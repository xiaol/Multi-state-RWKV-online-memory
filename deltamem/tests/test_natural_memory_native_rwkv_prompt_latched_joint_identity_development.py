from __future__ import annotations

import torch

from deltamem.core.cumulative_rwkv_residual import SourceBoundJointIdentityFFN
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_prompt_latched_joint_identity_development as runner,
)


def test_protocol_locks_open_only_joint_identity_route() -> None:
    protocol = runner.validate_protocol()

    assert protocol["open_development_only"] is True
    assert protocol["protected_mechanics_authorized"] is False
    assert protocol["protected_causal_authorized"] is False
    assert protocol["native_benchmark_authorized"] is False
    assert protocol["architecture"]["prompt_boundary_source_latched"] is True
    assert protocol["architecture"]["selected_native_rwkv_read_is_only_value"] is True
    assert (
        protocol["architecture"][
            "query_only_hidden_only_or_native_hidden_read_bypass"
        ]
        is False
    )
    assert protocol["architecture"]["identity_feature_dim"] == (
        runner.IDENTITY_FEATURE_DIM
    )


def test_joint_identity_parameter_inventory_and_staged_gradients() -> None:
    module = SourceBoundJointIdentityFFN(
        state_dim=2,
        hidden_dim=4,
        anchor_count=2,
        bottleneck_dim=3,
    )
    assert module.identity_dim == 8
    native_read = torch.tensor([[[1.0, -0.5]]])
    identity_features = torch.tensor(
        [[[0.5, -0.25, 0.5, 0.25, -0.5, 0.75, 1.5, 0.25]]]
    )
    base_hidden_read = torch.full((1, 1, 4), 9.0)

    direction, diagnostics = module(
        native_read=native_read,
        identity_features=identity_features,
        base_hidden_read=base_hidden_read,
    )
    assert torch.equal(direction, torch.zeros_like(direction))
    assert torch.equal(
        diagnostics["combined_hidden_read"], diagnostics["correction"]
    )
    direction.sub(0.25).square().mean().backward()
    assert torch.equal(
        module.state_down.weight.grad,
        torch.zeros_like(module.state_down.weight.grad),
    )
    assert torch.equal(
        module.query_gate.weight.grad,
        torch.zeros_like(module.query_gate.weight.grad),
    )
    assert bool(module.output_up.weight.grad.abs().max().gt(0.0).item())

    with torch.no_grad():
        module.output_up.weight.add_(0.01 * module.output_up.weight.grad)
    module.zero_grad(set_to_none=True)
    direction, _ = module(
        native_read=native_read,
        identity_features=identity_features,
        base_hidden_read=base_hidden_read,
    )
    direction.sub(0.25).square().mean().backward()
    assert all(
        parameter.grad is not None
        and bool(parameter.grad.abs().max().gt(0.0).item())
        for parameter in module.parameters()
    )


def test_joint_identity_zero_state_is_exact_zero_without_hidden_bypass() -> None:
    module = SourceBoundJointIdentityFFN(
        state_dim=2,
        hidden_dim=4,
        anchor_count=1,
        bottleneck_dim=3,
    )
    with torch.no_grad():
        module.output_up.weight.fill_(1.0)
        module.query_gate.weight.fill_(0.5)
    direction, diagnostics = module(
        native_read=torch.zeros(2, 1, 2),
        identity_features=torch.randn(2, 1, 4),
        base_hidden_read=torch.randn(2, 1, 4),
    )
    assert torch.equal(direction, torch.zeros_like(direction))
    assert torch.equal(
        diagnostics["correction"], torch.zeros_like(diagnostics["correction"])
    )


def test_prompt_latch_masks_every_anchor_to_one_source() -> None:
    batch_size = 3
    slots = 4
    states = {
        layer: torch.randn(batch_size, 1, slots, 2, 2)
        for layer in runner.base.ANCHORS
    }
    addresses = {
        layer: torch.randn(batch_size, slots, 5)
        for layer in runner.base.ANCHORS
    }
    occupied = {
        layer: torch.ones(batch_size, slots, dtype=torch.bool)
        for layer in runner.base.ANCHORS
    }
    source_ids = {
        layer: torch.tensor(
            [[10, 20, 30, 40], [40, 30, 20, 10], [10, 20, 30, 40]]
        )
        for layer in runner.base.ANCHORS
    }
    latched = runner.latch_banks(
        (states, addresses, occupied, source_ids),
        torch.tensor([30, 10, -1]),
    )

    for layer in runner.base.ANCHORS:
        assert latched[2][layer].sum(dim=1).tolist() == [1, 1, 0]
        retained = torch.where(latched[2][layer], source_ids[layer], -1)
        assert retained[0].max().item() == 30
        assert retained[1].max().item() == 10
        assert torch.equal(latched[0][layer], states[layer])
        assert torch.equal(latched[1][layer], addresses[layer])


def test_full_parameter_inventory_matches_protocol() -> None:
    module = SourceBoundJointIdentityFFN(
        state_dim=runner.base.NATIVE_READ_DIM,
        hidden_dim=runner.base.HIDDEN_DIM,
        anchor_count=len(runner.base.ANCHORS),
        bottleneck_dim=runner.base.BOTTLENECK_DIM,
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == (
        runner.TRAINABLE_ELEMENTS
    )
    assert len(tuple(module.parameters())) == runner.base.TRAINABLE_TENSORS
