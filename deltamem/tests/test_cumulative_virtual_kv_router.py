from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from deltamem.core.virtual_kv import (
    CumulativeRWKVCompatibilityRouter,
    ExplicitRWKVVirtualKV,
    VirtualKVShape,
)


def make_router(
    anchors: tuple[int, ...] = (5, 11, 17, 23),
    *,
    slots: int = 3,
) -> CumulativeRWKVCompatibilityRouter:
    builders = {
        layer: ExplicitRWKVVirtualKV(
            VirtualKVShape(
                key_dim=2,
                state_heads=1,
                rank=2,
                slots=slots,
                kv_heads=1,
                head_dim=2,
                probe_rank=2,
                value_hidden=4,
                seed=100 + layer,
            )
        )
        for layer in anchors
    }
    maps = {layer: (torch.eye(2), torch.eye(2)) for layer in anchors}
    return CumulativeRWKVCompatibilityRouter(
        builders=builders,
        maps=maps,
        anchor_layers=anchors,
        required_receptance_calls=2,
    )


def make_bank(
    anchors: tuple[int, ...],
    *,
    slots: int = 3,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    base_state = torch.arange(1, 1 + slots * 4, dtype=torch.float32).reshape(
        1, 1, slots, 2, 2
    )
    base_addresses = torch.tensor(
        [[[1.0, 0.25], [0.25, 1.0], [-1.0, 0.5]]], dtype=torch.float32
    )[:, :slots]
    source_ids = torch.arange(10, 10 + slots).unsqueeze(0)
    states = {layer: base_state + float(index) for index, layer in enumerate(anchors)}
    addresses = {
        layer: base_addresses + torch.tensor([0.1 * index, -0.05 * index])
        for index, layer in enumerate(anchors)
    }
    occupied = {
        layer: torch.ones(1, slots, dtype=torch.bool) for layer in anchors
    }
    sources = {layer: source_ids.clone() for layer in anchors}
    return states, addresses, occupied, sources


def invoke(
    router: CumulativeRWKVCompatibilityRouter,
    layer: int,
    receptance: torch.Tensor,
):
    return router.provider_for(layer)(
        query_states=torch.tensor([[[[0.5, -0.5]]]]),
        key_states=torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]]),
        value_states=torch.tensor([[[[0.1, 0.3], [0.5, 0.7]]]]),
        attention_mask=torch.zeros(1, 1, 1, 2),
        position_embeddings=None,
        module=SimpleNamespace(
            layer_idx=layer,
            rwkv_virtual_router_receptance=receptance.reshape(1, 1, 1, 2),
            rwkv_virtual_router_receptance_calls=2,
        ),
    )


def run_router(
    router: CumulativeRWKVCompatibilityRouter,
    banks,
    receptances: dict[int, torch.Tensor],
):
    states, addresses, occupied, sources = banks
    router.begin_forward(
        states=states,
        address_keys=addresses,
        occupied=occupied,
        source_ids=sources,
    )
    outputs = {layer: invoke(router, layer, receptances[layer]) for layer in router.anchor_layers}
    diagnostics = router.end_forward()
    return outputs, diagnostics


def test_cumulative_router_accumulates_in_strict_anchor_order() -> None:
    anchors = (5, 11, 17, 23)
    router = make_router(anchors)
    banks = make_bank(anchors)
    receptances = {
        5: torch.tensor([1.0, 0.0]),
        11: torch.tensor([0.0, 1.0]),
        17: torch.tensor([1.0, 1.0]),
        23: torch.tensor([-1.0, 1.0]),
    }
    outputs, diagnostics = run_router(router, banks, receptances)
    running = torch.zeros_like(diagnostics[0]["local_scores"])
    for index, layer in enumerate(anchors):
        running = running + diagnostics[index]["local_scores"]
        expected_scores = running / float(index + 1)
        torch.testing.assert_close(
            diagnostics[index]["accumulated_scores"], expected_scores
        )
        torch.testing.assert_close(
            diagnostics[index]["attention_bias"],
            2.0 * expected_scores,
        )
        keys, _, mask = outputs[layer]
        assert torch.equal(keys, torch.zeros_like(keys))
        torch.testing.assert_close(mask[..., -3:], 2.0 * expected_scores[:, None, None])
    assert not router.active
    assert not router.completed


def test_cumulative_router_fails_closed_on_skipped_or_repeated_anchor() -> None:
    anchors = (5, 11)
    router = make_router(anchors)
    banks = make_bank(anchors)
    router.begin_forward(
        states=banks[0],
        address_keys=banks[1],
        occupied=banks[2],
        source_ids=banks[3],
    )
    with pytest.raises(RuntimeError, match="expected=5 actual=11"):
        invoke(router, 11, torch.tensor([1.0, 0.0]))
    assert not router.active
    assert not router.completed
    assert router.diagnostics == ()

    router.begin_forward(
        states=banks[0],
        address_keys=banks[1],
        occupied=banks[2],
        source_ids=banks[3],
    )
    invoke(router, 5, torch.tensor([1.0, 0.0]))
    with pytest.raises(RuntimeError, match="expected=11 actual=5"):
        invoke(router, 5, torch.tensor([1.0, 0.0]))
    assert not router.active
    assert router.diagnostics == ()


def test_cumulative_router_rejects_cross_anchor_slot_misalignment() -> None:
    anchors = (5, 11)
    router = make_router(anchors)
    states, addresses, occupied, sources = make_bank(anchors)
    sources[11] = sources[11].roll(1, dims=1)
    with pytest.raises(ValueError, match="source alignment differs"):
        router.begin_forward(
            states=states,
            address_keys=addresses,
            occupied=occupied,
            source_ids=sources,
        )
    assert not router.active


def test_cumulative_router_is_slot_permutation_equivariant() -> None:
    anchors = (5, 11)
    router = make_router(anchors)
    banks = make_bank(anchors)
    receptances = {
        5: torch.tensor([1.0, -0.25]),
        11: torch.tensor([0.5, 1.0]),
    }
    outputs, diagnostics = run_router(router, banks, receptances)
    permutation = torch.tensor([2, 0, 1])
    permuted_banks = tuple(
        {
            layer: value.index_select(2 if index == 0 else 1, permutation)
            for layer, value in bank.items()
        }
        for index, bank in enumerate(banks)
    )
    permuted_outputs, permuted_diagnostics = run_router(
        router, permuted_banks, receptances
    )
    for index, layer in enumerate(anchors):
        torch.testing.assert_close(
            permuted_diagnostics[index]["attention_bias"],
            diagnostics[index]["attention_bias"].index_select(1, permutation),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            permuted_outputs[layer][1],
            outputs[layer][1].index_select(2, permutation),
            atol=1e-6,
            rtol=1e-6,
        )


def test_address_and_state_controls_change_separate_virtual_paths() -> None:
    anchors = (5,)
    router = make_router(anchors)
    banks = make_bank(anchors)
    receptances = {5: torch.tensor([1.0, -0.25])}
    outputs, diagnostics = run_router(router, banks, receptances)
    permutation = torch.tensor([1, 0, 2])

    address_control = (
        banks[0],
        {5: banks[1][5].index_select(1, permutation)},
        banks[2],
        banks[3],
    )
    address_outputs, address_diagnostics = run_router(
        router, address_control, receptances
    )
    assert torch.equal(address_outputs[5][1], outputs[5][1])
    torch.testing.assert_close(
        address_diagnostics[0]["attention_bias"],
        diagnostics[0]["attention_bias"].index_select(1, permutation),
        atol=0.0,
        rtol=0.0,
    )

    state_control = (
        {5: banks[0][5].index_select(2, permutation)},
        banks[1],
        banks[2],
        banks[3],
    )
    state_outputs, state_diagnostics = run_router(router, state_control, receptances)
    assert torch.equal(
        state_diagnostics[0]["attention_bias"], diagnostics[0]["attention_bias"]
    )
    torch.testing.assert_close(
        state_outputs[5][1],
        outputs[5][1].index_select(2, permutation),
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_address_disables_slot_across_the_remaining_prefix() -> None:
    anchors = (5, 11)
    router = make_router(anchors)
    states, addresses, occupied, sources = make_bank(anchors)
    addresses[5][:, 0] = 0.0
    outputs, diagnostics = run_router(
        router,
        (states, addresses, occupied, sources),
        {5: torch.tensor([1.0, 0.0]), 11: torch.tensor([0.0, 1.0])},
    )
    for index, layer in enumerate(anchors):
        assert not diagnostics[index]["active"][0, 0]
        assert outputs[layer][1][:, :, 0].square().sum().item() == 0.0
        assert outputs[layer][2][0, 0, 0, -3].item() == torch.finfo(torch.float32).min


def test_receptance_lifecycle_failure_clears_router_context() -> None:
    anchors = (5,)
    router = make_router(anchors)
    banks = make_bank(anchors)
    router.begin_forward(
        states=banks[0],
        address_keys=banks[1],
        occupied=banks[2],
        source_ids=banks[3],
    )
    provider = router.provider_for(5)
    with pytest.raises(RuntimeError, match="audited current RWKV receptance lifecycle"):
        provider(
            query_states=torch.ones(1, 1, 1, 2),
            key_states=torch.ones(1, 1, 1, 2),
            value_states=torch.ones(1, 1, 1, 2),
            attention_mask=torch.zeros(1, 1, 1, 1),
            position_embeddings=None,
            module=SimpleNamespace(
                layer_idx=5,
                rwkv_virtual_router_receptance=torch.ones(1, 1, 1, 2),
                rwkv_virtual_router_receptance_calls=1,
            ),
        )
    assert not router.active
    assert router.diagnostics == ()
