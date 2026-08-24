from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from deltamem.core.cumulative_rwkv_residual import (
    SourceBoundOuterFFN,
    SourceCumulativeResidualRouter,
)
from deltamem.core.delta import DeltaMemAttention, HFDeltaMemConfig
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention


ANCHORS = (5, 11, 17, 23)


class _IdentityReadout:
    def __init__(self) -> None:
        self.calls = 0

    def readout(self, reads: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return reads * gate


class _RouterModule:
    def __init__(
        self,
        layer: int,
        receptance: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_idx = layer
        self.rwkv_residual_router_receptance = receptance.to(dtype=dtype).reshape(
            1, 1, 1, 2
        )
        self.rwkv_residual_router_gate = torch.ones(1, 1, 2, dtype=dtype)
        self.rwkv_residual_router_receptance_calls = 2
        self.hrm_rwkv7_core = _IdentityReadout()
        self.delta_o_proj = torch.eye(2, dtype=dtype)

    @staticmethod
    def _project_delta_head(
        reads: torch.Tensor,
        weight: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor | None:
        assert head_name == "o"
        return F.linear(reads, weight)


def _router() -> SourceCumulativeResidualRouter:
    return SourceCumulativeResidualRouter(
        maps={layer: (torch.eye(2), torch.eye(2)) for layer in ANCHORS},
        anchor_layers=ANCHORS,
        compatibility_scale=8.0,
        residual_gain=1.0 / 32.0,
        required_receptance_calls=2,
    )


def _outer_router() -> SourceCumulativeResidualRouter:
    return SourceCumulativeResidualRouter(
        maps={layer: (torch.eye(2), torch.eye(2)) for layer in ANCHORS},
        anchor_layers=ANCHORS,
        compatibility_scale=8.0,
        residual_gain=1.0 / 32.0,
        required_receptance_calls=2,
        outer_ffn=SourceBoundOuterFFN(
            state_dim=2,
            query_dim=2,
            bottleneck_dim=3,
        ),
    )


def _banks():
    base_state = torch.tensor(
        [[[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [1.0, 0.0]], [[-1.0, 0.5], [0.25, 1.0]]]]]
    )
    base_address = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]])
    base_sources = torch.tensor([[30, 10, 20]], dtype=torch.long)
    states = {
        layer: base_state + float(index) / 10.0
        for index, layer in enumerate(ANCHORS)
    }
    addresses = {
        layer: base_address + torch.tensor([0.05 * index, -0.025 * index])
        for index, layer in enumerate(ANCHORS)
    }
    occupied = {
        layer: torch.ones(1, 3, dtype=torch.bool) for layer in ANCHORS
    }
    sources = {layer: base_sources.clone() for layer in ANCHORS}
    return states, addresses, occupied, sources


def _run(
    router: SourceCumulativeResidualRouter,
    banks,
    receptances: dict[int, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
):
    routed_banks = tuple(
        {layer: bank[layer] for layer in router.anchor_layers} for bank in banks
    )
    router.begin_forward(
        states=routed_banks[0],
        address_keys=routed_banks[1],
        occupied=routed_banks[2],
        source_ids=routed_banks[3],
    )
    residual = None
    modules = {}
    for layer in router.anchor_layers:
        module = _RouterModule(layer, receptances[layer], dtype=dtype)
        modules[layer] = module
        local = router.provider_for(layer)(
            hidden_states=torch.zeros(1, 1, 2, dtype=dtype),
            token_mask=torch.ones(1, 1, dtype=torch.bool),
            module=module,
        )
        if local is not None:
            residual = local
    diagnostics = router.end_forward()
    return residual, diagnostics, modules


def test_router_canonicalizes_independent_anchor_permutations_byte_exactly() -> None:
    receptances = {
        5: torch.tensor([1.0, 0.0]),
        11: torch.tensor([0.5, 1.0]),
        17: torch.tensor([-0.5, 1.0]),
        23: torch.tensor([1.0, 0.25]),
    }
    baseline_residual, baseline_diagnostics, _ = _run(
        _router(), _banks(), receptances
    )
    banks = _banks()
    permutations = {
        5: torch.tensor([2, 0, 1]),
        11: torch.tensor([1, 2, 0]),
        17: torch.tensor([0, 2, 1]),
        23: torch.tensor([2, 1, 0]),
    }
    permuted = tuple({} for _ in range(4))
    for layer in ANCHORS:
        permutation = permutations[layer]
        permuted[0][layer] = banks[0][layer].index_select(2, permutation)
        for bank_index in (1, 2, 3):
            permuted[bank_index][layer] = banks[bank_index][layer].index_select(
                1, permutation
            )
    residual, diagnostics, _ = _run(_router(), permuted, receptances)

    assert torch.equal(residual, baseline_residual)
    for actual, expected in zip(diagnostics, baseline_diagnostics):
        for name in (
            "source_ids",
            "local_scores",
            "accumulated_scores",
            "active",
        ):
            assert torch.equal(actual[name], expected[name])


def test_router_accumulates_scores_and_uses_hard_canonical_source_route() -> None:
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    residual, diagnostics, modules = _run(_router(), _banks(), receptances)

    running = torch.zeros_like(diagnostics[0]["local_scores"])
    for index, diagnostic in enumerate(diagnostics):
        running = running + diagnostic["local_scores"]
        assert torch.equal(
            diagnostic["accumulated_scores"], running / float(index + 1)
        )
    terminal = diagnostics[-1]
    assert torch.equal(terminal["source_ids"], torch.tensor([[10, 20, 30]]))
    assert torch.equal(
        terminal["source_routes"].sum(dim=-1), torch.ones(1, 1)
    )
    assert set(terminal["source_routes"].flatten().tolist()) <= {0.0, 1.0}
    assert torch.equal(residual, terminal["residual"])
    assert float(residual.abs().max()) <= 1.0 / 32.0
    assert modules[23].hrm_rwkv7_core.calls == 1
    selected_score = terminal["accumulated_scores"].gather(
        -1, terminal["selected_slot"].unsqueeze(-1)
    )
    torch.testing.assert_close(
        terminal["memory_mass"],
        torch.sigmoid(8.0 * selected_score),
    )


def test_selected_source_gate_is_invariant_to_lower_scoring_slot_count() -> None:
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    full_banks = _banks()
    _, full_diagnostics, _ = _run(_router(), full_banks, receptances)
    selected_source = full_diagnostics[-1]["source_ids"].gather(
        1, full_diagnostics[-1]["selected_slot"]
    )

    single_banks = list(_banks())
    single_banks[2] = {
        layer: source_ids.eq(selected_source)
        for layer, source_ids in single_banks[3].items()
    }
    _, single_diagnostics, _ = _run(
        _router(), tuple(single_banks), receptances
    )

    assert torch.equal(
        full_diagnostics[-1]["selected_slot"],
        single_diagnostics[-1]["selected_slot"],
    )
    assert torch.equal(
        full_diagnostics[-1]["memory_mass"],
        single_diagnostics[-1]["memory_mass"],
    )


def test_terminal_raw_read_matches_selected_native_rwkv_state() -> None:
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    _, diagnostics, _ = _run(_router(), _banks(), receptances)
    terminal = diagnostics[-1]
    selected_slot = int(terminal["selected_slot"].item())
    canonical_order = torch.argsort(_banks()[3][23], dim=1, stable=True)
    canonical_state = SourceCumulativeResidualRouter._canonical_gather(
        _banks()[0][23], canonical_order, 2
    )
    expected = canonical_state[0, 0, selected_slot] @ receptances[23]

    torch.testing.assert_close(terminal["raw_read"][0, 0], expected)
    assert torch.equal(terminal["native_read"], terminal["raw_read"])
    assert torch.equal(terminal["hidden_read"], terminal["native_read"])


def test_outer_ffn_zero_initialization_preserves_base_residual() -> None:
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    baseline, _, _ = _run(_router(), _banks(), receptances)
    outer, diagnostics, _ = _run(_outer_router(), _banks(), receptances)

    assert torch.equal(outer, baseline)
    terminal = diagnostics[-1]
    assert torch.equal(
        terminal["correction"], torch.zeros_like(terminal["correction"])
    )
    assert torch.equal(
        terminal["query_gate"], torch.ones_like(terminal["query_gate"])
    )


def test_outer_ffn_exact_zero_state_and_trainable_joint_path() -> None:
    outer_ffn = SourceBoundOuterFFN(
        state_dim=2,
        query_dim=4,
        bottleneck_dim=3,
    )
    with torch.no_grad():
        outer_ffn.output_up.weight.fill_(0.125)
        outer_ffn.query_gate.weight.fill_(0.25)
    zero_state = torch.zeros(2, 1, 2, requires_grad=True)
    hidden_query = torch.randn(2, 1, 4, requires_grad=True)
    base_hidden_read = torch.zeros(2, 1, 4)
    direction, diagnostics = outer_ffn(
        native_read=zero_state,
        hidden_query=hidden_query,
        base_hidden_read=base_hidden_read,
    )

    assert torch.equal(direction, torch.zeros_like(direction))
    assert all(layer.bias is None for layer in (
        outer_ffn.state_down,
        outer_ffn.query_gate,
        outer_ffn.output_up,
    ))
    assert torch.equal(
        diagnostics["state_value"], torch.zeros_like(diagnostics["state_value"])
    )

    live_state = torch.tensor([[[1.0, -0.5]], [[-0.25, 1.0]]], requires_grad=True)
    live_direction, _ = outer_ffn(
        native_read=live_state,
        hidden_query=hidden_query,
        base_hidden_read=torch.randn(2, 1, 4),
    )
    live_direction.square().mean().backward()
    assert outer_ffn.state_down.weight.grad is not None
    assert outer_ffn.query_gate.weight.grad is not None
    assert outer_ffn.output_up.weight.grad is not None
    assert bool(outer_ffn.output_up.weight.grad.abs().max().gt(0.0).item())


def test_outer_ffn_three_anchor_route_fires_at_layer_17_and_stays_canonical() -> None:
    anchors = (5, 11, 17)

    def router() -> SourceCumulativeResidualRouter:
        return SourceCumulativeResidualRouter(
            maps={layer: (torch.eye(2), torch.eye(2)) for layer in anchors},
            anchor_layers=anchors,
            compatibility_scale=1.0,
            residual_gain=1.0 / 32.0,
            required_receptance_calls=2,
            outer_ffn=SourceBoundOuterFFN(
                state_dim=2,
                query_dim=2,
                bottleneck_dim=3,
            ),
        )

    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in anchors}
    baseline, diagnostics, _ = _run(router(), _banks(), receptances)
    assert tuple(item["layer"] for item in diagnostics) == anchors
    assert "residual" not in diagnostics[0]
    assert "residual" not in diagnostics[1]
    assert torch.equal(baseline, diagnostics[2]["residual"])

    banks = list(_banks())
    permutations = {
        5: torch.tensor([2, 0, 1]),
        11: torch.tensor([1, 2, 0]),
        17: torch.tensor([0, 2, 1]),
    }
    for bank_index in range(4):
        banks[bank_index] = dict(banks[bank_index])
        for layer, permutation in permutations.items():
            axis = 2 if bank_index == 0 else 1
            banks[bank_index][layer] = banks[bank_index][layer].index_select(
                axis, permutation
            )
    permuted, _, _ = _run(router(), tuple(banks), receptances)
    assert torch.equal(permuted, baseline)

    zero_banks = list(_banks())
    zero_banks[0] = {
        layer: torch.zeros_like(value)
        for layer, value in zero_banks[0].items()
    }
    zero, zero_diagnostics, _ = _run(
        router(), tuple(zero_banks), receptances
    )
    assert torch.equal(zero, torch.zeros_like(zero))
    assert zero_diagnostics[-1]["selected_slot"].item() == -1


@pytest.mark.parametrize("zero_bank", ("state", "address"))
def test_zero_source_component_is_sticky_and_all_inactive_is_exact_zero(
    zero_bank: str,
) -> None:
    banks = list(_banks())
    bank_index = 0 if zero_bank == "state" else 1
    banks[bank_index] = {
        layer: torch.zeros_like(value) for layer, value in banks[bank_index].items()
    }
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    residual, diagnostics, modules = _run(_router(), tuple(banks), receptances)

    assert torch.equal(residual, torch.zeros_like(residual))
    assert all(not bool(item["active"].any().item()) for item in diagnostics)
    assert diagnostics[-1]["selected_slot"].item() == -1
    assert modules[23].hrm_rwkv7_core.calls == 0


def test_inactivity_remains_sticky_after_one_zero_address_anchor() -> None:
    states, addresses, occupied, sources = _banks()
    addresses[5][:, 0] = 0.0
    receptances = {layer: torch.tensor([1.0, 0.0]) for layer in ANCHORS}
    _, diagnostics, _ = _run(
        _router(), (states, addresses, occupied, sources), receptances
    )

    source_id = 30
    source_position = int((diagnostics[0]["source_ids"][0] == source_id).nonzero().item())
    assert all(not item["active"][0, source_position] for item in diagnostics)


def test_router_rejects_duplicate_or_cross_anchor_source_identity_sets() -> None:
    states, addresses, occupied, sources = _banks()
    sources[5][0, 1] = sources[5][0, 0]
    with pytest.raises(ValueError, match="unique"):
        _router().begin_forward(
            states=states,
            address_keys=addresses,
            occupied=occupied,
            source_ids=sources,
        )

    states, addresses, occupied, sources = _banks()
    sources[11][0, 1] = 99
    with pytest.raises(ValueError, match="identity set differs"):
        _router().begin_forward(
            states=states,
            address_keys=addresses,
            occupied=occupied,
            source_ids=sources,
        )


def test_router_fails_closed_on_anchor_or_capture_lifecycle_error() -> None:
    router = _router()
    banks = _banks()
    router.begin_forward(
        states=banks[0],
        address_keys=banks[1],
        occupied=banks[2],
        source_ids=banks[3],
    )
    with pytest.raises(RuntimeError, match="expected=5 actual=11"):
        router.provider_for(11)(
            hidden_states=torch.zeros(1, 1, 2),
            token_mask=None,
            module=_RouterModule(11, torch.ones(2)),
        )
    assert not router.active
    assert router.diagnostics == ()

    router.begin_forward(
        states=banks[0],
        address_keys=banks[1],
        occupied=banks[2],
        source_ids=banks[3],
    )
    module = _RouterModule(5, torch.ones(2))
    module.rwkv_residual_router_receptance_calls = 1
    with pytest.raises(RuntimeError, match="audited current RWKV receptance lifecycle"):
        router.provider_for(5)(
            hidden_states=torch.zeros(1, 1, 2),
            token_mask=None,
            module=module,
        )
    assert not router.active


def test_bfloat16_terminal_residual_is_finite_and_bounded() -> None:
    receptances = {layer: torch.tensor([1.0, 0.25]) for layer in ANCHORS}
    router = _router()
    residual, _, modules = _run(
        router, _banks(), receptances, dtype=torch.bfloat16
    )

    assert residual.dtype == torch.bfloat16
    assert torch.isfinite(residual).all()
    assert float(residual.abs().max()) <= 1.0 / 32.0
    assert modules[23].delta_o_proj.dtype == torch.bfloat16


def _attention_module() -> DeltaMemAttention:
    return DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_rwkv_hybrid",
            projected_kv_key_dim=2,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
            output_init="base_slice_fixed",
            base_slice_ref_width=2,
            rwkv_ms_hybrid_mode="addressed_moe_controller",
            delta_heads=("o",),
        ),
    )


def test_attention_wrapper_captures_two_identical_live_rwkv_reads() -> None:
    module = _attention_module()
    module.set_source_cumulative_residual_provider(lambda **kwargs: None)
    state = torch.randn(1, 1, 2, 2, 2)
    source = torch.randn(1, 1, 2)

    first = module._rwkv_ms_token_state_read_basis(state, source, None)
    second = module._rwkv_ms_token_state_read_basis(state, source, None)

    assert module.rwkv_residual_router_receptance_calls == 2
    assert module.rwkv_residual_router_receptance is first[0]
    assert module.rwkv_residual_router_gate is first[2]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])


def test_attention_wrapper_rejects_inconsistent_second_live_rwkv_read() -> None:
    module = _attention_module()
    module.set_source_cumulative_residual_provider(lambda **kwargs: None)
    state = torch.randn(1, 1, 2, 2, 2)
    source = torch.randn(1, 1, 2)
    module._rwkv_ms_token_state_read_basis(state, source, None)

    with pytest.raises(RuntimeError, match="inconsistent RWKV read captures"):
        module._rwkv_ms_token_state_read_basis(state, source + 1.0, None)


def test_zero_terminal_hook_returns_same_tensor_and_consumes_once() -> None:
    module = _attention_module()
    layernorm = torch.nn.Identity()
    module.bind_source_cumulative_residual_layernorm(layernorm)
    module.set_source_cumulative_residual_provider(lambda **kwargs: None)
    output = torch.randn(1, 1, module.hidden_size)
    module._pending_source_cumulative_residual = torch.zeros_like(output)

    actual = layernorm(output)

    assert actual is output
    assert module._pending_source_cumulative_residual is None
    with pytest.raises(RuntimeError, match="without a pending"):
        layernorm(output)
    module.clear_source_cumulative_residual_provider()
    module.remove_source_cumulative_residual_layernorm_hook()


def test_trainable_zero_terminal_hook_preserves_gradient_path() -> None:
    module = _attention_module()
    layernorm = torch.nn.Identity()
    module.bind_source_cumulative_residual_layernorm(layernorm)
    module.set_source_cumulative_residual_provider(lambda **kwargs: None)
    output = torch.randn(1, 1, module.hidden_size)
    trainable_zero = torch.zeros_like(output, requires_grad=True)
    module._pending_source_cumulative_residual = trainable_zero

    actual = layernorm(output)

    assert actual is not output
    assert torch.equal(actual, output)
    actual.square().mean().backward()
    assert trainable_zero.grad is not None
    assert bool(trainable_zero.grad.abs().max().gt(0.0).item())
    module.clear_source_cumulative_residual_provider()
    module.remove_source_cumulative_residual_layernorm_hook()


def test_virtual_and_no_suffix_residual_providers_are_mutually_exclusive() -> None:
    module = _attention_module()
    module.set_source_cumulative_residual_provider(lambda **kwargs: None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.set_virtual_kv_provider(lambda **kwargs: None)
    module.clear_source_cumulative_residual_provider()

    module.set_virtual_kv_provider(lambda **kwargs: None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.set_source_cumulative_residual_provider(lambda **kwargs: None)
