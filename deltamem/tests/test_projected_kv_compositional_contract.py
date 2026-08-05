from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from deltamem.core.delta import (
    DeltaMemAttention,
    HFDeltaMemConfig,
    collect_delta_mem_projected_kv_read_logits,
)
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention
from deltamem.train import delta_sft_experimental as experimental_train


def _module(*, slots: int = 4, key_dim: int = 4) -> DeltaMemAttention:
    torch.manual_seed(0)
    return DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=slots,
            rwkv_ms_chunk_size=2,
            memory_readout_mode="projected_kv_slots",
            projected_kv_key_dim=key_dim,
            projected_kv_temperature=4.0,
            projected_kv_update_cosine_threshold=1.0,
            memory_write_granularity="token",
        ),
    )


def test_separate_write_spans_project_key_and_value_from_distinct_tokens() -> None:
    module = _module()
    hidden = torch.randn(2, 5, module.hidden_size)
    key_mask = torch.tensor(
        [[False, True, True, False, False], [True, False, False, False, False]]
    )
    value_mask = torch.tensor(
        [[False, False, False, True, False], [False, False, True, True, False]]
    )
    slots = torch.tensor([2, 3])

    module.set_projected_kv_write_spans(key_mask, value_mask, slots)
    module._write_projected_kv_slots(hidden, torch.ones(2, 5, dtype=torch.bool))

    key_hidden = torch.stack((hidden[0, 1:3].mean(0), hidden[1, 0]))
    value_hidden = torch.stack((hidden[0, 3], hidden[1, 2:4].mean(0)))
    expected_keys, _ = module._projected_kv_project_hidden(key_hidden)
    _, expected_values = module._projected_kv_project_hidden(value_hidden)

    assert module.projected_kv_keys is not None
    assert module.projected_kv_values is not None
    assert module.projected_kv_occupied is not None
    assert torch.allclose(module.projected_kv_keys[torch.arange(2), slots], expected_keys)
    assert torch.allclose(
        module.projected_kv_values[torch.arange(2), slots], expected_values
    )
    assert torch.equal(
        module.projected_kv_occupied,
        F.one_hot(slots, num_classes=4).to(dtype=torch.bool),
    )


def test_forced_slots_fill_exact_capacity_even_when_projected_keys_match() -> None:
    module = _module()
    hidden = torch.randn(1, 3, module.hidden_size)
    key_mask = torch.tensor([[False, True, False]])
    value_mask = torch.tensor([[False, False, True]])
    token_mask = torch.ones(1, 3, dtype=torch.bool)

    for slot in range(4):
        module.set_projected_kv_write_spans(
            key_mask,
            value_mask,
            torch.tensor([slot]),
        )
        module._write_projected_kv_slots(hidden, token_mask)

    assert module.projected_kv_occupied is not None
    assert torch.equal(module.projected_kv_occupied, torch.ones(1, 4, dtype=torch.bool))


def test_projected_kv_write_spans_fail_closed() -> None:
    module = _module()
    mask = torch.tensor([[True, False]])

    with pytest.raises(ValueError, match="both be set"):
        module.set_projected_kv_write_spans(mask, None)
    with pytest.raises(ValueError, match="require key and value"):
        module.set_projected_kv_write_spans(None, None, torch.tensor([0]))

    module.set_projected_kv_write_spans(mask, mask, torch.tensor([0]))
    with pytest.raises(ValueError, match="must not overlap"):
        module._write_projected_kv_slots(
            torch.randn(1, 2, module.hidden_size),
            torch.ones(1, 2, dtype=torch.bool),
        )

    module.set_projected_kv_write_spans(
        torch.tensor([[False, True]]),
        torch.tensor([[True, False]]),
        torch.tensor([0]),
    )
    with pytest.raises(ValueError, match="only valid tokens"):
        module._write_projected_kv_slots(
            torch.randn(1, 2, module.hidden_size),
            torch.tensor([[True, False]]),
        )


def test_read_route_logits_are_graph_connected_for_cross_entropy() -> None:
    module = _module(slots=2, key_dim=2)
    with torch.no_grad():
        module.projected_kv_key_proj.zero_()
        module.projected_kv_key_proj[0, 0] = 1.0
        module.projected_kv_key_proj[1, 1] = 1.0
    module.projected_kv_keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    module.projected_kv_values = torch.randn(1, 2, module.state_read_dim)
    module.projected_kv_occupied = torch.ones(1, 2, dtype=torch.bool)
    module.projected_kv_surprise = torch.ones(1, 2)

    query = torch.zeros(1, 1, module.hidden_size)
    query[0, 0, 0] = 0.25
    query[0, 0, 1] = 1.0
    module._projected_kv_slot_token_reads(query)

    assert module.last_read_route_logits is not None
    model = torch.nn.Module()
    model.add_module("attn", module)
    collected = collect_delta_mem_projected_kv_read_logits(model)
    assert collected == {"attn": module.last_read_route_logits}
    assert collected["attn"] is module.last_read_route_logits
    loss = F.cross_entropy(
        collected["attn"][:, 0].float(),
        torch.tensor([0]),
    )
    loss.backward()
    assert module.projected_kv_key_proj.grad is not None
    assert torch.count_nonzero(module.projected_kv_key_proj.grad).item() > 0


def test_query_span_pooling_uses_one_semantic_route_for_the_full_read() -> None:
    module = _module(slots=2, key_dim=2)
    with torch.no_grad():
        module.projected_kv_key_proj.zero_()
        module.projected_kv_key_proj[0, 0] = 1.0
        module.projected_kv_key_proj[1, 1] = 1.0
    module.projected_kv_keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    module.projected_kv_values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    module.projected_kv_occupied = torch.ones(1, 2, dtype=torch.bool)
    module.projected_kv_surprise = torch.ones(1, 2)
    query = torch.zeros(1, 4, module.hidden_size)
    query[0, 0, 0] = 10.0
    query[0, 1:3, 1] = 1.0
    query[0, 3, 0] = 10.0
    module.set_projected_kv_read_query_mask(
        torch.tensor([[False, True, True, False]])
    )

    reads = module._projected_kv_slot_token_reads(query)

    assert module.last_read_routes is not None
    assert module.last_read_route_logits is not None
    assert module.last_read_routes.shape == (1, 4, 2)
    assert module.last_read_route_logits.shape == (1, 4, 2)
    assert torch.equal(
        module.last_read_routes.argmax(dim=-1),
        torch.ones(1, 4, dtype=torch.long),
    )
    assert torch.allclose(
        reads,
        module.projected_kv_values[:, 1:2].expand(-1, 4, -1),
    )


def test_reset_clears_compositional_runtime_controls_and_logits() -> None:
    module = _module()
    mask = torch.tensor([[True, False]])
    module.set_projected_kv_write_spans(mask, ~mask, torch.tensor([1]))
    module.set_projected_kv_read_query_mask(mask)
    module.last_read_route_logits = torch.randn(1, 2, 4)

    module.reset_state()

    assert module.projected_kv_write_key_mask is None
    assert module.projected_kv_write_value_mask is None
    assert module.projected_kv_write_slot_indices is None
    assert module.projected_kv_read_query_mask is None
    assert module.last_read_route_logits is None


def test_mode_transitions_clear_controls_from_the_previous_phase() -> None:
    module = _module()
    mask = torch.tensor([[True, False]])
    module.set_projected_kv_write_spans(mask, ~mask, torch.tensor([1]))
    module.last_read_route_logits = torch.randn(1, 2, 4)

    module.set_write_enabled(False)

    assert module.projected_kv_write_key_mask is None
    assert module.projected_kv_write_value_mask is None
    assert module.projected_kv_write_slot_indices is None
    assert module.last_read_route_logits is None

    module.set_projected_kv_read_query_mask(mask)
    module.last_read_route_logits = torch.randn(1, 2, 4)

    module.set_write_enabled(True)

    assert module.projected_kv_read_query_mask is None
    assert module.last_read_route_logits is None


def test_preserve_runtime_restores_compositional_controls_and_logits() -> None:
    module = _module()
    model = torch.nn.Module()
    model.add_module("attn", module)
    key_mask = torch.tensor([[True, False]])
    value_mask = ~key_mask
    slot_indices = torch.tensor([1])
    query_mask = torch.tensor([[False, True]])
    logits = torch.randn(1, 2, 4, requires_grad=True)
    module.set_projected_kv_write_spans(key_mask, value_mask, slot_indices)
    module.set_projected_kv_read_query_mask(query_mask)
    module.last_read_route_logits = logits

    with experimental_train._preserve_delta_runtime(model):
        module.set_projected_kv_write_spans(None, None)
        module.set_projected_kv_read_query_mask(None)
        module.last_read_route_logits = None

    assert module.projected_kv_write_key_mask is key_mask
    assert module.projected_kv_write_value_mask is value_mask
    assert module.projected_kv_write_slot_indices is slot_indices
    assert module.projected_kv_read_query_mask is query_mask
    assert module.last_read_route_logits is logits
