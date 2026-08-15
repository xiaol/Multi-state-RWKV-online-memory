from __future__ import annotations

import pytest
import torch

from deltamem.core.delta import (
    DeltaMemAttention,
    HFDeltaMemConfig,
    get_delta_mem_online_state,
    load_delta_mem_online_state,
)
from deltamem.tests.test_delta_mem_regressions import make_qwen3_attention


def _module(
    *,
    hybrid_mode: str = "residual",
    hybrid_gain: float = 0.125,
) -> DeltaMemAttention:
    torch.manual_seed(0)
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
            rwkv_ms_output_init_scale=0.02,
            rwkv_ms_hybrid_mode=hybrid_mode,
            rwkv_ms_hybrid_gain=hybrid_gain,
        ),
    )


@pytest.mark.parametrize("mode", ("residual", "vector_gate", "scalar_gate"))
def test_hybrid_modes_preserve_projected_carrier_for_zero_rwkv_state(
    mode: str,
) -> None:
    module = _module(hybrid_mode=mode)
    projected = torch.randn(2, 3, module.state_read_dim)

    fused = module._fuse_projected_rwkv_reads(
        projected,
        torch.zeros_like(projected),
    )

    assert torch.equal(fused, projected)


@pytest.mark.parametrize("mode", ("residual", "vector_gate", "scalar_gate"))
def test_hybrid_modes_are_sensitive_to_nonzero_rwkv_read(mode: str) -> None:
    module = _module(hybrid_mode=mode, hybrid_gain=0.25)
    projected = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    recurrent = torch.tensor([[[0.25, 0.75], [-0.5, 0.125]]])

    fused = module._fuse_projected_rwkv_reads(projected, recurrent)

    assert torch.isfinite(fused).all()
    assert not torch.equal(fused, projected)


def test_hybrid_write_populates_projected_and_recurrent_state() -> None:
    module = _module()
    hidden = torch.randn(1, 4, module.hidden_size)
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    state = module._ensure_state(1, hidden.device, hidden.dtype)

    next_state, reads, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        token_mask,
    )

    assert module.projected_kv_keys is not None
    assert module.projected_kv_values is not None
    assert module.projected_kv_occupied is not None
    assert module.projected_kv_occupied.any()
    assert torch.count_nonzero(next_state).item() > 0
    assert module.rwkv_ms_positions is not None
    assert torch.equal(module.rwkv_ms_positions, torch.tensor([4]))
    assert torch.count_nonzero(reads).item() == 0


def test_hybrid_online_state_round_trip_contains_both_carriers() -> None:
    module = _module()
    model = torch.nn.Module()
    model.add_module("attn", module)
    hidden = torch.randn(1, 3, module.hidden_size)
    state = module._ensure_state(1, hidden.device, hidden.dtype)
    next_state, _, _, _ = module._projected_rwkv_hybrid_step(
        state,
        hidden,
        torch.ones(1, 3, dtype=torch.bool),
    )
    module.delta_state = next_state
    saved = get_delta_mem_online_state(model)

    module.reset_state()
    load_delta_mem_online_state(model, saved)

    assert "attn" in saved
    assert "attn.__projected_kv_keys" in saved
    assert "attn.__projected_kv_values" in saved
    assert module.delta_state is not None
    assert torch.equal(module.delta_state.cpu(), saved["attn"])
    assert module.projected_kv_values is not None
    assert torch.equal(
        module.projected_kv_values.cpu(),
        saved["attn.__projected_kv_values"],
    )


def test_hybrid_configuration_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="rwkv_ms_hybrid_gain"):
        HFDeltaMemConfig(rwkv_ms_hybrid_gain=1.01)
    with pytest.raises(ValueError, match="hybrid mode"):
        HFDeltaMemConfig(rwkv_ms_hybrid_mode="unknown")
    with pytest.raises(ValueError, match="memory_write_granularity='token'"):
        HFDeltaMemConfig(
            memory_backend="rwkv_ms",
            memory_readout_mode="projected_kv_rwkv_hybrid",
            memory_write_granularity="message_mean",
        )
