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


@pytest.mark.parametrize("semantics_version", (1, 2))
def test_vectorized_recurrent_read_matches_token_loop(semantics_version: int) -> None:
    module = _module()
    module.rwkv_ms_semantics_version = semantics_version
    module.rwkv_ms_mask_empty_slots = True
    module.rwkv_ms_read_top_k = 1
    batch_size, seq_len = 2, 5
    source = torch.randn(
        batch_size,
        seq_len,
        module.state_read_dim,
        requires_grad=True,
    )
    state = torch.randn(
        batch_size,
        module.num_state_heads,
        module.rwkv_ms_num_states,
        module.rank,
        module.rank,
        requires_grad=True,
    )
    state = state.clone()
    state[:, :, -1] = 0.0
    token_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, False]]
    )

    features = module.hrm_rwkv7_core.project(
        source,
        previous_x=None,
        token_mask=token_mask,
        advance_within_sequence=False,
    )
    r_seq = module._rwkv_ms_project_heads(features.r).float()
    current_state = state.float()
    occupied = current_state.ne(0).any(dim=(-1, -2))
    reference_reads = []
    reference_routes = []
    for token_index in range(seq_len):
        r_t = r_seq[:, token_index]
        routes = module._rwkv_ms_read_routes(
            torch.einsum("bhsij,bhj->bhsi", current_state, r_t),
            r_t,
            token_mask[:, token_index],
            occupied_slots=occupied,
        )
        slot_reads = torch.einsum("bhsij,bhj->bhsi", current_state, r_t)
        reference_reads.append(
            torch.einsum("bhs,bhsi->bhi", routes, slot_reads).reshape(
                batch_size,
                module.state_read_dim,
            )
        )
        reference_routes.append(routes.mean(dim=1))
    reference = module.hrm_rwkv7_core.readout(
        torch.stack(reference_reads, dim=1).to(features.g.dtype),
        features.g,
    )
    targets = (state, source, module.hrm_rwkv7_core.output.weight)
    reference_grads = torch.autograd.grad(
        reference.float().square().mean(),
        targets,
        retain_graph=True,
    )

    vectorized = module._rwkv_ms_token_state_reads(state, source, token_mask)
    vectorized_routes = module.last_read_routes
    vectorized_grads = torch.autograd.grad(
        vectorized.float().square().mean(),
        targets,
    )

    assert torch.equal(vectorized, reference)
    assert torch.allclose(
        vectorized_routes,
        torch.stack(reference_routes, dim=1),
        atol=1e-7,
        rtol=0.0,
    )
    for vectorized_grad, reference_grad in zip(
        vectorized_grads,
        reference_grads,
    ):
        assert torch.allclose(
            vectorized_grad,
            reference_grad,
            atol=1e-8,
            rtol=1e-6,
        ), {
            "maximum_absolute_delta": float(
                (vectorized_grad - reference_grad).abs().max()
            ),
            "reference_maximum": float(reference_grad.abs().max()),
        }


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
