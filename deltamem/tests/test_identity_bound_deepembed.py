import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma.identity_bound_deepembed import (
    StateIdentityBinder,
    install,
)
from deltamem.tests.test_projected_rwkv_hybrid_contract import _ple_module


def test_identity_binder_zero_state_has_exact_zero_correction() -> None:
    torch.manual_seed(7)
    binder = StateIdentityBinder(8)
    projected = torch.randn(2, 3, 8)
    query = torch.randn(2, 3, 8)
    zero = torch.zeros_like(query)
    correction, score, gate = binder.correction(projected, zero, query)
    assert bool(torch.isfinite(score).all())
    assert bool(torch.isfinite(gate).all())
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0.0, rtol=0.0)


def test_identity_binder_score_has_live_pair_gradient() -> None:
    torch.manual_seed(8)
    binder = StateIdentityBinder(8)
    projected = torch.randn(2, 3, 8)
    state = torch.randn(2, 3, 8)
    score = binder.score(projected, state).mean()
    score.backward()
    gradients = [parameter.grad for parameter in binder.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None)
    assert any(bool(gradient.abs().gt(0).any()) for gradient in gradients if gradient is not None)


def test_identity_binder_pair_swap_changes_correction() -> None:
    torch.manual_seed(9)
    binder = StateIdentityBinder(8)
    projected = torch.randn(2, 3, 8)
    query = torch.randn(2, 3, 8)
    state = torch.randn(2, 3, 8)
    donor = torch.roll(state, shifts=1, dims=1)
    correct, _, _ = binder.correction(projected, state, query)
    swapped, _, _ = binder.correction(projected, donor, query)
    assert bool((correct - swapped).abs().gt(1e-7).any())


def test_identity_binder_deepembed_scale_is_zero_preserving_and_score_bound() -> None:
    torch.manual_seed(10)
    binder = StateIdentityBinder(8)
    native_scale = 1.0 + 0.25 * torch.randn(2, 3, 7)
    score = torch.randn(2, 3)
    gated = binder.gate_deepembed_scale(native_scale, score)
    assert gated.shape == native_scale.shape
    assert bool(torch.isfinite(gated).all())
    assert bool((gated - 1.0).abs().le((native_scale - 1.0).abs() + 1e-6).all())
    zero_scale = binder.gate_deepembed_scale(torch.ones_like(native_scale), score)
    assert torch.equal(zero_scale, torch.ones_like(native_scale))


def test_identity_binder_deepembed_scale_rejects_mismatched_score_shape() -> None:
    binder = StateIdentityBinder(4)
    native_scale = torch.ones(2, 3, 5)
    with pytest.raises(ValueError, match="scale and identity score"):
        binder.gate_deepembed_scale(native_scale, torch.ones(2, 4))


def test_ple_gate_only_gates_ple_delta_without_attention_correction() -> None:
    module, layer = _ple_module()
    layer.self_attn = module
    base_fuse = module._fuse_projected_rwkv_reads
    install(layer, mode="ple_gate")
    projected = torch.randn(1, 2, module.state_read_dim)
    recurrent = torch.randn_like(projected)
    global_recurrent = torch.randn_like(projected)
    hidden = torch.randn(1, 2, module.hidden_size)
    module.rwkv_query_state_identity_query_address = torch.randn_like(projected)
    expected = base_fuse(
        projected,
        recurrent,
        global_recurrent_reads=global_recurrent,
        hidden_states=hidden,
    )
    fused = module._fuse_projected_rwkv_reads(
        projected,
        recurrent,
        global_recurrent_reads=global_recurrent,
        hidden_states=hidden,
    )
    assert module.rwkv_identity_last_correction is None
    torch.testing.assert_close(fused, expected)
