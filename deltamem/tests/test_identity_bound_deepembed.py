import torch

from experiments.rethinking_rwkv_ms_gemma.identity_bound_deepembed import (
    StateIdentityBinder,
)


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
