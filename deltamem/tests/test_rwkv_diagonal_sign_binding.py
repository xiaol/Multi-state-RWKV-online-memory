from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma.rwkv_diagonal_sign_binding import (
    DiagonalSignBinding,
    deterministic_projection,
    fit_address_only_projection,
)


def test_matching_address_round_trip_is_exact() -> None:
    binding = DiagonalSignBinding(32, projection=deterministic_projection(32, 7))
    address = torch.randn(3, 5, 32)
    value = torch.randn(3, 5, 32)
    restored = binding.transfer(address, address, value)
    torch.testing.assert_close(restored, value, atol=0.0, rtol=0.0)


def test_diagonal_code_commutes_with_full_rwkv_value_axis_updates() -> None:
    torch.manual_seed(31)
    binding = DiagonalSignBinding(32, projection=deterministic_projection(32, 8))
    address = torch.randn(32)
    value = torch.randn(32)
    next_value = torch.randn(32)
    key = torch.randn(32)
    next_key = torch.randn(32)
    a = torch.randn(32)
    b = torch.randn(32)
    keep = torch.rand(32)
    erase = torch.rand(32)
    write = torch.rand(32)
    state = torch.outer(value, key)
    correction = torch.mv(state, a).outer(b)
    next_state = keep[:, None] * state + write[:, None] * torch.outer(next_value, next_key)
    next_state = next_state + erase[:, None] * correction

    code = binding.codes(address)
    bound_state = torch.outer(code * value, key)
    bound_correction = torch.mv(bound_state, a).outer(b)
    bound_next = keep[:, None] * bound_state + write[:, None] * torch.outer(code * next_value, next_key)
    bound_next = bound_next + erase[:, None] * bound_correction
    decoded = code[:, None] * bound_next
    torch.testing.assert_close(decoded, next_state, atol=0.0, rtol=0.0)


def test_donor_code_changes_values_and_zero_contract_holds() -> None:
    binding = DiagonalSignBinding(32, projection=deterministic_projection(32, 9))
    target = torch.randn(64, 32)
    donor = torch.roll(target, shifts=1, dims=0)
    value = torch.randn(64, 32)
    donor_values = binding.transfer(donor, target, value)
    changed = (donor_values - value).norm(dim=-1).gt(0.05 * value.norm(dim=-1).clamp_min(1e-6))
    assert float(changed.float().mean()) > 0.95
    zero = torch.zeros(4, 32)
    assert torch.equal(binding.bind(zero, zero), zero)
    assert torch.equal(binding.unbind(zero, zero), zero)


def test_projection_is_trainable_for_address_only_fit() -> None:
    torch.manual_seed(33)
    binding = DiagonalSignBinding(32)
    left = torch.randn(16, 32)
    right = torch.roll(left, shifts=1, dims=0)
    loss = binding.logits(left).square().mean() + binding.logits(right).square().mean()
    loss.backward()
    assert binding.projection.grad is not None
    assert bool(torch.isfinite(binding.projection.grad).all())


def test_separation_hinge_has_address_only_gradient() -> None:
    torch.manual_seed(34)
    binding = DiagonalSignBinding(32)
    left = torch.randn(16, 32)
    right = torch.roll(left, shifts=1, dims=0)
    loss = binding.separation_hinge(left, right, margin=1.0)
    loss.backward()
    assert bool(torch.isfinite(loss).item())
    assert binding.projection.grad is not None
    assert int(torch.count_nonzero(binding.projection.grad)) > 0


def test_address_only_projection_fit_is_reproducible_and_detached() -> None:
    torch.manual_seed(35)
    left = torch.randn(24, 8)
    right = torch.roll(left, shifts=1, dims=0)
    first, first_metrics = fit_address_only_projection(left, right, steps=4, seed=18)
    second, second_metrics = fit_address_only_projection(left, right, steps=4, seed=18)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    assert first.requires_grad is False
    assert first_metrics == second_metrics
