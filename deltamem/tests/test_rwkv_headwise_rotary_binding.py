from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma.rwkv_headwise_rotary_binding import (
    HeadwiseRotaryBinding,
    architecture_payload,
    parameter_count,
)


def test_matching_address_round_trip_is_invertible_and_norm_preserving() -> None:
    torch.manual_seed(201)
    binding = HeadwiseRotaryBinding(32, head_size=32)
    address = torch.randn(3, 5, 32)
    value = torch.randn(3, 5, 32)

    bound = binding.bind(address, value)
    restored = binding.unbind(address, bound)

    torch.testing.assert_close(restored, value, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        bound.square().sum(dim=-1),
        value.square().sum(dim=-1),
        atol=2e-5,
        rtol=2e-6,
    )


def test_binding_commutes_with_headwise_rwkv_outer_product_read() -> None:
    torch.manual_seed(202)
    batch_size, tokens, num_heads, head_size = 3, 5, 1, 32
    state_dim = num_heads * head_size
    binding = HeadwiseRotaryBinding(state_dim, head_size=head_size)
    row_address = torch.randn(batch_size, state_dim)
    write_address = row_address[:, None, :].expand(-1, tokens, -1)
    values = torch.randn(batch_size, tokens, state_dim)
    keys = torch.randn(batch_size, tokens, num_heads, head_size)
    query = torch.randn(batch_size, num_heads, head_size)

    bound_values = binding.bind(write_address, values).reshape(
        batch_size,
        tokens,
        num_heads,
        head_size,
    )
    bound_state = torch.einsum("bthi,bthj->bhij", bound_values, keys)
    bound_read = torch.einsum("bhij,bhj->bhi", bound_state, query).reshape(
        batch_size,
        state_dim,
    )
    decoded_read = binding.unbind(row_address, bound_read)

    values_by_head = values.reshape(batch_size, tokens, num_heads, head_size)
    reference_state = torch.einsum("bthi,bthj->bhij", values_by_head, keys)
    reference_read = torch.einsum("bhij,bhj->bhi", reference_state, query).reshape(
        batch_size,
        state_dim,
    )
    torch.testing.assert_close(decoded_read, reference_read, atol=4e-6, rtol=4e-6)


def test_binding_commutes_with_rwkv_right_linear_recurrence() -> None:
    torch.manual_seed(204)
    batch_size, width = 3, 32
    binding = HeadwiseRotaryBinding(width, head_size=width)
    address = torch.randn(batch_size, width)
    value = torch.randn(batch_size, width)
    key = torch.randn(batch_size, width)
    next_value = torch.randn(batch_size, width)
    next_key = torch.randn(batch_size, width)
    right_update = torch.randn(batch_size, width, width) / width**0.5
    query = torch.randn(batch_size, width)

    state = torch.einsum("bi,bj->bij", value, key)
    bound_state = torch.einsum("bi,bj->bij", binding.bind(address, value), key)
    next_state = (
        0.7 * state
        + 0.2 * torch.bmm(state, right_update)
        + torch.einsum("bi,bj->bij", next_value, next_key)
    )
    next_bound_state = (
        0.7 * bound_state
        + 0.2 * torch.bmm(bound_state, right_update)
        + torch.einsum(
            "bi,bj->bij",
            binding.bind(address, next_value),
            next_key,
        )
    )
    reference_read = torch.einsum("bij,bj->bi", next_state, query)
    bound_read = torch.einsum("bij,bj->bi", next_bound_state, query)

    torch.testing.assert_close(
        binding.unbind(address, bound_read),
        reference_read,
        atol=2e-5,
        rtol=2e-5,
    )


def test_matched_donor_uses_wrong_basis_and_projection_is_trainable() -> None:
    torch.manual_seed(203)
    binding = HeadwiseRotaryBinding(32, head_size=32)
    target_address = torch.randn(6, 32)
    donor_address = torch.roll(target_address, shifts=1, dims=0)
    value = torch.randn(6, 32)

    correct = binding.transfer(target_address, target_address, value)
    donor = binding.transfer(donor_address, target_address, value)
    distance = binding.phase_distance(target_address, donor_address)

    torch.testing.assert_close(correct, value, atol=2e-6, rtol=2e-6)
    assert float((donor - value).square().mean()) > 1e-3
    assert bool(distance.gt(0.0).all())
    (-distance.mean()).backward()
    assert binding.phase_projection.grad is not None
    assert bool(torch.isfinite(binding.phase_projection.grad).all())
    assert int(torch.count_nonzero(binding.phase_projection.grad)) > 0


def test_zero_contract_and_architecture_audit() -> None:
    binding = HeadwiseRotaryBinding(32, head_size=32)
    zero = torch.zeros(2, 32)
    assert torch.equal(binding.bind(zero, zero), zero)
    assert torch.equal(binding.unbind(zero, zero), zero)
    assert parameter_count(32, 32) == 512
    payload = architecture_payload()
    assert payload["head_size"] == 32
    assert payload["num_heads"] == 1
    assert payload["complex_pairs_per_head"] == 16
    assert payload["norm_preserving"] is True
    assert payload["scalar_gate"] is False
    assert payload["cosine_readout"] is False
    assert payload["parameters_per_layer"] == 512
