from __future__ import annotations

import pytest
import torch

from deltamem.kernels.rwkv_ms_write_scan import rwkv_ms_write_scan


DTYPES = (torch.float32, torch.bfloat16)


def _random_tensor(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * scale).to(dtype=dtype)


def _sign_codes(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    samples = torch.randn(shape, generator=generator)
    return torch.where(samples.ge(0), 1.0, -1.0).to(dtype=dtype)


def _encode_state(
    state: torch.Tensor,
    left_codes: torch.Tensor,
    right_codes: torch.Tensor,
) -> torch.Tensor:
    return state * left_codes.unsqueeze(-1) * right_codes.unsqueeze(-2)


def _decode_state(
    state: torch.Tensor,
    left_codes: torch.Tensor,
    right_codes: torch.Tensor,
) -> torch.Tensor:
    return state * left_codes.unsqueeze(-1) * right_codes.unsqueeze(-2)


def _selected_codes(codes: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, _, rank = codes.shape
    gather_indices = slots.clamp_min(0).view(batch_size, 1, -1, 1).expand(
        -1, num_heads, -1, rank
    )
    return codes.gather(2, gather_indices).permute(0, 2, 1, 3)


def _state_reads(
    state: torch.Tensor,
    query: torch.Tensor,
    *,
    left_decode_codes: torch.Tensor,
    right_query_codes: torch.Tensor,
) -> torch.Tensor:
    coded_query = query.unsqueeze(3) * right_query_codes.unsqueeze(1)
    raw_reads = torch.einsum("bhsij,bthsj->bthsi", state, coded_query)
    return raw_reads * left_decode_codes.unsqueeze(1)


def _scan_case(dtype: torch.dtype) -> dict[str, torch.Tensor | float]:
    generator = torch.Generator(device="cpu").manual_seed(1701)
    batch_size, sequence_length = 2, 7
    num_heads, num_slots, rank = 2, 3, 8
    state_shape = (batch_size, num_heads, num_slots, rank, rank)
    feature_shape = (batch_size, sequence_length, num_heads, rank)
    code_shape = (batch_size, num_heads, num_slots, rank)

    state = _random_tensor(
        state_shape, generator=generator, dtype=dtype, scale=0.25
    )
    decay = torch.sigmoid(
        _random_tensor(feature_shape, generator=generator, dtype=torch.float32)
    ).to(dtype=dtype)
    key = _random_tensor(feature_shape, generator=generator, dtype=dtype)
    value = _random_tensor(feature_shape, generator=generator, dtype=dtype)
    correction_key = _random_tensor(
        feature_shape, generator=generator, dtype=dtype
    )
    correction_value = _random_tensor(
        feature_shape, generator=generator, dtype=dtype
    )
    keep = (
        0.1
        + 0.8
        * torch.sigmoid(
            _random_tensor(feature_shape, generator=generator, dtype=torch.float32)
        )
    ).to(dtype=dtype)
    erase = (
        0.1
        + 0.8
        * torch.sigmoid(
            _random_tensor(feature_shape, generator=generator, dtype=torch.float32)
        )
    ).to(dtype=dtype)
    write = (
        0.1
        + 0.8
        * torch.sigmoid(
            _random_tensor(feature_shape, generator=generator, dtype=torch.float32)
        )
    ).to(dtype=dtype)
    slots = torch.tensor(
        [[0, 1, 2, -1, 0, 2, 1], [2, 2, -1, 1, 0, 1, 0]],
        dtype=torch.long,
    )
    left_codes = _sign_codes(
        code_shape, generator=generator, dtype=dtype
    )
    right_codes = _sign_codes(
        code_shape, generator=generator, dtype=dtype
    )
    query = _random_tensor(
        feature_shape, generator=generator, dtype=dtype
    )
    return {
        "state": state,
        "decay": decay,
        "key": key,
        "value": value,
        "correction_key": correction_key,
        "correction_value": correction_value,
        "keep": keep,
        "erase": erase,
        "write": write,
        "slots": slots,
        "left_codes": left_codes,
        "right_codes": right_codes,
        "query": query,
        "erase_gate": 0.625,
    }


def _run_baseline(case: dict[str, torch.Tensor | float]) -> torch.Tensor:
    return rwkv_ms_write_scan(
        case["state"],
        case["decay"],
        case["key"],
        case["value"],
        case["correction_key"],
        case["correction_value"],
        case["keep"],
        case["erase"],
        case["write"],
        case["slots"],
        case["erase_gate"],
    )


def _run_bound(case: dict[str, torch.Tensor | float]) -> torch.Tensor:
    left_codes = case["left_codes"]
    right_codes = case["right_codes"]
    slots = case["slots"]
    selected_left = _selected_codes(left_codes, slots)
    selected_right = _selected_codes(right_codes, slots)
    return rwkv_ms_write_scan(
        _encode_state(case["state"], left_codes, right_codes),
        case["decay"],
        case["key"] * selected_right,
        case["value"] * selected_left,
        case["correction_key"] * selected_right,
        case["correction_value"] * selected_right,
        case["keep"],
        case["erase"],
        case["write"],
        slots,
        case["erase_gate"],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_multitoken_multislot_two_axis_scan_commutes_exactly(
    dtype: torch.dtype,
) -> None:
    case = _scan_case(dtype)
    baseline = _run_baseline(case)
    bound = _run_bound(case)
    decoded = _decode_state(
        bound,
        case["left_codes"],
        case["right_codes"],
    )

    torch.testing.assert_close(decoded, baseline, atol=0.0, rtol=0.0)

    baseline_reads = torch.einsum(
        "bhsij,bthj->bthsi", baseline, case["query"]
    )
    decoded_reads = _state_reads(
        bound,
        case["query"],
        left_decode_codes=case["left_codes"],
        right_query_codes=case["right_codes"],
    )
    torch.testing.assert_close(decoded_reads, baseline_reads, atol=0.0, rtol=0.0)

    no_correction = rwkv_ms_write_scan(
        case["state"],
        case["decay"],
        case["key"],
        case["value"],
        case["correction_key"],
        case["correction_value"],
        case["keep"],
        torch.zeros_like(case["erase"]),
        case["write"],
        case["slots"],
        case["erase_gate"],
    )
    assert not torch.equal(no_correction, baseline)


@pytest.mark.parametrize("dtype", DTYPES)
def test_wrong_left_right_and_both_one_bit_codes_perturb_independently(
    dtype: torch.dtype,
) -> None:
    case = _scan_case(dtype)
    baseline = _run_baseline(case)
    bound = _run_bound(case)
    left_codes = case["left_codes"]
    right_codes = case["right_codes"]

    wrong_left = left_codes.clone()
    wrong_left[..., 0] = -wrong_left[..., 0]
    wrong_right = right_codes.clone()
    wrong_right[..., 1] = -wrong_right[..., 1]

    left_decoded = _decode_state(bound, wrong_left, right_codes)
    right_decoded = _decode_state(bound, left_codes, wrong_right)
    both_decoded = _decode_state(bound, wrong_left, wrong_right)

    expected_left_changes = torch.zeros_like(baseline, dtype=torch.bool)
    expected_left_changes[..., 0, :] = True
    expected_right_changes = torch.zeros_like(baseline, dtype=torch.bool)
    expected_right_changes[..., :, 1] = True
    expected_both_changes = expected_left_changes.logical_xor(
        expected_right_changes
    )

    assert torch.equal(left_decoded.ne(baseline), expected_left_changes)
    assert torch.equal(right_decoded.ne(baseline), expected_right_changes)
    assert torch.equal(both_decoded.ne(baseline), expected_both_changes)

    baseline_reads = torch.einsum(
        "bhsij,bthj->bthsi", baseline, case["query"]
    )
    left_reads = _state_reads(
        bound,
        case["query"],
        left_decode_codes=wrong_left,
        right_query_codes=right_codes,
    )
    right_reads = _state_reads(
        bound,
        case["query"],
        left_decode_codes=left_codes,
        right_query_codes=wrong_right,
    )
    both_reads = _state_reads(
        bound,
        case["query"],
        left_decode_codes=wrong_left,
        right_query_codes=wrong_right,
    )

    assert not torch.equal(left_reads, baseline_reads)
    assert not torch.equal(right_reads, baseline_reads)
    assert not torch.equal(both_reads, baseline_reads)
    assert not torch.equal(left_reads, right_reads)
    assert not torch.equal(left_reads, both_reads)
    assert not torch.equal(right_reads, both_reads)


@pytest.mark.parametrize("dtype", DTYPES)
def test_two_axis_involution_restores_signed_zero_bits(dtype: torch.dtype) -> None:
    state = torch.tensor(
        [[[[[0.0, -0.0], [1.0, -1.0]]]]],
        dtype=dtype,
    )
    left_codes = torch.tensor([[[[-1.0, 1.0]]]], dtype=dtype)
    right_codes = torch.tensor([[[[1.0, -1.0]]]], dtype=dtype)

    decoded = _decode_state(
        _encode_state(state, left_codes, right_codes),
        left_codes,
        right_codes,
    )

    assert torch.equal(decoded, state)
    assert torch.equal(torch.signbit(decoded), torch.signbit(state))
