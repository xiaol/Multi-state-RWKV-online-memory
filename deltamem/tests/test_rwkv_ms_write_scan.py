from __future__ import annotations

import pytest
import torch

from deltamem.kernels.rwkv_ms_write_scan import (
    _reference,
    fused_write_scan,
    write_slot_indices,
)


def test_write_scan_slot_indices_ignore_padding_across_boundary() -> None:
    positions = torch.tensor([1022, 2047])
    mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )

    slots = write_slot_indices(
        mask,
        batch_size=2,
        seq_len=5,
        positions=positions,
        chunk_size=1024,
        num_slots=4,
    )

    assert slots.tolist() == [[0, -1, 0, 1, -1], [-1, 1, 2, -1, 2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_write_scan_matches_reference_output_and_gradients() -> None:
    torch.manual_seed(123)
    batch_size, seq_len, num_heads, num_slots, rank = 2, 9, 1, 4, 32
    positions = torch.tensor([0, 1022], device="cuda")
    mask = torch.tensor(
        [
            [True, True, False, True, True, True, False, True, True],
            [True, False, True, True, True, False, True, True, True],
        ],
        device="cuda",
    )
    slots = write_slot_indices(
        mask,
        batch_size=batch_size,
        seq_len=seq_len,
        positions=positions,
        chunk_size=1024,
        num_slots=num_slots,
    )

    def feature() -> torch.Tensor:
        return (
            torch.randn(batch_size, seq_len, num_heads, rank, device="cuda")
            * 0.1
        ).requires_grad_()

    state = (
        torch.randn(
            batch_size,
            num_heads,
            num_slots,
            rank,
            rank,
            device="cuda",
        )
        * 0.1
    ).requires_grad_()
    w = torch.sigmoid(feature())
    k, v, a, b = feature(), feature(), feature(), feature()
    keep = torch.sigmoid(feature())
    erase = torch.sigmoid(feature())
    write = torch.sigmoid(feature())
    inputs = (state, w, k, v, a, b, keep, erase, write)

    reference = _reference(*inputs, slots, 1.0)
    output_weight = torch.randn_like(reference)
    reference_gradients = torch.autograd.grad(
        (reference * output_weight).sum(), inputs, retain_graph=True
    )
    fused = fused_write_scan(*inputs, slots, 1.0)
    fused_gradients = torch.autograd.grad((fused * output_weight).sum(), inputs)

    assert torch.allclose(fused, reference, atol=2e-6, rtol=2e-5)
    for fused_gradient, reference_gradient in zip(
        fused_gradients, reference_gradients
    ):
        assert torch.allclose(
            fused_gradient,
            reference_gradient,
            atol=5e-5,
            rtol=3e-4,
        ), float((fused_gradient - reference_gradient).abs().max())
