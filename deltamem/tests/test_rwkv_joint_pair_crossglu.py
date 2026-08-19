from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma.rwkv_joint_pair_crossglu import (
    JointPairGatedCrossGLU,
)


def test_zero_state_is_exact_projected_only() -> None:
    torch.manual_seed(41)
    bridge = JointPairGatedCrossGLU(32)
    projected = torch.randn(2, 7, 32)
    query = torch.randn(2, 7, 32)
    zero = torch.zeros_like(query)
    output, gate, value, correction = bridge(projected, query, zero)
    torch.testing.assert_close(output, projected, atol=0.0, rtol=0.0)
    assert bool(torch.isfinite(gate).all())
    assert bool(torch.equal(value, torch.zeros_like(value)))
    assert bool(torch.equal(correction, torch.zeros_like(correction)))


def test_pair_gate_changes_when_state_is_replaced() -> None:
    torch.manual_seed(42)
    bridge = JointPairGatedCrossGLU(32)
    projected = torch.randn(2, 7, 32)
    query = torch.randn(2, 7, 32)
    state = torch.randn(2, 7, 32)
    donor = torch.roll(state, shifts=1, dims=1)
    correct, correct_gate, _, _ = bridge(projected, query, state)
    wrong, wrong_gate, _, _ = bridge(projected, query, donor)
    assert bool(torch.isfinite(correct).all())
    assert bool((correct_gate - wrong_gate).abs().gt(1e-6).any())
    assert bool((correct - wrong).abs().gt(1e-6).any())


def test_gate_shuffle_and_fixed_gate_controls_are_distinct() -> None:
    torch.manual_seed(43)
    bridge = JointPairGatedCrossGLU(16)
    projected = torch.randn(2, 5, 16)
    query = torch.randn(2, 5, 16)
    state = torch.randn(2, 5, 16)
    _, gate, _, _ = bridge(projected, query, state)
    shuffled, _, _, _ = bridge(projected, query, state, gate_shuffle=True)
    fixed, _, _, _ = bridge(projected, query, state, gate_override=gate.roll(1, dims=-1))
    assert bool((shuffled - fixed).abs().max().lt(1e-6))
    assert bool((shuffled - projected).abs().gt(0.0).any())
