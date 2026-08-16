from __future__ import annotations

import math

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_spsa_gate_causal_train as training,
)


def test_protocol_and_failure_bindings_validate() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["training"]["autograd_backward_calls"] == 0


def test_spsa_direction_is_deterministic_rademacher() -> None:
    first = training.spsa_direction(1, 42)
    repeated = training.spsa_direction(1, 42)
    second = training.spsa_direction(2, 42)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)
    assert set(first.tolist()) == {-1.0, 1.0}


def test_spsa_gradient_uses_two_sided_difference() -> None:
    direction = torch.tensor([1.0, -1.0, 1.0])
    gradient = training.estimated_spsa_gradient(3.2, 3.0, direction)

    assert torch.allclose(gradient, direction)


def test_causal_hinge_objective_only_penalizes_weak_controls() -> None:
    objective = training.causal_hinge_objective(
        correct_ce=2.0,
        zero_ce=2.1,
        donor_ce=2.02,
        permuted_ce=1.99,
    )

    expected = 2.0 + 0.25 * ((0.05 - 0.02) + (0.05 - -0.01))
    assert math.isclose(objective, expected, rel_tol=0.0, abs_tol=1e-12)
