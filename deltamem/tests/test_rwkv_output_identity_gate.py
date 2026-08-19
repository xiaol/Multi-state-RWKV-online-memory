from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma.rwkv_output_identity_gate import (
    BoundedOutputIdentityGate,
    architecture_payload,
)


def test_zero_recurrent_correction_is_exact_projected_only() -> None:
    torch.manual_seed(113)
    gate = BoundedOutputIdentityGate(32)
    projected = torch.randn(2, 3, 32)
    output, values = gate(
        projected,
        torch.zeros_like(projected),
        torch.randn_like(projected),
        torch.randn_like(projected),
    )

    assert torch.equal(output, projected.float())
    assert bool(values.ge(0).all())
    assert bool(values.le(gate.max_gate).all())


def test_initial_identity_maps_and_output_coupling_gradients() -> None:
    torch.manual_seed(114)
    gate = BoundedOutputIdentityGate(32)
    reference = torch.randn(4, 32)
    audit = gate.audit(reference)

    assert audit["initialized_query_map_exact_identity"] is True
    assert audit["initialized_state_map_exact_identity"] is True
    assert audit["zero_recurrent_exact_projected_only"] is True
    assert 0.0 < audit["initial_gate_min"] <= audit["initial_gate_max"] <= 0.25

    output, _ = gate(
        reference,
        torch.randn_like(reference),
        torch.randn_like(reference),
        torch.randn_like(reference),
    )
    output.square().mean().backward()
    assert gate.identity.q_up.grad is not None
    assert gate.identity.s_up.grad is not None
    assert bool(torch.isfinite(gate.identity.q_up.grad).all())
    assert bool(torch.isfinite(gate.identity.s_up.grad).all())


def test_architecture_is_explicitly_output_coupled() -> None:
    payload = architecture_payload()
    assert payload["output_coupled"] is True
    assert "zero recurrent correction" in payload["zero_recurrent_identity"]
