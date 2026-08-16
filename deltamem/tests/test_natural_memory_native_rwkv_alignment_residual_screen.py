from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_alignment_residual_screen as screen,
)


def test_protocol_locks_alignment_residual_after_vector_partial_gain() -> None:
    protocol = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        screen.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["architecture"]["hybrid_mode"] == "alignment_residual"
    assert protocol["architecture"]["hybrid_gain"] == 0.125
    assert "layer_permuted_recurrent" in protocol["causal_conditions"]
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_selected_candidate_matches_locked_runtime_contract() -> None:
    assert screen.SELECTED_CANDIDATE == {
        "candidate_id": "alignment_residual_t16_k2_gate025_g0125",
        "hybrid_mode": "alignment_residual",
        "hybrid_gain": 0.125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    }


def test_local_evidence_requires_zero_donor_and_layer_controls(monkeypatch) -> None:
    states = {
        "correct": {"marker": torch.tensor(1.0)},
        "zero": {"marker": torch.tensor(0.0)},
        "matched_donor": {"marker": torch.tensor(2.0)},
        "layer_permuted": {"marker": torch.tensor(3.0)},
    }
    base = torch.zeros(1, 1, 4)

    def fake_read_logits(model, batch, state, *, readout_mode, **kwargs):
        if readout_mode == "projected_kv_slots":
            return base
        return base + float(state["marker"].item()) * 0.01

    monkeypatch.setattr(screen.hybrid_screen, "read_logits", fake_read_logits)

    evidence = screen.local_evidence(object(), object(), states)

    assert evidence["passed"] is True
    assert evidence["checks"] == {
        "zero_recurrent_exactly_equals_projected_only": True,
        "correct_vs_zero_material": True,
        "correct_vs_matched_donor_material": True,
        "correct_vs_layer_permuted_material": True,
        "correct_vs_projected_bounded": True,
        "all_condition_logits_finite": True,
    }
