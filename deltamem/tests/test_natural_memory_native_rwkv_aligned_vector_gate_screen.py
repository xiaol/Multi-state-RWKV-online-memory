from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_aligned_vector_gate_screen as screen,
)


def test_protocol_locks_aligned_vector_gate_after_alignment_failure() -> None:
    protocol = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        screen.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["architecture"]["hybrid_mode"] == "aligned_vector_gate"
    assert protocol["architecture"]["hybrid_gain"] == 0.125
    assert protocol["authorization_basis"]["prior_result_receipt"] == (
        screen.PRIOR_RESULT_RECEIPT
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_selected_candidate_matches_locked_runtime_contract() -> None:
    assert screen.SELECTED_CANDIDATE == {
        "candidate_id": "aligned_vector_gate_t16_k2_gate025_g0125",
        "hybrid_mode": "aligned_vector_gate",
        "hybrid_gain": 0.125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    }


def test_screen_bindings_configure_candidate_then_restore() -> None:
    names = (
        "SCHEMA",
        "PROTOCOL",
        "PROTOCOL_PAYLOAD_SHA256",
        "PRIOR_RESULT",
        "PRIOR_RESULT_FILE_SHA256",
        "PRIOR_RESULT_RECEIPT",
        "SEED",
        "SELECTED_CANDIDATE",
        "PASS_STATUS",
        "FAIL_STATUS",
        "MODEL_AUDIT_KEY",
        "PRIOR_RESULT_CODE_BINDING_KEY",
        "RUNNER_BINDING_PATH",
        "validate_protocol",
    )
    original = {name: getattr(screen.shared, name) for name in names}

    with screen.screen_bindings():
        assert screen.shared.SELECTED_CANDIDATE == screen.SELECTED_CANDIDATE
        assert screen.shared.PASS_STATUS == screen.PASS_STATUS
        assert screen.shared.RUNNER_BINDING_PATH == screen.Path(screen.__file__)
        assert screen.shared.validate_protocol is screen.validate_protocol

    assert {name: getattr(screen.shared, name) for name in names} == original


def test_shared_evidence_uses_bound_aligned_vector_gate(monkeypatch) -> None:
    states = {
        "correct": {"marker": torch.tensor(1.0)},
        "zero": {"marker": torch.tensor(0.0)},
        "matched_donor": {"marker": torch.tensor(2.0)},
        "layer_permuted": {"marker": torch.tensor(3.0)},
    }
    base = torch.zeros(1, 1, 4)
    modes: list[tuple[str, float]] = []

    def fake_read_logits(
        model,
        batch,
        state,
        *,
        readout_mode,
        hybrid_mode=None,
        hybrid_gain=None,
        **kwargs,
    ):
        if readout_mode == "projected_kv_slots":
            return base
        modes.append((hybrid_mode, hybrid_gain))
        return base + float(state["marker"].item()) * 0.01

    monkeypatch.setattr(screen.shared.hybrid_screen, "read_logits", fake_read_logits)

    with screen.screen_bindings():
        evidence = screen.shared.local_evidence(object(), object(), states)

    assert evidence["passed"] is True
    assert modes == [("aligned_vector_gate", 0.125)] * 4
