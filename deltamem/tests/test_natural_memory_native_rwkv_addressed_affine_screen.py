from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_affine_screen as screen,
)


RESULT = (
    Path(screen.__file__).resolve().parent
    / "local_artifacts/natural_memory_native_rwkv_addressed_affine_screen_v1/"
    "result.json"
)


def test_protocol_locks_addressed_affine_and_open_rows() -> None:
    protocol = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["hybrid_mode"] == "addressed_affine"
    assert protocol["architecture"]["hybrid_gain"] == 0.125
    assert protocol["architecture"]["recurrent_residual_ratio"] == 0.25
    assert protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_candidate_locks_addressed_affine() -> None:
    assert screen.SELECTED_CANDIDATE == {
        "candidate_id": "addressed_affine_t16_k2_gate025_g0125_r025",
        "hybrid_mode": "addressed_affine",
        "hybrid_gain": 0.125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    }


def test_screen_bindings_replace_and_restore_shared_contract() -> None:
    original_candidate = screen.shared.SELECTED_CANDIDATE
    original_validator = screen.shared.validate_protocol

    with screen.screen_bindings():
        assert screen.shared.SELECTED_CANDIDATE is screen.SELECTED_CANDIDATE
        assert screen.shared.validate_protocol is screen.validate_protocol
        assert str(screen.shared.RUNNER_BINDING_PATH) == screen.__file__

    assert screen.shared.SELECTED_CANDIDATE is original_candidate
    assert screen.shared.validate_protocol is original_validator


def test_signed_screen_result_authorizes_only_causal_training() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert screen.shared.sha256_file(RESULT) == (
        "3950c87909755234dec21880aa59a3fac7a8700717527500084e834385fae234"
    )
    assert screen.shared.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "fd96a0f8b5427cce1453773a07b6eb4071cbdce371f47a2508e8ea40d26b2d85"
    )
    assert result["status"] == screen.PASS_STATUS
    assert result["passed"] is True
    assert result["training_authorized"] is True
    assert result["native_generation_authorized"] is False
    assert result["protected_splits_opened"] == []
    assert result["checks"] == {
        "candidate_passed_on_all_ranks": True,
        "four_distinct_a100_ranks": True,
        "projected_carrier_fixed_on_all_ranks": True,
    }
    assert len(result["rank_evidence"]) == 4
    assert all(row["passed"] for row in result["rank_evidence"])
