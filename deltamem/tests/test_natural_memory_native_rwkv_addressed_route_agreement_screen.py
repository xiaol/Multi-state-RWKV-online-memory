from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_route_agreement_screen as screen,
)


def test_protocol_locks_query_verified_route_agreement() -> None:
    protocol = screen.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["hybrid_mode"] == "addressed_route_agreement"
    assert protocol["architecture"]["hybrid_gain"] == 0.125
    assert protocol["architecture"]["read_temperature"] == 16.0
    assert protocol["architecture"]["read_top_k"] == 2
    assert protocol["execution"]["hardware"] == "exactly four distinct A100 GPUs"
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_candidate_locks_query_verified_route_agreement() -> None:
    assert screen.SELECTED_CANDIDATE == {
        "candidate_id": "addressed_route_agreement_t16_k2_gate025_g0125",
        "hybrid_mode": "addressed_route_agreement",
        "hybrid_gain": 0.125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    }


def test_screen_bindings_replace_and_restore_shared_contract() -> None:
    original_candidate = screen.shared.SELECTED_CANDIDATE
    original_validator = screen.shared.validate_protocol
    original_loader = screen.shared.load_model

    with screen.screen_bindings():
        assert screen.shared.SELECTED_CANDIDATE is screen.SELECTED_CANDIDATE
        assert screen.shared.validate_protocol is screen.validate_protocol
        assert screen.shared.load_model is screen.load_model
        assert str(screen.shared.RUNNER_BINDING_PATH) == str(screen.RUNNER_BINDING_PATH)

    assert screen.shared.SELECTED_CANDIDATE is original_candidate
    assert screen.shared.validate_protocol is original_validator
    assert screen.shared.load_model is original_loader


def test_signed_screen_result_authorizes_only_causal_training() -> None:
    result_path = (
        Path(screen.__file__).resolve().parent
        / "local_artifacts/natural_memory_native_rwkv_addressed_route_agreement_screen_v1/"
        "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert screen.shared.sha256_file(result_path) == (
        "d23cad9a006f452b1fb61a640bdc03c82ef08d3d6ec3989ea47b1adbea610050"
    )
    assert screen.shared.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "d5dbb7f381e1630b2ed20ca8dba8ae84c99d3794353c910c2d9f580f59a203ab"
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
