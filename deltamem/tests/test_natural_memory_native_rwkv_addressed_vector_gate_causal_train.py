from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_vector_gate_causal_train as training,
)


RESULT = (
    Path(training.__file__).resolve().parent
    / "local_artifacts/"
    "natural_memory_native_rwkv_addressed_vector_gate_causal_train_v1/"
    "result.json"
)


def test_protocol_locks_screen_training_and_fresh_endpoint() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        training.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["architecture"]["selected_candidate"] == (
        training.SELECTED_CANDIDATE
    )
    assert protocol["training"]["optimizer_updates"] == 16
    assert protocol["heldout_causal_endpoint"]["candidate_rows_after_exclusions"] == 315
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        training.HELDOUT_ORDINALS
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_signed_screen_result_authorizes_training() -> None:
    result = training.validate_screen_result()
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert training.shared.sha256_file(training.SCREEN_RESULT) == (
        training.SCREEN_RESULT_FILE_SHA256
    )
    assert training.shared.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == training.SCREEN_RESULT_RECEIPT
    assert result["status"] == training.screen.PASS_STATUS
    assert result["training_authorized"] is True
    assert result["protected_splits_opened"] == []


def test_training_bindings_replace_and_restore_shared_contract() -> None:
    original_candidate = training.shared.SELECTED_CANDIDATE
    original_loader = training.shared.MODEL_LOADER

    with training.training_bindings():
        assert training.shared.SELECTED_CANDIDATE is training.SELECTED_CANDIDATE
        assert training.shared.MODEL_LOADER is training.load_model
        assert training.shared.UPDATES == 16
        assert training.causal_train.CONTRAST_WEIGHT == 1.0

    assert training.shared.SELECTED_CANDIDATE is original_candidate
    assert training.shared.MODEL_LOADER is original_loader


def test_endpoint_payload_is_hash_bound_in_protocol() -> None:
    protocol = json.loads(training.PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["heldout_causal_endpoint"]["source_donor_payload_sha256"] == (
        training.HELDOUT_PAYLOAD_SHA256
    )
    assert protocol["frozen_inputs"]["sixteen_update_schedule_prefix_sha256"] == (
        training.TRAINING_PREFIX_SHA256
    )


def test_signed_result_locks_donor_failure_and_blocks_generation() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert training.shared.sha256_file(RESULT) == (
        "e9a115cd7864e0c31738478c0393aed21c6db40e3ab1adccc50ba76cb8a898e4"
    )
    assert training.shared.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "e13f1a45139e28178b0e7ca28b3b647d42a427a23180bc2b7f695fda9dc109c3"
    )
    assert result["protocol_payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert result["status"] == training.FAIL_STATUS
    assert result["training_passed"] is True
    assert result["passed"] is False
    assert result["open_native_generation_authorized"] is False
    assert result["protected_splits_opened"] == []
    assert result["training"]["updates"] == 16
    assert result["training"]["row_filter"] == {
        "accepted_gradient_rows": 128,
        "enabled": True,
        "maximum_total_rejected_rows": 8,
        "minimum_accepted_rows_per_update": 8,
        "minimum_required_accepted_rows_per_update": 6,
        "rejected_gradient_rows": 0,
        "rejected_source_ordinals": [],
    }
    endpoint = result["heldout_causal_endpoint"]
    assert endpoint["rows"] == 32
    assert endpoint["answer_target_tokens"] == 371
    assert endpoint["mean_ce_margins"] == {
        "donor_minus_correct": -0.0025936959567416373,
        "layer_permuted_minus_correct": 0.018406582649827197,
        "zero_minus_correct": 0.9114194854571815,
    }
    assert endpoint["checks"]["zero_minus_correct_mean_ce_positive"] is True
    assert endpoint["checks"][
        "layer_permuted_minus_correct_mean_ce_positive"
    ] is True
    assert endpoint["checks"]["donor_minus_correct_mean_ce_positive"] is False
