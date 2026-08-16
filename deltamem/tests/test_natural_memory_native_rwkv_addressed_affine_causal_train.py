from __future__ import annotations

import json
from pathlib import Path

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_affine_causal_train as training,
)


RESULT = (
    Path(training.__file__).resolve().parent
    / "local_artifacts/"
    "natural_memory_native_rwkv_addressed_affine_causal_train_v1/"
    "result.json"
)


def test_protocol_locks_screen_training_and_fresh_endpoint() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["selected_candidate"] == (
        training.SELECTED_CANDIDATE
    )
    assert protocol["training"]["optimizer_updates"] == 16
    assert protocol["heldout_causal_endpoint"]["candidate_rows_after_exclusions"] == 229
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
    assert result["native_generation_authorized"] is False
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


def test_signed_result_authorizes_native_generation() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert training.shared.sha256_file(RESULT) == (
        "096e20bb01abbe86689745379b12b3f1b8d5de32c7be0ba682793855a85e0e2d"
    )
    assert training.shared.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert receipt["payload_sha256"] == (
        "c74dc75bee63bc3cae65671c0978f92969bb08e2e69d6fa1c8ea3c28c232a4e6"
    )
    assert result["protocol_payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert result["status"] == training.PASS_STATUS
    assert result["training_passed"] is True
    assert result["passed"] is True
    assert result["open_native_generation_authorized"] is True
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
    assert endpoint["answer_target_tokens"] == 404
    assert endpoint["mean_ce_margins"] == {
        "donor_minus_correct": 0.009505824287338704,
        "layer_permuted_minus_correct": 0.03961818997222588,
        "zero_minus_correct": 0.8378261056276828,
    }
    assert all(endpoint["checks"].values())
