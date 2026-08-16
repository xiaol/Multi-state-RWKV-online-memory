from __future__ import annotations

import json

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_affine_causal_train as training,
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
