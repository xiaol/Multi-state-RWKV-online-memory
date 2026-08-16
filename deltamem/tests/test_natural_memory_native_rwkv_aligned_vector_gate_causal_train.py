from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_aligned_vector_gate_causal_train as aligned,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_binds_screen_and_new_endpoint() -> None:
    protocol = aligned.validate_protocol()
    screen_result = aligned.validate_screen_result()

    assert protocol["receipt"]["payload_sha256"] == aligned.PROTOCOL_PAYLOAD_SHA256
    assert protocol["authorization_basis"]["screen_result_file_sha256"] == (
        aligned.SCREEN_RESULT_FILE_SHA256
    )
    assert screen_result["receipt"]["payload_sha256"] == (
        aligned.SCREEN_RESULT_RECEIPT
    )
    assert protocol["heldout_causal_endpoint"]["candidate_rows_after_exclusions"] == 618
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        aligned.HELDOUT_ORDINALS
    )
    assert protocol["heldout_causal_endpoint"]["source_donor_payload_sha256"] == (
        aligned.HELDOUT_PAYLOAD_SHA256
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_bindings_replace_all_run_inputs_then_restore() -> None:
    names = (
        "MODEL_LOADER",
        "SELECTED_CANDIDATE",
        "TRAINING_FUNCTION",
        "CALIBRATION_RESULT",
        "CALIBRATION_RESULT_FILE_SHA256",
        "CALIBRATION_RESULT_RECEIPT",
        "validate_calibration_result",
    )
    original_shared = {name: getattr(shared, name) for name in names}
    original_filter = causal.FILTER_NONFINITE_ROWS

    with aligned.training_bindings():
        assert shared.MODEL_LOADER is aligned.load_model
        assert shared.SELECTED_CANDIDATE == aligned.SELECTED_CANDIDATE
        assert shared.TRAINING_FUNCTION is causal.train
        assert shared.CALIBRATION_RESULT == aligned.SCREEN_RESULT
        assert (
            shared.CALIBRATION_RESULT_FILE_SHA256
            == aligned.SCREEN_RESULT_FILE_SHA256
        )
        assert shared.CALIBRATION_RESULT_RECEIPT == aligned.SCREEN_RESULT_RECEIPT
        assert shared.validate_calibration_result is aligned.validate_screen_result
        assert causal.FILTER_NONFINITE_ROWS is True

    assert {name: getattr(shared, name) for name in names} == original_shared
    assert causal.FILTER_NONFINITE_ROWS is original_filter
