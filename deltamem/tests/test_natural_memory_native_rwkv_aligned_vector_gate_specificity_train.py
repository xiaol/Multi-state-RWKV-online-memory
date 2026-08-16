from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_aligned_vector_gate_specificity_train as specificity,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_binds_prior_failure_and_fresh_endpoint() -> None:
    protocol = specificity.validate_protocol()
    prior = specificity.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == (
        specificity.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["authorization_basis"]["prior_result_file_sha256"] == (
        specificity.PRIOR_RESULT_FILE_SHA256
    )
    assert prior["receipt"]["payload_sha256"] == specificity.PRIOR_RESULT_RECEIPT
    assert protocol["heldout_causal_endpoint"]["candidate_rows_after_exclusions"] == 412
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        specificity.HELDOUT_ORDINALS
    )
    assert protocol["heldout_causal_endpoint"]["source_donor_payload_sha256"] == (
        specificity.HELDOUT_PAYLOAD_SHA256
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_bindings_strengthen_specificity_then_restore() -> None:
    shared_names = (
        "UPDATES",
        "TRAINING_PREFIX_SHA256",
        "HELDOUT_ORDINALS",
        "HELDOUT_PAYLOAD_SHA256",
        "TRAINING_FUNCTION",
        "MODEL_LOADER",
        "validate_protocol",
    )
    causal_names = (
        "TRAIN_UPDATES",
        "CONTRAST_WEIGHT",
        "FILTER_NONFINITE_ROWS",
        "MIN_ACCEPTED_ROWS_PER_UPDATE",
        "MAX_TOTAL_REJECTED_ROWS",
    )
    original_shared = {name: getattr(shared, name) for name in shared_names}
    original_causal = {name: getattr(causal, name) for name in causal_names}

    with specificity.training_bindings():
        assert shared.UPDATES == 16
        assert shared.TRAINING_PREFIX_SHA256 == specificity.TRAINING_PREFIX_SHA256
        assert shared.HELDOUT_ORDINALS == specificity.HELDOUT_ORDINALS
        assert shared.TRAINING_FUNCTION is causal.train
        assert shared.MODEL_LOADER is specificity.aligned.load_model
        assert causal.TRAIN_UPDATES == 16
        assert causal.CONTRAST_WEIGHT == 1.0
        assert causal.FILTER_NONFINITE_ROWS is True
        assert causal.MIN_ACCEPTED_ROWS_PER_UPDATE == 6
        assert causal.MAX_TOTAL_REJECTED_ROWS == 8

    assert {name: getattr(shared, name) for name in shared_names} == original_shared
    assert {name: getattr(causal, name) for name in causal_names} == original_causal
