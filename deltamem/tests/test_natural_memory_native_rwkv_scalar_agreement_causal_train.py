from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_scalar_agreement_causal_train as agreement,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_binds_scalar_hybrid_and_fresh_endpoint() -> None:
    protocol = agreement.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == agreement.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["selected_candidate"] == (
        agreement.SELECTED_CANDIDATE
    )
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        agreement.HELDOUT_ORDINALS
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_bindings_replace_loader_candidate_and_objective_then_restore() -> None:
    original_loader = shared.MODEL_LOADER
    original_candidate = shared.SELECTED_CANDIDATE
    original_training = shared.TRAINING_FUNCTION
    original_filter = causal.FILTER_NONFINITE_ROWS

    with agreement.training_bindings():
        assert shared.MODEL_LOADER is agreement.load_model
        assert shared.SELECTED_CANDIDATE == agreement.SELECTED_CANDIDATE
        assert shared.TRAINING_FUNCTION is causal.train
        assert causal.FILTER_NONFINITE_ROWS is True

    assert shared.MODEL_LOADER is original_loader
    assert shared.SELECTED_CANDIDATE == original_candidate
    assert shared.TRAINING_FUNCTION is original_training
    assert causal.FILTER_NONFINITE_ROWS is original_filter
