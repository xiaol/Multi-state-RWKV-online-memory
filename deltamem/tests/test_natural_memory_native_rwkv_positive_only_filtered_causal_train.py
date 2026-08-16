from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_positive_only_causal_train as positive,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_positive_only_filtered_causal_train as filtered,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_binds_failure_filter_and_unopened_endpoint() -> None:
    protocol = filtered.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == filtered.PROTOCOL_PAYLOAD_SHA256
    assert (
        protocol["training"]["minimum_accepted_rows_per_update"]
        == filtered.MIN_ACCEPTED_ROWS_PER_UPDATE
    )
    assert (
        protocol["training"]["maximum_total_rejected_rows"]
        == filtered.MAX_TOTAL_REJECTED_ROWS
    )
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        positive.HELDOUT_ORDINALS
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_training_bindings_enable_filter_then_restore() -> None:
    original_shared_protocol = shared.PROTOCOL_PAYLOAD_SHA256
    original_filter = positive.FILTER_NONFINITE_ROWS
    original_minimum = positive.MIN_ACCEPTED_ROWS_PER_UPDATE
    original_maximum = positive.MAX_TOTAL_REJECTED_ROWS

    with filtered.training_bindings():
        assert shared.TRAINING_FUNCTION is positive.train_positive_only
        assert shared.PROTOCOL_PAYLOAD_SHA256 == filtered.PROTOCOL_PAYLOAD_SHA256
        assert positive.PROTOCOL_PAYLOAD_SHA256 == filtered.PROTOCOL_PAYLOAD_SHA256
        assert positive.FILTER_NONFINITE_ROWS is True
        assert (
            positive.MIN_ACCEPTED_ROWS_PER_UPDATE
            == filtered.MIN_ACCEPTED_ROWS_PER_UPDATE
        )
        assert (
            positive.MAX_TOTAL_REJECTED_ROWS
            == filtered.MAX_TOTAL_REJECTED_ROWS
        )

    assert shared.PROTOCOL_PAYLOAD_SHA256 == original_shared_protocol
    assert positive.FILTER_NONFINITE_ROWS is original_filter
    assert positive.MIN_ACCEPTED_ROWS_PER_UPDATE == original_minimum
    assert positive.MAX_TOTAL_REJECTED_ROWS == original_maximum
