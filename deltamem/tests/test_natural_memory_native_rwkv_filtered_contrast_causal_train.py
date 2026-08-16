from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_filtered_contrast_causal_train as filtered,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_binds_fresh_endpoint_and_failed_positive_result() -> None:
    protocol = filtered.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == filtered.PROTOCOL_PAYLOAD_SHA256
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        filtered.HELDOUT_ORDINALS
    )
    assert protocol["heldout_causal_endpoint"]["source_donor_payload_sha256"] == (
        filtered.HELDOUT_PAYLOAD_SHA256
    )
    assert protocol["protected_splits_opened_by_this_protocol"] == []


def test_causal_row_filter_rejects_nan_without_contamination() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    named = (("first", first), ("second", second))
    clean: dict[str, torch.Tensor] = {}

    first.grad = torch.tensor([0.25, -0.5])
    second.grad = torch.tensor([0.75])
    finite = causal.accumulate_finite_row_gradients(named, clean)
    assert finite["passed"] is True

    first.grad = torch.tensor([float("nan"), 100.0])
    second.grad = torch.tensor([100.0])
    nonfinite = causal.accumulate_finite_row_gradients(named, clean)
    assert nonfinite["passed"] is False
    assert torch.equal(clean["first"], torch.tensor([0.25, -0.5]))
    assert torch.equal(clean["second"], torch.tensor([0.75]))

    causal.materialize_clean_gradients(named, clean, scale=2.0)
    assert torch.equal(first.grad, torch.tensor([0.5, -1.0]))
    assert torch.equal(second.grad, torch.tensor([1.5]))


def test_training_bindings_enable_causal_filter_then_restore() -> None:
    original_training = shared.TRAINING_FUNCTION
    original_filter = causal.FILTER_NONFINITE_ROWS
    original_minimum = causal.MIN_ACCEPTED_ROWS_PER_UPDATE
    original_maximum = causal.MAX_TOTAL_REJECTED_ROWS

    with filtered.training_bindings():
        assert shared.TRAINING_FUNCTION is causal.train
        assert causal.FILTER_NONFINITE_ROWS is True
        assert (
            causal.MIN_ACCEPTED_ROWS_PER_UPDATE
            == filtered.MIN_ACCEPTED_ROWS_PER_UPDATE
        )
        assert causal.MAX_TOTAL_REJECTED_ROWS == filtered.MAX_TOTAL_REJECTED_ROWS

    assert shared.TRAINING_FUNCTION is original_training
    assert causal.FILTER_NONFINITE_ROWS is original_filter
    assert causal.MIN_ACCEPTED_ROWS_PER_UPDATE == original_minimum
    assert causal.MAX_TOTAL_REJECTED_ROWS == original_maximum
