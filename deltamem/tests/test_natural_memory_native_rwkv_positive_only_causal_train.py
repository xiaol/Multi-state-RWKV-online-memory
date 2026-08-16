from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_positive_only_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def test_protocol_and_spsa_failure_binding_validate() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["training"]["control_branch_backward_calls"] == 0
    assert protocol["heldout_causal_endpoint"]["source_ordinals"] == list(
        training.HELDOUT_ORDINALS
    )


def test_training_bindings_replace_objective_and_endpoint_then_restore() -> None:
    original_function = shared.TRAINING_FUNCTION
    original_ordinals = shared.HELDOUT_ORDINALS
    original_hash = shared.HELDOUT_PAYLOAD_SHA256

    with training.training_bindings():
        assert shared.TRAINING_FUNCTION is training.train_positive_only
        assert shared.HELDOUT_ORDINALS == training.HELDOUT_ORDINALS
        assert shared.HELDOUT_PAYLOAD_SHA256 == training.HELDOUT_PAYLOAD_SHA256

    assert shared.TRAINING_FUNCTION is original_function
    assert shared.HELDOUT_ORDINALS == original_ordinals
    assert shared.HELDOUT_PAYLOAD_SHA256 == original_hash


def test_finite_row_accumulation_never_copies_nonfinite_gradients() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    named = (("first", first), ("second", second))
    clean: dict[str, torch.Tensor] = {}

    first.grad = torch.tensor([0.25, -0.5])
    second.grad = torch.tensor([0.75])
    finite = training.accumulate_finite_row_gradients(named, clean)

    assert finite["passed"] is True
    assert torch.equal(clean["first"], torch.tensor([0.25, -0.5]))
    assert torch.equal(clean["second"], torch.tensor([0.75]))

    first.grad = torch.tensor([float("nan"), 100.0])
    second.grad = torch.tensor([100.0])
    nonfinite = training.accumulate_finite_row_gradients(named, clean)

    assert nonfinite["passed"] is False
    assert torch.equal(clean["first"], torch.tensor([0.25, -0.5]))
    assert torch.equal(clean["second"], torch.tensor([0.75]))

    training.materialize_clean_gradients(named, clean, scale=2.0)
    assert torch.equal(first.grad, torch.tensor([0.5, -1.0]))
    assert torch.equal(second.grad, torch.tensor([1.5]))
