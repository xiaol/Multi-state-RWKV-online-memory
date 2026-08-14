from __future__ import annotations

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_crossfit_router as router,
)


def test_crossfit_router_protocol_is_hash_bound_and_train_only() -> None:
    protocol = router.validate_protocol()

    assert protocol["source"]["rows"] == 284
    assert protocol["crossfit"]["folds"] == 5
    assert protocol["authorization"]["publisher_validation_authorized"] is False
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["protected_splits_opened"] == []


def test_crossfit_fold_assignment_is_deterministic_and_bounded() -> None:
    row_hash = "a" * 64

    assert router.fold_for_hash(row_hash) == router.fold_for_hash(row_hash)
    assert 0 <= router.fold_for_hash(row_hash) < router.FOLDS


@pytest.mark.parametrize(
    ("rule", "expected"),
    (
        ("frozen_v9", {1, 3}),
        ("checkpoint16", {1, 2}),
        ("intersection", {1}),
        ("union", {1, 2, 3}),
        ("min_cardinality_v9_tie", {1, 3}),
        ("max_cardinality_v9_tie", {1, 3}),
        ("checkpoint_if_subset_else_v9", {1, 3}),
        ("v9_if_subset_else_checkpoint", {1, 2}),
    ),
)
def test_crossfit_router_rule_semantics(rule: str, expected: set[int]) -> None:
    assert router.apply_rule(rule, {1, 3}, {1, 2}) == expected


def test_crossfit_router_missing_prediction_falls_back_to_available_input() -> None:
    assert router.apply_rule("intersection", None, {4}) == {4}
    assert router.apply_rule("union", {5}, None) == {5}
    assert router.apply_rule("checkpoint16", None, None) is None


def test_crossfit_router_selection_uses_locked_tie_break_order() -> None:
    scores = {rule: 0.2 for rule in router.RULES}
    scores["checkpoint16"] = 0.3
    scores["intersection"] = 0.3

    assert router.select_rule(scores) == "checkpoint16"


def _metrics(f1: float, coverage: float = 1.0) -> dict[str, float]:
    return {"micro_f1": f1, "coverage": coverage}


def test_crossfit_router_gate_requires_gain_over_both_inputs() -> None:
    passing = router.evaluate_gates(
        router_metrics=_metrics(0.31),
        frozen_v9_metrics=_metrics(0.30),
        checkpoint16_metrics=_metrics(0.305),
        worst_fold_delta=-0.01,
    )
    failing = router.evaluate_gates(
        router_metrics=_metrics(0.304),
        frozen_v9_metrics=_metrics(0.30),
        checkpoint16_metrics=_metrics(0.303),
        worst_fold_delta=-0.01,
    )

    assert passing["passed"] is True
    assert failing["router_minus_frozen_v9_micro_f1_at_least_0.005"] is False
    assert failing["router_minus_checkpoint16_micro_f1_at_least_0.005"] is False
    assert failing["passed"] is False
