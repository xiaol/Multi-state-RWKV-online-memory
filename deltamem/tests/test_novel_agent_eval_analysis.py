from __future__ import annotations

from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import analyze_novel_agent_eval as analysis


def metric_names(kind: str) -> tuple[str, str]:
    if kind == "attribution":
        return "strict_accuracy", "format_recovered_accuracy"
    if kind == "narrative":
        return "strict_accuracy", "format_recovered_unit_accuracy"
    return "strict_f1", "format_recovered_micro_f1"


def score_from_contribution(kind: str, contribution: tuple[int, ...]) -> dict[str, Any]:
    if kind == "attribution":
        return {"correct": bool(contribution[0])}
    if kind == "narrative":
        return {
            "correct_units": contribution[0],
            "gold_units": contribution[1],
        }
    return {
        "tp": contribution[0],
        "fp": contribution[1],
        "fn": contribution[2],
    }


def build_inputs(
    specs: tuple[analysis.TaskSpec, ...],
    base_contributions: dict[str, list[tuple[int, ...]]],
    normal_contributions: dict[str, list[tuple[int, ...]]],
    *,
    base_predictions: dict[str, list[Any]] | None = None,
    normal_predictions: dict[str, list[Any]] | None = None,
    base_hits: dict[str, list[bool]] | None = None,
    normal_hits: dict[str, list[bool]] | None = None,
) -> dict[str, Any]:
    strict_summary: dict[str, Any] = {
        "conditions": {"base": {}, "normal": {}}
    }
    metrics: dict[str, dict[str, Any]] = {"base": {}, "normal": {}}
    predictions: dict[str, dict[str, list[Any]]] = {"base": {}, "normal": {}}
    contributions = {
        "base": base_contributions,
        "normal": normal_contributions,
    }
    records_by_condition: dict[str, dict[str, dict[str, Any]]] = {
        "base": {},
        "normal": {},
    }
    bootstraps: dict[str, dict[str, Any]] = {}
    for spec in specs:
        strict_name, recovered_name = metric_names(spec.kind)
        task_base = base_contributions[spec.name]
        task_normal = normal_contributions[spec.name]
        if len(task_base) != spec.expected_rows or len(task_normal) != spec.expected_rows:
            raise AssertionError("Synthetic contribution count differs from task spec")
        task_base_predictions = (
            base_predictions[spec.name]
            if base_predictions is not None and spec.name in base_predictions
            else [object() for _ in task_base]
        )
        task_normal_predictions = (
            normal_predictions[spec.name]
            if normal_predictions is not None and spec.name in normal_predictions
            else [object() for _ in task_normal]
        )
        predictions["base"][spec.name] = task_base_predictions
        predictions["normal"][spec.name] = task_normal_predictions
        for condition, task_contributions in (
            ("base", task_base),
            ("normal", task_normal),
        ):
            primary_metric = analysis.metric_from_contributions(
                spec.kind,
                task_contributions,
            )
            strict_summary["conditions"][condition][spec.name] = {
                "primary_metric": primary_metric,
                "primary_metric_name": strict_name,
            }
            metrics[condition][spec.name] = {
                "primary_metric": primary_metric,
                "primary_metric_name": recovered_name,
            }
            hit_values = (
                base_hits.get(spec.name, [False] * spec.expected_rows)
                if condition == "base" and base_hits is not None
                else normal_hits.get(spec.name, [False] * spec.expected_rows)
                if condition == "normal" and normal_hits is not None
                else [False] * spec.expected_rows
            )
            for index, (contribution, hit_max) in enumerate(
                zip(task_contributions, hit_values, strict=True)
            ):
                records_by_condition[condition][f"{spec.name}:{index}"] = {
                    "score": score_from_contribution(spec.kind, contribution),
                    "hit_max_new_tokens": hit_max,
                }
        bootstraps[spec.name] = {
            "metric_name": recovered_name,
            "base": analysis.metric_from_contributions(spec.kind, task_base),
            "normal": analysis.metric_from_contributions(spec.kind, task_normal),
            **analysis.paired_bootstrap(spec.kind, task_base, task_normal),
        }
    return {
        "strict_summary": strict_summary,
        "metrics": metrics,
        "predictions": predictions,
        "contributions": contributions,
        "records_by_condition": records_by_condition,
        "all_row_bootstraps": bootstraps,
    }


def test_clean_narrative_selection_excludes_val_index_25(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 2),
        analysis.TaskSpec("narrative-v3.2", "unused", "narrative", 26),
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 2),
    )
    base = {
        "attribution-v3.2": [(1, 1), (0, 1)],
        "narrative-v3.2": [(1, 1)] * 26,
        "scene-v4-current": [(1, 0, 0)] * 2,
    }
    normal = {
        **base,
        "narrative-v3.2": [(1, 1)] * 25 + [(0, 1)],
    }
    inputs = build_inputs(specs, base, normal)

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    narrative = result["tasks"]["narrative-v3.2"]
    assert narrative["evaluated_rows"] == 26
    assert narrative["selection_rows"] == 25
    assert narrative["applied_excluded_zero_based_indices"] == [25]
    assert narrative["all_rows"]["strict"]["normal_minus_base"] == pytest.approx(-1 / 26)
    assert narrative["all_rows"]["recovered"]["normal_minus_base"] == pytest.approx(-1 / 26)
    assert narrative["clean_selection"]["strict"]["normal_minus_base"] == 0.0
    assert narrative["clean_selection"]["recovered"]["normal_minus_base"] == 0.0
    assert narrative["gates"]["recovered_metric_delta_floor"]["passed"] is True


def test_selection_marks_each_failed_gate_and_missing_tasks(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 2),
    )
    base = {"attribution-v3.2": [(1, 1), (1, 1)]}
    normal = {"attribution-v3.2": [(0, 1), (0, 1)]}
    inputs = build_inputs(
        specs,
        base,
        normal,
        base_predictions={"attribution-v3.2": ["a", "b"]},
        normal_predictions={"attribution-v3.2": [None, None]},
        base_hits={"attribution-v3.2": [False, False]},
        normal_hits={"attribution-v3.2": [True, True]},
    )

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    attribution = result["tasks"]["attribution-v3.2"]
    assert attribution["status"] == "provisional_fail"
    assert all(gate["passed"] is False for gate in attribution["gates"].values())
    assert result["tasks"]["narrative-v3.2"]["status"] == "not_evaluated"
    assert result["tasks"]["scene-v4-current"]["status"] == "not_evaluated"
    assert result["status"] == "incomplete"
    assert result["complete"] is False
    assert result["overall_passed"] is False


def test_full_validation_selection_passes_equal_model_metrics(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAP_REPLICATES", 100)
    specs = (
        analysis.TaskSpec("attribution-v3.2", "unused", "attribution", 30),
        analysis.TaskSpec("narrative-v3.2", "unused", "narrative", 39),
        analysis.TaskSpec("scene-v4-current", "unused", "scene", 170),
    )
    base = {
        "attribution-v3.2": [(1, 1)] * 30,
        "narrative-v3.2": [(1, 1)] * 39,
        "scene-v4-current": [(1, 0, 0)] * 170,
    }
    inputs = build_inputs(specs, base, dict(base))

    result = analysis.build_selection_criterion(
        split="val",
        specs=specs,
        **inputs,
    )

    assert result["status"] == "pass"
    assert result["complete"] is True
    assert result["all_gates_passed"] is True
    assert result["overall_passed"] is True
    assert all(
        result["tasks"][task_name]["criterion_passed"] is True
        for task_name in analysis.CORE_SELECTION_TASKS
    )
