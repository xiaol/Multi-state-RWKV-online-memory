from __future__ import annotations

import copy

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    run_recurrent_routed_development_generation as development,
)


EXPECTED_ROWS = {
    "attribution": 32,
    "narrative": 128,
    "scene": 128,
}


def _system_summary(
    attribution: float,
    narrative: float,
    scene: float,
    *,
    strict_schema_rate: float = 1.0,
) -> dict[str, object]:
    metrics = {
        "attribution": ("accuracy", attribution),
        "narrative": ("unit_label_accuracy", narrative),
        "scene": ("micro_f1", scene),
    }
    by_task = {}
    for task, (metric, value) in metrics.items():
        rows = EXPECTED_ROWS[task]
        by_task[task] = {
            "rows": rows,
            metric: value,
            "strict_schema_valid": int(rows * strict_schema_rate),
        }
    return {
        "rows": sum(EXPECTED_ROWS.values()),
        "by_task": by_task,
    }


def test_summarize_reports_semantic_and_strict_schema_separately() -> None:
    records = [
        {
            "task": "attribution",
            "prompt_variant": 0,
            "score": {
                "schema_valid": False,
                "correct": False,
                "uncertain_correct": False,
                "joint_correct": False,
            },
            "recovered_score": {"covered": True, "correct": True},
        },
        {
            "task": "narrative",
            "prompt_variant": 1,
            "score": {
                "schema_valid": True,
                "correct_units": 1,
                "gold_units": 4,
            },
            "recovered_score": {
                "covered": True,
                "correct_units": 2,
                "gold_units": 4,
            },
        },
        {
            "task": "scene",
            "prompt_variant": 2,
            "score": {
                "schema_valid": True,
                "tp": 1,
                "fp": 1,
                "fn": 2,
            },
            "recovered_score": {
                "covered": True,
                "tp": 2,
                "fp": 1,
                "fn": 1,
            },
        },
    ]

    summary = development.summarize(records)

    assert summary["by_task"]["attribution"]["accuracy"] == 0.0
    assert (
        summary["by_task"]["attribution"][
            "recovered_accuracy_diagnostic"
        ]
        == 1.0
    )
    assert summary["by_task"]["attribution"]["strict_schema_valid"] == 0
    assert summary["by_task"]["narrative"]["unit_label_accuracy"] == 0.25
    assert summary["by_task"]["scene"]["micro_f1"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("task", "raw_generation"),
    [
        (
            "attribution",
            '{"best_candidate":"甲","uncertain":false}',
        ),
        (
            "narrative",
            '{"labels":[{"unit_id":"1","type":"dialogue"}]}',
        ),
        ("scene", '{"boundaries":[2,4]}'),
    ],
)
def test_strict_task_json_accepts_only_exact_raw_schema(
    task: str,
    raw_generation: str,
) -> None:
    from experiments.rethinking_rwkv_ms_gemma import (
        run_recurrent_routed_final as final,
    )

    assert final.strict_task_json(task, raw_generation) is not None
    assert final.strict_task_json(task, f"answer: {raw_generation}") is None


def test_promotion_criteria_compare_frozen_outputs_without_mutation() -> None:
    systems = {
        "frozen_gemma_base": _system_summary(0.50, 0.60, 0.40),
        "v9_projected_slot_baseline": _system_summary(0.60, 0.55, 0.50),
        "recurrent_routed_candidate": _system_summary(0.60, 0.65, 0.50),
    }
    original = copy.deepcopy(systems)

    result = development.evaluate_promotion_criteria(
        systems,
        EXPECTED_ROWS,
    )

    assert result["passed"] is True
    assert result["candidate_at_least_stronger_baseline_every_task"] is True
    assert result["candidate_strictly_better_than_both_one_task"] is True
    assert systems == original


def test_promotion_criteria_reject_task_regression() -> None:
    systems = {
        "frozen_gemma_base": _system_summary(0.50, 0.60, 0.40),
        "v9_projected_slot_baseline": _system_summary(0.60, 0.55, 0.50),
        "recurrent_routed_candidate": _system_summary(0.59, 0.70, 0.60),
    }

    result = development.evaluate_promotion_criteria(
        systems,
        EXPECTED_ROWS,
    )

    assert result["passed"] is False
    assert result["candidate_at_least_stronger_baseline_every_task"] is False


def test_cli_rejects_benchmark_time_candidate_gain_override() -> None:
    with pytest.raises(SystemExit):
        development.parse_args(
            [
                "--candidate-adapter",
                "adapter",
                "--candidate-training-receipt",
                "training",
                "--candidate-protocol-receipt",
                "protocol",
                "--candidate-gain",
                "0.5",
            ]
        )
