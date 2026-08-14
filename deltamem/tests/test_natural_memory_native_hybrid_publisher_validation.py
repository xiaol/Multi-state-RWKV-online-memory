from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_hybrid_publisher_validation as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_hybrid_publisher_validation as runner,
)


def test_hybrid_validation_protocol_is_hash_bound_and_narrowly_authorized() -> None:
    protocol = runner.validate_protocol()

    assert protocol["dataset"]["reported_rows_total"] == 238
    assert protocol["frozen_constraints"]["world_size"] == 4
    assert protocol["authorization"]["fresh_publisher_validation_replication_authorized"] is True
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False


def test_hybrid_validation_runner_code_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == analysis.EXPECTED_RUNNER_SHA256


def test_hybrid_validation_conditions_are_fixed_per_task() -> None:
    assert runner.TASKS["attribution"]["conditions"] == ("base",)
    assert runner.TASKS["narrative"]["conditions"] == ("base", "v9_memory")
    assert runner.TASKS["scene"]["conditions"] == (
        "base",
        "v9_memory",
        "checkpoint16_memory",
    )
    with pytest.raises(ValueError, match="Invalid hybrid validation condition"):
        runner.validate_condition("narrative", "checkpoint16_memory")


def test_hybrid_validation_four_way_sharding_is_complete_and_disjoint() -> None:
    rows = [{"source_index": index} for index in range(170)]
    shards = [
        runner.selected_rows(rows, task="scene", worker_index=worker_index)
        for worker_index in range(runner.WORLD_SIZE)
    ]

    assert {row["source_index"] for shard in shards for row in shard} == set(range(170))
    assert sum(len(shard) for shard in shards) == 170
    assert all(
        not ({row["source_index"] for row in shards[left]} & {
            row["source_index"] for row in shards[right]
        })
        for left in range(runner.WORLD_SIZE)
        for right in range(left + 1, runner.WORLD_SIZE)
    )


def test_hybrid_validation_retains_attribution_scope_exclusion() -> None:
    rows = [{"source_index": index} for index in range(30)]
    selected = [
        row
        for worker_index in range(runner.WORLD_SIZE)
        for row in runner.selected_rows(
            rows,
            task="attribution",
            worker_index=worker_index,
        )
    ]

    assert {row["source_index"] for row in selected} == set(range(1, 30))


def test_hybrid_narrative_router_changes_only_locked_label_pair() -> None:
    base = {
        0: {"prediction": {"1": "narration", "2": "action", "3": "thought"}}
    }
    v9 = {
        0: {
            "prediction": {
                "1": "scene_description",
                "2": "scene_description",
                "3": "narration",
            }
        }
    }

    output = analysis.routed.routed_narrative_records(base, v9)

    assert output[0]["prediction"] == {
        "1": "scene_description",
        "2": "action",
        "3": "thought",
    }


def _gate_metrics(*, scene_minus_v9: float) -> dict[str, dict[str, object]]:
    return {
        "attribution": {
            "candidate": {"coverage": 1.0},
            "candidate_minus_base": 0.0,
        },
        "narrative": {
            "candidate": {"coverage": 1.0},
            "candidate_minus_base": 0.001,
        },
        "scene": {
            "candidate": {"coverage": 1.0},
            "v9": {"coverage": 1.0},
            "candidate_minus_base": 0.02,
            "checkpoint16_minus_v9": scene_minus_v9,
        },
    }


def test_hybrid_validation_gate_requires_material_scene_gain_over_fresh_v9() -> None:
    passing = analysis.evaluate_gates(_gate_metrics(scene_minus_v9=0.005))
    failing = analysis.evaluate_gates(_gate_metrics(scene_minus_v9=0.004999))

    assert passing["passed"] is True
    assert failing["checkpoint16_scene_minus_fresh_v9_at_least_0.005"] is False
    assert failing["passed"] is False


def test_archived_hybrid_validation_failure_is_signed() -> None:
    path = Path(
        "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
        "natural_memory_native_hybrid_publisher_validation_v1/result.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    receipt = unsigned.pop("receipt")

    assert runner.canonical_sha256(unsigned) == receipt["payload_sha256"]
    assert result["passed"] is False
    assert result["gates"]["checkpoint16_scene_minus_fresh_v9"] < 0.0
    assert result["scope"]["prior_validation_artifacts_read"] is False
    assert result["scope"]["publisher_test_opened"] is False
