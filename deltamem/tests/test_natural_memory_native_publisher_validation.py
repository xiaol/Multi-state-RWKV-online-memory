from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_publisher_validation as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_publisher_validation as runner,
)


def test_publisher_validation_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["dataset"]["reported_rows_total"] == 238
    assert protocol["authorization"]["publisher_test_authorized"] is False
    assert protocol["authorization"]["hard32_authorized"] is False


def test_publisher_runner_code_hash_is_bound() -> None:
    assert runner.sha256_file(runner.Path(runner.__file__)) == (
        analysis.EXPECTED_RUNNER_SHA256
    )


def test_locked_execution_contract() -> None:
    runner.validate_task_contract(
        "attribution", shard_index=0, shard_count=1, conditions=("base",)
    )
    runner.validate_task_contract(
        "narrative",
        shard_index=3,
        shard_count=4,
        conditions=("base", "memory"),
    )
    runner.validate_task_contract(
        "scene",
        shard_index=1,
        shard_count=2,
        conditions=("base", "memory"),
    )


def test_attribution_exclusion_and_task_sharding() -> None:
    rows = [{"source_index": index} for index in range(30)]

    selected = runner.selected_rows(
        rows,
        task="attribution",
        shard_index=0,
        shard_count=1,
    )

    assert {row["source_index"] for row in selected} == set(range(1, 30))


def test_scene_two_way_sharding_is_complete_and_disjoint() -> None:
    rows = [{"source_index": index} for index in range(170)]

    shards = [
        runner.selected_rows(
            rows,
            task="scene",
            shard_index=shard_index,
            shard_count=2,
        )
        for shard_index in range(2)
    ]

    assert {row["source_index"] for shard in shards for row in shard} == set(
        range(170)
    )
    assert not ({row["source_index"] for row in shards[0]} & {
        row["source_index"] for row in shards[1]
    })


def test_publisher_narrative_uses_only_locked_pair_router() -> None:
    base = {
        0: {"prediction": {"1": "narration", "2": "action", "3": "thought"}}
    }
    memory = {
        0: {
            "prediction": {
                "1": "scene_description",
                "2": "scene_description",
                "3": "narration",
            }
        }
    }

    output = analysis.routed.routed_narrative_records(base, memory)

    assert output[0]["prediction"] == {
        "1": "scene_description",
        "2": "action",
        "3": "thought",
    }


def test_publisher_scene_uses_memory_directly() -> None:
    base = {0: {"prediction": [1, 3]}}
    memory = {0: {"prediction": [1]}}
    gold = {0: {"boundaries": [1]}}

    base_metrics = analysis.routed.scene_metrics(base, gold)
    memory_metrics = analysis.routed.scene_metrics(memory, gold)

    assert base_metrics["primary_metric"] == 2 / 3
    assert memory_metrics["primary_metric"] == 1.0
