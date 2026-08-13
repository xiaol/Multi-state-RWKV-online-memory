from __future__ import annotations

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_routed_benchmark as analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_routed_benchmark as runner,
)


def test_routed_protocol_receipt_is_bound() -> None:
    protocol = runner.validate_protocol()

    assert protocol["selection_gate"]["passed"] is True
    assert protocol["benchmark_scope"]["rows"]["total"] == 560


def test_scene_router_protocol_receipt_is_bound() -> None:
    protocol_path = (
        runner.SCRIPT_DIR / "natural_memory_native_scene_router_protocol_v1.json"
    )
    protocol = runner.json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = protocol.pop("receipt")

    assert (
        runner.canonical_sha256(protocol)
        == receipt["payload_sha256"]
        == runner.SCENE_ROUTER_PROTOCOL_PAYLOAD_SHA256
    )


def test_benchmark_excludes_selection_rows_and_shards_by_source_index() -> None:
    rows = [{"line_index": index} for index in range(20)]

    shards = [
        runner.selected_rows(rows, shard_index=index, world_size=4)
        for index in range(4)
    ]

    selected_indices = {
        int(row["line_index"])
        for shard in shards
        for row in shard
    }
    assert selected_indices == set(range(4, 20))
    assert all(
        all(int(row["line_index"]) % 4 == shard_index for row in shard)
        for shard_index, shard in enumerate(shards)
    )


def test_narrative_router_changes_only_preregistered_pair() -> None:
    base = {
        4: {"prediction": {"1": "narration", "2": "action", "3": "thought"}}
    }
    memory = {
        4: {
            "prediction": {
                "1": "scene_description",
                "2": "scene_description",
                "3": "narration",
            }
        }
    }

    routed = analysis.routed_narrative_records(base, memory)

    assert routed[4]["prediction"] == {
        "1": "scene_description",
        "2": "action",
        "3": "thought",
    }


def test_attribution_metrics_use_candidate_selection() -> None:
    records = {4: {"selected": "A"}, 5: {"selected": "B"}}
    gold = {
        4: {"best_candidate": "A"},
        5: {"best_candidate": "A"},
    }

    metrics = analysis.attribution_metrics(records, gold)

    assert metrics["correct"] == 1
    assert metrics["primary_metric"] == 0.5


def test_scene_routed_result_is_exact_base_fallback() -> None:
    base = {4: {"prediction": [1, 3]}, 5: {"prediction": []}}
    gold = {4: {"boundaries": [1]}, 5: {"boundaries": [2]}}

    metrics = analysis.scene_metrics(base, gold)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["primary_metric"] == 0.5


def test_attribution_singleton_candidate_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(runner.recovery, "parse_candidates", lambda _content: ("A",))

    record = runner.attribution_record(
        object(),
        object(),
        {
            "line_index": 4,
            "row_sha256": "row",
            "messages": [{"role": "user", "content": "singleton"}],
        },
        condition="base",
        device="cpu",
        shard_index=0,
    )

    assert record["selected"] == "A"
    assert record["candidate_scores"] == []
    assert record["deterministic_singleton_candidate"] is True
