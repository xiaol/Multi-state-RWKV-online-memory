from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import prepare_scene_failure_pairs as builder
from experiments.rethinking_rwkv_ms_gemma import run_scene_train_base_eval as producer


SYSTEM = (
    "你是一个中文小说 scene 边界检测助手。"
    "判断输入段落中哪些边界应切换 scene。"
    "只输出 boundaries，不输出原因。输出 JSON。"
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_row(label: str, boundaries: list[int]) -> str:
    user = "\n".join(
        [
            f"[P1] {label} 第一段。",
            f"[P2] {label} 第二段。",
            f"[P3] {label} 第三段。",
        ]
    )
    return compact_json(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": compact_json({"boundaries": boundaries}),
                },
            ]
        }
    )


def row_sha256(raw_line: str) -> str:
    return hashlib.sha256(raw_line.encode("utf-8")).hexdigest()


def make_base_record(
    raw_row: str,
    *,
    line_index: int,
    parsed_json: Any,
    gold: list[int],
) -> dict[str, Any]:
    return {
        "key": f"scene-v4-current:{line_index}",
        "condition": "base",
        "task": "scene-v4-current",
        "task_kind": "scene",
        "split": "train",
        "line_index": line_index,
        "row_sha256": row_sha256(raw_row),
        "gold": {"boundaries": gold},
        "status": "ok",
        "parsed_json": parsed_json,
        "raw_generation": compact_json(parsed_json),
    }


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_producer_bundle(
    output_dir: Path,
    *,
    dataset_file: Path,
    records: list[dict[str, Any]],
) -> Path:
    source_rows = builder.load_source_split(dataset_file, split="train")
    assert len(source_rows) == builder.DEFAULT_CANDIDATE_COUNT
    assert len(records) == builder.DEFAULT_CANDIDATE_COUNT
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = producer.selection_payload(
        dataset_file=dataset_file.resolve(),
        dataset_sha256=builder.sha256_file(dataset_file),
        selected_rows=source_rows,
        candidate_count=builder.DEFAULT_CANDIDATE_COUNT,
        selection_seed=producer.DEFAULT_SELECTION_SEED,
    )
    selection_path = output_dir / builder.PRODUCER_SELECTION_FILENAME
    write_json(selection_path, selection)

    weights = [
        {
            "relative_path": "model.safetensors",
            "size_bytes": 17,
            "sha256": hashlib.sha256(b"fake-base-weights").hexdigest(),
        }
    ]
    runtime_artifacts = [
        {
            "relative_path": "config.json",
            "size_bytes": 19,
            "sha256": hashlib.sha256(b"fake-base-config").hexdigest(),
        }
    ]
    base_model_artifacts = {
        "root": str((dataset_file.parent / "fake-base-model").resolve()),
        "weights": weights,
        "runtime_artifacts": runtime_artifacts,
        "aggregate_sha256": producer.canonical_json_sha256(
            {"weights": weights, "runtime_artifacts": runtime_artifacts}
        ),
    }
    fingerprint_payload = {
        "schema": builder.PRODUCER_SCHEMA,
        "task": builder.DEFAULT_TASK_NAME,
        "task_kind": "scene",
        "condition": "base",
        "split": "train",
        "base_model": base_model_artifacts["root"],
        "base_model_artifacts": base_model_artifacts,
        "dataset_file": str(dataset_file.resolve()),
        "dataset_sha256": builder.sha256_file(dataset_file),
        "selection_sha256": producer.canonical_json_sha256(selection),
        "selected_rows": selection["rows"],
        "candidate_count": builder.DEFAULT_CANDIDATE_COUNT,
        "selection_seed": producer.DEFAULT_SELECTION_SEED,
    }
    fingerprint = producer.canonical_json_sha256(fingerprint_payload)
    base_path = output_dir / "base.jsonl"
    write_jsonl(
        base_path,
        [compact_json({**record, "fingerprint": fingerprint}) for record in records],
    )
    manifest = {
        "schema": builder.PRODUCER_SCHEMA,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "selection": {
            "path": str(selection_path),
            "sha256": builder.sha256_file(selection_path),
            "rows": builder.DEFAULT_CANDIDATE_COUNT,
            "uses_gold_labels": False,
            "uses_model_output": False,
        },
        "output": {"base_records": str(base_path)},
    }
    write_json(output_dir / builder.PRODUCER_MANIFEST_FILENAME, manifest)
    write_json(
        output_dir / builder.PRODUCER_SUMMARY_FILENAME,
        {
            "schema": builder.PRODUCER_SCHEMA,
            "fingerprint": fingerprint,
            "complete": True,
            "completed": builder.DEFAULT_CANDIDATE_COUNT,
            "expected": builder.DEFAULT_CANDIDATE_COUNT,
            "condition": "base",
            "task": builder.DEFAULT_TASK_NAME,
            "split": "train",
        },
    )
    return base_path


@pytest.fixture
def scene_inputs(tmp_path: Path) -> dict[str, Any]:
    dataset_dir = tmp_path / "v4-scene-boundary-detection"
    train = [
        make_row("train-correct-exact", [1]),
        make_row("train-mixed-failure", [2]),
        make_row("train-correct-empty", []),
        make_row("train-correct-recovered", [2]),
        make_row("train-format-failure", [1]),
        make_row("train-fp-failure", []),
    ]
    train.extend(
        make_row(f"train-correct-filler-{index}", [1])
        for index in range(builder.DEFAULT_CANDIDATE_COUNT - len(train))
    )
    val = [
        make_row("val-a", [1]),
        make_row("val-b", []),
        make_row("val-c", [2]),
        make_row("val-d", [1, 2]),
    ]
    test = [
        make_row("test-a", []),
        make_row("test-b", [2]),
    ]
    write_jsonl(dataset_dir / "train.jsonl", train)
    write_jsonl(dataset_dir / "val.jsonl", val)
    write_jsonl(dataset_dir / "test.jsonl", test)

    records = [
        make_base_record(
            train[0], line_index=0, parsed_json={"boundaries": [1]}, gold=[1]
        ),
        make_base_record(
            train[1], line_index=1, parsed_json=[{"boundary": "P1"}], gold=[2]
        ),
        make_base_record(
            train[2], line_index=2, parsed_json={"boundaries": []}, gold=[]
        ),
        # This is strict-schema-invalid but accepted by the existing conservative recovery.
        make_base_record(
            train[3],
            line_index=3,
            parsed_json=[{"after_paragraph": "P2"}],
            gold=[2],
        ),
        make_base_record(
            train[4], line_index=4, parsed_json={"segments": [1]}, gold=[1]
        ),
        make_base_record(
            train[5], line_index=5, parsed_json={"boundaries": ["P1"]}, gold=[]
        ),
    ]
    records.extend(
        make_base_record(
            train[index],
            line_index=index,
            parsed_json={"boundaries": [1]},
            gold=[1],
        )
        for index in range(6, builder.DEFAULT_CANDIDATE_COUNT)
    )
    # Deliberately scramble evaluator order; the builder must join by row SHA-256.
    record_order = [5, 2, 0, 4, 1, 3, *range(6, builder.DEFAULT_CANDIDATE_COUNT)]
    eval_path = make_producer_bundle(
        tmp_path / "base-train-producer",
        dataset_file=dataset_dir / "train.jsonl",
        records=[records[index] for index in record_order],
    )
    return {
        "dataset_dir": dataset_dir,
        "eval_path": eval_path,
        "train": train,
        "val": val,
        "test": test,
        "records": records,
    }


def test_builds_train_failures_and_unbiased_validation_holdout(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pairs"
    test_path = scene_inputs["dataset_dir"] / "test.jsonl"
    test_before = test_path.read_bytes()

    manifest = builder.prepare_scene_failure_pairs(
        dataset_dir=scene_inputs["dataset_dir"],
        base_train_eval_jsonl=scene_inputs["eval_path"],
        output_dir=output_dir,
        train_failure_count=3,
        holdout_count=2,
        selection_seed=17,
    )

    emitted_train = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert emitted_train == [
        scene_inputs["train"][1],
        scene_inputs["train"][4],
        scene_inputs["train"][5],
    ]
    assert all(
        [message["role"] for message in json.loads(row)["messages"]]
        == ["system", "user", "assistant"]
        for row in emitted_train
    )
    assert all(set(json.loads(row)) == {"messages"} for row in emitted_train)

    train_manifest = read_jsonl(output_dir / "train_manifest.jsonl")
    assert [row["source_line_index"] for row in train_manifest] == [1, 4, 5]
    assert [row["failure_kind"] for row in train_manifest] == [
        "mixed_false_positive_false_negative",
        "unrecoverable_format",
        "false_positive_only",
    ]
    assert train_manifest[0]["base_recovered_boundaries"] == [1]
    assert train_manifest[1]["base_prediction_recovered"] is False

    holdout = (output_dir / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
    holdout_manifest = read_jsonl(output_dir / "holdout_manifest.jsonl")
    holdout_source_indices = json.loads(
        (output_dir / "holdout_source_indices.json").read_text(encoding="utf-8")
    )
    assert len(holdout) == 2
    assert len(holdout_manifest) == 2
    assert holdout_source_indices["schema"] == "rwkv_ms_scene_eval_selection.v1"
    assert holdout_source_indices["dataset"]["split"] == "val"
    assert holdout_source_indices["rows"] == [
        {
            "source_index": row["source_line_index"],
            "row_sha256": row["row_sha256"],
        }
        for row in holdout_manifest
    ]
    assert all(row["source_split"] == "val" for row in holdout_manifest)
    assert all(row["selection_uses_model_output"] is False for row in holdout_manifest)

    train_hashes = {row_sha256(row) for row in emitted_train}
    holdout_hashes = {row_sha256(row) for row in holdout}
    test_hashes = {row_sha256(row) for row in scene_inputs["test"]}
    assert train_hashes.isdisjoint(holdout_hashes)
    assert (train_hashes | holdout_hashes).isdisjoint(test_hashes)
    assert test_path.read_bytes() == test_before

    assert manifest["base_train_evaluation"]["selected_task_records"] == 64
    assert manifest["base_train_evaluation"]["selected_failures"] == 3
    assert manifest["partitions"]["train"]["rows"] == 3
    assert manifest["partitions"]["holdout"]["rows"] == 2
    assert manifest["validation"]["holdout_selection_uses_model_output"] is False
    assert manifest["validation"]["test_rows_emitted"] == 0
    assert manifest["partitions"]["holdout"]["official_source_indices"][
        "sha256"
    ] == builder.sha256_file(output_dir / "holdout_source_indices.json")
    assert manifest["sources"]["test"]["sha256"] == builder.sha256_file(test_path)
    assert manifest["partitions"]["train"]["data"]["sha256"] == builder.sha256_file(
        output_dir / "train.jsonl"
    )
    producer_bundle = manifest["base_train_evaluation"]["producer_bundle"]
    assert producer_bundle["fingerprint"] == json.loads(
        (scene_inputs["eval_path"].parent / "manifest.json").read_text(encoding="utf-8")
    )["fingerprint"]
    assert producer_bundle["selection"]["candidate_count"] == 64
    assert producer_bundle["base_model"]["artifact_aggregate_sha256"] == json.loads(
        (scene_inputs["eval_path"].parent / "manifest.json").read_text(encoding="utf-8")
    )["fingerprint_payload"]["base_model_artifacts"]["aggregate_sha256"]


def test_output_is_deterministic_across_base_record_order(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    reversed_eval = make_producer_bundle(
        tmp_path / "base-train-reversed",
        dataset_file=scene_inputs["dataset_dir"] / "train.jsonl",
        records=list(reversed(scene_inputs["records"])),
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.prepare_scene_failure_pairs(
        dataset_dir=scene_inputs["dataset_dir"],
        base_train_eval_jsonl=scene_inputs["eval_path"],
        output_dir=first,
        train_failure_count=2,
        holdout_count=3,
        selection_seed=99,
    )
    builder.prepare_scene_failure_pairs(
        dataset_dir=scene_inputs["dataset_dir"],
        base_train_eval_jsonl=reversed_eval,
        output_dir=second,
        train_failure_count=2,
        holdout_count=3,
        selection_seed=99,
    )

    for filename in (
        "train.jsonl",
        "holdout.jsonl",
        "holdout_source_indices.json",
        "train_manifest.jsonl",
        "holdout_manifest.jsonl",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert len((first / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_rejects_focused_evaluator_base_full_alias(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = []
    for original in scene_inputs["records"]:
        record = dict(original)
        record["condition"] = "base_full"
        record["source_index"] = record["line_index"]
        records.append(record)
    focused_eval = make_producer_bundle(
        tmp_path / "base-full-train",
        dataset_file=scene_inputs["dataset_dir"] / "train.jsonl",
        records=records,
    )

    with pytest.raises(ValueError, match="Invalid base train record"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=focused_eval,
            output_dir=tmp_path / "focused-pairs",
            train_failure_count=3,
            holdout_count=1,
        )


def test_rejects_base_record_that_does_not_join_to_official_train(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    bad_record = dict(scene_inputs["records"][0])
    bad_record["row_sha256"] = row_sha256(scene_inputs["val"][0])
    bad_record["gold"] = json.loads(scene_inputs["val"][0])["messages"][-1]["content"]
    # Keep the gold type valid so the row-hash ownership check is the failing invariant.
    bad_record["gold"] = json.loads(bad_record["gold"])
    bad_eval = make_producer_bundle(
        tmp_path / "bad-base-train",
        dataset_file=scene_inputs["dataset_dir"] / "train.jsonl",
        records=[bad_record, *scene_inputs["records"][1:]],
    )

    with pytest.raises(ValueError, match="does not belong to the official train split"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=bad_eval,
            output_dir=tmp_path / "bad-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_parsed_prediction_drift_from_raw_generation(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    bad_record = dict(scene_inputs["records"][1])
    bad_record["parsed_json"] = {"boundaries": [2]}
    bad_records = list(scene_inputs["records"])
    bad_records[1] = bad_record
    bad_eval = make_producer_bundle(
        tmp_path / "parsed-drift",
        dataset_file=scene_inputs["dataset_dir"] / "train.jsonl",
        records=bad_records,
    )

    with pytest.raises(ValueError, match="does not match raw_generation"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=bad_eval,
            output_dir=tmp_path / "parsed-drift-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_accepts_record_emitted_by_strict_train_base_producer(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    train_rows = builder.load_source_split(
        scene_inputs["dataset_dir"] / "train.jsonl",
        split="train",
    )
    source = train_rows[1]
    parsed_json = [{"boundary": "P1"}]
    record = producer.build_base_record(
        source,
        {
            "status": "ok",
            "raw_generation": compact_json(parsed_json),
            "parsed_json": parsed_json,
            "input_tokens": 24,
            "output_tokens": 5,
            "hit_max_new_tokens": False,
            "elapsed_seconds": 0.1,
            "peak_cuda_memory_bytes": None,
            "memory_trace": [],
        },
        fingerprint="f" * 64,
        max_new_tokens=128,
        completed_at="2026-01-01T00:00:00+00:00",
    )
    records = list(scene_inputs["records"])
    records[1] = record
    records[4] = make_base_record(
        scene_inputs["train"][4],
        line_index=4,
        parsed_json={"boundaries": [1]},
        gold=[1],
    )
    records[5] = make_base_record(
        scene_inputs["train"][5],
        line_index=5,
        parsed_json={"boundaries": []},
        gold=[],
    )
    eval_path = make_producer_bundle(
        tmp_path / "producer-base",
        dataset_file=scene_inputs["dataset_dir"] / "train.jsonl",
        records=records,
    )

    output_dir = tmp_path / "producer-pairs"
    manifest = builder.prepare_scene_failure_pairs(
        dataset_dir=scene_inputs["dataset_dir"],
        base_train_eval_jsonl=eval_path,
        output_dir=output_dir,
        train_failure_count=1,
        holdout_count=1,
        selection_seed=23,
    )

    assert (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() == [
        scene_inputs["train"][1]
    ]
    assert manifest["base_train_evaluation"]["selected_task_records"] == 64


def test_rejects_arbitrary_base_jsonl_without_producer_bundle(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    arbitrary_path = tmp_path / "arbitrary-base-records.jsonl"
    arbitrary_path.write_bytes(scene_inputs["eval_path"].read_bytes())

    with pytest.raises(ValueError, match="producer-managed base.jsonl"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=arbitrary_path,
            output_dir=tmp_path / "arbitrary-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_mixed_producer_record_fingerprints(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    records = read_jsonl(scene_inputs["eval_path"])
    records[0]["fingerprint"] = "0" * 64
    write_jsonl(scene_inputs["eval_path"], [compact_json(record) for record in records])

    with pytest.raises(ValueError, match="common producer fingerprint"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "mixed-fingerprint-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_candidate_selection_artifact_drift(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    selection_path = scene_inputs["eval_path"].parent / "candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["rows"][0]["row_sha256"] = "0" * 64
    write_json(selection_path, selection)

    with pytest.raises(ValueError, match="selection.sha256 differs"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "selection-drift-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_incomplete_producer_summary(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    summary_path = scene_inputs["eval_path"].parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["complete"] = False
    summary["completed"] = 63
    write_json(summary_path, summary)

    with pytest.raises(ValueError, match="not a complete 64-row run"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "incomplete-summary-output",
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_non_64_candidate_protocol(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires exactly 64"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "wrong-candidate-count-output",
            candidate_count=63,
            train_failure_count=1,
            holdout_count=1,
        )


def test_rejects_exact_prompt_overlap_between_official_splits(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    val_payload = json.loads(scene_inputs["val"][0])
    test_payload = json.loads(scene_inputs["test"][0])
    test_payload["messages"][1]["content"] = val_payload["messages"][1]["content"]
    write_jsonl(
        scene_inputs["dataset_dir"] / "test.jsonl",
        [compact_json(test_payload), scene_inputs["test"][1]],
    )

    with pytest.raises(ValueError, match="Official scene split overlap detected"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "overlap-output",
            train_failure_count=3,
            holdout_count=1,
        )


def test_aborts_when_fewer_than_declared_failure_count(
    scene_inputs: dict[str, Any],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fewer eligible failures"):
        builder.prepare_scene_failure_pairs(
            dataset_dir=scene_inputs["dataset_dir"],
            base_train_eval_jsonl=scene_inputs["eval_path"],
            output_dir=tmp_path / "too-few-failures",
            train_failure_count=4,
            holdout_count=1,
        )


def test_default_protocol_selects_exactly_32_of_64_independent_of_record_order(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "v4-scene-boundary-detection"
    train = [make_row(f"bulk-train-{index}", [2]) for index in range(64)]
    val = [make_row(f"bulk-val-{index}", [1]) for index in range(4)]
    test = [make_row(f"bulk-test-{index}", []) for index in range(2)]
    write_jsonl(dataset_dir / "train.jsonl", train)
    write_jsonl(dataset_dir / "val.jsonl", val)
    write_jsonl(dataset_dir / "test.jsonl", test)
    records = [
        make_base_record(
            raw_row,
            line_index=index,
            parsed_json={"boundaries": [1]},
            gold=[2],
        )
        for index, raw_row in enumerate(train)
    ]
    forward_eval = make_producer_bundle(
        tmp_path / "bulk-forward",
        dataset_file=dataset_dir / "train.jsonl",
        records=records,
    )
    reverse_eval = make_producer_bundle(
        tmp_path / "bulk-reverse",
        dataset_file=dataset_dir / "train.jsonl",
        records=list(reversed(records)),
    )

    forward_output = tmp_path / "bulk-forward-output"
    reverse_output = tmp_path / "bulk-reverse-output"
    forward_manifest = builder.prepare_scene_failure_pairs(
        dataset_dir=dataset_dir,
        base_train_eval_jsonl=forward_eval,
        output_dir=forward_output,
        holdout_count=2,
    )
    reverse_manifest = builder.prepare_scene_failure_pairs(
        dataset_dir=dataset_dir,
        base_train_eval_jsonl=reverse_eval,
        output_dir=reverse_output,
        holdout_count=2,
    )

    source_rows = builder.load_source_split(dataset_dir / "train.jsonl", split="train")
    expected_indices = sorted(
        row.line_index
        for row in sorted(
            source_rows,
            key=lambda row: (
                builder.train_failure_selection_sha256(row.prompt_sha256),
                row.prompt_sha256,
                row.line_index,
            ),
        )[:32]
    )
    forward_rows = read_jsonl(forward_output / "train_manifest.jsonl")
    reverse_rows = read_jsonl(reverse_output / "train_manifest.jsonl")

    assert [row["source_line_index"] for row in forward_rows] == expected_indices
    assert len(forward_rows) == 32
    assert forward_rows == reverse_rows
    assert (forward_output / "train.jsonl").read_bytes() == (
        reverse_output / "train.jsonl"
    ).read_bytes()
    assert forward_manifest["config"]["candidate_count"] == 64
    assert forward_manifest["config"]["train_failure_count"] == 32
    assert forward_manifest["base_train_evaluation"]["eligible_failures"] == 64
    assert forward_manifest["base_train_evaluation"]["selected_failures"] == 32
    assert forward_manifest["partitions"]["train"]["rows"] == 32
    assert forward_manifest["contract"]["failure_selection_uses_eval_record_order"] is False
    assert reverse_manifest["partitions"]["train"]["rows"] == 32
