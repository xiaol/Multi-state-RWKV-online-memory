#!/usr/bin/env python3
"""Aggregate and sign the four routed native benchmark shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_routed_benchmark as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_routed_benchmark.v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError(f"Routed record is not an object: {path}")
                records.append(record)
    return records


def collect_condition(
    root: Path,
    *,
    task: str,
    condition: str,
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index in range(4):
        path = root / f"shard-{shard_index}" / f"{task}.{condition}.jsonl"
        rows = read_records(path)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
        for record in rows:
            index = int(record["line_index"])
            expected = {
                "schema": runner.SCHEMA,
                "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                "task": task,
                "condition": condition,
                "shard_index": shard_index,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Routed shard record binding differs: {path}:{index}")
            if index % 4 != shard_index or index < runner.SELECTION_ROWS:
                raise ValueError(f"Routed shard index contract differs: {path}:{index}")
            if index in records:
                raise ValueError(f"Duplicate routed source index: {task}:{index}")
            records[index] = record
    expected_indices = set(
        range(runner.SELECTION_ROWS, int(runner.TASKS[task]["expected_rows"]))
    )
    if set(records) != expected_indices:
        missing = sorted(expected_indices - set(records))
        raise ValueError(f"Incomplete routed records for {task}:{condition}: {missing[:5]}")
    return records, artifacts


def load_gold(dataset_root: Path, task: str) -> dict[int, Mapping[str, Any]]:
    path = dataset_root / str(runner.TASKS[task]["path"])
    gold: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(line for line in handle if line.strip()):
            if index < runner.SELECTION_ROWS:
                continue
            row = json.loads(raw_line)
            parsed = recovery.extract_json(str(row["messages"][-1]["content"]))
            if not isinstance(parsed, Mapping):
                raise ValueError(f"Invalid routed benchmark gold: {path}:{index}")
            gold[index] = parsed
    return gold


def attribution_metrics(
    records: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    correct = sum(
        records[index]["selected"] == gold[index].get("best_candidate")
        for index in sorted(gold)
    )
    return {
        "rows": len(gold),
        "coverage": 1.0,
        "correct": correct,
        "primary_metric": correct / len(gold),
        "primary_metric_name": "candidate_likelihood_accuracy",
    }


def narrative_metrics(
    records: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    correct = 0
    total = 0
    covered = 0
    for index in sorted(gold):
        prediction = records[index].get("prediction")
        if isinstance(prediction, Mapping):
            covered += 1
            normalized = {str(key): value for key, value in prediction.items()}
        else:
            normalized = {}
        gold_labels = recovery.gold_label_map(gold[index])
        correct += sum(
            normalized.get(unit_id) == label_type
            for unit_id, label_type in gold_labels.items()
        )
        total += len(gold_labels)
    return {
        "rows": len(gold),
        "coverage": covered / len(gold),
        "correct_units": correct,
        "gold_units": total,
        "primary_metric": correct / total,
        "primary_metric_name": "format_recovered_unit_accuracy",
    }


def routed_narrative_records(
    base: Mapping[int, Mapping[str, Any]],
    memory: Mapping[int, Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    routed: dict[int, Mapping[str, Any]] = {}
    for index in sorted(base):
        base_prediction = base[index].get("prediction")
        memory_prediction = memory[index].get("prediction")
        base_labels = dict(base_prediction) if isinstance(base_prediction, Mapping) else {}
        memory_labels = (
            dict(memory_prediction) if isinstance(memory_prediction, Mapping) else {}
        )
        output = dict(base_labels)
        for unit_id, base_label in base_labels.items():
            if (
                base_label == "narration"
                and memory_labels.get(unit_id) == "scene_description"
            ):
                output[unit_id] = "scene_description"
        routed[index] = {"prediction": output}
    return routed


def scene_metrics(
    records: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    tp = fp = fn = covered = 0
    for index in sorted(gold):
        prediction = records[index].get("prediction")
        predicted = set(prediction) if isinstance(prediction, list) else set()
        covered += int(isinstance(prediction, list))
        expected = recovery.strict_gold_boundaries(gold[index])
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    denominator = 2 * tp + fp + fn
    return {
        "rows": len(gold),
        "coverage": covered / len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "primary_metric": 0.0 if denominator == 0 else 2 * tp / denominator,
        "primary_metric_name": "format_recovered_micro_f1",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner.validate_protocol()
    root = args.input_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Routed benchmark output must be fresh: {output}")

    metrics: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for task in runner.TASKS:
        gold = load_gold(dataset_root, task)
        base, base_artifacts = collect_condition(root, task=task, condition="base")
        if task == "attribution":
            memory, memory_artifacts = collect_condition(
                root, task=task, condition="memory"
            )
            base_metrics = attribution_metrics(base, gold)
            routed_metrics = attribution_metrics(memory, gold)
        elif task == "narrative":
            memory, memory_artifacts = collect_condition(
                root, task=task, condition="memory"
            )
            base_metrics = narrative_metrics(base, gold)
            routed_metrics = narrative_metrics(
                routed_narrative_records(base, memory), gold
            )
        else:
            memory_artifacts = []
            base_metrics = scene_metrics(base, gold)
            routed_metrics = dict(base_metrics)
        delta = float(routed_metrics["primary_metric"]) - float(
            base_metrics["primary_metric"]
        )
        metrics[task] = {
            "base": base_metrics,
            "routed": routed_metrics,
            "routed_minus_base": delta,
        }
        provenance[task] = {
            "base": base_artifacts,
            "memory": memory_artifacts,
        }

    coverage_pass = all(
        float(task["routed"]["coverage"]) >= 0.95 for task in metrics.values()
    )
    no_regression = all(
        float(task["routed_minus_base"]) >= 0.0 for task in metrics.values()
    )
    improved = sum(
        float(task["routed_minus_base"]) > 0.0 for task in metrics.values()
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "split": "publisher-TRAIN-derived development untouched remainder",
            "selection_rows_excluded_per_task": runner.SELECTION_ROWS,
            "rows": sum(
                int(task_spec["expected_rows"]) - runner.SELECTION_ROWS
                for task_spec in runner.TASKS.values()
            ),
            "protected_splits_opened": [],
        },
        "metrics": metrics,
        "gates": {
            "coverage_at_least_0.95_all_tasks": coverage_pass,
            "no_regression_all_tasks": no_regression,
            "strict_improvement_at_least_two_tasks": improved >= 2,
            "strictly_improved_tasks": improved,
            "passed": coverage_pass and no_regression and improved >= 2,
        },
        "provenance": provenance,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_benchmark_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
