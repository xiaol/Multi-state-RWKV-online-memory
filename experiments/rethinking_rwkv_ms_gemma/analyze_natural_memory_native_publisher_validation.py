#!/usr/bin/env python3
"""Aggregate, score, and sign the locked publisher native validation."""

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
    analyze_natural_memory_native_routed_benchmark as routed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_publisher_validation as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_publisher_validation.v1"
EXPECTED_RUNNER_SHA256 = (
    "dc044dc1b506ca9d548421c3066d56382961124140795026fdd744b6aec5405f"
)
EXPECTED_LIKELIHOOD_RUNNER_SHA256 = (
    "d1eb64ab7b926b92ac4b84ba3140af215ddfe94e477157718062b336ec696b22"
)
EXPECTED_GENERATION_RUNNER_SHA256 = (
    "69ebe18e2e4eaf5dd08217970bc85fc0ecdeab1ae7384382c644ce8641596ae8"
)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError(f"Publisher validation record is not an object: {path}")
                records.append(record)
    return records


def collect_condition(
    root: Path,
    *,
    task: str,
    condition: str,
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    shard_count = int(runner.TASKS[task]["allowed_shard_counts"][0])
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_dir = root / task / f"shard-{shard_index}-of-{shard_count}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "base_model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
            "base_config_sha256": runner.BASE_CONFIG_SHA256,
            "memory_adapter_sha256": runner.MEMORY_ADAPTER_SHA256,
            "dataset_sha256": runner.TASKS[task]["sha256"],
            "task": task,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "conditions": list(runner.TASKS[task]["conditions"]),
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "likelihood_runner_sha256": EXPECTED_LIKELIHOOD_RUNNER_SHA256,
            "generation_runner_sha256": EXPECTED_GENERATION_RUNNER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError(f"Publisher validation binding differs: {binding_path}")
        path = shard_dir / f"{condition}.jsonl"
        rows = read_records(path)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(path),
                "input_binding_path": str(binding_path.resolve()),
                "input_binding_sha256": sha256_file(binding_path),
            }
        )
        for record in rows:
            index = int(record["source_index"])
            expected_record = {
                "schema": runner.SCHEMA,
                "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                "task": task,
                "condition": condition,
                "shard_index": shard_index,
                "shard_count": shard_count,
            }
            if any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError(f"Publisher record binding differs: {path}:{index}")
            if index % shard_count != shard_index:
                raise ValueError(f"Publisher record shard differs: {path}:{index}")
            if index in records:
                raise ValueError(f"Duplicate publisher validation row: {task}:{index}")
            records[index] = record
    expected_indices = set(range(int(runner.TASKS[task]["source_rows"]))) - set(
        runner.TASKS[task]["excluded_source_indices"]
    )
    if set(records) != expected_indices:
        missing = sorted(expected_indices - set(records))
        raise ValueError(
            f"Incomplete publisher validation {task}:{condition}: {missing[:5]}"
        )
    return records, artifacts


def validate_record_hashes(
    records: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    condition: str,
) -> None:
    expected = {
        int(row["source_index"]): str(row["row_sha256"])
        for row in rows
        if int(row["source_index"])
        not in set(runner.TASKS[task]["excluded_source_indices"])
    }
    if set(records) != set(expected) or any(
        records[index].get("row_sha256") != digest
        for index, digest in expected.items()
    ):
        raise ValueError(f"Publisher source row hashes differ: {task}:{condition}")


def gold_by_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["source_index"]): row["gold"] for row in rows}


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
        raise ValueError(f"Publisher validation output must be fresh: {output}")

    metrics: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for task in runner.TASKS:
        rows = runner.load_rows(dataset_root, task)
        excluded = set(runner.TASKS[task]["excluded_source_indices"])
        gold = {
            index: value
            for index, value in gold_by_index(rows).items()
            if index not in excluded
        }
        base, base_artifacts = collect_condition(root, task=task, condition="base")
        validate_record_hashes(base, rows, task=task, condition="base")
        if task == "attribution":
            routed_metrics = routed.attribution_metrics(base, gold)
            base_metrics = dict(routed_metrics)
            memory_artifacts: list[dict[str, Any]] = []
        else:
            memory, memory_artifacts = collect_condition(
                root, task=task, condition="memory"
            )
            validate_record_hashes(memory, rows, task=task, condition="memory")
            if task == "narrative":
                base_metrics = routed.narrative_metrics(base, gold)
                routed_metrics = routed.narrative_metrics(
                    routed.routed_narrative_records(base, memory), gold
                )
            else:
                base_metrics = routed.scene_metrics(base, gold)
                routed_metrics = routed.scene_metrics(memory, gold)
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
            "split": "publisher validation",
            "rows": sum(
                int(spec["source_rows"]) - len(spec["excluded_source_indices"])
                for spec in runner.TASKS.values()
            ),
            "attribution_excluded_source_indices": list(
                runner.TASKS["attribution"]["excluded_source_indices"]
            ),
            "publisher_test_opened": False,
            "hard32_opened": False,
        },
        "decoder": {
            "attribution": "frozen-base candidate likelihood",
            "narrative": "base except base=narration,memory=scene_description",
            "scene": "v9 memory generation",
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
        "payload_scope": "canonical_publisher_validation_without_receipt",
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
