#!/usr/bin/env python3
"""Aggregate and sign the one-shot hybrid publisher-validation replication."""

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
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_hybrid_publisher_validation as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_hybrid_publisher_validation_result.v1"
EXPECTED_RUNNER_SHA256 = "8f196179c7a608b1855391773f3328440e90686e305d1714ce426974945124fb"
EXPECTED_LIKELIHOOD_RUNNER_SHA256 = "d1eb64ab7b926b92ac4b84ba3140af215ddfe94e477157718062b336ec696b22"
EXPECTED_GENERATION_RUNNER_SHA256 = "69ebe18e2e4eaf5dd08217970bc85fc0ecdeab1ae7384382c644ce8641596ae8"
EXPECTED_PATCH_LOADER_SHA256 = "2fb6ba9b864da4e9b5aa79d3cecba7af12b70fb95df0b88773c15989b919e39f"
SCENE_V9_MINIMUM_DELTA = 0.005


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
                    raise ValueError(f"Hybrid validation record is not an object: {path}")
                records.append(record)
    return records


def validate_input_bindings(root: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    for worker_index in range(runner.WORLD_SIZE):
        path = root / f"shard-{worker_index}" / "input_binding.json"
        binding = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "base_model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
            "base_config_sha256": runner.BASE_CONFIG_SHA256,
            "v9_adapter_files_sha256": runner.V9_ADAPTER_FILES_SHA256,
            "v9_adapter_weights_sha256": runner.V9_ADAPTER_WEIGHTS_SHA256,
            "hybrid_result_sha256": runner.HYBRID_RESULT_FILE_SHA256,
            "hybrid_result_receipt_sha256": runner.HYBRID_RESULT_RECEIPT_SHA256,
            "checkpoint_step": runner.SELECTED_STEP,
            "checkpoint_manifest_sha256": runner.SELECTED_MANIFEST_SHA256,
            "checkpoint_manifest_receipt_sha256": runner.SELECTED_MANIFEST_RECEIPT_SHA256,
            "checkpoint_gate_state_sha256": runner.SELECTED_GATE_STATE_SHA256,
            "checkpoint_patch_sha256": runner.SELECTED_PATCH_SHA256,
            "worker_index": worker_index,
            "world_size": runner.WORLD_SIZE,
            "task_conditions": {
                task: list(spec["conditions"])
                for task, spec in runner.TASKS.items()
            },
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "likelihood_runner_sha256": EXPECTED_LIKELIHOOD_RUNNER_SHA256,
            "generation_runner_sha256": EXPECTED_GENERATION_RUNNER_SHA256,
            "patch_loader_sha256": EXPECTED_PATCH_LOADER_SHA256,
            "protected_splits_opened": ["publisher_validation_fresh_replication"],
            "prior_validation_artifacts_read": False,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Hybrid validation input binding differs: {path}")
        expected_datasets = {
            task: spec["sha256"] for task, spec in runner.TASKS.items()
        }
        datasets = binding.get("dataset_files")
        if not isinstance(datasets, Mapping) or any(
            datasets.get(task, {}).get("sha256") != digest
            for task, digest in expected_datasets.items()
        ):
            raise ValueError(f"Hybrid validation dataset binding differs: {path}")
        comparable = dict(binding)
        comparable.pop("worker_index")
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError("Hybrid validation bindings differ across workers")
        bindings.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "payload": binding,
            }
        )
    return bindings


def collect_condition(
    root: Path,
    *,
    task: str,
    condition: str,
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    runner.validate_condition(task, condition)
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for worker_index in range(runner.WORLD_SIZE):
        path = root / f"shard-{worker_index}" / f"{task}.{condition}.jsonl"
        rows = read_records(path)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
        for record in rows:
            index = int(record["source_index"])
            expected = {
                "schema": runner.SCHEMA,
                "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                "task": task,
                "condition": condition,
                "worker_index": worker_index,
                "world_size": runner.WORLD_SIZE,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Hybrid validation record binding differs: {path}:{index}")
            if index % runner.WORLD_SIZE != worker_index:
                raise ValueError(f"Hybrid validation worker differs: {path}:{index}")
            if condition == "checkpoint16_memory" and (
                record.get("checkpoint_step") != runner.SELECTED_STEP
                or record.get("gate_state_sha256") != runner.SELECTED_GATE_STATE_SHA256
                or not isinstance(record.get("runtime_gate_state_sha256"), str)
            ):
                raise ValueError(f"Checkpoint-16 record binding differs: {path}:{index}")
            if index in records:
                raise ValueError(f"Duplicate hybrid validation row: {task}:{condition}:{index}")
            records[index] = record
    expected_indices = set(range(int(runner.TASKS[task]["source_rows"]))) - set(
        runner.TASKS[task]["excluded_source_indices"]
    )
    if set(records) != expected_indices:
        missing = sorted(expected_indices - set(records))
        raise ValueError(
            f"Incomplete hybrid validation {task}:{condition}: {missing[:5]}"
        )
    return records, artifacts


def load_gold(
    dataset_root: Path,
    task: str,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, str]]:
    spec = runner.TASKS[task]
    path = dataset_root / str(spec["path"])
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Hybrid validation gold dataset hash differs: {task}")
    excluded = set(spec["excluded_source_indices"])
    gold: dict[int, Mapping[str, Any]] = {}
    row_hashes: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(line for line in handle if line.strip()):
            if index in excluded:
                continue
            row = json.loads(raw_line)
            parsed = recovery.extract_json(str(row["messages"][-1]["content"]))
            if not isinstance(parsed, Mapping):
                raise ValueError(f"Invalid hybrid validation gold: {path}:{index}")
            gold[index] = parsed
            row_hashes[index] = hashlib.sha256(
                raw_line.rstrip("\n").encode("utf-8")
            ).hexdigest()
    expected = int(spec["source_rows"]) - len(excluded)
    if len(gold) != expected:
        raise ValueError(f"Hybrid validation gold row count differs: {task}")
    return gold, row_hashes


def validate_record_hashes(
    records: Mapping[int, Mapping[str, Any]],
    row_hashes: Mapping[int, str],
    *,
    task: str,
    condition: str,
) -> None:
    if set(records) != set(row_hashes) or any(
        records[index].get("row_sha256") != digest
        for index, digest in row_hashes.items()
    ):
        raise ValueError(f"Hybrid validation source hashes differ: {task}:{condition}")


def evaluate_gates(metrics: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    candidate_coverage = all(
        float(metrics[task]["candidate"]["coverage"]) >= 0.95
        for task in runner.TASKS
    )
    v9_scene_coverage = float(metrics["scene"]["v9"]["coverage"]) >= 0.95
    no_regression = all(
        float(metrics[task]["candidate_minus_base"]) >= 0.0
        for task in runner.TASKS
    )
    improved = sum(
        float(metrics[task]["candidate_minus_base"]) > 0.0
        for task in runner.TASKS
    )
    scene_minus_v9 = float(metrics["scene"]["checkpoint16_minus_v9"])
    scene_minus_base = float(metrics["scene"]["candidate_minus_base"])
    gates: dict[str, Any] = {
        "candidate_coverage_at_least_0.95_all_tasks": candidate_coverage,
        "fresh_v9_scene_coverage_at_least_0.95": v9_scene_coverage,
        "candidate_no_regression_vs_fresh_base_all_tasks": no_regression,
        "candidate_strict_improvement_at_least_one_task": improved >= 1,
        "candidate_strictly_improved_tasks": improved,
        "checkpoint16_scene_strict_improvement_vs_fresh_base": scene_minus_base > 0.0,
        "checkpoint16_scene_minus_fresh_v9_at_least_0.005": (
            scene_minus_v9 >= SCENE_V9_MINIMUM_DELTA
        ),
        "checkpoint16_scene_minus_fresh_v9": scene_minus_v9,
    }
    gates["passed"] = all(
        value for key, value in gates.items() if key not in {
            "candidate_strictly_improved_tasks",
            "checkpoint16_scene_minus_fresh_v9",
        }
    )
    return gates


def analyze(
    *,
    input_root: Path,
    dataset_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    runner.validate_protocol()
    if output.exists():
        raise ValueError(f"Hybrid validation result must be fresh: {output}")
    bindings = validate_input_bindings(input_root)
    metrics: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {"input_bindings": bindings}
    for task, spec in runner.TASKS.items():
        gold, row_hashes = load_gold(dataset_root, task)
        conditions: dict[str, Mapping[int, Mapping[str, Any]]] = {}
        artifacts: dict[str, list[dict[str, Any]]] = {}
        for condition in spec["conditions"]:
            records, condition_artifacts = collect_condition(
                input_root,
                task=task,
                condition=str(condition),
            )
            validate_record_hashes(
                records,
                row_hashes,
                task=task,
                condition=str(condition),
            )
            conditions[str(condition)] = records
            artifacts[str(condition)] = condition_artifacts
        if task == "attribution":
            base_metrics = routed.attribution_metrics(conditions["base"], gold)
            candidate_metrics = dict(base_metrics)
            task_metrics: dict[str, Any] = {
                "base": base_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_base": 0.0,
            }
        elif task == "narrative":
            base_metrics = routed.narrative_metrics(conditions["base"], gold)
            candidate_metrics = routed.narrative_metrics(
                routed.routed_narrative_records(
                    conditions["base"],
                    conditions["v9_memory"],
                ),
                gold,
            )
            task_metrics = {
                "base": base_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_base": (
                    float(candidate_metrics["primary_metric"])
                    - float(base_metrics["primary_metric"])
                ),
            }
        else:
            base_metrics = routed.scene_metrics(conditions["base"], gold)
            v9_metrics = routed.scene_metrics(conditions["v9_memory"], gold)
            candidate_metrics = routed.scene_metrics(
                conditions["checkpoint16_memory"], gold
            )
            task_metrics = {
                "base": base_metrics,
                "v9": v9_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_base": (
                    float(candidate_metrics["primary_metric"])
                    - float(base_metrics["primary_metric"])
                ),
                "checkpoint16_minus_v9": (
                    float(candidate_metrics["primary_metric"])
                    - float(v9_metrics["primary_metric"])
                ),
            }
        metrics[task] = task_metrics
        provenance[task] = artifacts

    gates = evaluate_gates(metrics)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "split": "publisher validation fresh replication",
            "rows": sum(
                int(spec["source_rows"]) - len(spec["excluded_source_indices"])
                for spec in runner.TASKS.values()
            ),
            "attribution_excluded_source_indices": list(
                runner.TASKS["attribution"]["excluded_source_indices"]
            ),
            "prior_validation_artifacts_read": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
            "unused_strength_holdout_opened": False,
            "protected_splits_opened": ["publisher_validation_fresh_replication"],
        },
        "decoder": {
            "attribution": "frozen-base candidate likelihood",
            "narrative": "base except base=narration,v9=scene_description",
            "scene": "checkpoint-16 correct-state memory generation",
        },
        "metrics": metrics,
        "gates": gates,
        "passed": bool(gates["passed"]),
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "provenance": {
            **provenance,
            "analyzer_sha256": sha256_file(Path(__file__)),
            "runner_sha256": EXPECTED_RUNNER_SHA256,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_hybrid_publisher_validation_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "receipt": result["receipt"]["payload_sha256"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
