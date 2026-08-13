#!/usr/bin/env python3
"""Aggregate and sign the native scene causal and router study."""

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
    run_natural_memory_native_scene_causal as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_causal_router.v1"
EXPECTED_RUNNER_SHA256 = (
    "37d6c88436caf5f097f284759e18488581f04de0c84405b9f14e276adf040fc0"
)
CANDIDATE_NAMES = (
    "base",
    "memory",
    "intersection",
    "conditional_intersection",
    "memory_else_base",
    "memory_else_small_base_1",
    "memory_else_small_base_2",
    "memory_else_small_base_3",
    "memory_plus_small_base_1",
    "memory_plus_small_base_2",
    "memory_plus_small_base_3",
    "memory_union_near_base_1",
    "memory_union_near_base_2",
    "snap_memory_to_base_1",
    "snap_memory_to_base_2",
    "intersection_else_memory",
)


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"Scene causal record is not an object: {path}")
                records.append(value)
    return records


def read_reference_condition(
    root: Path,
    condition: str,
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index in range(runner.WORLD_SIZE):
        path = root / f"shard-{shard_index}" / f"scene.{condition}.jsonl"
        digest = sha256_file(path)
        expected = runner.REFERENCE_ARTIFACT_SHA256[condition][shard_index]
        if digest != expected:
            raise ValueError(f"Scene causal reference hash differs: {path}")
        rows = read_jsonl(path)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": digest,
            }
        )
        for record in rows:
            index = int(record["line_index"])
            if index < runner.SELECTION_ROWS or index % runner.WORLD_SIZE != shard_index:
                raise ValueError(f"Scene causal reference shard differs: {path}:{index}")
            if index in records:
                raise ValueError(f"Duplicate scene causal reference index: {index}")
            records[index] = record
    expected_indices = set(range(runner.SELECTION_ROWS, runner.EXPECTED_ROWS))
    if set(records) != expected_indices:
        raise ValueError(f"Incomplete scene causal reference condition: {condition}")
    return records, artifacts


def read_causal_condition(
    root: Path,
    condition: str,
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    shared_binding: dict[str, Any] | None = None
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "base_model_revision": "a4c2d58be94dda072b918d9db64ee85c8ed34e3f",
            "base_config_sha256": runner.BASE_CONFIG_SHA256,
            "memory_adapter_sha256": runner.MEMORY_ADAPTER_SHA256,
            "dataset_sha256": runner.DATASET_SHA256,
            "donor_mapping_rows": runner.EXPECTED_ROWS - runner.SELECTION_ROWS,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "conditions": list(runner.CONDITIONS),
            "runner_sha256": EXPECTED_RUNNER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError(f"Scene causal input binding differs: {binding_path}")
        comparable = dict(binding)
        comparable.pop("shard_index")
        if shared_binding is None:
            shared_binding = comparable
        elif comparable != shared_binding:
            raise ValueError("Scene causal input bindings differ across shards")
        bindings.append(
            {
                "path": str(binding_path.resolve()),
                "sha256": sha256_file(binding_path),
                "payload": binding,
            }
        )
        path = shard_dir / f"{condition}.jsonl"
        rows = read_jsonl(path)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
        for record in rows:
            index = int(record["source_index"])
            expected_record = {
                "schema": runner.SCHEMA,
                "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                "condition": condition,
                "shard_index": shard_index,
                "world_size": runner.WORLD_SIZE,
            }
            if any(record.get(key) != value for key, value in expected_record.items()):
                raise ValueError(f"Scene causal record binding differs: {path}:{index}")
            if index < runner.SELECTION_ROWS or index % runner.WORLD_SIZE != shard_index:
                raise ValueError(f"Scene causal record shard differs: {path}:{index}")
            if condition == "layer_permuted_correct_state" and (
                record.get("correct_state_sha256")
                == record.get("permuted_state_sha256")
            ):
                raise ValueError(f"Scene causal state permutation is unchanged: {index}")
            if index in records:
                raise ValueError(f"Duplicate scene causal source index: {index}")
            records[index] = record
    expected_indices = set(range(runner.SELECTION_ROWS, runner.EXPECTED_ROWS))
    if set(records) != expected_indices:
        raise ValueError(f"Incomplete scene causal condition: {condition}")
    return records, artifacts, bindings


def gold_and_hashes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, set[int]], dict[int, str]]:
    gold: dict[int, set[int]] = {}
    hashes: dict[int, str] = {}
    for row in rows:
        index = int(row["source_index"])
        if index < runner.SELECTION_ROWS:
            continue
        gold[index] = recovery.strict_gold_boundaries(row["gold"])
        hashes[index] = str(row["row_sha256"])
    return gold, hashes


def validate_hashes(
    records: Mapping[int, Mapping[str, Any]],
    hashes: Mapping[int, str],
    *,
    index_key: str,
    condition: str,
) -> None:
    if set(records) != set(hashes) or any(
        records[index].get("row_sha256") != hashes[index]
        for index in hashes
    ):
        raise ValueError(f"Scene causal row hashes differ: {condition}:{index_key}")


def prediction_set(record: Mapping[str, Any]) -> set[int] | None:
    prediction = record.get("prediction")
    if not isinstance(prediction, list):
        return None
    return {int(value) for value in prediction}


def route(name: str, base: set[int], memory: set[int]) -> set[int]:
    if name == "base":
        return set(base)
    if name == "memory":
        return set(memory)
    if name == "intersection":
        return base & memory
    if name == "conditional_intersection":
        return base & memory if memory else set(base)
    if name == "memory_else_base":
        return set(memory) if memory else set(base)
    if name.startswith("memory_else_small_base_"):
        limit = int(name.rsplit("_", 1)[1])
        if memory:
            return set(memory)
        return set(base) if len(base) <= limit else set()
    if name.startswith("memory_plus_small_base_"):
        limit = int(name.rsplit("_", 1)[1])
        return (base | memory) if len(base) <= limit else set(memory)
    if name.startswith("memory_union_near_base_"):
        radius = int(name.rsplit("_", 1)[1])
        nearby = {
            boundary
            for boundary in base
            if any(abs(boundary - memory_boundary) <= radius for memory_boundary in memory)
        }
        return memory | nearby
    if name.startswith("snap_memory_to_base_"):
        radius = int(name.rsplit("_", 1)[1])
        output: set[int] = set()
        for memory_boundary in memory:
            candidates = [
                base_boundary
                for base_boundary in base
                if abs(base_boundary - memory_boundary) <= radius
            ]
            output.add(
                min(candidates, key=lambda value: (abs(value - memory_boundary), value))
                if candidates
                else memory_boundary
            )
        return output
    if name == "intersection_else_memory":
        shared = base & memory
        return shared if shared else set(memory)
    raise ValueError(f"Unknown scene causal router: {name}")


def metrics_from_sets(
    predictions: Mapping[int, set[int] | None],
    gold: Mapping[int, set[int]],
    indices: Sequence[int],
) -> Mapping[str, Any]:
    tp = fp = fn = covered = 0
    for index in indices:
        prediction = predictions[index]
        covered += int(prediction is not None)
        predicted = set() if prediction is None else prediction
        expected = gold[index]
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        "rows": len(indices),
        "coverage": covered / len(indices),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": 0.0 if denominator == 0 else 2 * tp / denominator,
    }


def condition_predictions(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[int, set[int] | None]:
    return {index: prediction_set(record) for index, record in records.items()}


def routed_predictions(
    base: Mapping[int, set[int] | None],
    memory: Mapping[int, set[int] | None],
    name: str,
    indices: Sequence[int],
) -> dict[int, set[int] | None]:
    return {
        index: route(name, base[index] or set(), memory[index] or set())
        for index in indices
    }


def output_change_fraction(
    reference: Mapping[int, set[int] | None],
    counterfactual: Mapping[int, set[int] | None],
    indices: Sequence[int],
) -> float:
    return sum(reference[index] != counterfactual[index] for index in indices) / len(indices)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = runner.validate_protocol()
    root = args.input_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Scene causal output must be fresh: {output}")

    rows = runner.load_rows(dataset_root)
    gold, row_hashes = gold_and_hashes(rows)
    all_indices = tuple(sorted(gold))
    fit_indices = tuple(index for index in all_indices if index % 5 != 4)
    holdout_indices = tuple(index for index in all_indices if index % 5 == 4)

    base_records, base_artifacts = read_reference_condition(reference_root, "base")
    memory_records, memory_artifacts = read_reference_condition(reference_root, "memory")
    validate_hashes(base_records, row_hashes, index_key="line_index", condition="base")
    validate_hashes(memory_records, row_hashes, index_key="line_index", condition="memory")
    predictions: dict[str, dict[int, set[int] | None]] = {
        "base": condition_predictions(base_records),
        "memory": condition_predictions(memory_records),
    }
    causal_artifacts: dict[str, Any] = {}
    input_bindings: list[dict[str, Any]] | None = None
    for condition in runner.CONDITIONS:
        records, artifacts, bindings = read_causal_condition(root, condition)
        validate_hashes(
            records,
            row_hashes,
            index_key="source_index",
            condition=condition,
        )
        predictions[condition] = condition_predictions(records)
        causal_artifacts[condition] = artifacts
        if input_bindings is None:
            input_bindings = bindings
        elif bindings != input_bindings:
            raise ValueError("Scene causal binding provenance differs by condition")

    causal_metrics = {
        condition: metrics_from_sets(predictions[condition], gold, all_indices)
        for condition in ("memory", *runner.CONDITIONS)
    }
    correct_f1 = float(causal_metrics["memory"]["micro_f1"])
    causal_deltas = {
        condition: correct_f1 - float(causal_metrics[condition]["micro_f1"])
        for condition in runner.CONDITIONS
    }
    change_fractions = {
        condition: output_change_fraction(
            predictions["memory"], predictions[condition], all_indices
        )
        for condition in runner.CONDITIONS
    }
    causal_gates = {
        "coverage_at_least_0.95_each_condition": all(
            float(metrics["coverage"]) >= 0.95
            for metrics in causal_metrics.values()
        ),
        "correct_minus_zero_state_at_least_0.02": causal_deltas["zero_state"] >= 0.02,
        "correct_minus_donor_state_at_least_0.01": causal_deltas["donor_state"] >= 0.01,
        "correct_minus_layer_permuted_at_least_0.01": (
            causal_deltas["layer_permuted_correct_state"] >= 0.01
        ),
        "output_change_fraction_at_least_0.05_each_counterfactual": all(
            fraction >= 0.05 for fraction in change_fractions.values()
        ),
    }
    causal_gates["passed"] = all(causal_gates.values())

    fit_candidates: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        candidate_predictions = routed_predictions(
            predictions["base"], predictions["memory"], name, fit_indices
        )
        fit_candidates[name] = metrics_from_sets(candidate_predictions, gold, fit_indices)
    selected = min(
        CANDIDATE_NAMES,
        key=lambda name: (
            -float(fit_candidates[name]["micro_f1"]),
            -float(fit_candidates[name]["precision"]),
            -float(fit_candidates[name]["recall"]),
            name,
        ),
    )
    holdout_metrics: dict[str, Any] = {}
    for name in ("base", "memory", selected):
        candidate_predictions = routed_predictions(
            predictions["base"], predictions["memory"], name, holdout_indices
        )
        holdout_metrics[name] = metrics_from_sets(
            candidate_predictions, gold, holdout_indices
        )
    fit_gain_over_memory = (
        float(fit_candidates[selected]["micro_f1"])
        - float(fit_candidates["memory"]["micro_f1"])
    )
    holdout_gain_over_memory = (
        float(holdout_metrics[selected]["micro_f1"])
        - float(holdout_metrics["memory"]["micro_f1"])
    )
    holdout_gain_over_base = (
        float(holdout_metrics[selected]["micro_f1"])
        - float(holdout_metrics["base"]["micro_f1"])
    )
    router_gates = {
        "fit_gain_over_memory_at_least_0.005": fit_gain_over_memory >= 0.005,
        "holdout_no_regression_vs_memory": holdout_gain_over_memory >= 0.0,
        "holdout_gain_over_base_at_least_0.005": holdout_gain_over_base >= 0.005,
        "holdout_coverage_at_least_0.95": (
            float(holdout_metrics[selected]["coverage"]) >= 0.95
        ),
    }
    router_gates["passed"] = all(router_gates.values())

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "split": "publisher-TRAIN-derived development",
            "rows": len(all_indices),
            "fit_rows": len(fit_indices),
            "holdout_rows": len(holdout_indices),
            "fit_partition": protocol["data_scope"]["router_fit_partition"],
            "holdout_partition": protocol["data_scope"]["router_holdout_partition"],
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
        },
        "causal": {
            "metrics": causal_metrics,
            "correct_minus_counterfactual_f1": causal_deltas,
            "paired_output_change_fraction": change_fractions,
            "gates": causal_gates,
        },
        "router": {
            "fit_candidates": fit_candidates,
            "selected": selected,
            "selected_fit_minus_memory": fit_gain_over_memory,
            "holdout": holdout_metrics,
            "selected_holdout_minus_memory": holdout_gain_over_memory,
            "selected_holdout_minus_base": holdout_gain_over_base,
            "gates": router_gates,
            "future_replication_candidate": selected if router_gates["passed"] else None,
            "accepted_validation_decoder_changed": False,
        },
        "overall": {
            "causal_passed": causal_gates["passed"],
            "router_passed": router_gates["passed"],
            "passed": causal_gates["passed"] and router_gates["passed"],
        },
        "provenance": {
            "input_bindings": input_bindings,
            "base": base_artifacts,
            "memory": memory_artifacts,
            **causal_artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_scene_causal_router_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["overall"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
