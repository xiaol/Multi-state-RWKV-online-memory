#!/usr/bin/env python3
"""Aggregate and sign the internally routed recurrent-value evaluation."""

from __future__ import annotations

import argparse
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
    analyze_natural_memory_native_rwkv_addressed_value_eval as addressed_analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_recurrent_value_eval as evaluation,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_value_result.v1"
MARGIN_MINIMUM = 0.005
COVERAGE_MINIMUM = 0.95
PARTITIONS_PER_SHARD = 1


def canonical_sha256(value: Any) -> str:
    return evaluation.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return evaluation.sha256_file(path)


def read_records(
    root: Path,
) -> tuple[dict[str, dict[int, Mapping[str, Any]]], list[Mapping[str, Any]]]:
    by_condition = {condition: {} for condition in evaluation.CONDITIONS}
    artifact_manifest: list[Mapping[str, Any]] = []
    partition_dirs = sorted(root.glob("shard-*/partition-*-of-*"))
    if len(partition_dirs) != evaluation.WORLD_SIZE * PARTITIONS_PER_SHARD:
        raise ValueError(
            f"Expected four recurrent-value partitions, found {len(partition_dirs)}"
        )
    for partition_dir in partition_dirs:
        binding_path = partition_dir / "input_binding.json"
        if not binding_path.is_file():
            raise ValueError(f"Missing recurrent-value binding: {binding_path}")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding.get("schema") != evaluation.INPUT_SCHEMA
            or binding.get("protocol_payload_sha256")
            != evaluation.PROTOCOL_PAYLOAD_SHA256
            or binding.get("partitions_per_shard") != PARTITIONS_PER_SHARD
            or binding.get("conditions") != list(evaluation.CONDITIONS)
            or binding.get("projected_bundle_read_inert") is not True
            or binding.get("rwkv_internal_router_active") is not True
        ):
            raise ValueError(f"Recurrent-value binding differs: {binding_path}")
        artifact_manifest.append(
            {
                "kind": "input_binding",
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
            }
        )
        for condition in evaluation.CONDITIONS:
            path = partition_dir / f"{condition}.jsonl"
            if not path.is_file():
                raise ValueError(f"Missing recurrent-value predictions: {path}")
            rows = 0
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                source_index = int(record["source_index"])
                required = {
                    "schema": evaluation.SCHEMA,
                    "protocol_payload_sha256": evaluation.PROTOCOL_PAYLOAD_SHA256,
                    "condition": condition,
                    "seed": evaluation.SEED,
                    "world_size": evaluation.WORLD_SIZE,
                    "projected_bundle_read_inert": True,
                    "rwkv_internal_router_active": True,
                }
                if any(record.get(key) != value for key, value in required.items()):
                    raise ValueError(
                        f"Recurrent-value prediction binding differs: "
                        f"{path}:{source_index}"
                    )
                if source_index in by_condition[condition]:
                    raise ValueError(
                        f"Duplicate recurrent-value prediction: {condition}:{source_index}"
                    )
                by_condition[condition][source_index] = record
                rows += 1
            artifact_manifest.append(
                {
                    "kind": "predictions",
                    "condition": condition,
                    "path": str(path),
                    "rows": rows,
                    "sha256": sha256_file(path),
                }
            )
    expected_indices = set(by_condition["correct_recurrent_state"])
    if len(expected_indices) != evaluation.EVALUATION_ROWS:
        raise ValueError("Recurrent-value correct-state row count differs")
    for condition, records in by_condition.items():
        if set(records) != expected_indices:
            raise ValueError(f"Recurrent-value condition row set differs: {condition}")
    return by_condition, artifact_manifest


def aggregate_condition(records: Mapping[int, Mapping[str, Any]]) -> Mapping[str, Any]:
    return addressed_analyzer.aggregate_condition(records)


def analyze(root: Path) -> Mapping[str, Any]:
    root = root.expanduser().resolve(strict=True)
    records, artifact_manifest = read_records(root)
    metrics = {
        condition: aggregate_condition(condition_records)
        for condition, condition_records in records.items()
    }
    correct_f1 = float(metrics["correct_recurrent_state"]["micro_f1"])
    margins = {
        "correct_minus_empty_micro_f1": correct_f1
        - float(metrics["empty_memory"]["micro_f1"]),
        "correct_minus_zero_micro_f1": correct_f1
        - float(metrics["zero_recurrent_state"]["micro_f1"]),
        "correct_minus_matched_donor_micro_f1": correct_f1
        - float(metrics["matched_donor_recurrent_state"]["micro_f1"]),
        "correct_minus_layer_permuted_micro_f1": correct_f1
        - float(metrics["layer_permuted_recurrent_state"]["micro_f1"]),
    }
    source_indices = sorted(records["correct_recurrent_state"])
    zero_empty_exact = all(
        records["zero_recurrent_state"][source]["prediction"]
        == records["empty_memory"][source]["prediction"]
        for source in source_indices
    )
    projected_carrier_fixed = all(
        record["projected_carrier_byte_identical"] is True
        for condition in (
            "correct_recurrent_state",
            "zero_recurrent_state",
            "matched_donor_recurrent_state",
            "layer_permuted_recurrent_state",
        )
        for record in records[condition].values()
    )
    paired_change = {
        condition: sum(
            records["correct_recurrent_state"][source]["prediction"]
            != records[condition][source]["prediction"]
            for source in source_indices
        )
        / len(source_indices)
        for condition in evaluation.CONDITIONS
        if condition != "correct_recurrent_state"
    }
    gates = {
        "coverage_minimum_every_condition": all(
            float(value["coverage"]) >= COVERAGE_MINIMUM
            for value in metrics.values()
        ),
        "correct_minus_empty_micro_f1_minimum": (
            margins["correct_minus_empty_micro_f1"] >= MARGIN_MINIMUM
        ),
        "correct_minus_zero_micro_f1_minimum": (
            margins["correct_minus_zero_micro_f1"] >= MARGIN_MINIMUM
        ),
        "correct_minus_matched_donor_micro_f1_minimum": (
            margins["correct_minus_matched_donor_micro_f1"] >= MARGIN_MINIMUM
        ),
        "correct_minus_layer_permuted_micro_f1_minimum": (
            margins["correct_minus_layer_permuted_micro_f1"] >= MARGIN_MINIMUM
        ),
        "zero_recurrent_exactly_matches_empty_memory_predictions": zero_empty_exact,
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
        "projected_bundle_read_inert_contract": True,
        "rwkv_internal_router_active_contract": True,
    }
    passed = all(gates.values())
    native_gain = (
        margins["correct_minus_empty_micro_f1"] >= MARGIN_MINIMUM
        and margins["correct_minus_zero_micro_f1"] >= MARGIN_MINIMUM
    )
    status = (
        "native_recurrent_causal_gain_established"
        if passed
        else "native_gain_without_full_recurrent_causal_pass"
        if native_gain
        else "recurrent_value_native_gain_not_established"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "passed": passed,
        "native_recurrent_causal_gain_established": passed,
        "protocol_payload_sha256": evaluation.PROTOCOL_PAYLOAD_SHA256,
        "seed": evaluation.SEED,
        "rows": evaluation.EVALUATION_ROWS,
        "condition_metrics": metrics,
        "causal_margins": margins,
        "paired_output_change_fraction_vs_correct": paired_change,
        "gates": gates,
        "thresholds": {
            "coverage_minimum": COVERAGE_MINIMUM,
            "causal_micro_f1_margin_minimum": MARGIN_MINIMUM,
        },
        "scope": {
            "split": "publisher-TRAIN-derived authorized development partition",
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
            "strength_holdout_opened": False,
        },
        "artifacts": artifact_manifest,
        "code_bindings": {
            "analyzer_sha256": sha256_file(Path(__file__)),
            "evaluation_runner_sha256": sha256_file(Path(evaluation.__file__)),
            "training_runner_sha256": sha256_file(Path(evaluation.training.__file__)),
            "addressed_analyzer_helper_sha256": sha256_file(
                Path(addressed_analyzer.__file__)
            ),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    return result


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Recurrent-value result output must be fresh: {path}")
    path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.evaluation_root)
    write_result(args.output.expanduser().resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "causal_margins": result["causal_margins"],
                "result_receipt": result["receipt"]["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
