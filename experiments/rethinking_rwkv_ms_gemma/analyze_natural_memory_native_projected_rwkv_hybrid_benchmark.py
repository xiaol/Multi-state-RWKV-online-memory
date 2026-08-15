#!/usr/bin/env python3
"""Analyze and sign the paired projected-KV/RWKV native benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_train as training,
)


SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_benchmark_result.v1"
PRIMARY_CONDITIONS = {
    "projected_control": "correct_state",
    "hybrid_candidate": "correct_recurrent_state",
}


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            records.append(json.loads(raw_line))
    return records


def prediction(record: Mapping[str, Any]) -> tuple[int, ...] | None:
    value = record.get("prediction")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Evaluation prediction is not a list or null")
    return tuple(sorted({int(item) for item in value}))


def metrics(
    records: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, set[int]],
) -> Mapping[str, Any]:
    if set(records) != set(gold):
        raise ValueError("Metric records do not cover the authorized rows")
    covered = 0
    tp = fp = fn = 0
    for source, record in records.items():
        predicted_tuple = prediction(record)
        if predicted_tuple is not None:
            covered += 1
        predicted = set() if predicted_tuple is None else set(predicted_tuple)
        expected = gold[source]
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return {
        "rows": len(records),
        "covered_rows": covered,
        "coverage": covered / len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
    }


def validate_binding(
    binding: Mapping[str, Any],
    *,
    architecture: str,
    seed: int,
    shard_index: int,
) -> None:
    expected = {
        "schema": evaluator.INPUT_SCHEMA,
        "protocol_payload_sha256": evaluator.PROTOCOL_PAYLOAD_SHA256,
        "architecture": architecture,
        "seed": seed,
        "shard_index": shard_index,
        "world_size": evaluator.WORLD_SIZE,
        "dataset_sha256": evaluator.DATASET_SHA256,
        "authorized_rows": evaluator.EVALUATION_ROWS,
        "authorized_rows_payload_sha256": evaluator.AUTHORIZED_ROWS_PAYLOAD_SHA256,
        "conditions": list(evaluator.CONDITIONS[architecture]),
        "hf_endpoint": evaluator.HF_MIRROR_ENDPOINT,
        "runner_sha256": sha256_file(Path(evaluator.__file__)),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Evaluation input binding differs: {architecture}:seed-{seed}:shard-{shard_index}"
        )


def load_outputs(
    root: Path,
    expected_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[int, dict[str, dict[int, Mapping[str, Any]]]]],
    list[dict[str, Any]],
]:
    expected_indices = set(expected_rows)
    outputs: dict[str, dict[int, dict[str, dict[int, Mapping[str, Any]]]]] = {}
    artifacts: list[dict[str, Any]] = []
    for architecture in evaluator.ARCHITECTURES:
        outputs[architecture] = {}
        for seed in evaluator.SEEDS:
            conditions = {
                condition: {} for condition in evaluator.CONDITIONS[architecture]
            }
            outputs[architecture][seed] = conditions
            for shard_index in range(evaluator.WORLD_SIZE):
                shard_dir = root / architecture / f"seed-{seed}" / f"shard-{shard_index}"
                binding_path = shard_dir / "input_binding.json"
                binding = json.loads(binding_path.read_text(encoding="utf-8"))
                validate_binding(
                    binding,
                    architecture=architecture,
                    seed=seed,
                    shard_index=shard_index,
                )
                artifacts.append(
                    {
                        "kind": "input_binding",
                        "architecture": architecture,
                        "seed": seed,
                        "shard_index": shard_index,
                        "path": str(binding_path),
                        "sha256": sha256_file(binding_path),
                    }
                )
                for condition in conditions:
                    path = shard_dir / f"{condition}.jsonl"
                    records = read_jsonl(path)
                    artifacts.append(
                        {
                            "kind": "predictions",
                            "architecture": architecture,
                            "seed": seed,
                            "condition": condition,
                            "shard_index": shard_index,
                            "path": str(path),
                            "rows": len(records),
                            "sha256": sha256_file(path),
                        }
                    )
                    for record in records:
                        source = int(record["source_index"])
                        expected = expected_rows.get(source)
                        required = {
                            "schema": evaluator.SCHEMA,
                            "protocol_payload_sha256": evaluator.PROTOCOL_PAYLOAD_SHA256,
                            "architecture": architecture,
                            "condition": condition,
                            "shard_index": shard_index,
                            "world_size": evaluator.WORLD_SIZE,
                            "source_index": source,
                            "row_sha256": None if expected is None else expected["row_sha256"],
                        }
                        if (
                            expected is None
                            or source % evaluator.WORLD_SIZE != shard_index
                            or any(record.get(key) != value for key, value in required.items())
                        ):
                            raise ValueError(f"Evaluation record binding differs: {path}:{source}")
                        expected_gold = evaluator.recovery.strict_gold_boundaries(
                            expected["gold"]
                        )
                        if record.get("gold") != sorted(expected_gold):
                            raise ValueError(f"Evaluation gold differs: {path}:{source}")
                        expected_score = evaluator.record_score(
                            None if prediction(record) is None else prediction(record),
                            expected_gold,
                        )
                        if record.get("score") != expected_score:
                            raise ValueError(f"Evaluation score differs: {path}:{source}")
                        if source in conditions[condition]:
                            raise ValueError(
                                f"Duplicate evaluation record: {architecture}:{seed}:{condition}:{source}"
                            )
                        conditions[condition][source] = record
            for condition, records in conditions.items():
                if set(records) != expected_indices:
                    raise ValueError(
                        f"Incomplete evaluation output: {architecture}:{seed}:{condition}"
                    )
    return outputs, artifacts


def output_change_fraction(
    left: Mapping[int, Mapping[str, Any]],
    right: Mapping[int, Mapping[str, Any]],
) -> float:
    if set(left) != set(right):
        raise ValueError("Paired output rows differ")
    return sum(prediction(left[index]) != prediction(right[index]) for index in left) / len(left)


def build_analysis(
    outputs: Mapping[str, Mapping[int, Mapping[str, Mapping[int, Mapping[str, Any]]]]],
    gold: Mapping[int, set[int]],
) -> Mapping[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    carrier_fixed = True
    zero_matches_bypass = True
    primary_change_fractions: list[float] = []
    primary_deltas: list[float] = []
    donor_deltas: list[float] = []
    zero_deltas: list[float] = []
    permutation_deltas: list[float] = []
    nonnegative_seeds = 0
    coverage_gates: list[bool] = []
    for seed in evaluator.SEEDS:
        projected = outputs["projected_control"][seed]["correct_state"]
        hybrid_conditions = outputs["hybrid_candidate"][seed]
        hybrid = hybrid_conditions["correct_recurrent_state"]
        projected_metrics = metrics(projected, gold)
        condition_metrics = {
            condition: metrics(records, gold)
            for condition, records in hybrid_conditions.items()
        }
        hybrid_metrics = condition_metrics["correct_recurrent_state"]
        primary_delta = float(hybrid_metrics["micro_f1"]) - float(
            projected_metrics["micro_f1"]
        )
        donor_delta = float(hybrid_metrics["micro_f1"]) - float(
            condition_metrics["matched_donor_recurrent_state"]["micro_f1"]
        )
        zero_delta = float(hybrid_metrics["micro_f1"]) - float(
            condition_metrics["zero_recurrent_state"]["micro_f1"]
        )
        permutation_delta = float(hybrid_metrics["micro_f1"]) - float(
            condition_metrics["layer_permuted_recurrent_state"]["micro_f1"]
        )
        change_fraction = output_change_fraction(hybrid, projected)
        for source in gold:
            carriers = {
                str(records[source].get("projected_carrier_sha256"))
                for records in hybrid_conditions.values()
            }
            carrier_fixed = carrier_fixed and len(carriers) == 1 and all(
                record[source].get("projected_carrier_byte_identical") is True
                for record in hybrid_conditions.values()
            )
            zero_matches_bypass = zero_matches_bypass and (
                prediction(hybrid_conditions["zero_recurrent_state"][source])
                == prediction(hybrid_conditions["projected_only_bypass"][source])
            )
        primary_deltas.append(primary_delta)
        donor_deltas.append(donor_delta)
        zero_deltas.append(zero_delta)
        permutation_deltas.append(permutation_delta)
        primary_change_fractions.append(change_fraction)
        nonnegative_seeds += int(primary_delta >= 0.0)
        coverage_gates.extend(
            (
                float(projected_metrics["coverage"]) >= 0.95,
                float(hybrid_metrics["coverage"]) >= 0.95,
            )
        )
        per_seed[str(seed)] = {
            "projected_control": projected_metrics,
            "hybrid_conditions": condition_metrics,
            "hybrid_minus_projected_micro_f1": primary_delta,
            "hybrid_correct_minus_donor_micro_f1": donor_delta,
            "hybrid_correct_minus_zero_micro_f1": zero_delta,
            "hybrid_correct_minus_layer_permuted_micro_f1": permutation_delta,
            "paired_output_change_fraction_hybrid_vs_projected": change_fraction,
        }
    aggregates = {
        "mean_hybrid_minus_projected_micro_f1": statistics.fmean(primary_deltas),
        "hybrid_nonnegative_vs_projected_seed_count": nonnegative_seeds,
        "mean_hybrid_correct_minus_donor_micro_f1": statistics.fmean(donor_deltas),
        "mean_hybrid_correct_minus_zero_micro_f1": statistics.fmean(zero_deltas),
        "mean_hybrid_correct_minus_layer_permuted_micro_f1": statistics.fmean(
            permutation_deltas
        ),
        "mean_paired_output_change_fraction_hybrid_vs_projected": statistics.fmean(
            primary_change_fractions
        ),
        "zero_recurrent_exactly_matches_projected_bypass_predictions": zero_matches_bypass,
        "projected_carrier_hash_fixed_for_every_hybrid_intervention": carrier_fixed,
    }
    gates = {
        "coverage_minimum_every_primary_arm": all(coverage_gates),
        "mean_hybrid_minus_projected_micro_f1_minimum": aggregates[
            "mean_hybrid_minus_projected_micro_f1"
        ]
        >= 0.005,
        "hybrid_nonnegative_vs_projected_seed_count_minimum": nonnegative_seeds >= 2,
        "mean_hybrid_correct_minus_donor_micro_f1_minimum": aggregates[
            "mean_hybrid_correct_minus_donor_micro_f1"
        ]
        >= 0.005,
        "mean_hybrid_correct_minus_zero_micro_f1_minimum": aggregates[
            "mean_hybrid_correct_minus_zero_micro_f1"
        ]
        >= 0.02,
        "mean_hybrid_correct_minus_layer_permuted_micro_f1_minimum": aggregates[
            "mean_hybrid_correct_minus_layer_permuted_micro_f1"
        ]
        >= 0.005,
        "zero_recurrent_exactly_matches_projected_bypass_predictions": zero_matches_bypass,
        "projected_carrier_hash_fixed_for_every_hybrid_intervention": carrier_fixed,
        "paired_output_change_fraction_hybrid_vs_projected_minimum": aggregates[
            "mean_paired_output_change_fraction_hybrid_vs_projected"
        ]
        >= 0.05,
    }
    benchmark_gates = (
        "coverage_minimum_every_primary_arm",
        "mean_hybrid_minus_projected_micro_f1_minimum",
        "hybrid_nonnegative_vs_projected_seed_count_minimum",
        "paired_output_change_fraction_hybrid_vs_projected_minimum",
    )
    benchmark_gain = all(gates[name] for name in benchmark_gates)
    passed = all(gates.values())
    return {
        "per_seed": per_seed,
        "aggregates": aggregates,
        "gates": {**gates, "passed": passed},
        "benchmark_gain_established": benchmark_gain,
        "recurrent_rwkv_causal_attribution_established": passed,
        "status": (
            "native_benchmark_and_recurrent_causal_pass"
            if passed
            else "native_benchmark_gain_without_recurrent_causal_pass"
            if benchmark_gain
            else "native_benchmark_gain_not_established"
        ),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Benchmark analysis output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=training.DATASET_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training.validate_protocol()
    input_root = args.input_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    dataset_file = dataset_root / evaluator.DATASET_RELATIVE_PATH
    if sha256_file(dataset_file) != evaluator.DATASET_SHA256:
        raise ValueError("Authorized benchmark dataset hash differs")
    rows = evaluator.parse_authorized_rows(evaluator.raw_line_metadata(dataset_file))
    expected_rows = {int(row["source_index"]): row for row in rows}
    gold = {
        source: evaluator.recovery.strict_gold_boundaries(row["gold"])
        for source, row in expected_rows.items()
    }
    outputs, artifacts = load_outputs(input_root, expected_rows)
    analysis = build_analysis(outputs, gold)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": evaluator.PROTOCOL_PAYLOAD_SHA256,
        "evaluation_runner_sha256": sha256_file(Path(evaluator.__file__)),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "scope": {
            "split": "publisher-TRAIN-derived authorized fit remainder",
            "rows_per_arm": evaluator.EVALUATION_ROWS,
            "seeds": list(evaluator.SEEDS),
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
            "strength_holdout_opened": False,
        },
        **analysis,
        "provenance": {"evaluation_artifacts": artifacts},
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["gates"]["passed"],
                "aggregates": result["aggregates"],
                "receipt": result["receipt"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
