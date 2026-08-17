#!/usr/bin/env python3
"""Aggregate and sign the address-keyed RWKV plus DeepEmbed native benchmark."""

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
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_eval
    as evaluation,
)


SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_result.v1"
)
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
            f"Expected four DeepEmbed partitions, found {len(partition_dirs)}"
        )
    for partition_dir in partition_dirs:
        binding_path = partition_dir / "input_binding.json"
        if not binding_path.is_file():
            raise ValueError(f"Missing DeepEmbed binding: {binding_path}")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (
            binding.get("schema") != evaluation.INPUT_SCHEMA
            or binding.get("protocol_payload_sha256")
            != evaluation.PROTOCOL_PAYLOAD_SHA256
            or binding.get("partitions_per_shard") != PARTITIONS_PER_SHARD
            or binding.get("conditions") != list(evaluation.CONDITIONS)
            or binding.get("hybrid_mode")
            != "address_keyed_moe_deepembed_ffn"
            or binding.get("generation_batch_shape_control")
            != "four_by_four_same_position"
            or binding.get("projected_carrier_active") is not True
            or binding.get("recurrent_attention_path_active") is not True
            or binding.get("deepembed_outer_ffn_path_active") is not True
            or binding.get("deepembed_outer_ffn_layers")
            != list(evaluation.OUTER_FFN_LAYERS)
            or binding.get(
                "zero_recurrent_identity_requires_explicit_bypass_match"
            )
            is not True
        ):
            raise ValueError(f"DeepEmbed binding differs: {binding_path}")
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
                raise ValueError(f"Missing DeepEmbed predictions: {path}")
            rows = 0
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                source_index = int(record["source_index"])
                combined_path_active = condition != "projected_only_bypass"
                required = {
                    "schema": evaluation.SCHEMA,
                    "protocol_payload_sha256": evaluation.PROTOCOL_PAYLOAD_SHA256,
                    "training_protocol_payload_sha256": (
                        evaluation.TRAINING_PROTOCOL_PAYLOAD_SHA256
                    ),
                    "condition": condition,
                    "seed": evaluation.SEED,
                    "world_size": evaluation.WORLD_SIZE,
                    "projected_carrier_active": True,
                    "serialized_adapter_hybrid_mode": (
                        "address_keyed_moe_deepembed_ffn"
                    ),
                    "generation_batch_shape_control": (
                        "four_by_four_same_position"
                    ),
                    "recurrent_attention_path_active": combined_path_active,
                    "deepembed_outer_ffn_path_active": combined_path_active,
                    "deepembed_outer_ffn_layers": list(
                        evaluation.OUTER_FFN_LAYERS
                    ),
                    "explicit_projected_only_bypass": not combined_path_active,
                }
                if any(record.get(key) != value for key, value in required.items()):
                    raise ValueError(
                        f"DeepEmbed prediction binding differs: {path}:{source_index}"
                    )
                if source_index in by_condition[condition]:
                    raise ValueError(
                        f"Duplicate DeepEmbed prediction: {condition}:{source_index}"
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
        raise ValueError("DeepEmbed correct-state row count differs")
    for condition, records in by_condition.items():
        if set(records) != expected_indices:
            raise ValueError(f"DeepEmbed condition row set differs: {condition}")
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
        "correct_minus_zero_micro_f1": correct_f1
        - float(metrics["zero_recurrent_state"]["micro_f1"]),
        "correct_minus_projected_only_micro_f1": correct_f1
        - float(metrics["projected_only_bypass"]["micro_f1"]),
        "correct_minus_matched_donor_micro_f1": correct_f1
        - float(metrics["matched_donor_recurrent_state"]["micro_f1"]),
        "correct_minus_layer_permuted_micro_f1": correct_f1
        - float(metrics["layer_permuted_recurrent_state"]["micro_f1"]),
    }
    source_indices = sorted(records["correct_recurrent_state"])
    zero_projected_prediction_exact = all(
        records["zero_recurrent_state"][source]["prediction"]
        == records["projected_only_bypass"][source]["prediction"]
        for source in source_indices
    )
    zero_projected_raw_exact = all(
        records["zero_recurrent_state"][source]["raw_generation"]
        == records["projected_only_bypass"][source]["raw_generation"]
        for source in source_indices
    )
    projected_carrier_fixed = all(
        record["projected_carrier_byte_identical"] is True
        for condition in evaluation.CONDITIONS
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
        "correct_minus_projected_only_micro_f1_minimum": (
            margins["correct_minus_projected_only_micro_f1"] >= MARGIN_MINIMUM
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
        "zero_recurrent_exactly_matches_projected_only_predictions": (
            zero_projected_prediction_exact
        ),
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
    }
    passed = all(gates.values())
    native_gain = (
        gates["correct_minus_projected_only_micro_f1_minimum"]
        and gates["correct_minus_zero_micro_f1_minimum"]
    )
    status = (
        "address_keyed_deepembed_native_recurrent_ffn_causal_gain_established"
        if passed
        else "address_keyed_deepembed_native_gain_without_full_causal_pass"
        if native_gain
        else "address_keyed_deepembed_native_gain_not_established"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "passed": passed,
        "native_recurrent_ffn_causal_gain_established": passed,
        "protocol_payload_sha256": evaluation.PROTOCOL_PAYLOAD_SHA256,
        "training_protocol_payload_sha256": (
            evaluation.TRAINING_PROTOCOL_PAYLOAD_SHA256
        ),
        "seed": evaluation.SEED,
        "rows": evaluation.EVALUATION_ROWS,
        "condition_metrics": metrics,
        "causal_margins": margins,
        "paired_output_change_fraction_vs_correct": paired_change,
        "diagnostics": {
            "zero_recurrent_raw_generation_exactly_matches_projected_only": (
                zero_projected_raw_exact
            )
        },
        "gates": gates,
        "thresholds": {
            "coverage_minimum": COVERAGE_MINIMUM,
            "causal_micro_f1_margin_minimum": MARGIN_MINIMUM,
        },
        "attribution": {
            "established_if_passed": (
                "Combined recurrent RWKV attention plus recurrent-conditioned "
                "sparse DeepEmbed outer-FFN path."
            ),
            "not_separately_identified": (
                "Attention-path contribution versus outer-FFN contribution."
            ),
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
        raise ValueError(f"DeepEmbed result output must be fresh: {path}")
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
                "condition_micro_f1": {
                    condition: metrics["micro_f1"]
                    for condition, metrics in result["condition_metrics"].items()
                },
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
