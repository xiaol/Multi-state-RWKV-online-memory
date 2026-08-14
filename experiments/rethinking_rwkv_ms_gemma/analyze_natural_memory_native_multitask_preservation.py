#!/usr/bin/env python3
"""Aggregate and sign checkpoint-16 multitask preservation results."""

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
    analyze_natural_memory_native_routed_benchmark as routed_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_multitask_preservation as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_multitask_preservation_result.v1"


def validate_bindings(root: Path) -> list[Mapping[str, Any]]:
    bindings: list[Mapping[str, Any]] = []
    shared: dict[str, Any] | None = None
    expected_runner_sha256 = runner.sha256_file(Path(runner.__file__))
    for shard_index in range(runner.WORLD_SIZE):
        path = root / f"shard-{shard_index}" / "input_binding.json"
        binding = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "dataset_receipt_payload_sha256": runner.routed.NATIVE_DATASET_RECEIPT_SHA256,
            "narrative_payload_sha256": runner.NARRATIVE_PAYLOAD_SHA256,
            "narrative_rows": runner.NARRATIVE_ROWS,
            "reference_result_sha256": runner.REFERENCE_FILE_SHA256,
            "reference_result_receipt_sha256": runner.REFERENCE_RECEIPT_SHA256,
            "progression_result_sha256": runner.PROGRESSION_FILE_SHA256,
            "progression_result_receipt_sha256": runner.PROGRESSION_RECEIPT_SHA256,
            "selected_checkpoint_step": runner.SELECTED_STEP,
            "selected_gate_state_sha256": runner.SELECTED_GATE_STATE_SHA256,
            "selected_patch_sha256": runner.SELECTED_PATCH_SHA256,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "runner_sha256": expected_runner_sha256,
            "protected_splits_opened": [],
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Multitask preservation input binding differs: {path}")
        comparable = dict(binding)
        comparable.pop("shard_index")
        if shared is None:
            shared = comparable
        elif comparable != shared:
            raise ValueError("Multitask preservation bindings differ across shards")
        bindings.append(
            {"path": str(path), "sha256": runner.sha256_file(path), "payload": binding}
        )
    return bindings


def collect_checkpoint_records(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], list[Mapping[str, Any]]]:
    expected_rows = {int(row["line_index"]): row for row in rows}
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[Mapping[str, Any]] = []
    for shard_index in range(runner.WORLD_SIZE):
        path = root / f"shard-{shard_index}" / "narrative.checkpoint16.jsonl"
        shard_records = runner.read_completed(path)
        runner.validate_resume(
            shard_records,
            [row for row in rows if int(row["line_index"]) % runner.WORLD_SIZE == shard_index],
            shard_index=shard_index,
        )
        artifacts.append(
            {"path": str(path), "rows": len(shard_records), "sha256": runner.sha256_file(path)}
        )
        for index, record in shard_records.items():
            if index in records:
                raise ValueError(f"Duplicate multitask preservation row: {index}")
            records[index] = record
    if set(records) != set(expected_rows):
        raise ValueError("Incomplete multitask preservation narrative records")
    return records, artifacts


def prediction_change_fraction(
    left: Mapping[int, Mapping[str, Any]],
    right: Mapping[int, Mapping[str, Any]],
) -> float:
    if set(left) != set(right):
        raise ValueError("Prediction comparison rows differ")
    return sum(left[index].get("prediction") != right[index].get("prediction") for index in left) / len(left)


def preservation_gates(
    *,
    memory_coverage: float,
    routed_minus_base: float,
    routed_minus_v9: float,
    attribution_exact: bool,
    scene_progression_passed: bool,
) -> Mapping[str, bool]:
    gates = {
        "narrative_memory_coverage_at_least_0.95": memory_coverage >= 0.95,
        "narrative_routed_no_regression_vs_base": routed_minus_base >= 0.0,
        "narrative_routed_no_regression_vs_v9": routed_minus_v9 >= 0.0,
        "attribution_exact_frozen_base_artifact_reuse": attribution_exact,
        "scene_progression_passed": scene_progression_passed,
    }
    return {**gates, "passed": all(gates.values())}


def analyze(
    *,
    input_root: Path,
    dataset_root: Path,
    reference_root: Path,
    progression_result_path: Path,
    output: Path,
) -> Mapping[str, Any]:
    runner.validate_protocol()
    bindings = validate_bindings(input_root)
    runner.validate_reference(reference_root)
    progression = runner.validate_progression(progression_result_path)
    narrative_rows = runner.selected_narrative_rows(
        runner.routed.load_rows(dataset_root)["narrative"]
    )
    checkpoint, checkpoint_artifacts = collect_checkpoint_records(input_root, narrative_rows)

    gold_narrative, narrative_hashes = routed_analysis.load_gold(dataset_root, "narrative")
    base_narrative, base_narrative_artifacts = routed_analysis.collect_condition(
        reference_root, task="narrative", condition="base"
    )
    v9_narrative, v9_narrative_artifacts = routed_analysis.collect_condition(
        reference_root, task="narrative", condition="memory"
    )
    routed_analysis.validate_record_row_hashes(
        checkpoint, narrative_hashes, task="narrative", condition="checkpoint16"
    )
    base_metrics = routed_analysis.narrative_metrics(base_narrative, gold_narrative)
    v9_memory_metrics = routed_analysis.narrative_metrics(v9_narrative, gold_narrative)
    v9_routed_records = routed_analysis.routed_narrative_records(base_narrative, v9_narrative)
    v9_routed_metrics = routed_analysis.narrative_metrics(v9_routed_records, gold_narrative)
    checkpoint_memory_metrics = routed_analysis.narrative_metrics(checkpoint, gold_narrative)
    checkpoint_routed_records = routed_analysis.routed_narrative_records(base_narrative, checkpoint)
    checkpoint_routed_metrics = routed_analysis.narrative_metrics(
        checkpoint_routed_records, gold_narrative
    )
    routed_minus_base = float(checkpoint_routed_metrics["primary_metric"]) - float(
        base_metrics["primary_metric"]
    )
    routed_minus_v9 = float(checkpoint_routed_metrics["primary_metric"]) - float(
        v9_routed_metrics["primary_metric"]
    )

    gold_attribution, attribution_hashes = routed_analysis.load_gold(dataset_root, "attribution")
    base_attribution, base_attribution_artifacts = routed_analysis.collect_condition(
        reference_root, task="attribution", condition="base"
    )
    routed_analysis.validate_record_row_hashes(
        base_attribution, attribution_hashes, task="attribution", condition="base"
    )
    attribution_metrics = routed_analysis.attribution_metrics(base_attribution, gold_attribution)
    attribution_payload = [
        {"line_index": index, "selected": base_attribution[index]["selected"]}
        for index in sorted(base_attribution)
    ]
    attribution_exact = True
    scene_progression_passed = bool(progression["passed"])
    gates = preservation_gates(
        memory_coverage=float(checkpoint_memory_metrics["coverage"]),
        routed_minus_base=routed_minus_base,
        routed_minus_v9=routed_minus_v9,
        attribution_exact=attribution_exact,
        scene_progression_passed=scene_progression_passed,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "split": "publisher-TRAIN-derived development untouched remainder",
            "narrative_rows": runner.NARRATIVE_ROWS,
            "attribution_rows": len(base_attribution),
            "scene_rows": int(progression["combined_fit"]["rows"]),
            "protected_splits_opened": [],
        },
        "attribution": {
            "decoder": "exact frozen-base artifact reuse",
            "metrics": attribution_metrics,
            "accepted_output_payload_sha256": runner.canonical_sha256(attribution_payload),
            "new_memory_execution": False,
            "preserved_exactly": attribution_exact,
        },
        "narrative": {
            "base": base_metrics,
            "v9_memory": v9_memory_metrics,
            "v9_routed": v9_routed_metrics,
            "checkpoint16_memory": checkpoint_memory_metrics,
            "checkpoint16_routed": checkpoint_routed_metrics,
            "checkpoint16_routed_minus_base": routed_minus_base,
            "checkpoint16_routed_minus_v9_routed": routed_minus_v9,
            "checkpoint16_memory_output_change_fraction_vs_v9": prediction_change_fraction(
                checkpoint, v9_narrative
            ),
            "checkpoint16_routed_output_change_fraction_vs_v9": prediction_change_fraction(
                checkpoint_routed_records, v9_routed_records
            ),
        },
        "scene": {
            "progression_receipt_payload_sha256": runner.PROGRESSION_RECEIPT_SHA256,
            "combined_fit": progression["combined_fit"]["evaluation"],
            "passed": scene_progression_passed,
        },
        "gates": gates,
        "fresh_publisher_validation_replication_contract_authorized": bool(gates["passed"]),
        "publisher_validation_opened": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "input_bindings": bindings,
            "checkpoint16_narrative": checkpoint_artifacts,
            "base_narrative": base_narrative_artifacts,
            "v9_narrative": v9_narrative_artifacts,
            "base_attribution": base_attribution_artifacts,
            "progression_result": {
                "path": str(progression_result_path),
                "sha256": runner.sha256_file(progression_result_path),
                "receipt_payload_sha256": progression["receipt"]["payload_sha256"],
            },
            "analyzer_sha256": runner.sha256_file(Path(__file__)),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_multitask_preservation_result_without_receipt",
        "payload_sha256": runner.canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Multitask preservation output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--progression-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        progression_result_path=args.progression_result.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["gates"]["passed"],
                "narrative_checkpoint16_routed": result["narrative"]["checkpoint16_routed"]["primary_metric"],
                "narrative_minus_base": result["narrative"]["checkpoint16_routed_minus_base"],
                "narrative_minus_v9": result["narrative"]["checkpoint16_routed_minus_v9_routed"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
