#!/usr/bin/env python3
"""Cross-fit and sign the locked native scene repair-bridge study."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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
    analyze_natural_memory_native_scene_contrast_probe as probe_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_contrast_progression as progression_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_repair_bridges as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_repair_bridge as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_progression as progression,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_repair_bridge_result.v1"
FOLD_SALT = "rwkv-ms-native-scene-repair-bridge-v1:"
FOLDS = 5
FOLD_ASSIGNMENT_PAYLOAD_SHA256 = "fc949e9492b9864d6f5bcac55aaa71751ba3beb492015ca10d81d846305828e4"
FOLD_COUNTS = {0: 43, 1: 39, 2: 49, 3: 45, 4: 44}
CHECKPOINT_ID = "checkpoint_16"
PROGRESSION_RESULT_FILE_SHA256 = "af852ce316d83fb90b18bbd97fb302cdb1fe99b305c96736478759143d897cb2"
PROGRESSION_RESULT_RECEIPT_SHA256 = "23bc133d82590890308ac5b0779e54427f51fbee615d941393a023538be80b2b"
GATE_THRESHOLDS = {
    "coverage": 0.95,
    "oof_minus_checkpoint_16_micro_f1": 0.005,
    "oof_minus_v9_micro_f1": 0.005,
    "oof_output_change_fraction_vs_checkpoint_16": 0.02,
    "learned_bridge_selected_folds": 4,
    "modal_learned_bridge_selected_folds": 3,
    "modal_bridge_minus_checkpoint_16_micro_f1": 0.005,
}


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def fold_for_row(row: Mapping[str, Any]) -> int:
    digest = hashlib.sha256((FOLD_SALT + str(row["row_sha256"])).encode("ascii")).hexdigest()
    return int(digest, 16) % FOLDS


def validate_folds(rows: Sequence[Mapping[str, Any]]) -> Mapping[int, tuple[int, ...]]:
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
            "fold": fold_for_row(row),
        }
        for row in rows
    ]
    if canonical_sha256(payload) != FOLD_ASSIGNMENT_PAYLOAD_SHA256:
        raise ValueError("Repair-bridge fold assignment differs")
    indices = {
        fold: tuple(
            int(row["source_index"])
            for row in rows
            if fold_for_row(row) == fold
        )
        for fold in range(FOLDS)
    }
    if {fold: len(value) for fold, value in indices.items()} != FOLD_COUNTS:
        raise ValueError("Repair-bridge fold counts differ")
    return indices


def read_bridge_outputs(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    materialization_root: Path,
) -> tuple[
    dict[str, dict[int, Mapping[str, Any]]],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    expected = {int(row["source_index"]): row for row in rows}
    materialization, manifests = runner.validate_materialization(materialization_root)
    manifest_by_id = {str(item["candidate_id"]): item for item in manifests}
    outputs = {candidate_id: {} for candidate_id in manifest_by_id}
    bindings: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    runtime_hashes: dict[str, str] = {}
    runner_sha256 = sha256_file(Path(runner.__file__))
    materialization_file_sha256 = sha256_file(materialization_root / "result.json")
    materialization_receipt = materialization["receipt"]["payload_sha256"]
    expected_artifacts = [
        {
            "candidate_id": manifest["candidate_id"],
            "gate_state_sha256": manifest["gate_state_sha256"],
            "manifest_sha256": sha256_file(
                materialization_root / manifest["candidate_id"] / "manifest.json"
            ),
            "patch_sha256": manifest["patch_file"]["sha256"],
        }
        for manifest in manifests
    ]
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        required_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "materialization_result_file_sha256": materialization_file_sha256,
            "materialization_result_receipt_sha256": materialization_receipt,
            "candidate_artifacts": expected_artifacts,
            "row_payload_sha256": runner.ROW_PAYLOAD_SHA256,
            "rows": runner.ROWS,
            "condition": runner.CONDITION,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "runner_sha256": runner_sha256,
            "protected_splits_opened": [],
        }
        if any(binding.get(key) != value for key, value in required_binding.items()):
            raise ValueError(f"Repair-bridge input binding differs: {binding_path}")
        bindings.append(
            {"path": str(binding_path), "sha256": sha256_file(binding_path), "payload": binding}
        )
        shard_expected = {
            source_index
            for source_index in expected
            if source_index % runner.WORLD_SIZE == shard_index
        }
        for candidate_id, manifest in manifest_by_id.items():
            path = runner.output_path(shard_dir, candidate_id)
            records = probe_analysis.read_jsonl(path)
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(records),
                    "candidate_id": candidate_id,
                    "shard_index": shard_index,
                }
            )
            seen: set[int] = set()
            for record in records:
                source_index = int(record["source_index"])
                runtime_sha256 = str(record.get("runtime_gate_state_sha256"))
                prior_runtime_sha256 = runtime_hashes.setdefault(candidate_id, runtime_sha256)
                required = {
                    "schema": runner.SCHEMA,
                    "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                    "candidate_id": candidate_id,
                    "gate_state_sha256": manifest["gate_state_sha256"],
                    "condition": runner.CONDITION,
                    "state_kind": "row_correct",
                    "shard_index": shard_index,
                    "world_size": runner.WORLD_SIZE,
                    "source_index": source_index,
                }
                if (
                    source_index not in shard_expected
                    or source_index in seen
                    or runtime_sha256 != prior_runtime_sha256
                    or record.get("row_sha256") != expected[source_index]["row_sha256"]
                    or any(record.get(key) != value for key, value in required.items())
                ):
                    raise ValueError(f"Repair-bridge output differs: {candidate_id}:{source_index}")
                seen.add(source_index)
                outputs[candidate_id][source_index] = record
            if seen != shard_expected:
                raise ValueError(f"Incomplete repair-bridge shard: {candidate_id}:{shard_index}")
    for candidate_id, records in outputs.items():
        if set(records) != set(expected):
            raise ValueError(f"Incomplete repair-bridge candidate: {candidate_id}")
    return outputs, materialization, manifests, bindings, artifacts


def validate_progression_result(root: Path) -> Mapping[str, Any]:
    path = root / "result.json"
    result = probe.validate_signed_json(path, description="Checkpoint 16 progression result")
    if (
        sha256_file(path) != PROGRESSION_RESULT_FILE_SHA256
        or result["receipt"].get("payload_sha256") != PROGRESSION_RESULT_RECEIPT_SHA256
        or result.get("schema") != progression_analysis.SCHEMA
        or result.get("selected_checkpoint_step") != progression.SELECTED_STEP
        or result.get("passed") is not True
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Repair-bridge progression result binding differs")
    return result


def metric_rank(candidate_id: str, metric: Mapping[str, Any]) -> tuple[float, float, float, str]:
    return (
        -float(metric["micro_f1"]),
        -float(metric["precision"]),
        -float(metric["recall"]),
        candidate_id,
    )


def analyze(
    *,
    input_root: Path,
    materialization_root: Path,
    progression_root: Path,
    dataset_root: Path,
    reference_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    materializer.validate_protocol()
    rows = causal.load_rows(dataset_root)
    evaluation_rows = progression.progression_rows(rows)
    indices = tuple(int(row["source_index"]) for row in evaluation_rows)
    fold_indices = validate_folds(evaluation_rows)
    all_gold, hashes = shared.gold_and_hashes(rows)
    bridge_outputs, materialization, manifests, bindings, artifacts = read_bridge_outputs(
        input_root,
        rows=evaluation_rows,
        materialization_root=materialization_root,
    )
    bridge_predictions = {
        candidate_id: shared.predictions_from_records(records)
        for candidate_id, records in bridge_outputs.items()
    }
    progression_result = validate_progression_result(progression_root)
    progression_outputs, progression_bindings, progression_artifacts = (
        progression_analysis.read_progression_outputs(
            progression_root,
            remaining_rows=evaluation_rows,
        )
    )
    checkpoint_predictions = shared.predictions_from_records(
        progression_outputs[runner.CONDITION]
    )
    v9_records, v9_artifacts = shared.read_reference_condition(reference_root, "memory", hashes)
    v9_predictions = shared.predictions_from_records(v9_records)
    predictions = {**bridge_predictions, CHECKPOINT_ID: checkpoint_predictions}
    aggregate_metrics = {
        candidate_id: shared.metrics_from_sets(candidate_predictions, all_gold, indices)
        for candidate_id, candidate_predictions in predictions.items()
    }
    v9_metrics = shared.metrics_from_sets(v9_predictions, all_gold, indices)
    fold_results: list[dict[str, Any]] = []
    oof_predictions: dict[int, set[int] | None] = {}
    all_indices = set(indices)
    for fold in range(FOLDS):
        heldout = fold_indices[fold]
        training_indices = tuple(sorted(all_indices - set(heldout)))
        training_metrics = {
            candidate_id: shared.metrics_from_sets(
                candidate_predictions,
                all_gold,
                training_indices,
            )
            for candidate_id, candidate_predictions in predictions.items()
        }
        selected_id = min(
            training_metrics,
            key=lambda candidate_id: metric_rank(candidate_id, training_metrics[candidate_id]),
        )
        heldout_metrics = shared.metrics_from_sets(
            predictions[selected_id],
            all_gold,
            heldout,
        )
        checkpoint_heldout_metrics = shared.metrics_from_sets(
            checkpoint_predictions,
            all_gold,
            heldout,
        )
        for source_index in heldout:
            oof_predictions[source_index] = predictions[selected_id][source_index]
        fold_results.append(
            {
                "fold": fold,
                "fit_rows": len(training_indices),
                "heldout_rows": len(heldout),
                "selected_candidate_id": selected_id,
                "fit_ranking": sorted(
                    training_metrics,
                    key=lambda candidate_id: metric_rank(
                        candidate_id,
                        training_metrics[candidate_id],
                    ),
                ),
                "selected_fit_metrics": training_metrics[selected_id],
                "selected_heldout_metrics": heldout_metrics,
                "checkpoint_16_heldout_metrics": checkpoint_heldout_metrics,
            }
        )
    if set(oof_predictions) != set(indices):
        raise ValueError("Repair-bridge out-of-fold coverage differs")
    oof_metrics = shared.metrics_from_sets(oof_predictions, all_gold, indices)
    checkpoint_metrics = aggregate_metrics[CHECKPOINT_ID]
    selected_ids = [result["selected_candidate_id"] for result in fold_results]
    learned_selected = [candidate_id for candidate_id in selected_ids if candidate_id != CHECKPOINT_ID]
    learned_counts = Counter(learned_selected)
    modal_candidate_id = (
        None
        if not learned_counts
        else min(learned_counts, key=lambda candidate_id: (-learned_counts[candidate_id], candidate_id))
    )
    modal_count = 0 if modal_candidate_id is None else learned_counts[modal_candidate_id]
    modal_metrics = None if modal_candidate_id is None else aggregate_metrics[modal_candidate_id]
    deltas = {
        "oof_minus_checkpoint_16_micro_f1": float(oof_metrics["micro_f1"])
        - float(checkpoint_metrics["micro_f1"]),
        "oof_minus_v9_micro_f1": float(oof_metrics["micro_f1"])
        - float(v9_metrics["micro_f1"]),
        "oof_output_change_fraction_vs_checkpoint_16": shared.output_change_fraction(
            oof_predictions,
            checkpoint_predictions,
            indices,
        ),
        "modal_bridge_minus_checkpoint_16_micro_f1": (
            None
            if modal_metrics is None
            else float(modal_metrics["micro_f1"]) - float(checkpoint_metrics["micro_f1"])
        ),
    }
    gates = {
        "coverage_at_least_0.95": float(oof_metrics["coverage"])
        >= GATE_THRESHOLDS["coverage"],
        "oof_minus_checkpoint_16_micro_f1_at_least_0.005": deltas[
            "oof_minus_checkpoint_16_micro_f1"
        ]
        >= GATE_THRESHOLDS["oof_minus_checkpoint_16_micro_f1"],
        "oof_minus_v9_micro_f1_at_least_0.005": deltas["oof_minus_v9_micro_f1"]
        >= GATE_THRESHOLDS["oof_minus_v9_micro_f1"],
        "oof_output_change_fraction_vs_checkpoint_16_at_least_0.02": deltas[
            "oof_output_change_fraction_vs_checkpoint_16"
        ]
        >= GATE_THRESHOLDS["oof_output_change_fraction_vs_checkpoint_16"],
        "learned_bridge_selected_in_at_least_4_folds": len(learned_selected)
        >= GATE_THRESHOLDS["learned_bridge_selected_folds"],
        "modal_learned_bridge_selected_in_at_least_3_folds": modal_count
        >= GATE_THRESHOLDS["modal_learned_bridge_selected_folds"],
        "modal_bridge_minus_checkpoint_16_micro_f1_at_least_0.005": (
            deltas["modal_bridge_minus_checkpoint_16_micro_f1"] is not None
            and deltas["modal_bridge_minus_checkpoint_16_micro_f1"]
            >= GATE_THRESHOLDS["modal_bridge_minus_checkpoint_16_micro_f1"]
        ),
    }
    gates["passed"] = all(gates.values())
    passed = bool(gates["passed"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "materialization_result_receipt_sha256": materialization["receipt"]["payload_sha256"],
        "runner_sha256": sha256_file(Path(runner.__file__)),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "rows": len(indices),
        "source_indices": list(indices),
        "fold_assignment_payload_sha256": FOLD_ASSIGNMENT_PAYLOAD_SHA256,
        "fold_counts": FOLD_COUNTS,
        "candidate_ids": list(bridge_predictions),
        "aggregate_metrics": aggregate_metrics,
        "v9_metrics": v9_metrics,
        "fold_results": fold_results,
        "fold_selected_candidate_ids": selected_ids,
        "learned_bridge_selected_folds": len(learned_selected),
        "learned_bridge_selection_counts": dict(sorted(learned_counts.items())),
        "modal_candidate_id": modal_candidate_id,
        "modal_candidate_metrics": modal_metrics,
        "oof_metrics": oof_metrics,
        "deltas": deltas,
        "gate_thresholds": GATE_THRESHOLDS,
        "gates": gates,
        "passed": passed,
        "selected_candidate_id": modal_candidate_id if passed else None,
        "study_scope": "adaptive_development_only",
        "external_replication_authorized": False,
        "multitask_preservation_authorized": False,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "input_bindings": bindings,
            "candidate_outputs": artifacts,
            "materialization_manifests": [
                {
                    "candidate_id": manifest["candidate_id"],
                    "gate_state_sha256": manifest["gate_state_sha256"],
                    "patch_sha256": manifest["patch_file"]["sha256"],
                }
                for manifest in manifests
            ],
            "checkpoint_16_input_bindings": progression_bindings,
            "checkpoint_16_outputs": progression_artifacts,
            "checkpoint_16_result": {
                "path": str(progression_root / "result.json"),
                "sha256": sha256_file(progression_root / "result.json"),
                "receipt_payload_sha256": progression_result["receipt"]["payload_sha256"],
            },
            "frozen_v9_outputs": v9_artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_repair_bridge_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Repair-bridge result must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--progression-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
        materialization_root=args.materialization_root.expanduser().resolve(strict=True),
        progression_root=args.progression_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "selected_candidate_id": result["selected_candidate_id"],
                "oof_micro_f1": result["oof_metrics"]["micro_f1"],
                "checkpoint_16_micro_f1": result["aggregate_metrics"][CHECKPOINT_ID][
                    "micro_f1"
                ],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
