#!/usr/bin/env python3
"""Analyze and sign the locked native scene contrast-dropout probe."""

from __future__ import annotations

import argparse
from datetime import datetime
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
    analyze_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_probe_result.v1"
SELECTION_SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_probe_selection.v1"
GATE_THRESHOLDS = {
    "coverage": 0.95,
    "correct_minus_v9_micro_f1": 0.005,
    "correct_minus_donor_micro_f1": 0.005,
    "correct_minus_zero_micro_f1": 0.02,
    "paired_output_change_fraction_vs_v9": 0.05,
}


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"Scene contrast probe record must be an object: {path}")
            records.append(value)
    return records


def read_candidate_outputs(
    root: Path,
    *,
    selected_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[int, dict[str, dict[int, Mapping[str, Any]]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    expected = {int(row["source_index"]): row for row in selected_rows}
    runner_sha256 = sha256_file(Path(runner.__file__))
    outputs = {
        step: {condition: {} for condition in runner.CONDITIONS}
        for step in runner.CANDIDATE_STEPS
    }
    bindings: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    gate_hash_by_step: dict[int, str] = {}
    runtime_gate_hash_by_step: dict[int, str] = {}
    donor_mapping_sha256: str | None = None
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        required_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "training_result_receipt_sha256": runner.TRAINING_RESULT_RECEIPT_SHA256,
            "selection_payload_sha256": runner.PROBE_PAYLOAD_SHA256,
            "selection_rows": runner.PROBE_ROWS,
            "candidate_steps": list(runner.CANDIDATE_STEPS),
            "conditions": list(runner.CONDITIONS),
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "runner_sha256": runner_sha256,
            "protected_splits_opened": [],
        }
        if any(binding.get(key) != value for key, value in required_binding.items()):
            raise ValueError(f"Scene contrast probe input binding differs: {binding_path}")
        current_donor_sha256 = str(binding.get("donor_mapping_payload_sha256"))
        if donor_mapping_sha256 is None:
            donor_mapping_sha256 = current_donor_sha256
        elif current_donor_sha256 != donor_mapping_sha256:
            raise ValueError("Scene contrast probe donor binding differs across shards")
        bindings.append(
            {"path": str(binding_path), "sha256": sha256_file(binding_path), "payload": binding}
        )
        shard_expected = {
            source_index
            for source_index in expected
            if source_index % runner.WORLD_SIZE == shard_index
        }
        for step in runner.CANDIDATE_STEPS:
            checkpoint_artifacts = binding.get("checkpoint_artifacts")
            if not isinstance(checkpoint_artifacts, list):
                raise ValueError("Scene contrast probe checkpoint binding is missing")
            bound_checkpoint = next(
                (item for item in checkpoint_artifacts if item.get("step") == step),
                None,
            )
            if not isinstance(bound_checkpoint, Mapping):
                raise ValueError(f"Scene contrast probe checkpoint {step} is unbound")
            expected_gate_hash = str(bound_checkpoint.get("gate_state_sha256"))
            prior_gate_hash = gate_hash_by_step.setdefault(step, expected_gate_hash)
            if expected_gate_hash != prior_gate_hash:
                raise ValueError(f"Scene contrast probe checkpoint {step} differs across shards")
            for condition in runner.CONDITIONS:
                path = runner.output_path(shard_dir, step, condition)
                records = read_jsonl(path)
                artifacts.append(
                    {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "rows": len(records),
                        "checkpoint_step": step,
                        "condition": condition,
                        "shard_index": shard_index,
                    }
                )
                seen: set[int] = set()
                for record in records:
                    source_index = int(record["source_index"])
                    runtime_gate_hash = str(record.get("runtime_gate_state_sha256"))
                    prior_runtime_hash = runtime_gate_hash_by_step.setdefault(
                        step, runtime_gate_hash
                    )
                    required = {
                        "schema": runner.SCHEMA,
                        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
                        "training_result_receipt_sha256": runner.TRAINING_RESULT_RECEIPT_SHA256,
                        "checkpoint_step": step,
                        "gate_state_sha256": expected_gate_hash,
                        "condition": condition,
                        "shard_index": shard_index,
                        "world_size": runner.WORLD_SIZE,
                        "source_index": source_index,
                    }
                    if (
                        source_index not in shard_expected
                        or source_index in seen
                        or runtime_gate_hash != prior_runtime_hash
                        or record.get("row_sha256") != expected[source_index]["row_sha256"]
                        or any(record.get(key) != value for key, value in required.items())
                    ):
                        raise ValueError(
                            f"Scene contrast probe output differs: {step}:{condition}:{source_index}"
                        )
                    if condition == "matched_donor_state" and (
                        record.get("donor_source_index") == source_index
                        or not isinstance(record.get("absolute_write_token_delta"), int)
                    ):
                        raise ValueError(f"Scene contrast donor metadata differs: {source_index}")
                    seen.add(source_index)
                    outputs[step][condition][source_index] = record
                if seen != shard_expected:
                    raise ValueError(
                        f"Incomplete scene contrast probe shard: {step}:{condition}:{shard_index}"
                    )
    for step in runner.CANDIDATE_STEPS:
        for condition in runner.CONDITIONS:
            if set(outputs[step][condition]) != set(expected):
                raise ValueError(f"Incomplete scene contrast probe: {step}:{condition}")
    return outputs, bindings, artifacts


def candidate_result(
    *,
    step: int,
    predictions: Mapping[str, Mapping[int, set[int] | None]],
    v9_predictions: Mapping[int, set[int] | None],
    gold: Mapping[int, set[int]],
    indices: Sequence[int],
) -> Mapping[str, Any]:
    metrics = {
        condition: shared.metrics_from_sets(condition_predictions, gold, indices)
        for condition, condition_predictions in predictions.items()
    }
    v9_metrics = shared.metrics_from_sets(v9_predictions, gold, indices)
    correct_f1 = float(metrics["correct_state"]["micro_f1"])
    donor_f1 = float(metrics["matched_donor_state"]["micro_f1"])
    zero_f1 = float(metrics["zero_state"]["micro_f1"])
    v9_f1 = float(v9_metrics["micro_f1"])
    change_fraction = shared.output_change_fraction(
        predictions["correct_state"],
        v9_predictions,
        indices,
    )
    deltas = {
        "correct_minus_v9_micro_f1": correct_f1 - v9_f1,
        "correct_minus_matched_donor_micro_f1": correct_f1 - donor_f1,
        "correct_minus_zero_micro_f1": correct_f1 - zero_f1,
        "paired_output_change_fraction_vs_v9": change_fraction,
    }
    gates: dict[str, bool] = {
        "coverage_at_least_0.95": float(metrics["correct_state"]["coverage"]) >= 0.95,
        "correct_minus_v9_micro_f1_at_least_0.005": deltas["correct_minus_v9_micro_f1"] >= 0.005,
        "correct_minus_matched_donor_micro_f1_at_least_0.005": deltas[
            "correct_minus_matched_donor_micro_f1"
        ]
        >= 0.005,
        "correct_minus_zero_micro_f1_at_least_0.02": deltas[
            "correct_minus_zero_micro_f1"
        ]
        >= 0.02,
        "paired_output_change_fraction_vs_v9_at_least_0.05": change_fraction >= 0.05,
    }
    gates["passed"] = all(gates.values())
    return {
        "checkpoint_step": step,
        "metrics": metrics,
        "frozen_v9_metrics": v9_metrics,
        "deltas": deltas,
        "gates": gates,
        "passed": bool(gates["passed"]),
    }


def ranking_key(value: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        -float(value["metrics"]["correct_state"]["micro_f1"]),
        -float(value["deltas"]["correct_minus_matched_donor_micro_f1"]),
        -float(value["deltas"]["correct_minus_zero_micro_f1"]),
        int(value["checkpoint_step"]),
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Scene contrast probe output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analyze(
    *,
    input_root: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
    output: Path,
    selection_output: Path,
) -> Mapping[str, Any]:
    training_manifests = runner.validate_training_root(training_root)
    rows = causal.load_rows(dataset_root)
    selected_rows = runner.selected_probe_rows(rows)
    indices = tuple(int(row["source_index"]) for row in selected_rows)
    all_gold, hashes = shared.gold_and_hashes(rows)
    gold = {source_index: all_gold[source_index] for source_index in indices}
    outputs, bindings, artifacts = read_candidate_outputs(
        input_root,
        selected_rows=selected_rows,
    )
    v9_records, v9_artifacts = shared.read_reference_condition(reference_root, "memory", hashes)
    v9_predictions = shared.predictions_from_records(v9_records)
    candidates: list[Mapping[str, Any]] = []
    for step in runner.CANDIDATE_STEPS:
        predictions = {
            condition: shared.predictions_from_records(outputs[step][condition])
            for condition in runner.CONDITIONS
        }
        candidates.append(
            candidate_result(
                step=step,
                predictions=predictions,
                v9_predictions=v9_predictions,
                gold=gold,
                indices=indices,
            )
        )
    ranked = sorted(candidates, key=ranking_key)
    passing = [candidate for candidate in ranked if candidate["passed"]]
    selected = passing[0] if passing else None
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "training_result_receipt_sha256": runner.TRAINING_RESULT_RECEIPT_SHA256,
        "runner_sha256": sha256_file(Path(runner.__file__)),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase": "selection_probe",
        "selection_payload_sha256": runner.PROBE_PAYLOAD_SHA256,
        "source_indices": list(indices),
        "rows": len(indices),
        "gate_thresholds": GATE_THRESHOLDS,
        "candidates": candidates,
        "ranking": [int(candidate["checkpoint_step"]) for candidate in ranked],
        "selected_checkpoint_step": None if selected is None else int(selected["checkpoint_step"]),
        "passed": selected is not None,
        "remaining_fit_evaluation_authorized": selected is not None,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "input_bindings": bindings,
            "candidate_outputs": artifacts,
            "frozen_v9_outputs": v9_artifacts,
            "training_result": {
                "path": str(training_root / "result.json"),
                "sha256": sha256_file(training_root / "result.json"),
                "receipt_payload_sha256": runner.TRAINING_RESULT_RECEIPT_SHA256,
            },
            "training_checkpoints": [
                {
                    "step": int(manifest["step"]),
                    "gate_state_sha256": str(manifest["gate_state_sha256"]),
                    "patch_sha256": str(manifest["patch_file"]["sha256"]),
                }
                for manifest in training_manifests
            ],
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_contrast_probe_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output, result)
    if selected is not None:
        step = int(selected["checkpoint_step"])
        manifest = next(item for item in training_manifests if int(item["step"]) == step)
        selection: dict[str, Any] = {
            "schema": SELECTION_SCHEMA,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "runner_sha256": sha256_file(Path(runner.__file__)),
            "analyzer_sha256": sha256_file(Path(__file__)),
            "result": {
                "path": str(output),
                "file_sha256": sha256_file(output),
                "receipt_payload_sha256": result["receipt"]["payload_sha256"],
            },
            "selected_checkpoint_step": step,
            "selected_gate_state_sha256": manifest["gate_state_sha256"],
            "selected_patch_sha256": manifest["patch_file"]["sha256"],
            "selected_metrics": selected,
            "passed": True,
            "remaining_fit_evaluation_authorized": True,
            "publisher_validation_authorized": False,
            "publisher_test_authorized": False,
            "hard32_authorized": False,
            "unused_strength_holdout_authorized": False,
            "protected_splits_opened": [],
        }
        selection["receipt"] = {
            "algorithm": "sha256",
            "payload_scope": "canonical_contrast_probe_selection_without_receipt",
            "payload_sha256": canonical_sha256(selection),
        }
        write_json(selection_output, selection)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        training_root=args.training_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
        selection_output=args.selection_output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "ranking": result["ranking"],
                "selected_checkpoint_step": result["selected_checkpoint_step"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
