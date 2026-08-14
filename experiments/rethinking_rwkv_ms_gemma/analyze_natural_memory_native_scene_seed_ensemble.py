#!/usr/bin/env python3
"""Analyze and sign the locked native-scene seed-ensemble candidate."""

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
    analyze_natural_memory_native_scene_contrast_probe as probe_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_contrast_progression as progression_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_seed_ensemble as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_progression as progression,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble_eval as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_ensemble_result.v1"
PROGRESSION_RESULT_FILE_SHA256 = "af852ce316d83fb90b18bbd97fb302cdb1fe99b305c96736478759143d897cb2"
PROGRESSION_RESULT_RECEIPT_SHA256 = "23bc133d82590890308ac5b0779e54427f51fbee615d941393a023538be80b2b"
GATE_THRESHOLDS = {
    "coverage": 0.95,
    "candidate_minus_checkpoint_16_micro_f1": 0.005,
    "candidate_minus_v9_micro_f1": 0.005,
    "output_change_fraction_vs_checkpoint_16": 0.02,
}


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def read_candidate_outputs(
    root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    materialization_root: Path,
) -> tuple[
    dict[int, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    materialization, manifest = runner.validate_materialization(materialization_root)
    expected = {int(row["source_index"]): row for row in rows}
    outputs: dict[int, Mapping[str, Any]] = {}
    bindings: list[Mapping[str, Any]] = []
    artifacts: list[Mapping[str, Any]] = []
    runtime_hash: str | None = None
    runner_sha256 = sha256_file(Path(runner.__file__))
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        required_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "materialization_result_file_sha256": sha256_file(
                materialization_root / "result.json"
            ),
            "materialization_result_receipt_sha256": materialization["receipt"][
                "payload_sha256"
            ],
            "candidate_id": materializer.CANDIDATE_ID,
            "candidate_gate_state_sha256": manifest["gate_state_sha256"],
            "candidate_manifest_sha256": sha256_file(
                materialization_root / "manifest.json"
            ),
            "candidate_patch_sha256": manifest["patch_file"]["sha256"],
            "row_payload_sha256": runner.ROW_PAYLOAD_SHA256,
            "rows": runner.ROWS,
            "condition": runner.CONDITION,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "runner_sha256": runner_sha256,
            "protected_splits_opened": [],
        }
        if any(binding.get(key) != value for key, value in required_binding.items()):
            raise ValueError(f"Seed-ensemble input binding differs: {binding_path}")
        bindings.append(
            {"path": str(binding_path), "sha256": sha256_file(binding_path), "payload": binding}
        )
        shard_expected = {
            source_index
            for source_index in expected
            if source_index % runner.WORLD_SIZE == shard_index
        }
        path = runner.output_path(shard_dir)
        records = probe_analysis.read_jsonl(path)
        artifacts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(records),
                "shard_index": shard_index,
            }
        )
        seen: set[int] = set()
        for record in records:
            source_index = int(record["source_index"])
            candidate_runtime_hash = str(record.get("runtime_gate_state_sha256"))
            if runtime_hash is None:
                runtime_hash = candidate_runtime_hash
            required = {
                "schema": runner.SCHEMA,
                "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
                "candidate_id": materializer.CANDIDATE_ID,
                "gate_state_sha256": manifest["gate_state_sha256"],
                "runtime_gate_state_sha256": runtime_hash,
                "condition": runner.CONDITION,
                "state_kind": "row_correct",
                "shard_index": shard_index,
                "world_size": runner.WORLD_SIZE,
                "source_index": source_index,
            }
            if (
                source_index not in shard_expected
                or source_index in seen
                or record.get("row_sha256") != expected[source_index]["row_sha256"]
                or any(record.get(key) != value for key, value in required.items())
            ):
                raise ValueError(f"Seed-ensemble output differs: {source_index}")
            seen.add(source_index)
            outputs[source_index] = record
        if seen != shard_expected:
            raise ValueError(f"Incomplete seed-ensemble shard: {shard_index}")
    if set(outputs) != set(expected):
        raise ValueError("Incomplete seed-ensemble outputs")
    return outputs, materialization, manifest, bindings, artifacts


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
        raise ValueError("Seed-ensemble checkpoint 16 baseline binding differs")
    return result


def analyze(
    *,
    input_root: Path,
    materialization_root: Path,
    progression_root: Path,
    dataset_root: Path,
    reference_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    training.validate_protocol()
    rows = causal.load_rows(dataset_root)
    evaluation_rows = progression.progression_rows(rows)
    indices = tuple(int(row["source_index"]) for row in evaluation_rows)
    all_gold, hashes = shared.gold_and_hashes(rows)
    candidate_outputs, materialization, manifest, bindings, artifacts = (
        read_candidate_outputs(
            input_root,
            rows=evaluation_rows,
            materialization_root=materialization_root,
        )
    )
    candidate_predictions = shared.predictions_from_records(candidate_outputs)
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
    v9_records, v9_artifacts = shared.read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    v9_predictions = shared.predictions_from_records(v9_records)
    metrics = {
        "seed_delta_mean": shared.metrics_from_sets(
            candidate_predictions,
            all_gold,
            indices,
        ),
        "checkpoint_16": shared.metrics_from_sets(
            checkpoint_predictions,
            all_gold,
            indices,
        ),
        "v9": shared.metrics_from_sets(v9_predictions, all_gold, indices),
    }
    deltas = {
        "candidate_minus_checkpoint_16_micro_f1": float(
            metrics["seed_delta_mean"]["micro_f1"]
        )
        - float(metrics["checkpoint_16"]["micro_f1"]),
        "candidate_minus_v9_micro_f1": float(metrics["seed_delta_mean"]["micro_f1"])
        - float(metrics["v9"]["micro_f1"]),
        "output_change_fraction_vs_checkpoint_16": shared.output_change_fraction(
            candidate_predictions,
            checkpoint_predictions,
            indices,
        ),
    }
    gates = {
        "coverage_at_least_0.95": float(metrics["seed_delta_mean"]["coverage"])
        >= GATE_THRESHOLDS["coverage"],
        "candidate_minus_checkpoint_16_micro_f1_at_least_0.005": deltas[
            "candidate_minus_checkpoint_16_micro_f1"
        ]
        >= GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"],
        "candidate_minus_v9_micro_f1_at_least_0.005": deltas[
            "candidate_minus_v9_micro_f1"
        ]
        >= GATE_THRESHOLDS["candidate_minus_v9_micro_f1"],
        "output_change_fraction_vs_checkpoint_16_at_least_0.02": deltas[
            "output_change_fraction_vs_checkpoint_16"
        ]
        >= GATE_THRESHOLDS["output_change_fraction_vs_checkpoint_16"],
    }
    gates["passed"] = all(gates.values())
    passed = bool(gates["passed"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if passed else "failed",
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "candidate_id": materializer.CANDIDATE_ID,
        "candidate_gate_state_sha256": manifest["gate_state_sha256"],
        "rows": len(indices),
        "row_payload_sha256": runner.ROW_PAYLOAD_SHA256,
        "metrics": metrics,
        "deltas": deltas,
        "gate_thresholds": GATE_THRESHOLDS,
        "gates": gates,
        "materialization": {
            "path": str(materialization_root / "result.json"),
            "sha256": sha256_file(materialization_root / "result.json"),
            "receipt_sha256": materialization["receipt"]["payload_sha256"],
        },
        "candidate_input_bindings": bindings,
        "candidate_generation_artifacts": artifacts,
        "checkpoint_16_baseline": {
            "path": str(progression_root / "result.json"),
            "sha256": sha256_file(progression_root / "result.json"),
            "receipt_sha256": progression_result["receipt"]["payload_sha256"],
            "input_bindings": progression_bindings,
            "generation_artifacts": progression_artifacts,
        },
        "v9_reference_artifacts": v9_artifacts,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "decision": (
            "TRAIN-only seed ensemble cleared every preregistered gate; external replication still requires a separate protocol."
            if passed
            else "TRAIN-only seed ensemble failed at least one preregistered gate; archive without external replication."
        ),
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Seed-ensemble analysis output must be fresh: {output}")
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
                "status": result["status"],
                "metrics": result["metrics"],
                "deltas": result["deltas"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
