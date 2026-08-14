#!/usr/bin/env python3
"""Analyze and sign the locked checkpoint-16 residual candidate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_seed_ensemble as shared_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_c16_residual as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_c16_residual as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_c16_residual_eval as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_c16_residual_result.v1"
GATE_THRESHOLDS = dict(shared_analysis.GATE_THRESHOLDS)


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


@contextmanager
def _configured_analysis() -> Iterator[None]:
    replacements = {
        "training": training,
        "materializer": materializer,
        "runner": runner,
    }
    original = {name: getattr(shared_analysis, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(shared_analysis, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(shared_analysis, name, value)


def read_candidate_outputs(*args: Any, **kwargs: Any) -> Any:
    with _configured_analysis():
        return shared_analysis.read_candidate_outputs(*args, **kwargs)


def validate_progression_result(root: Path) -> Mapping[str, Any]:
    return shared_analysis.validate_progression_result(root)


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
    rows = shared_analysis.causal.load_rows(dataset_root)
    evaluation_rows = shared_analysis.progression.progression_rows(rows)
    indices = tuple(int(row["source_index"]) for row in evaluation_rows)
    all_gold, hashes = shared_analysis.shared.gold_and_hashes(rows)
    candidate_outputs, materialization, manifest, bindings, artifacts = (
        read_candidate_outputs(
            input_root,
            rows=evaluation_rows,
            materialization_root=materialization_root,
        )
    )
    candidate_predictions = shared_analysis.shared.predictions_from_records(
        candidate_outputs
    )
    progression_result = validate_progression_result(progression_root)
    progression_outputs, progression_bindings, progression_artifacts = (
        shared_analysis.progression_analysis.read_progression_outputs(
            progression_root,
            remaining_rows=evaluation_rows,
        )
    )
    checkpoint_predictions = shared_analysis.shared.predictions_from_records(
        progression_outputs[runner.CONDITION]
    )
    v9_records, v9_artifacts = shared_analysis.shared.read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    v9_predictions = shared_analysis.shared.predictions_from_records(v9_records)
    metrics = {
        materializer.CANDIDATE_ID: shared_analysis.shared.metrics_from_sets(
            candidate_predictions,
            all_gold,
            indices,
        ),
        "checkpoint_16": shared_analysis.shared.metrics_from_sets(
            checkpoint_predictions,
            all_gold,
            indices,
        ),
        "v9": shared_analysis.shared.metrics_from_sets(
            v9_predictions,
            all_gold,
            indices,
        ),
    }
    candidate_metrics = metrics[materializer.CANDIDATE_ID]
    deltas = {
        "candidate_minus_checkpoint_16_micro_f1": float(
            candidate_metrics["micro_f1"]
        )
        - float(metrics["checkpoint_16"]["micro_f1"]),
        "candidate_minus_v9_micro_f1": float(candidate_metrics["micro_f1"])
        - float(metrics["v9"]["micro_f1"]),
        "output_change_fraction_vs_checkpoint_16": (
            shared_analysis.shared.output_change_fraction(
                candidate_predictions,
                checkpoint_predictions,
                indices,
            )
        ),
    }
    gates = {
        "coverage_at_least_0.95": float(candidate_metrics["coverage"])
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
            "TRAIN-only checkpoint-16 residual mean cleared every preregistered gate; external replication still requires a separate protocol."
            if passed
            else "TRAIN-only checkpoint-16 residual mean failed at least one preregistered gate; archive without external replication."
        ),
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"C16-residual analysis output must be fresh: {output}")
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
