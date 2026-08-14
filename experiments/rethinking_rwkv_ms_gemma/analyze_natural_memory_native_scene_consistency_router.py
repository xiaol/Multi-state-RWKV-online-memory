#!/usr/bin/env python3
"""Cross-fit and sign the adaptive native scene consistency router."""

from __future__ import annotations

import argparse
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
    analyze_natural_memory_native_scene_contrast_progression as progression_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_repair_bridge as bridge_analysis,
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
    run_natural_memory_native_scene_contrast_progression as progression,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_repair_bridge as bridge_runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_consistency_router_result.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_consistency_router_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "9601af58d9bcebf908fcbcae6bb79e65b207fec238f45ee553377f5f18a52056"
FOLD_SALT = "rwkv-ms-native-scene-consistency-router-v1:"
FOLDS = 5
FOLD_ASSIGNMENT_PAYLOAD_SHA256 = "accfb48afb38fa2eff4504dff52845e059cd2c4e6538cf6993304757e7297999"
FOLD_COUNTS = {0: 38, 1: 42, 2: 44, 3: 45, 4: 51}
CHECKPOINT_ID = "checkpoint_16"
PROPOSAL_ID = "onpolicy75_dualpath25"
POLICY_ORDER = (
    "checkpoint",
    "strict_subset",
    "abstention_singleton",
    "combined",
)
GATE_THRESHOLDS = {
    "coverage": 0.95,
    "oof_minus_checkpoint_16_micro_f1": 0.005,
    "oof_minus_v9_micro_f1": 0.005,
    "oof_output_change_fraction_vs_checkpoint_16": 0.02,
    "combined_selected_folds": 4,
    "combined_tp_gain": 1,
    "combined_fp_delta_maximum": 0,
}


def canonical_sha256(value: Any) -> str:
    return bridge_analysis.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return bridge_analysis.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Consistency-router protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Consistency-router protocol hash differs")
    policy_ids = [policy.get("policy_id") for policy in value.get("router", {}).get("policies", [])]
    if policy_ids != list(POLICY_ORDER):
        raise ValueError("Consistency-router policy order differs")
    return value


def proposal_reason(
    checkpoint: set[int] | None,
    proposal: set[int] | None,
) -> str | None:
    if checkpoint is None or proposal is None:
        return None
    if proposal < checkpoint:
        return "strict_subset"
    if not checkpoint and len(proposal) == 1:
        return "abstention_singleton"
    return None


def route_prediction(
    checkpoint: set[int] | None,
    proposal: set[int] | None,
    *,
    policy_id: str,
) -> set[int] | None:
    if policy_id not in POLICY_ORDER:
        raise ValueError(f"Unknown consistency-router policy: {policy_id}")
    reason = proposal_reason(checkpoint, proposal)
    if policy_id == "checkpoint":
        return checkpoint
    if policy_id == "strict_subset" and reason == "strict_subset":
        return proposal
    if policy_id == "abstention_singleton" and reason == "abstention_singleton":
        return proposal
    if policy_id == "combined" and reason is not None:
        return proposal
    return checkpoint


def policy_predictions(
    checkpoint_predictions: Mapping[int, set[int] | None],
    proposal_predictions: Mapping[int, set[int] | None],
) -> dict[str, dict[int, set[int] | None]]:
    if set(checkpoint_predictions) != set(proposal_predictions):
        raise ValueError("Consistency-router prediction rows differ")
    return {
        policy_id: {
            source_index: route_prediction(
                checkpoint_predictions[source_index],
                proposal_predictions[source_index],
                policy_id=policy_id,
            )
            for source_index in checkpoint_predictions
        }
        for policy_id in POLICY_ORDER
    }


def fold_for_row(row: Mapping[str, Any]) -> int:
    digest = hashlib.sha256((FOLD_SALT + str(row["row_sha256"])).encode("ascii")).hexdigest()
    return int(digest, 16) % FOLDS


def validate_folds(rows: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, ...]]:
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
            "fold": fold_for_row(row),
        }
        for row in rows
    ]
    if canonical_sha256(payload) != FOLD_ASSIGNMENT_PAYLOAD_SHA256:
        raise ValueError("Consistency-router fold assignment differs")
    indices = {
        fold: tuple(
            int(row["source_index"])
            for row in rows
            if fold_for_row(row) == fold
        )
        for fold in range(FOLDS)
    }
    if {fold: len(value) for fold, value in indices.items()} != FOLD_COUNTS:
        raise ValueError("Consistency-router fold counts differ")
    return indices


def validate_bridge_failure(root: Path) -> Mapping[str, Any]:
    protocol = validate_protocol()
    frozen = protocol["frozen_inputs"]
    path = root / "result.json"
    result = probe.validate_signed_json(path, description="Repair-bridge failure result")
    if (
        sha256_file(path) != frozen["bridge_failure_result_file_sha256"]
        or result["receipt"].get("payload_sha256")
        != frozen["bridge_failure_result_receipt_sha256"]
        or result.get("schema") != bridge_analysis.SCHEMA
        or result.get("passed") is not False
        or PROPOSAL_ID not in result.get("candidate_ids", [])
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Consistency-router bridge-failure binding differs")
    return result


def validate_materialization(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = validate_protocol()
    frozen = protocol["frozen_inputs"]
    result, manifests = bridge_runner.validate_materialization(root)
    manifest_by_id = {str(manifest["candidate_id"]): manifest for manifest in manifests}
    proposal = manifest_by_id.get(PROPOSAL_ID)
    if proposal is None:
        raise ValueError("Consistency-router proposal materialization is missing")
    required = {
        "gate_state_sha256": frozen["proposal_gate_state_sha256"],
    }
    if (
        sha256_file(root / "result.json")
        != frozen["bridge_materialization_result_file_sha256"]
        or result["receipt"].get("payload_sha256")
        != frozen["bridge_materialization_result_receipt_sha256"]
        or sha256_file(root / PROPOSAL_ID / "manifest.json")
        != frozen["proposal_manifest_sha256"]
        or proposal["patch_file"].get("sha256") != frozen["proposal_patch_sha256"]
        or any(proposal.get(key) != value for key, value in required.items())
    ):
        raise ValueError("Consistency-router proposal materialization differs")
    return result, proposal


def metric_rank(policy_id: str, metric: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        -float(metric["micro_f1"]),
        -float(metric["precision"]),
        -float(metric["recall"]),
        POLICY_ORDER.index(policy_id),
    )


def sorted_prediction(value: set[int] | None) -> list[int] | None:
    return None if value is None else sorted(value)


def route_trace(
    *,
    rows: Sequence[Mapping[str, Any]],
    checkpoint_predictions: Mapping[int, set[int] | None],
    proposal_predictions: Mapping[int, set[int] | None],
    gold: Mapping[int, set[int]],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for row in rows:
        source_index = int(row["source_index"])
        checkpoint = checkpoint_predictions[source_index]
        proposal = proposal_predictions[source_index]
        reason = proposal_reason(checkpoint, proposal)
        if reason is None:
            continue
        selected = route_prediction(checkpoint, proposal, policy_id="combined")
        gold_boundaries = gold[source_index]
        trace.append(
            {
                "source_index": source_index,
                "row_sha256": row["row_sha256"],
                "fold": fold_for_row(row),
                "reason": reason,
                "checkpoint_prediction": sorted_prediction(checkpoint),
                "proposal_prediction": sorted_prediction(proposal),
                "selected_prediction": sorted_prediction(selected),
                "gold_for_post_route_audit_only": sorted(gold_boundaries),
                "checkpoint_false_positives": (
                    None if checkpoint is None else len(checkpoint - gold_boundaries)
                ),
                "checkpoint_false_negatives": (
                    None if checkpoint is None else len(gold_boundaries - checkpoint)
                ),
                "selected_false_positives": (
                    None if selected is None else len(selected - gold_boundaries)
                ),
                "selected_false_negatives": (
                    None if selected is None else len(gold_boundaries - selected)
                ),
            }
        )
    return trace


def analyze(
    *,
    bridge_input_root: Path,
    bridge_materialization_root: Path,
    bridge_result_root: Path,
    progression_root: Path,
    dataset_root: Path,
    reference_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    protocol = validate_protocol()
    frozen = protocol["frozen_inputs"]
    rows = causal.load_rows(dataset_root)
    evaluation_rows = progression.progression_rows(rows)
    indices = tuple(int(row["source_index"]) for row in evaluation_rows)
    folds = validate_folds(evaluation_rows)
    all_gold, hashes = shared.gold_and_hashes(rows)
    bridge_failure = validate_bridge_failure(bridge_result_root)
    materialization, proposal_manifest = validate_materialization(
        bridge_materialization_root
    )
    bridge_outputs, _, _, bridge_bindings, bridge_artifacts = (
        bridge_analysis.read_bridge_outputs(
            bridge_input_root,
            rows=evaluation_rows,
            materialization_root=bridge_materialization_root,
        )
    )
    proposal_records = bridge_outputs[PROPOSAL_ID]
    runtime_hashes = {
        str(record.get("runtime_gate_state_sha256"))
        for record in proposal_records.values()
    }
    if runtime_hashes != {frozen["proposal_runtime_gate_state_sha256"]}:
        raise ValueError("Consistency-router proposal runtime state differs")
    proposal_predictions = shared.predictions_from_records(proposal_records)
    progression_result = bridge_analysis.validate_progression_result(progression_root)
    progression_outputs, progression_bindings, progression_artifacts = (
        progression_analysis.read_progression_outputs(
            progression_root,
            remaining_rows=evaluation_rows,
        )
    )
    checkpoint_predictions = shared.predictions_from_records(
        progression_outputs[bridge_runner.CONDITION]
    )
    v9_records, v9_artifacts = shared.read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    v9_predictions = shared.predictions_from_records(v9_records)
    policies = policy_predictions(checkpoint_predictions, proposal_predictions)
    aggregate_metrics = {
        policy_id: shared.metrics_from_sets(predictions, all_gold, indices)
        for policy_id, predictions in policies.items()
    }
    checkpoint_metrics = aggregate_metrics["checkpoint"]
    combined_metrics = aggregate_metrics["combined"]
    v9_metrics = shared.metrics_from_sets(v9_predictions, all_gold, indices)
    all_indices = set(indices)
    fold_results: list[dict[str, Any]] = []
    oof_predictions: dict[int, set[int] | None] = {}
    for fold in range(FOLDS):
        heldout = folds[fold]
        fit_indices = tuple(sorted(all_indices - set(heldout)))
        fit_metrics = {
            policy_id: shared.metrics_from_sets(
                predictions,
                all_gold,
                fit_indices,
            )
            for policy_id, predictions in policies.items()
        }
        selected_id = min(
            POLICY_ORDER,
            key=lambda policy_id: metric_rank(policy_id, fit_metrics[policy_id]),
        )
        for source_index in heldout:
            oof_predictions[source_index] = policies[selected_id][source_index]
        fold_results.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "heldout_rows": len(heldout),
                "selected_policy_id": selected_id,
                "fit_ranking": sorted(
                    POLICY_ORDER,
                    key=lambda policy_id: metric_rank(policy_id, fit_metrics[policy_id]),
                ),
                "selected_fit_metrics": fit_metrics[selected_id],
                "selected_heldout_metrics": shared.metrics_from_sets(
                    policies[selected_id],
                    all_gold,
                    heldout,
                ),
                "checkpoint_heldout_metrics": shared.metrics_from_sets(
                    policies["checkpoint"],
                    all_gold,
                    heldout,
                ),
            }
        )
    if set(oof_predictions) != set(indices):
        raise ValueError("Consistency-router out-of-fold coverage differs")
    oof_metrics = shared.metrics_from_sets(oof_predictions, all_gold, indices)
    selected_ids = [result["selected_policy_id"] for result in fold_results]
    combined_selected_folds = selected_ids.count("combined")
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
        "combined_minus_checkpoint_16_micro_f1": float(combined_metrics["micro_f1"])
        - float(checkpoint_metrics["micro_f1"]),
        "combined_tp_gain": int(combined_metrics["tp"]) - int(checkpoint_metrics["tp"]),
        "combined_fp_delta": int(combined_metrics["fp"]) - int(checkpoint_metrics["fp"]),
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
        "combined_selected_in_at_least_4_folds": combined_selected_folds
        >= GATE_THRESHOLDS["combined_selected_folds"],
        "combined_tp_gain_at_least_1": deltas["combined_tp_gain"]
        >= GATE_THRESHOLDS["combined_tp_gain"],
        "combined_fp_delta_at_most_0": deltas["combined_fp_delta"]
        <= GATE_THRESHOLDS["combined_fp_delta_maximum"],
    }
    gates["passed"] = all(gates.values())
    passed = bool(gates["passed"])
    trace = route_trace(
        rows=evaluation_rows,
        checkpoint_predictions=checkpoint_predictions,
        proposal_predictions=proposal_predictions,
        gold=all_gold,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "rows": len(indices),
        "source_indices": list(indices),
        "checkpoint_candidate_id": CHECKPOINT_ID,
        "proposal_candidate_id": PROPOSAL_ID,
        "policy_order": list(POLICY_ORDER),
        "fold_assignment_payload_sha256": FOLD_ASSIGNMENT_PAYLOAD_SHA256,
        "fold_counts": FOLD_COUNTS,
        "aggregate_metrics": aggregate_metrics,
        "v9_metrics": v9_metrics,
        "fold_results": fold_results,
        "fold_selected_policy_ids": selected_ids,
        "combined_selected_folds": combined_selected_folds,
        "oof_metrics": oof_metrics,
        "deltas": deltas,
        "route_trace": trace,
        "routed_rows": len(trace),
        "route_reason_counts": {
            reason: sum(item["reason"] == reason for item in trace)
            for reason in ("strict_subset", "abstention_singleton")
        },
        "gate_thresholds": GATE_THRESHOLDS,
        "gates": gates,
        "passed": passed,
        "selected_policy_id": "combined" if passed else None,
        "study_scope": "post_hoc_adaptive_development_only",
        "gold_used_by_router": False,
        "dual_pass_required": True,
        "external_replication_authorized": False,
        "multitask_preservation_authorized": False,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "bridge_failure_result": {
                "path": str(bridge_result_root / "result.json"),
                "sha256": sha256_file(bridge_result_root / "result.json"),
                "receipt_payload_sha256": bridge_failure["receipt"]["payload_sha256"],
            },
            "bridge_materialization_result": {
                "path": str(bridge_materialization_root / "result.json"),
                "sha256": sha256_file(bridge_materialization_root / "result.json"),
                "receipt_payload_sha256": materialization["receipt"]["payload_sha256"],
            },
            "proposal_manifest": {
                "candidate_id": proposal_manifest["candidate_id"],
                "gate_state_sha256": proposal_manifest["gate_state_sha256"],
                "patch_sha256": proposal_manifest["patch_file"]["sha256"],
            },
            "bridge_input_bindings": bridge_bindings,
            "bridge_candidate_outputs": bridge_artifacts,
            "checkpoint_input_bindings": progression_bindings,
            "checkpoint_outputs": progression_artifacts,
            "checkpoint_result": {
                "path": str(progression_root / "result.json"),
                "sha256": sha256_file(progression_root / "result.json"),
                "receipt_payload_sha256": progression_result["receipt"]["payload_sha256"],
            },
            "frozen_v9_outputs": v9_artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_consistency_router_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Consistency-router result must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-input-root", type=Path, required=True)
    parser.add_argument("--bridge-materialization-root", type=Path, required=True)
    parser.add_argument("--bridge-result-root", type=Path, required=True)
    parser.add_argument("--progression-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        bridge_input_root=args.bridge_input_root.expanduser().resolve(strict=True),
        bridge_materialization_root=args.bridge_materialization_root.expanduser().resolve(
            strict=True
        ),
        bridge_result_root=args.bridge_result_root.expanduser().resolve(strict=True),
        progression_root=args.progression_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "selected_policy_id": result["selected_policy_id"],
                "oof_micro_f1": result["oof_metrics"]["micro_f1"],
                "checkpoint_16_micro_f1": result["aggregate_metrics"]["checkpoint"][
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
