#!/usr/bin/env python3
"""Cross-fit simple V9/checkpoint-16 routers on open native scene rows."""

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
    analyze_natural_memory_native_scene_contrast_progression as progression_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_progression as progression,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_crossfit_router_result.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_crossfit_router_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "6e7d5d155fcb2557b6ecb9a2b23423c47e24faff09b0fae4870cacad776a4a5c"
FOLDS = 5
FOLD_SALT = "rwkv-ms-native-scene-crossfit-router-v1:"
RULES = (
    "frozen_v9",
    "checkpoint16",
    "intersection",
    "union",
    "min_cardinality_v9_tie",
    "max_cardinality_v9_tie",
    "checkpoint_if_subset_else_v9",
    "v9_if_subset_else_checkpoint",
)
PROGRESSION_RESULT_FILE_SHA256 = "af852ce316d83fb90b18bbd97fb302cdb1fe99b305c96736478759143d897cb2"
PROGRESSION_RESULT_RECEIPT_SHA256 = "23bc133d82590890308ac5b0779e54427f51fbee615d941393a023538be80b2b"
PROBE_RESULT_FILE_SHA256 = "e8854b74567d9217ee32c3d0f76623cd4703185e5121b4a11c67f8bdfc7e6f03"
PROBE_RESULT_RECEIPT_SHA256 = "c7883da73a72faa5d866617ecf98e0182dea7db88cf40f6a3d4172718c19d594"
REFERENCE_RESULT_FILE_SHA256 = "36549b036f104bc665d367b65668aef80ad94b98bdf5c63d441cb0d6ef9b422f"
REFERENCE_RESULT_RECEIPT_SHA256 = "26a2248976ff009804744a19a738fd2124061cf6d909bdc74c9e7c040098c091"


def canonical_sha256(value: Any) -> str:
    return progression.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return progression.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Scene crossfit-router protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Scene crossfit-router protocol hash differs")
    if tuple(value.get("candidate_rules_in_tie_break_order", ())) != RULES:
        raise ValueError("Scene crossfit-router candidate order differs")
    return value


def validate_signed_source(
    path: Path,
    *,
    description: str,
    expected_file_sha256: str,
    expected_receipt_sha256: str,
) -> Mapping[str, Any]:
    value = probe.validate_signed_json(path, description=description)
    if (
        sha256_file(path) != expected_file_sha256
        or value["receipt"].get("payload_sha256") != expected_receipt_sha256
    ):
        raise ValueError(f"{description} binding differs")
    return value


def fold_for_hash(row_sha256: str) -> int:
    digest = hashlib.sha256((FOLD_SALT + row_sha256).encode("ascii")).hexdigest()
    return int(digest[:8], 16) % FOLDS


def apply_rule(
    rule: str,
    frozen_v9: set[int] | None,
    checkpoint16: set[int] | None,
) -> set[int] | None:
    if rule not in RULES:
        raise ValueError(f"Unknown scene crossfit-router rule: {rule}")
    if frozen_v9 is None:
        return None if checkpoint16 is None else set(checkpoint16)
    if checkpoint16 is None:
        return set(frozen_v9)
    v9 = set(frozen_v9)
    checkpoint = set(checkpoint16)
    if rule == "frozen_v9":
        return v9
    if rule == "checkpoint16":
        return checkpoint
    if rule == "intersection":
        return v9 & checkpoint
    if rule == "union":
        return v9 | checkpoint
    if rule == "min_cardinality_v9_tie":
        return checkpoint if len(checkpoint) < len(v9) else v9
    if rule == "max_cardinality_v9_tie":
        return checkpoint if len(checkpoint) > len(v9) else v9
    if rule == "checkpoint_if_subset_else_v9":
        return checkpoint if checkpoint <= v9 else v9
    if rule == "v9_if_subset_else_checkpoint":
        return v9 if v9 <= checkpoint else checkpoint
    raise AssertionError(f"Unhandled scene crossfit-router rule: {rule}")


def routed_predictions(
    rule: str,
    frozen_v9: Mapping[int, set[int] | None],
    checkpoint16: Mapping[int, set[int] | None],
) -> dict[int, set[int] | None]:
    if set(frozen_v9) != set(checkpoint16):
        raise ValueError("Scene crossfit-router input coverage differs")
    return {
        source_index: apply_rule(
            rule,
            frozen_v9[source_index],
            checkpoint16[source_index],
        )
        for source_index in frozen_v9
    }


def select_rule(train_scores: Mapping[str, float]) -> str:
    if set(train_scores) != set(RULES):
        raise ValueError("Scene crossfit-router score candidates differ")
    best = max(float(train_scores[rule]) for rule in RULES)
    return next(rule for rule in RULES if float(train_scores[rule]) == best)


def evaluate_gates(
    *,
    router_metrics: Mapping[str, Any],
    frozen_v9_metrics: Mapping[str, Any],
    checkpoint16_metrics: Mapping[str, Any],
    worst_fold_delta: float,
) -> Mapping[str, Any]:
    router_f1 = float(router_metrics["micro_f1"])
    v9_delta = router_f1 - float(frozen_v9_metrics["micro_f1"])
    checkpoint_delta = router_f1 - float(checkpoint16_metrics["micro_f1"])
    gates: dict[str, Any] = {
        "router_coverage_at_least_0.95": float(router_metrics["coverage"]) >= 0.95,
        "router_minus_frozen_v9_micro_f1_at_least_0.005": v9_delta >= 0.005,
        "router_minus_checkpoint16_micro_f1_at_least_0.005": checkpoint_delta >= 0.005,
        "worst_fold_router_minus_better_baseline_at_least_minus_0.02": (
            worst_fold_delta >= -0.02
        ),
        "router_minus_frozen_v9_micro_f1": v9_delta,
        "router_minus_checkpoint16_micro_f1": checkpoint_delta,
        "worst_fold_router_minus_better_baseline_micro_f1": worst_fold_delta,
    }
    numeric = {
        "router_minus_frozen_v9_micro_f1",
        "router_minus_checkpoint16_micro_f1",
        "worst_fold_router_minus_better_baseline_micro_f1",
    }
    gates["passed"] = all(value for key, value in gates.items() if key not in numeric)
    return gates


def load_bound_predictions(
    *,
    progression_root: Path,
    probe_root: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
) -> tuple[
    tuple[int, ...],
    dict[int, set[int]],
    dict[int, str],
    dict[int, set[int] | None],
    dict[int, set[int] | None],
    Mapping[str, Any],
]:
    progression_result = validate_signed_source(
        progression_root / "result.json",
        description="Scene contrast progression result",
        expected_file_sha256=PROGRESSION_RESULT_FILE_SHA256,
        expected_receipt_sha256=PROGRESSION_RESULT_RECEIPT_SHA256,
    )
    if progression_result.get("passed") is not True or progression_result.get("protected_splits_opened") != []:
        raise ValueError("Scene contrast progression result authorization differs")
    probe_result = progression_analysis.validate_probe_result(probe_root)
    if (
        sha256_file(probe_root / "result.json") != PROBE_RESULT_FILE_SHA256
        or probe_result["receipt"].get("payload_sha256") != PROBE_RESULT_RECEIPT_SHA256
    ):
        raise ValueError("Scene contrast probe result binding differs")
    reference_result = validate_signed_source(
        reference_root / "result.json",
        description="Frozen V9 routed result",
        expected_file_sha256=REFERENCE_RESULT_FILE_SHA256,
        expected_receipt_sha256=REFERENCE_RESULT_RECEIPT_SHA256,
    )
    if reference_result.get("scope", {}).get("protected_splits_opened") != []:
        raise ValueError("Frozen V9 reference authorization differs")
    progression.selected_manifest(training_root)

    rows = causal.load_rows(dataset_root)
    probe_rows = probe.selected_probe_rows(rows)
    remaining_rows = progression.progression_rows(rows)
    fit_rows = [*probe_rows, *remaining_rows]
    indices = tuple(sorted(int(row["source_index"]) for row in fit_rows))
    if len(indices) != 284 or len(set(indices)) != 284:
        raise ValueError("Scene crossfit-router fit coverage differs")
    gold, hashes = shared.gold_and_hashes(rows)
    remaining_outputs, progression_bindings, progression_artifacts = (
        progression_analysis.read_progression_outputs(
            progression_root,
            remaining_rows=remaining_rows,
        )
    )
    probe_outputs, probe_bindings, probe_artifacts = (
        progression_analysis.probe_analysis.read_candidate_outputs(
            probe_root,
            selected_rows=probe_rows,
        )
    )
    selected_probe = probe_outputs[progression.SELECTED_STEP]["correct_state"]
    checkpoint_records = dict(selected_probe)
    checkpoint_records.update(remaining_outputs["correct_state"])
    v9_records, v9_artifacts = shared.read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    checkpoint_predictions = shared.predictions_from_records(checkpoint_records)
    v9_predictions = shared.predictions_from_records(v9_records)
    selected_gold = {source_index: gold[source_index] for source_index in indices}
    selected_hashes = {source_index: hashes[source_index] for source_index in indices}
    selected_checkpoint = {
        source_index: checkpoint_predictions[source_index] for source_index in indices
    }
    selected_v9 = {source_index: v9_predictions[source_index] for source_index in indices}
    provenance = {
        "progression_result": {
            "path": str(progression_root / "result.json"),
            "sha256": PROGRESSION_RESULT_FILE_SHA256,
            "receipt_payload_sha256": PROGRESSION_RESULT_RECEIPT_SHA256,
        },
        "probe_result": {
            "path": str(probe_root / "result.json"),
            "sha256": PROBE_RESULT_FILE_SHA256,
            "receipt_payload_sha256": PROBE_RESULT_RECEIPT_SHA256,
        },
        "reference_result": {
            "path": str(reference_root / "result.json"),
            "sha256": REFERENCE_RESULT_FILE_SHA256,
            "receipt_payload_sha256": REFERENCE_RESULT_RECEIPT_SHA256,
        },
        "progression_input_bindings": progression_bindings,
        "progression_outputs": progression_artifacts,
        "probe_input_bindings": probe_bindings,
        "probe_outputs": probe_artifacts,
        "frozen_v9_outputs": v9_artifacts,
    }
    return (
        indices,
        selected_gold,
        selected_hashes,
        selected_v9,
        selected_checkpoint,
        provenance,
    )


def analyze(
    *,
    progression_root: Path,
    probe_root: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    validate_protocol()
    if output.exists():
        raise ValueError(f"Scene crossfit-router output must be fresh: {output}")
    (
        indices,
        gold,
        hashes,
        frozen_v9,
        checkpoint16,
        provenance,
    ) = load_bound_predictions(
        progression_root=progression_root,
        probe_root=probe_root,
        dataset_root=dataset_root,
        reference_root=reference_root,
        training_root=training_root,
    )
    predictions = {
        rule: routed_predictions(rule, frozen_v9, checkpoint16) for rule in RULES
    }
    fold_by_index = {
        source_index: fold_for_hash(hashes[source_index]) for source_index in indices
    }
    fold_counts = Counter(fold_by_index.values())
    if set(fold_counts) != set(range(FOLDS)):
        raise ValueError("Scene crossfit-router fold coverage differs")

    crossfit_predictions: dict[int, set[int] | None] = {}
    fold_results: list[dict[str, Any]] = []
    selected_rules: list[str] = []
    fold_deltas: list[float] = []
    for fold in range(FOLDS):
        heldout = tuple(
            source_index for source_index in indices if fold_by_index[source_index] == fold
        )
        train = tuple(
            source_index for source_index in indices if fold_by_index[source_index] != fold
        )
        train_metrics = {
            rule: shared.metrics_from_sets(predictions[rule], gold, train)
            for rule in RULES
        }
        train_scores = {
            rule: float(train_metrics[rule]["micro_f1"]) for rule in RULES
        }
        selected_rule = select_rule(train_scores)
        selected_rules.append(selected_rule)
        for source_index in heldout:
            crossfit_predictions[source_index] = predictions[selected_rule][source_index]
        heldout_router = shared.metrics_from_sets(predictions[selected_rule], gold, heldout)
        heldout_v9 = shared.metrics_from_sets(frozen_v9, gold, heldout)
        heldout_checkpoint = shared.metrics_from_sets(checkpoint16, gold, heldout)
        better_baseline = max(
            float(heldout_v9["micro_f1"]),
            float(heldout_checkpoint["micro_f1"]),
        )
        delta = float(heldout_router["micro_f1"]) - better_baseline
        fold_deltas.append(delta)
        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "heldout_payload_sha256": canonical_sha256(
                    [
                        {
                            "source_index": source_index,
                            "row_sha256": hashes[source_index],
                        }
                        for source_index in heldout
                    ]
                ),
                "selected_rule": selected_rule,
                "train_rule_metrics": train_metrics,
                "heldout": {
                    "router": heldout_router,
                    "frozen_v9": heldout_v9,
                    "checkpoint16": heldout_checkpoint,
                    "router_minus_better_baseline_micro_f1": delta,
                },
            }
        )
    if set(crossfit_predictions) != set(indices):
        raise ValueError("Scene crossfit-router prediction coverage differs")

    router_metrics = shared.metrics_from_sets(crossfit_predictions, gold, indices)
    v9_metrics = shared.metrics_from_sets(frozen_v9, gold, indices)
    checkpoint_metrics = shared.metrics_from_sets(checkpoint16, gold, indices)
    gates = evaluate_gates(
        router_metrics=router_metrics,
        frozen_v9_metrics=v9_metrics,
        checkpoint16_metrics=checkpoint_metrics,
        worst_fold_delta=min(fold_deltas),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "scope": {
            "split": "publisher-TRAIN-derived combined open scene fit",
            "rows": len(indices),
            "folds": FOLDS,
            "fold_counts": {str(fold): fold_counts[fold] for fold in range(FOLDS)},
            "protected_splits_opened": [],
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
            "unused_strength_holdout_opened": False,
        },
        "candidate_rules_in_tie_break_order": list(RULES),
        "fold_results": fold_results,
        "selected_rule_counts": dict(sorted(Counter(selected_rules).items())),
        "aggregate": {
            "crossfit_router": router_metrics,
            "frozen_v9": v9_metrics,
            "checkpoint16": checkpoint_metrics,
            "output_change_fraction_vs_v9": shared.output_change_fraction(
                crossfit_predictions,
                frozen_v9,
                indices,
            ),
            "output_change_fraction_vs_checkpoint16": shared.output_change_fraction(
                crossfit_predictions,
                checkpoint16,
                indices,
            ),
        },
        "gates": gates,
        "passed": bool(gates["passed"]),
        "new_external_replication_authorized": bool(gates["passed"]),
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            **provenance,
            "analyzer_sha256": sha256_file(Path(__file__)),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_scene_crossfit_router_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progression-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        progression_root=args.progression_root.expanduser().resolve(strict=True),
        probe_root=args.probe_root.expanduser().resolve(strict=True),
        dataset_root=args.dataset_root.expanduser().resolve(strict=True),
        reference_root=args.reference_root.expanduser().resolve(strict=True),
        training_root=args.training_root.expanduser().resolve(strict=True),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "selected_rule_counts": result["selected_rule_counts"],
                "aggregate": result["aggregate"],
                "gates": result["gates"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
