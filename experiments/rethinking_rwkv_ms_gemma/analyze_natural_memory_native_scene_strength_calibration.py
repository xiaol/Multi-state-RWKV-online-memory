#!/usr/bin/env python3
"""Analyze and sign the two-phase native scene strength-calibration study."""

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
    analyze_natural_memory_native_scene_state_retrieval as shared_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal_runner,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_strength_calibration as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_calibration_result.v1"
EXPECTED_RUNNER_SHA256 = "073435b930b9f3f80875dadef432fdcf6d75de89be5c5919d335d3c3abcbdf85"
FULL_STRENGTH_NAME = "scale_1p0"
ZERO_STRENGTH_NAME = "scale_0p0"


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def phase_indices(hashes: Mapping[int, str], phase: str) -> tuple[int, ...]:
    indices = tuple(
        source_index
        for source_index in sorted(hashes)
        if runner.partition_for_hash(hashes[source_index]) == phase
    )
    expected = 284 if phase == "fit" else 73
    if len(indices) != expected:
        raise ValueError(f"Strength-calibration {phase} partition count differs")
    return indices


def read_outputs(
    root: Path,
    *,
    phase: str,
    strength_names: Sequence[str],
    indices: Sequence[int],
    hashes: Mapping[int, str],
    selection: Mapping[str, Any] | None,
    selection_path: Path | None,
) -> tuple[
    dict[str, dict[int, Mapping[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    expected_indices = set(indices)
    outputs: dict[str, dict[int, Mapping[str, Any]]] = {
        name: {} for name in strength_names
    }
    artifacts: dict[str, list[dict[str, Any]]] = {
        name: [] for name in strength_names
    }
    bindings: list[dict[str, Any]] = []
    settings_digest_by_strength: dict[str, str] = {}
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / phase / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "phase": phase,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "strengths": {name: runner.STRENGTHS[name] for name in strength_names},
            "runner_sha256": EXPECTED_RUNNER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError(f"Strength-calibration input binding differs: {binding_path}")
        if phase == "fit":
            if any(
                binding.get(key) is not None
                for key in (
                    "selection_path",
                    "selection_file_sha256",
                    "selection_payload_sha256",
                )
            ):
                raise ValueError("Strength-calibration fit binding contains selection")
        else:
            if selection is None or selection_path is None:
                raise ValueError("Strength-calibration holdout selection is missing")
            if binding.get("selection_file_sha256") != sha256_file(selection_path):
                raise ValueError("Strength-calibration selection file hash differs")
            if binding.get("selection_payload_sha256") != selection["receipt"]["payload_sha256"]:
                raise ValueError("Strength-calibration selection payload differs")
        bindings.append(
            {
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
                "payload": binding,
            }
        )
        for strength_name in strength_names:
            path = shard_dir / f"{strength_name}.jsonl"
            rows = shared_analysis.read_jsonl(path)
            artifacts[strength_name].append(
                {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
            )
            for record in rows:
                source_index = int(record["source_index"])
                expected_record = {
                    "schema": runner.SCHEMA,
                    "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                    "phase": phase,
                    "strength_name": strength_name,
                    "strength": runner.STRENGTHS[strength_name],
                    "wrapped_layers": 42,
                    "shard_index": shard_index,
                    "world_size": runner.WORLD_SIZE,
                    "row_sha256": hashes[source_index],
                }
                if any(record.get(key) != value for key, value in expected_record.items()):
                    raise ValueError(f"Strength-calibration output binding differs: {path}:{source_index}")
                if source_index not in expected_indices or source_index % 4 != shard_index:
                    raise ValueError(f"Strength-calibration output shard differs: {path}:{source_index}")
                settings_digest = record.get("settings_sha256")
                if not isinstance(settings_digest, str):
                    raise ValueError(f"Strength-calibration settings digest missing: {path}:{source_index}")
                prior = settings_digest_by_strength.setdefault(strength_name, settings_digest)
                if settings_digest != prior:
                    raise ValueError(f"Strength-calibration settings differ: {strength_name}")
                if source_index in outputs[strength_name]:
                    raise ValueError(f"Duplicate strength-calibration row: {strength_name}:{source_index}")
                outputs[strength_name][source_index] = record
    for strength_name in strength_names:
        if set(outputs[strength_name]) != expected_indices:
            raise ValueError(f"Incomplete strength-calibration output: {phase}:{strength_name}")
    return outputs, artifacts, bindings


def gate_result(
    selected_metrics: Mapping[str, Any],
    selected_predictions: Mapping[int, set[int] | None],
    full_metrics: Mapping[str, Any],
    full_predictions: Mapping[int, set[int] | None],
    zero_metrics: Mapping[str, Any],
    indices: Sequence[int],
    *,
    selected_is_intermediate: bool,
) -> Mapping[str, Any]:
    selected_f1 = float(selected_metrics["micro_f1"])
    full_f1 = float(full_metrics["micro_f1"])
    zero_f1 = float(zero_metrics["micro_f1"])
    change_fraction = shared_analysis.output_change_fraction(
        selected_predictions,
        full_predictions,
        indices,
    )
    gates: dict[str, Any] = {
        "selected_is_intermediate_strength": selected_is_intermediate,
        "coverage_at_least_0.95": float(selected_metrics["coverage"]) >= 0.95,
        "minus_full_strength_micro_f1_at_least_0.005": selected_f1 - full_f1 >= 0.005,
        "minus_zero_strength_micro_f1_at_least_0.02": selected_f1 - zero_f1 >= 0.02,
        "paired_output_change_fraction_vs_full_strength_at_least_0.05": change_fraction >= 0.05,
    }
    gates["passed"] = all(gates.values())
    return {
        "selected_minus_full_strength_micro_f1": selected_f1 - full_f1,
        "selected_minus_zero_strength_micro_f1": selected_f1 - zero_f1,
        "paired_output_change_fraction_vs_full_strength": change_fraction,
        "gates": gates,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Strength-calibration output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_selection(
    path: Path,
    *,
    result_path: Path,
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected_name = str(result["selected_strength_name"])
    if not result["passed"] or selected_name not in runner.STRENGTHS:
        raise ValueError("Strength-calibration selection is unauthorized")
    value: dict[str, Any] = {
        "schema": runner.SELECTION_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase_one_result": {
            "path": result_path.name,
            "file_sha256": sha256_file(result_path),
            "receipt_payload_sha256": result["receipt"]["payload_sha256"],
        },
        "selected_strength_name": selected_name,
        "selected_strength": runner.STRENGTHS[selected_name],
        "selected_phase_one": result["selected"],
        "phase_one_passed": True,
        "holdout_authorized": True,
        "publisher_validation_predictions_opened": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "accepted_validation_decoder_changed": False,
    }
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_selection_without_receipt",
        "payload_sha256": canonical_sha256(value),
    }
    write_json(path, value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase", choices=runner.PHASES, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner.validate_protocol()
    if sha256_file(runner.Path(runner.__file__)) != EXPECTED_RUNNER_SHA256:
        raise ValueError("Strength-calibration runner hash differs")
    input_root = args.input_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    zero_root = args.zero_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    selection: Mapping[str, Any] | None = None
    selection_path: Path | None = None
    if args.phase == "fit":
        if args.selection is not None:
            raise ValueError("Strength-calibration fit analysis forbids selection")
        strength_names = tuple(runner.STRENGTHS)
    else:
        if args.selection is None or args.selection_output is not None:
            raise ValueError("Strength-calibration holdout analysis requires only --selection")
        selection_path = args.selection.expanduser().resolve(strict=True)
        selection = runner.validate_selection(
            selection_path,
            runner_sha256=EXPECTED_RUNNER_SHA256,
        )
        strength_names = (str(selection["selected_strength_name"]),)
    rows = causal_runner.load_rows(dataset_root)
    runner.validate_partitions(rows)
    gold, hashes = shared_analysis.gold_and_hashes(rows)
    indices = phase_indices(hashes, args.phase)
    base_records, base_artifacts = shared_analysis.read_reference_condition(
        reference_root,
        "base",
        hashes,
    )
    full_records, full_artifacts = shared_analysis.read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    zero_records, zero_artifacts = shared_analysis.read_zero_condition(zero_root, hashes)
    outputs, artifacts, bindings = read_outputs(
        input_root,
        phase=args.phase,
        strength_names=strength_names,
        indices=indices,
        hashes=hashes,
        selection=selection,
        selection_path=selection_path,
    )
    base_predictions = shared_analysis.predictions_from_records(base_records)
    full_predictions = shared_analysis.predictions_from_records(full_records)
    zero_predictions = shared_analysis.predictions_from_records(zero_records)
    candidate_predictions = {
        name: shared_analysis.predictions_from_records(records)
        for name, records in outputs.items()
    }
    baseline_metrics = {
        "base": shared_analysis.metrics_from_sets(base_predictions, gold, indices),
        ZERO_STRENGTH_NAME: shared_analysis.metrics_from_sets(zero_predictions, gold, indices),
        FULL_STRENGTH_NAME: shared_analysis.metrics_from_sets(full_predictions, gold, indices),
    }
    candidate_metrics = {
        name: shared_analysis.metrics_from_sets(predictions, gold, indices)
        for name, predictions in candidate_predictions.items()
    }
    if args.phase == "fit":
        all_metrics = {**candidate_metrics, FULL_STRENGTH_NAME: baseline_metrics[FULL_STRENGTH_NAME]}
        strengths = {**runner.STRENGTHS, FULL_STRENGTH_NAME: 1.0}
        selected_name = min(
            all_metrics,
            key=lambda name: (
                -float(all_metrics[name]["micro_f1"]),
                -float(all_metrics[name]["precision"]),
                -float(all_metrics[name]["recall"]),
                strengths[name],
            ),
        )
    else:
        if selection is None:
            raise ValueError("Strength-calibration holdout selection is missing")
        selected_name = str(selection["selected_strength_name"])
    selected_metrics = (
        candidate_metrics[selected_name]
        if selected_name in candidate_metrics
        else baseline_metrics[FULL_STRENGTH_NAME]
    )
    selected_predictions = (
        candidate_predictions[selected_name]
        if selected_name in candidate_predictions
        else full_predictions
    )
    selected_gate = gate_result(
        selected_metrics,
        selected_predictions,
        baseline_metrics[FULL_STRENGTH_NAME],
        full_predictions,
        baseline_metrics[ZERO_STRENGTH_NAME],
        indices,
        selected_is_intermediate=selected_name in runner.STRENGTHS,
    )
    passed = bool(selected_gate["gates"]["passed"])
    selected_strength = runner.STRENGTHS.get(selected_name, 1.0)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase": args.phase,
        "scope": {
            "split": "publisher-TRAIN-derived development",
            "rows": len(indices),
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
        },
        "baselines": baseline_metrics,
        "candidate_strengths": candidate_metrics,
        "selected_strength_name": selected_name,
        "selected_strength": selected_strength,
        "selected": {"metrics": selected_metrics, **selected_gate},
        "passed": passed,
        "holdout_authorized": args.phase == "fit" and passed,
        "accepted_validation_decoder_changed": False,
        "selection_receipt_payload_sha256": (
            None if selection is None else selection["receipt"]["payload_sha256"]
        ),
        "provenance": {
            "input_bindings": bindings,
            "baselines": {
                "base": base_artifacts,
                ZERO_STRENGTH_NAME: zero_artifacts,
                FULL_STRENGTH_NAME: full_artifacts,
            },
            "intermediate_strengths": artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": f"canonical_strength_calibration_{args.phase}_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output, result)
    if args.phase == "fit" and passed:
        if args.selection_output is None:
            raise ValueError("Passing strength calibration requires --selection-output")
        write_selection(args.selection_output.expanduser().resolve(), result_path=output, result=result)
    elif args.phase == "holdout" and args.selection_output is not None:
        raise ValueError("Strength-calibration selection output is unauthorized")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "selected_strength_name": selected_name,
                "selected_strength": selected_strength,
                "selected": result["selected"],
                "passed": passed,
                "holdout_authorized": result["holdout_authorized"],
                "receipt": result["receipt"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
