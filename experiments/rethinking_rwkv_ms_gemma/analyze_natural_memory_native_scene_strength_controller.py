#!/usr/bin/env python3
"""Analyze and sign effective native scene strength-controller phases."""

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
    analyze_natural_memory_native_scene_strength_calibration as v1_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_strength_controller as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_controller_result.v2"
EXPECTED_RUNNER_SHA256 = "6db74e482f55eb24d9f7afca8ad0e7f0761a786b13894415645e9b6a3c366380"
FULL_NAME = "scale_1p0"
ZERO_NAME = "scale_0p0"


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Strength-controller output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_phase_outputs(
    root: Path,
    *,
    phase: str,
    strengths: Mapping[str, float],
    indices: Sequence[int],
    hashes: Mapping[int, str],
    preflight: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> tuple[
    dict[str, dict[int, Mapping[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    expected_indices = set(indices)
    outputs: dict[str, dict[int, Mapping[str, Any]]] = {
        name: {} for name in strengths
    }
    artifacts: dict[str, list[dict[str, Any]]] = {
        name: [] for name in strengths
    }
    bindings: list[dict[str, Any]] = []
    controller_digests: dict[str, str] = {}
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / phase / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "runtime_sha256": runner.RUNTIME_SHA256,
            "phase": phase,
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "strengths": dict(strengths),
            "runner_sha256": EXPECTED_RUNNER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError(f"Strength-controller input binding differs: {binding_path}")
        expected_preflight = None if preflight is None else preflight["receipt"]["payload_sha256"]
        expected_selection = None if selection is None else selection["receipt"]["payload_sha256"]
        if binding.get("preflight_payload_sha256") != expected_preflight:
            raise ValueError(f"Strength-controller preflight binding differs: {binding_path}")
        if binding.get("selection_payload_sha256") != expected_selection:
            raise ValueError(f"Strength-controller selection binding differs: {binding_path}")
        bindings.append(
            {
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
                "payload": binding,
            }
        )
        for strength_name, strength in strengths.items():
            path = shard_dir / f"{strength_name}.jsonl"
            rows = shared.read_jsonl(path)
            artifacts[strength_name].append(
                {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
            )
            for record in rows:
                source_index = int(record["source_index"])
                required = {
                    "schema": runner.SCHEMA,
                    "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                    "phase": phase,
                    "strength_name": strength_name,
                    "strength": strength,
                    "wrapped_layers": 42,
                    "shard_index": shard_index,
                    "world_size": runner.WORLD_SIZE,
                    "row_sha256": hashes[source_index],
                }
                if any(record.get(key) != value for key, value in required.items()):
                    raise ValueError(f"Strength-controller record differs: {path}:{source_index}")
                calls = record.get("controller_calls")
                if not isinstance(calls, Mapping) or int(calls.get("q", 0)) <= 0 or int(calls.get("o", 0)) <= 0:
                    raise ValueError(f"Strength-controller calls differ: {path}:{source_index}")
                digest = str(record.get("controller_sha256"))
                prior = controller_digests.setdefault(strength_name, digest)
                if digest != prior:
                    raise ValueError(f"Strength-controller digest differs: {strength_name}")
                if source_index not in expected_indices or source_index % 4 != shard_index:
                    raise ValueError(f"Strength-controller shard differs: {path}:{source_index}")
                if source_index in outputs[strength_name]:
                    raise ValueError(f"Duplicate strength-controller row: {strength_name}:{source_index}")
                outputs[strength_name][source_index] = record
    for strength_name in strengths:
        if set(outputs[strength_name]) != expected_indices:
            raise ValueError(f"Incomplete strength-controller output: {phase}:{strength_name}")
    return outputs, artifacts, bindings


def analyze_preflight(
    *,
    input_root: Path,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> tuple[Mapping[str, Any], bool]:
    hashes = {int(row["source_index"]): str(row["row_sha256"]) for row in rows}
    indices = (0, 1, 2, 3)
    outputs, artifacts, bindings = read_phase_outputs(
        input_root,
        phase="preflight",
        strengths=runner.PREFLIGHT_STRENGTHS,
        indices=indices,
        hashes=hashes,
        preflight=None,
        selection=None,
    )
    predictions = {
        name: shared.predictions_from_records(records)
        for name, records in outputs.items()
    }
    endpoint_changes = sum(
        predictions[ZERO_NAME][index] != predictions[FULL_NAME][index]
        for index in indices
    )
    midpoint_changes = sum(
        predictions["scale_0p5"][index] != predictions[ZERO_NAME][index]
        or predictions["scale_0p5"][index] != predictions[FULL_NAME][index]
        for index in indices
    )
    format_recovered = all(
        predictions[name][index] is not None
        for name in predictions
        for index in indices
    )
    gates: dict[str, Any] = {
        "controller_attached_to_42_layers": all(
            record["wrapped_layers"] == 42
            for records in outputs.values()
            for record in records.values()
        ),
        "controller_head_calls_include_q_and_o": all(
            record["controller_calls"]["q"] > 0
            and record["controller_calls"]["o"] > 0
            for records in outputs.values()
            for record in records.values()
        ),
        "scale_0_output_differs_from_scale_1_on_at_least_one_row": endpoint_changes >= 1,
        "scale_0p5_output_differs_from_at_least_one_endpoint_on_at_least_one_row": midpoint_changes >= 1,
        "all_generations_format_recover": format_recovered,
    }
    gates["passed"] = all(gates.values())
    result: dict[str, Any] = {
        "schema": runner.PREFLIGHT_SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase": "preflight",
        "source_indices": list(indices),
        "endpoint_output_change_rows": endpoint_changes,
        "midpoint_output_change_rows": midpoint_changes,
        "predictions": {
            name: {str(index): sorted(value) if value is not None else None for index, value in values.items()}
            for name, values in predictions.items()
        },
        "gates": gates,
        "passed": bool(gates["passed"]),
        "protected_splits_opened": [],
        "provenance": {"input_bindings": bindings, "outputs": artifacts},
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_strength_controller_preflight_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output, result)
    return result, bool(gates["passed"])


def phase_indices(hashes: Mapping[int, str], phase: str) -> tuple[int, ...]:
    indices = tuple(
        source_index
        for source_index in sorted(hashes)
        if runner.v1.partition_for_hash(hashes[source_index]) == phase
    )
    expected = 284 if phase == "fit" else 73
    if len(indices) != expected:
        raise ValueError(f"Strength-controller {phase} count differs")
    return indices


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
    change_fraction = shared.output_change_fraction(
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


def write_selection(path: Path, *, result_path: Path, result: Mapping[str, Any]) -> None:
    selected_name = str(result["selected_strength_name"])
    if not result["passed"] or selected_name not in runner.FIT_STRENGTHS:
        raise ValueError("Strength-controller selection is unauthorized")
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
        "selected_strength": runner.FIT_STRENGTHS[selected_name],
        "selected_phase_one": result["selected"],
        "passed": True,
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


def analyze_reported_phase(
    *,
    input_root: Path,
    reference_root: Path,
    zero_root: Path,
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    preflight: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    output: Path,
) -> tuple[Mapping[str, Any], bool]:
    gold, hashes = shared.gold_and_hashes(rows)
    indices = phase_indices(hashes, phase)
    strengths = (
        runner.FIT_STRENGTHS
        if phase == "fit"
        else {str(selection["selected_strength_name"]): float(selection["selected_strength"])}
    )
    outputs, artifacts, bindings = read_phase_outputs(
        input_root,
        phase=phase,
        strengths=strengths,
        indices=indices,
        hashes=hashes,
        preflight=preflight,
        selection=selection,
    )
    base_records, base_artifacts = shared.read_reference_condition(reference_root, "base", hashes)
    full_records, full_artifacts = shared.read_reference_condition(reference_root, "memory", hashes)
    zero_records, zero_artifacts = shared.read_zero_condition(zero_root, hashes)
    base_predictions = shared.predictions_from_records(base_records)
    full_predictions = shared.predictions_from_records(full_records)
    zero_predictions = shared.predictions_from_records(zero_records)
    candidate_predictions = {
        name: shared.predictions_from_records(records)
        for name, records in outputs.items()
    }
    baselines = {
        "base": shared.metrics_from_sets(base_predictions, gold, indices),
        ZERO_NAME: shared.metrics_from_sets(zero_predictions, gold, indices),
        FULL_NAME: shared.metrics_from_sets(full_predictions, gold, indices),
    }
    candidate_metrics = {
        name: shared.metrics_from_sets(predictions, gold, indices)
        for name, predictions in candidate_predictions.items()
    }
    if phase == "fit":
        all_metrics = {**candidate_metrics, FULL_NAME: baselines[FULL_NAME]}
        strength_values = {**runner.FIT_STRENGTHS, FULL_NAME: 1.0}
        selected_name = min(
            all_metrics,
            key=lambda name: (
                -float(all_metrics[name]["micro_f1"]),
                -float(all_metrics[name]["precision"]),
                -float(all_metrics[name]["recall"]),
                strength_values[name],
            ),
        )
    else:
        selected_name = str(selection["selected_strength_name"])
    selected_metrics = (
        candidate_metrics[selected_name]
        if selected_name in candidate_metrics
        else baselines[FULL_NAME]
    )
    selected_predictions = (
        candidate_predictions[selected_name]
        if selected_name in candidate_predictions
        else full_predictions
    )
    selected_gate = gate_result(
        selected_metrics,
        selected_predictions,
        baselines[FULL_NAME],
        full_predictions,
        baselines[ZERO_NAME],
        indices,
        selected_is_intermediate=selected_name in runner.FIT_STRENGTHS,
    )
    passed = bool(selected_gate["gates"]["passed"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase": phase,
        "scope": {
            "split": "publisher-TRAIN-derived development",
            "rows": len(indices),
            "publisher_validation_predictions_opened": False,
            "publisher_test_opened": False,
            "hard32_opened": False,
        },
        "baselines": baselines,
        "candidate_strengths": candidate_metrics,
        "selected_strength_name": selected_name,
        "selected_strength": runner.FIT_STRENGTHS.get(selected_name, 1.0),
        "selected": {"metrics": selected_metrics, **selected_gate},
        "passed": passed,
        "holdout_authorized": phase == "fit" and passed,
        "accepted_validation_decoder_changed": False,
        "preflight_receipt_payload_sha256": preflight["receipt"]["payload_sha256"],
        "selection_receipt_payload_sha256": (
            None if selection is None else selection["receipt"]["payload_sha256"]
        ),
        "provenance": {
            "input_bindings": bindings,
            "baselines": {
                "base": base_artifacts,
                ZERO_NAME: zero_artifacts,
                FULL_NAME: full_artifacts,
            },
            "controller_strengths": artifacts,
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": f"canonical_strength_controller_{phase}_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    write_json(output, result)
    return result, passed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--zero-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase", choices=runner.PHASES, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner.validate_protocol()
    if sha256_file(runner.Path(runner.__file__)) != EXPECTED_RUNNER_SHA256:
        raise ValueError("Strength-controller runner hash differs")
    input_root = args.input_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    rows = causal.load_rows(dataset_root)
    runner.v1.validate_partitions(rows)
    if args.phase == "preflight":
        if any(value is not None for value in (args.preflight, args.selection, args.selection_output)):
            raise ValueError("Strength-controller preflight analysis forbids receipts")
        result, passed = analyze_preflight(input_root=input_root, rows=rows, output=output)
    else:
        if args.preflight is None or args.reference_root is None or args.zero_root is None:
            raise ValueError("Strength-controller reported analysis requires preflight and baselines")
        preflight = runner.validate_signed_receipt(
            args.preflight.expanduser().resolve(strict=True),
            schema=runner.PREFLIGHT_SCHEMA,
            runner_sha256=EXPECTED_RUNNER_SHA256,
            require_passed=True,
        )
        selection: Mapping[str, Any] | None = None
        if args.phase == "fit":
            if args.selection is not None:
                raise ValueError("Strength-controller fit analysis forbids selection")
        else:
            if args.selection is None or args.selection_output is not None:
                raise ValueError("Strength-controller holdout analysis requires only selection")
            selection = runner.validate_selection(
                args.selection.expanduser().resolve(strict=True),
                runner_sha256=EXPECTED_RUNNER_SHA256,
            )
        result, passed = analyze_reported_phase(
            input_root=input_root,
            reference_root=args.reference_root.expanduser().resolve(strict=True),
            zero_root=args.zero_root.expanduser().resolve(strict=True),
            rows=rows,
            phase=args.phase,
            preflight=preflight,
            selection=selection,
            output=output,
        )
        if args.phase == "fit" and passed:
            if args.selection_output is None:
                raise ValueError("Passing strength-controller fit requires selection output")
            write_selection(
                args.selection_output.expanduser().resolve(),
                result_path=output,
                result=result,
            )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "passed": passed,
                "holdout_authorized": bool(result.get("holdout_authorized", False)),
                "selected_strength_name": result.get("selected_strength_name"),
                "selected": result.get("selected"),
                "gates": result.get("gates"),
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
