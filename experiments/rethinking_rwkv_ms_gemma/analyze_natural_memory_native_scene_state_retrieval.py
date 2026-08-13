#!/usr/bin/env python3
"""Analyze and sign the two-phase native scene state-retrieval study."""

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
    analyze_novel_agent_eval as recovery,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    prepare_natural_memory_native_scene_state_retrieval as mapping_builder,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal_runner,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_state_retrieval as retrieval_runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_state_retrieval_result.v1"
EXPECTED_RUNNER_SHA256 = (
    "3dadaa229bfd1210a0c002d15e39b27e27d547a773cf630f67b5f083d98a9690"
)
ZERO_ARTIFACT_SHA256 = (
    "fc6cd8d3535a1d05d17b3f9a230b04c058312f377a60652233b8d34c4abbd373",
    "4a8a2cea9d5095f401210f199c3dc600183f40be74cf5869f69fd0493c9b6eb8",
    "3d6fed9994d0c9c68796d1866701408093420ba7cb15e87b068f4630f2141817",
    "0572edc842c1014d3e6933642de114cf424b11309ddb29b96c907d1b0d510300",
)


def canonical_sha256(value: Any) -> str:
    return mapping_builder.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return mapping_builder.sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"State-retrieval record is not an object: {path}")
                records.append(value)
    return records


def gold_and_hashes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, set[int]], dict[int, str]]:
    gold: dict[int, set[int]] = {}
    hashes: dict[int, str] = {}
    for row in rows:
        source_index = int(row["source_index"])
        if source_index < mapping_builder.EXCLUDED_TARGET_ROWS:
            continue
        gold[source_index] = recovery.strict_gold_boundaries(row["gold"])
        hashes[source_index] = str(row["row_sha256"])
    return gold, hashes


def phase_indices(hashes: Mapping[int, str], phase: str) -> tuple[int, ...]:
    indices = tuple(
        source_index
        for source_index in sorted(hashes)
        if mapping_builder.partition_for_hash(hashes[source_index]) == phase
    )
    expected_count = 289 if phase == "fit" else 68
    if len(indices) != expected_count:
        raise ValueError(f"State-retrieval {phase} partition count differs")
    return indices


def read_reference_condition(
    root: Path,
    condition: str,
    hashes: Mapping[int, str],
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index in range(retrieval_runner.WORLD_SIZE):
        path = root / f"shard-{shard_index}" / f"scene.{condition}.jsonl"
        digest = sha256_file(path)
        expected = causal_runner.REFERENCE_ARTIFACT_SHA256[condition][shard_index]
        if digest != expected:
            raise ValueError(f"State-retrieval reference hash differs: {path}")
        rows = read_jsonl(path)
        artifacts.append({"path": str(path), "rows": len(rows), "sha256": digest})
        for record in rows:
            source_index = int(record["line_index"])
            if source_index in records:
                raise ValueError(f"Duplicate state-retrieval reference row: {source_index}")
            if record.get("row_sha256") != hashes.get(source_index):
                raise ValueError(f"State-retrieval reference row hash differs: {source_index}")
            records[source_index] = record
    if set(records) != set(hashes):
        raise ValueError(f"Incomplete state-retrieval reference: {condition}")
    return records, artifacts


def read_zero_condition(
    root: Path,
    hashes: Mapping[int, str],
) -> tuple[dict[int, Mapping[str, Any]], list[dict[str, Any]]]:
    records: dict[int, Mapping[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for shard_index, expected_hash in enumerate(ZERO_ARTIFACT_SHA256):
        path = root / f"shard-{shard_index}" / "zero_state.jsonl"
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ValueError(f"State-retrieval zero-state hash differs: {path}")
        rows = read_jsonl(path)
        artifacts.append({"path": str(path), "rows": len(rows), "sha256": digest})
        for record in rows:
            source_index = int(record["source_index"])
            if source_index in records:
                raise ValueError(f"Duplicate state-retrieval zero-state row: {source_index}")
            if record.get("row_sha256") != hashes.get(source_index):
                raise ValueError(f"State-retrieval zero-state row hash differs: {source_index}")
            records[source_index] = record
    if set(records) != set(hashes):
        raise ValueError("Incomplete state-retrieval zero-state reference")
    return records, artifacts


def prediction_set(record: Mapping[str, Any]) -> set[int] | None:
    prediction = record.get("prediction")
    if not isinstance(prediction, list):
        return None
    return {int(value) for value in prediction}


def predictions_from_records(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[int, set[int] | None]:
    return {source_index: prediction_set(record) for source_index, record in records.items()}


def metrics_from_sets(
    predictions: Mapping[int, set[int] | None],
    gold: Mapping[int, set[int]],
    indices: Sequence[int],
) -> Mapping[str, Any]:
    tp = fp = fn = covered = 0
    for source_index in indices:
        prediction = predictions[source_index]
        covered += int(prediction is not None)
        predicted = set() if prediction is None else prediction
        expected = gold[source_index]
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        "rows": len(indices),
        "coverage": covered / len(indices),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": 0.0 if denominator == 0 else 2 * tp / denominator,
    }


def output_change_fraction(
    left: Mapping[int, set[int] | None],
    right: Mapping[int, set[int] | None],
    indices: Sequence[int],
) -> float:
    return sum(left[source_index] != right[source_index] for source_index in indices) / len(indices)


def mapping_by_index() -> tuple[Mapping[str, Any], dict[int, Mapping[str, Any]]]:
    mapping = retrieval_runner.load_mapping()
    records = {
        int(record["target_source_index"]): record for record in mapping["records"]
    }
    return mapping, records


def read_retrieval_outputs(
    root: Path,
    *,
    phase: str,
    methods: Sequence[str],
    indices: Sequence[int],
    hashes: Mapping[int, str],
    mapping_records: Mapping[int, Mapping[str, Any]],
    selection: Mapping[str, Any] | None,
    selection_path: Path | None,
) -> tuple[
    dict[str, dict[int, Mapping[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    expected_indices = set(indices)
    outputs: dict[str, dict[int, Mapping[str, Any]]] = {method: {} for method in methods}
    artifacts: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    bindings: list[dict[str, Any]] = []
    for shard_index in range(retrieval_runner.WORLD_SIZE):
        shard_dir = root / phase / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_binding = {
            "schema": retrieval_runner.INPUT_SCHEMA,
            "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
            "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
            "mapping_payload_sha256": retrieval_runner.MAPPING_PAYLOAD_SHA256,
            "phase": phase,
            "shard_index": shard_index,
            "world_size": retrieval_runner.WORLD_SIZE,
            "methods": list(methods),
            "runner_sha256": EXPECTED_RUNNER_SHA256,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ValueError(f"State-retrieval input binding differs: {binding_path}")
        if phase == "fit":
            if any(
                binding.get(key) is not None
                for key in (
                    "selection_path",
                    "selection_file_sha256",
                    "selection_payload_sha256",
                )
            ):
                raise ValueError("State-retrieval fit binding contains a selection")
        else:
            if selection is None or selection_path is None:
                raise ValueError("State-retrieval holdout selection is missing")
            if binding.get("selection_file_sha256") != sha256_file(selection_path):
                raise ValueError("State-retrieval holdout selection file hash differs")
            if binding.get("selection_payload_sha256") != selection["receipt"]["payload_sha256"]:
                raise ValueError("State-retrieval holdout selection payload differs")
        bindings.append(
            {
                "path": str(binding_path),
                "sha256": sha256_file(binding_path),
                "payload": binding,
            }
        )
        for method in methods:
            path = shard_dir / f"{method}.jsonl"
            rows = read_jsonl(path)
            artifacts[method].append(
                {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
            )
            for record in rows:
                source_index = int(record["target_source_index"])
                expected_record = {
                    "schema": retrieval_runner.SCHEMA,
                    "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
                    "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
                    "mapping_payload_sha256": retrieval_runner.MAPPING_PAYLOAD_SHA256,
                    "phase": phase,
                    "method": method,
                    "shard_index": shard_index,
                    "world_size": retrieval_runner.WORLD_SIZE,
                    "target_row_sha256": hashes[source_index],
                }
                if any(record.get(key) != value for key, value in expected_record.items()):
                    raise ValueError(f"State-retrieval output binding differs: {path}:{source_index}")
                if source_index not in expected_indices or source_index % 4 != shard_index:
                    raise ValueError(f"State-retrieval output shard differs: {path}:{source_index}")
                mapping_record = mapping_records[source_index]
                method_mapping = mapping_record["methods"][method]
                mapping_expected = {
                    "bank_source_index": method_mapping["bank_source_index"],
                    "bank_row_sha256": method_mapping["bank_row_sha256"],
                    "target_write_tokens": mapping_record["target_write_tokens"],
                    "bank_write_tokens": method_mapping["bank_write_tokens"],
                    "absolute_write_token_delta": method_mapping["absolute_write_token_delta"],
                    "char_tfidf_cosine": method_mapping["char_tfidf_cosine"],
                    "selection_score": method_mapping["selection_score"],
                }
                if any(record.get(key) != value for key, value in mapping_expected.items()):
                    raise ValueError(f"State-retrieval mapping output differs: {path}:{source_index}")
                if source_index in outputs[method]:
                    raise ValueError(f"Duplicate state-retrieval method output: {method}:{source_index}")
                outputs[method][source_index] = record
    for method in methods:
        if set(outputs[method]) != expected_indices:
            raise ValueError(f"Incomplete state-retrieval method output: {phase}:{method}")
    return outputs, artifacts, bindings


def gates_for_method(
    candidate_metrics: Mapping[str, Any],
    candidate_predictions: Mapping[int, set[int] | None],
    correct_metrics: Mapping[str, Any],
    correct_predictions: Mapping[int, set[int] | None],
    zero_metrics: Mapping[str, Any],
    indices: Sequence[int],
) -> Mapping[str, Any]:
    candidate_f1 = float(candidate_metrics["micro_f1"])
    correct_f1 = float(correct_metrics["micro_f1"])
    zero_f1 = float(zero_metrics["micro_f1"])
    change_fraction = output_change_fraction(
        candidate_predictions,
        correct_predictions,
        indices,
    )
    gates: dict[str, Any] = {
        "coverage_at_least_0.95": float(candidate_metrics["coverage"]) >= 0.95,
        "minus_correct_state_micro_f1_at_least_0.005": candidate_f1 - correct_f1 >= 0.005,
        "minus_zero_state_micro_f1_at_least_0.02": candidate_f1 - zero_f1 >= 0.02,
        "paired_output_change_fraction_vs_correct_state_at_least_0.05": change_fraction >= 0.05,
    }
    gates["passed"] = all(gates.values())
    return {
        "candidate_minus_correct_state_micro_f1": candidate_f1 - correct_f1,
        "candidate_minus_zero_state_micro_f1": candidate_f1 - zero_f1,
        "paired_output_change_fraction_vs_correct_state": change_fraction,
        "gates": gates,
    }


def result_payload(
    *,
    phase: str,
    methods: Sequence[str],
    indices: Sequence[int],
    gold: Mapping[int, set[int]],
    baseline_records: Mapping[str, Mapping[int, Mapping[str, Any]]],
    baseline_artifacts: Mapping[str, Any],
    retrieval_records: Mapping[str, Mapping[int, Mapping[str, Any]]],
    retrieval_artifacts: Mapping[str, Any],
    input_bindings: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, bool]:
    baseline_predictions = {
        name: predictions_from_records(records)
        for name, records in baseline_records.items()
    }
    baseline_metrics = {
        name: metrics_from_sets(predictions, gold, indices)
        for name, predictions in baseline_predictions.items()
    }
    method_predictions = {
        method: predictions_from_records(records)
        for method, records in retrieval_records.items()
    }
    method_metrics = {
        method: metrics_from_sets(predictions, gold, indices)
        for method, predictions in method_predictions.items()
    }
    if phase == "fit":
        selected = min(
            methods,
            key=lambda method: (
                -float(method_metrics[method]["micro_f1"]),
                -float(method_metrics[method]["precision"]),
                -float(method_metrics[method]["recall"]),
                method,
            ),
        )
    else:
        if selection is None:
            raise ValueError("State-retrieval holdout selection is missing")
        selected = str(selection["selected_method"])
        if tuple(methods) != (selected,):
            raise ValueError("State-retrieval holdout method differs from selection")
    gate_result = gates_for_method(
        method_metrics[selected],
        method_predictions[selected],
        baseline_metrics["correct_state"],
        baseline_predictions["correct_state"],
        baseline_metrics["zero_state"],
        indices,
    )
    passed = bool(gate_result["gates"]["passed"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
        "mapping_payload_sha256": retrieval_runner.MAPPING_PAYLOAD_SHA256,
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
        "baselines": baseline_metrics,
        "candidate_methods": method_metrics,
        "selected_method": selected,
        "selected": {
            "metrics": method_metrics[selected],
            **gate_result,
        },
        "passed": passed,
        "holdout_authorized": phase == "fit" and passed,
        "accepted_validation_decoder_changed": False,
        "selection_receipt_payload_sha256": (
            None if selection is None else selection["receipt"]["payload_sha256"]
        ),
        "provenance": {
            "input_bindings": list(input_bindings),
            "baselines": dict(baseline_artifacts),
            "retrieval": dict(retrieval_artifacts),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": f"canonical_state_retrieval_{phase}_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    return result, selected, passed


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"State-retrieval analysis output must be fresh: {path}")
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
    selected: str,
) -> Mapping[str, Any]:
    if not result["passed"] or not result["holdout_authorized"]:
        raise ValueError("State-retrieval selection requires a passing phase one")
    value: dict[str, Any] = {
        "schema": retrieval_runner.SELECTION_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": mapping_builder.PROTOCOL_PAYLOAD_SHA256,
        "amendment_payload_sha256": mapping_builder.AMENDMENT_PAYLOAD_SHA256,
        "mapping_payload_sha256": retrieval_runner.MAPPING_PAYLOAD_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "analyzer_sha256": sha256_file(Path(__file__)),
        "phase_one_result": {
            "path": result_path.name,
            "file_sha256": sha256_file(result_path),
            "receipt_payload_sha256": result["receipt"]["payload_sha256"],
        },
        "selected_method": selected,
        "selection_ordering": "micro-F1 descending, precision descending, recall descending, candidate name ascending",
        "selected_phase_one_metrics": result["selected"],
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
    parser.add_argument("--phase", choices=retrieval_runner.PHASES, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mapping_builder.validate_protocol()
    if sha256_file(retrieval_runner.Path(retrieval_runner.__file__)) != EXPECTED_RUNNER_SHA256:
        raise ValueError("State-retrieval runner hash differs")
    input_root = args.input_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    zero_root = args.zero_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    selection_path: Path | None = None
    selection: Mapping[str, Any] | None = None
    if args.phase == "fit":
        if args.selection is not None:
            raise ValueError("State-retrieval fit analysis forbids a selection")
        methods = mapping_builder.CANDIDATE_METHODS
    else:
        if args.selection is None or args.selection_output is not None:
            raise ValueError("State-retrieval holdout analysis requires only --selection")
        selection_path = args.selection.expanduser().resolve(strict=True)
        selection = retrieval_runner.validate_selection(
            selection_path,
            runner_sha256=EXPECTED_RUNNER_SHA256,
        )
        methods = (str(selection["selected_method"]),)
    rows = causal_runner.load_rows(dataset_root)
    gold, hashes = gold_and_hashes(rows)
    indices = phase_indices(hashes, args.phase)
    _, retrieval_mapping = mapping_by_index()
    base_records, base_artifacts = read_reference_condition(
        reference_root,
        "base",
        hashes,
    )
    memory_records, memory_artifacts = read_reference_condition(
        reference_root,
        "memory",
        hashes,
    )
    zero_records, zero_artifacts = read_zero_condition(zero_root, hashes)
    outputs, retrieval_artifacts, bindings = read_retrieval_outputs(
        input_root,
        phase=args.phase,
        methods=methods,
        indices=indices,
        hashes=hashes,
        mapping_records=retrieval_mapping,
        selection=selection,
        selection_path=selection_path,
    )
    result, selected, passed = result_payload(
        phase=args.phase,
        methods=methods,
        indices=indices,
        gold=gold,
        baseline_records={
            "base": base_records,
            "correct_state": memory_records,
            "zero_state": zero_records,
        },
        baseline_artifacts={
            "base": base_artifacts,
            "correct_state": memory_artifacts,
            "zero_state": zero_artifacts,
        },
        retrieval_records=outputs,
        retrieval_artifacts=retrieval_artifacts,
        input_bindings=bindings,
        selection=selection,
    )
    write_json(output, result)
    if args.phase == "fit" and passed:
        if args.selection_output is None:
            raise ValueError("Passing state-retrieval fit analysis requires --selection-output")
        write_selection(
            args.selection_output.expanduser().resolve(),
            result_path=output,
            result=result,
            selected=selected,
        )
    elif args.selection_output is not None:
        raise ValueError("State-retrieval selection output is unauthorized")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "selected_method": selected,
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
