#!/usr/bin/env python3
"""Analyze selected checkpoint 16 on remaining and combined open scene fit rows."""

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
    analyze_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_progression as runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_progression_result.v1"


def canonical_sha256(value: Any) -> str:
    return runner.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return runner.sha256_file(path)


def read_progression_outputs(
    root: Path,
    *,
    remaining_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, Mapping[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {int(row["source_index"]): row for row in remaining_rows}
    outputs = {condition: {} for condition in runner.CONDITIONS}
    bindings: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    runner_sha256 = sha256_file(Path(runner.__file__))
    runtime_gate_hash: str | None = None
    donor_mapping_sha256: str | None = None
    for shard_index in range(runner.WORLD_SIZE):
        shard_dir = root / f"shard-{shard_index}"
        binding_path = shard_dir / "input_binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        required_binding = {
            "schema": runner.INPUT_SCHEMA,
            "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
            "selection_receipt_sha256": runner.SELECTION_RECEIPT_SHA256,
            "selection_file_sha256": runner.SELECTION_FILE_SHA256,
            "selected_checkpoint_step": runner.SELECTED_STEP,
            "selected_gate_state_sha256": runner.SELECTED_GATE_STATE_SHA256,
            "selected_patch_sha256": runner.SELECTED_PATCH_SHA256,
            "remaining_payload_sha256": runner.REMAINING_PAYLOAD_SHA256,
            "remaining_rows": runner.REMAINING_ROWS,
            "conditions": list(runner.CONDITIONS),
            "shard_index": shard_index,
            "world_size": runner.WORLD_SIZE,
            "runner_sha256": runner_sha256,
            "protected_splits_opened": [],
        }
        if any(binding.get(key) != value for key, value in required_binding.items()):
            raise ValueError(f"Scene contrast progression binding differs: {binding_path}")
        current_donor_hash = str(binding.get("donor_mapping_payload_sha256"))
        if donor_mapping_sha256 is None:
            donor_mapping_sha256 = current_donor_hash
        elif donor_mapping_sha256 != current_donor_hash:
            raise ValueError("Scene contrast progression donor binding differs across shards")
        bindings.append(
            {"path": str(binding_path), "sha256": sha256_file(binding_path), "payload": binding}
        )
        shard_expected = {
            source_index
            for source_index in expected
            if source_index % runner.WORLD_SIZE == shard_index
        }
        for condition in runner.CONDITIONS:
            path = shard_dir / f"{condition}.jsonl"
            records = probe_analysis.read_jsonl(path)
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": len(records),
                    "condition": condition,
                    "shard_index": shard_index,
                }
            )
            seen: set[int] = set()
            for record in records:
                source_index = int(record["source_index"])
                current_runtime_hash = str(record.get("runtime_gate_state_sha256"))
                if runtime_gate_hash is None:
                    runtime_gate_hash = current_runtime_hash
                required = {
                    "schema": runner.SCHEMA,
                    "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
                    "selection_receipt_sha256": runner.SELECTION_RECEIPT_SHA256,
                    "checkpoint_step": runner.SELECTED_STEP,
                    "gate_state_sha256": runner.SELECTED_GATE_STATE_SHA256,
                    "condition": condition,
                    "shard_index": shard_index,
                    "world_size": runner.WORLD_SIZE,
                    "source_index": source_index,
                }
                if (
                    source_index not in shard_expected
                    or source_index in seen
                    or current_runtime_hash != runtime_gate_hash
                    or record.get("row_sha256") != expected[source_index]["row_sha256"]
                    or any(record.get(key) != value for key, value in required.items())
                ):
                    raise ValueError(
                        f"Scene contrast progression output differs: {condition}:{source_index}"
                    )
                if condition == "matched_donor_state" and (
                    record.get("donor_source_index") == source_index
                    or not isinstance(record.get("absolute_write_token_delta"), int)
                ):
                    raise ValueError(f"Scene contrast progression donor differs: {source_index}")
                seen.add(source_index)
                outputs[condition][source_index] = record
            if seen != shard_expected:
                raise ValueError(
                    f"Incomplete scene contrast progression shard: {condition}:{shard_index}"
                )
    for condition in runner.CONDITIONS:
        if set(outputs[condition]) != set(expected):
            raise ValueError(f"Incomplete scene contrast progression condition: {condition}")
    return outputs, bindings, artifacts


def validate_probe_result(probe_root: Path) -> Mapping[str, Any]:
    result = probe.validate_signed_json(
        probe_root / "result.json",
        description="Scene contrast probe result",
    )
    selection = runner.validate_selection(probe_root / "selection.json")
    if (
        result.get("schema") != probe_analysis.SCHEMA
        or result["receipt"].get("payload_sha256")
        != selection["result"]["receipt_payload_sha256"]
        or result.get("selected_checkpoint_step") != runner.SELECTED_STEP
        or result.get("passed") is not True
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Scene contrast progression probe result differs")
    return result


def analyze(
    *,
    input_root: Path,
    probe_root: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    runner.validate_protocol()
    probe_result = validate_probe_result(probe_root)
    runner.selected_manifest(training_root)
    rows = causal.load_rows(dataset_root)
    probe_rows = probe.selected_probe_rows(rows)
    remaining_rows = runner.progression_rows(rows)
    fit_rows = [*probe_rows, *remaining_rows]
    probe_indices = tuple(int(row["source_index"]) for row in probe_rows)
    remaining_indices = tuple(int(row["source_index"]) for row in remaining_rows)
    fit_indices = tuple(int(row["source_index"]) for row in fit_rows)
    all_gold, hashes = shared.gold_and_hashes(rows)
    remaining_outputs, bindings, artifacts = read_progression_outputs(
        input_root,
        remaining_rows=remaining_rows,
    )
    probe_outputs, probe_bindings, probe_artifacts = probe_analysis.read_candidate_outputs(
        probe_root,
        selected_rows=probe_rows,
    )
    v9_records, v9_artifacts = shared.read_reference_condition(reference_root, "memory", hashes)
    v9_predictions = shared.predictions_from_records(v9_records)
    remaining_predictions = {
        condition: shared.predictions_from_records(records)
        for condition, records in remaining_outputs.items()
    }
    selected_probe_outputs = probe_outputs[runner.SELECTED_STEP]
    probe_predictions = {
        condition: shared.predictions_from_records(records)
        for condition, records in selected_probe_outputs.items()
    }
    combined_predictions: dict[str, dict[int, set[int] | None]] = {}
    for condition in runner.CONDITIONS:
        combined = dict(probe_predictions[condition])
        combined.update(remaining_predictions[condition])
        combined_predictions[condition] = combined
    remaining_result = probe_analysis.candidate_result(
        step=runner.SELECTED_STEP,
        predictions=remaining_predictions,
        v9_predictions=v9_predictions,
        gold=all_gold,
        indices=remaining_indices,
    )
    combined_result = probe_analysis.candidate_result(
        step=runner.SELECTED_STEP,
        predictions=combined_predictions,
        v9_predictions=v9_predictions,
        gold=all_gold,
        indices=fit_indices,
    )
    if tuple(sorted(set(probe_indices) | set(remaining_indices))) != tuple(sorted(fit_indices)):
        raise ValueError("Scene contrast progression combined fit coverage differs")
    passed = bool(remaining_result["passed"] and combined_result["passed"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_payload_sha256": runner.PROTOCOL_PAYLOAD_SHA256,
        "selection_receipt_sha256": runner.SELECTION_RECEIPT_SHA256,
        "runner_sha256": sha256_file(Path(runner.__file__)),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "selected_checkpoint_step": runner.SELECTED_STEP,
        "selected_gate_state_sha256": runner.SELECTED_GATE_STATE_SHA256,
        "locked_probe": {
            "rows": len(probe_indices),
            "source_indices": list(probe_indices),
            "signed_result_metrics": next(
                item
                for item in probe_result["candidates"]
                if int(item["checkpoint_step"]) == runner.SELECTED_STEP
            ),
        },
        "remaining_fit": {
            "rows": len(remaining_indices),
            "source_indices": list(remaining_indices),
            "evaluation": remaining_result,
        },
        "combined_fit": {
            "rows": len(fit_indices),
            "source_indices": list(fit_indices),
            "evaluation": combined_result,
        },
        "passed": passed,
        "multitask_preservation_authorized": passed,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
        "provenance": {
            "input_bindings": bindings,
            "progression_outputs": artifacts,
            "probe_input_bindings": probe_bindings,
            "probe_outputs": probe_artifacts,
            "frozen_v9_outputs": v9_artifacts,
            "probe_result": {
                "path": str(probe_root / "result.json"),
                "sha256": sha256_file(probe_root / "result.json"),
                "receipt_payload_sha256": probe_result["receipt"]["payload_sha256"],
            },
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_contrast_progression_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    if output.exists():
        raise ValueError(f"Scene contrast progression output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_root=args.input_root.expanduser().resolve(strict=True),
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
                "remaining_fit_f1": result["remaining_fit"]["evaluation"]["metrics"][
                    "correct_state"
                ]["micro_f1"],
                "combined_fit_f1": result["combined_fit"]["evaluation"]["metrics"][
                    "correct_state"
                ]["micro_f1"],
                "receipt": result["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
