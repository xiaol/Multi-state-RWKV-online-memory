#!/usr/bin/env python3
"""Generate one checkpoint-16 narrative preservation shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    analyze_natural_memory_native_routed_benchmark as routed_analysis,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_routed_benchmark as routed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)


SCHEMA = "rwkv_ms_natural_memory_native_multitask_preservation_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_multitask_preservation_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_multitask_preservation_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "5e97c578ee5e89e7f8de904e316513b03ab03cd6d5cae7fc25d64a03c374c920"
WORLD_SIZE = 4
SELECTED_STEP = 16
NARRATIVE_ROWS = 114
NARRATIVE_PAYLOAD_SHA256 = "5dbd1295a94ae3c8015e9ebb7446e8c2dd9dc25d1a98d2082f4321e7bba853c9"
PROGRESSION_RECEIPT_SHA256 = "23bc133d82590890308ac5b0779e54427f51fbee615d941393a023538be80b2b"
PROGRESSION_FILE_SHA256 = "af852ce316d83fb90b18bbd97fb302cdb1fe99b305c96736478759143d897cb2"
REFERENCE_RECEIPT_SHA256 = "26a2248976ff009804744a19a738fd2124061cf6d909bdc74c9e7c040098c091"
REFERENCE_FILE_SHA256 = "36549b036f104bc665d367b65668aef80ad94b98bdf5c63d441cb0d6ef9b422f"
SELECTED_GATE_STATE_SHA256 = "5b0670683046e9701c24171a5c9d8cfc58e1078f25b72806662732260bab7d4f"
SELECTED_PATCH_SHA256 = "c7bd4f6a396c06404cd884f3f6ff92ad5d891621fe295a3e91e9b33d73d4834a"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return probe.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Multitask preservation protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Multitask preservation protocol hash differs")
    return value


def validate_signed_result(
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


def validate_progression(path: Path) -> Mapping[str, Any]:
    value = validate_signed_result(
        path,
        description="Scene contrast progression result",
        expected_file_sha256=PROGRESSION_FILE_SHA256,
        expected_receipt_sha256=PROGRESSION_RECEIPT_SHA256,
    )
    required = {
        "schema": "rwkv_ms_natural_memory_native_scene_contrast_progression_result.v1",
        "selected_checkpoint_step": SELECTED_STEP,
        "selected_gate_state_sha256": SELECTED_GATE_STATE_SHA256,
        "passed": True,
        "multitask_preservation_authorized": True,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("Scene contrast progression authorization differs")
    return value


def validate_reference(root: Path) -> Mapping[str, Any]:
    routed_analysis.validate_input_bindings(root)
    value = validate_signed_result(
        root / "result.json",
        description="Routed native benchmark result",
        expected_file_sha256=REFERENCE_FILE_SHA256,
        expected_receipt_sha256=REFERENCE_RECEIPT_SHA256,
    )
    if (
        value.get("schema") != routed_analysis.SCHEMA
        or value.get("scope", {}).get("protected_splits_opened") != []
    ):
        raise ValueError("Routed native benchmark reference differs")
    return value


def selected_narrative_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if int(row["line_index"]) >= routed.SELECTION_ROWS]
    payload = [
        {"line_index": int(row["line_index"]), "row_sha256": str(row["row_sha256"])}
        for row in selected
    ]
    if len(selected) != NARRATIVE_ROWS or canonical_sha256(payload) != NARRATIVE_PAYLOAD_SHA256:
        raise ValueError("Narrative preservation row binding differs")
    return selected


def selected_manifest(training_root: Path) -> Mapping[str, Any]:
    manifests = probe.validate_training_root(training_root)
    manifest = next(item for item in manifests if int(item["step"]) == SELECTED_STEP)
    if (
        manifest.get("gate_state_sha256") != SELECTED_GATE_STATE_SHA256
        or manifest.get("patch_file", {}).get("sha256") != SELECTED_PATCH_SHA256
    ):
        raise ValueError("Multitask preservation checkpoint differs")
    return manifest


def reference_artifacts(root: Path) -> list[Mapping[str, Any]]:
    artifacts: list[Mapping[str, Any]] = []
    for task, conditions in (("attribution", ("base",)), ("narrative", ("base", "memory"))):
        for condition in conditions:
            _, condition_artifacts = routed_analysis.collect_condition(
                root,
                task=task,
                condition=condition,
            )
            artifacts.extend(
                {"task": task, "condition": condition, **artifact}
                for artifact in condition_artifacts
            )
    return artifacts


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    reference_root: Path,
    progression_result: Path,
    training_root: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    shard_index: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    dataset_manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("receipt", {}).get("payload_sha256") != routed.NATIVE_DATASET_RECEIPT_SHA256:
        raise ValueError("Native development dataset receipt differs")
    if sha256_file(base_model / "config.json") != training.BASE_CONFIG_SHA256:
        raise ValueError("Multitask preservation base config differs")
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != training.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Multitask preservation V9 adapter differs")
    validate_progression(progression_result)
    validate_reference(reference_root)
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": training.BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_files_sha256": training.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "dataset_receipt_payload_sha256": routed.NATIVE_DATASET_RECEIPT_SHA256,
        "narrative_payload_sha256": NARRATIVE_PAYLOAD_SHA256,
        "narrative_rows": len(rows),
        "reference_root": str(reference_root),
        "reference_result_sha256": REFERENCE_FILE_SHA256,
        "reference_result_receipt_sha256": REFERENCE_RECEIPT_SHA256,
        "reference_artifacts": reference_artifacts(reference_root),
        "progression_result": str(progression_result),
        "progression_result_sha256": PROGRESSION_FILE_SHA256,
        "progression_result_receipt_sha256": PROGRESSION_RECEIPT_SHA256,
        "training_root": str(training_root),
        "selected_checkpoint_step": SELECTED_STEP,
        "selected_gate_state_sha256": manifest["gate_state_sha256"],
        "selected_patch_sha256": manifest["patch_file"]["sha256"],
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Multitask preservation binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            record = json.loads(raw_line)
            index = int(record["line_index"])
            if index in records:
                raise ValueError(f"Duplicate narrative preservation row: {path}:{index}")
            records[index] = record
    return records


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
) -> None:
    expected = {int(row["line_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError("Unexpected narrative preservation rows")
    for index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "task": "narrative",
            "condition": "checkpoint16_correct_state",
            "checkpoint_step": SELECTED_STEP,
            "gate_state_sha256": SELECTED_GATE_STATE_SHA256,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "line_index": index,
            "row_sha256": expected[index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed narrative preservation row differs: {index}")


def append_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    shard_index: int,
    device: str,
    runtime_gate_state_sha256: str,
) -> Mapping[str, Any]:
    result = routed.evaluator.generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=list(row["messages"]),
        max_new_tokens=int(routed.TASKS["narrative"]["max_new_tokens"]),
        device=device,
        online_memory_protocol="write_then_read",
    )
    return {
        "schema": SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "task": "narrative",
        "condition": "checkpoint16_correct_state",
        "checkpoint_step": SELECTED_STEP,
        "gate_state_sha256": SELECTED_GATE_STATE_SHA256,
        "runtime_gate_state_sha256": runtime_gate_state_sha256,
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "line_index": row["line_index"],
        "row_sha256": row["row_sha256"],
        "prediction": routed.recovery.recover_narrative(result["parsed_json"]),
        "raw_generation": result["raw_generation"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "hit_max_new_tokens": result["hit_max_new_tokens"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--progression-result", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Multitask preservation requires a valid four-way shard")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    progression_result = args.progression_result.expanduser().resolve(strict=True)
    training_root = args.training_root.expanduser().resolve(strict=True)
    output_dir = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}"
    all_rows = selected_narrative_rows(routed.load_rows(dataset_root)["narrative"])
    shard_rows = [
        row for row in all_rows if int(row["line_index"]) % WORLD_SIZE == args.shard_index
    ]
    manifest = selected_manifest(training_root)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        reference_root=reference_root,
        progression_result=progression_result,
        training_root=training_root,
        manifest=manifest,
        rows=all_rows,
        shard_index=args.shard_index,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    output_path = output_dir / "narrative.checkpoint16.jsonl"
    existing = read_completed(output_path)
    validate_resume(existing, shard_rows, shard_index=args.shard_index)
    if len(existing) == len(shard_rows):
        print(f"MULTITASK_PRESERVATION_SHARD_COMPLETE shard={args.shard_index}", flush=True)
        return 0
    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    loaded = probe.load_gate_patch(
        model,
        patch_path=training_root / f"checkpoint-{SELECTED_STEP}" / "gate_patch.pt",
        manifest=manifest,
    )
    for ordinal, row in enumerate(shard_rows, start=1):
        index = int(row["line_index"])
        if index in existing:
            continue
        record = generate_record(
            model,
            tokenizer,
            row,
            shard_index=args.shard_index,
            device=args.device,
            runtime_gate_state_sha256=str(loaded["runtime_gate_state_sha256"]),
        )
        append_record(output_path, record)
        print(
            f"MULTITASK_PRESERVATION_PROGRESS shard={args.shard_index} "
            f"row={index} ordinal={ordinal}/{len(shard_rows)}",
            flush=True,
        )
    print(f"MULTITASK_PRESERVATION_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
