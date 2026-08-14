#!/usr/bin/env python3
"""Evaluate selected contrast checkpoint 16 on the remaining open scene fit rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_probe as probe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_strength_calibration as strength,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_progression_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_progression_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_contrast_progression_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "e9945957bf942d646152d79a471ae561eca0ea4bafc511d48ea84cffce62746e"
WORLD_SIZE = 4
SELECTED_STEP = 16
CONDITIONS = probe.CONDITIONS
REMAINING_ROWS = 220
REMAINING_PAYLOAD_SHA256 = "0493e75da858d4ddebba580cc7b5aaaa32249527e5e44502e6ff06591cd82d09"
FIT_PAYLOAD_SHA256 = "5743ac67571f8359585402fb31579fef801f5804cc2f3b9874dccad795350c38"
SELECTION_RECEIPT_SHA256 = "64ad8298040e3935f1024a6fe2bb7cfaea96bdabd91237633be95f2d0a134c8c"
SELECTION_FILE_SHA256 = "02a55b13f77020cea3b21e66ba3600587560701bf467c151db9ac178c54d8d55"
SELECTED_GATE_STATE_SHA256 = "5b0670683046e9701c24171a5c9d8cfc58e1078f25b72806662732260bab7d4f"
SELECTED_PATCH_SHA256 = "c7bd4f6a396c06404cd884f3f6ff92ad5d891621fe295a3e91e9b33d73d4834a"


def canonical_sha256(value: Any) -> str:
    return probe.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return probe.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Scene contrast progression protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Scene contrast progression protocol hash differs")
    return value


def validate_selection(path: Path) -> Mapping[str, Any]:
    value = probe.validate_signed_json(path, description="Scene contrast selection")
    required = {
        "schema": "rwkv_ms_natural_memory_native_scene_contrast_probe_selection.v1",
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "selected_checkpoint_step": SELECTED_STEP,
        "selected_gate_state_sha256": SELECTED_GATE_STATE_SHA256,
        "selected_patch_sha256": SELECTED_PATCH_SHA256,
        "passed": True,
        "remaining_fit_evaluation_authorized": True,
        "publisher_validation_authorized": False,
        "publisher_test_authorized": False,
        "hard32_authorized": False,
        "unused_strength_holdout_authorized": False,
        "protected_splits_opened": [],
    }
    if (
        any(value.get(key) != expected for key, expected in required.items())
        or value["receipt"].get("payload_sha256") != SELECTION_RECEIPT_SHA256
        or sha256_file(path) != SELECTION_FILE_SHA256
    ):
        raise ValueError("Scene contrast progression selection binding differs")
    return value


def progression_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    probe_rows = probe.selected_probe_rows(rows)
    probe_indices = {int(row["source_index"]) for row in probe_rows}
    fit = [
        row
        for row in rows
        if int(row["source_index"]) >= 4
        and strength.partition_for_hash(str(row["row_sha256"])) == "fit"
    ]
    fit_payload = [
        {"source_index": int(row["source_index"]), "row_sha256": str(row["row_sha256"])}
        for row in fit
    ]
    remaining = [row for row in fit if int(row["source_index"]) not in probe_indices]
    remaining_payload = [
        {"source_index": int(row["source_index"]), "row_sha256": str(row["row_sha256"])}
        for row in remaining
    ]
    if len(fit) != 284 or canonical_sha256(fit_payload) != FIT_PAYLOAD_SHA256:
        raise ValueError("Scene contrast progression fit partition differs")
    if (
        len(remaining) != REMAINING_ROWS
        or canonical_sha256(remaining_payload) != REMAINING_PAYLOAD_SHA256
    ):
        raise ValueError("Scene contrast progression remaining partition differs")
    return remaining


def selected_manifest(training_root: Path) -> Mapping[str, Any]:
    manifests = probe.validate_training_root(training_root)
    manifest = next(item for item in manifests if int(item["step"]) == SELECTED_STEP)
    if (
        manifest.get("gate_state_sha256") != SELECTED_GATE_STATE_SHA256
        or manifest["patch_file"].get("sha256") != SELECTED_PATCH_SHA256
    ):
        raise ValueError("Scene contrast progression selected patch differs")
    return manifest


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
    selection_path: Path,
    manifest: Mapping[str, Any],
    donor_payload: Sequence[Mapping[str, Any]],
    shard_index: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != training.BASE_CONFIG_SHA256:
        raise ValueError("Scene contrast progression base config differs")
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != training.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Scene contrast progression V9 adapter differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "selection_file": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "base_model": str(base_model),
        "base_config_sha256": training.BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_files_sha256": training.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_file_sha256": causal.DATASET_SHA256,
        "reference_root": str(reference_root),
        "reference_artifacts": causal.reference_artifacts(reference_root),
        "training_root": str(training_root),
        "selected_checkpoint_step": SELECTED_STEP,
        "selected_gate_state_sha256": manifest["gate_state_sha256"],
        "selected_patch_sha256": manifest["patch_file"]["sha256"],
        "remaining_payload_sha256": REMAINING_PAYLOAD_SHA256,
        "remaining_rows": REMAINING_ROWS,
        "donor_mapping_payload_sha256": canonical_sha256(list(donor_payload)),
        "donor_mapping_rows": len(donor_payload),
        "conditions": list(CONDITIONS),
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "probe_runner_sha256": sha256_file(Path(probe.__file__)),
        "protected_splits_opened": [],
    }


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Scene contrast progression binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            record = json.loads(raw_line)
            source_index = int(record["source_index"])
            if source_index in records:
                raise ValueError(f"Duplicate scene contrast progression row: {path}:{source_index}")
            records[source_index] = record
    return records


def append_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected scene contrast progression rows: {condition}")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
            "checkpoint_step": SELECTED_STEP,
            "gate_state_sha256": SELECTED_GATE_STATE_SHA256,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source_index,
            "row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(
                f"Resumed scene contrast progression row differs: {condition}:{source_index}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
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
        raise ValueError("Scene contrast progression requires a valid four-way shard")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    training_root = args.training_root.expanduser().resolve(strict=True)
    selection_path = args.selection.expanduser().resolve(strict=True)
    output_dir = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}"
    validate_selection(selection_path)
    manifest = selected_manifest(training_root)
    rows = causal.load_rows(dataset_root)
    remaining = progression_rows(rows)
    shard_rows = [
        row for row in remaining if int(row["source_index"]) % WORLD_SIZE == args.shard_index
    ]
    tokenizer_for_mapping = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    token_counts = causal.write_prompt_token_counts(tokenizer_for_mapping, rows)
    donor_mapping = causal.build_donor_mapping(rows, token_counts)
    donor_payload = causal.donor_mapping_payload(rows, token_counts, donor_mapping)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        reference_root=reference_root,
        training_root=training_root,
        selection_path=selection_path,
        manifest=manifest,
        donor_payload=donor_payload,
        shard_index=args.shard_index,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    completed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        existing = read_completed(output_dir / f"{condition}.jsonl")
        validate_resume(
            existing,
            shard_rows,
            condition=condition,
            shard_index=args.shard_index,
        )
        completed[condition] = existing
    if all(len(value) == len(shard_rows) for value in completed.values()):
        print(f"SCENE_CONTRAST_PROGRESSION_SHARD_COMPLETE shard={args.shard_index}", flush=True)
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
    by_index = {int(row["source_index"]): row for row in rows}
    for ordinal, row in enumerate(shard_rows, start=1):
        source_index = int(row["source_index"])
        donor_index = donor_mapping[source_index]
        donor_row = by_index[donor_index]
        donor_delta = abs(token_counts[source_index] - token_counts[donor_index])
        for condition in CONDITIONS:
            if source_index in completed[condition]:
                continue
            record = dict(
                probe.generate_condition(
                    model,
                    tokenizer,
                    row,
                    donor_row=donor_row,
                    donor_token_delta=donor_delta,
                    step=SELECTED_STEP,
                    condition=condition,
                    gate_state_sha256=str(loaded["gate_state_sha256"]),
                    runtime_gate_state_sha256=str(loaded["runtime_gate_state_sha256"]),
                    shard_index=args.shard_index,
                    device=args.device,
                )
            )
            record.update(
                {
                    "schema": SCHEMA,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
                }
            )
            record.pop("training_result_receipt_sha256", None)
            append_record(output_dir / f"{condition}.jsonl", record)
            print(
                f"SCENE_CONTRAST_PROGRESSION_PROGRESS shard={args.shard_index} "
                f"condition={condition} row={source_index} "
                f"ordinal={ordinal}/{len(shard_rows)}",
                flush=True,
            )
    print(f"SCENE_CONTRAST_PROGRESSION_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
