#!/usr/bin/env python3
"""Evaluate contrast-dropout gate patches on the locked native scene probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer, reset_delta_state, set_delta_write_enabled  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_strength_calibration as strength,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_probe_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_probe_input.v1"
WORLD_SIZE = 4
CANDIDATE_STEPS = (8, 16, 32)
CONDITIONS = ("correct_state", "matched_donor_state", "zero_state")
PROBE_SALT = "rwkv-ms-contrast-probe-v1:"
PROBE_ROWS = 64
PROBE_PAYLOAD_SHA256 = "5c8a10f1e373ec6661481caf79bae340fdf5a92ab6afa9cf04e22f6bda254994"
TRAINING_RESULT_RECEIPT_SHA256 = "ef72604d84a379413e8c518b4d41ce4a844eaa536df05d30163ac6bf51f5c9cc"
REFERENCE_ROOT_NAME = "natural_memory_native_routed_benchmark_v1_r2"


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def selected_probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    fit = [
        row
        for row in rows
        if int(row["source_index"]) >= 4
        and strength.partition_for_hash(str(row["row_sha256"])) == "fit"
    ]
    if len(fit) != 284:
        raise ValueError("Scene contrast probe fit partition count differs")
    selected = sorted(
        fit,
        key=lambda row: (
            hashlib.sha256(
                (PROBE_SALT + str(row["row_sha256"])).encode("ascii")
            ).hexdigest(),
            int(row["source_index"]),
        ),
    )[:PROBE_ROWS]
    payload = [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
        }
        for row in selected
    ]
    if len(selected) != PROBE_ROWS or canonical_sha256(payload) != PROBE_PAYLOAD_SHA256:
        raise ValueError("Scene contrast probe selection hash differs")
    return selected


def validate_signed_json(path: Path, *, description: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{description} receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if canonical_sha256(unsigned) != receipt.get("payload_sha256"):
        raise ValueError(f"{description} receipt differs")
    return value


def validate_training_root(training_root: Path) -> list[Mapping[str, Any]]:
    result = validate_signed_json(
        training_root / "result.json",
        description="Scene contrast training result",
    )
    if (
        result.get("schema") != training.SCHEMA
        or result.get("status") != "training_complete_evaluation_pending"
        or result.get("protected_splits_opened") != []
        or result["receipt"].get("payload_sha256") != TRAINING_RESULT_RECEIPT_SHA256
    ):
        raise ValueError("Scene contrast training result binding differs")
    training_value = result.get("training")
    if not isinstance(training_value, Mapping) or training_value.get("non_gate_unchanged") is not True:
        raise ValueError("Scene contrast training isolation audit differs")
    manifests: list[Mapping[str, Any]] = []
    checkpoints = training_value.get("checkpoints")
    if not isinstance(checkpoints, list) or [item.get("step") for item in checkpoints] != list(CANDIDATE_STEPS):
        raise ValueError("Scene contrast checkpoint list differs")
    for step, result_manifest in zip(CANDIDATE_STEPS, checkpoints, strict=True):
        checkpoint_dir = training_root / f"checkpoint-{step}"
        manifest = validate_signed_json(
            checkpoint_dir / "manifest.json",
            description=f"Scene contrast checkpoint {step}",
        )
        patch_path = checkpoint_dir / "gate_patch.pt"
        patch_file = manifest.get("patch_file")
        if (
            manifest != result_manifest
            or manifest.get("schema") != training.PATCH_SCHEMA
            or manifest.get("protocol_payload_sha256") != training.PROTOCOL_PAYLOAD_SHA256
            or manifest.get("source_adapter_files_sha256") != training.V9_ADAPTER_FILES_SHA256
            or manifest.get("step") != step
            or not isinstance(patch_file, Mapping)
            or patch_file.get("sha256") != sha256_file(patch_path)
            or patch_file.get("bytes") != patch_path.stat().st_size
        ):
            raise ValueError(f"Scene contrast checkpoint {step} binding differs")
        manifests.append(manifest)
    return manifests


def load_gate_patch(
    model: torch.nn.Module,
    *,
    patch_path: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Scene contrast gate patch payload differs")
    state = payload["state_dict"]
    required = {
        "schema": training.PATCH_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "source_adapter_files_sha256": training.V9_ADAPTER_FILES_SHA256,
        "step": manifest["step"],
        "gate_state_sha256": manifest["gate_state_sha256"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("Scene contrast gate patch metadata differs")
    if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
        raise ValueError("Scene contrast gate patch state hash differs")
    parameters = dict(model.named_parameters())
    gate_names = {
        name
        for name in parameters
        if any(name.endswith(f".{family}") for family in training.GATE_FAMILIES)
    }
    if set(state) != gate_names or len(gate_names) != 126:
        raise ValueError("Scene contrast gate patch parameter names differ")
    with torch.no_grad():
        for name in sorted(gate_names):
            source = state[name]
            target = parameters[name]
            if source.shape != target.shape:
                raise ValueError(f"Scene contrast gate patch shape differs: {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    loaded = {
        name: parameters[name].detach().cpu().clone()
        for name in sorted(gate_names)
    }
    expected_runtime = {
        name: state[name].to(dtype=parameters[name].dtype).detach().cpu().clone()
        for name in sorted(gate_names)
    }
    loaded_sha256 = runtime._state_dict_sha256(loaded)
    if loaded_sha256 != runtime._state_dict_sha256(expected_runtime):
        raise ValueError("Scene contrast runtime-cast gate state differs")
    return {
        "step": int(manifest["step"]),
        "gate_state_sha256": str(manifest["gate_state_sha256"]),
        "runtime_gate_state_sha256": loaded_sha256,
        "parameter_tensors": len(loaded),
    }


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    reference_root: Path,
    training_root: Path,
    manifests: Sequence[Mapping[str, Any]],
    donor_payload: Sequence[Mapping[str, Any]],
    shard_index: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != training.BASE_CONFIG_SHA256:
        raise ValueError("Scene contrast probe base config differs")
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != training.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Scene contrast probe V9 adapter differs")
    if reference_root.name != REFERENCE_ROOT_NAME:
        raise ValueError("Scene contrast probe reference root differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "training_result_receipt_sha256": TRAINING_RESULT_RECEIPT_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": training.BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_files_sha256": training.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_file_sha256": causal.DATASET_SHA256,
        "reference_root": str(reference_root),
        "reference_artifacts": causal.reference_artifacts(reference_root),
        "training_root": str(training_root),
        "checkpoint_artifacts": [
            {
                "step": int(manifest["step"]),
                "gate_state_sha256": str(manifest["gate_state_sha256"]),
                "manifest_sha256": sha256_file(
                    training_root / f"checkpoint-{manifest['step']}" / "manifest.json"
                ),
                "patch_sha256": str(manifest["patch_file"]["sha256"]),
            }
            for manifest in manifests
        ],
        "selection_payload_sha256": PROBE_PAYLOAD_SHA256,
        "selection_rows": PROBE_ROWS,
        "donor_mapping_payload_sha256": canonical_sha256(list(donor_payload)),
        "donor_mapping_rows": len(donor_payload),
        "candidate_steps": list(CANDIDATE_STEPS),
        "conditions": list(CONDITIONS),
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "causal_runner_sha256": sha256_file(Path(causal.__file__)),
        "protected_splits_opened": [],
    }


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Scene contrast probe binding differs: {path}")
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
                raise ValueError(f"Duplicate scene contrast probe row: {path}:{source_index}")
            records[source_index] = record
    return records


def append_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def output_path(output_dir: Path, step: int, condition: str) -> Path:
    return output_dir / f"checkpoint-{step}.{condition}.jsonl"


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    step: int,
    condition: str,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected scene contrast probe rows: {step}:{condition}")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "training_result_receipt_sha256": TRAINING_RESULT_RECEIPT_SHA256,
            "checkpoint_step": step,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source_index,
            "row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed scene contrast probe row differs: {step}:{condition}:{source_index}")


def generate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    donor_row: Mapping[str, Any],
    donor_token_delta: int,
    step: int,
    condition: str,
    gate_state_sha256: str,
    runtime_gate_state_sha256: str,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    try:
        if condition == "correct_state":
            state = causal.prime_messages(model, tokenizer, row["messages"], device=device)
            state_metadata = {
                "state_kind": "row_correct",
                "state_sha256": causal.tensor_digest(state),
            }
        elif condition == "matched_donor_state":
            state = causal.prime_messages(model, tokenizer, donor_row["messages"], device=device)
            state_metadata = {
                "state_kind": "different_gold_length_matched_donor",
                "state_sha256": causal.tensor_digest(state),
                "donor_source_index": int(donor_row["source_index"]),
                "donor_row_sha256": str(donor_row["row_sha256"]),
                "absolute_write_token_delta": donor_token_delta,
            }
        elif condition == "zero_state":
            reset_delta_state(model)
            state_metadata = {"state_kind": "empty", "state_sha256": None}
        else:
            raise ValueError(f"Unknown scene contrast probe condition: {condition}")
        generated = causal.generate_read(model, tokenizer, row["messages"], device=device)
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "training_result_receipt_sha256": TRAINING_RESULT_RECEIPT_SHA256,
            "checkpoint_step": step,
            "gate_state_sha256": gate_state_sha256,
            "runtime_gate_state_sha256": runtime_gate_state_sha256,
            "condition": condition,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
            **state_metadata,
            **generated,
        }
    finally:
        reset_delta_state(model)
        set_delta_write_enabled(model, True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training.validate_protocol()
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Scene contrast probe requires a valid four-way shard")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    training_root = args.training_root.expanduser().resolve(strict=True)
    output_dir = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}"
    manifests = validate_training_root(training_root)
    rows = causal.load_rows(dataset_root)
    selected = selected_probe_rows(rows)
    shard_rows = [
        row for row in selected if int(row["source_index"]) % WORLD_SIZE == args.shard_index
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
        manifests=manifests,
        donor_payload=donor_payload,
        shard_index=args.shard_index,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    completed: dict[tuple[int, str], dict[int, Mapping[str, Any]]] = {}
    for step in CANDIDATE_STEPS:
        for condition in CONDITIONS:
            existing = read_completed(output_path(output_dir, step, condition))
            validate_resume(
                existing,
                shard_rows,
                step=step,
                condition=condition,
                shard_index=args.shard_index,
            )
            completed[(step, condition)] = existing
    if all(len(value) == len(shard_rows) for value in completed.values()):
        print(f"SCENE_CONTRAST_PROBE_SHARD_COMPLETE shard={args.shard_index}", flush=True)
        return 0
    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    by_index = {int(row["source_index"]): row for row in rows}
    for manifest in manifests:
        step = int(manifest["step"])
        loaded = load_gate_patch(
            model,
            patch_path=training_root / f"checkpoint-{step}" / "gate_patch.pt",
            manifest=manifest,
        )
        for ordinal, row in enumerate(shard_rows, start=1):
            source_index = int(row["source_index"])
            donor_index = donor_mapping[source_index]
            donor_row = by_index[donor_index]
            donor_delta = abs(token_counts[source_index] - token_counts[donor_index])
            for condition in CONDITIONS:
                if source_index in completed[(step, condition)]:
                    continue
                record = generate_condition(
                    model,
                    tokenizer,
                    row,
                    donor_row=donor_row,
                    donor_token_delta=donor_delta,
                    step=step,
                    condition=condition,
                    gate_state_sha256=str(loaded["gate_state_sha256"]),
                    runtime_gate_state_sha256=str(loaded["runtime_gate_state_sha256"]),
                    shard_index=args.shard_index,
                    device=args.device,
                )
                append_record(output_path(output_dir, step, condition), record)
                print(
                    f"SCENE_CONTRAST_PROBE_PROGRESS shard={args.shard_index} "
                    f"step={step} condition={condition} row={source_index} "
                    f"ordinal={ordinal}/{len(shard_rows)}",
                    flush=True,
                )
    print(f"SCENE_CONTRAST_PROBE_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
