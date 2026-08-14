#!/usr/bin/env python3
"""Generate the locked seed-ensemble candidate on open TRAIN scene rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import (  # noqa: E402
    load_model_and_tokenizer,
    reset_delta_state,
    set_delta_write_enabled,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_scene_seed_ensemble as materializer,
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
    run_natural_memory_native_scene_seed_ensemble as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_ensemble_eval_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_ensemble_eval_input.v1"
WORLD_SIZE = 4
ROWS = progression.REMAINING_ROWS
ROW_PAYLOAD_SHA256 = progression.REMAINING_PAYLOAD_SHA256
CONDITION = "correct_state"


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


def validate_materialization(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = probe.validate_signed_json(
        root / "result.json",
        description="Seed-ensemble materialization result",
    )
    manifest = probe.validate_signed_json(
        root / "manifest.json",
        description="Seed-ensemble candidate manifest",
    )
    patch_path = root / "gate_patch.pt"
    patch_file = manifest.get("patch_file")
    if (
        result.get("schema") != materializer.SCHEMA
        or result.get("protocol_payload_sha256") != training.PROTOCOL_PAYLOAD_SHA256
        or result.get("candidate_id") != materializer.CANDIDATE_ID
        or result.get("candidate_manifest") != manifest
        or result.get("candidate_fixed_before_generation") is not True
        or result.get("protected_splits_opened") != []
        or manifest.get("schema") != materializer.PATCH_SCHEMA
        or manifest.get("protocol_payload_sha256") != training.PROTOCOL_PAYLOAD_SHA256
        or manifest.get("candidate_id") != materializer.CANDIDATE_ID
        or manifest.get("parameter_tensors") != 126
        or manifest.get("parameter_elements") != 108906
        or not isinstance(patch_file, Mapping)
        or patch_file.get("bytes") != patch_path.stat().st_size
        or patch_file.get("sha256") != sha256_file(patch_path)
    ):
        raise ValueError("Seed-ensemble materialization binding differs")
    return result, manifest


def load_candidate_patch(
    model: torch.nn.Module,
    *,
    patch_path: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = torch.load(patch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Seed-ensemble patch payload differs")
    required = {
        "schema": materializer.PATCH_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "candidate_id": materializer.CANDIDATE_ID,
        "source_gate_state_sha256": manifest["source_gate_state_sha256"],
        "seed_weights": manifest["seed_weights"],
        "denominator": manifest["denominator"],
        "gate_state_sha256": manifest["gate_state_sha256"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("Seed-ensemble patch metadata differs")
    state = payload["state_dict"]
    if runtime._state_dict_sha256(state) != manifest["gate_state_sha256"]:
        raise ValueError("Seed-ensemble patch state hash differs")
    parameters = dict(model.named_parameters())
    names = {
        name
        for name in parameters
        if any(name.endswith(f".{family}") for family in training.contrast.GATE_FAMILIES)
    }
    if set(state) != names or len(names) != 126:
        raise ValueError("Seed-ensemble runtime gate names differ")
    with torch.no_grad():
        for name in sorted(names):
            source = state[name]
            target = parameters[name]
            if source.shape != target.shape:
                raise ValueError(f"Seed-ensemble runtime shape differs: {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    loaded = {
        name: parameters[name].detach().cpu().clone() for name in sorted(names)
    }
    expected = {
        name: state[name].to(dtype=parameters[name].dtype).detach().cpu().clone()
        for name in sorted(names)
    }
    runtime_sha256 = runtime._state_dict_sha256(loaded)
    if runtime_sha256 != runtime._state_dict_sha256(expected):
        raise ValueError("Seed-ensemble runtime-cast state differs")
    return {
        "candidate_id": materializer.CANDIDATE_ID,
        "gate_state_sha256": manifest["gate_state_sha256"],
        "runtime_gate_state_sha256": runtime_sha256,
    }


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    reference_root: Path,
    materialization_root: Path,
    materialization: Mapping[str, Any],
    manifest: Mapping[str, Any],
    shard_index: int,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != training.contrast.BASE_CONFIG_SHA256:
        raise ValueError("Seed-ensemble base config differs")
    adapter_files = training.gate.snapshot_directory_files(memory_dir)
    if training.gate._sha256_json(adapter_files) != training.contrast.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Seed-ensemble V9 adapter differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": training.contrast.BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_files_sha256": training.contrast.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_file_sha256": causal.DATASET_SHA256,
        "reference_root": str(reference_root),
        "reference_artifacts": causal.reference_artifacts(reference_root),
        "materialization_root": str(materialization_root),
        "materialization_result_file_sha256": sha256_file(
            materialization_root / "result.json"
        ),
        "materialization_result_receipt_sha256": materialization["receipt"][
            "payload_sha256"
        ],
        "candidate_id": materializer.CANDIDATE_ID,
        "candidate_gate_state_sha256": manifest["gate_state_sha256"],
        "candidate_manifest_sha256": sha256_file(materialization_root / "manifest.json"),
        "candidate_patch_sha256": manifest["patch_file"]["sha256"],
        "row_payload_sha256": ROW_PAYLOAD_SHA256,
        "rows": ROWS,
        "condition": CONDITION,
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
            raise ValueError(f"Seed-ensemble input binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def output_path(output_dir: Path) -> Path:
    return output_dir / f"{materializer.CANDIDATE_ID}.{CONDITION}.jsonl"


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            record = json.loads(raw_line)
            source_index = int(record["source_index"])
            if source_index in records:
                raise ValueError(f"Duplicate seed-ensemble output: {path}:{source_index}")
            records[source_index] = record
    return records


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    gate_state_sha256: str,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError("Unexpected seed-ensemble rows")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "candidate_id": materializer.CANDIDATE_ID,
            "gate_state_sha256": gate_state_sha256,
            "condition": CONDITION,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source_index,
            "row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Resumed seed-ensemble row differs: {source_index}")


def append_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    gate_state_sha256: str,
    runtime_gate_state_sha256: str,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    try:
        state = causal.prime_messages(model, tokenizer, row["messages"], device=device)
        generated = causal.generate_read(model, tokenizer, row["messages"], device=device)
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": training.PROTOCOL_PAYLOAD_SHA256,
            "candidate_id": materializer.CANDIDATE_ID,
            "gate_state_sha256": gate_state_sha256,
            "runtime_gate_state_sha256": runtime_gate_state_sha256,
            "condition": CONDITION,
            "state_kind": "row_correct",
            "state_sha256": causal.tensor_digest(state),
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
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
    parser.add_argument("--materialization-root", type=Path, required=True)
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
        raise ValueError("Seed-ensemble evaluation requires a valid four-way shard")
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    reference_root = args.reference_root.expanduser().resolve(strict=True)
    materialization_root = args.materialization_root.expanduser().resolve(strict=True)
    output_dir = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}"
    materialization, manifest = validate_materialization(materialization_root)
    rows = causal.load_rows(dataset_root)
    selected = progression.progression_rows(rows)
    shard_rows = [
        row for row in selected if int(row["source_index"]) % WORLD_SIZE == args.shard_index
    ]
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        reference_root=reference_root,
        materialization_root=materialization_root,
        materialization=materialization,
        manifest=manifest,
        shard_index=args.shard_index,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    existing = read_completed(output_path(output_dir))
    validate_resume(
        existing,
        shard_rows,
        gate_state_sha256=str(manifest["gate_state_sha256"]),
        shard_index=args.shard_index,
    )
    if len(existing) == len(shard_rows):
        print(f"SCENE_SEED_ENSEMBLE_SHARD_COMPLETE shard={args.shard_index}", flush=True)
        return 0
    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    loaded = load_candidate_patch(
        model,
        patch_path=materialization_root / "gate_patch.pt",
        manifest=manifest,
    )
    for ordinal, row in enumerate(shard_rows, start=1):
        source_index = int(row["source_index"])
        if source_index in existing:
            continue
        record_value = generate(
            model,
            tokenizer,
            row,
            gate_state_sha256=str(loaded["gate_state_sha256"]),
            runtime_gate_state_sha256=str(loaded["runtime_gate_state_sha256"]),
            shard_index=args.shard_index,
            device=args.device,
        )
        append_record(output_path(output_dir), record_value)
        print(
            f"SCENE_SEED_ENSEMBLE_PROGRESS shard={args.shard_index} "
            f"row={source_index} ordinal={ordinal}/{len(shard_rows)}",
            flush=True,
        )
    print(f"SCENE_SEED_ENSEMBLE_SHARD_COMPLETE shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
