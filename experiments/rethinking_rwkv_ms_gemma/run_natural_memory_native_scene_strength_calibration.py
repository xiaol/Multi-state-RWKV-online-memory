#!/usr/bin/env python3
"""Run one append-only shard of the native scene strength-calibration study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer, reset_delta_state, set_delta_write_enabled  # noqa: E402
from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    prepare_natural_memory_native_scene_state_retrieval as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal_runner,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_calibration_shard.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_calibration_input.v1"
SELECTION_SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_calibration_selection.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_strength_calibration_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "06ed3ac29ce73863b1b760b1c542d39b18cd9618f2b7ceb48bae858ff81b0989"
MEMORY_ADAPTER_SHA256 = causal_runner.MEMORY_ADAPTER_SHA256
BASE_CONFIG_SHA256 = causal_runner.BASE_CONFIG_SHA256
WORLD_SIZE = 4
PHASES = ("fit", "holdout")
PARTITION_NAMESPACE = "rwkv-ms-scale-v1:"
STRENGTHS = {
    "scale_0p125": 0.125,
    "scale_0p25": 0.25,
    "scale_0p5": 0.5,
    "scale_0p75": 0.75,
}
FIT_PARTITION_SHA256 = "902baf8f8af6552765514e17545b4234266eb89f31ca100dae3866eba30bcc99"
HOLDOUT_PARTITION_SHA256 = "0c2a3a3ee8dfd6d1bb1077b2b690289e734241476410816fd09247f0a8d2a655"
ALL_PARTITION_SHA256 = "ac9c20c2d0641e31bf98210a13d416cc8a64d65c47c96c7fb272889d14570eb1"


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Strength-calibration protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Strength-calibration protocol hash differs")
    return value


def partition_for_hash(row_sha256: str) -> str:
    digest = hashlib.sha256(f"{PARTITION_NAMESPACE}{row_sha256}".encode("ascii")).hexdigest()
    return "holdout" if int(digest[:8], 16) % 5 == 0 else "fit"


def partition_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": int(row["source_index"]),
            "row_sha256": str(row["row_sha256"]),
            "partition": partition_for_hash(str(row["row_sha256"])),
        }
        for row in rows
        if int(row["source_index"]) >= shared.EXCLUDED_TARGET_ROWS
    ]


def validate_partitions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload = partition_payload(rows)
    fit = [record for record in payload if record["partition"] == "fit"]
    holdout = [record for record in payload if record["partition"] == "holdout"]
    for records, count, digest in (
        (payload, 357, ALL_PARTITION_SHA256),
        (fit, 284, FIT_PARTITION_SHA256),
        (holdout, 73, HOLDOUT_PARTITION_SHA256),
    ):
        if len(records) != count or canonical_sha256(records) != digest:
            raise ValueError("Strength-calibration partition binding differs")
    return payload


def validate_selection(path: Path, *, runner_sha256: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Strength-calibration selection receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if canonical_sha256(unsigned) != receipt.get("payload_sha256"):
        raise ValueError("Strength-calibration selection receipt differs")
    required = {
        "schema": SELECTION_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": runner_sha256,
        "phase_one_passed": True,
        "holdout_authorized": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("Strength-calibration selection binding differs")
    selected = value.get("selected_strength_name")
    if selected not in STRENGTHS or value.get("selected_strength") != STRENGTHS[selected]:
        raise ValueError("Strength-calibration selected strength differs")
    analyzer_sha256 = value.get("analyzer_sha256")
    if not isinstance(analyzer_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", analyzer_sha256) is None:
        raise ValueError("Strength-calibration analyzer hash is invalid")
    analyzer_path = SCRIPT_DIR / "analyze_natural_memory_native_scene_strength_calibration.py"
    if sha256_file(analyzer_path) != analyzer_sha256:
        raise ValueError("Strength-calibration analyzer file hash differs")
    return value


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(row["source_index"]) >= shared.EXCLUDED_TARGET_ROWS
        and partition_for_hash(str(row["row_sha256"])) == phase
        and int(row["source_index"]) % WORLD_SIZE == shard_index
    ]


def set_strength(model, strength: float) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    modules = list(iter_delta_mem_modules(model))
    if len(modules) != 42:
        raise ValueError(f"Strength calibration requires 42 wrapped layers, found {len(modules)}")
    for name, module in modules:
        if module.memory_fusion_placement != "attention_output":
            raise ValueError(f"Strength calibration placement differs: {name}")
        if float(module.memory_fusion_residual_scale_max) != 1.0:
            raise ValueError(f"Strength calibration maximum differs: {name}")
        module.memory_fusion_residual_scale = float(strength)
        settings.append(
            {
                "module_name": name,
                "layer_index": int(module.layer_idx),
                "memory_fusion_placement": str(module.memory_fusion_placement),
                "memory_fusion_residual_scale": float(module.memory_fusion_residual_scale),
                "memory_fusion_residual_scale_max": float(module.memory_fusion_residual_scale_max),
            }
        )
    if any(setting["memory_fusion_residual_scale"] != strength for setting in settings):
        raise ValueError("Strength calibration did not apply uniformly")
    return settings


def settings_sha256(settings: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(list(settings))


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                record = json.loads(raw_line)
                source_index = int(record["source_index"])
                if source_index in records:
                    raise ValueError(f"Duplicate strength-calibration row: {path}:{source_index}")
                records[source_index] = record
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Strength-calibration binding differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def input_binding(
    *,
    base_model: Path,
    memory_dir: Path,
    dataset_root: Path,
    phase: str,
    shard_index: int,
    strength_names: Sequence[str],
    selection: Mapping[str, Any] | None,
    selection_path: Path | None,
    dtype: str,
    attn_implementation: str,
) -> Mapping[str, Any]:
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Strength-calibration base config differs")
    if sha256_file(memory_dir / "delta_mem_adapter.pt") != MEMORY_ADAPTER_SHA256:
        raise ValueError("Strength-calibration memory adapter differs")
    return {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_sha256": MEMORY_ADAPTER_SHA256,
        "memory_config_sha256": sha256_file(memory_dir / "delta_mem_config.json"),
        "dataset_root": str(dataset_root),
        "dataset_sha256": shared.TARGET_SHA256,
        "phase": phase,
        "shard_index": shard_index,
        "world_size": WORLD_SIZE,
        "strengths": {name: STRENGTHS[name] for name in strength_names},
        "selection_path": None if selection_path is None else str(selection_path),
        "selection_file_sha256": None if selection_path is None else sha256_file(selection_path),
        "selection_payload_sha256": None if selection is None else selection["receipt"]["payload_sha256"],
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "runner_sha256": sha256_file(Path(__file__)),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_natural_memory_native_scene_causal.py"
        ),
    }


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    strength_name: str,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected strength-calibration indices: {phase}:{strength_name}")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "phase": phase,
            "strength_name": strength_name,
            "strength": STRENGTHS[strength_name],
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source_index,
            "row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Strength-calibration resumed record differs: {phase}:{strength_name}:{source_index}")


def generate_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    phase: str,
    strength_name: str,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    try:
        strength = STRENGTHS[strength_name]
        settings = set_strength(model, strength)
        state = causal_runner.prime_messages(
            model,
            tokenizer,
            row["messages"],
            device=device,
        )
        generated = causal_runner.generate_read(
            model,
            tokenizer,
            row["messages"],
            device=device,
        )
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "phase": phase,
            "strength_name": strength_name,
            "strength": strength,
            "settings_sha256": settings_sha256(settings),
            "wrapped_layers": len(settings),
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": row["source_index"],
            "row_sha256": row["row_sha256"],
            "correct_state_sha256": causal_runner.tensor_digest(state),
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    if not 0 <= args.shard_index < WORLD_SIZE:
        raise ValueError("Strength calibration requires a valid four-way shard")
    runner_sha256 = sha256_file(Path(__file__))
    selection: Mapping[str, Any] | None = None
    selection_path: Path | None = None
    if args.phase == "fit":
        if args.selection is not None:
            raise ValueError("Strength-calibration fit phase forbids a selection")
        strength_names = tuple(STRENGTHS)
    else:
        if args.selection is None:
            raise ValueError("Strength-calibration holdout phase requires a signed selection")
        selection_path = args.selection.expanduser().resolve(strict=True)
        selection = validate_selection(selection_path, runner_sha256=runner_sha256)
        strength_names = (str(selection["selected_strength_name"]),)
    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.phase / f"shard-{args.shard_index}"
    rows = causal_runner.load_rows(dataset_root)
    validate_partitions(rows)
    shard_rows = selected_rows(rows, phase=args.phase, shard_index=args.shard_index)
    binding = input_binding(
        base_model=base_model,
        memory_dir=memory_dir,
        dataset_root=dataset_root,
        phase=args.phase,
        shard_index=args.shard_index,
        strength_names=strength_names,
        selection=selection,
        selection_path=selection_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    write_or_validate_json(output_dir / "input_binding.json", binding)
    existing_by_strength: dict[str, dict[int, Mapping[str, Any]]] = {}
    for strength_name in strength_names:
        existing = read_completed(output_dir / f"{strength_name}.jsonl")
        validate_resume(
            existing,
            shard_rows,
            phase=args.phase,
            strength_name=strength_name,
            shard_index=args.shard_index,
        )
        existing_by_strength[strength_name] = existing
    if all(len(existing_by_strength[name]) == len(shard_rows) for name in strength_names):
        print(f"STRENGTH_CALIBRATION_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}", flush=True)
        return 0
    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(memory_dir),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for ordinal, row in enumerate(shard_rows, start=1):
        source_index = int(row["source_index"])
        for strength_name in strength_names:
            if source_index in existing_by_strength[strength_name]:
                continue
            record = generate_record(
                model,
                tokenizer,
                row,
                phase=args.phase,
                strength_name=strength_name,
                shard_index=args.shard_index,
                device=args.device,
            )
            append_record(output_dir / f"{strength_name}.jsonl", record)
            print(
                f"STRENGTH_CALIBRATION_PROGRESS phase={args.phase} shard={args.shard_index} "
                f"strength={strength_name} row={source_index} ordinal={ordinal}/{len(shard_rows)}",
                flush=True,
            )
    print(f"STRENGTH_CALIBRATION_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
