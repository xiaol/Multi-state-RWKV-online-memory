#!/usr/bin/env python3
"""Run one append-only shard of effective native scene strength control."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import load_model_and_tokenizer, reset_delta_state, set_delta_write_enabled  # noqa: E402
from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_strength_calibration as v1,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_controller_shard.v2"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_controller_input.v2"
PREFLIGHT_SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_controller_preflight.v2"
SELECTION_SCHEMA = "rwkv_ms_natural_memory_native_scene_strength_controller_selection.v2"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_strength_controller_protocol_v2.json"
PROTOCOL_PAYLOAD_SHA256 = "2c15c7c56c769c91599a4b6590ec72ba0004cc358b8f3133d4c43733bfccede9"
RUNTIME_SHA256 = "2f7d97a7dbe014f635ca4f08bed5f527e0ea74144de9cf1a7b6fdf4c2b815ada"
WORLD_SIZE = 4
PHASES = ("preflight", "fit", "holdout")
PREFLIGHT_STRENGTHS = {
    "scale_0p0": 0.0,
    "scale_0p5": 0.5,
    "scale_1p0": 1.0,
}
FIT_STRENGTHS = dict(v1.STRENGTHS)


def canonical_sha256(value: Any) -> str:
    return v1.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return v1.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Strength-controller protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Strength-controller protocol hash differs")
    if sha256_file(PROJECT_ROOT / "deltamem/core/delta_impl.py") != RUNTIME_SHA256:
        raise ValueError("Strength-controller runtime hash differs")
    return value


def validate_signed_receipt(
    path: Path,
    *,
    schema: str,
    runner_sha256: str,
    require_passed: bool,
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Strength-controller receipt missing: {path}")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if canonical_sha256(unsigned) != receipt.get("payload_sha256"):
        raise ValueError(f"Strength-controller receipt differs: {path}")
    required = {
        "schema": schema,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": runner_sha256,
        "passed": require_passed,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError(f"Strength-controller binding differs: {path}")
    analyzer_hash = value.get("analyzer_sha256")
    analyzer_path = SCRIPT_DIR / "analyze_natural_memory_native_scene_strength_controller.py"
    if not isinstance(analyzer_hash, str) or re.fullmatch(r"[0-9a-f]{64}", analyzer_hash) is None:
        raise ValueError("Strength-controller analyzer hash is invalid")
    if sha256_file(analyzer_path) != analyzer_hash:
        raise ValueError("Strength-controller analyzer file hash differs")
    return value


def validate_selection(path: Path, *, runner_sha256: str) -> Mapping[str, Any]:
    value = validate_signed_receipt(
        path,
        schema=SELECTION_SCHEMA,
        runner_sha256=runner_sha256,
        require_passed=True,
    )
    if value.get("holdout_authorized") is not True:
        raise ValueError("Strength-controller holdout is unauthorized")
    selected = value.get("selected_strength_name")
    if selected not in FIT_STRENGTHS or value.get("selected_strength") != FIT_STRENGTHS[selected]:
        raise ValueError("Strength-controller selected strength differs")
    return value


def attach_controller(model, strength: float) -> tuple[Counter[str], str]:
    modules = list(iter_delta_mem_modules(model))
    if len(modules) != 42:
        raise ValueError(f"Strength controller requires 42 wrapped layers, found {len(modules)}")
    calls: Counter[str] = Counter()

    def controller(module, head_name, reference, raw_delta, token_mask):
        del module, token_mask
        if head_name not in {"q", "o"}:
            raise ValueError(f"Strength controller received unsupported head: {head_name}")
        if reference.shape != raw_delta.shape:
            raise ValueError("Strength controller projection shape differs")
        if reference.device != raw_delta.device:
            raise ValueError("Strength controller projection device differs")
        calls[head_name] += 1
        return reference + raw_delta.to(dtype=reference.dtype) * float(strength)

    for _, module in modules:
        if module.training:
            raise ValueError("Strength controller requires eval mode")
        module._eval_memory_delta_controller = controller
    payload = {
        "strength": float(strength),
        "heads": ["o", "q"],
        "wrapped_layers": [name for name, _ in modules],
        "runtime_hook": "_eval_memory_delta_controller",
    }
    return calls, canonical_sha256(payload)


def rows_for_phase(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    shard_index: int,
) -> list[Mapping[str, Any]]:
    if phase == "preflight":
        return [row for row in rows if int(row["source_index"]) == shard_index]
    return v1.selected_rows(rows, phase=phase, shard_index=shard_index)


def read_completed(path: Path) -> dict[int, Mapping[str, Any]]:
    return v1.read_completed(path)


def validate_resume(
    existing: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    strength_name: str,
    strength: float,
    shard_index: int,
) -> None:
    expected = {int(row["source_index"]): row for row in rows}
    if not set(existing) <= set(expected):
        raise ValueError(f"Unexpected strength-controller indices: {phase}:{strength_name}")
    for source_index, record in existing.items():
        required = {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "phase": phase,
            "strength_name": strength_name,
            "strength": strength,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": source_index,
            "row_sha256": expected[source_index]["row_sha256"],
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise ValueError(f"Strength-controller resumed record differs: {phase}:{strength_name}:{source_index}")


def generate_record(
    model,
    tokenizer,
    row: Mapping[str, Any],
    *,
    phase: str,
    strength_name: str,
    strength: float,
    shard_index: int,
    device: str,
) -> Mapping[str, Any]:
    try:
        calls, controller_sha256 = attach_controller(model, strength)
        state = causal.prime_messages(model, tokenizer, row["messages"], device=device)
        generated = causal.generate_read(model, tokenizer, row["messages"], device=device)
        if calls["q"] <= 0 or calls["o"] <= 0:
            raise RuntimeError("Strength controller did not receive both Q and O calls")
        return {
            "schema": SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "phase": phase,
            "strength_name": strength_name,
            "strength": strength,
            "controller_sha256": controller_sha256,
            "controller_calls": {"q": calls["q"], "o": calls["o"]},
            "wrapped_layers": 42,
            "shard_index": shard_index,
            "world_size": WORLD_SIZE,
            "source_index": row["source_index"],
            "row_sha256": row["row_sha256"],
            "correct_state_sha256": causal.tensor_digest(state),
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
    parser.add_argument("--preflight", type=Path)
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
        raise ValueError("Strength controller requires a valid four-way shard")
    runner_sha256 = sha256_file(Path(__file__))
    preflight: Mapping[str, Any] | None = None
    selection: Mapping[str, Any] | None = None
    if args.phase == "preflight":
        if args.preflight is not None or args.selection is not None:
            raise ValueError("Strength-controller preflight forbids receipts")
        strengths = PREFLIGHT_STRENGTHS
    elif args.phase == "fit":
        if args.preflight is None or args.selection is not None:
            raise ValueError("Strength-controller fit requires only signed preflight")
        preflight = validate_signed_receipt(
            args.preflight.expanduser().resolve(strict=True),
            schema=PREFLIGHT_SCHEMA,
            runner_sha256=runner_sha256,
            require_passed=True,
        )
        strengths = FIT_STRENGTHS
    else:
        if args.preflight is None or args.selection is None:
            raise ValueError("Strength-controller holdout requires preflight and selection")
        preflight = validate_signed_receipt(
            args.preflight.expanduser().resolve(strict=True),
            schema=PREFLIGHT_SCHEMA,
            runner_sha256=runner_sha256,
            require_passed=True,
        )
        selection = validate_selection(
            args.selection.expanduser().resolve(strict=True),
            runner_sha256=runner_sha256,
        )
        selected_name = str(selection["selected_strength_name"])
        strengths = {selected_name: FIT_STRENGTHS[selected_name]}

    base_model = args.base_model.expanduser().resolve(strict=True)
    memory_dir = args.memory_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_dir = output_root / args.phase / f"shard-{args.shard_index}"
    rows = causal.load_rows(dataset_root)
    v1.validate_partitions(rows)
    shard_rows = rows_for_phase(rows, phase=args.phase, shard_index=args.shard_index)
    binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "base_model": str(base_model),
        "base_config_sha256": v1.BASE_CONFIG_SHA256,
        "memory_dir": str(memory_dir),
        "memory_adapter_sha256": v1.MEMORY_ADAPTER_SHA256,
        "runtime_sha256": RUNTIME_SHA256,
        "dataset_root": str(dataset_root),
        "dataset_sha256": v1.shared.TARGET_SHA256,
        "phase": args.phase,
        "shard_index": args.shard_index,
        "world_size": WORLD_SIZE,
        "strengths": strengths,
        "preflight_payload_sha256": None if preflight is None else preflight["receipt"]["payload_sha256"],
        "selection_payload_sha256": None if selection is None else selection["receipt"]["payload_sha256"],
        "runner_sha256": runner_sha256,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    v1.write_or_validate_json(output_dir / "input_binding.json", binding)
    existing_by_strength: dict[str, dict[int, Mapping[str, Any]]] = {}
    for strength_name, strength in strengths.items():
        existing = read_completed(output_dir / f"{strength_name}.jsonl")
        validate_resume(
            existing,
            shard_rows,
            phase=args.phase,
            strength_name=strength_name,
            strength=strength,
            shard_index=args.shard_index,
        )
        existing_by_strength[strength_name] = existing
    if all(len(existing_by_strength[name]) == len(shard_rows) for name in strengths):
        print(f"STRENGTH_CONTROLLER_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}", flush=True)
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
        for strength_name, strength in strengths.items():
            if source_index in existing_by_strength[strength_name]:
                continue
            record = generate_record(
                model,
                tokenizer,
                row,
                phase=args.phase,
                strength_name=strength_name,
                strength=strength,
                shard_index=args.shard_index,
                device=args.device,
            )
            v1.append_record(output_dir / f"{strength_name}.jsonl", record)
            print(
                f"STRENGTH_CONTROLLER_PROGRESS phase={args.phase} shard={args.shard_index} "
                f"strength={strength_name} row={source_index} ordinal={ordinal}/{len(shard_rows)}",
                flush=True,
            )
    print(f"STRENGTH_CONTROLLER_SHARD_COMPLETE phase={args.phase} shard={args.shard_index}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
