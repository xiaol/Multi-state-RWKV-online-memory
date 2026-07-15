#!/usr/bin/env python3
"""Validate a RWKV-MS math fixture by recomputing the PyTorch reference path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rwkv_ms_math_fixture_common import (
    DEFAULT_OUTPUT,
    compare_tensor_records,
    compute_expected_tensors,
    dtype_from_name,
    file_sha256,
    load_layer_artifacts,
    raw_tensor_sha256,
    read_json,
    records_from_tensors,
    tensor_from_record,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a RWKV-MS PyTorch math fixture.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-dir", type=Path, default=None, help="Override memory_dir recorded in the fixture.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--emit-actual", type=Path, default=None, help="Write recomputed tensor records as JSON.")
    parser.add_argument("--json", action="store_true", help="Emit JSON result.")
    return parser.parse_args()


def verify_record_hashes(section_name: str, records: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, record in sorted(records.items()):
        actual = raw_tensor_sha256(tensor_from_record(record))
        expected = record.get("sha256")
        if actual != expected:
            errors.append(f"{section_name}.{name} sha256 mismatch: {actual} != {expected}")
    return errors


def source_hash_errors(fixture: dict[str, Any], memory_dir: Path) -> list[str]:
    source = fixture.get("source", {})
    checks = {
        "adapter_sha256": memory_dir / "delta_mem_adapter.pt",
        "config_sha256": memory_dir / "delta_mem_config.json",
        "metadata_sha256": memory_dir / "adapter_metadata.json",
    }
    errors: list[str] = []
    for key, path in checks.items():
        expected = source.get(key)
        if expected is None:
            continue
        actual = file_sha256(path)
        if actual != expected:
            errors.append(f"{key} mismatch for {path}: {actual} != {expected}")
    return errors


def load_inputs(records: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    return {
        "hidden_states": tensor_from_record(records["hidden_states"]),
        "initial_state": tensor_from_record(records["initial_state"]),
        "initial_positions": tensor_from_record(records["initial_positions"]),
        "initial_previous_source": (
            tensor_from_record(records["initial_previous_source"])
            if "initial_previous_source" in records
            else None
        ),
        "token_mask": tensor_from_record(records["token_mask"]) if "token_mask" in records else None,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = args.fixture.expanduser().resolve()
    fixture = read_json(fixture_path)
    errors: list[str] = []
    if fixture.get("schema") != "delta_mem_rwkv_ms_math_fixture.v1":
        errors.append(f"unexpected schema: {fixture.get('schema')}")

    input_records = fixture.get("inputs", {})
    expected_records = fixture.get("expected", {})
    if not isinstance(input_records, dict) or not isinstance(expected_records, dict):
        raise ValueError("Fixture must contain object-valued inputs and expected sections")
    errors.extend(verify_record_hashes("inputs", input_records))
    errors.extend(verify_record_hashes("expected", expected_records))

    memory_dir = args.memory_dir
    if memory_dir is None:
        memory_dir = Path(str(fixture.get("source", {}).get("memory_dir", "")))
    memory_dir = memory_dir.expanduser().resolve()
    source_errors = source_hash_errors(fixture, memory_dir)
    if source_errors and not args.allow_source_mismatch:
        errors.extend(source_errors)

    config = fixture["config"]
    dtype = dtype_from_name(str(config.get("dtype", "float32")))
    layer_artifacts = load_layer_artifacts(memory_dir, int(config["layer"]), dtype)
    actual = compute_expected_tensors(
        layer_artifacts=layer_artifacts,
        inputs=load_inputs(input_records),
        scan_config=fixture["scan_config"],
        include_step_trace=bool(config.get("include_step_trace", False)),
    )
    comparison = compare_tensor_records(
        expected_records,
        actual,
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_errors=max(1, int(args.max_errors)),
    )
    errors.extend(comparison["errors"])

    if args.emit_actual is not None:
        write_json(
            args.emit_actual.expanduser(),
            {
                "schema": "delta_mem_rwkv_ms_math_actual.v1",
                "fixture": str(fixture_path),
                "memory_dir": str(memory_dir),
                "tensors": records_from_tensors(actual),
            },
        )

    return {
        "fixture": str(fixture_path),
        "memory_dir": str(memory_dir),
        "schema": fixture.get("schema"),
        "ok": not errors,
        "source_ok": not source_errors,
        "source_errors": source_errors,
        "compared_tensors": comparison["compared_tensors"],
        "max_abs_diff": comparison["max_abs_diff"],
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "errors": errors[: max(1, int(args.max_errors))],
    }


if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()
        result = validate(args)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"fixture: {result['fixture']}")
            print(f"memory_dir: {result['memory_dir']}")
            print(f"ok: {result['ok']}")
            print(f"source_ok: {result['source_ok']}")
            print(f"compared_tensors: {result['compared_tensors']}")
            print(f"max_abs_diff: {result['max_abs_diff']}")
            for error in result["errors"]:
                print(f"error: {error}")
        if not result["ok"]:
            raise SystemExit(1)
