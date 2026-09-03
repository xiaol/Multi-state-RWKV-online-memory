#!/usr/bin/env python3
"""Materialize a reproducible convex blend of recurrent-routed adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_adapter_blend.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def snapshot_directory(path: Path) -> Mapping[str, Mapping[str, Any]]:
    return {
        file.name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
        for file in sorted(path.iterdir())
        if file.is_file()
    }


def materialize(first: Path, second: Path, output: Path, weight_second: float) -> Mapping[str, Any]:
    if not 0.0 < weight_second < 1.0:
        raise ValueError("Blend weight must be strictly between zero and one")
    if output.exists():
        raise ValueError(f"Blend output must be fresh: {output}")
    first = first.expanduser().resolve(strict=True)
    second = second.expanduser().resolve(strict=True)
    first_config = first / "delta_mem_config.json"
    second_config = second / "delta_mem_config.json"
    if first_config.read_bytes() != second_config.read_bytes():
        raise ValueError("Adapter configurations differ")
    first_state = torch.load(first / "delta_mem_adapter.pt", map_location="cpu", weights_only=True)
    second_state = torch.load(second / "delta_mem_adapter.pt", map_location="cpu", weights_only=True)
    if set(first_state) != set(second_state):
        raise ValueError("Adapter parameter names differ")
    mixed = {}
    for name in sorted(first_state):
        left = first_state[name]
        right = second_state[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
            raise ValueError(f"Adapter tensor differs: {name}")
        mixed[name] = (left.float() * (1.0 - weight_second) + right.float() * weight_second).to(left.dtype)
    output.mkdir(parents=True)
    adapter = output / "adapter"
    adapter.mkdir()
    torch.save(mixed, adapter / "delta_mem_adapter.pt")
    (adapter / "delta_mem_config.json").write_bytes(first_config.read_bytes())
    adapter_files = snapshot_directory(adapter)
    binding = {
        "schema": SCHEMA + "_input",
        "first_adapter": str(first),
        "second_adapter": str(second),
        "first_adapter_files": snapshot_directory(first),
        "second_adapter_files": snapshot_directory(second),
        "second_weight": weight_second,
        "final_rows_opened": False,
    }
    result = {
        "schema": SCHEMA,
        "status": "adapter_blend_complete_development_v2_evaluation_authorized",
        "passed": True,
        "final_rows_opened": False,
        "open_development_evaluation_authorized": True,
        "adapter_files": adapter_files,
        "adapter_files_sha256": canonical_sha256(adapter_files),
        "input_binding": binding,
        "receipt": {"algorithm": "sha256", "payload_scope": "canonical_result_without_receipt"},
    }
    unsigned_result = dict(result)
    unsigned_result.pop("receipt")
    result["receipt"]["payload_sha256"] = canonical_sha256(unsigned_result)
    (output / "input_binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, default=0.5)
    args = parser.parse_args()
    result = materialize(args.first, args.second, args.output_dir.expanduser().resolve(), args.second_weight)
    print(json.dumps({"status": result["status"], "receipt": result["receipt"]["payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
