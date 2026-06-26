#!/usr/bin/env python3
"""Compare two delta-Mem adapter checkpoint directories at tensor byte level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_ORIGINAL = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/"
    "gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"
)
DEFAULT_REBUILT = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/from_gguf/"
    "gemma-4-E4B-it-rwkv-ms-memory"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two delta-Mem checkpoint directories.")
    parser.add_argument("--left", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--right", type=Path, default=DEFAULT_REBUILT)
    return parser.parse_args()


def load_adapter(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    if torch.is_tensor(obj):
        return {prefix or "<root>": obj}
    result: dict[str, torch.Tensor] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(iter_tensors(value, name))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            name = f"{prefix}.{index}" if prefix else str(index)
            result.update(iter_tensors(value, name))
    return result


def raw_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left = iter_tensors(load_adapter(left_dir / "delta_mem_adapter.pt"))
    right = iter_tensors(load_adapter(right_dir / "delta_mem_adapter.pt"))
    errors: list[str] = []
    for name in sorted(set(left) | set(right)):
        if name not in left:
            errors.append(f"missing left tensor: {name}")
            continue
        if name not in right:
            errors.append(f"missing right tensor: {name}")
            continue
        lt = left[name].detach().cpu().contiguous()
        rt = right[name].detach().cpu().contiguous()
        if tuple(lt.shape) != tuple(rt.shape):
            errors.append(f"shape mismatch {name}: {tuple(lt.shape)} != {tuple(rt.shape)}")
        if lt.dtype != rt.dtype:
            errors.append(f"dtype mismatch {name}: {lt.dtype} != {rt.dtype}")
        if raw_hash(lt) != raw_hash(rt):
            errors.append(f"raw bytes mismatch {name}")
    return {
        "left": str(left_dir),
        "right": str(right_dir),
        "tensor_count_left": len(left),
        "tensor_count_right": len(right),
        "total_numel_left": sum(int(t.numel()) for t in left.values()),
        "total_numel_right": sum(int(t.numel()) for t in right.values()),
        "ok": not errors,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    result = compare(args.left.expanduser().resolve(), args.right.expanduser().resolve())
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
