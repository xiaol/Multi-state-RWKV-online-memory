#!/usr/bin/env python3
"""Inspect a delta-Mem online-memory checkpoint for a future GGUF/GGML port.

The output is a JSON manifest of config values and tensor inventory. It does not
convert tensors to GGUF and does not make llama.cpp run RWKV-MS memory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


DEFAULT_MEMORY_REPO = "xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit a JSON manifest for a delta-Mem memory checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--memory-dir", type=Path, help="Local checkpoint directory.")
    source.add_argument("--memory-repo", default=None, help="HF repo id to download with huggingface_hub.")
    parser.add_argument("--output", type=Path, default=None, help="Manifest JSON path. Defaults to stdout.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache dir for --memory-repo.")
    return parser.parse_args()


def resolve_memory_dir(args: argparse.Namespace) -> Path:
    if args.memory_dir is not None:
        return args.memory_dir.expanduser().resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise SystemExit("Install huggingface_hub or pass --memory-dir.") from exc
    return Path(snapshot_download(repo_id=args.memory_repo or DEFAULT_MEMORY_REPO, cache_dir=args.cache_dir)).resolve()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_adapter(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_tensors(obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = []
    if torch.is_tensor(obj):
        tensors.append((prefix or "<root>", obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            tensors.extend(iter_tensors(value, name))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            name = f"{prefix}.{index}" if prefix else str(index)
            tensors.extend(iter_tensors(value, name))
    return tensors


def tensor_entry(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
    }


def infer_layer_ids(tensor_names: list[str]) -> list[int]:
    ids: set[int] = set()
    patterns = [
        re.compile(r"(?:layers|layer|model\.layers)\.(\d+)"),
        re.compile(r"(?:layers|layer)_(\d+)"),
    ]
    for name in tensor_names:
        for pattern in patterns:
            match = pattern.search(name)
            if match:
                ids.add(int(match.group(1)))
    return sorted(ids)


def summarize_dtype(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        dtype = entry["dtype"]
        bucket = summary.setdefault(dtype, {"tensors": 0, "numel": 0, "bytes": 0})
        bucket["tensors"] += 1
        bucket["numel"] += int(entry["numel"])
        bucket["bytes"] += int(entry["bytes"])
    return summary


def build_manifest(memory_dir: Path) -> dict[str, Any]:
    config = read_json(memory_dir / "delta_mem_config.json")
    metadata = read_json(memory_dir / "adapter_metadata.json")
    adapter_path = memory_dir / "delta_mem_adapter.pt"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing {adapter_path}")
    tensors = [tensor_entry(name, tensor) for name, tensor in iter_tensors(load_adapter(adapter_path))]
    tensor_names = [entry["name"] for entry in tensors]
    return {
        "schema": "delta_mem_rwkv_ms_memory_manifest.v1",
        "memory_dir": str(memory_dir),
        "files": {
            "delta_mem_config.json": (memory_dir / "delta_mem_config.json").is_file(),
            "adapter_metadata.json": (memory_dir / "adapter_metadata.json").is_file(),
            "delta_mem_adapter.pt": True,
        },
        "config": config,
        "adapter_metadata": metadata,
        "tensor_summary": {
            "tensor_count": len(tensors),
            "total_numel": sum(int(entry["numel"]) for entry in tensors),
            "total_bytes": sum(int(entry["bytes"]) for entry in tensors),
            "by_dtype": summarize_dtype(tensors),
            "inferred_layer_ids": infer_layer_ids(tensor_names),
        },
        "tensors": tensors,
        "gguf_port_notes": [
            "This manifest is an inventory only; llama.cpp cannot consume it yet.",
            "A real port must implement RWKV-MS state, Gemma4 layer hooks, q/o delta injection, and session-state sync in GGML.",
        ],
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(resolve_memory_dir(args))
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
