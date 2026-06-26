#!/usr/bin/env python3
"""Inspect and optionally validate a RWKV-MS memory GGUF sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_GGUF = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_GGUF_PY_ROOT = Path("/run/media/xiaol/B214449214445C0B/tools/llama.cpp/gguf-py")
DEFAULT_MEMORY_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/"
    "gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a delta-Mem RWKV-MS GGUF sidecar.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--gguf-py-root", type=Path, default=DEFAULT_GGUF_PY_ROOT)
    parser.add_argument("--memory-dir", type=Path, default=None, help="Optional original checkpoint dir for byte validation.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a compact text summary.")
    return parser.parse_args()


def add_gguf_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "gguf").is_dir():
        raise FileNotFoundError(f"{root} does not look like llama.cpp gguf-py")
    sys.path.insert(0, str(root))


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def field_to_py(field: Any) -> Any:
    # ReaderField.data points at the payload part indexes, excluding type tags.
    if len(field.data) == 1:
        part = field.parts[field.data[0]]
        if getattr(part, "dtype", None) is not None and part.dtype.kind == "u" and field.types[-1].name == "STRING":
            return bytes(part).decode("utf-8")
        if len(part) == 1:
            return scalar(part[0])
        return [scalar(item) for item in part]
    values = []
    for index in field.data:
        part = field.parts[index]
        if getattr(part, "dtype", None) is not None and part.dtype.kind == "u" and field.types[-1].name == "STRING":
            values.append(bytes(part).decode("utf-8"))
        elif len(part) == 1:
            values.append(scalar(part[0]))
        else:
            values.append([scalar(item) for item in part])
    return values


def inspect_sidecar(path: Path, gguf_py_root: Path) -> dict[str, Any]:
    add_gguf_to_path(gguf_py_root)
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(path)
    fields = {key: field_to_py(field) for key, field in reader.fields.items()}
    tensors = [
        {
            "name": tensor.name,
            "gguf_type": tensor.tensor_type.name,
            "logical_shape": list(reversed(tensor.shape.tolist())),
            "reader_shape": tensor.shape.tolist(),
            "data_shape": list(tensor.data.shape),
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
        }
        for tensor in reader.tensors
    ]
    manifest_json = fields.get("delta_mem.tensor_manifest_json", "[]")
    try:
        tensor_manifest = json.loads(manifest_json)
    except json.JSONDecodeError:
        tensor_manifest = []
    expected_by_name = {item["name"]: item for item in tensor_manifest if isinstance(item, dict) and "name" in item}
    per_layer_attention = json.loads(fields.get("delta_mem.per_layer_attention_json", "[]"))
    validation_errors: list[str] = []
    for tensor in tensors:
        expected = expected_by_name.get(tensor["name"])
        if expected is None:
            validation_errors.append(f"tensor missing from embedded manifest: {tensor['name']}")
            continue
        if list(expected.get("shape", [])) != tensor["logical_shape"]:
            validation_errors.append(
                f"shape mismatch for {tensor['name']}: gguf={tensor['logical_shape']} manifest={expected.get('shape')}"
            )
        if str(expected.get("gguf_type")) != tensor["gguf_type"]:
            validation_errors.append(
                f"type mismatch for {tensor['name']}: gguf={tensor['gguf_type']} manifest={expected.get('gguf_type')}"
            )
    if len(expected_by_name) != len(tensors):
        validation_errors.append(f"tensor count mismatch: gguf={len(tensors)} manifest={len(expected_by_name)}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "fields": fields,
        "summary": {
            "architecture": fields.get("general.architecture"),
            "schema": fields.get("delta_mem.schema"),
            "status": fields.get("delta_mem.status"),
            "base_model": fields.get("delta_mem.base_model"),
            "base_gguf": fields.get("delta_mem.base_gguf"),
            "base_gguf_sha256": fields.get("delta_mem.base_gguf_sha256"),
            "mmproj_gguf": fields.get("delta_mem.mmproj_gguf"),
            "mmproj_gguf_sha256": fields.get("delta_mem.mmproj_gguf_sha256"),
            "memory_backend": fields.get("delta_mem.memory_backend"),
            "rank": fields.get("delta_mem.rank"),
            "alpha": fields.get("delta_mem.alpha"),
            "target_layers": json.loads(fields.get("delta_mem.target_layers_json", "[]")),
            "delta_heads": json.loads(fields.get("delta_mem.delta_heads_json", "[]")),
            "per_layer_attention": per_layer_attention,
            "tensor_count": len(tensors),
            "total_tensor_bytes": sum(item["n_bytes"] for item in tensors),
            "validation_ok": not validation_errors,
        },
        "validation_errors": validation_errors,
        "tensors": tensors,
    }


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


def validate_against_checkpoint(result: dict[str, Any], memory_dir: Path) -> list[str]:
    errors: list[str] = []
    adapter_path = memory_dir / "delta_mem_adapter.pt"
    if not adapter_path.is_file():
        return [f"missing adapter checkpoint: {adapter_path}"]
    expected = {
        name: hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, tensor in iter_tensors(load_adapter(adapter_path))
    }
    actual = {
        tensor.get("source_name", tensor["name"]): tensor.get("sha256_raw")
        for tensor in json.loads(result["fields"].get("delta_mem.tensor_manifest_json", "[]"))
    }
    for name, expected_hash in expected.items():
        actual_hash = actual.get(name)
        if actual_hash != expected_hash:
            errors.append(f"raw hash mismatch for {name}: gguf={actual_hash} checkpoint={expected_hash}")
    for name in actual:
        if name not in expected:
            errors.append(f"extra tensor in GGUF manifest: {name}")
    return errors


def main() -> None:
    args = parse_args()
    result = inspect_sidecar(args.gguf.expanduser().resolve(), args.gguf_py_root)
    if args.memory_dir is not None:
        checkpoint_errors = validate_against_checkpoint(result, args.memory_dir.expanduser().resolve())
        result["checkpoint_validation_errors"] = checkpoint_errors
        result["summary"]["checkpoint_validation_ok"] = not checkpoint_errors
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        summary = result["summary"]
        print(f"path: {result['path']}")
        print(f"schema: {summary['schema']}")
        print(f"status: {summary['status']}")
        print(f"base_model: {summary['base_model']}")
        print(f"memory_backend: {summary['memory_backend']}")
        print(f"target_layers: {summary['target_layers']}")
        print(f"delta_heads: {summary['delta_heads']}")
        print(f"per_layer_attention: {summary['per_layer_attention']}")
        print(f"rank/alpha: {summary['rank']} / {summary['alpha']}")
        print(f"tensors: {summary['tensor_count']}")
        print(f"tensor_bytes: {summary['total_tensor_bytes']}")
        print(f"validation_ok: {summary['validation_ok']}")
        if "checkpoint_validation_ok" in summary:
            print(f"checkpoint_validation_ok: {summary['checkpoint_validation_ok']}")
            for error in result["checkpoint_validation_errors"]:
                print(f"checkpoint_error: {error}")
        if result["validation_errors"]:
            for error in result["validation_errors"]:
                print(f"error: {error}")


if __name__ == "__main__":
    main()
