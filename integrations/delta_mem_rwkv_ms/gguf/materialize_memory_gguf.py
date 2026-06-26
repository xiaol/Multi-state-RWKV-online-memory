#!/usr/bin/env python3
"""Materialize a delta-Mem checkpoint directory from a RWKV-MS GGUF sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_GGUF = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/from_gguf/"
    "gemma-4-E4B-it-rwkv-ms-memory"
)
DEFAULT_GGUF_PY_ROOT = Path("/run/media/xiaol/B214449214445C0B/tools/llama.cpp/gguf-py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild delta-Mem checkpoint files from a GGUF memory sidecar.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gguf-py-root", type=Path, default=DEFAULT_GGUF_PY_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def add_gguf_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "gguf").is_dir():
        raise FileNotFoundError(f"{root} does not look like llama.cpp gguf-py")
    sys.path.insert(0, str(root))


def field_to_py(field: Any) -> Any:
    if len(field.data) == 1:
        part = field.parts[field.data[0]]
        if getattr(part, "dtype", None) is not None and part.dtype.kind == "u" and field.types[-1].name == "STRING":
            return bytes(part).decode("utf-8")
        if len(part) == 1:
            value = part[0]
            return value.item() if hasattr(value, "item") else value
        return [(item.item() if hasattr(item, "item") else item) for item in part]
    return [field_to_py(type("FieldPart", (), {"data": [idx], "parts": field.parts, "types": field.types})()) for idx in field.data]


def tensor_manifest(fields: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = json.loads(fields.get("delta_mem.tensor_manifest_json", "[]"))
    return {entry["name"]: entry for entry in manifest}


def source_name(entry: dict[str, Any]) -> str:
    return str(entry.get("source_name", entry["name"]))


def tensor_to_torch(reader_tensor: Any, expected: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in expected["shape"])
    gguf_type = expected["gguf_type"]
    data = np.array(reader_tensor.data, copy=True)
    if gguf_type == "BF16":
        return torch.from_numpy(data).view(torch.bfloat16).reshape(shape).contiguous()
    if gguf_type == "F16":
        return torch.from_numpy(data.astype(np.float16, copy=False)).reshape(shape).contiguous()
    if gguf_type == "F32":
        return torch.from_numpy(data.astype(np.float32, copy=False)).reshape(shape).contiguous()
    if gguf_type == "F64":
        return torch.from_numpy(data.astype(np.float64, copy=False)).reshape(shape).contiguous()
    if gguf_type == "I8":
        return torch.from_numpy(data.astype(np.int8, copy=False)).reshape(shape).contiguous()
    if gguf_type == "I16":
        return torch.from_numpy(data.astype(np.int16, copy=False)).reshape(shape).contiguous()
    if gguf_type == "I32":
        return torch.from_numpy(data.astype(np.int32, copy=False)).reshape(shape).contiguous()
    if gguf_type == "I64":
        return torch.from_numpy(data.astype(np.int64, copy=False)).reshape(shape).contiguous()
    raise ValueError(f"Unsupported sidecar tensor type: {gguf_type}")


def raw_tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    add_gguf_to_path(args.gguf_py_root)
    from gguf.gguf_reader import GGUFReader

    gguf_path = args.gguf.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reader = GGUFReader(gguf_path)
    fields = {key: field_to_py(field) for key, field in reader.fields.items()}
    expected = tensor_manifest(fields)

    config = json.loads(fields["delta_mem.config_json"])
    metadata = json.loads(fields["delta_mem.adapter_metadata_json"])
    state_dict: dict[str, torch.Tensor] = {}
    errors: list[str] = []
    for reader_tensor in reader.tensors:
        entry = expected.get(reader_tensor.name)
        if entry is None:
            errors.append(f"tensor missing from embedded manifest: {reader_tensor.name}")
            continue
        tensor = tensor_to_torch(reader_tensor, entry)
        actual_hash = raw_tensor_sha256(tensor)
        expected_hash = entry.get("sha256_raw")
        if actual_hash != expected_hash:
            errors.append(f"raw hash mismatch for {reader_tensor.name}: {actual_hash} != {expected_hash}")
        state_dict[source_name(entry)] = tensor
    missing = sorted(
        source_name(entry)
        for name, entry in expected.items()
        if source_name(entry) not in state_dict
    )
    if missing:
        errors.extend(f"missing tensor from GGUF payload: {name}" for name in missing)
    if errors:
        raise ValueError("GGUF sidecar validation failed:\n" + "\n".join(errors[:20]))

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "delta_mem_config.json": output_dir / "delta_mem_config.json",
        "adapter_metadata.json": output_dir / "adapter_metadata.json",
        "delta_mem_adapter.pt": output_dir / "delta_mem_adapter.pt",
    }
    if not args.force:
        existing = [str(path) for path in targets.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing files without --force: " + ", ".join(existing))

    targets["delta_mem_config.json"].write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    targets["adapter_metadata.json"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(state_dict, targets["delta_mem_adapter.pt"])
    return {
        "output_dir": str(output_dir),
        "tensor_count": len(state_dict),
        "total_numel": sum(int(tensor.numel()) for tensor in state_dict.values()),
        "total_bytes": sum(int(tensor.numel() * tensor.element_size()) for tensor in state_dict.values()),
        "files": {name: str(path) for name, path in targets.items()},
    }


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2))


if __name__ == "__main__":
    main()
