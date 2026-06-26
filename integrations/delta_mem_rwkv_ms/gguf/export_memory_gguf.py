#!/usr/bin/env python3
"""Export a delta-Mem RWKV-MS memory checkpoint to a GGUF sidecar.

This creates a real GGUF tensor container for the online-memory adapter. It is
still a sidecar artifact: llama.cpp cannot use it until the RWKV-MS runtime hooks
described in GGUF_PORT_PLAN.md are implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_GGUF_PY_ROOT = Path("/run/media/xiaol/B214449214445C0B/tools/llama.cpp/gguf-py")
DEFAULT_MEMORY_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/"
    "gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"
)
DEFAULT_OUTPUT = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_BASE_GGUF = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-Q8_0.gguf"
)
DEFAULT_MMPROJ_GGUF = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "mmproj-gemma-4-E4B-it-Q8_0.gguf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export delta-Mem RWKV-MS adapter tensors to a GGUF sidecar.")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--gguf-py-root", type=Path, default=DEFAULT_GGUF_PY_ROOT)
    parser.add_argument("--base-gguf", type=Path, default=DEFAULT_BASE_GGUF)
    parser.add_argument("--mmproj-gguf", type=Path, default=DEFAULT_MMPROJ_GGUF)
    parser.add_argument("--checkpoint-repo", default="xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1")
    parser.add_argument("--source-repo", default="https://github.com/xiaol/Multi-state-RWKV-online-memory")
    return parser.parse_args()


def add_gguf_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "gguf").is_dir():
        raise FileNotFoundError(f"{root} does not look like llama.cpp gguf-py")
    sys.path.insert(0, str(root))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tensor_to_gguf_payload(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...] | None, Any | None, str]:
    from gguf import GGMLQuantizationType

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        # gguf-py stores BF16 as a raw byte tensor with raw_dtype=BF16.
        raw = tensor.view(torch.uint8).numpy()
        return raw, tuple(raw.shape), GGMLQuantizationType.BF16, "BF16"
    if tensor.dtype == torch.float16:
        return tensor.numpy(), None, None, "F16"
    if tensor.dtype == torch.float32:
        return tensor.numpy(), None, None, "F32"
    if tensor.dtype == torch.float64:
        return tensor.numpy(), None, None, "F64"
    if tensor.dtype == torch.int8:
        return tensor.numpy(), None, None, "I8"
    if tensor.dtype == torch.int16:
        return tensor.numpy(), None, None, "I16"
    if tensor.dtype == torch.int32:
        return tensor.numpy(), None, None, "I32"
    if tensor.dtype == torch.int64:
        return tensor.numpy(), None, None, "I64"
    raise TypeError(f"Unsupported tensor dtype for GGUF export: {tensor.dtype}")


COMPACT_TENSOR_NAMES = {
    "memory_q_proj": "mem_q",
    "memory_k_proj": "mem_k",
    "memory_v_proj": "mem_v",
    "beta_proj": "beta",
    "beta_bias": "beta_b",
    "lambda_proj": "lambda",
    "lambda_bias": "lambda_b",
    "delta_q_proj": "delta_q",
    "delta_k_proj": "delta_k",
    "delta_v_proj": "delta_v",
    "delta_o_proj": "delta_o",
}


def compact_tensor_name(source_name: str) -> str:
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.self_attn\.(.+)", source_name)
    if match is None:
        digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:16]
        return f"tensor.{digest}"
    layer = match.group(1)
    tail = match.group(2)
    if tail.startswith("hrm_rwkv7_core."):
        name = f"blk.{layer}.rwkv.{tail.removeprefix('hrm_rwkv7_core.')}"
    else:
        name = f"blk.{layer}.{COMPACT_TENSOR_NAMES.get(tail, tail)}"
    if len(name.encode("utf-8")) >= 64:
        raise ValueError(f"compact tensor name is too long for llama.cpp GGUF reader: {name}")
    return name


def tensor_entry(name: str, source_name: str, tensor: torch.Tensor, gguf_type: str) -> dict[str, Any]:
    return {
        "name": name,
        "source_name": source_name,
        "shape": list(tensor.shape),
        "torch_dtype": str(tensor.dtype).replace("torch.", ""),
        "gguf_type": gguf_type,
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "sha256_raw": hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def build_manifest(
    *,
    memory_dir: Path,
    output: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    tensor_entries: list[dict[str, Any]],
    base_gguf: Path,
    mmproj_gguf: Path,
    checkpoint_repo: str,
    source_repo: str,
) -> dict[str, Any]:
    per_layer_attention = infer_per_layer_attention(tensor_entries, config.get("target_layers", []))
    return {
        "schema": "delta_mem_rwkv_ms_memory_gguf_sidecar.v1",
        "status": "sidecar_only_runtime_port_required",
        "memory_dir": str(memory_dir),
        "output": str(output),
        "checkpoint_repo": checkpoint_repo,
        "source_repo": source_repo,
        "base_gguf": str(base_gguf),
        "base_gguf_sha256": sha256_file(base_gguf),
        "mmproj_gguf": str(mmproj_gguf),
        "mmproj_gguf_sha256": sha256_file(mmproj_gguf),
        "config": config,
        "adapter_metadata": metadata,
        "summary": {
            "tensor_count": len(tensor_entries),
            "total_numel": sum(int(entry["numel"]) for entry in tensor_entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in tensor_entries),
            "target_layers": config.get("target_layers"),
            "delta_heads": config.get("delta_heads"),
            "rank": config.get("rank"),
            "alpha": config.get("alpha"),
            "memory_backend": config.get("memory_backend"),
            "rwkv_ms_num_states": config.get("rwkv_ms_num_states"),
            "rwkv_ms_chunk_size": config.get("rwkv_ms_chunk_size"),
        },
        "per_layer_attention": per_layer_attention,
        "tensors": tensor_entries,
    }


def infer_per_layer_attention(tensor_entries: list[dict[str, Any]], target_layers: Any) -> list[dict[str, Any]]:
    by_name = {entry.get("source_name", entry["name"]): entry for entry in tensor_entries}
    layers = [int(layer) for layer in target_layers] if isinstance(target_layers, list) else []
    result: list[dict[str, Any]] = []
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}.self_attn."

        def rows(name: str) -> int:
            shape = by_name.get(prefix + name, {}).get("shape", [])
            return int(shape[0]) if shape else 0

        q_out = rows("delta_q_proj")
        k_out = rows("delta_k_proj")
        v_out = rows("delta_v_proj")
        o_out = rows("delta_o_proj")
        result.append(
            {
                "layer": layer,
                "attention_kind": "full_attention" if q_out > 2048 else "sliding_attention",
                "q_out": q_out,
                "k_out": k_out,
                "v_out": v_out,
                "o_out": o_out,
                "active_delta_heads": ["q", "o"],
                "compat_delta_heads": ["k", "v"],
            }
        )
    return result


def add_metadata(writer: Any, manifest: dict[str, Any]) -> None:
    config = manifest["config"]
    metadata = manifest["adapter_metadata"] or {}
    summary = manifest["summary"]

    writer.add_name("Gemma4 E4B RWKV-MS delta-Mem memory sidecar")
    writer.add_description(
        "RWKV-MS online-memory adapter tensors for Gemma4 E4B. "
        "This is a GGUF sidecar; llama.cpp runtime hooks are still required."
    )
    writer.add_source_repo_url(manifest["source_repo"])
    writer.add_string("delta_mem.schema", manifest["schema"])
    writer.add_string("delta_mem.status", manifest["status"])
    writer.add_string("delta_mem.memory_type", str(metadata.get("memory_type", "delta-mem-rwkv-ms")))
    writer.add_string("delta_mem.memory_backend", str(config.get("memory_backend", "rwkv_ms")))
    writer.add_string("delta_mem.memory_readout_mode", str(config.get("memory_readout_mode", "")))
    writer.add_string("delta_mem.base_model", str(metadata.get("base_model", "google/gemma-4-E4B-it")))
    writer.add_string("delta_mem.checkpoint_repo", str(manifest["checkpoint_repo"]))
    writer.add_string("delta_mem.base_gguf", str(manifest["base_gguf"]))
    writer.add_string("delta_mem.base_gguf_sha256", str(manifest["base_gguf_sha256"] or ""))
    writer.add_string("delta_mem.mmproj_gguf", str(manifest["mmproj_gguf"]))
    writer.add_string("delta_mem.mmproj_gguf_sha256", str(manifest["mmproj_gguf_sha256"] or ""))
    writer.add_string("delta_mem.tensor_name_format", "compact_with_source_name_manifest")
    writer.add_uint32("delta_mem.format_version", 1)
    writer.add_uint32("delta_mem.tensor_count", int(summary["tensor_count"]))
    writer.add_uint64("delta_mem.total_numel", int(summary["total_numel"]))
    writer.add_uint64("delta_mem.total_bytes", int(summary["total_bytes"]))
    writer.add_uint32("delta_mem.rank", int(config.get("rank", 0)))
    writer.add_float32("delta_mem.alpha", float(config.get("alpha", 0.0)))
    writer.add_uint32("delta_mem.num_state_heads", int(config.get("num_state_heads", 0)))
    writer.add_uint32("delta_mem.rwkv_ms_num_states", int(config.get("rwkv_ms_num_states", 0)))
    writer.add_uint32("delta_mem.rwkv_ms_chunk_size", int(config.get("rwkv_ms_chunk_size", 0)))
    writer.add_string("delta_mem.rwkv_ms_boundary_mode", str(config.get("rwkv_ms_boundary_mode", "")))
    writer.add_string("delta_mem.target_layers_json", json.dumps(config.get("target_layers", []), separators=(",", ":")))
    writer.add_string("delta_mem.delta_heads_json", json.dumps(config.get("delta_heads", []), separators=(",", ":")))
    writer.add_string("delta_mem.per_layer_attention_json", json.dumps(manifest["per_layer_attention"], sort_keys=True, separators=(",", ":")))
    writer.add_string("delta_mem.config_json", json.dumps(config, sort_keys=True, separators=(",", ":")))
    writer.add_string("delta_mem.adapter_metadata_json", json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    writer.add_string("delta_mem.tensor_manifest_json", json.dumps(manifest["tensors"], sort_keys=True, separators=(",", ":")))
    writer.add_string(
        "delta_mem.required_runtime_hooks_json",
        json.dumps(
            [
                "Gemma4 text attention layers 0-5",
                "skip KV-shared tail layers",
                "read hidden states before attention delta injection",
                "RWKV-MS recurrent state buffers",
                "read-before-write token update",
                "q/o delta injection",
                "KV-cache and memory-state synchronization",
                "reset/save/load session state",
            ],
            separators=(",", ":"),
        ),
    )


def export_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    add_gguf_to_path(args.gguf_py_root)
    from gguf import GGUFWriter

    memory_dir = args.memory_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config = read_json(memory_dir / "delta_mem_config.json")
    metadata = read_json(memory_dir / "adapter_metadata.json")
    adapter = load_adapter(memory_dir / "delta_mem_adapter.pt")
    tensors = iter_tensors(adapter)
    if not tensors:
        raise ValueError(f"No tensors found in {memory_dir / 'delta_mem_adapter.pt'}")

    tensor_entries: list[dict[str, Any]] = []
    payloads: list[tuple[str, np.ndarray, tuple[int, ...] | None, Any | None]] = []
    seen_names: set[str] = set()
    for source_name, tensor in tensors:
        name = compact_tensor_name(source_name)
        if name in seen_names:
            raise ValueError(f"compact tensor name collision: {name}")
        seen_names.add(name)
        payload, raw_shape, raw_dtype, gguf_type = tensor_to_gguf_payload(tensor)
        tensor_entries.append(tensor_entry(name, source_name, tensor, gguf_type))
        payloads.append((name, payload, raw_shape, raw_dtype))

    manifest = build_manifest(
        memory_dir=memory_dir,
        output=output,
        config=config,
        metadata=metadata,
        tensor_entries=tensor_entries,
        base_gguf=args.base_gguf.expanduser().resolve(),
        mmproj_gguf=args.mmproj_gguf.expanduser().resolve(),
        checkpoint_repo=args.checkpoint_repo,
        source_repo=args.source_repo,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(output, "rwkv_ms_memory")
    writer.add_custom_alignment(32)
    add_metadata(writer, manifest)
    for name, payload, raw_shape, raw_dtype in payloads:
        writer.add_tensor(name, payload, raw_shape=raw_shape, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest["output_sha256"] = sha256_file(output)
    manifest["output_size_bytes"] = output.stat().st_size
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = export_sidecar(parse_args())
    print(json.dumps({"output": manifest["output"], "sha256": manifest["output_sha256"], **manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
