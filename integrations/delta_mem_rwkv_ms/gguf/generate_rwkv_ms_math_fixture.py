#!/usr/bin/env python3
"""Generate a deterministic RWKV-MS math fixture from the Gemma4 memory sidecar.

The fixture is a PyTorch golden reference for the future GGML port. It uses the
sidecar-rebuilt delta-Mem checkpoint and does not imply stock llama.cpp can
execute the RWKV-MS memory yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rwkv_ms_math_fixture_common import (
    DEFAULT_MEMORY_DIR,
    DEFAULT_OUTPUT,
    build_summary,
    compute_expected_tensors,
    dtype_from_name,
    dtype_name,
    file_sha256,
    load_layer_artifacts,
    make_deterministic_inputs,
    parse_token_mask,
    records_from_tensors,
    tensor_record,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a RWKV-MS PyTorch math fixture for GGML parity work.")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=0, help="Wrapped Gemma4 layer to exercise.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--initial-position", type=int, default=None)
    parser.add_argument(
        "--token-mask",
        default="auto",
        help="none, auto, or a comma-separated mask such as 1,1,0,1. Auto inserts one invalid token.",
    )
    parser.add_argument("--rwkv-ms-read-top-k", type=int, default=None)
    parser.add_argument("--include-step-trace", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON.")
    return parser.parse_args()


def scan_config_from_checkpoint(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    read_top_k = int(config.get("rwkv_ms_read_top_k", 0))
    if args.rwkv_ms_read_top_k is not None:
        read_top_k = int(args.rwkv_ms_read_top_k)
    return {
        "rwkv_ms_num_states": int(config["rwkv_ms_num_states"]),
        "rwkv_ms_chunk_size": int(config["rwkv_ms_chunk_size"]),
        "rwkv_ms_boundary_mode": str(config.get("rwkv_ms_boundary_mode", "fixed_chunk")),
        "rwkv_ms_erase_gate": float(config.get("rwkv_ms_erase_gate", 1.0)),
        "rwkv_ms_read_top_k": read_top_k,
    }


def default_initial_position(chunk_size: int, seq_len: int) -> int:
    if seq_len <= 1:
        return 0
    return max(0, int(chunk_size) - min(2, int(seq_len) - 1))


def generate_fixture(args: argparse.Namespace) -> dict[str, Any]:
    dtype = dtype_from_name(args.dtype)
    layer_artifacts = load_layer_artifacts(args.memory_dir, args.layer, dtype)
    config = layer_artifacts["config"]
    metadata = layer_artifacts["metadata"]
    scan_config = scan_config_from_checkpoint(config, args)
    if scan_config["rwkv_ms_boundary_mode"] != "fixed_chunk":
        raise ValueError(f"Unsupported RWKV-MS boundary mode: {scan_config['rwkv_ms_boundary_mode']}")

    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    if batch_size < 1 or seq_len < 1:
        raise ValueError("--batch-size and --seq-len must be >= 1")
    initial_position = args.initial_position
    if initial_position is None:
        initial_position = default_initial_position(scan_config["rwkv_ms_chunk_size"], seq_len)

    token_mask = parse_token_mask(args.token_mask, batch_size=batch_size, seq_len=seq_len)
    inputs = make_deterministic_inputs(
        seed=args.seed,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=int(layer_artifacts["hidden_size"]),
        num_state_heads=int(layer_artifacts["num_state_heads"]),
        num_states=int(scan_config["rwkv_ms_num_states"]),
        rank=int(layer_artifacts["rank"]),
        dtype=dtype,
        token_mask=token_mask,
        initial_position=int(initial_position),
    )
    expected = compute_expected_tensors(
        layer_artifacts=layer_artifacts,
        inputs=inputs,
        scan_config=scan_config,
        include_step_trace=bool(args.include_step_trace),
    )

    input_records = {
        "hidden_states": tensor_record(inputs["hidden_states"]),  # type: ignore[arg-type]
        "initial_state": tensor_record(inputs["initial_state"]),  # type: ignore[arg-type]
        "initial_positions": tensor_record(inputs["initial_positions"]),  # type: ignore[arg-type]
    }
    if inputs["token_mask"] is not None:
        input_records["token_mask"] = tensor_record(inputs["token_mask"])  # type: ignore[arg-type]
    expected_records = records_from_tensors(expected)
    fixture = {
        "schema": "delta_mem_rwkv_ms_math_fixture.v1",
        "created_at": utc_now(),
        "source": {
            "memory_dir": str(layer_artifacts["memory_dir"]),
            "adapter_sha256": file_sha256(layer_artifacts["memory_dir"] / "delta_mem_adapter.pt"),
            "config_sha256": file_sha256(layer_artifacts["memory_dir"] / "delta_mem_config.json"),
            "metadata_sha256": file_sha256(layer_artifacts["memory_dir"] / "adapter_metadata.json"),
            "base_model": metadata.get("base_model"),
            "memory_backend": config.get("memory_backend"),
        },
        "config": {
            "layer": int(args.layer),
            "seed": int(args.seed),
            "dtype": dtype_name(dtype),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": int(layer_artifacts["hidden_size"]),
            "rank": int(layer_artifacts["rank"]),
            "num_state_heads": int(layer_artifacts["num_state_heads"]),
            "state_read_dim": int(layer_artifacts["state_read_dim"]),
            "initial_position": int(initial_position),
            "token_mask": args.token_mask,
            "include_step_trace": bool(args.include_step_trace),
        },
        "delta_mem_config": {
            "alpha": float(config["alpha"]),
            "delta_heads": config.get("delta_heads", []),
            "normalize_qk": bool(config.get("normalize_qk", True)),
            "couple_lambda": bool(config.get("couple_lambda", True)),
            "rankwise_gates": bool(config.get("rankwise_gates", True)),
            "state_update_mode": str(config.get("state_update_mode", "standard")),
            "trainable_delta_scale": bool(config.get("trainable_delta_scale", False)),
        },
        "scan_config": scan_config,
        "adapter_layer_tensors": {
            "count": len(layer_artifacts["tensor_names"]),
            "names": layer_artifacts["tensor_names"],
            "sha256": layer_artifacts["tensor_hashes"],
        },
        "inputs": input_records,
        "expected": expected_records,
    }
    fixture["summary"] = build_summary(fixture["inputs"], fixture["expected"])
    return fixture


def main() -> None:
    args = parse_args()
    fixture = generate_fixture(args)
    write_json(args.output.expanduser(), fixture, compact=args.compact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": fixture["schema"],
                "layer": fixture["config"]["layer"],
                "dtype": fixture["config"]["dtype"],
                "tensor_count": fixture["summary"]["tensor_count"],
                "total_values": fixture["summary"]["total_values"],
                "slot_indices": fixture["expected"]["slot_indices"]["data"],
                "final_positions": fixture["expected"]["final_positions"]["data"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
