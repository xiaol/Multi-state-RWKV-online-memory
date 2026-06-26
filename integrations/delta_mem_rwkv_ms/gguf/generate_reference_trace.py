#!/usr/bin/env python3
"""Generate a PyTorch delta-Mem reference trace for future GGUF/GGML parity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_DELTA_MEM_ROOT = Path("/home/xiaol/X/delta-Mem")
DEFAULT_BASE_MODEL = Path("/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it")
DEFAULT_MEMORY_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/models/delta_mem/from_gguf/"
    "gemma-4-E4B-it-rwkv-ms-memory"
)
DEFAULT_MEMORY_GGUF = Path(
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_OUTPUT = Path(".openresearch/artifacts/gguf_reference_trace.json")
DEFAULT_PROMPT = """You are a telecom solo-mode tool agent. Return exactly one tool call in this format:
[ACTION]
tool_name(arg_name="value")
[/ACTION]

Available tools:
- get_customer_by_phone(phone_number: str)
- check_network_status(line_id: str)
- toggle_data(line_id: str, enabled: bool)
- run_speed_test(line_id: str)
- done()

Ticket: Customer phone number 555-123-2002 reports no usable mobile data.
First step: identify the customer account from the phone number. Return only the next tool call."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch delta-Mem reference and write a trace JSON.")
    parser.add_argument("--delta-mem-root", type=Path, default=DEFAULT_DELTA_MEM_ROOT)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--memory-gguf", type=Path, default=DEFAULT_MEMORY_GGUF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--write-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-snapshot-dir", type=Path, default=None)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_delta_mem_root(path: Path) -> None:
    if not (path / "deltamem").is_dir():
        raise FileNotFoundError(f"{path} does not look like a delta-Mem checkout")
    sys.path.insert(0, str(path.resolve()))


def require_memory_dir(path: Path) -> None:
    missing = [
        name
        for name in ("delta_mem_config.json", "adapter_metadata.json", "delta_mem_adapter.pt")
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{path} is missing: {', '.join(missing)}")


def tensor_state_summary(snapshot_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    entries = []
    for name, tensor in sorted(snapshot_state.items()):
        tensor_cpu = tensor.detach().cpu()
        entries.append(
            {
                "name": name,
                "shape": list(tensor_cpu.shape),
                "dtype": str(tensor_cpu.dtype).replace("torch.", ""),
                "numel": int(tensor_cpu.numel()),
                "sum_abs": float(tensor_cpu.float().abs().sum().item()) if tensor_cpu.numel() else 0.0,
            }
        )
    return {
        "tensor_count": len(entries),
        "total_numel": sum(item["numel"] for item in entries),
        "tensors": entries,
    }


def main() -> None:
    args = parse_args()
    delta_mem_root = args.delta_mem_root.expanduser().resolve()
    memory_dir = args.memory_dir.expanduser().resolve()
    require_delta_mem_root(delta_mem_root)
    require_memory_dir(memory_dir)

    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model

    started = time.perf_counter()
    model, tokenizer = load_delta_mem_chat_model(
        model_path=args.base_model,
        adapter_dir=memory_dir,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    result = session.generate_reply(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_seed=args.sample_seed,
        write_enabled=args.write_enabled,
        include_debug=True,
    )
    snapshot = session.snapshot()
    if args.save_snapshot_dir is not None:
        session.save_snapshot(args.save_snapshot_dir)

    trace = {
        "schema": "delta_mem_rwkv_ms_reference_trace.v1",
        "created_at": utc_now(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "source": {
            "delta_mem_root": str(delta_mem_root),
            "base_model": args.base_model,
            "memory_dir": str(memory_dir),
            "memory_gguf": str(args.memory_gguf.expanduser().resolve()),
        },
        "generation": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.sample),
            "sample_seed": args.sample_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "write_enabled": bool(args.write_enabled),
        },
        "result": result,
        "session": {
            "messages": snapshot.messages,
            "processed_input_ids": snapshot.processed_input_ids,
            "processed_token_count": len(snapshot.processed_input_ids),
            "write_message_ids": snapshot.write_message_ids,
            "write_sentence_ids": snapshot.write_sentence_ids,
            "delta_state_summary": tensor_state_summary(snapshot.delta_state),
            "snapshot_dir": None if args.save_snapshot_dir is None else str(args.save_snapshot_dir),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "assistant": result.get("assistant_display"), "elapsed_ms": trace["elapsed_ms"]}, indent=2))


if __name__ == "__main__":
    main()
