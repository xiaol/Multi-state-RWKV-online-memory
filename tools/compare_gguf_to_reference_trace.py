#!/usr/bin/env python3
"""Compare a llama.cpp GGUF endpoint against a PyTorch delta-Mem reference trace."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_TRACE = ".openresearch/artifacts/gguf_reference_trace_from_sidecar_64.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-e4b-it-q8"
DEFAULT_RWKV_MS_MODEL = "gemma-4-e4b-it-rwkv-ms-q8"
DEFAULT_RWKV_MS_SIDECAR_PATH = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_OUTPUT = ".openresearch/artifacts/gguf_ui/trace_compare_cli.jsonl"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/") + "/", api_key=os.environ.get("OPENAI_API_KEY", "llama.cpp"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local GGUF output against a saved delta-Mem reference trace.")
    parser.add_argument("--trace", type=Path, default=Path(DEFAULT_TRACE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL))
    rwkv_ms_default = env_flag("LLAMA_RWKV_MS", bool(os.environ.get("GGUF_RWKV_MS_SIDECAR_PATH")))
    parser.add_argument("--model", default=os.environ.get("LLAMA_MODEL", DEFAULT_RWKV_MS_MODEL if rwkv_ms_default else DEFAULT_MODEL))
    parser.add_argument("--rwkv-ms", action="store_true", default=rwkv_ms_default)
    parser.add_argument("--rwkv-ms-sidecar", default=os.environ.get("GGUF_RWKV_MS_SIDECAR_PATH", DEFAULT_RWKV_MS_SIDECAR_PATH))
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    if trace.get("schema") != "delta_mem_rwkv_ms_reference_trace.v1":
        raise ValueError(f"Unexpected trace schema in {path}")
    return trace


def assistant_message_text(raw_response: dict[str, Any]) -> tuple[str, str | None, str]:
    message = raw_response.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content")
    return content or reasoning or "", reasoning, content


def main() -> None:
    args = parse_args()
    trace = load_trace(args.trace)
    generation = trace["generation"]
    prompt = generation["prompt"]
    reference_text = trace["result"]["assistant_display"]
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(generation.get("max_new_tokens", 64)),
        "temperature": float(generation.get("temperature", 1.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "extra_body": {
            "top_k": int(generation.get("top_k", 0)),
            "repeat_penalty": float(args.repeat_penalty),
        },
    }
    seed = generation.get("sample_seed")
    if seed is not None:
        request["extra_body"]["seed"] = int(seed)

    started = time.perf_counter()
    row: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp": utc_now(),
        "comparison": "gguf_rwkv_ms_sidecar_vs_delta_mem_reference_trace" if args.rwkv_ms else "gguf_base_vs_delta_mem_reference_trace",
        "backend_mode": "rwkv_ms_sidecar" if args.rwkv_ms else "base_gguf",
        "rwkv_ms_requested": bool(args.rwkv_ms),
        "rwkv_ms_sidecar_path": args.rwkv_ms_sidecar if args.rwkv_ms else "",
        "trace": str(args.trace),
        "base_url": args.base_url,
        "request": request,
        "reference_text": reference_text,
        "reference_source": trace.get("source"),
    }
    try:
        response = client(args.base_url).chat.completions.create(**request)
        raw = response.model_dump(mode="json")
        text, reasoning, content = assistant_message_text(raw)
        row.update(
            {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "gguf_text": text,
                "gguf_content": content,
                "gguf_reasoning_content": reasoning,
                "exact_match": text.strip() == reference_text.strip(),
                "usage": raw.get("usage"),
                "raw_response": raw,
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"latency_ms": (time.perf_counter() - started) * 1000.0, "error": repr(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row.get(k) for k in ("exact_match", "reference_text", "gguf_text", "gguf_content", "gguf_reasoning_content", "error")}, indent=2, ensure_ascii=False))
    if "error" in row:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
