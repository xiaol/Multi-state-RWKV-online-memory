#!/usr/bin/env python3
"""Batch prompt evaluator for a Gemma4 GGUF served by llama.cpp."""

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


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "gemma-4-e4b-it-q8"
DEFAULT_RWKV_MS_MODEL = "gemma-4-e4b-it-rwkv-ms-q8"
DEFAULT_RWKV_MS_SIDECAR_PATH = (
    "/run/media/xiaol/B214449214445C0B/models/gguf/gemma-4-E4B-it/"
    "gemma-4-E4B-it-rwkv-ms-memory.gguf"
)
DEFAULT_OUTPUT_DIR = ".openresearch/artifacts/gguf_ui"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/") + "/", api_key=os.environ.get("OPENAI_API_KEY", "llama.cpp"))


def read_jsonl_or_text(path: Path, system_prompt: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if path.suffix.lower() == ".jsonl":
                item = json.loads(text)
            else:
                item = {"id": str(index), "prompt": text}
            if "messages" not in item:
                messages = []
                item_system = item.get("system", system_prompt)
                if item_system:
                    messages.append({"role": "system", "content": item_system})
                messages.append({"role": "user", "content": item["prompt"]})
                item["messages"] = messages
            item.setdefault("id", str(index))
            items.append(item)
    return items


def output_path(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(DEFAULT_OUTPUT_DIR) / f"eval_{stamp}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def assistant_message_text(raw_response: dict[str, Any]) -> tuple[str, str | None]:
    message = raw_response.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content")
    return content or reasoning or "", reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prompts against a local llama.cpp OpenAI endpoint.")
    parser.add_argument("input", type=Path, help="Text file with one prompt per line, or JSONL with prompt/messages.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL))
    rwkv_ms_default = env_flag("LLAMA_RWKV_MS", bool(os.environ.get("GGUF_RWKV_MS_SIDECAR_PATH")))
    parser.add_argument("--model", default=os.environ.get("LLAMA_MODEL", DEFAULT_RWKV_MS_MODEL if rwkv_ms_default else DEFAULT_MODEL))
    parser.add_argument("--rwkv-ms", action="store_true", default=rwkv_ms_default)
    parser.add_argument("--rwkv-ms-sidecar", default=os.environ.get("GGUF_RWKV_MS_SIDECAR_PATH", DEFAULT_RWKV_MS_SIDECAR_PATH))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N prompts; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl_or_text(args.input, args.system_prompt)
    if args.limit > 0:
        rows = rows[: args.limit]
    out = output_path(args.output)
    openai_client = client(args.base_url)

    extra_body: dict[str, Any] = {
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
    }
    if args.seed >= 0:
        extra_body["seed"] = args.seed

    for item in rows:
        run_id = str(uuid.uuid4())
        request = {
            "model": args.model,
            "messages": item["messages"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "extra_body": extra_body,
        }
        result: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": utc_now(),
            "input_id": item.get("id"),
            "backend": "llama.cpp_openai_compatible",
            "backend_mode": "rwkv_ms_sidecar" if args.rwkv_ms else "base_gguf",
            "rwkv_ms_requested": bool(args.rwkv_ms),
            "rwkv_ms_sidecar_path": args.rwkv_ms_sidecar if args.rwkv_ms else "",
            "base_url": args.base_url,
            "model": args.model,
            "request": request,
        }
        started = time.perf_counter()
        try:
            response = openai_client.chat.completions.create(**request)
            raw = response.model_dump(mode="json")
            text, reasoning = assistant_message_text(raw)
            result.update(
                {
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "response_text": text,
                    "reasoning_content": reasoning,
                    "finish_reason": raw.get("choices", [{}])[0].get("finish_reason"),
                    "usage": raw.get("usage"),
                    "raw_response": raw,
                }
            )
        except Exception as exc:  # noqa: BLE001
            result.update({"latency_ms": (time.perf_counter() - started) * 1000.0, "error": repr(exc)})
        append_jsonl(out, result)
        print(json.dumps({"input_id": item.get("id"), "run_id": run_id, "ok": "error" not in result}))
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
