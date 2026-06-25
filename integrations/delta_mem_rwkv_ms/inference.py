#!/usr/bin/env python3
"""Inference entry point for the Gemma4 E4B RWKV-MS delta-Mem HF adapter.

This repository owns the RWKV-MS integration and experiment docs. The actual
model wrapper/session runtime comes from a delta-Mem checkout patched with
`delta_mem_rwkv_ms.patch`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ADAPTER_REPO = "xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_PROMPT = "Help me troubleshoot a mobile data issue where the customer has no usable data."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gemma4 E4B with RWKV-MS delta-Mem adapter.")
    parser.add_argument("--delta-mem-root", default=None, help="Patched delta-Mem checkout to add to sys.path.")
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-dir", default=None, help="Local adapter folder; skips Hub download.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    return parser


def resolve_adapter_dir(adapter_repo: str, adapter_dir: str | None) -> str:
    if adapter_dir:
        path = Path(adapter_dir).expanduser().resolve()
        if not (path / "delta_mem_config.json").is_file():
            raise FileNotFoundError(f"Missing delta_mem_config.json in {path}")
        if not (path / "delta_mem_adapter.pt").is_file():
            raise FileNotFoundError(f"Missing delta_mem_adapter.pt in {path}")
        return str(path)
    return snapshot_download(repo_id=adapter_repo)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.delta_mem_root:
        delta_mem_root = Path(args.delta_mem_root).expanduser().resolve()
        if not (delta_mem_root / "deltamem").is_dir():
            raise FileNotFoundError(f"{delta_mem_root} does not look like a delta-Mem checkout")
        sys.path.insert(0, str(delta_mem_root))

    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model

    adapter_dir = resolve_adapter_dir(args.adapter_repo, args.adapter_dir)
    model, tokenizer = load_delta_mem_chat_model(
        model_path=args.base_model,
        adapter_dir=adapter_dir,
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
        include_debug=True,
    )
    print(result["assistant_display"])
    print("\n[state_stats]", result["state_stats"])
    print("[turn_stats]", result["turn_stats"])


if __name__ == "__main__":
    main()
