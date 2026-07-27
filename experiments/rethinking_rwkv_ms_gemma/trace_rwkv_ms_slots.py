#!/usr/bin/env python3
"""Record RWKV-MS query read mass for the slot containing the needle marker."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_MEMORY_REPO,
    collect_rwkv_trace,
    find_marker_token_index,
    forward_logits,
    load_model_and_tokenizer,
    load_samples,
    memory_condition,
    summarize_numeric,
    tokenize_prompt,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-mem-root", default="../delta-Mem")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--memory-repo", default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", default="normal", choices=("normal", "no_write", "no_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset, limit=args.limit)
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root,
        memory_dir=args.memory_dir,
        memory_repo=args.memory_repo or DEFAULT_MEMORY_REPO,
    )

    rows: list[dict] = []
    for sample in samples:
        encoded = tokenize_prompt(tokenizer, sample.prompt, args.device)
        needle_index = find_marker_token_index(tokenizer, encoded["input_ids"], sample.needle_marker)
        with memory_condition(model, args.condition):
            _ = forward_logits(
                model,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            trace = collect_rwkv_trace(model, needle_token_index=needle_index)
        for metric in trace:
            rows.append(
                {
                    "id": sample.sample_id,
                    "condition": args.condition,
                    "answer": sample.answer,
                    "needle_token_index": needle_index,
                    "prompt_tokens": int(encoded["input_ids"].size(1)),
                    **metric,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.condition}_slot_trace.jsonl", rows)
    summary = {
        "condition": args.condition,
        "num_rows": len(rows),
        **summarize_numeric(
            rows,
            keys=("needle_slot_mass", "read_entropy", "read_max", "state_norm", "delta_o_ratio"),
        ),
    }
    write_json(args.output_dir / f"{args.condition}_slot_trace_summary.json", summary)
    print(
        f"{args.condition}: rows={len(rows)} "
        f"mean_needle_slot_mass={summary.get('mean_needle_slot_mass', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
