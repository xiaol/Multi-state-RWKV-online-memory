#!/usr/bin/env python3
"""Run answer-token NIAH ablations for Gemma + RWKV-MS."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from common import (
    DEFAULT_MEMORY_REPO,
    collect_rwkv_trace,
    find_marker_token_index,
    first_candidate_token_ids,
    forward_logits,
    load_model_and_tokenizer,
    load_samples,
    memory_condition,
    parse_conditions,
    reset_interval_for_condition,
    score_candidates_from_logits,
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
    parser.add_argument("--conditions", default="base,normal,no_write,no_delta")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def evaluate_condition(
    *,
    model,
    tokenizer,
    samples,
    condition: str,
    device: str,
) -> tuple[list[dict], dict]:
    reset_interval = reset_interval_for_condition(condition)
    memory_mode = "normal" if condition == "base" else condition
    rows: list[dict] = []
    for sample in samples:
        encoded = tokenize_prompt(tokenizer, sample.prompt, device)
        candidate_token_ids = first_candidate_token_ids(tokenizer, sample.candidates)
        needle_index = find_marker_token_index(tokenizer, encoded["input_ids"], sample.needle_marker)
        with memory_condition(model, memory_mode):
            logits = forward_logits(
                model,
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                reset_interval=reset_interval,
            )
            scores = score_candidates_from_logits(logits, candidate_token_ids)
            trace = collect_rwkv_trace(model, needle_token_index=needle_index)
        pred = max(scores.items(), key=lambda item: item[1])[0]
        non_gold = [score for candidate, score in scores.items() if candidate != sample.answer]
        gold_margin = scores[sample.answer] - max(non_gold)
        rows.append(
            {
                "id": sample.sample_id,
                "condition": condition,
                "answer": sample.answer,
                "prediction": pred,
                "correct": pred == sample.answer,
                "gold_logprob_among_candidates": scores[sample.answer],
                "gold_margin": gold_margin,
                "needle_token_index": needle_index,
                "prompt_tokens": int(encoded["input_ids"].size(1)),
                "trace": trace,
            }
        )
    flat_trace = [
        metric
        for row in rows
        for metric in row["trace"]
    ]
    summary = {
        "condition": condition,
        "num_samples": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / max(len(rows), 1),
        "mean_gold_margin": sum(row["gold_margin"] for row in rows) / max(len(rows), 1),
        "mean_gold_logprob_among_candidates": sum(
            row["gold_logprob_among_candidates"] for row in rows
        )
        / max(len(rows), 1),
    }
    summary.update(
        summarize_numeric(
            flat_trace,
            keys=("needle_slot_mass", "read_entropy", "read_max", "state_norm", "delta_o_ratio"),
        )
    )
    return rows, summary


def clear_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    conditions = parse_conditions(args.conditions)
    samples = load_samples(args.dataset, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    adapter_loaded = False
    adapter_model = None
    adapter_tokenizer = None
    base_model = None
    base_tokenizer = None

    for condition in conditions:
        if condition == "base":
            if adapter_model is not None:
                del adapter_model
                adapter_model = None
                adapter_tokenizer = None
                adapter_loaded = False
                clear_model_memory()
            if base_model is None:
                base_model, base_tokenizer = load_model_and_tokenizer(
                    base_model=args.base_model,
                    device=args.device,
                    dtype=args.dtype,
                    attn_implementation=args.attn_implementation,
                )
            model, tokenizer = base_model, base_tokenizer
        else:
            if base_model is not None:
                del base_model
                base_model = None
                base_tokenizer = None
                clear_model_memory()
            if not adapter_loaded:
                adapter_model, adapter_tokenizer = load_model_and_tokenizer(
                    base_model=args.base_model,
                    device=args.device,
                    dtype=args.dtype,
                    attn_implementation=args.attn_implementation,
                    delta_mem_root=args.delta_mem_root,
                    memory_dir=args.memory_dir,
                    memory_repo=args.memory_repo or DEFAULT_MEMORY_REPO,
                )
                adapter_loaded = True
            model, tokenizer = adapter_model, adapter_tokenizer
        rows, summary = evaluate_condition(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            condition=condition,
            device=args.device,
        )
        write_jsonl(args.output_dir / f"{condition}.jsonl", rows)
        summaries.append(summary)
        print(
            f"{condition}: accuracy={summary['accuracy']:.3f} "
            f"margin={summary['mean_gold_margin']:.3f}"
        )
        # Do not keep the previous condition's model alive while switching
        # between the frozen base model and the adapter-backed model.
        del model, tokenizer

    write_json(args.output_dir / "summary.json", {"conditions": summaries})


if __name__ == "__main__":
    main()
