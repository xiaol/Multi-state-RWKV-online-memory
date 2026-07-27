#!/usr/bin/env python3
"""Compute input-embedding gradient influence by distance from the query token."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_MEMORY_REPO,
    first_candidate_token_ids,
    load_model_and_tokenizer,
    load_samples,
    memory_condition,
    reset_delta_state,
    tokenize_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-mem-root", default="../delta-Mem")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--memory-repo", default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--condition", default="normal", choices=("base", "normal", "no_write", "no_delta"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    samples = load_samples(args.dataset, limit=args.limit)
    use_adapter = args.condition != "base"
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        delta_mem_root=args.delta_mem_root if use_adapter else None,
        memory_dir=args.memory_dir if use_adapter else None,
        memory_repo=(args.memory_repo or DEFAULT_MEMORY_REPO) if use_adapter else None,
    )
    model.eval()

    sum_by_distance: dict[int, float] = {}
    count_by_distance: dict[int, int] = {}
    per_sample_rows: list[dict] = []

    for sample in samples:
        reset_delta_state(model)
        encoded = tokenize_prompt(tokenizer, sample.prompt, args.device)
        candidate_token_ids = first_candidate_token_ids(tokenizer, sample.candidates)
        gold_token_id = candidate_token_ids[sample.answer]
        embeddings = model.get_input_embeddings()(encoded["input_ids"]).detach()
        embeddings.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with memory_condition(model, "normal" if args.condition == "base" else args.condition):
            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=encoded["attention_mask"],
                use_cache=False,
            )
            target = outputs.logits[0, -1, gold_token_id]
            target.backward()
        grad_norm = embeddings.grad.detach().float()[0].norm(dim=-1).cpu().numpy()
        seq_len = int(grad_norm.shape[0])
        for pos, value in enumerate(grad_norm.tolist()):
            distance = seq_len - 1 - pos
            sum_by_distance[distance] = sum_by_distance.get(distance, 0.0) + float(value)
            count_by_distance[distance] = count_by_distance.get(distance, 0) + 1
        per_sample_rows.append(
            {
                "id": sample.sample_id,
                "prompt_tokens": seq_len,
                "target_grad_norm": float(grad_norm[-1]),
                "max_grad_norm": float(grad_norm.max()),
            }
        )

    distances = np.asarray(sorted(sum_by_distance), dtype=np.int64)
    mean_grad_norm = np.asarray(
        [sum_by_distance[int(distance)] / count_by_distance[int(distance)] for distance in distances],
        dtype=np.float32,
    )
    counts = np.asarray([count_by_distance[int(distance)] for distance in distances], dtype=np.int64)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        distance=distances,
        mean_grad_norm=mean_grad_norm,
        count=counts,
        condition=args.condition,
        per_sample=np.asarray(per_sample_rows, dtype=object),
    )
    print(f"wrote gradient influence for {len(samples)} samples to {args.output_npz}")


if __name__ == "__main__":
    main()
