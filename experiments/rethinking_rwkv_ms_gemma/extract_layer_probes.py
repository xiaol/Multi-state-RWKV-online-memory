#!/usr/bin/env python3
"""Extract final-query hidden states for layer-wise probing."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_MEMORY_REPO,
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
    parser.add_argument("--limit", type=int, default=None)
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
    label_to_id = {label: idx for idx, label in enumerate(samples[0].candidates)}
    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    for sample in samples:
        reset_delta_state(model)
        encoded = tokenize_prompt(tokenizer, sample.prompt, args.device)
        with memory_condition(model, "normal" if args.condition == "base" else args.condition):
            with torch.inference_mode():
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
        layer_vectors = [
            hidden_state[0, -1].detach().float().cpu().numpy()
            for hidden_state in outputs.hidden_states
        ]
        features.append(np.stack(layer_vectors, axis=0))
        labels.append(label_to_id[sample.answer])
        sample_ids.append(sample.sample_id)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        features=np.stack(features, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        label_names=np.asarray(samples[0].candidates),
        condition=args.condition,
    )
    print(f"wrote probe features {np.stack(features, axis=0).shape} to {args.output_npz}")


if __name__ == "__main__":
    main()
