#!/usr/bin/env python3
"""Create a synthetic NIAH-style candidate-recall dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import write_jsonl


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def build_prompt(
    *,
    rng: random.Random,
    sample_idx: int,
    num_records: int,
    num_candidates: int,
) -> dict:
    candidates = LABELS[:num_candidates]
    answer = rng.choice(candidates)
    target_marker = f"NEEDLE_{sample_idx:04d}"
    target_pos = rng.randrange(num_records)
    lines: list[str] = [
        "You are reading a noisy key-value ledger.",
        "Each record may contain a marker and an answer letter.",
        "Return only the answer letter requested by the final question.",
        "",
    ]
    for record_idx in range(num_records):
        if record_idx == target_pos:
            marker = target_marker
            label = answer
        else:
            marker = f"DISTRACTOR_{sample_idx:04d}_{record_idx:04d}"
            label = rng.choice(candidates)
        noise_a = rng.randrange(10_000, 99_999)
        noise_b = rng.randrange(10_000, 99_999)
        lines.append(
            f"Record {record_idx:04d}: marker {marker} maps to answer {label}. "
            f"Checksum {noise_a}; unrelated note {noise_b}."
        )
    lines.extend(
        [
            "",
            f"Question: what answer letter maps to marker {target_marker}?",
            "Answer:",
        ]
    )
    return {
        "id": f"niah-{sample_idx:05d}",
        "prompt": "\n".join(lines),
        "answer": answer,
        "candidates": list(candidates),
        "needle_marker": target_marker,
        "needle_record_index": target_pos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-records", type=int, default=96)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 2 <= args.num_candidates <= len(LABELS):
        raise ValueError(f"--num-candidates must be between 2 and {len(LABELS)}")
    rng = random.Random(args.seed)
    rows = [
        build_prompt(
            rng=rng,
            sample_idx=idx,
            num_records=args.num_records,
            num_candidates=args.num_candidates,
        )
        for idx in range(args.num_samples)
    ]
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
