#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from task_config import (
    DEFAULT_TRAIN_LENGTHS,
    OFFICIAL_EVAL_SEED,
    TRAIN_TASKS,
    parse_csv_ints,
    parse_csv_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a converted RULER online-memory corpus.")
    parser.add_argument("--training-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--expected-rows", type=int, default=4224)
    parser.add_argument("--expected-rows-per-task-length", type=int, default=128)
    parser.add_argument("--lengths", default=",".join(str(value) for value in DEFAULT_TRAIN_LENGTHS))
    parser.add_argument("--tasks", default=",".join(TRAIN_TASKS))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    expected_lengths = parse_csv_ints(args.lengths)
    expected_tasks = parse_csv_strings(args.tasks)
    if set(expected_tasks) - set(TRAIN_TASKS):
        raise ValueError("Verifier task list contains QA or unknown tasks")
    manifest_path = args.manifest_file or args.training_file.with_suffix(
        args.training_file.suffix + ".manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(args.training_file)
    if manifest.get("output_sha256") != digest:
        raise ValueError("Training JSONL hash does not match its manifest")
    if manifest.get("rows") != args.expected_rows:
        raise ValueError(f"Manifest row count is not {args.expected_rows}")
    if tuple(manifest.get("tasks", [])) != expected_tasks:
        raise ValueError("Manifest task order does not match the 11 non-QA RULER tasks")
    if tuple(manifest.get("lengths", [])) != expected_lengths:
        raise ValueError("Manifest lengths do not match the requested matrix")
    if manifest.get("expected_rows_per_task_length") != args.expected_rows_per_task_length:
        raise ValueError("Manifest rows-per-task-length does not match")
    seeds = [int(value) for value in manifest.get("generation_seeds", [])]
    if len(seeds) != len(expected_tasks) * len(expected_lengths):
        raise ValueError("Manifest does not contain one seed per task/length job")
    if len(set(seeds)) != len(seeds) or OFFICIAL_EVAL_SEED in seeds:
        raise ValueError("Training seeds are not unique and eval-disjoint")
    if manifest.get("assistant_loss_mode") != "final_assistant_only":
        raise ValueError("Training manifest has the wrong assistant loss mode")
    if manifest.get("episode_recent_messages") != 1:
        raise ValueError("Training manifest has the wrong episode visibility")
    if int(manifest.get("observed_write_tokens", {}).get("max", 0)) > 8192:
        raise ValueError("Training manifest contains an overlength write")
    if int(manifest.get("observed_read_tokens", {}).get("max", 0)) > 512:
        raise ValueError("Training manifest contains an overlength read")

    counts: Counter[tuple[str, int]] = Counter()
    source_hashes = set()
    rows = 0
    with args.training_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or [message.get("role") for message in messages] != [
                "user",
                "assistant",
                "user",
                "assistant",
            ]:
                raise ValueError(f"Invalid four-turn episode at line {line_number}")
            if messages[1].get("content") != "Memory loaded.":
                raise ValueError(f"Invalid memory acknowledgement at line {line_number}")
            if int(row.get("write_tokens", 8193)) > 8192 or int(row.get("read_tokens", 513)) > 512:
                raise ValueError(f"Overlength episode at line {line_number}")
            task = str(row.get("task"))
            length = int(row.get("target_context_length"))
            counts[(task, length)] += 1
            source_hash = str(row.get("source_prompt_sha256"))
            if source_hash in source_hashes:
                raise ValueError(f"Duplicate source prompt at line {line_number}")
            source_hashes.add(source_hash)
            rows += 1
    if rows != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} JSONL rows, found {rows}")
    expected_counts = {
        (task, length): args.expected_rows_per_task_length
        for length in expected_lengths
        for task in expected_tasks
    }
    if dict(counts) != expected_counts:
        raise ValueError("Per-task/per-length row matrix is incomplete")

    result = {
        "status": "ok",
        "training_file": str(args.training_file.resolve()),
        "sha256": digest,
        "rows": rows,
        "jobs": len(counts),
        "unique_seeds": len(seeds),
        "max_write_tokens": manifest["observed_write_tokens"]["max"],
        "max_read_tokens": manifest["observed_read_tokens"]["max"],
    }
    print("RULER_TRAINING_VERIFY=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
