#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from task_config import (
    DEFAULT_TRAIN_LENGTHS,
    GEMMA4_PROMPT_PREFIX,
    GEMMA4_PROMPT_SUFFIX,
    OFFICIAL_EVAL_SEED,
    TASK_TOKEN_BUDGETS,
    TRAIN_TASKS,
    parse_csv_ints,
    parse_csv_strings,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert seed-disjoint RULER rows into strict online-memory episodes."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--generation-manifest", type=Path)
    parser.add_argument("--lengths", default=",".join(str(value) for value in DEFAULT_TRAIN_LENGTHS))
    parser.add_argument("--tasks", default=",".join(TRAIN_TASKS))
    parser.add_argument("--subset", default="train")
    parser.add_argument("--expected-rows", type=int, default=128)
    parser.add_argument("--max-write-tokens", type=int, default=8192)
    parser.add_argument("--max-read-tokens", type=int, default=512)
    parser.add_argument("--memory-ack", default="Memory loaded.")
    parser.add_argument("--shuffle-seed", type=int, default=742)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--eval-subset", default="validation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def canonical_prompt(prompt: str) -> str:
    has_prefix = prompt.startswith(GEMMA4_PROMPT_PREFIX)
    has_suffix = prompt.endswith(GEMMA4_PROMPT_SUFFIX)
    if has_prefix != has_suffix:
        raise ValueError("RULER prompt has only part of the Gemma4 chat wrapper")
    if has_prefix:
        return prompt[len(GEMMA4_PROMPT_PREFIX) : -len(GEMMA4_PROMPT_SUFFIX)]
    if any(token in prompt for token in ("<bos>", "<|turn>", "<turn|>")):
        raise ValueError("Training prompt contains unexpected chat-template control tokens")
    return prompt


def split_document_query(task: str, prompt_body: str) -> tuple[str, str]:
    marker = "\nWhat " if task.startswith("niah_") else "\nQuestion:"
    split_at = prompt_body.rfind(marker)
    if split_at < 0:
        raise ValueError(f"Cannot find final query delimiter {marker!r} for task {task}")
    document = prompt_body[:split_at].rstrip()
    query = prompt_body[split_at + 1 :].strip()
    if not document or not query:
        raise ValueError(f"Empty document or query after splitting task {task}")
    return document, query


def format_answer(task: str, row: dict) -> str:
    references = [str(value).strip() for value in row.get("outputs", []) if str(value).strip()]
    if not references:
        raise ValueError(f"RULER row for {task} has no non-empty outputs")
    answer = ", ".join(references)
    lowered = answer.lower()
    if any(reference.lower() not in lowered for reference in references):
        raise ValueError(f"Formatted answer lost a reference for task {task}")
    return answer


def token_count(apply_chat_template, tokenizer, messages: list[dict[str, str]]) -> int:
    tokenized = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    elif isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "shape"):
        return int(tokenized.shape[-1])
    if tokenized and isinstance(tokenized[0], list):
        return len(tokenized[0])
    return len(tokenized)


def prompt_hashes(root: Path, lengths: tuple[int, ...], tasks: tuple[str, ...], subset: str) -> set[str]:
    hashes = set()
    for length in lengths:
        for task in tasks:
            path = root / str(length) / task / f"{subset}.jsonl"
            if not path.is_file():
                continue
            for row in load_jsonl(path):
                prompt = canonical_prompt(str(row.get("input") or ""))
                hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return hashes


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_file}")
    if args.expected_rows <= 0 or args.max_write_tokens <= 0 or args.max_read_tokens <= 0:
        raise ValueError("Expected rows and token limits must be positive")
    lengths = parse_csv_ints(args.lengths)
    tasks = parse_csv_strings(args.tasks)
    unknown = sorted(set(tasks) - set(TRAIN_TASKS))
    if unknown:
        raise ValueError(f"Training data may only use the 11 non-QA tasks; invalid tasks: {unknown}")

    generation_manifest_path = args.generation_manifest or args.input_root / "generation_manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if generation_manifest.get("mode") != "train":
        raise ValueError("Generation manifest is not a training-data manifest")
    if generation_manifest.get("template_name") != "base":
        raise ValueError("Training sources must be generated with RULER's base template")
    generation_seeds = [int(value) for value in generation_manifest.get("seeds", [])]
    if OFFICIAL_EVAL_SEED in generation_seeds or len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("Training generation seeds are not unique and eval-disjoint")
    if tuple(generation_manifest.get("lengths", [])) != lengths:
        raise ValueError("Requested lengths do not match the generation manifest")
    if tuple(generation_manifest.get("tasks", [])) != tasks:
        raise ValueError("Requested tasks do not match the generation manifest")

    sys.path.insert(0, str(args.runtime_root.resolve()))
    from deltamem.chat_templates import apply_chat_template
    from deltamem.train.delta_sft_experimental import (
        _tokenize_chat_messages,
        build_episode_training_examples,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    eval_hashes = (
        prompt_hashes(args.eval_root, lengths, tasks, args.eval_subset)
        if args.eval_root is not None
        else set()
    )
    episodes = []
    source_files = []
    seen_source_ids = set()
    seen_prompt_answers = set()
    write_counts = []
    read_counts = []
    for length in lengths:
        for task in tasks:
            path = args.input_root / str(length) / task / f"{args.subset}.jsonl"
            rows = load_jsonl(path)
            if len(rows) != args.expected_rows:
                raise ValueError(f"Expected {args.expected_rows} rows in {path}, found {len(rows)}")
            source_files.append(
                {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256_file(path)}
            )
            for ordinal, row in enumerate(rows):
                prompt = canonical_prompt(str(row.get("input") or ""))
                prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if prompt_sha256 in eval_hashes:
                    raise ValueError(f"Training/eval prompt overlap detected at {path}:{ordinal + 1}")
                document, query = split_document_query(task, prompt)
                answer = format_answer(task, row)
                references = [str(value) for value in row.get("outputs", [])]
                lowered_document = document.lower()
                if any(reference.lower() not in lowered_document for reference in references):
                    raise ValueError(f"Reference is absent from source document at {path}:{ordinal + 1}")
                reconstructed = prompt + str(row.get("answer_prefix") or "")
                reconstructed_length = len(tokenizer.tokenize(reconstructed)) + TASK_TOKEN_BUDGETS[task]
                if reconstructed_length != int(row.get("length", -1)):
                    raise ValueError(
                        f"RULER length mismatch at {path}:{ordinal + 1}: "
                        f"expected {row.get('length')}, reconstructed {reconstructed_length}"
                    )
                messages = [
                    {"role": "user", "content": document},
                    {"role": "assistant", "content": args.memory_ack},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ]
                full_write_ids = _tokenize_chat_messages(tokenizer, messages[:2])
                full_read_ids = _tokenize_chat_messages(tokenizer, messages[2:])
                write_tokens = len(full_write_ids)
                read_tokens = len(full_read_ids)
                if write_tokens > args.max_write_tokens:
                    raise ValueError(
                        f"Write history has {write_tokens} tokens, above {args.max_write_tokens}: "
                        f"{path}:{ordinal + 1}"
                    )
                if read_tokens > args.max_read_tokens:
                    raise ValueError(
                        f"Read turn has {read_tokens} tokens, above {args.max_read_tokens}: "
                        f"{path}:{ordinal + 1}"
                    )
                built = build_episode_training_examples(
                    tokenizer,
                    messages,
                    args.max_read_tokens,
                    assistant_loss_mode="final_assistant_only",
                    episode_recent_messages=1,
                    max_write_length=args.max_write_tokens,
                    include_sentence_ids=False,
                )
                if len(built) != 1:
                    raise ValueError(f"Expected one trainer episode at {path}:{ordinal + 1}")
                episode = built[0]
                if (
                    episode["episode_target_message_index"] != 3
                    or episode["write_message_count"] != 2
                    or episode["visible_message_count"] != 1
                ):
                    raise ValueError(f"Trainer episode partition mismatch at {path}:{ordinal + 1}")
                if episode["write_input_ids"] != full_write_ids or episode["input_ids"] != full_read_ids:
                    raise ValueError(f"Trainer silently truncated an episode at {path}:{ordinal + 1}")
                if (
                    episode["state_only_write_input_ids"] != episode["write_input_ids"]
                    or episode["state_only_input_ids"] != episode["input_ids"]
                ):
                    raise ValueError(f"State-only episode path differs at {path}:{ordinal + 1}")
                if not any(label != -100 for label in episode["labels"]):
                    raise ValueError(f"Episode has no supervised answer tokens at {path}:{ordinal + 1}")

                prompt_answer_id = hashlib.sha256(
                    (prompt + "\0" + answer).encode("utf-8")
                ).hexdigest()
                if prompt_answer_id in seen_prompt_answers:
                    raise ValueError(f"Duplicate canonical prompt/answer at {path}:{ordinal + 1}")
                seen_prompt_answers.add(prompt_answer_id)
                source_id = f"{task}:{length}:{ordinal}:{prompt_sha256}"
                if source_id in seen_source_ids:
                    raise ValueError(f"Duplicate source row: {source_id}")
                seen_source_ids.add(source_id)
                write_counts.append(write_tokens)
                read_counts.append(read_tokens)
                episodes.append(
                    {
                        "messages": messages,
                        "task": task,
                        "target_context_length": length,
                        "source_index": row.get("index", ordinal),
                        "source_ordinal": ordinal,
                        "source_prompt_sha256": prompt_sha256,
                        "write_tokens": write_tokens,
                        "read_tokens": read_tokens,
                    }
                )

    random.Random(args.shuffle_seed).shuffle(episodes)
    atomic_write_jsonl(args.output_file, episodes)
    manifest = {
        "output_file": str(args.output_file.resolve()),
        "output_sha256": sha256_file(args.output_file),
        "generation_manifest": str(generation_manifest_path.resolve()),
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "runtime_root": str(args.runtime_root.resolve()),
        "tasks": list(tasks),
        "lengths": list(lengths),
        "generation_seeds": generation_seeds,
        "official_eval_seed": OFFICIAL_EVAL_SEED,
        "rows": len(episodes),
        "expected_rows_per_task_length": args.expected_rows,
        "shuffle_seed": args.shuffle_seed,
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 1,
        "memory_ack": args.memory_ack,
        "max_write_tokens": args.max_write_tokens,
        "observed_write_tokens": {"min": min(write_counts), "max": max(write_counts)},
        "max_read_tokens": args.max_read_tokens,
        "observed_read_tokens": {"min": min(read_counts), "max": max(read_counts)},
        "eval_overlap_checked": args.eval_root is not None,
        "eval_root": None if args.eval_root is None else str(args.eval_root.resolve()),
        "eval_subset": args.eval_subset if args.eval_root is not None else None,
        "eval_prompt_hashes": len(eval_hashes),
        "source_files": source_files,
    }
    manifest_path = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("RULER_TRAINING_DATA=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
