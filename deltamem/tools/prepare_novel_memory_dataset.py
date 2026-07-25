#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


EXPECTED_ROLES = ("system", "user", "assistant")
DEFAULT_BREAK_QUERY = "请从断点处继续写小说，不要复述前文。"
DEFAULT_TARGET_SUFFIX_CAP = 112
DEFAULT_MIN_PREFIX_TOKENS = 32
DEFAULT_MIN_TOTAL_TOKENS = 64
SHORT_SKIP_REASON = "assistant_below_min_total_tokens"


@dataclass
class SourceStats:
    input_index: int
    path: str
    rows: int = 0
    blank_lines: int = 0
    dedupe_winners: int = 0
    duplicates: int = 0
    emitted: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    sha256: str = ""

    def count_skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_index": self.input_index,
            "path": self.path,
            "sha256": self.sha256,
            "rows": self.rows,
            "blank_lines": self.blank_lines,
            "dedupe_winners": self.dedupe_winners,
            "duplicates": self.duplicates,
            "emitted": self.emitted,
            "skipped": dict(sorted(self.skipped.items())),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert three-turn novel SFT rows into deterministic online-memory episodes."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_paths",
        action="append",
        type=Path,
        required=True,
        help="Input JSONL path. Repeat to set source precedence.",
    )
    parser.add_argument("--model-path", required=True, help="Local tokenizer/model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path (default: <output>.summary.json).",
    )
    parser.add_argument("--break-query", default=DEFAULT_BREAK_QUERY)
    parser.add_argument(
        "--target-suffix-cap",
        type=int,
        default=DEFAULT_TARGET_SUFFIX_CAP,
    )
    parser.add_argument(
        "--min-prefix-tokens",
        type=int,
        default=DEFAULT_MIN_PREFIX_TOKENS,
    )
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=DEFAULT_MIN_TOTAL_TOKENS,
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dedupe_key(user: str, assistant: str) -> str:
    _, separator, user_after_header = user.partition("\n")
    canonical_user = user_after_header if separator else user
    canonical = re.sub(r"\s+", "", canonical_user + assistant)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_row(row: Any, source: Path, line_number: int) -> tuple[str, str, str]:
    location = f"{source}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at {location}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
        raise ValueError(
            f"Expected exactly three messages with roles {EXPECTED_ROLES} at {location}"
        )
    roles = []
    contents = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {message_index} is not an object at {location}")
        role = message.get("role")
        content = message.get("content")
        roles.append(role)
        if not isinstance(content, str):
            raise ValueError(f"Message {message_index} content is not a string at {location}")
        contents.append(content)
    if tuple(roles) != EXPECTED_ROLES:
        raise ValueError(
            f"Expected exact roles {EXPECTED_ROLES}, found {tuple(roles)} at {location}"
        )
    return contents[0], contents[1], contents[2]


def _flat_input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("Expected a single tokenized sequence")
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _flat_input_ids(tokenized)
    offsets: Any = tokenized["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        if len(offsets) != 1:
            raise ValueError("Expected offsets for a single tokenized sequence")
        offsets = offsets[0]
    normalized_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(input_ids) != len(normalized_offsets):
        raise ValueError(
            "Tokenizer returned different input-id and offset-mapping lengths "
            f"({len(input_ids)} != {len(normalized_offsets)})"
        )
    return input_ids, normalized_offsets


def _token_count(tokenizer: Any, text: str) -> int:
    tokenized = tokenizer(text, add_special_tokens=False)
    return len(_flat_input_ids(tokenized))


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(_flat_input_ids(tokenized))


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        index = max(0, math.ceil((percent / 100) * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _report_progress(stats: SourceStats) -> None:
    print(
        "NOVEL_MEMORY_PROGRESS "
        f"source={stats.input_index} rows={stats.rows} "
        f"winners={stats.dedupe_winners} emitted={stats.emitted}",
        file=sys.stderr,
        flush=True,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _validate_config(
    input_paths: Sequence[Path],
    output_path: Path,
    summary_path: Path,
    break_query: str,
    target_suffix_cap: int,
    min_prefix_tokens: int,
    min_total_tokens: int,
) -> None:
    if not input_paths:
        raise ValueError("At least one input path is required")
    if not break_query.strip():
        raise ValueError("break_query must not be empty")
    if target_suffix_cap <= 0:
        raise ValueError("target_suffix_cap must be > 0")
    if min_prefix_tokens <= 0:
        raise ValueError("min_prefix_tokens must be > 0")
    if min_total_tokens <= min_prefix_tokens:
        raise ValueError("min_total_tokens must be greater than min_prefix_tokens")
    resolved_output = output_path.resolve()
    resolved_summary = summary_path.resolve()
    if resolved_output == resolved_summary:
        raise ValueError("Output and summary paths must be different")
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
        resolved_input = input_path.resolve()
        if resolved_input in {resolved_output, resolved_summary}:
            raise ValueError(f"Input path cannot also be an output path: {input_path}")


def preprocess_dataset(
    *,
    input_paths: Sequence[str | Path],
    tokenizer: Any,
    model_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    break_query: str = DEFAULT_BREAK_QUERY,
    target_suffix_cap: int = DEFAULT_TARGET_SUFFIX_CAP,
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS,
    min_total_tokens: int = DEFAULT_MIN_TOTAL_TOKENS,
) -> dict[str, Any]:
    normalized_inputs = [Path(path) for path in input_paths]
    normalized_output = Path(output_path)
    normalized_summary = (
        Path(summary_path)
        if summary_path is not None
        else Path(str(normalized_output) + ".summary.json")
    )
    _validate_config(
        normalized_inputs,
        normalized_output,
        normalized_summary,
        break_query,
        target_suffix_cap,
        min_prefix_tokens,
        min_total_tokens,
    )

    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    source_stats = [
        SourceStats(input_index=index, path=str(path.resolve()))
        for index, path in enumerate(normalized_inputs)
    ]
    seen_keys: set[str] = set()
    skip_counts: dict[str, int] = {}
    assistant_token_counts: list[int] = []
    prefix_boundary_token_counts: list[int] = []
    target_token_counts: list[int] = []
    full_write_token_counts: list[int] = []
    full_read_token_counts: list[int] = []
    cut_adjustments: list[int] = []
    chat_token_stats_error: str | None = None
    output_digest = hashlib.sha256()
    output_rows = 0
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=normalized_output.parent,
            prefix=f".{normalized_output.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            for input_index, input_path in enumerate(normalized_inputs):
                stats = source_stats[input_index]
                input_digest = hashlib.sha256()
                with input_path.open("rb") as input_handle:
                    for line_number, raw_line in enumerate(input_handle, start=1):
                        input_digest.update(raw_line)
                        if not raw_line.strip():
                            stats.blank_lines += 1
                            continue
                        stats.rows += 1
                        try:
                            row = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(f"Invalid UTF-8 JSON at {input_path}:{line_number}") from exc
                        system, user, assistant = validate_row(row, input_path, line_number)
                        row_dedupe_key = dedupe_key(user, assistant)
                        if row_dedupe_key in seen_keys:
                            stats.duplicates += 1
                            if stats.rows % 1000 == 0:
                                _report_progress(stats)
                            continue
                        seen_keys.add(row_dedupe_key)
                        stats.dedupe_winners += 1

                        assistant_ids, offsets = _tokenize_with_offsets(tokenizer, assistant)
                        assistant_tokens = len(assistant_ids)
                        assistant_token_counts.append(assistant_tokens)
                        if assistant_tokens < min_total_tokens:
                            stats.count_skip(SHORT_SKIP_REASON)
                            skip_counts[SHORT_SKIP_REASON] = skip_counts.get(SHORT_SKIP_REASON, 0) + 1
                            if stats.rows % 1000 == 0:
                                _report_progress(stats)
                            continue

                        initial_cut = max(
                            min_prefix_tokens,
                            assistant_tokens - target_suffix_cap,
                        )
                        cut = initial_cut
                        while cut < assistant_tokens:
                            split_at = offsets[cut][0]
                            if split_at <= 0 or split_at >= len(assistant):
                                raise ValueError(
                                    "Tokenizer returned an unusable split offset "
                                    f"at {input_path}:{line_number} token {cut}: {split_at}"
                                )
                            assistant_suffix = assistant[split_at:]
                            target_tokens = _token_count(tokenizer, assistant_suffix)
                            if target_tokens <= target_suffix_cap:
                                break
                            cut += 1
                        else:
                            raise ValueError(
                                f"Could not produce a non-empty capped suffix at {input_path}:{line_number}"
                            )

                        assistant_prefix = assistant[:split_at]
                        output_messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant_prefix},
                            {"role": "user", "content": break_query},
                            {"role": "assistant", "content": assistant_suffix},
                        ]
                        output_row = {
                            "messages": output_messages,
                            "memory_preprocessing": {
                                "assistant_tokens": assistant_tokens,
                                "cut_adjustment_tokens": cut - initial_cut,
                                "dedupe_sha256": row_dedupe_key,
                                "input_index": input_index,
                                "prefix_boundary_tokens": cut,
                                "source_line": line_number,
                                "target_tokens": target_tokens,
                            },
                        }
                        serialized = json.dumps(
                            output_row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ) + "\n"
                        output_handle.write(serialized)
                        output_digest.update(serialized.encode("utf-8"))
                        output_rows += 1
                        stats.emitted += 1
                        prefix_boundary_token_counts.append(cut)
                        target_token_counts.append(target_tokens)
                        cut_adjustments.append(cut - initial_cut)

                        if chat_token_stats_error is None:
                            try:
                                full_write_token_counts.append(
                                    _chat_token_count(tokenizer, output_messages[:3])
                                )
                                full_read_token_counts.append(
                                    _chat_token_count(
                                        tokenizer,
                                        [output_messages[0], output_messages[3], output_messages[4]],
                                    )
                                )
                            except Exception as exc:
                                chat_token_stats_error = f"{type(exc).__name__}: {exc}"
                                full_write_token_counts.clear()
                                full_read_token_counts.clear()
                        if stats.rows % 1000 == 0:
                            _report_progress(stats)
                stats.sha256 = input_digest.hexdigest()
        os.replace(temporary_path, normalized_output)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    duplicate_rows = sum(stats.duplicates for stats in source_stats)
    input_rows = sum(stats.rows for stats in source_stats)
    dedupe_winners = sum(stats.dedupe_winners for stats in source_stats)
    token_stats: dict[str, Any] = {
        "assistant_total": _distribution(assistant_token_counts),
        "prefix_boundary": _distribution(prefix_boundary_token_counts),
        "target_suffix": _distribution(target_token_counts),
        "cut_adjustment": _distribution(cut_adjustments),
        "full_write": _distribution(full_write_token_counts),
        "full_read": _distribution(full_read_token_counts),
    }
    if chat_token_stats_error is not None:
        token_stats["chat_template_stats_error"] = chat_token_stats_error

    summary = {
        "version": 1,
        "inputs": [stats.as_dict() for stats in source_stats],
        "output": {
            "path": str(normalized_output.resolve()),
            "summary_path": str(normalized_summary.resolve()),
            "rows": output_rows,
            "sha256": output_digest.hexdigest(),
        },
        "counts": {
            "input_rows": input_rows,
            "dedupe_winners": dedupe_winners,
            "duplicates": duplicate_rows,
            "emitted": output_rows,
            "skipped": dict(sorted(skip_counts.items())),
        },
        "config": {
            "model_path": str(model_path),
            "input_precedence": "argument_order_then_line_order_first_occurrence_wins",
            "expected_roles": list(EXPECTED_ROLES),
            "dedupe": "sha256(remove_whitespace(user_after_first_newline + assistant))",
            "break_query": break_query,
            "target_suffix_cap": target_suffix_cap,
            "min_prefix_tokens": min_prefix_tokens,
            "min_total_tokens": min_total_tokens,
            "tokenizer_add_special_tokens": False,
        },
        "token_stats": token_stats,
    }
    _atomic_write_json(normalized_summary, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for assistant offset mappings")
    summary = preprocess_dataset(
        input_paths=args.input_paths,
        tokenizer=tokenizer,
        model_path=args.model_path,
        output_path=args.output,
        summary_path=args.summary,
        break_query=args.break_query,
        target_suffix_cap=args.target_suffix_cap,
        min_prefix_tokens=args.min_prefix_tokens,
        min_total_tokens=args.min_total_tokens,
    )
    print(
        "NOVEL_MEMORY_DATA="
        + json.dumps(
            {
                "output": summary["output"],
                "counts": summary["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
