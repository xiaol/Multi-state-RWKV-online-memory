#!/usr/bin/env python3
"""Make a memory probe whose visible read prompt is identical for every row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument("--user-prompt", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _validated_messages(row: Any, *, line_number: int) -> list[dict[str, str]]:
    if not isinstance(row, dict):
        raise ValueError(f"Expected a JSON object at line {line_number}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 5:
        raise ValueError(f"Expected at least five messages at line {line_number}")
    normalized: list[dict[str, str]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Message {message_index} is not an object at line {line_number}"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(
                f"Invalid message {message_index} at line {line_number}"
            )
        normalized.append({"role": role, "content": content})
    if normalized[0]["role"] != "system":
        raise ValueError(f"First message must be system at line {line_number}")
    if normalized[1]["role"] != "user":
        raise ValueError(f"Second message must be user at line {line_number}")
    if normalized[-2]["role"] != "user" or normalized[-1]["role"] != "assistant":
        raise ValueError(
            f"Last two messages must be user, assistant at line {line_number}"
        )
    if not normalized[-1]["content"].strip():
        raise ValueError(f"Final assistant target is empty at line {line_number}")
    return normalized


def build_content_control_rows(
    rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("Content-control probes require at least two rows")
    if not system_prompt.strip() or not user_prompt.strip():
        raise ValueError("Visible system and user prompts must be non-empty")

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        messages = _validated_messages(row, line_number=row_index + 1)
        original_system = messages[0]["content"]
        original_final_user = messages[-2]["content"]

        # Keep all sample-specific task information available, but only through
        # the history that is written to online memory.
        messages[1]["content"] = f"{original_system}\n\n{messages[1]['content']}"
        messages[0]["content"] = system_prompt
        messages[-2]["content"] = user_prompt

        transformed = dict(row)
        transformed["messages"] = messages
        transformed["content_control_probe"] = {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": source_sha256,
            "source_row_index": row_index,
            "original_system_sha256": hashlib.sha256(
                original_system.encode("utf-8")
            ).hexdigest(),
            "original_final_user_sha256": hashlib.sha256(
                original_final_user.encode("utf-8")
            ).hexdigest(),
        }
        output_rows.append(transformed)

    visible_contexts = {
        (row["messages"][0]["content"], row["messages"][-2]["content"])
        for row in output_rows
    }
    if visible_contexts != {(system_prompt, user_prompt)}:
        raise AssertionError("Content-control transformation left non-identical read prompts")
    return output_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = (
        args.summary.expanduser().resolve()
        if args.summary is not None
        else Path(str(output) + ".summary.json")
    )
    if not source.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {source}")
    if len({source, output, summary}) != 3:
        raise ValueError("Input, output, and summary paths must be different")

    source_sha256 = sha256_file(source)
    rows = build_content_control_rows(
        read_jsonl(source),
        source_sha256=source_sha256,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(output, payload)
    output_sha256 = sha256_file(output)
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "rows": len(rows),
        "unique_visible_read_contexts": 1,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
    }
    _atomic_write_text(
        summary,
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
