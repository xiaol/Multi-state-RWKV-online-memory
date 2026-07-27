#!/usr/bin/env python3
"""Validate the content-controlled 32-row one-layer CE probe source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any


DEFAULT_SOURCE = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/novel_agent_memory/"
    "novel_memory_content_control_probe_seed20260724_n32.jsonl"
)
EXPECTED_SHA256 = "0aa7472d3c7fe3b5501801fc380f570b82a048c6e535e800263c6e1c2ee08a2d"
PROVENANCE_SHA256 = "026f58d1ee06a3cf79363db274f803e8264b1a2e42cc2d7698ea8477a5d4b9ca"
EXPECTED_ROWS = 32
EXPECTED_ROLES = ("system", "user", "assistant", "user", "assistant")
VISIBLE_SYSTEM = "你是一位小说作家。"
VISIBLE_FINAL_USER = "请从断点处继续写小说，不要复述前文。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_visible_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not a JSON object")
            rows.append(row)
    return rows


def validate_source(*, source: Path, expected_sha256: str) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"controlled probe source is missing: {source}")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"controlled probe checksum mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    rows = read_jsonl(source)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"controlled probe row-count mismatch: expected={EXPECTED_ROWS} actual={len(rows)}"
        )

    normalized_system = normalize_visible_text(VISIBLE_SYSTEM)
    normalized_final_user = normalize_visible_text(VISIBLE_FINAL_USER)
    source_row_indices: list[int] = []
    original_system_hashes: list[str] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
            raise ValueError(f"row {row_index}: expected exactly five messages")
        roles = tuple(message.get("role") for message in messages if isinstance(message, dict))
        if roles != EXPECTED_ROLES:
            raise ValueError(
                f"row {row_index}: expected roles {EXPECTED_ROLES}, got {roles}"
            )
        contents = [message.get("content") for message in messages]
        if not all(isinstance(content, str) and content for content in contents):
            raise ValueError(f"row {row_index}: every message must have nonempty string content")
        if contents[0] != VISIBLE_SYSTEM:
            raise ValueError(f"row {row_index}: visible system prompt is not byte-identical")
        if contents[-2] != VISIBLE_FINAL_USER:
            raise ValueError(f"row {row_index}: final-user prompt is not byte-identical")
        if normalize_visible_text(contents[0]) != normalized_system:
            raise ValueError(f"row {row_index}: normalized system prompt differs")
        if normalize_visible_text(contents[-2]) != normalized_final_user:
            raise ValueError(f"row {row_index}: normalized final-user prompt differs")

        control = row.get("content_control_probe")
        if not isinstance(control, dict):
            raise ValueError(f"row {row_index}: content_control_probe metadata is missing")
        if control.get("schema_version") != 1:
            raise ValueError(f"row {row_index}: unsupported content-control schema")
        if control.get("source_sha256") != PROVENANCE_SHA256:
            raise ValueError(f"row {row_index}: old-source provenance checksum differs")
        source_row_index = control.get("source_row_index")
        if source_row_index != row_index:
            raise ValueError(
                f"row {row_index}: source_row_index must preserve controlled row order"
            )
        source_row_indices.append(int(source_row_index))

        original_system, separator, _ = contents[1].partition("\n\n")
        if not separator:
            raise ValueError(
                f"row {row_index}: first written user turn does not contain moved system text"
            )
        original_system_sha256 = str(control.get("original_system_sha256", ""))
        if sha256_text(original_system) != original_system_sha256:
            raise ValueError(f"row {row_index}: moved system provenance hash differs")
        original_system_hashes.append(original_system_sha256)
        if control.get("original_final_user_sha256") != sha256_text(VISIBLE_FINAL_USER):
            raise ValueError(f"row {row_index}: original final-user provenance hash differs")

    if source_row_indices != list(range(EXPECTED_ROWS)):
        raise ValueError("content-control source row indices are not contiguous")
    visible_payload = json.dumps(
        {
            "system": normalized_system,
            "final_user": normalized_final_user,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "path": str(source),
        "sha256": actual_sha256,
        "rows": len(rows),
        "visible_system": VISIBLE_SYSTEM,
        "visible_final_user": VISIBLE_FINAL_USER,
        "normalized_visible_context_sha256": sha256_bytes(visible_payload),
        "visible_system_unique_count": 1,
        "visible_final_user_unique_count": 1,
        "read_phase_writes_required": False,
        "provenance": {
            "source_sha256": PROVENANCE_SHA256,
            "source_rows": len(source_row_indices),
            "moved_system_rows": len(original_system_hashes),
            "unique_original_systems": len(set(original_system_hashes)),
        },
    }


def main() -> int:
    args = parse_args()
    result = validate_source(
        source=args.source,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
