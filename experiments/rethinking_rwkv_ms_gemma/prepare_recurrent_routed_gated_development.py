#!/usr/bin/env python3
"""Commit an open development subset for gated recurrent routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_development_v1"
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_gated_query_value_development_split_v1"
)
SOURCE_MANIFEST_RECEIPT = (
    "7f1056c33009a30d63179b49e9f95fe1c9fb4438b434d2ad3a22cd22039704e4"
)
SELECTION_SALT = "rwkv-ms-recurrent-routed-gated-development-v1"
ROWS_PER_TASK = 32
TASK_FILES = {
    "attribution": Path(
        "v3.2-attribution-best-candidate/train_derived_development.jsonl"
    ),
    "narrative": Path(
        "v3.2-narrative-type-classification/train_derived_development.jsonl"
    ),
    "scene": Path(
        "v4-scene-boundary-detection/train_derived_development.jsonl"
    ),
}
SOURCE_FILES = {
    "attribution": {
        "rows": 93,
        "sha256": "c8c8203bc460e294bf5756b6600e569c1f5be5b72111d1b969f89d649689a1d3",
    },
    "narrative": {
        "rows": 118,
        "sha256": "f6b84bdbea009caba590a2eb8ab1e94e395ede13ba868199a2d6f9c36a1b2974",
    },
    "scene": {
        "rows": 361,
        "sha256": "b383625cee07e6a7565142e38bb0b0a4d4a2468b2c91171570115b7b311e1e68",
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_manifest_without_receipt",
        "payload_sha256": canonical_sha256(value),
    }
    return result


def main() -> int:
    source_manifest = json.loads(
        (SOURCE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    receipt = source_manifest.pop("receipt")
    if (
        receipt.get("payload_sha256") != SOURCE_MANIFEST_RECEIPT
        or canonical_sha256(source_manifest) != SOURCE_MANIFEST_RECEIPT
    ):
        raise ValueError("Native development source manifest differs")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    tasks = {}
    for task, relative_path in TASK_FILES.items():
        source_path = SOURCE_ROOT / relative_path
        raw_lines = tuple(
            line
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line
        )
        expected = SOURCE_FILES[task]
        if (
            len(raw_lines) != expected["rows"]
            or sha256_file(source_path) != expected["sha256"]
        ):
            raise ValueError(f"Gated development source differs for {task}")
        candidates = [
            {
                "source_ordinal": ordinal,
                "row_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
                "raw_line": raw_line,
            }
            for ordinal, raw_line in enumerate(raw_lines)
        ]
        selected = sorted(
            candidates,
            key=lambda row: (
                hashlib.sha256(
                    (
                        f"{SELECTION_SALT}:{task}:{row['row_sha256']}"
                    ).encode("utf-8")
                ).hexdigest(),
                int(row["source_ordinal"]),
            ),
        )[:ROWS_PER_TASK]
        output_path = OUTPUT_ROOT / task / "development.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=False)
        output_path.write_text(
            "\n".join(str(row["raw_line"]) for row in selected) + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                "source_ordinal": row["source_ordinal"],
                "row_sha256": row["row_sha256"],
            }
            for row in selected
        ]
        tasks[task] = {
            "source_relative_path": str(relative_path),
            "source_rows": len(raw_lines),
            "source_sha256": expected["sha256"],
            "selected_rows": rows,
            "selected_rows_sha256": canonical_sha256(rows),
            "materialized_relative_path": str(output_path.relative_to(OUTPUT_ROOT)),
            "materialized_bytes": output_path.stat().st_size,
            "materialized_sha256": sha256_file(output_path),
        }
    manifest = signed(
        {
            "schema": "rwkv_ms_recurrent_routed_gated_development_split.v1",
            "source_manifest_receipt": SOURCE_MANIFEST_RECEIPT,
            "selection_salt": SELECTION_SALT,
            "rows_per_task": ROWS_PER_TASK,
            "tasks": tasks,
            "fit_rows_opened": False,
            "publisher_validation_opened": False,
            "publisher_test_opened": False,
        }
    )
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "manifest_receipt": manifest["receipt"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
