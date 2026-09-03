#!/usr/bin/env python3
"""Create an independent development-v2 holdout from unused open train rows."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common


OUTPUT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
DEV_COUNTS = {"attribution": 8, "narrative": 32, "scene": 32}
TRAIN_COUNTS = {"attribution": 12, "narrative": 14, "scene": 14}
SELECTION_SALT = "rwkv-ms-recurrent-routed-development-v2-selection"


def row_user_content(row: common.SourceRow) -> str:
    value = json.loads(row.raw_line)
    return str(value["messages"][1]["content"])


def choose_donor(
    target: common.SourceRow,
    rows: Sequence[common.SourceRow],
) -> common.SourceRow:
    target_user = row_user_content(target)
    candidates = [
        row
        for row in rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"Development-v2 row has no different-answer donor: {target}")
    return max(
        candidates,
        key=lambda row: (
            SequenceMatcher(None, target_user, row_user_content(row)).ratio(),
            -abs(row.user_characters - target.user_characters),
            row.row_sha256,
        ),
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(root: Path, split: str, rows_by_task: Mapping[str, Sequence[common.SourceRow]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for task in common.TASKS:
        path = root / split / task / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(row.raw_line + "\n" for row in rows_by_task[task])
        path.write_text(payload, encoding="utf-8")
        files[str(path.relative_to(root))] = {
            "bytes": len(payload.encode("utf-8")),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "rows": len(rows_by_task[task]),
            "row_payload_sha256": common.canonical_sha256(
                [row.row_sha256 for row in rows_by_task[task]]
            ),
            "source_ordinals": [row.source_ordinal for row in rows_by_task[task]],
        }
    return files


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise ValueError(f"Development-v2 output must be fresh: {OUTPUT_ROOT}")
    manifest, _ = common.validate_split_artifacts()
    open_rows = common.load_open_rows("train", manifest=manifest)
    prior_schedule, prior_payload = common.build_training_schedule(open_rows, updates=96)
    used = {
        task: {row.target.source_ordinal for row in prior_schedule if row.target.task == task}
        for task in common.TASKS
    }
    remaining: dict[str, tuple[common.SourceRow, ...]] = {}
    for task in common.TASKS:
        candidates = [row for row in open_rows[task] if row.source_ordinal not in used[task]]
        remaining[task] = tuple(
            sorted(
                candidates,
                key=lambda row: (
                    hashlib.sha256(
                        f"{SELECTION_SALT}:{task}:{row.row_sha256}".encode("utf-8")
                    ).hexdigest(),
                    row.source_ordinal,
                ),
            )
        )
        required = DEV_COUNTS[task] + TRAIN_COUNTS[task]
        if len(remaining[task]) < required:
            raise RuntimeError(f"Insufficient unused rows for {task}")
    development = {
        task: remaining[task][: DEV_COUNTS[task]] for task in common.TASKS
    }
    training = {
        task: remaining[task][DEV_COUNTS[task] : DEV_COUNTS[task] + TRAIN_COUNTS[task]]
        for task in common.TASKS
    }
    development_ordinals = {
        task: {row.source_ordinal for row in development[task]} for task in common.TASKS
    }
    training_ordinals = {
        task: {row.source_ordinal for row in training[task]} for task in common.TASKS
    }
    if any(development_ordinals[task] & training_ordinals[task] for task in common.TASKS):
        raise RuntimeError("Development-v2 and candidate-training rows overlap")
    development_files = write_rows(OUTPUT_ROOT / "open", "development_v2", development)
    training_files = write_rows(OUTPUT_ROOT / "candidate_train", "candidate_train", training)
    donors = {
        task: {
            str(row.source_ordinal): choose_donor(row, development[task]).source_ordinal
            for row in development[task]
        }
        for task in common.TASKS
    }
    payload = {
        "schema": "rwkv_ms_natural_memory_native_recurrent_routed_development_v2.v1",
        "selection_salt": SELECTION_SALT,
        "source_split": "open/train",
        "source_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "source_open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "prior_96_step_schedule_sha256": common.canonical_sha256(prior_payload),
        "prior_training_rows_excluded": {
            task: sorted(used[task]) for task in common.TASKS
        },
        "development_counts": DEV_COUNTS,
        "candidate_training_counts": TRAIN_COUNTS,
        "development_files": development_files,
        "candidate_training_files": training_files,
        "development_source_ordinals": {
            task: sorted(development_ordinals[task]) for task in common.TASKS
        },
        "candidate_training_source_ordinals": {
            task: sorted(training_ordinals[task]) for task in common.TASKS
        },
        "development_donor_source_ordinals": donors,
        "final_rows_opened": False,
    }
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_payload_without_receipt",
        "payload_sha256": common.canonical_sha256(payload),
    }
    write_json(OUTPUT_ROOT / "manifest.json", payload)
    print(json.dumps(payload["receipt"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
