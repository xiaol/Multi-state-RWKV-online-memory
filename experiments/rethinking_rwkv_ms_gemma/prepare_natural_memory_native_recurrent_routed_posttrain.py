#!/usr/bin/env python3
"""Commit fresh fit, development, and final rows for recurrent-routed post-training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = (
    SCRIPT_DIR / "local_artifacts/natural_memory_native_development_v1"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_recurrent_routed_posttrain_split_v1"
)
SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_split.v1"
FINAL_SCHEMA = (
    "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_final_commitment.v1"
)
OPEN_SPLIT_SCHEMA = (
    "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_open_splits.v1"
)
SPLIT_SALT = "rwkv-ms-recurrent-routed-posttrain-split-v1"
DEVELOPMENT_ROWS_PER_TASK = 32
FINAL_ROWS_PER_TASK = 64
TASK_FILES = {
    "attribution": Path(
        "v3.2-attribution-best-candidate/train_derived_fit.jsonl"
    ),
    "narrative": Path(
        "v3.2-narrative-type-classification/train_derived_fit.jsonl"
    ),
    "scene": Path("v4-scene-boundary-detection/train_derived_fit.jsonl"),
}
PREVIOUSLY_INSPECTED_ROWS = {
    "attribution": (),
    "narrative": (),
    "scene": (0, 1),
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_sha256(raw_line: str) -> str:
    return hashlib.sha256(raw_line.encode("utf-8")).hexdigest()


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_json_without_receipt",
        "payload_sha256": canonical_sha256(value),
    }
    return payload


def write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Post-training split output must be fresh: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_task_split(task: str, path: Path) -> Mapping[str, Any]:
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [
        {
            "task": task,
            "source_ordinal": ordinal,
            "row_sha256": row_sha256(raw_line),
        }
        for ordinal, raw_line in enumerate(raw_lines)
    ]
    inspected = frozenset(PREVIOUSLY_INSPECTED_ROWS[task])
    eligible_holdout = [
        row for row in rows if int(row["source_ordinal"]) not in inspected
    ]
    ordered = sorted(
        eligible_holdout,
        key=lambda row: (
            hashlib.sha256(
                (
                    f"{SPLIT_SALT}:{task}:" + str(row["row_sha256"])
                ).encode("utf-8")
            ).hexdigest(),
            int(row["source_ordinal"]),
        ),
    )
    final_rows = ordered[:FINAL_ROWS_PER_TASK]
    development_rows = ordered[
        FINAL_ROWS_PER_TASK : FINAL_ROWS_PER_TASK + DEVELOPMENT_ROWS_PER_TASK
    ]
    heldout_ordinals = {
        int(row["source_ordinal"])
        for row in (*final_rows, *development_rows)
    }
    train_rows = [
        row for row in rows if int(row["source_ordinal"]) not in heldout_ordinals
    ]
    split_rows = {
        "train": train_rows,
        "development": development_rows,
        "final": final_rows,
    }
    ordinal_sets = {
        name: {int(row["source_ordinal"]) for row in selected}
        for name, selected in split_rows.items()
    }
    if (
        ordinal_sets["train"] & ordinal_sets["development"]
        or ordinal_sets["train"] & ordinal_sets["final"]
        or ordinal_sets["development"] & ordinal_sets["final"]
        or set.union(*ordinal_sets.values()) != set(range(len(rows)))
    ):
        raise RuntimeError(f"Post-training split overlap for {task}")
    if inspected & ordinal_sets["final"]:
        raise RuntimeError(f"Previously inspected {task} row entered final split")
    return {
        "source_file": str(path),
        "source_file_sha256": sha256_file(path),
        "source_rows": len(rows),
        "previously_inspected_source_ordinals": sorted(inspected),
        "splits": {
            name: {
                "rows": selected,
                "count": len(selected),
                "payload_sha256": canonical_sha256(selected),
            }
            for name, selected in split_rows.items()
        },
    }


def build_manifest(dataset_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    tasks = {
        task: build_task_split(task, dataset_root / relative_path)
        for task, relative_path in TASK_FILES.items()
    }
    final_rows = {
        task: value["splits"]["final"]
        for task, value in tasks.items()
    }
    final_commitment = signed(
        {
            "schema": FINAL_SCHEMA,
            "split_salt": SPLIT_SALT,
            "rows_per_task": FINAL_ROWS_PER_TASK,
            "tasks": final_rows,
            "protected_until": (
                "candidate selection and development causal gates are complete"
            ),
            "semantic_content_opened_during_commitment": False,
        }
    )
    manifest = signed(
        {
            "schema": SCHEMA,
            "split_salt": SPLIT_SALT,
            "development_rows_per_task": DEVELOPMENT_ROWS_PER_TASK,
            "final_rows_per_task": FINAL_ROWS_PER_TASK,
            "tasks": tasks,
            "final_commitment_payload_sha256": final_commitment["receipt"][
                "payload_sha256"
            ],
            "leakage_audit": {
                "task_local_splits_disjoint": True,
                "all_source_rows_assigned_once": True,
                "previously_inspected_rows_excluded_from_final": True,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
            },
        }
    )
    return manifest, final_commitment


def materialize_open_splits(
    dataset_root: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    files: dict[str, Any] = {}
    for task, relative_path in TASK_FILES.items():
        source_lines = [
            line
            for line in (dataset_root / relative_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        task_manifest = manifest["tasks"][task]
        for split in ("train", "development"):
            rows = task_manifest["splits"][split]["rows"]
            selected_lines = [
                source_lines[int(row["source_ordinal"])]
                for row in rows
            ]
            for row, raw_line in zip(rows, selected_lines):
                if row_sha256(raw_line) != row["row_sha256"]:
                    raise RuntimeError(
                        f"Open {task}/{split} row hash differs at "
                        f"{row['source_ordinal']}"
                    )
            path = output_dir / "open" / task / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ValueError(f"Open split output must be fresh: {path}")
            path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
            files[str(path.relative_to(output_dir))] = {
                "rows": len(selected_lines),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_payload_sha256": canonical_sha256(rows),
            }
    receipt = signed(
        {
            "schema": OPEN_SPLIT_SCHEMA,
            "manifest_receipt": manifest["receipt"]["payload_sha256"],
            "files": files,
            "materialized_splits": ["train", "development"],
            "final_files_written": [],
        }
    )
    write_fresh_json(output_dir / "open_split_receipt.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, final_commitment = build_manifest(dataset_root)
    write_fresh_json(output_dir / "manifest.json", manifest)
    write_fresh_json(output_dir / "final_commitment.json", final_commitment)
    open_split_receipt = materialize_open_splits(
        dataset_root,
        output_dir,
        manifest,
    )
    print(
        json.dumps(
            {
                "manifest_receipt": manifest["receipt"]["payload_sha256"],
                "final_commitment_receipt": final_commitment["receipt"][
                    "payload_sha256"
                ],
                "open_split_receipt": open_split_receipt["receipt"][
                    "payload_sha256"
                ],
                "output_dir": str(output_dir),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
