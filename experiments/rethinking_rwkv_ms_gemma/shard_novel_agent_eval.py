#!/usr/bin/env python3
"""Prepare and merge auditable shards for the batch-one Novel Agent evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PLACEHOLDER_FIELD = "distributed_shard_placeholder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--main-eval-dir", type=Path, required=True)
    prepare.add_argument("--shard-dir", type=Path, required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--target-condition", required=True)
    prepare.add_argument("--owned-indices", required=True)
    prepare.add_argument("--physical-gpu", type=int, required=True)
    prepare.add_argument("--replace", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--main-eval-dir", type=Path, required=True)
    merge.add_argument("--target-condition", required=True)
    merge.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def manifest_context(main_eval_dir: Path, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(main_eval_dir / "manifest.json")
    fingerprint_payload = manifest.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ValueError("Main manifest lacks fingerprint_payload")
    datasets = fingerprint_payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {task}:
        raise ValueError(f"Main manifest does not select only task {task!r}")
    dataset = datasets[task]
    if not isinstance(dataset, dict):
        raise ValueError("Main manifest dataset entry is invalid")
    return manifest, dataset


def dataset_row_hashes(dataset: dict[str, Any]) -> list[str]:
    path = Path(str(dataset["path"])).expanduser().resolve()
    selected_rows = int(dataset["selected_rows"])
    hashes: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            hashes.append(sha256_bytes(raw_line.rstrip("\n").encode("utf-8")))
            if len(hashes) == selected_rows:
                break
    if len(hashes) != selected_rows:
        raise ValueError(
            f"Expected {selected_rows} selected rows in {path}, found {len(hashes)}"
        )
    return hashes


def parse_indices(raw: str, row_count: int) -> list[int]:
    indices = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= row_count:
        raise ValueError(f"Owned indices must be within [0, {row_count})")
    return indices


def prepare(args: argparse.Namespace) -> None:
    main_eval_dir = args.main_eval_dir.expanduser().resolve()
    shard_dir = args.shard_dir.expanduser().resolve()
    if shard_dir.exists() and any(shard_dir.iterdir()):
        if not args.replace:
            raise ValueError(f"Shard directory is not empty: {shard_dir}")
        expected_names = {
            "base.jsonl",
            "manifest.json",
            "normal.jsonl",
            "progress.json",
            "shard_plan.json",
            "summary.json",
        }
        unexpected = {path.name for path in shard_dir.iterdir()} - expected_names
        if unexpected:
            raise ValueError(
                f"Refusing to replace shard directory with unexpected files: {sorted(unexpected)}"
            )
        for path in shard_dir.iterdir():
            path.unlink()
    manifest, dataset = manifest_context(main_eval_dir, args.task)
    fingerprint_payload = manifest["fingerprint_payload"]
    conditions = fingerprint_payload.get("conditions")
    if not isinstance(conditions, list) or args.target_condition not in conditions:
        raise ValueError("Target condition is absent from the main manifest")
    row_hashes = dataset_row_hashes(dataset)
    owned_indices = parse_indices(args.owned_indices, len(row_hashes))
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("Main manifest fingerprint is invalid")

    shard_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(shard_dir / "manifest.json", manifest)
    for condition in conditions:
        skipped = (
            set(range(len(row_hashes)))
            if condition != args.target_condition
            else set(range(len(row_hashes))) - set(owned_indices)
        )
        placeholders = [
            {
                "fingerprint": fingerprint,
                "key": f"{args.task}:{line_index}",
                "condition": condition,
                "status": "ok",
                "row_sha256": row_hashes[line_index],
                PLACEHOLDER_FIELD: True,
            }
            for line_index in sorted(skipped)
        ]
        write_jsonl_atomic(shard_dir / f"{condition}.jsonl", placeholders)
    write_json_atomic(
        shard_dir / "shard_plan.json",
        {
            "schema": "novel_agent_eval_distributed_shard.v1",
            "main_eval_dir": str(main_eval_dir),
            "main_fingerprint": fingerprint,
            "task": args.task,
            "target_condition": args.target_condition,
            "owned_indices": owned_indices,
            "physical_gpu": args.physical_gpu,
            "logical_device": fingerprint_payload.get("device"),
            "row_count": len(row_hashes),
        },
    )
    print(
        f"SHARD_PREPARED dir={shard_dir} owned={len(owned_indices)} "
        f"condition={args.target_condition}",
        flush=True,
    )


def validate_generated_row(
    row: dict[str, Any],
    *,
    fingerprint: str,
    task: str,
    condition: str,
    row_hashes: list[str],
    owned_indices: set[int],
) -> int:
    if row.get(PLACEHOLDER_FIELD):
        raise ValueError("Placeholder passed to generated-row validation")
    line_index = row.get("line_index")
    if isinstance(line_index, bool) or not isinstance(line_index, int):
        raise ValueError("Generated shard row lacks line_index")
    expected = {
        "fingerprint": fingerprint,
        "key": f"{task}:{line_index}",
        "condition": condition,
        "task": task,
        "status": "ok",
        "row_sha256": row_hashes[line_index],
    }
    if line_index not in owned_indices or any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Generated shard row violates its plan at index {line_index}")
    for key in ("raw_generation", "parsed_json", "score", "gold"):
        if key not in row:
            raise ValueError(f"Generated shard row {line_index} lacks {key}")
    return line_index


def merge(args: argparse.Namespace) -> None:
    main_eval_dir = args.main_eval_dir.expanduser().resolve()
    shard_dirs = [path.expanduser().resolve() for path in args.shard_dir]
    first_plan = read_json(shard_dirs[0] / "shard_plan.json")
    task = first_plan["task"]
    manifest, dataset = manifest_context(main_eval_dir, task)
    fingerprint = str(manifest["fingerprint"])
    row_hashes = dataset_row_hashes(dataset)
    output_path = main_eval_dir / f"{args.target_condition}.jsonl"
    merged_by_index: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(output_path):
        if row.get(PLACEHOLDER_FIELD):
            raise ValueError("Main evaluation output contains a shard placeholder")
        line_index = validate_generated_row(
            row,
            fingerprint=fingerprint,
            task=task,
            condition=args.target_condition,
            row_hashes=row_hashes,
            owned_indices=set(range(len(row_hashes))),
        )
        merged_by_index[line_index] = row

    shard_audit: list[dict[str, Any]] = []
    claimed_indices: set[int] = set()
    for shard_dir in shard_dirs:
        plan_path = shard_dir / "shard_plan.json"
        plan = read_json(plan_path)
        if (
            plan.get("schema") != "novel_agent_eval_distributed_shard.v1"
            or plan.get("main_eval_dir") != str(main_eval_dir)
            or plan.get("main_fingerprint") != fingerprint
            or plan.get("task") != task
            or plan.get("target_condition") != args.target_condition
            or plan.get("row_count") != len(row_hashes)
        ):
            raise ValueError(f"Shard plan differs from the main evaluation: {plan_path}")
        owned_indices = set(plan["owned_indices"])
        overlap = claimed_indices & owned_indices
        if overlap:
            raise ValueError(f"Shard plans overlap at indices {sorted(overlap)}")
        claimed_indices.update(owned_indices)
        generated: dict[int, dict[str, Any]] = {}
        records_path = shard_dir / f"{args.target_condition}.jsonl"
        for row in read_jsonl(records_path):
            if row.get(PLACEHOLDER_FIELD):
                continue
            line_index = validate_generated_row(
                row,
                fingerprint=fingerprint,
                task=task,
                condition=args.target_condition,
                row_hashes=row_hashes,
                owned_indices=owned_indices,
            )
            if line_index in generated:
                raise ValueError(f"Duplicate generated shard row {line_index}")
            generated[line_index] = row
        if set(generated) != owned_indices:
            missing = sorted(owned_indices - set(generated))
            raise ValueError(f"Shard is incomplete at indices {missing}: {shard_dir}")
        for line_index, row in generated.items():
            existing = merged_by_index.get(line_index)
            if existing is not None and existing.get("raw_generation") != row.get("raw_generation"):
                raise ValueError(f"Conflicting generation at index {line_index}")
            merged_by_index.setdefault(line_index, row)
        shard_audit.append(
            {
                "path": str(shard_dir),
                "plan_sha256": sha256_file(plan_path),
                "records_sha256": sha256_file(records_path),
                "owned_indices": sorted(owned_indices),
                "generated_rows": len(generated),
                "physical_gpu": plan.get("physical_gpu"),
            }
        )

    expected_indices = set(range(len(row_hashes)))
    complete = set(merged_by_index) == expected_indices
    if args.require_complete and not complete:
        raise ValueError(
            f"Merged output is incomplete; missing {sorted(expected_indices - set(merged_by_index))}"
        )
    write_jsonl_atomic(output_path, [merged_by_index[index] for index in sorted(merged_by_index)])
    write_json_atomic(
        main_eval_dir / "distributed_shard_merge.json",
        {
            "schema": "novel_agent_eval_distributed_merge.v1",
            "task": task,
            "condition": args.target_condition,
            "fingerprint": fingerprint,
            "complete": complete,
            "rows": len(merged_by_index),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "shards": shard_audit,
        },
    )
    print(
        f"SHARD_MERGE_COMPLETE rows={len(merged_by_index)} complete={str(complete).lower()} "
        f"output={output_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
