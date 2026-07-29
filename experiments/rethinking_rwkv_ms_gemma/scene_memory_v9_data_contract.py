#!/usr/bin/env python3
"""Validate the frozen V9 Value14 pair-level curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    canonical_sha256,
    load_json_object,
    read_jsonl,
    require,
    sha256_file,
    validate_self_hash,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v9_data import (
    ARTIFACT_FILENAMES,
    BUNDLE_SCHEMA,
    CANONICAL_VALUE14_PAIRS,
    CHECKPOINT_STEPS,
    CURRICULUM_BINDING_SCHEMA,
    DEFAULT_OUTPUT_DIR,
    DIRECTED_PRESENTATIONS,
    PAIRS_PER_PASS,
    PAIR_PASSES,
    PAIR_SCHEDULE_ENTRY_SCHEMA,
    PAIR_SCHEDULE_MANIFEST_SCHEMA,
    SOURCE_LOCK,
    SOURCE_LOCK_SCHEMA,
    SOURCE_SCHEMA,
    TOTAL_PAIR_STEPS,
    TRAIN32_SHA256,
    VALUE14_ORDINALS,
    build_bundle_manifest,
    build_pair_schedule,
    build_pair_schedule_manifest,
    build_source_lock,
    build_source_manifest,
    hard32_exclusion,
    load_input_contract,
    schedule_audit,
)


def load_source_lock(path: Path = SOURCE_LOCK) -> dict[str, Any]:
    lock = load_json_object(path, description="V9 source lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "V9 source-lock schema differs")
    unsigned = dict(lock)
    recorded = unsigned.pop("lock_sha256", None)
    require(recorded == canonical_sha256(unsigned), "V9 source-lock checksum differs")
    require(
        lock.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V9 source-lock Hard32 exclusion differs",
    )
    require(
        lock.get("curriculum")
        == {
            "canonical_value14_pairs": [list(pair) for pair in CANONICAL_VALUE14_PAIRS],
            "value14_ordinals": list(VALUE14_ORDINALS),
            "pair_passes": PAIR_PASSES,
            "pairs_per_pass": PAIRS_PER_PASS,
            "pair_steps": TOTAL_PAIR_STEPS,
            "directed_presentations": DIRECTED_PRESENTATIONS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
        },
        "V9 source-lock curriculum differs",
    )
    return lock


def _validate_locked_artifacts(
    root: Path,
    source_lock: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    locked = source_lock.get("artifacts")
    require(isinstance(locked, dict), "V9 source-lock artifacts are missing")
    bindings: dict[str, dict[str, Any]] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        record = locked.get(name)
        require(isinstance(record, dict), f"V9 source-lock artifact missing: {name}")
        path = root / filename
        require(
            Path(str(record.get("path"))).resolve() == path.resolve(),
            f"V9 source-lock path differs: {name}",
        )
        require(path.is_file() and not path.is_symlink(), f"V9 artifact missing: {path}")
        actual_sha256 = sha256_file(path)
        require(
            actual_sha256 == record.get("sha256"),
            f"V9 artifact SHA-256 differs: {name}",
        )
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "bytes": path.stat().st_size,
        }
    return bindings


def _validate_pair_schedule_entries(
    schedule_path: Path,
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = read_jsonl(schedule_path)
    require(len(records) == TOTAL_PAIR_STEPS, "V9 materialized schedule length differs")
    entries = [payload for _, payload in records]
    directed = {
        int(entry["train_row_ordinal"]): entry
        for entry in pair_manifest["directed_pairs"]
    }
    for schedule_index, entry in enumerate(entries):
        require(
            entry.get("schema") == PAIR_SCHEDULE_ENTRY_SCHEMA,
            "V9 pair schedule entry schema differs",
        )
        validate_self_hash(entry, field="entry_sha256")
        require(
            entry.get("schedule_index") == schedule_index
            and entry.get("step") == schedule_index + 1,
            "V9 pair schedule indexing differs",
        )
        pair = entry.get("canonical_pair_ordinals")
        require(
            isinstance(pair, list)
            and len(pair) == 2
            and tuple(pair) in CANONICAL_VALUE14_PAIRS,
            "V9 schedule contains a non-canonical pair",
        )
        low, high = pair
        members = entry.get("members")
        require(
            isinstance(members, list)
            and len(members) == 2
            and entry.get("pair_batch_size") == 2,
            "V9 schedule pair batch differs",
        )
        for member, ordinal, donor, role in (
            (members[0], low, high, "canonical_low"),
            (members[1], high, low, "canonical_high"),
        ):
            require(isinstance(member, dict), "V9 schedule pair member is invalid")
            row = rows[ordinal]
            pair_entry = directed[ordinal]
            require(
                member
                == {
                    "member_role": role,
                    "train_row_ordinal": ordinal,
                    "official_source_index": row["official_source_index"],
                    "row_sha256": row["row_sha256"],
                    "row_record_sha256": row["record_sha256"],
                    "directed_pair_entry_sha256": pair_entry["entry_sha256"],
                    "donor_train_row_ordinal": donor,
                    "donor_row_sha256": pair_entry["donor_row_sha256"],
                },
                "V9 schedule pair member binding differs",
            )
    expected = build_pair_schedule(rows, pair_manifest)
    require(entries == expected, "V9 schedule is not the deterministic contract")
    schedule_audit(entries)
    return entries


def validate_bundle(
    root: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_lock_path: Path = SOURCE_LOCK,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_lock_path = source_lock_path.expanduser().resolve()
    rows, pair_manifest, input_source, input_bindings = load_input_contract()
    source_lock = load_source_lock(source_lock_path)
    artifacts = _validate_locked_artifacts(root, source_lock)
    require(source_lock.get("inputs") == input_bindings, "V9 locked inputs differ")

    schedule_path = root / "pair_schedule.jsonl"
    schedule_manifest_path = root / "pair_schedule_manifest.json"
    entries = _validate_pair_schedule_entries(schedule_path, rows, pair_manifest)
    schedule_manifest = load_json_object(
        schedule_manifest_path,
        description="V9 pair schedule manifest",
    )
    validate_self_hash(schedule_manifest)
    require(
        schedule_manifest.get("schema") == PAIR_SCHEDULE_MANIFEST_SCHEMA,
        "V9 pair schedule-manifest schema differs",
    )
    require(
        schedule_manifest.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V9 schedule-manifest Hard32 exclusion differs",
    )
    expected_schedule_manifest = build_pair_schedule_manifest(
        schedule_path=schedule_path,
        entries=entries,
        input_bindings=input_bindings,
    )
    require(
        schedule_manifest == expected_schedule_manifest,
        "V9 pair schedule manifest differs from deterministic reconstruction",
    )

    source_manifest_path = root / "source_manifest.json"
    source_manifest = load_json_object(
        source_manifest_path,
        description="V9 source manifest",
    )
    validate_self_hash(source_manifest)
    require(source_manifest.get("schema") == SOURCE_SCHEMA, "V9 source schema differs")
    curriculum = source_manifest.get("v9_pair_curriculum")
    require(
        isinstance(curriculum, dict)
        and curriculum.get("schema") == CURRICULUM_BINDING_SCHEMA,
        "V9 source curriculum binding differs",
    )
    require(
        source_manifest.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V9 source-manifest Hard32 exclusion differs",
    )
    expected_source_manifest = build_source_manifest(
        input_source_manifest=input_source,
        input_bindings=input_bindings,
        schedule_path=schedule_path,
        schedule_manifest_path=schedule_manifest_path,
        schedule_manifest=schedule_manifest,
    )
    require(
        source_manifest == expected_source_manifest,
        "V9 source manifest differs from locked inputs",
    )

    bundle = load_json_object(root / "manifest.json", description="V9 bundle manifest")
    validate_self_hash(bundle)
    require(bundle.get("schema") == BUNDLE_SCHEMA, "V9 bundle schema differs")
    expected_bundle = build_bundle_manifest(
        output_dir=root,
        input_bindings=input_bindings,
        schedule_manifest=schedule_manifest,
    )
    require(bundle == expected_bundle, "V9 bundle manifest differs from its artifacts")
    require(
        bundle.get("leakage")
        == {
            "source_split": "train",
            "val_rows_in_schedule": 0,
            "test_rows_in_schedule": 0,
            "hard32_rows_in_schedule": 0,
            "proof": "hash_locked_train32_pair_ordinals_only_v1",
        },
        "V9 leakage proof differs",
    )
    require(
        bundle.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V9 bundle Hard32 exclusion differs",
    )

    expected_lock = build_source_lock(root)
    require(source_lock == expected_lock, "V9 checked-in source lock differs from bundle")
    audit = schedule_manifest["curriculum"]
    return {
        "status": "pass",
        "root": str(root),
        "train32_sha256": TRAIN32_SHA256,
        "pair_schedule_sha256": sha256_file(schedule_path),
        "pair_schedule_entries_sha256": audit["entries_sha256"],
        "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
        "pair_schedule_manifest_sha256": schedule_manifest["manifest_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "pair_steps": audit["pair_steps"],
        "directed_presentations": audit["directed_presentations"],
        "checkpoint_steps": audit["checkpoint_steps"],
        "canonical_value14_pairs": audit["canonical_value14_pairs"],
        "hard32_rows_in_schedule": bundle["leakage"]["hard32_rows_in_schedule"],
        "hard32_exclusion": bundle["excluded_artifacts"]["hard32"],
        "source_lock": {
            "path": str(source_lock_path),
            "sha256": sha256_file(source_lock_path),
            "lock_sha256": source_lock["lock_sha256"],
        },
        "artifacts": artifacts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_bundle(args.root, source_lock_path=args.source_lock)
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
