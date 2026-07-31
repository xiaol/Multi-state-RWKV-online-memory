#!/usr/bin/env python3
"""Validate the V15 all-Train32 symmetric-pair data and schedule bundle."""

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
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v15_data import (
    ALL32_ORDINALS,
    ARTIFACT_FILENAMES,
    BUNDLE_SCHEMA,
    CANONICAL_ALL32_PAIRS,
    CHECKPOINT_CYCLES,
    CURRICULUM_BINDING_SCHEMA,
    DEFAULT_OUTPUT_DIR,
    DIRECTED_PRESENTATIONS,
    DIRECTED_STRATUM_COUNTS,
    EMPTY_ORDINALS,
    FULL_PAIR_CYCLES,
    PAIR_MANIFEST_FILE_SHA256,
    PAIRS_PER_CYCLE,
    PAIR_CYCLES,
    PAIR_PREFIX_SHA256_BY_CHECKPOINT,
    PAIR_SCHEDULE_ENTRY_SCHEMA,
    PAIR_SCHEDULE_MANIFEST_SCHEMA,
    PAIR_STRATUM_COUNTS,
    PRESENTATION_CHECKPOINTS,
    SOURCE_LOCK,
    SOURCE_LOCK_SCHEMA,
    SOURCE_MANIFEST_FILE_SHA256,
    SOURCE_SCHEMA,
    TOTAL_PAIR_PRESENTATIONS,
    TRAIN32_SHA256,
    TRAIN32_ROWS_SHA256,
    V7_ROOT,
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
    lock = load_json_object(path, description="V15 source lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "V15 source-lock schema differs")
    unsigned = dict(lock)
    recorded = unsigned.pop("lock_sha256", None)
    require(recorded == canonical_sha256(unsigned), "V15 source-lock checksum differs")
    require(
        lock.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V15 source-lock Hard32 exclusion differs",
    )
    require(
        lock.get("curriculum")
        == {
            "canonical_all32_pairs": [list(pair) for pair in CANONICAL_ALL32_PAIRS],
            "all32_ordinals": list(ALL32_ORDINALS),
            "empty_ordinals": list(EMPTY_ORDINALS),
            "pair_stratum_counts_per_cycle": dict(PAIR_STRATUM_COUNTS),
            "pair_cycles": PAIR_CYCLES,
            "pairs_per_cycle": PAIRS_PER_CYCLE,
            "pair_presentations": TOTAL_PAIR_PRESENTATIONS,
            "directed_presentations": DIRECTED_PRESENTATIONS,
            "checkpoint_cycles": list(CHECKPOINT_CYCLES),
            "presentation_checkpoints": list(PRESENTATION_CHECKPOINTS),
            "pair_prefix_sha256_by_checkpoint": dict(
                PAIR_PREFIX_SHA256_BY_CHECKPOINT
            ),
        },
        "V15 source-lock curriculum differs",
    )
    return lock


def _validate_locked_artifacts(
    root: Path,
    source_lock: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    locked = source_lock.get("artifacts")
    require(isinstance(locked, dict), "V15 source-lock artifacts are missing")
    bindings: dict[str, dict[str, Any]] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        record = locked.get(name)
        require(isinstance(record, dict), f"V15 source-lock artifact missing: {name}")
        path = root / filename
        require(
            Path(str(record.get("path"))).resolve() == path.resolve(),
            f"V15 source-lock path differs: {name}",
        )
        require(path.is_file() and not path.is_symlink(), f"V15 artifact missing: {path}")
        actual_sha256 = sha256_file(path)
        require(
            actual_sha256 == record.get("sha256"),
            f"V15 artifact SHA-256 differs: {name}",
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
    require(
        len(records) == TOTAL_PAIR_PRESENTATIONS,
        "V15 materialized schedule length differs",
    )
    entries = [payload for _, payload in records]
    directed = {
        int(entry["train_row_ordinal"]): entry
        for entry in pair_manifest["directed_pairs"]
    }
    for schedule_index, entry in enumerate(entries):
        require(
            entry.get("schema") == PAIR_SCHEDULE_ENTRY_SCHEMA,
            "V15 pair schedule entry schema differs",
        )
        validate_self_hash(entry, field="entry_sha256")
        require(
            entry.get("schedule_index") == schedule_index
            and entry.get("presentation") == schedule_index + 1,
            "V15 pair schedule indexing differs",
        )
        cycle_index = schedule_index // PAIRS_PER_CYCLE + 1
        cycle_position = schedule_index % PAIRS_PER_CYCLE
        expected_pair = FULL_PAIR_CYCLES[cycle_index - 1][cycle_position]
        pair = entry.get("canonical_pair_ordinals")
        require(
            pair == list(expected_pair)
            and entry.get("cycle_index") == cycle_index
            and entry.get("cycle_position") == cycle_position
            and entry.get("phase") == "all32_symmetric_pair"
            and entry.get("pair_batch_size") == 2,
            "V15 pair schedule cycle order differs",
        )
        low, high = expected_pair
        members = entry.get("members")
        require(
            isinstance(members, list)
            and len(members) == 2
            and [member.get("train_row_ordinal") for member in members]
            == [low, high]
            and [member.get("donor_train_row_ordinal") for member in members]
            == [high, low],
            "V15 schedule pair members are not reciprocal",
        )
        for member, ordinal in zip(members, (low, high)):
            row = rows[ordinal]
            pair_entry = directed[ordinal]
            require(
                member.get("source_split") == "train"
                and member.get("official_source_index")
                == row["official_source_index"]
                and member.get("row_sha256") == row["row_sha256"]
                and member.get("row_record_sha256") == row["record_sha256"]
                and member.get("label_sha256") == row["label_sha256"]
                and member.get("base_record_sha256")
                == row["base_record_sha256"]
                and member.get("strict_failure_stratum")
                == row["strict_failure_stratum"]
                and member.get("directed_pair_entry_sha256")
                == pair_entry["entry_sha256"],
                "V15 schedule member binding differs",
            )
        empty_members = [
            member
            for member in members
            if member.get("target_presence_role") == "empty"
        ]
        if entry.get("target_stratum") == "presence":
            require(
                len(empty_members) == 1
                and entry.get("empty_member_ordinals")
                == [empty_members[0]["train_row_ordinal"]],
                "V15 presence pair does not bind exactly one empty target",
            )
        else:
            require(
                not empty_members and entry.get("empty_member_ordinals") == [],
                "V15 value pair unexpectedly binds an empty target",
            )
    expected = build_pair_schedule(rows, pair_manifest)
    require(entries == expected, "V15 schedule is not the deterministic contract")
    schedule_audit(entries)
    return entries


def validate_bundle(
    root: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_lock_path: Path = SOURCE_LOCK,
    v7_root: Path = V7_ROOT,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_lock_path = source_lock_path.expanduser().resolve()
    rows, pair_manifest, input_source, input_bindings = load_input_contract(v7_root)
    source_lock = load_source_lock(source_lock_path)
    artifacts = _validate_locked_artifacts(root, source_lock)
    require(source_lock.get("inputs") == input_bindings, "V15 locked inputs differ")

    schedule_path = root / "pair_schedule.jsonl"
    schedule_manifest_path = root / "pair_schedule_manifest.json"
    entries = _validate_pair_schedule_entries(schedule_path, rows, pair_manifest)
    schedule_manifest = load_json_object(
        schedule_manifest_path,
        description="V15 pair schedule manifest",
    )
    validate_self_hash(schedule_manifest)
    require(
        schedule_manifest.get("schema") == PAIR_SCHEDULE_MANIFEST_SCHEMA,
        "V15 pair schedule-manifest schema differs",
    )
    require(
        schedule_manifest.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V15 schedule-manifest Hard32 exclusion differs",
    )
    expected_schedule_manifest = build_pair_schedule_manifest(
        schedule_path=schedule_path,
        entries=entries,
        input_bindings=input_bindings,
    )
    require(
        schedule_manifest == expected_schedule_manifest,
        "V15 pair schedule manifest differs from deterministic reconstruction",
    )

    source_manifest_path = root / "source_manifest.json"
    source_manifest = load_json_object(
        source_manifest_path,
        description="V15 source manifest",
    )
    validate_self_hash(source_manifest)
    require(
        source_manifest.get("schema") == SOURCE_SCHEMA,
        "V15 source schema differs",
    )
    curriculum = source_manifest.get("v15_pair_curriculum")
    require(
        isinstance(curriculum, dict)
        and curriculum.get("schema") == CURRICULUM_BINDING_SCHEMA,
        "V15 source curriculum binding differs",
    )
    require(
        source_manifest.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V15 source-manifest Hard32 exclusion differs",
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
        "V15 source manifest differs from locked inputs",
    )

    bundle = load_json_object(root / "manifest.json", description="V15 bundle manifest")
    validate_self_hash(bundle)
    require(bundle.get("schema") == BUNDLE_SCHEMA, "V15 bundle schema differs")
    expected_bundle = build_bundle_manifest(
        output_dir=root,
        input_bindings=input_bindings,
        schedule_manifest=schedule_manifest,
    )
    require(bundle == expected_bundle, "V15 bundle manifest differs from its artifacts")
    require(
        bundle.get("leakage")
        == {
            "source_split": "train",
            "val_rows_in_schedule": 0,
            "test_rows_in_schedule": 0,
            "hard32_rows_in_schedule": 0,
            "proof": "four_hash_locked_train32_inputs_only_all32_ordinals_v1",
        },
        "V15 leakage proof differs",
    )
    require(
        bundle.get("excluded_artifacts") == {"hard32": hard32_exclusion()},
        "V15 bundle Hard32 exclusion differs",
    )

    expected_lock = build_source_lock(root)
    require(source_lock == expected_lock, "V15 checked-in source lock differs from bundle")
    audit = schedule_manifest["curriculum"]
    return {
        "status": "pass",
        "root": str(root),
        "train32_sha256": TRAIN32_SHA256,
        "scheduled_train_rows": len(audit["all32_ordinals"]),
        "base_failure_rows": 32,
        "empty_rows": len(audit["empty_ordinals"]),
        "pair_cycles": audit["pair_cycles"],
        "pairs_per_cycle": audit["pairs_per_cycle"],
        "pair_presentations": audit["pair_presentations"],
        "directed_presentations": audit["directed_presentations"],
        "pair_stratum_counts_per_cycle": audit["pair_stratum_counts_per_cycle"],
        "directed_stratum_counts_per_cycle": audit[
            "directed_stratum_counts_per_cycle"
        ],
        "presentation_checkpoints": audit["presentation_checkpoints"],
        "pair_prefix_sha256_by_checkpoint": audit[
            "pair_prefix_sha256_by_checkpoint"
        ],
        "pair_schedule_sha256": sha256_file(schedule_path),
        "pair_schedule_entries_sha256": audit["entries_sha256"],
        "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
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
    parser.add_argument("--v7-root", type=Path, default=V7_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_bundle(
            args.root,
            source_lock_path=args.source_lock,
            v7_root=args.v7_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
