#!/usr/bin/env python3
"""Validate the frozen V8 value-first and balanced Train32 curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    HARD32_FILE,
    HARD32_FILE_SHA256,
    canonical_sha256,
    load_json_object,
    read_jsonl,
    require,
    sha256_file,
    validate_self_hash,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v8_data import (
    ARTIFACT_FILENAMES,
    BALANCED_QUOTAS,
    BUNDLE_SCHEMA,
    CHECKPOINT_STEPS,
    CURRICULUM_BINDING_SCHEMA,
    DEFAULT_OUTPUT_DIR,
    PARENT_TRAIN32_SHA256,
    SCHEDULE_ENTRY_SCHEMA,
    SCHEDULE_MANIFEST_SCHEMA,
    SOURCE_LOCK,
    SOURCE_LOCK_SCHEMA,
    SOURCE_SCHEMA,
    TOTAL_STEPS,
    VALUE14_ORDINALS,
    VALUE14_PASSES,
    VALUE14_STEPS,
    build_bundle_manifest,
    build_schedule,
    build_schedule_manifest,
    build_source_lock,
    build_source_manifest,
    load_parent_contract,
    schedule_audit,
)


def load_source_lock(path: Path = SOURCE_LOCK) -> dict[str, Any]:
    lock = load_json_object(path, description="V8 source lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "V8 source-lock schema differs")
    unsigned = dict(lock)
    recorded = unsigned.pop("lock_sha256", None)
    require(recorded == canonical_sha256(unsigned), "V8 source-lock checksum differs")
    require(
        lock.get("parent_train32_sha256") == PARENT_TRAIN32_SHA256,
        "V8 source lock does not bind the frozen Train32",
    )
    require(
        lock.get("fixed_hard32")
        == {
            "path": str(HARD32_FILE),
            "sha256": HARD32_FILE_SHA256,
            "role": "protected_evaluation_only_not_scheduled",
        },
        "V8 source-lock Hard32 role differs",
    )
    require(
        lock.get("curriculum")
        == {
            "value14_ordinals": list(VALUE14_ORDINALS),
            "value14_passes": VALUE14_PASSES,
            "value14_steps": VALUE14_STEPS,
            "balanced_quotas": dict(BALANCED_QUOTAS),
            "balanced_steps": 96,
            "total_steps": TOTAL_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
        },
        "V8 source-lock curriculum differs",
    )
    return lock


def _validate_locked_artifacts(
    root: Path,
    source_lock: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    locked = source_lock.get("artifacts")
    require(isinstance(locked, dict), "V8 source-lock artifacts are missing")
    bindings: dict[str, dict[str, Any]] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        record = locked.get(name)
        require(isinstance(record, dict), f"V8 source-lock artifact is missing: {name}")
        path = root / filename
        require(
            Path(str(record.get("path"))).resolve() == path.resolve(),
            f"V8 source-lock path differs: {name}",
        )
        require(path.is_file() and not path.is_symlink(), f"V8 artifact is missing: {path}")
        actual = sha256_file(path)
        require(actual == record.get("sha256"), f"V8 artifact SHA-256 differs: {name}")
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return bindings


def _validate_schedule_entries(
    schedule_path: Path,
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = read_jsonl(schedule_path)
    require(len(records) == TOTAL_STEPS, "V8 materialized schedule length differs")
    entries = [payload for _, payload in records]
    pair_by_ordinal = {
        int(entry["train_row_ordinal"]): entry
        for entry in pair_manifest["directed_pairs"]
    }
    for schedule_index, entry in enumerate(entries):
        require(entry.get("schema") == SCHEDULE_ENTRY_SCHEMA, "V8 schedule entry schema differs")
        validate_self_hash(entry, field="entry_sha256")
        require(entry.get("schedule_index") == schedule_index, "V8 schedule index differs")
        require(entry.get("step") == schedule_index + 1, "V8 schedule step differs")
        ordinal = entry.get("train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < len(rows),
            "V8 schedule original row ordinal is invalid",
        )
        row = rows[ordinal]
        pair = pair_by_ordinal[ordinal]
        require(entry.get("row_sha256") == row["row_sha256"], "V8 row hash binding differs")
        require(
            entry.get("row_record_sha256") == row["record_sha256"],
            "V8 row-record binding differs",
        )
        require(
            entry.get("official_source_index") == row["official_source_index"],
            "V8 official train source index differs",
        )
        require(
            entry.get("pair_entry_sha256") == pair["entry_sha256"],
            "V8 pair-entry binding differs",
        )
        require(
            entry.get("target_stratum") == pair["target_stratum"],
            "V8 target stratum differs",
        )
        require(
            entry.get("donor_train_row_ordinal") == pair["donor_train_row_ordinal"]
            and entry.get("donor_row_sha256") == pair["donor_row_sha256"],
            "V8 donor binding differs",
        )
    expected = build_schedule(rows, pair_manifest)
    require(entries == expected, "V8 materialized schedule is not the deterministic contract")
    schedule_audit(entries)
    return entries


def validate_bundle(
    root: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_lock_path: Path = SOURCE_LOCK,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    source_lock_path = source_lock_path.expanduser().resolve()
    rows, pair_manifest, parent_source, parent_bindings = load_parent_contract()
    source_lock = load_source_lock(source_lock_path)
    artifacts = _validate_locked_artifacts(root, source_lock)
    require(source_lock.get("parent_v7") == parent_bindings, "V8 parent V7 lock differs")

    schedule_path = root / "schedule.jsonl"
    schedule_manifest_path = root / "schedule_manifest.json"
    entries = _validate_schedule_entries(schedule_path, rows, pair_manifest)

    schedule_manifest = load_json_object(
        schedule_manifest_path,
        description="V8 schedule manifest",
    )
    validate_self_hash(schedule_manifest)
    require(
        schedule_manifest.get("schema") == SCHEDULE_MANIFEST_SCHEMA,
        "V8 schedule-manifest schema differs",
    )
    expected_schedule_manifest = build_schedule_manifest(
        schedule_path=schedule_path,
        entries=entries,
        parent_bindings=parent_bindings,
    )
    require(
        schedule_manifest == expected_schedule_manifest,
        "V8 schedule manifest differs from its deterministic reconstruction",
    )

    source_manifest_path = root / "source_manifest.json"
    source_manifest = load_json_object(
        source_manifest_path,
        description="V8 source manifest",
    )
    validate_self_hash(source_manifest)
    require(source_manifest.get("schema") == SOURCE_SCHEMA, "V8 source schema differs")
    binding = source_manifest.get("v8_curriculum")
    require(
        isinstance(binding, dict)
        and binding.get("schema") == CURRICULUM_BINDING_SCHEMA,
        "V8 curriculum binding differs",
    )
    expected_source_manifest = build_source_manifest(
        v7_root=Path(parent_bindings["train32"]["path"]).parent,
        parent_source_manifest=parent_source,
        schedule_path=schedule_path,
        schedule_manifest_path=schedule_manifest_path,
        schedule_manifest=schedule_manifest,
    )
    require(
        source_manifest == expected_source_manifest,
        "V8 source manifest differs from its locked inputs",
    )

    bundle = load_json_object(root / "manifest.json", description="V8 bundle manifest")
    validate_self_hash(bundle)
    require(bundle.get("schema") == BUNDLE_SCHEMA, "V8 bundle schema differs")
    expected_bundle = build_bundle_manifest(
        output_dir=root,
        parent_bindings=parent_bindings,
        schedule_manifest=schedule_manifest,
    )
    require(bundle == expected_bundle, "V8 bundle manifest differs from its artifacts")
    require(
        bundle.get("leakage")
        == {
            "source_split": "train",
            "val_rows_in_schedule": 0,
            "test_rows_in_schedule": 0,
            "hard32_rows_in_schedule": 0,
            "train32_hard32_shared_normalized_paragraphs": 0,
            "proof": "validated_parent_v7_contract_plus_ordinal_only_schedule_v1",
        },
        "V8 leakage proof differs",
    )

    expected_lock = build_source_lock(root)
    require(source_lock == expected_lock, "V8 checked-in source lock differs from bundle")
    audit = schedule_manifest["curriculum"]
    return {
        "status": "pass",
        "root": str(root),
        "parent_train32_sha256": PARENT_TRAIN32_SHA256,
        "schedule_sha256": sha256_file(schedule_path),
        "schedule_entries_sha256": audit["entries_sha256"],
        "schedule_manifest_sha256": schedule_manifest["manifest_sha256"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "total_steps": audit["total_steps"],
        "checkpoint_steps": audit["checkpoint_steps"],
        "value14_ordinals": audit["phases"]["value14"]["value14_ordinals"],
        "balanced_counts": audit["phases"]["balanced"]["observed_counts"],
        "hard32_rows_in_schedule": bundle["leakage"]["hard32_rows_in_schedule"],
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
