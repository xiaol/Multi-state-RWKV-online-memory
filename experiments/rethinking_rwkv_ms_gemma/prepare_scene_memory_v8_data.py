#!/usr/bin/env python3
"""Build the hash-locked V8 curriculum over the frozen V7 Train32 rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    ContractError,
    DEFAULT_OUTPUT_DIR as V7_ROOT,
    HARD32_FILE,
    HARD32_FILE_SHA256,
    TARGET_STRATA,
    TASK,
    canonical_sha256,
    load_json_object,
    read_jsonl,
    require,
    sha256_file,
    validate_self_hash,
    with_self_hash,
    write_json,
    write_jsonl,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v7_data_contract import (
    SOURCE_LOCK as V7_SOURCE_LOCK,
)


SCHEDULE_SCHEMA = "rwkv_ms_scene_memory_v8_schedule.v1"
SCHEDULE_ENTRY_SCHEMA = "rwkv_ms_scene_memory_v8_schedule_entry.v1"
SCHEDULE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v8_schedule_manifest.v1"
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v8_source.v1"
CURRICULUM_BINDING_SCHEMA = "rwkv_ms_scene_memory_v8_curriculum_binding.v1"
BUNDLE_SCHEMA = "rwkv_ms_scene_memory_v8_bundle.v1"
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v8_source_lock.v1"

PARENT_TRAIN32_SHA256 = (
    "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9"
)
PARENT_ROWS_SHA256 = (
    "af80b1938196319e6595a0e6d0e2f2c9a6009963a82fea520d608e060a4fe957"
)
PARENT_PAIR_MANIFEST_SHA256 = (
    "13555da56823d9597bf061d51ec6575db25cde49c044cab378f2050373fd78b6"
)
PARENT_SOURCE_MANIFEST_SHA256 = (
    "57626d3629e055a5f7900ed2a8526d357890d63730aa55d56ee72bd56a05017b"
)
PARENT_BUNDLE_MANIFEST_SHA256 = (
    "d5e2152dde19cbb1d3840fecc3540a1c057b85af2dfac26dd96b399d19cb1578"
)
PARENT_SOURCE_LOCK_SHA256 = (
    "6d5ef6a0db7a41f87140a9d56801727b92670500b3e8907ac8c82de0061afdb9"
)

VALUE14_ORDINALS = (1, 3, 5, 9, 10, 14, 19, 20, 22, 23, 24, 26, 28, 31)
VALUE14_PASSES = 4
VALUE14_STEPS = 56
BALANCED_STEPS_PER_STRATUM = 32
BALANCED_STEPS = 96
TOTAL_STEPS = 152
CHECKPOINT_STEPS = (14, 28, 42, 56, 80, 104, 128, 152)
BALANCED_QUOTAS = {
    "presence": 32,
    "same_cardinality_value": 32,
    "cross_cardinality_value": 32,
}
VALUE_SHUFFLE_NAMESPACE = "rwkv_ms_scene_memory_v8_value14_pass_shuffle.v1"
BALANCED_DRAW_NAMESPACE = "rwkv_ms_scene_memory_v8_balanced_stratum_draw.v1"
BALANCED_INTERLEAVE_NAMESPACE = "rwkv_ms_scene_memory_v8_balanced_round_order.v1"

DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v8_value14_balanced152_v1"
)
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v8_source_lock.json")
ARTIFACT_FILENAMES = {
    "bundle_manifest": "manifest.json",
    "schedule": "schedule.jsonl",
    "schedule_manifest": "schedule_manifest.json",
    "source_manifest": "source_manifest.json",
}
PARENT_FILENAMES = {
    "bundle_manifest": ("manifest.json", PARENT_BUNDLE_MANIFEST_SHA256),
    "train32": ("train32.jsonl", PARENT_TRAIN32_SHA256),
    "train32_rows": ("train32_rows.jsonl", PARENT_ROWS_SHA256),
    "train32_pair_manifest": (
        "train32_pair_manifest.json",
        PARENT_PAIR_MANIFEST_SHA256,
    ),
    "train32_source_manifest": (
        "train32_source_manifest.json",
        PARENT_SOURCE_MANIFEST_SHA256,
    ),
}


def _verify_file(path: Path, expected_sha256: str, description: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    require(sha256_file(path) == expected_sha256, f"{description} SHA-256 differs")


def _parent_bindings(v7_root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for name, (filename, expected_sha256) in PARENT_FILENAMES.items():
        path = v7_root / filename
        _verify_file(path, expected_sha256, f"V7 {name}")
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": expected_sha256,
            "bytes": path.stat().st_size,
        }
    _verify_file(V7_SOURCE_LOCK, PARENT_SOURCE_LOCK_SHA256, "V7 source lock")
    bindings["source_lock"] = {
        "path": str(V7_SOURCE_LOCK.resolve()),
        "sha256": PARENT_SOURCE_LOCK_SHA256,
        "bytes": V7_SOURCE_LOCK.stat().st_size,
    }
    return bindings


def load_parent_contract(
    v7_root: Path = V7_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    v7_root = v7_root.expanduser().resolve()
    bindings = _parent_bindings(v7_root)

    # V8 pre-gate validation must not inspect the protected Hard32 rows. The
    # exact, pinned V7 bundle and source-lock bytes already carry that binding.
    parent_bundle = load_json_object(
        v7_root / "manifest.json",
        description="V7 bundle manifest",
    )
    validate_self_hash(parent_bundle)
    require(
        parent_bundle.get("schema") == "rwkv_ms_scene_memory_v7_bundle.v1",
        "V7 bundle schema differs",
    )
    require(parent_bundle.get("task") == TASK, "V7 bundle task differs")
    parent_artifacts = parent_bundle.get("artifacts")
    require(isinstance(parent_artifacts, dict), "V7 bundle artifacts are missing")
    for name in ("train32", "train32_rows", "train32_pair_manifest", "train32_source_manifest"):
        require(
            parent_artifacts.get(name) == bindings[name],
            f"V7 bundle Train32 artifact differs: {name}",
        )
    leakage = parent_bundle.get("leakage")
    require(
        isinstance(leakage, dict)
        and leakage.get("val_rows_in_training") == 0
        and leakage.get("test_rows_in_training") == 0
        and leakage.get("train32_hard32_shared_normalized_paragraphs") == 0,
        "V7 bundle leakage contract differs",
    )
    fixed_hard32 = parent_bundle.get("fixed_hard32")
    require(
        isinstance(fixed_hard32, dict)
        and isinstance(fixed_hard32.get("data"), dict)
        and fixed_hard32["data"].get("path") == str(HARD32_FILE)
        and fixed_hard32["data"].get("sha256") == HARD32_FILE_SHA256,
        "V7 bundle protected Hard32 binding differs",
    )

    parent_lock = load_json_object(V7_SOURCE_LOCK, description="V7 source lock")
    unsigned_parent_lock = dict(parent_lock)
    recorded_parent_lock_sha256 = unsigned_parent_lock.pop("lock_sha256", None)
    require(
        recorded_parent_lock_sha256 == canonical_sha256(unsigned_parent_lock),
        "V7 source-lock checksum differs",
    )
    require(
        parent_lock.get("schema") == "rwkv_ms_scene_memory_v7_source_lock.v1",
        "V7 source-lock schema differs",
    )
    locked_artifacts = parent_lock.get("artifacts")
    require(isinstance(locked_artifacts, dict), "V7 source-lock artifacts are missing")
    for name in ("train32", "train32_rows", "train32_pair_manifest", "train32_source_manifest"):
        expected = bindings[name]
        require(
            locked_artifacts.get(name)
            == {"path": expected["path"], "sha256": expected["sha256"]},
            f"V7 source-lock Train32 artifact differs: {name}",
        )
    locked_hard32 = parent_lock.get("fixed_hard32")
    require(
        isinstance(locked_hard32, dict)
        and locked_hard32.get("data_sha256") == HARD32_FILE_SHA256,
        "V7 source-lock protected Hard32 binding differs",
    )

    data_records = read_jsonl(v7_root / "train32.jsonl")
    row_records = read_jsonl(v7_root / "train32_rows.jsonl")
    require(len(data_records) == len(row_records) == 32, "V7 Train32 must contain 32 rows")
    rows = [payload for _, payload in row_records]
    for ordinal, ((raw_line, _), row) in enumerate(zip(data_records, rows)):
        validate_self_hash(row, field="record_sha256")
        require(row.get("train_row_ordinal") == ordinal, "V7 row ordinal differs")
        require(row.get("source_split") == "train", "V7 row is not official train")
        require(
            row.get("row_sha256") == hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            "V7 row data binding differs",
        )
    pair_manifest = load_json_object(
        v7_root / "train32_pair_manifest.json",
        description="V7 Train32 pair manifest",
    )
    validate_self_hash(pair_manifest)
    source_manifest = load_json_object(
        v7_root / "train32_source_manifest.json",
        description="V7 Train32 source manifest",
    )
    validate_self_hash(source_manifest)
    contract = source_manifest.get("contract")
    require(isinstance(contract, dict), "V7 source contract is missing")
    require(contract.get("source_split") == "train", "V7 source split differs")
    require(
        contract.get("val_rows") == contract.get("test_rows") == 0,
        "V7 source includes validation or test rows",
    )
    return rows, pair_manifest, source_manifest, bindings


def _pair_entries_by_ordinal(
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    entries = pair_manifest.get("directed_pairs")
    require(isinstance(entries, list) and len(entries) == 32, "V7 pair entries differ")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "V7 pair entry is invalid")
        validate_self_hash(entry, field="entry_sha256")
        ordinal = entry.get("train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < 32
            and ordinal not in by_ordinal,
            "V7 pair ordinal is invalid or duplicated",
        )
        require(entry.get("target_stratum") in TARGET_STRATA, "V7 target stratum differs")
        by_ordinal[ordinal] = entry
    require(set(by_ordinal) == set(range(32)), "V7 pair entries do not cover Train32")
    observed_value = tuple(
        ordinal
        for ordinal in range(32)
        if by_ordinal[ordinal]["target_stratum"]
        in {"same_cardinality_value", "cross_cardinality_value"}
    )
    require(observed_value == VALUE14_ORDINALS, "locked Value14 ordinals differ")
    return by_ordinal


def _shuffle_key(namespace: str, *parts: object) -> str:
    material = "\0".join((namespace, *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _ordered_ordinals(
    ordinals: Sequence[int],
    *,
    rows: Sequence[Mapping[str, Any]],
    namespace: str,
    pass_index: int,
) -> list[int]:
    return sorted(
        ordinals,
        key=lambda ordinal: (
            _shuffle_key(
                namespace,
                pass_index,
                ordinal,
                rows[ordinal]["row_sha256"],
            ),
            ordinal,
        ),
    )


def _schedule_entry(
    *,
    schedule_index: int,
    phase: str,
    phase_step: int,
    ordinal: int,
    row: Mapping[str, Any],
    pair: Mapping[str, Any],
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SCHEDULE_ENTRY_SCHEMA,
        "schedule_index": schedule_index,
        "step": schedule_index + 1,
        "phase": phase,
        "phase_step": phase_step,
        "train_row_ordinal": ordinal,
        "official_source_index": row["official_source_index"],
        "row_sha256": row["row_sha256"],
        "row_record_sha256": row["record_sha256"],
        "pair_entry_sha256": pair["entry_sha256"],
        "target_stratum": pair["target_stratum"],
        "donor_train_row_ordinal": pair["donor_train_row_ordinal"],
        "donor_row_sha256": pair["donor_row_sha256"],
        "sampling": dict(sampling),
    }
    return with_self_hash(payload, field="entry_sha256")


def build_schedule(
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    require(len(rows) == 32, "V8 schedule requires the original 32 V7 rows")
    by_ordinal = _pair_entries_by_ordinal(pair_manifest)
    entries: list[dict[str, Any]] = []
    for pass_index in range(VALUE14_PASSES):
        ordered = _ordered_ordinals(
            VALUE14_ORDINALS,
            rows=rows,
            namespace=VALUE_SHUFFLE_NAMESPACE,
            pass_index=pass_index,
        )
        for pass_position, ordinal in enumerate(ordered):
            entries.append(
                _schedule_entry(
                    schedule_index=len(entries),
                    phase="value14",
                    phase_step=len(entries),
                    ordinal=ordinal,
                    row=rows[ordinal],
                    pair=by_ordinal[ordinal],
                    sampling={
                        "mode": "deterministic_hash_sorted_pass",
                        "namespace": VALUE_SHUFFLE_NAMESPACE,
                        "pass_index": pass_index,
                        "pass_position": pass_position,
                        "shuffle_key_sha256": _shuffle_key(
                            VALUE_SHUFFLE_NAMESPACE,
                            pass_index,
                            ordinal,
                            rows[ordinal]["row_sha256"],
                        ),
                    },
                )
            )
    require(len(entries) == VALUE14_STEPS, "V8 Value14 phase length differs")

    ordinals_by_stratum = {
        stratum: tuple(
            ordinal
            for ordinal in range(32)
            if by_ordinal[ordinal]["target_stratum"] == stratum
        )
        for stratum in TARGET_STRATA
    }
    draws: dict[str, list[tuple[int, int, int, str]]] = {}
    for stratum, pool in ordinals_by_stratum.items():
        require(bool(pool), f"V8 balanced pool is empty: {stratum}")
        stratum_draws: list[tuple[int, int, int, str]] = []
        cycle = 0
        while len(stratum_draws) < BALANCED_STEPS_PER_STRATUM:
            ordered = _ordered_ordinals(
                pool,
                rows=rows,
                namespace=f"{BALANCED_DRAW_NAMESPACE}:{stratum}",
                pass_index=cycle,
            )
            for cycle_position, ordinal in enumerate(ordered):
                if len(stratum_draws) == BALANCED_STEPS_PER_STRATUM:
                    break
                key = _shuffle_key(
                    f"{BALANCED_DRAW_NAMESPACE}:{stratum}",
                    cycle,
                    ordinal,
                    rows[ordinal]["row_sha256"],
                )
                stratum_draws.append((ordinal, cycle, cycle_position, key))
            cycle += 1
        draws[stratum] = stratum_draws

    for round_index in range(BALANCED_STEPS_PER_STRATUM):
        stratum_order = sorted(
            TARGET_STRATA,
            key=lambda stratum: (
                _shuffle_key(BALANCED_INTERLEAVE_NAMESPACE, round_index, stratum),
                stratum,
            ),
        )
        for round_position, stratum in enumerate(stratum_order):
            ordinal, cycle, cycle_position, draw_key = draws[stratum][round_index]
            entries.append(
                _schedule_entry(
                    schedule_index=len(entries),
                    phase="balanced",
                    phase_step=len(entries) - VALUE14_STEPS,
                    ordinal=ordinal,
                    row=rows[ordinal],
                    pair=by_ordinal[ordinal],
                    sampling={
                        "mode": "deterministic_balanced_stratum_round",
                        "draw_namespace": f"{BALANCED_DRAW_NAMESPACE}:{stratum}",
                        "interleave_namespace": BALANCED_INTERLEAVE_NAMESPACE,
                        "round_index": round_index,
                        "round_position": round_position,
                        "stratum_draw_index": round_index,
                        "stratum_cycle": cycle,
                        "stratum_cycle_position": cycle_position,
                        "draw_shuffle_key_sha256": draw_key,
                        "interleave_key_sha256": _shuffle_key(
                            BALANCED_INTERLEAVE_NAMESPACE,
                            round_index,
                            stratum,
                        ),
                    },
                )
            )
    require(len(entries) == TOTAL_STEPS, "V8 total schedule length differs")
    return entries


def schedule_audit(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(entries) == TOTAL_STEPS, "V8 schedule audit length differs")
    value_entries = entries[:VALUE14_STEPS]
    balanced_entries = entries[VALUE14_STEPS:]
    value_passes = []
    for pass_index in range(VALUE14_PASSES):
        start = pass_index * len(VALUE14_ORDINALS)
        pass_entries = value_entries[start : start + len(VALUE14_ORDINALS)]
        ordinals = [int(entry["train_row_ordinal"]) for entry in pass_entries]
        require(set(ordinals) == set(VALUE14_ORDINALS), "V8 Value14 pass differs")
        require(len(ordinals) == len(set(ordinals)), "V8 Value14 pass duplicates a row")
        require(
            all(entry["phase"] == "value14" for entry in pass_entries),
            "V8 Value14 phase label differs",
        )
        value_passes.append(
            {
                "pass_index": pass_index,
                "ordinals": ordinals,
                "ordered_ordinals_sha256": canonical_sha256(ordinals),
            }
        )
    balanced_counts = Counter(str(entry["target_stratum"]) for entry in balanced_entries)
    require(
        {stratum: balanced_counts[stratum] for stratum in TARGET_STRATA}
        == BALANCED_QUOTAS,
        "V8 balanced quotas differ",
    )
    require(
        all(entry["phase"] == "balanced" for entry in balanced_entries),
        "V8 balanced phase label differs",
    )
    return {
        "total_steps": len(entries),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "ordered_train_row_ordinals_sha256": canonical_sha256(
            [entry["train_row_ordinal"] for entry in entries]
        ),
        "entries_sha256": canonical_sha256(list(entries)),
        "phases": {
            "value14": {
                "steps": len(value_entries),
                "passes": VALUE14_PASSES,
                "value14_ordinals": list(VALUE14_ORDINALS),
                "target_strata": [
                    "same_cardinality_value",
                    "cross_cardinality_value",
                ],
                "pass_audit": value_passes,
            },
            "balanced": {
                "steps": len(balanced_entries),
                "rounds": BALANCED_STEPS_PER_STRATUM,
                "quotas": dict(BALANCED_QUOTAS),
                "observed_counts": {
                    stratum: balanced_counts[stratum] for stratum in TARGET_STRATA
                },
            },
        },
    }


def build_schedule_manifest(
    *,
    schedule_path: Path,
    entries: Sequence[Mapping[str, Any]],
    parent_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    audit = schedule_audit(entries)
    return with_self_hash(
        {
            "schema": SCHEDULE_MANIFEST_SCHEMA,
            "task": TASK,
            "schedule_schema": SCHEDULE_SCHEMA,
            "indexing": "zero_based_original_v7_train32_ordinal",
            "sampling_contract": {
                "sampler": "explicit_ordered_train_row_ordinal_v1",
                "replacement": "phase_specific_and_fully_materialized",
                "random_number_generator": "none_sha256_sort_keys_only",
                "value_shuffle_namespace": VALUE_SHUFFLE_NAMESPACE,
                "balanced_draw_namespace": BALANCED_DRAW_NAMESPACE,
                "balanced_interleave_namespace": BALANCED_INTERLEAVE_NAMESPACE,
            },
            "schedule": {
                "path": str(schedule_path.resolve()),
                "sha256": sha256_file(schedule_path),
                "rows": len(entries),
                "entries_sha256": audit["entries_sha256"],
                "ordered_train_row_ordinals_sha256": audit[
                    "ordered_train_row_ordinals_sha256"
                ],
            },
            "parent_v7": {key: dict(value) for key, value in parent_bindings.items()},
            "curriculum": audit,
            "split_contract": {
                "source_split": "train",
                "parent_train_rows": 32,
                "scheduled_validation_rows": 0,
                "scheduled_test_rows": 0,
                "scheduled_hard32_rows": 0,
                "hard32_role": "protected_evaluation_only",
            },
        }
    )


def curriculum_binding(
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    curriculum = schedule_manifest["curriculum"]
    return {
        "schema": CURRICULUM_BINDING_SCHEMA,
        "schedule": {
            "path": str(schedule_path.resolve()),
            "sha256": sha256_file(schedule_path),
            "rows": TOTAL_STEPS,
            "entries_sha256": curriculum["entries_sha256"],
        },
        "schedule_manifest": {
            "path": str(schedule_manifest_path.resolve()),
            "sha256": sha256_file(schedule_manifest_path),
            "manifest_sha256": schedule_manifest["manifest_sha256"],
        },
        "parent_train32_sha256": PARENT_TRAIN32_SHA256,
        "value14_ordinals": list(VALUE14_ORDINALS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "total_steps": TOTAL_STEPS,
    }


def build_source_manifest(
    *,
    v7_root: Path,
    parent_source_manifest: Mapping[str, Any],
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    parent_train = parent_source_manifest["partitions"]["train"]
    require(parent_train["data"]["sha256"] == PARENT_TRAIN32_SHA256, "V7 train binding differs")
    return with_self_hash(
        {
            "schema": SOURCE_SCHEMA,
            "task": TASK,
            "purpose": "v8_value_first_then_balanced_generation_aligned_training",
            "contract": {
                "source_split": "train",
                "val_rows": 0,
                "test_rows": 0,
                "hard32_rows": 0,
                "episode_contract": dict(
                    parent_source_manifest["contract"]["episode_contract"]
                ),
            },
            "partitions": {
                "train": {
                    "source_split": "train",
                    "rows": 32,
                    "data": {
                        "path": str((v7_root / "train32.jsonl").resolve()),
                        "sha256": PARENT_TRAIN32_SHA256,
                    },
                    "row_manifest": {
                        "path": str((v7_root / "train32_rows.jsonl").resolve()),
                        "sha256": PARENT_ROWS_SHA256,
                    },
                }
            },
            "v7_pairing": dict(parent_source_manifest["v7_pairing"]),
            "v8_curriculum": curriculum_binding(
                schedule_path,
                schedule_manifest_path,
                schedule_manifest,
            ),
            "parent_v7_source_manifest": {
                "path": str((v7_root / "train32_source_manifest.json").resolve()),
                "sha256": PARENT_SOURCE_MANIFEST_SHA256,
                "manifest_sha256": parent_source_manifest["manifest_sha256"],
            },
            "parent_train32_sha256": PARENT_TRAIN32_SHA256,
        }
    )


def build_bundle_manifest(
    *,
    output_dir: Path,
    parent_bindings: Mapping[str, Mapping[str, Any]],
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {}
    for name, filename in (
        ("schedule", "schedule.jsonl"),
        ("schedule_manifest", "schedule_manifest.json"),
        ("source_manifest", "source_manifest.json"),
    ):
        path = output_dir / filename
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return with_self_hash(
        {
            "schema": BUNDLE_SCHEMA,
            "task": TASK,
            "parent_v7": {key: dict(value) for key, value in parent_bindings.items()},
            "artifacts": artifacts,
            "curriculum": dict(schedule_manifest["curriculum"]),
            "leakage": {
                "source_split": "train",
                "val_rows_in_schedule": 0,
                "test_rows_in_schedule": 0,
                "hard32_rows_in_schedule": 0,
                "train32_hard32_shared_normalized_paragraphs": 0,
                "proof": "validated_parent_v7_contract_plus_ordinal_only_schedule_v1",
            },
        }
    )


def build_source_lock(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    bundle = load_json_object(output_dir / "manifest.json", description="V8 bundle manifest")
    validate_self_hash(bundle)
    artifacts = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        path = output_dir / filename
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return with_self_hash(
        {
            "schema": SOURCE_LOCK_SCHEMA,
            "parent_train32_sha256": PARENT_TRAIN32_SHA256,
            "parent_v7": dict(bundle["parent_v7"]),
            "fixed_hard32": {
                "path": str(HARD32_FILE),
                "sha256": HARD32_FILE_SHA256,
                "role": "protected_evaluation_only_not_scheduled",
            },
            "curriculum": {
                "value14_ordinals": list(VALUE14_ORDINALS),
                "value14_passes": VALUE14_PASSES,
                "value14_steps": VALUE14_STEPS,
                "balanced_quotas": dict(BALANCED_QUOTAS),
                "balanced_steps": BALANCED_STEPS,
                "total_steps": TOTAL_STEPS,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
            },
            "artifacts": artifacts,
        },
        field="lock_sha256",
    )


def _ensure_fresh_output(output_dir: Path, *, overwrite: bool) -> None:
    paths = [output_dir / filename for filename in ARTIFACT_FILENAMES.values()]
    existing = [path for path in paths if path.exists()]
    require(overwrite or not existing, "V8 output already exists: " + ", ".join(map(str, existing)))
    if overwrite:
        for path in existing:
            require(path.is_file() and not path.is_symlink(), f"refusing to replace output: {path}")
            path.unlink()


def prepare_v8_data(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    v7_root: Path = V7_ROOT,
    source_lock_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    v7_root = v7_root.expanduser().resolve()
    _ensure_fresh_output(output_dir, overwrite=overwrite)
    rows, pair_manifest, parent_source, parent_bindings = load_parent_contract(v7_root)
    entries = build_schedule(rows, pair_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "schedule.jsonl"
    schedule_manifest_path = output_dir / "schedule_manifest.json"
    source_manifest_path = output_dir / "source_manifest.json"
    write_jsonl(schedule_path, entries)
    schedule_manifest = build_schedule_manifest(
        schedule_path=schedule_path,
        entries=entries,
        parent_bindings=parent_bindings,
    )
    write_json(schedule_manifest_path, schedule_manifest)
    source_manifest = build_source_manifest(
        v7_root=v7_root,
        parent_source_manifest=parent_source,
        schedule_path=schedule_path,
        schedule_manifest_path=schedule_manifest_path,
        schedule_manifest=schedule_manifest,
    )
    write_json(source_manifest_path, source_manifest)
    bundle = build_bundle_manifest(
        output_dir=output_dir,
        parent_bindings=parent_bindings,
        schedule_manifest=schedule_manifest,
    )
    write_json(output_dir / "manifest.json", bundle)
    if source_lock_output is not None:
        write_json(source_lock_output.expanduser().resolve(), build_source_lock(output_dir))
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v7-root", type=Path, default=V7_ROOT)
    parser.add_argument("--source-lock-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_v8_data(
            output_dir=args.output_dir,
            v7_root=args.v7_root,
            source_lock_output=args.source_lock_output,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "pass",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "total_steps": manifest["curriculum"]["total_steps"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
