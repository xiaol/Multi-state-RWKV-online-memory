#!/usr/bin/env python3
"""Build the hash-locked V9 pair-level curriculum over frozen Train32."""

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


PAIR_SCHEDULE_SCHEMA = "rwkv_ms_scene_memory_v9_pair_schedule.v1"
PAIR_SCHEDULE_ENTRY_SCHEMA = "rwkv_ms_scene_memory_v9_pair_schedule_entry.v1"
PAIR_SCHEDULE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v9_pair_schedule_manifest.v1"
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v9_source.v1"
CURRICULUM_BINDING_SCHEMA = "rwkv_ms_scene_memory_v9_pair_curriculum_binding.v1"
BUNDLE_SCHEMA = "rwkv_ms_scene_memory_v9_bundle.v1"
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v9_source_lock.v1"

TRAIN32_SHA256 = "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9"
TRAIN32_ROWS_SHA256 = (
    "af80b1938196319e6595a0e6d0e2f2c9a6009963a82fea520d608e060a4fe957"
)
PAIR_MANIFEST_FILE_SHA256 = (
    "13555da56823d9597bf061d51ec6575db25cde49c044cab378f2050373fd78b6"
)
PAIR_MANIFEST_SHA256 = (
    "a5d04a2f6cecb1f87681cd39a8e558bac6322df502e5749f907d9fdd6cd1b3c4"
)
PAIR_ENTRIES_SHA256 = (
    "6234a72df756c7fe93dd1b2ebd0caddb58b00445cffc2d5c03f76d31878cbc99"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "57626d3629e055a5f7900ed2a8526d357890d63730aa55d56ee72bd56a05017b"
)
SOURCE_MANIFEST_SHA256 = (
    "56e3d031895289df4414034d1abb562738b097206b78384bf045623856942911"
)

CANONICAL_VALUE14_PAIRS = (
    (1, 14),
    (22, 26),
    (3, 24),
    (5, 9),
    (10, 23),
    (19, 28),
    (20, 31),
)
VALUE14_ORDINALS = tuple(sorted(ordinal for pair in CANONICAL_VALUE14_PAIRS for ordinal in pair))
PAIR_PASSES = 4
PAIRS_PER_PASS = len(CANONICAL_VALUE14_PAIRS)
TOTAL_PAIR_STEPS = PAIR_PASSES * PAIRS_PER_PASS
DIRECTED_PRESENTATIONS = TOTAL_PAIR_STEPS * 2
CHECKPOINT_STEPS = (7, 14, 21, 28)
PAIR_SHUFFLE_NAMESPACE = "rwkv_ms_scene_memory_v9_value14_pair_pass_shuffle.v1"

DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v9/value14_pair28_v1"
)
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v9_source_lock.json")
ARTIFACT_FILENAMES = {
    "bundle_manifest": "manifest.json",
    "pair_schedule": "pair_schedule.jsonl",
    "pair_schedule_manifest": "pair_schedule_manifest.json",
    "source_manifest": "source_manifest.json",
}
INPUT_FILENAMES = {
    "train32": ("train32.jsonl", TRAIN32_SHA256),
    "train32_rows": ("train32_rows.jsonl", TRAIN32_ROWS_SHA256),
    "pair_manifest": ("train32_pair_manifest.json", PAIR_MANIFEST_FILE_SHA256),
    "source_manifest": (
        "train32_source_manifest.json",
        SOURCE_MANIFEST_FILE_SHA256,
    ),
}


def hard32_exclusion() -> dict[str, Any]:
    """Return a policy marker that deliberately carries no Hard32 locator."""
    return {
        "name": "Hard32",
        "included": False,
        "path": None,
        "sha256": None,
        "policy": "forbidden_not_resolved_opened_or_hashed",
    }


def _verify_file(path: Path, expected_sha256: str, description: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    require(sha256_file(path) == expected_sha256, f"{description} SHA-256 differs")


def _input_bindings(v7_root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for name, (filename, expected_sha256) in INPUT_FILENAMES.items():
        path = v7_root / filename
        _verify_file(path, expected_sha256, f"V9 input {name}")
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": expected_sha256,
            "bytes": path.stat().st_size,
        }
    return bindings


def _directed_entries_by_ordinal(
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    entries = pair_manifest.get("directed_pairs")
    require(isinstance(entries, list) and len(entries) == 32, "V9 pair entries differ")
    require(
        pair_manifest.get("entries_sha256") == PAIR_ENTRIES_SHA256,
        "V9 pair entries hash differs",
    )
    require(canonical_sha256(entries) == PAIR_ENTRIES_SHA256, "V9 pair entries drifted")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "V9 directed pair entry is invalid")
        validate_self_hash(entry, field="entry_sha256")
        ordinal = entry.get("train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < 32
            and ordinal not in by_ordinal,
            "V9 directed pair ordinal is invalid or duplicated",
        )
        by_ordinal[ordinal] = entry
    require(set(by_ordinal) == set(range(32)), "V9 pair entries do not cover Train32")
    return by_ordinal


def _validate_canonical_value_pairs(
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    require(len(rows) == 32, "V9 requires the frozen 32 Train32 rows")
    by_ordinal = _directed_entries_by_ordinal(pair_manifest)
    observed_value_ordinals = tuple(
        ordinal
        for ordinal in range(32)
        if by_ordinal[ordinal].get("target_stratum")
        in {"same_cardinality_value", "cross_cardinality_value"}
    )
    require(observed_value_ordinals == VALUE14_ORDINALS, "V9 locked Value14 ordinals differ")
    require(
        len(set(VALUE14_ORDINALS)) == 14
        and all(low < high for low, high in CANONICAL_VALUE14_PAIRS),
        "V9 canonical undirected-pair definition is invalid",
    )
    for low, high in CANONICAL_VALUE14_PAIRS:
        low_entry = by_ordinal[low]
        high_entry = by_ordinal[high]
        require(
            low_entry.get("donor_train_row_ordinal") == high
            and high_entry.get("donor_train_row_ordinal") == low,
            f"V9 canonical pair is not reciprocal: {(low, high)}",
        )
        require(
            low_entry.get("target_stratum") == high_entry.get("target_stratum")
            and low_entry.get("target_stratum")
            in {"same_cardinality_value", "cross_cardinality_value"},
            f"V9 canonical pair stratum differs: {(low, high)}",
        )
        for ordinal, donor, entry in (
            (low, high, low_entry),
            (high, low, high_entry),
        ):
            require(
                entry.get("source_row_sha256") == rows[ordinal].get("row_sha256")
                and entry.get("donor_row_sha256") == rows[donor].get("row_sha256"),
                f"V9 canonical pair row binding differs: {(low, high)}",
            )
    return by_ordinal


def load_input_contract(
    v7_root: Path = V7_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load only the four allowed Train32/source/pair inputs."""
    v7_root = v7_root.expanduser().resolve()
    bindings = _input_bindings(v7_root)
    data_records = read_jsonl(v7_root / "train32.jsonl")
    row_records = read_jsonl(v7_root / "train32_rows.jsonl")
    require(len(data_records) == len(row_records) == 32, "V9 Train32 must contain 32 rows")
    rows = [payload for _, payload in row_records]
    for ordinal, ((raw_line, _), row) in enumerate(zip(data_records, rows)):
        validate_self_hash(row, field="record_sha256")
        require(row.get("train_row_ordinal") == ordinal, "V9 Train32 row ordinal differs")
        require(row.get("source_split") == "train", "V9 input row is not official train")
        require(
            row.get("row_sha256") == hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            "V9 Train32 row data binding differs",
        )

    pair_manifest = load_json_object(
        v7_root / "train32_pair_manifest.json",
        description="V9 input pair manifest",
    )
    validate_self_hash(pair_manifest)
    require(
        pair_manifest.get("schema") == "rwkv_ms_scene_memory_v7_pairing.v1"
        and pair_manifest.get("manifest_sha256") == PAIR_MANIFEST_SHA256,
        "V9 input pair manifest identity differs",
    )
    dataset = pair_manifest.get("dataset")
    require(
        isinstance(dataset, dict)
        and dataset.get("sha256") == TRAIN32_SHA256
        and dataset.get("rows") == 32,
        "V9 pair manifest Train32 binding differs",
    )
    _validate_canonical_value_pairs(rows, pair_manifest)

    source_manifest = load_json_object(
        v7_root / "train32_source_manifest.json",
        description="V9 input source manifest",
    )
    validate_self_hash(source_manifest)
    require(
        source_manifest.get("schema") == "rwkv_ms_scene_memory_v7_source.v1"
        and source_manifest.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
        "V9 input source manifest identity differs",
    )
    contract = source_manifest.get("contract")
    require(
        isinstance(contract, dict)
        and contract.get("source_split") == "train"
        and contract.get("val_rows") == 0
        and contract.get("test_rows") == 0,
        "V9 source manifest is not Train32-only",
    )
    train_partition = source_manifest.get("partitions", {}).get("train")
    require(
        isinstance(train_partition, dict)
        and train_partition.get("rows") == 32
        and train_partition.get("source_split") == "train"
        and train_partition.get("data", {}).get("sha256") == TRAIN32_SHA256
        and train_partition.get("row_manifest", {}).get("sha256")
        == TRAIN32_ROWS_SHA256,
        "V9 source manifest Train32 artifacts differ",
    )
    pairing = source_manifest.get("v7_pairing")
    require(
        isinstance(pairing, dict)
        and pairing.get("dataset_sha256") == TRAIN32_SHA256
        and pairing.get("entries_sha256") == PAIR_ENTRIES_SHA256
        and pairing.get("pair_manifest", {}).get("sha256")
        == PAIR_MANIFEST_FILE_SHA256
        and pairing.get("pair_manifest", {}).get("manifest_sha256")
        == PAIR_MANIFEST_SHA256,
        "V9 source manifest pair binding differs",
    )
    return rows, pair_manifest, source_manifest, bindings


def _pair_identity_sha256(
    pair: tuple[int, int],
    rows: Sequence[Mapping[str, Any]],
    directed: Mapping[int, Mapping[str, Any]],
) -> str:
    low, high = pair
    return canonical_sha256(
        {
            "canonical_pair_ordinals": [low, high],
            "row_sha256": [rows[low]["row_sha256"], rows[high]["row_sha256"]],
            "directed_pair_entry_sha256": [
                directed[low]["entry_sha256"],
                directed[high]["entry_sha256"],
            ],
        }
    )


def _shuffle_key(
    *,
    pass_index: int,
    pair: tuple[int, int],
    pair_identity_sha256: str,
) -> str:
    material = "\0".join(
        (
            PAIR_SHUFFLE_NAMESPACE,
            str(pass_index),
            str(pair[0]),
            str(pair[1]),
            pair_identity_sha256,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _member_binding(
    *,
    ordinal: int,
    donor_ordinal: int,
    member_role: str,
    rows: Sequence[Mapping[str, Any]],
    directed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    row = rows[ordinal]
    pair_entry = directed[ordinal]
    return {
        "member_role": member_role,
        "train_row_ordinal": ordinal,
        "official_source_index": row["official_source_index"],
        "row_sha256": row["row_sha256"],
        "row_record_sha256": row["record_sha256"],
        "directed_pair_entry_sha256": pair_entry["entry_sha256"],
        "donor_train_row_ordinal": donor_ordinal,
        "donor_row_sha256": pair_entry["donor_row_sha256"],
    }


def build_pair_schedule(
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directed = _validate_canonical_value_pairs(rows, pair_manifest)
    pair_identities = {
        pair: _pair_identity_sha256(pair, rows, directed)
        for pair in CANONICAL_VALUE14_PAIRS
    }
    entries: list[dict[str, Any]] = []
    for pass_index in range(PAIR_PASSES):
        ordered_pairs = sorted(
            CANONICAL_VALUE14_PAIRS,
            key=lambda pair: (
                _shuffle_key(
                    pass_index=pass_index,
                    pair=pair,
                    pair_identity_sha256=pair_identities[pair],
                ),
                pair,
            ),
        )
        for pass_position, pair in enumerate(ordered_pairs):
            low, high = pair
            target_stratum = directed[low]["target_stratum"]
            pair_identity = pair_identities[pair]
            entry = {
                "schema": PAIR_SCHEDULE_ENTRY_SCHEMA,
                "schedule_index": len(entries),
                "step": len(entries) + 1,
                "phase": "value14_pair",
                "pass_index": pass_index,
                "pass_position": pass_position,
                "pair_batch_size": 2,
                "canonical_pair_ordinals": [low, high],
                "canonical_pair_sha256": pair_identity,
                "target_stratum": target_stratum,
                "members": [
                    _member_binding(
                        ordinal=low,
                        donor_ordinal=high,
                        member_role="canonical_low",
                        rows=rows,
                        directed=directed,
                    ),
                    _member_binding(
                        ordinal=high,
                        donor_ordinal=low,
                        member_role="canonical_high",
                        rows=rows,
                        directed=directed,
                    ),
                ],
                "sampling": {
                    "mode": "deterministic_hash_sorted_undirected_pair_pass",
                    "namespace": PAIR_SHUFFLE_NAMESPACE,
                    "shuffle_key_sha256": _shuffle_key(
                        pass_index=pass_index,
                        pair=pair,
                        pair_identity_sha256=pair_identity,
                    ),
                },
            }
            entries.append(with_self_hash(entry, field="entry_sha256"))
    require(len(entries) == TOTAL_PAIR_STEPS, "V9 pair schedule length differs")
    return entries


def schedule_audit(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(entries) == TOTAL_PAIR_STEPS, "V9 pair schedule audit length differs")
    pass_audit = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    ordinal_counts: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()
    for pass_index in range(PAIR_PASSES):
        start = pass_index * PAIRS_PER_PASS
        pass_entries = entries[start : start + PAIRS_PER_PASS]
        pairs = [tuple(entry["canonical_pair_ordinals"]) for entry in pass_entries]
        require(
            len(pairs) == len(set(pairs)) == PAIRS_PER_PASS
            and set(pairs) == set(CANONICAL_VALUE14_PAIRS),
            f"V9 pair pass differs: {pass_index}",
        )
        require(
            all(
                entry.get("pass_index") == pass_index
                and entry.get("pass_position") == position
                and entry.get("pair_batch_size") == 2
                for position, entry in enumerate(pass_entries)
            ),
            f"V9 pair pass indexing differs: {pass_index}",
        )
        pair_counts.update(pairs)
        for pair in pairs:
            ordinal_counts.update(pair)
        target_counts.update(str(entry["target_stratum"]) for entry in pass_entries)
        pass_audit.append(
            {
                "pass_index": pass_index,
                "canonical_pair_ordinals": [list(pair) for pair in pairs],
                "ordered_pairs_sha256": canonical_sha256([list(pair) for pair in pairs]),
            }
        )
    require(
        pair_counts == Counter({pair: PAIR_PASSES for pair in CANONICAL_VALUE14_PAIRS}),
        "V9 pair exposure counts differ",
    )
    require(
        ordinal_counts == Counter({ordinal: PAIR_PASSES for ordinal in VALUE14_ORDINALS}),
        "V9 directed row exposure counts differ",
    )
    require(
        target_counts
        == Counter(
            {
                "same_cardinality_value": 5 * PAIR_PASSES,
                "cross_cardinality_value": 2 * PAIR_PASSES,
            }
        ),
        "V9 pair stratum counts differ",
    )
    ordered_pairs = [entry["canonical_pair_ordinals"] for entry in entries]
    return {
        "pair_steps": len(entries),
        "directed_presentations": DIRECTED_PRESENTATIONS,
        "pair_batch_size": 2,
        "pair_passes": PAIR_PASSES,
        "pairs_per_pass": PAIRS_PER_PASS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "canonical_value14_pairs": [list(pair) for pair in CANONICAL_VALUE14_PAIRS],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "ordered_pairs_sha256": canonical_sha256(ordered_pairs),
        "entries_sha256": canonical_sha256(list(entries)),
        "pair_exposure_counts": {
            f"{low}:{high}": pair_counts[(low, high)]
            for low, high in CANONICAL_VALUE14_PAIRS
        },
        "target_stratum_pair_steps": dict(sorted(target_counts.items())),
        "pass_audit": pass_audit,
    }


def build_pair_schedule_manifest(
    *,
    schedule_path: Path,
    entries: Sequence[Mapping[str, Any]],
    input_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    audit = schedule_audit(entries)
    return with_self_hash(
        {
            "schema": PAIR_SCHEDULE_MANIFEST_SCHEMA,
            "task": TASK,
            "schedule_schema": PAIR_SCHEDULE_SCHEMA,
            "step_unit": "one_canonical_undirected_pair_with_two_reciprocal_directions",
            "sampling_contract": {
                "sampler": "explicit_ordered_canonical_pair_v1",
                "replacement": "four_complete_passes",
                "random_number_generator": "none_sha256_sort_keys_only",
                "namespace": PAIR_SHUFFLE_NAMESPACE,
            },
            "schedule": {
                "path": str(schedule_path.resolve()),
                "sha256": sha256_file(schedule_path),
                "rows": len(entries),
                "entries_sha256": audit["entries_sha256"],
                "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
            },
            "inputs": {key: dict(value) for key, value in input_bindings.items()},
            "curriculum": audit,
            "split_contract": {
                "source_split": "train",
                "train32_rows": 32,
                "scheduled_train_ordinals": list(VALUE14_ORDINALS),
                "scheduled_validation_rows": 0,
                "scheduled_test_rows": 0,
                "scheduled_hard32_rows": 0,
            },
            "excluded_artifacts": {"hard32": hard32_exclusion()},
        }
    )


def curriculum_binding(
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit = schedule_manifest["curriculum"]
    return {
        "schema": CURRICULUM_BINDING_SCHEMA,
        "pair_schedule": {
            "path": str(schedule_path.resolve()),
            "sha256": sha256_file(schedule_path),
            "rows": TOTAL_PAIR_STEPS,
            "entries_sha256": audit["entries_sha256"],
            "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
        },
        "pair_schedule_manifest": {
            "path": str(schedule_manifest_path.resolve()),
            "sha256": sha256_file(schedule_manifest_path),
            "manifest_sha256": schedule_manifest["manifest_sha256"],
        },
        "train32_sha256": TRAIN32_SHA256,
        "pair_manifest_file_sha256": PAIR_MANIFEST_FILE_SHA256,
        "canonical_value14_pairs": [list(pair) for pair in CANONICAL_VALUE14_PAIRS],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "pair_steps": TOTAL_PAIR_STEPS,
    }


def build_source_manifest(
    *,
    input_source_manifest: Mapping[str, Any],
    input_bindings: Mapping[str, Mapping[str, Any]],
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return with_self_hash(
        {
            "schema": SOURCE_SCHEMA,
            "task": TASK,
            "purpose": "v9_pair_level_value14_training",
            "contract": {
                "source_split": "train",
                "train_rows": 32,
                "scheduled_train_rows": 14,
                "val_rows": 0,
                "test_rows": 0,
                "hard32_rows": 0,
                "episode_contract": dict(
                    input_source_manifest["contract"]["episode_contract"]
                ),
            },
            "inputs": {key: dict(value) for key, value in input_bindings.items()},
            "partitions": {
                "train": {
                    "source_split": "train",
                    "rows": 32,
                    "data": dict(input_source_manifest["partitions"]["train"]["data"]),
                    "row_manifest": dict(
                        input_source_manifest["partitions"]["train"]["row_manifest"]
                    ),
                }
            },
            "v7_pairing": dict(input_source_manifest["v7_pairing"]),
            "v9_pair_curriculum": curriculum_binding(
                schedule_path,
                schedule_manifest_path,
                schedule_manifest,
            ),
            "excluded_artifacts": {"hard32": hard32_exclusion()},
        }
    )


def build_bundle_manifest(
    *,
    output_dir: Path,
    input_bindings: Mapping[str, Mapping[str, Any]],
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {}
    for name, filename in (
        ("pair_schedule", "pair_schedule.jsonl"),
        ("pair_schedule_manifest", "pair_schedule_manifest.json"),
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
            "inputs": {key: dict(value) for key, value in input_bindings.items()},
            "artifacts": artifacts,
            "curriculum": dict(schedule_manifest["curriculum"]),
            "leakage": {
                "source_split": "train",
                "val_rows_in_schedule": 0,
                "test_rows_in_schedule": 0,
                "hard32_rows_in_schedule": 0,
                "proof": "hash_locked_train32_pair_ordinals_only_v1",
            },
            "excluded_artifacts": {"hard32": hard32_exclusion()},
        }
    )


def build_source_lock(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    bundle = load_json_object(output_dir / "manifest.json", description="V9 bundle manifest")
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
            "inputs": dict(bundle["inputs"]),
            "curriculum": {
                "canonical_value14_pairs": [
                    list(pair) for pair in CANONICAL_VALUE14_PAIRS
                ],
                "value14_ordinals": list(VALUE14_ORDINALS),
                "pair_passes": PAIR_PASSES,
                "pairs_per_pass": PAIRS_PER_PASS,
                "pair_steps": TOTAL_PAIR_STEPS,
                "directed_presentations": DIRECTED_PRESENTATIONS,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
            },
            "excluded_artifacts": {"hard32": hard32_exclusion()},
            "artifacts": artifacts,
        },
        field="lock_sha256",
    )


def _ensure_fresh_output(output_dir: Path, *, overwrite: bool) -> None:
    paths = [output_dir / filename for filename in ARTIFACT_FILENAMES.values()]
    existing = [path for path in paths if path.exists()]
    require(
        overwrite or not existing,
        "V9 output already exists: " + ", ".join(map(str, existing)),
    )
    if overwrite:
        for path in existing:
            require(path.is_file() and not path.is_symlink(), f"refusing to replace: {path}")
            path.unlink()


def prepare_v9_data(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    v7_root: Path = V7_ROOT,
    source_lock_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    v7_root = v7_root.expanduser().resolve()
    _ensure_fresh_output(output_dir, overwrite=overwrite)
    rows, pair_manifest, input_source, input_bindings = load_input_contract(v7_root)
    entries = build_pair_schedule(rows, pair_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "pair_schedule.jsonl"
    schedule_manifest_path = output_dir / "pair_schedule_manifest.json"
    source_manifest_path = output_dir / "source_manifest.json"
    write_jsonl(schedule_path, entries)
    schedule_manifest = build_pair_schedule_manifest(
        schedule_path=schedule_path,
        entries=entries,
        input_bindings=input_bindings,
    )
    write_json(schedule_manifest_path, schedule_manifest)
    source_manifest = build_source_manifest(
        input_source_manifest=input_source,
        input_bindings=input_bindings,
        schedule_path=schedule_path,
        schedule_manifest_path=schedule_manifest_path,
        schedule_manifest=schedule_manifest,
    )
    write_json(source_manifest_path, source_manifest)
    bundle = build_bundle_manifest(
        output_dir=output_dir,
        input_bindings=input_bindings,
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
        manifest = prepare_v9_data(
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
                "pair_steps": manifest["curriculum"]["pair_steps"],
                "checkpoint_steps": manifest["curriculum"]["checkpoint_steps"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
