#!/usr/bin/env python3
"""Build the leakage-closed V15 all-Train32 symmetric-pair schedule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    ContractError,
    DEFAULT_OUTPUT_DIR as V7_ROOT,
    FAILURE_STRATA,
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


PAIR_SCHEDULE_SCHEMA = "rwkv_ms_scene_memory_v15_pair_schedule.v1"
PAIR_SCHEDULE_ENTRY_SCHEMA = "rwkv_ms_scene_memory_v15_pair_schedule_entry.v1"
PAIR_SCHEDULE_MANIFEST_SCHEMA = (
    "rwkv_ms_scene_memory_v15_pair_schedule_manifest.v1"
)
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v15_source.v1"
CURRICULUM_BINDING_SCHEMA = (
    "rwkv_ms_scene_memory_v15_pair_curriculum_binding.v1"
)
BUNDLE_SCHEMA = "rwkv_ms_scene_memory_v15_bundle.v1"
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v15_source_lock.v1"

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

# These are the sixteen reciprocal pairs already bound by the authoritative
# Train32 pair manifest. Every Train32 ordinal occurs exactly once.
CANONICAL_ALL32_PAIRS = (
    (0, 17),
    (1, 14),
    (2, 29),
    (3, 24),
    (4, 27),
    (5, 9),
    (6, 21),
    (7, 25),
    (8, 15),
    (10, 23),
    (11, 18),
    (12, 30),
    (13, 16),
    (19, 28),
    (20, 31),
    (22, 26),
)
ALL32_ORDINALS = tuple(range(32))
EMPTY_ORDINALS = (4, 6, 7, 11, 12, 15, 16, 17, 29)
GOLD_CARDINALITY_COUNTS = {0: 9, 1: 16, 2: 5, 3: 2}
PAIR_STRATUM_COUNTS = {
    "presence": 9,
    "same_cardinality_value": 5,
    "cross_cardinality_value": 2,
}
DIRECTED_STRATUM_COUNTS = {
    "presence": 18,
    "same_cardinality_value": 10,
    "cross_cardinality_value": 4,
}

PAIR_CYCLES = 4
PAIRS_PER_CYCLE = len(CANONICAL_ALL32_PAIRS)
TOTAL_PAIR_PRESENTATIONS = PAIR_CYCLES * PAIRS_PER_CYCLE
DIRECTED_PRESENTATIONS = TOTAL_PAIR_PRESENTATIONS * 2
CHECKPOINT_CYCLES = (1, 2, 3, 4)
PRESENTATION_CHECKPOINTS = (16, 32, 48, 64)
PAIR_SHUFFLE_NAMESPACE = "rwkv_ms_scene_memory_v15_all32_cycle_shuffle.v1"

DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v15/all32_pair64_v1"
)
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v15_source_lock.json")
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
    """Return an exclusion marker without carrying a Hard32 locator."""
    return {
        "name": "Hard32",
        "included": False,
        "path": None,
        "sha256": None,
        "policy": "forbidden_not_resolved_opened_or_hashed",
    }


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def allowed_input_paths(v7_root: Path = V7_ROOT) -> frozenset[Path]:
    """Return the only upstream files V15 may inspect."""
    root = _lexical_absolute(v7_root)
    return frozenset(root / filename for filename, _ in INPUT_FILENAMES.values())


def guard_v15_input_path(
    path: Path | str,
    *,
    v7_root: Path = V7_ROOT,
) -> Path:
    """Reject every input path outside the four pinned Train32 artifacts."""
    candidate = _lexical_absolute(path)
    require(
        candidate in allowed_input_paths(v7_root),
        f"V15 input is not an allowed pinned Train32 artifact: {candidate}",
    )
    return candidate


def _verify_file(
    path: Path,
    expected_sha256: str,
    description: str,
    *,
    v7_root: Path,
) -> Path:
    guarded = guard_v15_input_path(path, v7_root=v7_root)
    require(
        guarded.is_file() and not guarded.is_symlink(),
        f"missing {description}: {guarded}",
    )
    require(sha256_file(guarded) == expected_sha256, f"{description} SHA-256 differs")
    return guarded


def _input_bindings(v7_root: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for name, (filename, expected_sha256) in INPUT_FILENAMES.items():
        path = _verify_file(
            v7_root / filename,
            expected_sha256,
            f"V15 input {name}",
            v7_root=v7_root,
        )
        bindings[name] = {
            "path": str(path),
            "sha256": expected_sha256,
            "bytes": path.stat().st_size,
        }
    return bindings


def _directed_entries_by_ordinal(
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    require(
        pair_manifest.get("schema") == "rwkv_ms_scene_memory_v7_pairing.v1",
        "V15 pair-manifest schema differs",
    )
    require(
        pair_manifest.get("manifest_sha256") == PAIR_MANIFEST_SHA256,
        "V15 pair-manifest identity differs",
    )
    entries = pair_manifest.get("directed_pairs")
    require(isinstance(entries, list) and len(entries) == 32, "V15 pair entries differ")
    require(
        pair_manifest.get("entries_sha256") == PAIR_ENTRIES_SHA256
        and canonical_sha256(entries) == PAIR_ENTRIES_SHA256,
        "V15 pair entries hash differs",
    )
    require(
        pair_manifest.get("quotas") == DIRECTED_STRATUM_COUNTS,
        "V15 directed pair strata differ",
    )
    by_ordinal: dict[int, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "V15 directed pair entry is invalid")
        validate_self_hash(entry, field="entry_sha256")
        ordinal = entry.get("train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < 32
            and ordinal not in by_ordinal,
            "V15 directed pair ordinal is invalid or duplicated",
        )
        by_ordinal[ordinal] = entry
    require(set(by_ordinal) == set(ALL32_ORDINALS), "V15 pairs do not cover Train32")
    return by_ordinal


def _expected_pair_stratum(left_count: int, right_count: int) -> str:
    if left_count == 0 or right_count == 0:
        require(
            (left_count == 0) != (right_count == 0),
            "V15 cannot form a presence pair from two empty targets",
        )
        return "presence"
    if left_count == right_count:
        return "same_cardinality_value"
    return "cross_cardinality_value"


def _validate_all32_pairs(
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    require(len(rows) == 32, "V15 requires all 32 frozen Train32 rows")
    directed = _directed_entries_by_ordinal(pair_manifest)
    covered = tuple(sorted(ordinal for pair in CANONICAL_ALL32_PAIRS for ordinal in pair))
    require(
        covered == ALL32_ORDINALS
        and all(low < high for low, high in CANONICAL_ALL32_PAIRS),
        "V15 canonical pairs must partition Train32 exactly once",
    )

    observed_empty = tuple(
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("gold_boundary_count") == 0
    )
    require(observed_empty == EMPTY_ORDINALS, "V15 empty Train32 ordinals differ")
    cardinalities = Counter(int(row["gold_boundary_count"]) for row in rows)
    require(dict(sorted(cardinalities.items())) == GOLD_CARDINALITY_COUNTS, "V15 gold cardinalities differ")
    require(
        all(
            row.get("source_split") == "train"
            and row.get("strict_failure_stratum") in FAILURE_STRATA
            for row in rows
        ),
        "V15 requires official-train base-failure rows only",
    )
    require(
        len({int(row["official_source_index"]) for row in rows}) == 32,
        "V15 official train source indices are duplicated",
    )

    pair_strata: Counter[str] = Counter()
    paired_empty: list[int] = []
    for low, high in CANONICAL_ALL32_PAIRS:
        low_row = rows[low]
        high_row = rows[high]
        low_entry = directed[low]
        high_entry = directed[high]
        low_count = int(low_row["gold_boundary_count"])
        high_count = int(high_row["gold_boundary_count"])
        expected_stratum = _expected_pair_stratum(low_count, high_count)
        require(
            low_entry.get("donor_train_row_ordinal") == high
            and high_entry.get("donor_train_row_ordinal") == low,
            f"V15 canonical pair is not reciprocal: {(low, high)}",
        )
        require(
            low_entry.get("target_stratum")
            == high_entry.get("target_stratum")
            == expected_stratum,
            f"V15 canonical pair stratum differs: {(low, high)}",
        )
        require(
            low_entry.get("source_row_sha256") == low_row.get("row_sha256")
            and low_entry.get("donor_row_sha256") == high_row.get("row_sha256")
            and high_entry.get("source_row_sha256") == high_row.get("row_sha256")
            and high_entry.get("donor_row_sha256") == low_row.get("row_sha256"),
            f"V15 canonical pair row binding differs: {(low, high)}",
        )
        require(
            low_row.get("label_sha256") != high_row.get("label_sha256")
            and low_entry.get("source_label_sha256")
            != low_entry.get("donor_label_sha256"),
            f"V15 canonical pair targets are not distinct: {(low, high)}",
        )
        if expected_stratum == "presence":
            empties = [ordinal for ordinal in (low, high) if int(rows[ordinal]["gold_boundary_count"]) == 0]
            require(len(empties) == 1, f"V15 presence pair lacks one empty target: {(low, high)}")
            paired_empty.extend(empties)
        pair_strata[expected_stratum] += 1

    require(pair_strata == Counter(PAIR_STRATUM_COUNTS), "V15 pair stratum counts differ")
    require(
        tuple(sorted(paired_empty)) == EMPTY_ORDINALS,
        "V15 does not route every empty target through one presence pair",
    )
    return directed


def load_input_contract(
    v7_root: Path = V7_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load only the four hash-pinned official-train artifacts."""
    v7_root = _lexical_absolute(v7_root)
    bindings = _input_bindings(v7_root)
    train_path = guard_v15_input_path(v7_root / "train32.jsonl", v7_root=v7_root)
    rows_path = guard_v15_input_path(v7_root / "train32_rows.jsonl", v7_root=v7_root)
    data_records = read_jsonl(train_path)
    row_records = read_jsonl(rows_path)
    require(len(data_records) == len(row_records) == 32, "V15 Train32 must contain 32 rows")
    rows = [payload for _, payload in row_records]
    for ordinal, ((raw_line, _), row) in enumerate(zip(data_records, rows)):
        validate_self_hash(row, field="record_sha256")
        require(row.get("train_row_ordinal") == ordinal, "V15 Train32 row ordinal differs")
        require(row.get("source_split") == "train", "V15 input row is not official train")
        require(
            row.get("row_sha256") == hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
            "V15 Train32 row data binding differs",
        )

    pair_path = guard_v15_input_path(
        v7_root / "train32_pair_manifest.json",
        v7_root=v7_root,
    )
    pair_manifest = load_json_object(pair_path, description="V15 input pair manifest")
    validate_self_hash(pair_manifest)
    dataset = pair_manifest.get("dataset")
    require(
        isinstance(dataset, dict)
        and guard_v15_input_path(Path(str(dataset.get("path", ""))), v7_root=v7_root)
        == train_path
        and dataset.get("sha256") == TRAIN32_SHA256
        and dataset.get("rows") == 32,
        "V15 pair manifest Train32 binding differs",
    )
    _validate_all32_pairs(rows, pair_manifest)

    source_path = guard_v15_input_path(
        v7_root / "train32_source_manifest.json",
        v7_root=v7_root,
    )
    source_manifest = load_json_object(source_path, description="V15 input source manifest")
    validate_self_hash(source_manifest)
    require(
        source_manifest.get("schema") == "rwkv_ms_scene_memory_v7_source.v1"
        and source_manifest.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
        "V15 input source-manifest identity differs",
    )
    contract = source_manifest.get("contract")
    require(
        isinstance(contract, dict)
        and contract.get("source_split") == "train"
        and contract.get("val_rows") == 0
        and contract.get("test_rows") == 0,
        "V15 input source manifest is not train-only",
    )
    train_partition = source_manifest.get("partitions", {}).get("train")
    require(
        isinstance(train_partition, dict)
        and train_partition.get("source_split") == "train"
        and train_partition.get("rows") == 32
        and train_partition.get("data", {}).get("sha256") == TRAIN32_SHA256
        and train_partition.get("row_manifest", {}).get("sha256")
        == TRAIN32_ROWS_SHA256,
        "V15 source-manifest Train32 partition differs",
    )
    pairing = source_manifest.get("v7_pairing")
    require(
        isinstance(pairing, dict)
        and pairing.get("dataset_sha256") == TRAIN32_SHA256
        and pairing.get("entries_sha256") == PAIR_ENTRIES_SHA256
        and pairing.get("quotas") == DIRECTED_STRATUM_COUNTS
        and pairing.get("pair_manifest", {}).get("sha256")
        == PAIR_MANIFEST_FILE_SHA256
        and pairing.get("pair_manifest", {}).get("manifest_sha256")
        == PAIR_MANIFEST_SHA256,
        "V15 source-manifest pair binding differs",
    )
    return rows, pair_manifest, source_manifest, bindings


def _shuffle_key(*, cycle_index: int, pair: tuple[int, int]) -> str:
    material = "\0".join(
        (PAIR_SHUFFLE_NAMESPACE, str(cycle_index), str(pair[0]), str(pair[1]))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def ordered_pairs_for_cycle(cycle_index: int) -> tuple[tuple[int, int], ...]:
    require(
        isinstance(cycle_index, int)
        and not isinstance(cycle_index, bool)
        and 1 <= cycle_index <= PAIR_CYCLES,
        "V15 cycle index is outside the full schedule",
    )
    return tuple(
        sorted(
            CANONICAL_ALL32_PAIRS,
            key=lambda pair: (_shuffle_key(cycle_index=cycle_index, pair=pair), pair),
        )
    )


FULL_PAIR_CYCLES = tuple(
    ordered_pairs_for_cycle(cycle_index)
    for cycle_index in range(1, PAIR_CYCLES + 1)
)
ALL_CYCLE_PAIRS = tuple(pair for cycle in FULL_PAIR_CYCLES for pair in cycle)
PAIR_PREFIX_SHA256_BY_CHECKPOINT = {
    str(cycle_index): canonical_sha256(
        [
            list(pair)
            for pair in ALL_CYCLE_PAIRS[: cycle_index * PAIRS_PER_CYCLE]
        ]
    )
    for cycle_index in CHECKPOINT_CYCLES
}


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
            "label_sha256": [
                rows[low]["label_sha256"],
                rows[high]["label_sha256"],
            ],
            "directed_pair_entry_sha256": [
                directed[low]["entry_sha256"],
                directed[high]["entry_sha256"],
            ],
        }
    )


def _member_binding(
    *,
    ordinal: int,
    donor_ordinal: int,
    member_role: str,
    target_stratum: str,
    rows: Sequence[Mapping[str, Any]],
    directed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    row = rows[ordinal]
    pair_entry = directed[ordinal]
    boundary_count = int(row["gold_boundary_count"])
    return {
        "member_role": member_role,
        "target_presence_role": (
            "empty" if boundary_count == 0 else "nonempty"
        )
        if target_stratum == "presence"
        else "nonempty_value",
        "train_row_ordinal": ordinal,
        "official_source_index": row["official_source_index"],
        "source_split": row["source_split"],
        "row_sha256": row["row_sha256"],
        "row_record_sha256": row["record_sha256"],
        "label_sha256": row["label_sha256"],
        "gold_boundary_count": boundary_count,
        "strict_failure_stratum": row["strict_failure_stratum"],
        "base_record_sha256": row["base_record_sha256"],
        "base_failure_score_sha256": canonical_sha256(row["strict_score"]),
        "directed_pair_entry_sha256": pair_entry["entry_sha256"],
        "donor_train_row_ordinal": donor_ordinal,
        "donor_row_sha256": pair_entry["donor_row_sha256"],
        "donor_label_sha256": pair_entry["donor_label_sha256"],
    }


def build_pair_schedule(
    rows: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directed = _validate_all32_pairs(rows, pair_manifest)
    pair_identities = {
        pair: _pair_identity_sha256(pair, rows, directed)
        for pair in CANONICAL_ALL32_PAIRS
    }
    entries: list[dict[str, Any]] = []
    for cycle_index, ordered_pairs in enumerate(FULL_PAIR_CYCLES, start=1):
        for cycle_position, pair in enumerate(ordered_pairs):
            low, high = pair
            target_stratum = str(directed[low]["target_stratum"])
            empty_ordinals = [
                ordinal
                for ordinal in pair
                if int(rows[ordinal]["gold_boundary_count"]) == 0
            ]
            entry = {
                "schema": PAIR_SCHEDULE_ENTRY_SCHEMA,
                "schedule_index": len(entries),
                "presentation": len(entries) + 1,
                "phase": "all32_symmetric_pair",
                "cycle_index": cycle_index,
                "cycle_position": cycle_position,
                "pair_batch_size": 2,
                "canonical_pair_ordinals": [low, high],
                "canonical_pair_sha256": pair_identities[pair],
                "target_stratum": target_stratum,
                "empty_member_ordinals": empty_ordinals,
                "members": [
                    _member_binding(
                        ordinal=low,
                        donor_ordinal=high,
                        member_role="canonical_low",
                        target_stratum=target_stratum,
                        rows=rows,
                        directed=directed,
                    ),
                    _member_binding(
                        ordinal=high,
                        donor_ordinal=low,
                        member_role="canonical_high",
                        target_stratum=target_stratum,
                        rows=rows,
                        directed=directed,
                    ),
                ],
                "sampling": {
                    "mode": "deterministic_hash_sorted_full_pair_cycle",
                    "namespace": PAIR_SHUFFLE_NAMESPACE,
                    "shuffle_key_sha256": _shuffle_key(
                        cycle_index=cycle_index,
                        pair=pair,
                    ),
                },
            }
            entries.append(with_self_hash(entry, field="entry_sha256"))
    require(
        len(entries) == TOTAL_PAIR_PRESENTATIONS,
        "V15 pair schedule length differs",
    )
    return entries


def schedule_audit(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(
        len(entries) == TOTAL_PAIR_PRESENTATIONS,
        "V15 pair schedule audit length differs",
    )
    pair_counts: Counter[tuple[int, int]] = Counter()
    ordinal_counts: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()
    empty_counts: Counter[int] = Counter()
    cycle_audit: list[dict[str, Any]] = []
    flattened_pairs: list[list[int]] = []
    for cycle_index, expected_pairs in enumerate(FULL_PAIR_CYCLES, start=1):
        start = (cycle_index - 1) * PAIRS_PER_CYCLE
        cycle_entries = entries[start : start + PAIRS_PER_CYCLE]
        pairs = [tuple(entry["canonical_pair_ordinals"]) for entry in cycle_entries]
        require(
            tuple(pairs) == expected_pairs
            and len(pairs) == len(set(pairs)) == PAIRS_PER_CYCLE
            and set(pairs) == set(CANONICAL_ALL32_PAIRS),
            f"V15 full pair cycle differs: {cycle_index}",
        )
        require(
            all(
                entry.get("cycle_index") == cycle_index
                and entry.get("cycle_position") == position
                and entry.get("pair_batch_size") == 2
                for position, entry in enumerate(cycle_entries)
            ),
            f"V15 cycle indexing differs: {cycle_index}",
        )
        cycle_strata = Counter(str(entry["target_stratum"]) for entry in cycle_entries)
        require(cycle_strata == Counter(PAIR_STRATUM_COUNTS), f"V15 cycle strata differ: {cycle_index}")
        cycle_ordinals = sorted(ordinal for pair in pairs for ordinal in pair)
        require(tuple(cycle_ordinals) == ALL32_ORDINALS, f"V15 cycle row coverage differs: {cycle_index}")
        for entry, pair in zip(cycle_entries, pairs):
            empties = list(entry["empty_member_ordinals"])
            if entry["target_stratum"] == "presence":
                require(len(empties) == 1, "V15 presence schedule entry lacks one empty member")
                empty_counts.update(empties)
            else:
                require(not empties, "V15 value schedule entry contains an empty member")
            pair_counts[pair] += 1
            ordinal_counts.update(pair)
            target_counts[str(entry["target_stratum"])] += 1
            flattened_pairs.append(list(pair))
        prefix_sha256 = canonical_sha256(flattened_pairs)
        require(
            prefix_sha256 == PAIR_PREFIX_SHA256_BY_CHECKPOINT[str(cycle_index)],
            f"V15 checkpoint prefix differs: {cycle_index}",
        )
        cycle_audit.append(
            {
                "cycle_index": cycle_index,
                "presentation_start": start + 1,
                "presentation_end": start + PAIRS_PER_CYCLE,
                "canonical_pair_ordinals": [list(pair) for pair in pairs],
                "cycle_pairs_sha256": canonical_sha256([list(pair) for pair in pairs]),
                "checkpoint_prefix_sha256": prefix_sha256,
                "pair_stratum_counts": dict(sorted(cycle_strata.items())),
            }
        )

    require(
        pair_counts == Counter({pair: PAIR_CYCLES for pair in CANONICAL_ALL32_PAIRS}),
        "V15 pair exposure counts differ",
    )
    require(
        ordinal_counts == Counter({ordinal: PAIR_CYCLES for ordinal in ALL32_ORDINALS}),
        "V15 row exposure counts differ",
    )
    require(
        empty_counts == Counter({ordinal: PAIR_CYCLES for ordinal in EMPTY_ORDINALS}),
        "V15 empty-target exposure counts differ",
    )
    require(
        target_counts
        == Counter(
            {
                stratum: count * PAIR_CYCLES
                for stratum, count in PAIR_STRATUM_COUNTS.items()
            }
        ),
        "V15 schedule stratum counts differ",
    )
    ordered_pairs = [entry["canonical_pair_ordinals"] for entry in entries]
    return {
        "pair_presentations": len(entries),
        "directed_presentations": DIRECTED_PRESENTATIONS,
        "pair_batch_size": 2,
        "pair_cycles": PAIR_CYCLES,
        "pairs_per_cycle": PAIRS_PER_CYCLE,
        "checkpoint_cycles": list(CHECKPOINT_CYCLES),
        "presentation_checkpoints": list(PRESENTATION_CHECKPOINTS),
        "pair_prefix_sha256_by_checkpoint": dict(PAIR_PREFIX_SHA256_BY_CHECKPOINT),
        "canonical_all32_pairs": [list(pair) for pair in CANONICAL_ALL32_PAIRS],
        "all32_ordinals": list(ALL32_ORDINALS),
        "empty_ordinals": list(EMPTY_ORDINALS),
        "gold_cardinality_counts": {
            str(key): value for key, value in GOLD_CARDINALITY_COUNTS.items()
        },
        "pair_stratum_counts_per_cycle": dict(PAIR_STRATUM_COUNTS),
        "directed_stratum_counts_per_cycle": dict(DIRECTED_STRATUM_COUNTS),
        "scheduled_pair_stratum_presentations": dict(sorted(target_counts.items())),
        "empty_target_presentations": sum(empty_counts.values()),
        "nonempty_target_presentations": DIRECTED_PRESENTATIONS - sum(empty_counts.values()),
        "base_failure_target_presentations": DIRECTED_PRESENTATIONS,
        "ordered_pairs_sha256": canonical_sha256(ordered_pairs),
        "entries_sha256": canonical_sha256(list(entries)),
        "cycle_audit": cycle_audit,
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
                "sampler": "explicit_ordered_v15_full_pair_cycle_v1",
                "replacement": "four_complete_cycles",
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
                "scheduled_train_ordinals": list(ALL32_ORDINALS),
                "scheduled_validation_rows": 0,
                "scheduled_test_rows": 0,
                "scheduled_hard32_rows": 0,
            },
            "failure_contract": {
                "all_rows_are_frozen_base_failures": True,
                "base_failure_rows": 32,
                "empty_rows": 9,
                "empty_rows_are_presence_paired": True,
                "reciprocal_pair_rows": 32,
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
            "rows": TOTAL_PAIR_PRESENTATIONS,
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
        "canonical_all32_pairs": [list(pair) for pair in CANONICAL_ALL32_PAIRS],
        "all32_ordinals": list(ALL32_ORDINALS),
        "empty_ordinals": list(EMPTY_ORDINALS),
        "checkpoint_cycles": list(CHECKPOINT_CYCLES),
        "presentation_checkpoints": list(PRESENTATION_CHECKPOINTS),
        "pair_prefix_sha256_by_checkpoint": dict(PAIR_PREFIX_SHA256_BY_CHECKPOINT),
        "pair_presentations": TOTAL_PAIR_PRESENTATIONS,
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
            "purpose": "v15_all32_symmetric_base_failure_training",
            "contract": {
                "source_split": "train",
                "train_rows": 32,
                "scheduled_train_rows": 32,
                "base_failure_rows": 32,
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
            "v15_pair_curriculum": curriculum_binding(
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
                "proof": "four_hash_locked_train32_inputs_only_all32_ordinals_v1",
            },
            "excluded_artifacts": {"hard32": hard32_exclusion()},
        }
    )


def build_source_lock(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    bundle = load_json_object(output_dir / "manifest.json", description="V15 bundle manifest")
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
        "V15 output already exists: " + ", ".join(map(str, existing)),
    )
    if overwrite:
        for path in existing:
            require(
                path.is_file() and not path.is_symlink(),
                f"refusing to replace: {path}",
            )
            path.unlink()


def prepare_v15_data(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    v7_root: Path = V7_ROOT,
    source_lock_output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    v7_root = _lexical_absolute(v7_root)
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
        manifest = prepare_v15_data(
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
                "pair_presentations": manifest["curriculum"]["pair_presentations"],
                "presentation_checkpoints": manifest["curriculum"][
                    "presentation_checkpoints"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
