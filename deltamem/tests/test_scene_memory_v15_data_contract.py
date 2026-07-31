from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_scene_memory_v15_data as builder,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_data_contract as contract,
)


EXPECTED_ENTRIES_SHA256 = (
    "6a315c5c3be51cf9dc1ecf49ca4f0d649023d92c40407a55ddad08da696d224c"
)
EXPECTED_ORDERED_PAIRS_SHA256 = (
    "5aea39c86edd9460454172bc046961b607c1792f27a2abb198ed2996ed10b26e"
)
EXPECTED_PREFIXES = {
    "1": "2253f0b7a678bad0a4d0a2c8150a8f1b8f1acb88185f411bd4bc03cf65a0f733",
    "2": "55538c8dff030eb30c6dc66708ab798eb2a2b7191fc95db4eaf4afaac30ede65",
    "3": "8a35c31f28c572aa8a2d2fa6c3b6d1a8e9df9931ca47c0c0cf45405667dcecc1",
    "4": EXPECTED_ORDERED_PAIRS_SHA256,
}
EXPECTED_PAIRS = (
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


@pytest.fixture(scope="module")
def input_contract() -> tuple[list[dict], dict, dict, dict]:
    return builder.load_input_contract()


def test_v15_pairs_cover_all_base_failures_and_all_empty_targets(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, source_manifest, _ = input_contract
    directed = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }

    assert builder.CANONICAL_ALL32_PAIRS == EXPECTED_PAIRS
    assert sorted(ordinal for pair in EXPECTED_PAIRS for ordinal in pair) == list(
        range(32)
    )
    assert builder.EMPTY_ORDINALS == (4, 6, 7, 11, 12, 15, 16, 17, 29)
    assert Counter(row["gold_boundary_count"] for row in rows) == {
        0: 9,
        1: 16,
        2: 5,
        3: 2,
    }
    assert all(row["source_split"] == "train" for row in rows)
    assert all(row["strict_failure_stratum"] is not None for row in rows)
    assert source_manifest["contract"] == {
        "episode_contract": {
            "episode_recent_messages": 0,
            "read_supervision": "system + assistant",
            "write_phase": "system + user",
        },
        "source_split": "train",
        "test_rows": 0,
        "val_rows": 0,
    }

    pair_strata: Counter[str] = Counter()
    paired_empty: list[int] = []
    for low, high in EXPECTED_PAIRS:
        assert directed[low]["donor_train_row_ordinal"] == high
        assert directed[high]["donor_train_row_ordinal"] == low
        assert directed[low]["target_stratum"] == directed[high]["target_stratum"]
        pair_strata[directed[low]["target_stratum"]] += 1
        empties = [
            ordinal
            for ordinal in (low, high)
            if rows[ordinal]["gold_boundary_count"] == 0
        ]
        if directed[low]["target_stratum"] == "presence":
            assert len(empties) == 1
            paired_empty.extend(empties)
        else:
            assert empties == []
    assert pair_strata == builder.PAIR_STRATUM_COUNTS
    assert sorted(paired_empty) == list(builder.EMPTY_ORDINALS)


def test_v15_four_full_cycles_and_checkpoint_prefixes_are_locked(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    entries = builder.build_pair_schedule(rows, pair_manifest)
    audit = builder.schedule_audit(entries)

    assert builder.PAIR_CYCLES == 4
    assert builder.PAIRS_PER_CYCLE == 16
    assert builder.TOTAL_PAIR_PRESENTATIONS == 64
    assert builder.DIRECTED_PRESENTATIONS == 128
    assert builder.CHECKPOINT_CYCLES == (1, 2, 3, 4)
    assert builder.PRESENTATION_CHECKPOINTS == (16, 32, 48, 64)
    assert builder.PAIR_PREFIX_SHA256_BY_CHECKPOINT == EXPECTED_PREFIXES
    assert audit["entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert audit["ordered_pairs_sha256"] == EXPECTED_ORDERED_PAIRS_SHA256
    assert audit["pair_prefix_sha256_by_checkpoint"] == EXPECTED_PREFIXES
    assert len(set(builder.FULL_PAIR_CYCLES)) == 4

    for cycle_index in range(4):
        start = cycle_index * 16
        cycle = entries[start : start + 16]
        pairs = [tuple(entry["canonical_pair_ordinals"]) for entry in cycle]
        assert tuple(pairs) == builder.FULL_PAIR_CYCLES[cycle_index]
        assert set(pairs) == set(EXPECTED_PAIRS)
        assert len(pairs) == len(set(pairs)) == 16
        assert sorted(ordinal for pair in pairs for ordinal in pair) == list(range(32))
        assert Counter(entry["target_stratum"] for entry in cycle) == {
            "presence": 9,
            "same_cardinality_value": 5,
            "cross_cardinality_value": 2,
        }


def test_v15_schedule_members_bind_reciprocal_train_failures(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    entries = builder.build_pair_schedule(rows, pair_manifest)

    empty_exposures: Counter[int] = Counter()
    row_exposures: Counter[int] = Counter()
    for entry in entries:
        low, high = entry["canonical_pair_ordinals"]
        members = entry["members"]
        assert [member["train_row_ordinal"] for member in members] == [low, high]
        assert [member["donor_train_row_ordinal"] for member in members] == [
            high,
            low,
        ]
        for member in members:
            ordinal = member["train_row_ordinal"]
            row_exposures[ordinal] += 1
            assert member["source_split"] == "train"
            assert member["strict_failure_stratum"] == rows[ordinal][
                "strict_failure_stratum"
            ]
            assert member["base_record_sha256"] == rows[ordinal][
                "base_record_sha256"
            ]
            assert member["label_sha256"] == rows[ordinal]["label_sha256"]
        if entry["target_stratum"] == "presence":
            assert len(entry["empty_member_ordinals"]) == 1
            empty_exposures.update(entry["empty_member_ordinals"])
        else:
            assert entry["empty_member_ordinals"] == []

    assert row_exposures == Counter({ordinal: 4 for ordinal in range(32)})
    assert empty_exposures == Counter(
        {ordinal: 4 for ordinal in builder.EMPTY_ORDINALS}
    )


@pytest.mark.parametrize(
    "filename",
    ("val.jsonl", "test.jsonl", "holdout.jsonl", "hard32.jsonl"),
)
def test_v15_input_guard_rejects_non_train32_artifacts(filename: str) -> None:
    with pytest.raises(builder.ContractError, match="not an allowed pinned Train32"):
        builder.guard_v15_input_path(builder.V7_ROOT / filename)


def test_v15_build_and_validation_read_only_four_pinned_external_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    original_read_text = Path.read_text
    read_paths: set[Path] = set()

    def record(path: Path) -> None:
        read_paths.add(Path(path).absolute())

    def tracked_open(path: Path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode:
            record(path)
        return original_open(path, *args, **kwargs)

    def tracked_read_text(path: Path, *args, **kwargs):
        record(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "read_text", tracked_read_text)

    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v15_data(output_dir=root, source_lock_output=lock_path)
    result = contract.validate_bundle(root, source_lock_path=lock_path)

    allowed = builder.allowed_input_paths()
    external_reads = {
        path
        for path in read_paths
        if not path.is_relative_to(tmp_path.absolute())
    }
    assert external_reads == allowed
    assert result["status"] == "pass"
    assert result["hard32_rows_in_schedule"] == 0
    assert result["hard32_exclusion"]["path"] is None
    assert result["hard32_exclusion"]["sha256"] is None


def test_v15_schedule_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.prepare_v15_data(output_dir=first)
    builder.prepare_v15_data(output_dir=second)

    assert (first / "pair_schedule.jsonl").read_bytes() == (
        second / "pair_schedule.jsonl"
    ).read_bytes()
    first_manifest = json.loads(
        (first / "pair_schedule_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second / "pair_schedule_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["curriculum"]["entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert second_manifest["curriculum"]["entries_sha256"] == EXPECTED_ENTRIES_SHA256


def test_v15_source_lock_excludes_hard32_without_a_locator() -> None:
    lock = contract.load_source_lock()
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    exclusion = lock["excluded_artifacts"]["hard32"]

    assert lock["schema"] == builder.SOURCE_LOCK_SCHEMA
    assert lock["lock_sha256"] == builder.canonical_sha256(unsigned)
    assert exclusion == {
        "name": "Hard32",
        "included": False,
        "path": None,
        "sha256": None,
        "policy": "forbidden_not_resolved_opened_or_hashed",
    }
    assert "holdout.jsonl" not in json.dumps(lock, sort_keys=True)


def test_v15_frozen_bundle_passes_full_contract() -> None:
    result = contract.validate_bundle()

    assert result["status"] == "pass"
    assert result["train32_sha256"] == builder.TRAIN32_SHA256
    assert result["scheduled_train_rows"] == 32
    assert result["base_failure_rows"] == 32
    assert result["empty_rows"] == 9
    assert result["pair_cycles"] == 4
    assert result["pairs_per_cycle"] == 16
    assert result["pair_presentations"] == 64
    assert result["directed_presentations"] == 128
    assert result["pair_stratum_counts_per_cycle"] == {
        "presence": 9,
        "same_cardinality_value": 5,
        "cross_cardinality_value": 2,
    }
    assert result["presentation_checkpoints"] == [16, 32, 48, 64]
    assert result["pair_prefix_sha256_by_checkpoint"] == EXPECTED_PREFIXES
    assert result["pair_schedule_entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert result["ordered_pairs_sha256"] == EXPECTED_ORDERED_PAIRS_SHA256
    assert result["hard32_rows_in_schedule"] == 0


def test_v15_validator_rejects_schedule_tampering(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v15_data(output_dir=root, source_lock_output=lock_path)
    schedule = root / "pair_schedule.jsonl"
    schedule.write_bytes(schedule.read_bytes() + b"\n")

    with pytest.raises(builder.ContractError, match="artifact SHA-256 differs: pair_schedule"):
        contract.validate_bundle(root, source_lock_path=lock_path)
