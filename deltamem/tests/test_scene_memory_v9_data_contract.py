from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import prepare_scene_memory_v9_data as builder
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v9_data_contract as contract


EXPECTED_ENTRIES_SHA256 = (
    "3b87877c33a73e9295e07a1e354beeca9a18e00cf66358c263ddb92d3d0f75c1"
)
EXPECTED_ORDERED_PAIRS_SHA256 = (
    "d710fec2abee4e7b8b5ecf7f75f005d7d872a17b6c0aea2f193dcb29e5f3a55d"
)
EXPECTED_PAIRS = (
    (1, 14),
    (22, 26),
    (3, 24),
    (5, 9),
    (10, 23),
    (19, 28),
    (20, 31),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def input_contract() -> tuple[list[dict], dict, dict, dict]:
    return builder.load_input_contract()


def test_v9_canonical_pairs_are_exact_reciprocal_value14(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    directed = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }

    assert builder.CANONICAL_VALUE14_PAIRS == EXPECTED_PAIRS
    assert builder.VALUE14_ORDINALS == (
        1,
        3,
        5,
        9,
        10,
        14,
        19,
        20,
        22,
        23,
        24,
        26,
        28,
        31,
    )
    for low, high in EXPECTED_PAIRS:
        assert directed[low]["donor_train_row_ordinal"] == high
        assert directed[high]["donor_train_row_ordinal"] == low
        assert directed[low]["source_row_sha256"] == rows[low]["row_sha256"]
        assert directed[high]["source_row_sha256"] == rows[high]["row_sha256"]
        assert directed[low]["target_stratum"] == directed[high]["target_stratum"]
        assert directed[low]["target_stratum"] in {
            "same_cardinality_value",
            "cross_cardinality_value",
        }


def test_v9_schedule_has_four_complete_deterministic_pair_passes(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    entries = builder.build_pair_schedule(rows, pair_manifest)
    audit = builder.schedule_audit(entries)

    assert len(entries) == builder.TOTAL_PAIR_STEPS == 28
    assert builder.CHECKPOINT_STEPS == (7, 14, 21, 28)
    assert audit["entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert audit["ordered_pairs_sha256"] == EXPECTED_ORDERED_PAIRS_SHA256
    assert audit["directed_presentations"] == 56
    assert audit["pair_batch_size"] == 2
    pass_orders = []
    for pass_index in range(builder.PAIR_PASSES):
        start = pass_index * builder.PAIRS_PER_PASS
        current = entries[start : start + builder.PAIRS_PER_PASS]
        pairs = [tuple(entry["canonical_pair_ordinals"]) for entry in current]
        assert set(pairs) == set(EXPECTED_PAIRS)
        assert len(pairs) == len(set(pairs)) == 7
        assert {entry["pass_index"] for entry in current} == {pass_index}
        assert [entry["pass_position"] for entry in current] == list(range(7))
        pass_orders.append(pairs)
    assert len({tuple(order) for order in pass_orders}) == builder.PAIR_PASSES

    pair_counts = Counter(
        tuple(entry["canonical_pair_ordinals"]) for entry in entries
    )
    assert pair_counts == Counter({pair: 4 for pair in EXPECTED_PAIRS})


def test_v9_pair_step_binds_both_reciprocal_directions(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    directed = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }
    entries = builder.build_pair_schedule(rows, pair_manifest)

    for schedule_index, entry in enumerate(entries):
        low, high = entry["canonical_pair_ordinals"]
        assert entry["schedule_index"] == schedule_index
        assert entry["step"] == schedule_index + 1
        assert entry["pair_batch_size"] == 2
        assert [member["train_row_ordinal"] for member in entry["members"]] == [
            low,
            high,
        ]
        assert [member["donor_train_row_ordinal"] for member in entry["members"]] == [
            high,
            low,
        ]
        for member in entry["members"]:
            ordinal = member["train_row_ordinal"]
            assert member["row_sha256"] == rows[ordinal]["row_sha256"]
            assert member["row_record_sha256"] == rows[ordinal]["record_sha256"]
            assert member["directed_pair_entry_sha256"] == directed[ordinal][
                "entry_sha256"
            ]


def test_v9_input_contract_hash_locks_existing_train_source_and_pairs(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    _, pair_manifest, source_manifest, bindings = input_contract

    assert bindings["train32"]["sha256"] == builder.TRAIN32_SHA256
    assert bindings["train32_rows"]["sha256"] == builder.TRAIN32_ROWS_SHA256
    assert bindings["pair_manifest"]["sha256"] == builder.PAIR_MANIFEST_FILE_SHA256
    assert bindings["source_manifest"]["sha256"] == builder.SOURCE_MANIFEST_FILE_SHA256
    assert pair_manifest["manifest_sha256"] == builder.PAIR_MANIFEST_SHA256
    assert pair_manifest["entries_sha256"] == builder.PAIR_ENTRIES_SHA256
    assert source_manifest["manifest_sha256"] == builder.SOURCE_MANIFEST_SHA256


def test_v9_rejects_rehashed_pair_manifest_drift(
    input_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = input_contract
    drifted = json.loads(json.dumps(pair_manifest))
    drifted_entry = dict(drifted["directed_pairs"][1])
    drifted_entry.pop("entry_sha256")
    drifted_entry["donor_train_row_ordinal"] = 22
    drifted["directed_pairs"][1] = builder.with_self_hash(
        drifted_entry,
        field="entry_sha256",
    )
    drifted["entries_sha256"] = builder.canonical_sha256(drifted["directed_pairs"])

    with pytest.raises(builder.ContractError, match="pair entries hash differs"):
        builder.build_pair_schedule(rows, drifted)


def test_v9_pair_schedule_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.prepare_v9_data(output_dir=first)
    builder.prepare_v9_data(output_dir=second)

    assert (first / "pair_schedule.jsonl").read_bytes() == (
        second / "pair_schedule.jsonl"
    ).read_bytes()
    first_manifest = load_json(first / "pair_schedule_manifest.json")
    second_manifest = load_json(second / "pair_schedule_manifest.json")
    assert first_manifest["curriculum"]["entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert second_manifest["curriculum"]["entries_sha256"] == EXPECTED_ENTRIES_SHA256


def test_v9_source_lock_explicitly_excludes_hard32_without_locator() -> None:
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


def test_v9_build_and_validation_never_access_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = Path(
        "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
        "pairs_candidate64_failure32_holdout32_v1/holdout.jsonl"
    )
    original_open = Path.open
    original_read_text = Path.read_text
    original_resolve = Path.resolve
    original_builder_sha256 = builder.sha256_file
    original_contract_sha256 = contract.sha256_file

    def is_protected(path: Path) -> bool:
        return str(path) == str(protected)

    def guarded_open(path: Path, *args, **kwargs):
        if is_protected(path):
            raise AssertionError("V9 attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if is_protected(path):
            raise AssertionError("V9 attempted to read Hard32")
        return original_read_text(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if is_protected(path):
            raise AssertionError("V9 attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    def guarded_builder_sha256(path: Path) -> str:
        if is_protected(path):
            raise AssertionError("V9 attempted to hash Hard32")
        return original_builder_sha256(path)

    def guarded_contract_sha256(path: Path) -> str:
        if is_protected(path):
            raise AssertionError("V9 attempted to hash Hard32")
        return original_contract_sha256(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(builder, "sha256_file", guarded_builder_sha256)
    monkeypatch.setattr(contract, "sha256_file", guarded_contract_sha256)

    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v9_data(output_dir=root, source_lock_output=lock_path)
    result = contract.validate_bundle(root, source_lock_path=lock_path)

    assert result["status"] == "pass"
    assert result["hard32_rows_in_schedule"] == 0
    assert result["hard32_exclusion"]["path"] is None
    assert result["hard32_exclusion"]["sha256"] is None


def test_v9_frozen_bundle_passes_full_contract() -> None:
    result = contract.validate_bundle()

    assert result["status"] == "pass"
    assert result["train32_sha256"] == builder.TRAIN32_SHA256
    assert result["pair_schedule_entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert result["ordered_pairs_sha256"] == EXPECTED_ORDERED_PAIRS_SHA256
    assert result["pair_steps"] == 28
    assert result["directed_presentations"] == 56
    assert result["checkpoint_steps"] == [7, 14, 21, 28]
    assert result["canonical_value14_pairs"] == [list(pair) for pair in EXPECTED_PAIRS]
    assert result["hard32_rows_in_schedule"] == 0


def test_v9_validator_rejects_pair_schedule_tampering(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v9_data(output_dir=root, source_lock_output=lock_path)
    schedule = root / "pair_schedule.jsonl"
    schedule.write_bytes(schedule.read_bytes() + b"\n")

    with pytest.raises(builder.ContractError, match="artifact SHA-256 differs: pair_schedule"):
        contract.validate_bundle(root, source_lock_path=lock_path)
