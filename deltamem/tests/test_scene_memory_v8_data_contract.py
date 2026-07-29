from __future__ import annotations

from argparse import Namespace
from collections import Counter
import json
from pathlib import Path

import pytest

from deltamem.train.delta_sft_experimental import _scene_state_source_manifest_identity
from experiments.rethinking_rwkv_ms_gemma import prepare_scene_memory_v8_data as builder
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v8_data_contract as contract


EXPECTED_ENTRIES_SHA256 = (
    "979ca0c2dc253373eed6b4221cd6fa4c37f4a7a6e93173e8ce7f86f811e23df0"
)
EXPECTED_ORDINALS_SHA256 = (
    "dfd2efa5f0fb8e5969fbb7f36689cc4d47d66166b40b2ee08c8d26d70f2d17f3"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def parent_contract() -> tuple[list[dict], dict, dict, dict]:
    return builder.load_parent_contract()


def test_v8_value14_is_exactly_the_locked_value_strata(
    parent_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    _, pair_manifest, _, _ = parent_contract
    directed = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }
    observed = tuple(
        ordinal
        for ordinal in range(32)
        if directed[ordinal]["target_stratum"]
        in {"same_cardinality_value", "cross_cardinality_value"}
    )

    assert observed == builder.VALUE14_ORDINALS
    assert observed == (1, 3, 5, 9, 10, 14, 19, 20, 22, 23, 24, 26, 28, 31)


def test_v8_schedule_has_four_deterministic_value_passes_then_balanced96(
    parent_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = parent_contract
    entries = builder.build_schedule(rows, pair_manifest)
    audit = builder.schedule_audit(entries)

    assert len(entries) == builder.TOTAL_STEPS == 152
    assert audit["entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert audit["ordered_train_row_ordinals_sha256"] == EXPECTED_ORDINALS_SHA256
    value = entries[: builder.VALUE14_STEPS]
    assert all(entry["phase"] == "value14" for entry in value)
    pass_orders = []
    for pass_index in range(builder.VALUE14_PASSES):
        start = pass_index * len(builder.VALUE14_ORDINALS)
        current = value[start : start + len(builder.VALUE14_ORDINALS)]
        ordinals = [entry["train_row_ordinal"] for entry in current]
        assert set(ordinals) == set(builder.VALUE14_ORDINALS)
        assert len(ordinals) == len(set(ordinals)) == 14
        assert {entry["sampling"]["pass_index"] for entry in current} == {pass_index}
        pass_orders.append(ordinals)
    assert len({tuple(order) for order in pass_orders}) == builder.VALUE14_PASSES

    balanced = entries[builder.VALUE14_STEPS :]
    assert len(balanced) == 96
    assert Counter(entry["target_stratum"] for entry in balanced) == builder.BALANCED_QUOTAS
    for round_index in range(builder.BALANCED_STEPS_PER_STRATUM):
        current = balanced[round_index * 3 : round_index * 3 + 3]
        assert {entry["target_stratum"] for entry in current} == set(builder.TARGET_STRATA)
        assert {entry["sampling"]["round_index"] for entry in current} == {round_index}


def test_v8_schedule_preserves_original_ordinals_and_pair_bindings(
    parent_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = parent_contract
    entries = builder.build_schedule(rows, pair_manifest)
    pairs = {
        entry["train_row_ordinal"]: entry
        for entry in pair_manifest["directed_pairs"]
    }

    for schedule_index, entry in enumerate(entries):
        ordinal = entry["train_row_ordinal"]
        assert entry["schedule_index"] == schedule_index
        assert entry["step"] == schedule_index + 1
        assert entry["row_sha256"] == rows[ordinal]["row_sha256"]
        assert entry["row_record_sha256"] == rows[ordinal]["record_sha256"]
        assert entry["official_source_index"] == rows[ordinal]["official_source_index"]
        assert entry["pair_entry_sha256"] == pairs[ordinal]["entry_sha256"]
        assert entry["donor_train_row_ordinal"] == pairs[ordinal][
            "donor_train_row_ordinal"
        ]


def test_v8_schedule_rejects_category_drift(
    parent_contract: tuple[list[dict], dict, dict, dict],
) -> None:
    rows, pair_manifest, _, _ = parent_contract
    drifted = json.loads(json.dumps(pair_manifest))
    entry = drifted["directed_pairs"][1]
    entry.pop("entry_sha256")
    entry["target_stratum"] = "presence"
    drifted["directed_pairs"][1] = builder.with_self_hash(
        entry,
        field="entry_sha256",
    )

    with pytest.raises(builder.ContractError, match="locked Value14 ordinals differ"):
        builder.build_schedule(rows, drifted)


def test_v8_builder_is_byte_deterministic_for_materialized_schedule(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.prepare_v8_data(output_dir=first)
    builder.prepare_v8_data(output_dir=second)

    assert (first / "schedule.jsonl").read_bytes() == (second / "schedule.jsonl").read_bytes()
    assert builder.sha256_file(first / "schedule.jsonl") == builder.sha256_file(
        second / "schedule.jsonl"
    )
    assert load_json(first / "schedule_manifest.json")["curriculum"][
        "entries_sha256"
    ] == EXPECTED_ENTRIES_SHA256


def test_v8_source_lock_is_self_hashed_and_hard32_is_eval_only() -> None:
    lock = contract.load_source_lock()
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}

    assert lock["schema"] == builder.SOURCE_LOCK_SCHEMA
    assert lock["lock_sha256"] == builder.canonical_sha256(unsigned)
    assert lock["parent_train32_sha256"] == builder.PARENT_TRAIN32_SHA256
    assert lock["fixed_hard32"] == {
        "path": str(builder.HARD32_FILE),
        "sha256": builder.HARD32_FILE_SHA256,
        "role": "protected_evaluation_only_not_scheduled",
    }
    assert lock["curriculum"]["value14_ordinals"] == list(builder.VALUE14_ORDINALS)
    assert lock["curriculum"]["balanced_quotas"] == builder.BALANCED_QUOTAS


def test_v8_frozen_bundle_passes_full_contract() -> None:
    result = contract.validate_bundle()

    assert result["status"] == "pass"
    assert result["parent_train32_sha256"] == builder.PARENT_TRAIN32_SHA256
    assert result["schedule_entries_sha256"] == EXPECTED_ENTRIES_SHA256
    assert result["total_steps"] == 152
    assert result["checkpoint_steps"] == list(builder.CHECKPOINT_STEPS)
    assert result["balanced_counts"] == builder.BALANCED_QUOTAS
    assert result["hard32_rows_in_schedule"] == 0


def test_v8_pre_gate_build_and_validation_never_access_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard32 = Path(str(builder.HARD32_FILE))
    original_open = Path.open
    original_read_text = Path.read_text
    original_resolve = Path.resolve

    def guarded_open(path: Path, *args, **kwargs):
        if Path(str(path)) == hard32:
            raise AssertionError("V8 pre-gate validation attempted to open Hard32")
        return original_open(path, *args, **kwargs)

    def guarded_read_text(path: Path, *args, **kwargs):
        if Path(str(path)) == hard32:
            raise AssertionError("V8 pre-gate validation attempted to read Hard32")
        return original_read_text(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args, **kwargs):
        if Path(str(path)) == hard32:
            raise AssertionError("V8 pre-gate validation attempted to resolve Hard32")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v8_data(output_dir=root, source_lock_output=lock_path)
    result = contract.validate_bundle(root, source_lock_path=lock_path)

    assert result["status"] == "pass"
    assert result["hard32_rows_in_schedule"] == 0


def test_v8_source_manifest_binds_original_train32_with_trainer_validator() -> None:
    source_path = builder.DEFAULT_OUTPUT_DIR / "source_manifest.json"
    train_path = builder.V7_ROOT / "train32.jsonl"
    identity = _scene_state_source_manifest_identity(
        Namespace(
            scene_state_source_manifest=source_path,
            expected_scene_state_source_manifest_sha256=builder.sha256_file(source_path),
            train_file=train_path,
        )
    )

    assert identity is not None
    assert identity["schema"] == builder.SOURCE_SCHEMA
    assert identity["train_file_sha256"] == builder.PARENT_TRAIN32_SHA256
    assert identity["train_rows"] == 32
    assert identity["train_source_split"] == "train"


def test_v8_validator_rejects_schedule_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    lock_path = tmp_path / "source_lock.json"
    builder.prepare_v8_data(output_dir=root, source_lock_output=lock_path)
    schedule = root / "schedule.jsonl"
    schedule.write_bytes(schedule.read_bytes() + b"\n")

    with pytest.raises(builder.ContractError, match="artifact SHA-256 differs: schedule"):
        contract.validate_bundle(root, source_lock_path=lock_path)
