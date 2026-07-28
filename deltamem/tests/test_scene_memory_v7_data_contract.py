from __future__ import annotations

from argparse import Namespace
from collections import Counter
import json
from pathlib import Path

import pytest

from deltamem.train.delta_sft_experimental import _scene_state_source_manifest_identity
from experiments.rethinking_rwkv_ms_gemma import prepare_scene_memory_v7_data as builder
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v7_data_contract as contract
from experiments.rethinking_rwkv_ms_gemma.run_novel_agent_eval import score_prediction


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v7_source_lock_is_self_hashed_and_binds_fixed_hard32() -> None:
    lock = contract.load_source_lock()

    assert lock["schema"] == "rwkv_ms_scene_memory_v7_source_lock.v1"
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    assert lock["lock_sha256"] == builder.canonical_sha256(unsigned)
    assert lock["fixed_hard32"] == {
        "base_records_sha256": builder.HARD32_BASE_RECORDS_SHA256,
        "data_sha256": builder.HARD32_FILE_SHA256,
        "selection_sha256": builder.HARD32_SELECTION_SHA256,
    }
    assert lock["pairing_quotas"] == {
        "train32": builder.TRAIN_DIRECTED_PAIR_QUOTAS,
        "tiny2": builder.TINY_DIRECTED_PAIR_QUOTAS,
    }


def test_v7_frozen_bundle_passes_full_contract() -> None:
    result = contract.validate_bundle()

    assert result["status"] == "pass"
    assert result["minimum_failure_stratum_l1"] == 2
    assert result["train32_pair_quotas"] == builder.TRAIN_DIRECTED_PAIR_QUOTAS
    assert result["tiny2_pair_quotas"] == builder.TINY_DIRECTED_PAIR_QUOTAS
    assert result["train32_sha256"] == (
        "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9"
    )


def test_v7_train32_matches_hard_cardinality_and_closest_failure_strata() -> None:
    manifest = load_json(builder.DEFAULT_OUTPUT_DIR / "manifest.json")
    rows = load_jsonl(builder.DEFAULT_OUTPUT_DIR / "train32_rows.jsonl")

    assert Counter(row["gold_boundary_count"] for row in rows) == builder.GOLD_CARDINALITY_QUOTAS
    assert all(row["strict_failure_stratum"] in builder.FAILURE_STRATA for row in rows)
    assert manifest["selection"]["minimum_cellwise_l1"] == 2
    assert manifest["selection"]["overall_failure_count_l1"] == 2
    assert manifest["selection"]["relaxed_silently"] is False
    assert manifest["selection"]["hard32_failure_counts"] == {
        "invalid_schema": 25,
        "false_positive_only": 2,
        "false_negative_only": 1,
        "mixed": 4,
    }
    assert manifest["selection"]["selected_failure_counts"] == {
        "invalid_schema": 25,
        "false_positive_only": 2,
        "false_negative_only": 0,
        "mixed": 5,
    }


def test_v7_uses_current_exact_strict_scorer_not_format_recovery() -> None:
    gold = {"boundaries": [1]}
    recoverable_noncanonical = [{"boundary": "P1"}]
    score = score_prediction("scene", recoverable_noncanonical, gold)

    assert score["schema_valid"] is False
    assert score["tp"] == 0
    assert score["fn"] == 1
    assert builder.strict_failure_stratum(score) == "invalid_schema"


def test_v7_train32_pairing_is_exact_symmetric_and_token_bound() -> None:
    pair_manifest = load_json(builder.DEFAULT_OUTPUT_DIR / "train32_pair_manifest.json")
    rows = load_jsonl(builder.DEFAULT_OUTPUT_DIR / "train32_rows.jsonl")
    directed = pair_manifest["directed_pairs"]
    by_ordinal = {entry["train_row_ordinal"]: entry for entry in directed}

    assert pair_manifest["schema"] == builder.PAIRING_SCHEMA
    assert pair_manifest["quotas"] == builder.TRAIN_DIRECTED_PAIR_QUOTAS
    assert Counter(entry["target_stratum"] for entry in directed) == (
        builder.TRAIN_DIRECTED_PAIR_QUOTAS
    )
    assert pair_manifest["optimization"]["global_minimum_after_exact_quotas"] is True
    assert pair_manifest["optimization"]["evaluated_cross_choices"] == 7381
    assert pair_manifest["optimization"]["feasible_cross_choices"] == 5860
    for ordinal, entry in by_ordinal.items():
        donor_ordinal = entry["donor_train_row_ordinal"]
        reverse = by_ordinal[donor_ordinal]
        assert reverse["donor_train_row_ordinal"] == ordinal
        assert entry["source_row_sha256"] == rows[ordinal]["row_sha256"]
        assert entry["donor_row_sha256"] == rows[donor_ordinal]["row_sha256"]
        assert entry["source_write_sha256"] == rows[ordinal]["token_metadata"][
            "write_input_ids_sha256"
        ]
        assert entry["source_generation_prefix_sha256"] == rows[ordinal][
            "token_metadata"
        ]["generation_prefix_input_ids_sha256"]
        assert entry["selected_target_predictor_positions"] == [
            entry["selected_target_positions"][0] - 1
        ]
        assert entry["selected_target_token_ids"] != entry["donor_target_token_ids"]


@pytest.mark.parametrize(
    ("prefix", "rows"),
    (("train32", 32), ("tiny2", 2)),
)
def test_v7_source_manifests_bind_with_existing_trainer_validator(
    prefix: str,
    rows: int,
) -> None:
    source_path = builder.DEFAULT_OUTPUT_DIR / f"{prefix}_source_manifest.json"
    train_path = builder.DEFAULT_OUTPUT_DIR / f"{prefix}.jsonl"
    identity = _scene_state_source_manifest_identity(
        Namespace(
            scene_state_source_manifest=source_path,
            expected_scene_state_source_manifest_sha256=builder.sha256_file(source_path),
            train_file=train_path,
        )
    )

    assert identity is not None
    assert identity["schema"] == builder.SOURCE_SCHEMA
    assert identity["train_file"] == str(train_path.resolve())
    assert identity["train_file_sha256"] == builder.sha256_file(train_path)
    assert identity["train_rows"] == rows


def test_v7_tiny2_is_deterministic_hardest_positive_same_cardinality_pair() -> None:
    train_rows = load_jsonl(builder.DEFAULT_OUTPUT_DIR / "train32_rows.jsonl")
    pair_manifest = load_json(builder.DEFAULT_OUTPUT_DIR / "train32_pair_manifest.json")
    tiny_rows = load_jsonl(builder.DEFAULT_OUTPUT_DIR / "tiny2_rows.jsonl")
    directed = pair_manifest["directed_pairs"]
    candidates = []
    for entry in directed:
        source = entry["train_row_ordinal"]
        donor = entry["donor_train_row_ordinal"]
        if source >= donor or entry["target_stratum"] != "same_cardinality_value":
            continue
        source_row = train_rows[source]
        donor_row = train_rows[donor]
        assert source_row["gold_boundary_count"] > 0
        severity = sum(
            row["strict_score"]["fp"]
            + row["strict_score"]["fn"]
            + int(not row["strict_score"]["schema_valid"])
            for row in (source_row, donor_row)
        )
        pair_hashes = tuple(sorted((source_row["row_sha256"], donor_row["row_sha256"])))
        candidates.append((-severity, pair_hashes))
    expected_hashes = set(min(candidates)[1])

    assert {row["row_sha256"] for row in tiny_rows} == expected_hashes
    assert len({row["gold_boundary_count"] for row in tiny_rows}) == 1
    assert len({row["label_sha256"] for row in tiny_rows}) == 2
    tiny_pair = load_json(builder.DEFAULT_OUTPUT_DIR / "tiny2_pair_manifest.json")
    assert tiny_pair["quotas"] == builder.TINY_DIRECTED_PAIR_QUOTAS


def test_v7_selection_fails_closed_when_cardinality_quota_is_missing() -> None:
    target = {
        cardinality: {stratum: 0 for stratum in builder.FAILURE_STRATA}
        for cardinality in builder.GOLD_CARDINALITY_QUOTAS
    }

    with pytest.raises(builder.ContractError, match="cannot satisfy cardinality quota"):
        builder.select_train32([], target)
