from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import scene_memory_v6_data_contract as contract


def test_identity_data_contract_binds_failure32_and_fixed_val32() -> None:
    payload = contract.build_official_contract()

    assert payload["schema"] == "rwkv_ms_scene_v6_identity_data.v1"
    assert payload["experiment"] == "scene_memory_v6_identity_proof"
    assert payload["manifest_sha256"] == contract.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )

    train = payload["training_partition"]
    assert train == {
        "source_split": "train",
        "selection_rule": "frozen_base_failure32_from_predeclared_official_train_candidate64",
        "rows": 32,
        "path": str(contract.PROOF_TRAIN),
        "sha256": contract.PROOF_TRAIN_SHA256,
        "row_hashes_sha256": contract.PROOF_TRAIN_ROW_HASHES_SHA256,
        "prompt_hashes_sha256": contract.PROOF_TRAIN_PROMPT_HASHES_SHA256,
        "row_manifest": {
            "path": str(contract.PROOF_TRAIN_ROW_MANIFEST),
            "sha256": contract.PROOF_TRAIN_ROW_MANIFEST_SHA256,
        },
        "val_or_test_rows_emitted_for_training": 0,
    }

    selection = payload["hard_evaluation_selection"]
    assert selection["source_split"] == "val"
    assert selection["selection_rule"] == (
        "lowest_predeclared_prompt_hash_ranks_without_labels_or_model_outputs"
    )
    assert selection["rows"] == 32
    assert selection["source_indices"] == list(contract.HARD32_SOURCE_INDICES)
    assert selection["path"] == str(contract.HARD32_DATA)
    assert selection["sha256"] == contract.HARD32_DATA_SHA256


def test_fixed_val32_is_not_described_as_failure_mined() -> None:
    payload = contract.build_official_contract()
    selection = payload["hard_evaluation_selection"]

    assert "failure" not in selection["selection_rule"]
    assert selection["checkpoint_selection_only"] is True
    assert payload["split_policy"]["full_val"] == (
        "forbidden until a hard32 pass receipt exists"
    )


def test_test_split_is_never_emitted() -> None:
    payload = contract.build_official_contract()

    assert payload["test_policy"] == {
        "rows_emitted_for_training": 0,
        "rows_emitted_for_checkpoint_selection": 0,
        "full_validation_before_hard32_pass": "forbidden",
        "test_before_validation_selection_receipt": "forbidden",
    }
    assert payload["splits"]["test"]["rows"] == 149
    assert payload["pair_manifest"]["sha256"] == contract.PAIR_MANIFEST_SHA256


def test_contract_records_paragraph_overlap_without_claiming_disjointness() -> None:
    payload = contract.build_official_contract()
    overlap = payload["overlap_audit"]

    assert overlap["passage_disjoint"] is False
    assert overlap["pairs"]["train__val"]["exact_normalized_full_prompts_shared"] == 0
    assert overlap["pairs"]["train__val"]["exact_normalized_paragraphs_shared"] == 1451
    assert "Do not claim passage-level" in overlap["warning"]


def test_selected_failure32_and_fixed_val32_are_passage_disjoint() -> None:
    payload = contract.build_official_contract()
    audit = payload["selected_slice_overlap_audit"]

    assert audit["passage_disjoint"] is True
    assert audit["training_partition"]["normalized_paragraphs"] == {
        "per_row_unique_instances": 433,
        "unique": 426,
        "per_row_hashes_sha256": (
            "69cc46065213dfd9bbc0f08061b11a7512bb38c31d903b9687c2498f08a0181c"
        ),
        "unique_hashes_sha256": (
            "988d08886f5749848553809989746effdcfb4088064c256ac806554063713bbc"
        ),
    }
    assert audit["evaluation_partition"]["normalized_paragraphs"] == {
        "per_row_unique_instances": 412,
        "unique": 412,
        "per_row_hashes_sha256": (
            "d897ef2a6b4d42b14e4e7ede53177fcdf5c9d3e8242ac6f21c55791883a0bc6e"
        ),
        "unique_hashes_sha256": (
            "e01ee3766bdc0093cec804101bdf0bf8b83355bdd26725772b4990a375044f3c"
        ),
    }
    assert audit["comparison"] == {
        "left_split": "failure32_train",
        "right_split": "fixed_val32",
        "exact_normalized_full_prompts_shared": 0,
        "exact_normalized_paragraphs_shared": 0,
        "left_rows_with_shared_paragraph": 0,
        "right_rows_with_shared_paragraph": 0,
        "shared_paragraph_hashes_sha256": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
    }


def test_output_write_is_exclusive_and_checksum_bound(tmp_path: Path) -> None:
    output = tmp_path / "data_contract_manifest.json"
    assert contract.main(["--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == contract.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    assert contract.main(["--output", str(output)]) == 2


def test_pair_bundle_hash_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "PROOF_TRAIN_SHA256", "0" * 64)
    with pytest.raises(contract.ContractError, match="frozen pair artifact SHA-256"):
        contract.build_official_contract()
