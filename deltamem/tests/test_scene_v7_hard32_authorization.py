from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as evaluator
from experiments.rethinking_rwkv_ms_gemma import run_scene_train32_eval as v7


def _receipt_payload(*, contract: str, status: str) -> dict[str, object]:
    return {
        "contract": contract,
        "gate": {"status": status},
        "input_artifacts": {
            name: {
                "path": f"/locked/{name}",
                "actual_sha256": character * 64,
            }
            for name, character in (
                ("dataset", "a"),
                ("row_manifest", "b"),
                ("pair_manifest", "c"),
                ("source_manifest", "d"),
            )
        },
    }


def test_cli_exposes_clearly_scoped_v7_train32_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scene_state_eval.py",
            "--base-model",
            "/model",
            "--memory-dir",
            "/checkpoint",
            "--dataset-file",
            "/data/val.jsonl",
            "--output-dir",
            "/output",
            "--row-indices",
            "3",
            "--evaluation-contract",
            "scene_v6_identity_hard32",
            "--scene-v7-train32-receipt",
            "/receipts/train32.json",
        ],
    )

    args = evaluator.parse_args()

    assert args.scene_v7_train32_receipt == Path("/receipts/train32.json")


@pytest.mark.parametrize(
    "contract",
    ["generic", "scene_v6_matched_donor_validation"],
)
def test_v7_train32_receipt_is_scoped_only_to_fixed_hard32(contract: str) -> None:
    with pytest.raises(ValueError, match="accepted only by scene_v6_identity_hard32"):
        evaluator.validate_scene_v7_train32_receipt_scope(
            evaluation_contract=contract,
            receipt_path=Path("train32_receipt.json"),
        )


def test_v7_train32_receipt_reconstructs_contract_and_binds_exact_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "train32_receipt.json"
    receipt_path.write_text(
        json.dumps(
            _receipt_payload(
                contract="scene_v7_train32_overfit",
                status="pass",
            )
        ),
        encoding="utf-8",
    )
    memory_dir = tmp_path / "checkpoint-64"
    input_contract = {"contract": "scene_v7_train32_overfit"}
    checkpoint = {
        "memory_dir": str(memory_dir),
        "adapter_sha256": "e" * 64,
        "config_sha256": "f" * 64,
        "training_protocol_sha256": "1" * 64,
        "trainer_state_sha256": "2" * 64,
    }
    calls: dict[str, object] = {}

    def fake_validate_contract(**kwargs):
        calls["contract_kwargs"] = kwargs
        return input_contract

    def fake_validate_checkpoint(path, *, input_contract):
        calls["checkpoint_path"] = path
        calls["input_contract"] = input_contract
        return checkpoint

    def fake_validate_authorization(path, *, expected_checkpoint):
        calls["receipt_path"] = path
        calls["expected_checkpoint"] = expected_checkpoint
        return {
            "receipt_sha256": "3" * 64,
            "evaluation_fingerprint": "4" * 64,
            "source_lock": {
                "path": str(v7.DEFAULT_SOURCE_LOCK),
                "file_sha256": "5" * 64,
                "lock_sha256": "6" * 64,
            },
        }

    monkeypatch.setattr(v7, "validate_v7_contract", fake_validate_contract)
    monkeypatch.setattr(v7, "validate_v7_checkpoint", fake_validate_checkpoint)
    monkeypatch.setattr(
        v7,
        "validate_fixed_hard32_authorization",
        fake_validate_authorization,
    )

    authorization = evaluator.validate_scene_v7_train32_hard32_authorization(
        receipt_path,
        memory_dir=memory_dir,
    )

    contract_kwargs = calls["contract_kwargs"]
    assert contract_kwargs["contract"] == "scene_v7_train32_overfit"
    assert contract_kwargs["source_lock_file"] == v7.DEFAULT_SOURCE_LOCK
    assert calls["checkpoint_path"] == memory_dir
    assert calls["input_contract"] is input_contract
    assert calls["expected_checkpoint"] is checkpoint
    assert authorization["checkpoint"] is checkpoint
    assert authorization["scope"] == "fixed_hard32_only_no_full170"
    assert authorization["receipt"]["path"] == str(receipt_path.resolve())


@pytest.mark.parametrize(
    ("contract", "status", "message"),
    [
        ("scene_v7_tiny_overfit", "pass", "Train32 receipt, not a Tiny2"),
        ("scene_v7_train32_overfit", "fail", "passed V7 Train32 receipt"),
    ],
)
def test_tiny_or_failed_v7_receipt_cannot_authorize_hard32(
    tmp_path: Path,
    contract: str,
    status: str,
    message: str,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(_receipt_payload(contract=contract, status=status)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        evaluator.validate_scene_v7_train32_hard32_authorization(
            receipt_path,
            memory_dir=tmp_path / "checkpoint-32",
        )


def test_mismatched_v7_checkpoint_cannot_authorize_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            _receipt_payload(
                contract="scene_v7_train32_overfit",
                status="pass",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(v7, "validate_v7_contract", lambda **_: {"rows": 32})
    monkeypatch.setattr(
        v7,
        "validate_v7_checkpoint",
        lambda *_args, **_kwargs: {"adapter_sha256": "a" * 64},
    )

    def reject_mismatch(*_args, **_kwargs):
        raise v7.V7EvaluationContractError("V7 receipt checkpoint differs")

    monkeypatch.setattr(
        v7,
        "validate_fixed_hard32_authorization",
        reject_mismatch,
    )

    with pytest.raises(v7.V7EvaluationContractError, match="checkpoint differs"):
        evaluator.validate_scene_v7_train32_hard32_authorization(
            receipt_path,
            memory_dir=tmp_path / "checkpoint-64",
        )


def test_v7_authorized_hard32_receipt_cannot_unlock_full170(tmp_path: Path) -> None:
    receipt = {
        "schema": evaluator.HARD32_RECEIPT_SCHEMA,
        "status": "pass",
        "objective_interpretation": (
            evaluator.SCENE_V6_IDENTITY_OBJECTIVE_INTERPRETATION
        ),
        "contract": {
            "name": "scene_v6_identity_hard32",
            "rows": 32,
            "conditions": list(evaluator.CONDITIONS),
        },
        "checkpoint": {
            "candidate_lineage": {
                "lineage_kind": evaluator.SCENE_V7_TRAIN32_AUTHORIZATION_KIND,
            }
        },
    }
    receipt["receipt_sha256"] = evaluator.fingerprint_payload_sha256(receipt)
    receipt_path = tmp_path / "hard32_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="does not authorize full170"):
        evaluator.validate_hard32_pass_receipt(
            receipt_path,
            memory_dir=tmp_path / "checkpoint-64",
        )


def test_v7_hard32_receipt_records_v7_objective_and_removes_full170_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256",
        evaluator.sha256_text("[]"),
    )
    monkeypatch.setattr(
        evaluator,
        "file_binding",
        lambda path: {
            "path": str(path),
            "bytes": 1,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(evaluator, "sha256_file", lambda _path: "b" * 64)
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "full170_authorized_for_bound_checkpoint": True,
    }

    receipt = evaluator.build_hard32_receipt(
        output_dir=tmp_path / "eval",
        fingerprint="f" * 64,
        contract={
            "name": "scene_v6_identity_hard32",
            "rows": 32,
            "conditions": list(evaluator.CONDITIONS),
        },
        candidate_lineage={
            "lineage_kind": evaluator.SCENE_V7_TRAIN32_AUTHORIZATION_KIND,
            "authorization": {"scope": "fixed_hard32_only_no_full170"},
        },
        code_fingerprint={"evaluator_sha256": "c" * 64},
        dataset_file=tmp_path / "val.jsonl",
        selection_file=tmp_path / "hard32.json",
        donor_mapping=[],
        gate=gate,
        semantic_evidence={"rows": []},
        base_outcome_evidence={"rows": []},
        memory_dir=tmp_path / "checkpoint-64",
        conditions=list(evaluator.CONDITIONS),
    )

    assert receipt["objective_interpretation"] == (
        evaluator.SCENE_V7_HARD32_OBJECTIVE_INTERPRETATION
    )
    assert receipt["authorization_scope"] == "fixed_hard32_only_no_full170"
    assert receipt["gate"]["full170_authorized_for_bound_checkpoint"] is False
    assert gate["full170_authorized_for_bound_checkpoint"] is True
