from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v14_gate as v14
from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as evaluator


def _artifact(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": str(path.resolve()),
        "bytes": len(content),
        "sha256": evaluator.sha256_file(path),
    }


def _protected_benchmark() -> dict[str, Any]:
    return {
        "official_val_path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL.resolve()),
        "official_val_sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        "official_dataset_revision": evaluator.OFFICIAL_SCENE_V4_DATASET_REVISION,
        "selection_path": str(
            evaluator.HISTORICAL_V6_HARD32_SELECTION.resolve()
        ),
        "selection_sha256": evaluator.HARD32_SELECTION_SHA256,
        "holdout_path": str(evaluator.HISTORICAL_V6_HARD32_HOLDOUT.resolve()),
        "holdout_sha256": evaluator.HARD32_HOLDOUT_SHA256,
        "pair_manifest_sha256": evaluator.HARD32_PAIR_MANIFEST_SHA256,
        "rows": 32,
        "split": "val",
        "authorization_scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def _benchmark_lock(protected: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "selected_task": evaluator.TASK_NAME,
        "authorization_scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "protected_benchmark": (
            _protected_benchmark() if protected is None else protected
        ),
    }


def _validated_gate(tmp_path: Path) -> dict[str, Any]:
    launch = {
        "artifact": _artifact(tmp_path / "run.launch.json", b"launch"),
        "receipt_sha256": "1" * 64,
    }
    completion = {
        "artifact": _artifact(tmp_path / "run.completion.json", b"completion"),
        "receipt_sha256": "2" * 64,
    }
    provenance = {"launch": launch, "completion": completion}
    checkpoint = {
        "memory_dir": str((tmp_path / "trainer" / "checkpoint-2").resolve()),
        "global_step": 2,
        "artifacts": {
            "delta_mem_adapter": _artifact(
                tmp_path / "checkpoint-artifacts" / "delta_mem_adapter.pt",
                b"adapter",
            ),
            "delta_mem_config": _artifact(
                tmp_path / "checkpoint-artifacts" / "delta_mem_config.json",
                b"{}",
            ),
        },
        "training_provenance": provenance,
    }
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    return {
        "status": "pass",
        "receipt_path": str((tmp_path / "gate_receipt.json").resolve()),
        "receipt_file_sha256": "3" * 64,
        "receipt_sha256": "4" * 64,
        "evaluation_fingerprint": "5" * 64,
        "checkpoint": checkpoint,
        "training_provenance": provenance,
        "gate": gate,
        "candidate_designation": {
            "kind": v14.CANDIDATE_DESIGNATION_KIND,
            "designated": True,
            "checkpoint_binding": checkpoint,
        },
        "training_authorization": {
            "hard32_authorized": True,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }


def _v14_authorization() -> dict[str, Any]:
    return {
        "authorization_kind": evaluator.SCENE_V14_VALUE14_AUTHORIZATION_KIND,
        "scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def _contract_kwargs(authorization: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "contract": evaluator.SCENE_V14_HARD32_CONTRACT,
        "row_indices": list(evaluator.HARD32_ROW_INDICES),
        "expected_hashes": dict(evaluator.HARD32_ROW_HASHES),
        "selection_dataset_contract": {
            "path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL),
            "split": "val",
            "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        "conditions": list(evaluator.CONDITIONS),
        "donor_rule": evaluator.DONOR_RULE_LENGTH_MATCHED,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "selection_manifest_sha256": evaluator.HARD32_SELECTION_SHA256,
        "scene_v14_value14_authorization": authorization,
    }


def test_v14_receipts_are_required_together_and_scoped_to_hard32() -> None:
    receipts = {
        "gate_receipt": Path("gate.json"),
        "launch_receipt": Path("launch.json"),
        "completion_receipt": Path("completion.json"),
    }
    evaluator.validate_scene_v14_receipt_scope(
        evaluation_contract=evaluator.SCENE_V14_HARD32_CONTRACT,
        **receipts,
    )

    with pytest.raises(ValueError, match="missing: --scene-v14-completion-receipt"):
        evaluator.validate_scene_v14_receipt_scope(
            evaluation_contract=evaluator.SCENE_V14_HARD32_CONTRACT,
            **{**receipts, "completion_receipt": None},
        )
    with pytest.raises(ValueError, match="accepted only"):
        evaluator.validate_scene_v14_receipt_scope(
            evaluation_contract="generic",
            **receipts,
        )


def test_v14_authorization_validates_both_training_receipts_before_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_gate(tmp_path)
    inputs = {"benchmark_lock": _benchmark_lock()}
    gate_receipt = tmp_path / "gate_receipt.json"
    launch_receipt = tmp_path / "run.launch.json"
    completion_receipt = tmp_path / "run.completion.json"
    checkpoint = tmp_path / "trainer" / "checkpoint-2"
    selection = tmp_path / "holdout_source_indices.json"
    dataset = tmp_path / "val.jsonl"
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(v14, "validate_v14_train_inputs", lambda: inputs)

    def validate_receipt(path, **kwargs):
        calls.append(("receipt", (path, kwargs)))
        return validated

    def validate_hard32(**kwargs):
        calls.append(("hard32", kwargs))
        return {"official_selection_reproduction": {"rows": 32}}

    monkeypatch.setattr(v14, "validate_gate_receipt_for_checkpoint", validate_receipt)
    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        validate_hard32,
    )

    authorization = evaluator.validate_scene_v14_value14_hard32_authorization(
        gate_receipt,
        launch_receipt=launch_receipt,
        completion_receipt=completion_receipt,
        memory_dir=checkpoint,
        dataset_file=dataset,
        selection_file=selection,
    )

    assert [name for name, _ in calls] == ["receipt", "hard32"]
    receipt_kwargs = calls[0][1][1]
    assert receipt_kwargs["memory_dir"] == checkpoint
    assert receipt_kwargs["launch_receipt"] == launch_receipt
    assert receipt_kwargs["completion_receipt"] == completion_receipt
    assert receipt_kwargs["input_contract"] is inputs
    assert authorization["authorization_kind"] == (
        evaluator.SCENE_V14_VALUE14_AUTHORIZATION_KIND
    )
    assert authorization["checkpoint"] == validated["checkpoint"]
    assert authorization["training_provenance"] == validated["training_provenance"]
    assert authorization["frozen_hard32"]["official_selection_reproduction"][
        "rows"
    ] == 32
    assert authorization["full170_authorized"] is False
    assert authorization["test_authorized"] is False
    assert authorization["other_benchmarks_authorized"] is False


@pytest.mark.parametrize("failure", ["failed_gate", "benchmark_drift"])
def test_v14_rejection_never_touches_hard32(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_gate(tmp_path)
    protected = _protected_benchmark()
    if failure == "failed_gate":
        validated["status"] = "fail"
    else:
        protected["holdout_sha256"] = "0" * 64
    inputs = {"benchmark_lock": _benchmark_lock(protected)}
    monkeypatch.setattr(v14, "validate_v14_train_inputs", lambda: inputs)
    monkeypatch.setattr(
        v14,
        "validate_gate_receipt_for_checkpoint",
        lambda *_args, **_kwargs: validated,
    )
    hard32_calls = 0

    def forbidden_hard32(**_kwargs):
        nonlocal hard32_calls
        hard32_calls += 1
        raise AssertionError("Hard32 must remain unopened")

    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        forbidden_hard32,
    )

    with pytest.raises(ValueError):
        evaluator.validate_scene_v14_value14_hard32_authorization(
            tmp_path / "gate.json",
            launch_receipt=tmp_path / "launch.json",
            completion_receipt=tmp_path / "completion.json",
            memory_dir=tmp_path / "checkpoint-2",
            dataset_file=tmp_path / "val.jsonl",
            selection_file=tmp_path / "selection.json",
        )
    assert hard32_calls == 0


def test_v14_contract_accepts_only_frozen_hard32_and_forbids_broader_scope() -> None:
    authorization = _v14_authorization()
    contract = evaluator.validate_scene_v6_matched_donor_contract(
        **_contract_kwargs(authorization)
    )

    assert contract["name"] == evaluator.SCENE_V14_HARD32_CONTRACT
    assert contract["rows"] == 32
    assert contract["scene_v14_value14_authorization"] is authorization
    assert contract["authorization_scope"] == (
        evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE
    )
    assert contract["full170_authorized"] is False
    assert contract["test_authorized"] is False
    assert contract["other_benchmarks_authorized"] is False

    with pytest.raises(ValueError, match="authoritative fixed 32-row"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(authorization),
                "row_indices": list(evaluator.HARD32_ROW_INDICES[:-1]),
            }
        )
    with pytest.raises(ValueError, match="restricted"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(authorization),
                "contract": "scene_v6_matched_donor_validation",
            }
        )
    with pytest.raises(ValueError, match="mixed or broader"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(authorization),
                "scene_v8_train32_authorization": {},
            }
        )


def test_v14_record_lineage_binds_gate_checkpoint_launch_and_completion(
    tmp_path: Path,
) -> None:
    validated = _validated_gate(tmp_path)
    authorization = {
        **_v14_authorization(),
        "receipt": {
            "path": validated["receipt_path"],
            "file_sha256": validated["receipt_file_sha256"],
            "payload_sha256": validated["receipt_sha256"],
            "evaluation_fingerprint": validated["evaluation_fingerprint"],
        },
        "checkpoint": validated["checkpoint"],
        "training_provenance": validated["training_provenance"],
    }
    candidate = {
        "lineage_kind": evaluator.SCENE_V14_VALUE14_AUTHORIZATION_KIND,
        "authorization": authorization,
    }

    binding = evaluator.build_candidate_lineage_record_binding(candidate)

    assert binding is not None
    assert binding["lineage_sha256"] == evaluator.fingerprint_payload_sha256(
        candidate
    )
    assert binding["gate_receipt"] == authorization["receipt"]
    assert binding["checkpoint"]["global_step"] == 2
    assert binding["checkpoint"]["delta_mem_adapter"] == validated["checkpoint"][
        "artifacts"
    ]["delta_mem_adapter"]
    assert binding["training_provenance"]["launch"]["receipt_sha256"] == (
        "1" * 64
    )
    assert binding["training_provenance"]["completion"]["receipt_sha256"] == (
        "2" * 64
    )
    json.dumps(binding, sort_keys=True)
