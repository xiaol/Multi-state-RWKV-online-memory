from __future__ import annotations

import copy
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
        "path": str(path.absolute()),
        "bytes": len(content),
        "sha256": evaluator.sha256_file(path),
    }


def _protected_benchmark() -> dict[str, Any]:
    return {
        "official_val_path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL.resolve()),
        "official_val_sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        "official_dataset_revision": evaluator.OFFICIAL_SCENE_V4_DATASET_REVISION,
        "selection_path": str(evaluator.HISTORICAL_V6_HARD32_SELECTION.resolve()),
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


def _write_lock(path: Path, payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "lock_sha256"}
    payload["lock_sha256"] = evaluator.fingerprint_payload_sha256(unsigned)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    checkpoint_dir = tmp_path / "trainer" / "checkpoint-3"
    adapter = _artifact(checkpoint_dir / "delta_mem_adapter.pt", b"adapter-3")
    config = _artifact(checkpoint_dir / "delta_mem_config.json", b"{}")
    trainer_state = _artifact(checkpoint_dir / "trainer_state.json", b"state-3")

    launch_path = tmp_path / "run.launch.json"
    completion_path = tmp_path / "run.completion.json"
    launch_artifact = _artifact(launch_path, b"launch")
    completion_artifact = _artifact(completion_path, b"completion")
    launch_binding = {
        "path": launch_artifact["path"],
        "bytes": launch_artifact["bytes"],
        "file_sha256": launch_artifact["sha256"],
        "payload_sha256": "1" * 64,
    }
    completion_binding = {
        "path": completion_artifact["path"],
        "bytes": completion_artifact["bytes"],
        "file_sha256": completion_artifact["sha256"],
        "payload_sha256": "2" * 64,
    }
    provenance = {
        "launch": {
            "artifact": launch_artifact,
            "receipt_sha256": launch_binding["payload_sha256"],
        },
        "completion": {
            "artifact": completion_artifact,
            "receipt_sha256": completion_binding["payload_sha256"],
        },
    }
    checkpoint = {
        "memory_dir": str(checkpoint_dir.absolute()),
        "global_step": 3,
        "consumed_pair_presentations": 21,
        "artifacts": {
            "delta_mem_adapter": adapter,
            "delta_mem_config": config,
            "trainer_state": trainer_state,
        },
        "training_provenance": provenance,
    }

    historical_runtime = {
        "path": str(Path(evaluator.__file__).absolute()),
        "bytes": 100,
        "sha256": "a" * 64,
    }
    live_runtime = {
        "path": str(Path(evaluator.__file__).absolute()),
        "bytes": 200,
        "sha256": "b" * 64,
    }
    common_code = {
        "other_runtime": {
            "path": "/repo/other.py",
            "bytes": 10,
            "sha256": "c" * 64,
        }
    }
    historical_code = {"state_runtime": historical_runtime, **common_code}
    live_code = {"state_runtime": live_runtime, **copy.deepcopy(common_code)}
    diagnostic = {
        "schema": v14.GATE_RECEIPT_SCHEMA,
        "status": "fail",
        "code": historical_code,
    }
    diagnostic["receipt_sha256"] = v14.self_hash_payload(
        diagnostic,
        hash_field="receipt_sha256",
    )
    diagnostic_path = tmp_path / "gate_receipt.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic_binding = {
        "path": str(diagnostic_path.absolute()),
        "bytes": diagnostic_path.stat().st_size,
        "file_sha256": evaluator.sha256_file(diagnostic_path),
        "payload_sha256": diagnostic["receipt_sha256"],
        "evaluation_fingerprint": "3" * 64,
        "status": "fail",
    }

    output_dir = tmp_path / "hard32" / "checkpoint-3"
    selected = {
        "checkpoint": {
            "memory_dir": checkpoint["memory_dir"],
            "global_step": 3,
            "consumed_pair_presentations": 21,
            "delta_mem_adapter": adapter,
            "delta_mem_config": config,
            "trainer_state": trainer_state,
        },
        "diagnostic_receipt": diagnostic_binding,
        "legacy_gate": {
            "status": "fail",
            "all_gates_passed": False,
            "hard32_authorized": False,
            "correct_strict_exact_rows": 6,
            "donor_identity_strict_exact_rows": 6,
            "correct_donor_semantic_switch_rows": 4,
            "role": "diagnostic_only_superseded_by_exact_candidate_lock",
        },
        "launch_receipt": launch_binding,
        "completion_receipt": completion_binding,
        "historical_state_runtime": historical_runtime,
    }
    lock_payload = {
        "schema": evaluator.SCENE_V14_CANDIDATE_LOCK_SCHEMA,
        "authorization_kind": (
            evaluator.SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
        ),
        "selection_policy": evaluator.SCENE_V14_CANDIDATE_SELECTION_POLICY,
        "candidate_count": 1,
        "rejected_checkpoint_steps": [1, 2, 4],
        "hard32_output_dir": str(output_dir.absolute()),
        "selected_candidate": selected,
        "authorization": {
            "scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
            "hard32_authorized": True,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
        "exception_reason": evaluator.SCENE_V14_EVALUATOR_BRIDGE_REASON,
    }
    lock_path = tmp_path / "candidate_lock.json"
    _write_lock(lock_path, lock_payload)
    monkeypatch.setattr(evaluator, "SCENE_V14_CANDIDATE_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        evaluator,
        "SCENE_V14_CANDIDATE_LOCK_FILE_SHA256",
        evaluator.sha256_file(lock_path),
    )

    validated = {
        "status": "fail",
        "receipt_sha256": diagnostic["receipt_sha256"],
        "evaluation_fingerprint": diagnostic_binding["evaluation_fingerprint"],
        "checkpoint": checkpoint,
        "training_provenance": provenance,
        "gate": {
            "status": "fail",
            "all_gates_passed": False,
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
        "candidate_designation": {
            "kind": v14.CANDIDATE_DESIGNATION_KIND,
            "designated": False,
            "checkpoint_binding": checkpoint,
        },
        "training_authorization": {
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }
    inputs = {
        "benchmark_lock": {
            "selected_task": evaluator.TASK_NAME,
            "authorization_scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
            "protected_benchmark": _protected_benchmark(),
        }
    }
    hard32_calls: list[dict[str, Any]] = []
    base_model_calls: list[Path | str] = []

    monkeypatch.setattr(v14, "validate_v14_train_inputs", lambda: inputs)
    monkeypatch.setattr(v14, "evaluator_code_binding", lambda: live_code)

    def validate_base_model(path: Path | str) -> Path:
        base_model_calls.append(path)
        supplied = Path(path).expanduser().absolute()
        if supplied != base_model.absolute():
            raise v14.V14EvaluationContractError("base model differs")
        return base_model.absolute()

    monkeypatch.setattr(v14, "validate_base_model_path", validate_base_model)

    def validate_receipt(receipt, **kwargs):
        assert receipt["code"] == v14.evaluator_code_binding()
        assert receipt["code"] == historical_code
        assert receipt["status"] == "fail"
        assert kwargs["memory_dir"] == checkpoint_dir
        return copy.deepcopy(validated)

    monkeypatch.setattr(v14, "validate_gate_receipt_for_checkpoint", validate_receipt)

    def validate_hard32(**kwargs):
        hard32_calls.append(kwargs)
        return {"official_selection_reproduction": {"rows": 32}}

    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        validate_hard32,
    )
    return {
        "lock_path": lock_path,
        "lock_payload": lock_payload,
        "diagnostic_path": diagnostic_path,
        "launch_path": launch_path,
        "completion_path": completion_path,
        "base_model": base_model,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "validated": validated,
        "live_code": live_code,
        "base_model_calls": base_model_calls,
        "hard32_calls": hard32_calls,
    }


def _authorize(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = {
        "candidate_lock_path": fixture["lock_path"],
        "receipt_path": fixture["diagnostic_path"],
        "launch_receipt": fixture["launch_path"],
        "completion_receipt": fixture["completion_path"],
        "base_model": fixture["base_model"],
        "memory_dir": fixture["checkpoint_dir"],
        "output_dir": fixture["output_dir"],
        "overwrite": False,
        "dataset_file": Path("val.jsonl"),
        "selection_file": Path("holdout_source_indices.json"),
    }
    kwargs.update(overrides)
    return evaluator.validate_scene_v14_candidate_hard32_authorization(**kwargs)


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
        "scene_v14_candidate_authorization": authorization,
    }


def test_tracked_production_lock_designates_only_failed_checkpoint3() -> None:
    lock = evaluator.validate_scene_v14_candidate_lock(
        evaluator.SCENE_V14_CANDIDATE_LOCK_PATH
    )["payload"]

    assert lock["candidate_count"] == 1
    assert lock["rejected_checkpoint_steps"] == [1, 2, 4]
    assert lock["selected_candidate"]["checkpoint"]["global_step"] == 3
    assert lock["selected_candidate"]["diagnostic_receipt"]["status"] == "fail"
    assert lock["selected_candidate"]["legacy_gate"]["hard32_authorized"] is False
    assert lock["authorization"]["hard32_authorized"] is True
    assert lock["authorization"]["full170_authorized"] is False


def test_v14_receipts_and_candidate_lock_are_required_together() -> None:
    receipts = {
        "gate_receipt": Path("gate.json"),
        "candidate_lock": Path("candidate-lock.json"),
        "launch_receipt": Path("launch.json"),
        "completion_receipt": Path("completion.json"),
    }
    evaluator.validate_scene_v14_receipt_scope(
        evaluation_contract=evaluator.SCENE_V14_HARD32_CONTRACT,
        **receipts,
    )

    with pytest.raises(ValueError, match="--scene-v14-candidate-lock"):
        evaluator.validate_scene_v14_receipt_scope(
            evaluation_contract=evaluator.SCENE_V14_HARD32_CONTRACT,
            **{**receipts, "candidate_lock": None},
        )
    with pytest.raises(ValueError, match="accepted only"):
        evaluator.validate_scene_v14_receipt_scope(
            evaluation_contract="generic",
            **receipts,
        )


def test_failed_diagnostic_receipt_is_accepted_only_by_exact_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    authorization = _authorize(fixture)

    assert fixture["base_model_calls"] == [fixture["base_model"]]
    assert len(fixture["hard32_calls"]) == 1
    assert v14.evaluator_code_binding() == fixture["live_code"]
    assert authorization["authorization_kind"] == (
        evaluator.SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND
    )
    assert authorization["hard32_authorized"] is True
    assert authorization["diagnostic_receipt"]["status"] == "fail"
    assert authorization["checkpoint"]["global_step"] == 3
    assert authorization["evaluator_code_bridge"]["substitution_scope"] == (
        "diagnostic_replay.evaluator_code_binding.state_runtime_only"
    )
    assert authorization["evaluator_code_bridge"][
        "other_code_bindings_unchanged"
    ] is True
    assert authorization["full170_authorized"] is False
    assert authorization["test_authorized"] is False
    assert authorization["other_benchmarks_authorized"] is False


def test_checkpoint4_cannot_reuse_checkpoint3_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="rejects every checkpoint"):
        _authorize(fixture, memory_dir=tmp_path / "trainer" / "checkpoint-4")

    assert fixture["hard32_calls"] == []


def test_other_base_model_cannot_reuse_checkpoint3_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    with pytest.raises(v14.V14EvaluationContractError, match="base model differs"):
        _authorize(fixture, base_model=tmp_path / "other-model")

    assert fixture["hard32_calls"] == []


def test_base_model_artifact_drift_rejects_before_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    def reject_artifact_drift(path: Path | str) -> Path:
        raise v14.V14EvaluationContractError("base model artifact differs")

    monkeypatch.setattr(v14, "validate_base_model_path", reject_artifact_drift)

    with pytest.raises(
        v14.V14EvaluationContractError,
        match="base model artifact differs",
    ):
        _authorize(fixture)

    assert fixture["hard32_calls"] == []


@pytest.mark.parametrize(
    "tamper",
    ["lock", "diagnostic", "launch", "completion", "adapter", "config"],
)
def test_candidate_or_provenance_tampering_rejects_before_hard32(
    tamper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if tamper == "lock":
        fixture["lock_path"].write_text("{}\n", encoding="utf-8")
    elif tamper == "diagnostic":
        fixture["diagnostic_path"].write_text("{}\n", encoding="utf-8")
    elif tamper == "launch":
        fixture["launch_path"].write_bytes(b"changed launch")
    elif tamper == "completion":
        fixture["completion_path"].write_bytes(b"changed completion")
    elif tamper == "adapter":
        fixture["validated"]["checkpoint"]["artifacts"][
            "delta_mem_adapter"
        ]["sha256"] = "d" * 64
    else:
        fixture["validated"]["checkpoint"]["artifacts"][
            "delta_mem_config"
        ]["sha256"] = "e" * 64

    with pytest.raises(ValueError):
        _authorize(fixture)

    assert fixture["hard32_calls"] == []


def test_non_evaluator_code_binding_drift_is_not_bridged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["live_code"]["other_runtime"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="non-evaluator code drift"):
        _authorize(fixture)

    assert fixture["hard32_calls"] == []


@pytest.mark.parametrize("failure", ["different_output", "overwrite"])
def test_candidate_lock_allows_only_one_resumable_hard32_output(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    overrides = (
        {"output_dir": tmp_path / "hard32" / "second-run"}
        if failure == "different_output"
        else {"overwrite": True}
    )

    with pytest.raises(ValueError):
        _authorize(fixture, **overrides)

    assert fixture["hard32_calls"] == []


def test_failed_receipt_cannot_claim_broader_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["validated"]["training_authorization"]["full170_authorized"] = True

    with pytest.raises(ValueError, match="no benchmark authorization"):
        _authorize(fixture)

    assert fixture["hard32_calls"] == []


def test_v14_contract_preserves_frozen_rows_conditions_and_causal_thresholds() -> None:
    authorization = {
        "authorization_kind": evaluator.SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "scope": evaluator.SCENE_V14_HARD32_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }

    contract = evaluator.validate_scene_v6_matched_donor_contract(
        **_contract_kwargs(authorization)
    )

    assert contract["rows"] == 32
    assert contract["conditions"] == list(evaluator.CONDITIONS)
    assert contract["gate_requirements"] == evaluator.HARD32_GATE_REQUIREMENTS
    assert contract["scene_v14_candidate_authorization"] is authorization
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
    with pytest.raises(ValueError, match="exact order"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(authorization),
                "conditions": list(reversed(evaluator.CONDITIONS)),
            }
        )
    with pytest.raises(ValueError, match="scope differs"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(
                    {**authorization, "full170_authorized": True}
                )
            }
        )


def test_v14_record_lineage_binds_lock_failed_receipt_and_code_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    authorization = _authorize(fixture)
    candidate = {
        "lineage_kind": evaluator.SCENE_V14_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "authorization": authorization,
    }

    binding = evaluator.build_candidate_lineage_record_binding(candidate)

    assert binding is not None
    assert binding["candidate_lock"] == authorization["candidate_lock"]
    assert binding["base_model"] == str(fixture["base_model"].absolute())
    assert binding["diagnostic_receipt"]["status"] == "fail"
    assert binding["checkpoint"]["global_step"] == 3
    assert binding["evaluator_code_bridge"]["historical_state_runtime"] == (
        authorization["evaluator_code_bridge"]["historical_state_runtime"]
    )
    assert binding["training_provenance"]["launch"]["receipt_sha256"] == (
        "1" * 64
    )
    json.dumps(binding, sort_keys=True)
