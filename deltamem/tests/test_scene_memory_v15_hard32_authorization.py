from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as evaluator
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma import (
    select_scene_memory_v15_checkpoint as selector,
)


def _authorization() -> dict[str, Any]:
    return {
        "authorization_kind": evaluator.SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "scope": evaluator.SCENE_V15_HARD32_AUTHORIZATION_SCOPE,
        "hard32_authorized": True,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def _contract_kwargs(authorization: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "contract": evaluator.SCENE_V15_HARD32_CONTRACT,
        "row_indices": list(evaluator.HARD32_ROW_INDICES),
        "expected_hashes": dict(evaluator.HARD32_ROW_HASHES),
        "selection_dataset_contract": {
            "path": str(evaluator.HISTORICAL_V6_OFFICIAL_VAL),
            "split": "val",
            "sha256": evaluator.OFFICIAL_SCENE_V4_VAL_SHA256,
        },
        "conditions": list(evaluator.SCENE_V15_HARD32_CONDITIONS),
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
        "scene_v15_candidate_authorization": authorization,
    }


def _summary(metric: float) -> dict[str, Any]:
    return {
        "format_recovered": {"primary_metric": metric},
        "strict": {
            "metric_name": evaluator.BENCHMARK_SCENE_METRIC_NAME,
            "primary_metric": metric,
            "precision": metric,
            "recall": metric,
            "tp": 1,
            "fp": 1,
            "fn": 1,
            "schema_valid_rate": 1.0,
        },
    }


def _nonexact_record() -> dict[str, Any]:
    return {
        "score_strict": {"schema_valid": True, "fp": 1, "fn": 0},
        "score_recovered": {"schema_recovered": True, "fp": 1, "fn": 0},
    }


def _candidate_lineage() -> dict[str, Any]:
    checkpoint = {
        "path": "/ssd/run/trainer/checkpoint-4",
        "checkpoint_step": 4,
        "artifacts": {
            "delta_mem_adapter.pt": {
                "path": "/ssd/run/trainer/checkpoint-4/delta_mem_adapter.pt",
                "sha256": "a" * 64,
            },
            "delta_mem_config.json": {
                "path": "/ssd/run/trainer/checkpoint-4/delta_mem_config.json",
                "sha256": "b" * 64,
            },
        },
    }
    authorization = {
        **_authorization(),
        "candidate_lock": {"payload_sha256": "c" * 64},
        "selection_receipt": {"payload_sha256": "d" * 64},
        "selection_fingerprint": "e" * 64,
        "selector_manifest": {
            "path": "/ssd/selection/manifest.json",
            "bytes": 1,
            "sha256": "9" * 64,
        },
        "selected_checkpoint_step": 4,
        "post_save_audit": {"sha256": "f" * 64},
        "base_model": "/ssd/base-model",
        "checkpoint": checkpoint,
        "training_provenance": {
            "launch": {
                "path": "/ssd/run/run.launch.json",
                "file_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
            },
            "completion": {
                "path": "/ssd/run/run.completion.json",
                "file_sha256": "3" * 64,
                "receipt_sha256": "4" * 64,
            },
        },
        "authorization_consumption": {
            "path": "/ssd/selection/hard32_authorization_consumed.json",
            "bytes": 1,
            "file_sha256": "5" * 64,
            "claim_sha256": "6" * 64,
            "schema": evaluator.SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA,
        },
    }
    return {
        "lineage_kind": evaluator.SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "authorization": authorization,
    }


def _authorization_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    selection_root = tmp_path / "selection"
    selection_dir = selection_root / "run-v15"
    selection_dir.mkdir(parents=True)
    selection_path = selection_dir / selector.SELECTION_RECEIPT_FILENAME
    selector_manifest_path = selection_dir / selector.SELECTOR_MANIFEST_FILENAME
    lock_path = selection_dir / selector.CANDIDATE_LOCK_FILENAME
    launch_path = tmp_path / "training" / "run.launch.json"
    completion_path = tmp_path / "training" / "run.completion.json"
    for path, content in (
        (selection_path, b"selection"),
        (selector_manifest_path, b"selector-manifest"),
        (lock_path, b"lock"),
        (launch_path, b"launch"),
        (completion_path, b"completion"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    checkpoints = []
    for step in selector.CHECKPOINT_STEPS:
        checkpoint_path = tmp_path / "training" / "trainer" / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True)
        checkpoints.append(
            {
                "path": str(checkpoint_path.absolute()),
                "checkpoint_step": step,
                "artifacts": {
                    "delta_mem_adapter.pt": {"sha256": f"{step}" * 64},
                    "delta_mem_config.json": {"sha256": "a" * 64},
                },
            }
        )
    selected_checkpoint = checkpoints[-1]
    launch_binding = {"path": str(launch_path.absolute()), "sha256": "b" * 64}
    completion_binding = {
        "path": str(completion_path.absolute()),
        "sha256": "c" * 64,
    }
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    base_model_identity = {"path": str(base_model.absolute())}
    selected_receipt = {
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_step": 4,
        "ranked_checkpoint_steps": [4, 3, 2, 1],
        "candidates": [{"checkpoint": checkpoint} for checkpoint in checkpoints],
        "launch_receipt": launch_binding,
        "completion_receipt": completion_binding,
        "selector_manifest": {
            "path": str(selector_manifest_path.absolute()),
            "bytes": selector_manifest_path.stat().st_size,
            "sha256": "7" * 64,
        },
        "base_model": base_model_identity,
        "validated_receipt_sha256": "d" * 64,
        "fingerprint": "e" * 64,
    }
    output_dir = tmp_path / "hard32" / "run-v15-checkpoint-4"
    protected_dataset = tmp_path / "protected" / "val.jsonl"
    protected_selection = tmp_path / "protected" / "holdout_source_indices.json"
    candidate_lock = {
        "schema": evaluator.SCENE_V15_CANDIDATE_LOCK_SCHEMA,
        "authorization_kind": evaluator.SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND,
        "selection_policy": evaluator.SCENE_V15_CANDIDATE_SELECTION_POLICY,
        "selected_candidate": {
            "checkpoint": selected_checkpoint,
            "post_save_audit": {"sha256": "f" * 64},
            "selector_manifest": {
                "path": str(selector_manifest_path.absolute()),
                "bytes": selector_manifest_path.stat().st_size,
                "sha256": "7" * 64,
            },
            "selector_manifest_fingerprint": "e" * 64,
            "base_model": base_model_identity,
        },
        "hard32_output_dir": str(output_dir.absolute()),
        "validated_lock_sha256": "0" * 64,
    }
    monkeypatch.setattr(selector, "SELECTION_ROOT", selection_root)
    monkeypatch.setattr(
        evaluator,
        "HISTORICAL_V6_OFFICIAL_VAL",
        protected_dataset,
    )
    monkeypatch.setattr(
        evaluator,
        "HISTORICAL_V6_HARD32_SELECTION",
        protected_selection,
    )
    monkeypatch.setattr(
        selector,
        "validate_selection_receipt",
        lambda path: copy.deepcopy(selected_receipt),
    )
    monkeypatch.setattr(
        selector,
        "validate_candidate_lock",
        lambda path, **kwargs: copy.deepcopy(candidate_lock),
    )

    artifact_validations: list[tuple[dict[str, Any], Path]] = []

    def validate_artifact_binding(
        binding: dict[str, Any],
        *,
        description: str,
        expected_path: Path,
    ) -> Path:
        del description
        artifact_validations.append((binding, expected_path))
        assert Path(str(binding["path"])).absolute() == expected_path.absolute()
        return expected_path.absolute()

    monkeypatch.setattr(selector, "validate_artifact_binding", validate_artifact_binding)
    monkeypatch.setattr(launch, "validate_data_contract", lambda: {"data": True})
    monkeypatch.setattr(launch, "validate_warm_start_contract", lambda: {"warm": True})
    monkeypatch.setattr(
        launch,
        "validate_base_model_contract",
        lambda path: {"path": str(path.absolute())},
    )
    monkeypatch.setattr(
        launch,
        "validate_checkpoint_contract",
        lambda path, **kwargs: copy.deepcopy(selected_checkpoint),
    )
    monkeypatch.setattr(
        evaluator,
        "resolved_memory_layer_count",
        lambda path, requested: 42 if requested in (None, 42) else requested,
    )
    monkeypatch.setattr(
        evaluator,
        "memory_architecture_contract",
        lambda path: {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
    )
    expected_checkpoints = {
        Path(str(checkpoint["path"])).name: checkpoint for checkpoint in checkpoints
    }
    monkeypatch.setattr(
        launch,
        "validate_launch_receipt",
        lambda path, **kwargs: {
            "path": str(path.absolute()),
            "file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "payload": {
                "checkpoints": {
                    name: str(checkpoint["path"])
                    for name, checkpoint in expected_checkpoints.items()
                }
            },
        },
    )
    monkeypatch.setattr(
        launch,
        "validate_completion_receipt",
        lambda path, **kwargs: {
            "path": str(path.absolute()),
            "file_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
            "payload": {"checkpoints": copy.deepcopy(expected_checkpoints)},
        },
    )
    hard32_calls: list[dict[str, Any]] = []
    claim_observed: list[bool] = []

    def validate_hard32(**kwargs: Any) -> dict[str, Any]:
        marker = evaluator.scene_v15_consumption_marker_path(selection_path)
        claim_observed.append(marker.is_file())
        assert not output_dir.exists()
        hard32_calls.append(kwargs)
        return {"official_selection_reproduction": {"rows": 32}}

    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        validate_hard32,
    )
    return {
        "selection_path": selection_path,
        "lock_path": lock_path,
        "selector_manifest_path": selector_manifest_path,
        "launch_path": launch_path,
        "completion_path": completion_path,
        "checkpoints": checkpoints,
        "selected_checkpoint": selected_checkpoint,
        "output_dir": output_dir,
        "base_model": base_model,
        "protected_dataset": protected_dataset,
        "protected_selection": protected_selection,
        "artifact_validations": artifact_validations,
        "hard32_calls": hard32_calls,
        "claim_observed": claim_observed,
    }


def _authorization_kwargs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_receipt_path": fixture["selection_path"],
        "candidate_lock_path": fixture["lock_path"],
        "launch_receipt": fixture["launch_path"],
        "completion_receipt": fixture["completion_path"],
        "base_model": fixture["base_model"],
        "memory_dir": Path(fixture["selected_checkpoint"]["path"]),
        "output_dir": fixture["output_dir"],
        "overwrite": False,
        "dataset_file": fixture["protected_dataset"],
        "selection_file": fixture["protected_selection"],
        "device": "cuda:0",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "delta_mem_root": evaluator.PROJECT_ROOT,
        "conditions": list(evaluator.SCENE_V15_HARD32_CONDITIONS),
        "donor_rule": evaluator.DONOR_RULE_LENGTH_MATCHED,
        "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS,
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "inline_row_indices": None,
        "preflight_only": False,
        "broader_authorization_supplied": False,
    }


def test_v15_receipts_and_candidate_lock_are_required_together() -> None:
    receipts = {
        "selection_receipt": Path("selection.json"),
        "candidate_lock": Path("candidate-lock.json"),
        "launch_receipt": Path("launch.json"),
        "completion_receipt": Path("completion.json"),
    }
    evaluator.validate_scene_v15_receipt_scope(
        evaluation_contract=evaluator.SCENE_V15_HARD32_CONTRACT,
        **receipts,
    )

    with pytest.raises(ValueError, match="--scene-v15-candidate-lock"):
        evaluator.validate_scene_v15_receipt_scope(
            evaluation_contract=evaluator.SCENE_V15_HARD32_CONTRACT,
            **{**receipts, "candidate_lock": None},
        )
    with pytest.raises(ValueError, match="accepted only"):
        evaluator.validate_scene_v15_receipt_scope(
            evaluation_contract="generic",
            **receipts,
        )


def test_v15_selected_checkpoint_chain_validates_before_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    kwargs = _authorization_kwargs(fixture)

    authorization = evaluator.validate_scene_v15_candidate_hard32_authorization(
        **kwargs
    )

    assert len(fixture["artifact_validations"]) == 3
    assert len(fixture["hard32_calls"]) == 1
    assert fixture["claim_observed"] == [True]
    assert authorization["selected_checkpoint_step"] == 4
    assert authorization["checkpoint"] == fixture["selected_checkpoint"]
    assert authorization["hard32_output_dir"] == str(
        fixture["output_dir"].absolute()
    )
    marker = evaluator.scene_v15_consumption_marker_path(
        fixture["selection_path"]
    )
    assert marker == (
        fixture["selection_path"].parent
        / evaluator.SCENE_V15_HARD32_CONSUMPTION_MARKER_FILENAME
    )
    assert authorization["authorization_consumption"]["path"] == str(marker)
    assert authorization["full170_authorized"] is False
    assert authorization["test_authorized"] is False

    with pytest.raises(ValueError, match="rejects every unselected checkpoint"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(
            **{
                **kwargs,
                "memory_dir": Path(fixture["checkpoints"][2]["path"]),
            }
        )
    assert len(fixture["hard32_calls"]) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"preflight_only": True}, "preflight-only"),
        ({"overwrite": True}, "overwrite"),
        ({"broader_authorization_supplied": True}, "broader"),
        ({"inline_row_indices": "3,6"}, "inline"),
        ({"conditions": ["state_only"]}, "exact order"),
        ({"donor_rule": evaluator.DONOR_RULE_CYCLIC}, "donor_rule"),
        ({"max_new_tokens": 64}, "max_new_tokens"),
        ({"normal_fusion_profile": "native_gate_open"}, "native"),
        ({"device": "cuda:1"}, "CUDA 0"),
        ({"expected_memory_layer_count": 41}, "42-layer"),
        ({"dataset_file": Path("/wrong/val.jsonl")}, "official val"),
        ({"selection_file": Path("/wrong/hard32.json")}, "selection path"),
    ),
)
def test_v15_bad_invocation_is_rejected_before_hard32_or_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    message: str,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    kwargs = {**_authorization_kwargs(fixture), **mutation}

    with pytest.raises(ValueError, match=message):
        evaluator.validate_scene_v15_candidate_hard32_authorization(**kwargs)

    assert fixture["hard32_calls"] == []
    assert not evaluator.scene_v15_consumption_marker_path(
        fixture["selection_path"]
    ).exists()


def test_v15_bad_args_do_not_probe_protected_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    protected = {
        str(fixture["protected_dataset"].absolute()),
        str(fixture["protected_selection"].absolute()),
    }
    accesses: list[tuple[str, str]] = []
    original_stat = os.stat
    original_lstat = os.lstat
    original_open = Path.open

    def guarded_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if str(Path(path).absolute()) in protected:
            accesses.append(("stat", str(path)))
            raise AssertionError("protected Hard32 stat before authorization")
        return original_stat(path, *args, **kwargs)

    def guarded_lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if str(Path(path).absolute()) in protected:
            accesses.append(("lstat", str(path)))
            raise AssertionError("protected Hard32 lstat before authorization")
        return original_lstat(path, *args, **kwargs)

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if str(path.absolute()) in protected:
            accesses.append(("open", str(path)))
            raise AssertionError("protected Hard32 open before authorization")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(Path, "open", guarded_open)

    with pytest.raises(ValueError, match="max_new_tokens"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(
            **{
                **_authorization_kwargs(fixture),
                "max_new_tokens": evaluator.DEFAULT_MAX_NEW_TOKENS - 1,
            }
        )

    assert accesses == []
    assert fixture["hard32_calls"] == []


def test_v15_main_rejects_preflight_before_other_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "parse_args",
        lambda: SimpleNamespace(
            preflight_only=True,
            evaluation_contract=evaluator.SCENE_V15_HARD32_CONTRACT,
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "selected_conditions",
        lambda _raw: pytest.fail("main continued past V15 preflight rejection"),
    )

    with pytest.raises(ValueError, match="preflight-only"):
        evaluator.main()


def test_v15_existing_output_rejects_without_consuming_or_opening_hard32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    fixture["output_dir"].mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot resume"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(
            **_authorization_kwargs(fixture)
        )

    assert fixture["hard32_calls"] == []
    assert not evaluator.scene_v15_consumption_marker_path(
        fixture["selection_path"]
    ).exists()


def test_v15_consumption_marker_is_canonical_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    authorization = evaluator.validate_scene_v15_candidate_hard32_authorization(
        **_authorization_kwargs(fixture)
    )
    marker_path = evaluator.scene_v15_consumption_marker_path(
        fixture["selection_path"]
    )
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    claim_sha256 = unsigned.pop("claim_sha256")

    assert marker_path.parent == fixture["selection_path"].parent
    assert payload["schema"] == (
        evaluator.SCENE_V15_HARD32_CONSUMPTION_MARKER_SCHEMA
    )
    assert claim_sha256 == evaluator.fingerprint_payload_sha256(unsigned)
    assert payload["selection_receipt"] == authorization["selection_receipt"]
    assert payload["candidate_lock"] == authorization["candidate_lock"]
    assert payload["checkpoint"] == fixture["selected_checkpoint"]
    assert payload["hard32_output_dir"] == str(fixture["output_dir"].absolute())
    assert payload["protected_dataset_path"] == str(
        fixture["protected_dataset"].absolute()
    )
    assert payload["protected_selection_path"] == str(
        fixture["protected_selection"].absolute()
    )
    assert payload["retry_authorized"] is False
    assert marker_path.stat().st_mode & 0o777 == 0o400


def test_v15_consumption_marker_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[str] = []
    real_fsync = os.fsync

    def tracked_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        synced.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    selection_receipt = tmp_path / selector.SELECTION_RECEIPT_FILENAME

    evaluator.claim_scene_v15_hard32_authorization(
        selection_receipt_path=selection_receipt,
        payload={"authorization_kind": "unit_test"},
    )

    assert synced == ["file", "directory"]
    assert evaluator.scene_v15_consumption_marker_path(selection_receipt).is_file()


def test_v15_marker_rejects_replay_even_after_output_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    kwargs = _authorization_kwargs(fixture)
    evaluator.validate_scene_v15_candidate_hard32_authorization(**kwargs)
    fixture["output_dir"].mkdir(parents=True)
    fixture["output_dir"].rmdir()

    with pytest.raises(ValueError, match="already consumed"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(**kwargs)

    assert len(fixture["hard32_calls"]) == 1


def test_v15_failure_after_claim_is_not_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    kwargs = _authorization_kwargs(fixture)

    def fail_protected_validation(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError("protected validation failed after claim")

    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        fail_protected_validation,
    )
    with pytest.raises(ValueError, match="after claim"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(**kwargs)

    later_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        evaluator,
        "validate_historical_v6_hard32_artifacts",
        lambda **call: later_calls.append(call) or {},
    )
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(**kwargs)

    assert later_calls == []


def test_v15_checkpoint_topology_rejects_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _authorization_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        evaluator,
        "memory_architecture_contract",
        lambda _path: {
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 8,
            "rwkv_ms_semantics_version": 2,
            "memory_backend": "rwkv_ms",
        },
    )

    with pytest.raises(ValueError, match="rank-4"):
        evaluator.validate_scene_v15_candidate_hard32_authorization(
            **_authorization_kwargs(fixture)
        )

    assert fixture["hard32_calls"] == []
    assert not evaluator.scene_v15_consumption_marker_path(
        fixture["selection_path"]
    ).exists()


def test_v15_contract_authorizes_exactly_three_conditions() -> None:
    authorization = _authorization()

    contract = evaluator.validate_scene_v6_matched_donor_contract(
        **_contract_kwargs(authorization)
    )

    assert contract["conditions"] == list(evaluator.SCENE_V15_HARD32_CONDITIONS)
    assert contract["gate_requirements"] == evaluator.SCENE_V15_HARD32_GATE_POLICY
    assert contract["scene_v15_candidate_authorization"] is authorization
    assert contract["full170_authorized"] is False
    assert contract["test_authorized"] is False
    assert contract["other_benchmarks_authorized"] is False

    with pytest.raises(ValueError, match="exact order"):
        evaluator.validate_scene_v6_matched_donor_contract(
            **{
                **_contract_kwargs(authorization),
                "conditions": list(reversed(evaluator.SCENE_V15_HARD32_CONDITIONS)),
            }
        )


def test_v15_gate_uses_only_strict_f1_against_donor_and_no_write() -> None:
    summaries = {
        "state_only": _summary(0.30),
        "state_only_donor": _summary(0.20),
        "state_only_no_write": _summary(0.10),
    }
    ordered_records = {
        condition: [_nonexact_record()]
        for condition in evaluator.SCENE_V15_HARD32_CONDITIONS
    }
    semantic_evidence = {
        "donor_pair_target_minus_correct": {"mean_gap": -1.0},
        "donor_all_semantic_minus_correct_diagnostic": {"mean_gap": -1.0},
        "zero_all_semantic_minus_correct": {"mean_gap": -1.0},
    }
    contract = {"name": evaluator.SCENE_V15_HARD32_CONTRACT, "rows": 32}

    passed = evaluator.build_scene_v15_hard32_gate(
        summaries=summaries,
        comparisons=evaluator.build_comparisons(summaries),
        semantic_evidence=semantic_evidence,
        ordered_records=ordered_records,
        contract=contract,
    )
    tied = {**summaries, "state_only_donor": _summary(0.30)}
    failed = evaluator.build_scene_v15_hard32_gate(
        summaries=tied,
        comparisons=evaluator.build_comparisons(tied),
        semantic_evidence=semantic_evidence,
        ordered_records=ordered_records,
        contract=contract,
    )

    assert passed["status"] == "pass"
    assert passed["all_gates_passed"] is True
    assert passed["exact_generation_evidence"]["state_only"][
        "strict_parsed_exact_rows"
    ] == 0
    assert passed["semantic_nll_evidence"] == semantic_evidence
    assert passed["full170_authorized_for_bound_checkpoint"] is False
    assert failed["status"] == "fail"
    assert failed["gates"][
        "state_only_strict_f1_greater_than_donor"
    ]["passed"] is False


def test_v15_lineage_and_receipt_bind_only_three_condition_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lineage = _candidate_lineage()
    binding = evaluator.build_candidate_lineage_record_binding(lineage)
    assert binding is not None
    assert binding["selected_checkpoint_step"] == 4
    assert binding["candidate_lock"] == lineage["authorization"]["candidate_lock"]
    assert binding["selection_receipt"] == lineage["authorization"][
        "selection_receipt"
    ]
    assert binding["selector_manifest"] == lineage["authorization"][
        "selector_manifest"
    ]
    assert binding["authorization_consumption"] == lineage["authorization"][
        "authorization_consumption"
    ]
    assert binding["checkpoint"]["delta_mem_adapter"] == lineage[
        "authorization"
    ]["checkpoint"]["artifacts"]["delta_mem_adapter.pt"]

    monkeypatch.setattr(
        evaluator,
        "HARD32_FROZEN_DONOR_MAPPING_ROWS_SHA256",
        evaluator.sha256_text("[]"),
    )
    monkeypatch.setattr(
        evaluator,
        "file_binding",
        lambda path: {"path": str(path), "bytes": 1, "sha256": "5" * 64},
    )
    monkeypatch.setattr(evaluator, "sha256_file", lambda _path: "6" * 64)
    conditions = list(evaluator.SCENE_V15_HARD32_CONDITIONS)
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "full170_authorized_for_bound_checkpoint": False,
    }

    receipt = evaluator.build_hard32_receipt(
        output_dir=tmp_path / "hard32",
        fingerprint="7" * 64,
        contract={
            "name": evaluator.SCENE_V15_HARD32_CONTRACT,
            "rows": 32,
            "conditions": conditions,
        },
        candidate_lineage=lineage,
        code_fingerprint={"evaluator_sha256": "8" * 64},
        dataset_file=tmp_path / "val.jsonl",
        selection_file=tmp_path / "hard32.json",
        donor_mapping=[],
        gate=gate,
        semantic_evidence={"rows": []},
        base_outcome_evidence=None,
        memory_dir=tmp_path / "checkpoint-4",
        conditions=conditions,
    )

    assert list(receipt["outputs"]["conditions"]) == conditions
    assert receipt["upstream_authorization_kind"] == (
        evaluator.SCENE_V15_CANDIDATE_LOCK_AUTHORIZATION_KIND
    )
    assert receipt["objective_interpretation"] == (
        evaluator.SCENE_V15_HARD32_OBJECTIVE_INTERPRETATION
    )
    assert receipt["checkpoint"]["candidate_lineage_binding"] == binding
    assert receipt["full170_authorized"] is False
    assert receipt["other_benchmarks_authorized"] is False

    with pytest.raises(ValueError, match="conditions differ"):
        evaluator.build_hard32_receipt(
            output_dir=tmp_path / "hard32",
            fingerprint="7" * 64,
            contract={"name": evaluator.SCENE_V15_HARD32_CONTRACT, "rows": 32},
            candidate_lineage=lineage,
            code_fingerprint={},
            dataset_file=tmp_path / "val.jsonl",
            selection_file=tmp_path / "hard32.json",
            donor_mapping=[],
            gate=gate,
            semantic_evidence={},
            base_outcome_evidence=None,
            memory_dir=tmp_path / "checkpoint-4",
            conditions=[*conditions, "base_full"],
        )
