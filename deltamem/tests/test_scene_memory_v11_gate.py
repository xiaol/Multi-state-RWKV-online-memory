from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v11_gate as gate


_GENERATION_FIELDS = (
    "canonical_correct_outputs",
    "correct_strict_exact_rows",
    "donor_identity_strict_exact_rows",
    "correct_strict_micro_f1",
)


def _diagnostic(
    *,
    values: dict[str, int | float] | None = None,
    zero_invariant: bool = True,
) -> dict[str, Any]:
    metrics = {
        name: value
        for name, value in gate.GATE_REQUIREMENTS.items()
        if name != "zero_reset_control_is_row_invariant"
    }
    if values is not None:
        metrics.update(values)
    generation = {name: metrics[name] for name in _GENERATION_FIELDS}
    identity = {
        name: value
        for name, value in metrics.items()
        if name not in _GENERATION_FIELDS
    }
    return {
        "contract": gate.v10.GATE_CONTRACT,
        "metrics": {
            "value14_generation": generation,
            "value14_selected_token_identity": {"overall": identity},
        },
        "gates": {"zero_reset_control_is_row_invariant": zero_invariant},
        "status": "synthetic",
        "all_gates_passed": False,
    }


def test_v11_gate_constants_pin_custom_candidate_thresholds() -> None:
    assert gate.CHECKPOINT_STEPS == (1,)
    assert gate.GATE_REQUIREMENTS == {
        "canonical_correct_outputs": 14,
        "correct_strict_exact_rows": 4,
        "donor_identity_strict_exact_rows": 4,
        "correct_strict_micro_f1": 0.3783783783783784,
        "bidirectional_identity_switch_rows": 8,
        "correct_state_beats_donor_state_on_source_token_rows": 14,
        "correct_state_prefers_source_token_rows": 11,
        "donor_state_prefers_donor_token_rows": 11,
        "correct_state_beats_zero_on_source_token_rows": 11,
        "zero_reset_control_is_row_invariant": True,
    }
    assert gate.V10_DIAGNOSTIC_BASELINE == {
        "canonical_correct_outputs": 14,
        "correct_strict_exact_rows": 3,
        "donor_identity_strict_exact_rows": 3,
        "correct_strict_micro_f1": 0.3783783783783784,
        "bidirectional_identity_switch_rows": 8,
        "correct_state_beats_donor_state_on_source_token_rows": 14,
        "correct_state_prefers_source_token_rows": 11,
        "donor_state_prefers_donor_token_rows": 11,
        "correct_state_beats_zero_on_source_token_rows": 11,
    }
    assert gate.GATE_REQUIREMENTS["correct_strict_exact_rows"] == (
        gate.V10_DIAGNOSTIC_BASELINE["correct_strict_exact_rows"] + 1
    )
    assert gate.GATE_REQUIREMENTS["donor_identity_strict_exact_rows"] == (
        gate.V10_DIAGNOSTIC_BASELINE["donor_identity_strict_exact_rows"] + 1
    )


def test_v11_objective_binds_suffix_repair_v8_warm_and_v10_diagnostic() -> None:
    assert gate.V11_OBJECTIVE["training_objective_version"] == (
        gate.launch.OBJECTIVE_VERSION
    )
    assert gate.V11_OBJECTIVE["suffix_repair_mode"] == (
        gate.launch.SUFFIX_REPAIR_MODE
    )
    assert gate.V11_OBJECTIVE["suffix_repair_weight"] == 0.5
    assert gate.V11_OBJECTIVE["suffix_repair_divergence"] == (
        "first_raw_token_divergence_including_length_mismatch_v1"
    )
    assert gate.V11_OBJECTIVE["warm_start_source"] == (
        "pinned_v8_checkpoint56_adapter_only"
    )
    assert gate.V11_OBJECTIVE["v10_role"] == (
        "frozen_diagnostic_only_never_warm_start"
    )
    assert gate.V11_OBJECTIVE["training_continuation"] == (
        "forbidden_one_cycle_only_regardless_of_gate_status"
    )


def test_v11_passing_gate_designates_candidate_but_never_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def build(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _diagnostic()

    monkeypatch.setattr(gate.v10, "build_v10_gate", build)

    result = gate.build_v11_gate(
        records_by_condition={"synthetic": []},
        pairing={"synthetic": True},
        checkpoint_step=1,
    )

    assert calls == [
        {
            "records_by_condition": {"synthetic": []},
            "pairing": {"synthetic": True},
            "checkpoint_step": 1,
            "previous_gate": None,
        }
    ]
    assert result["contract"] == gate.GATE_CONTRACT
    assert result["criterion"] == (
        "train32_value14_suffix_repair_candidate_v1"
    )
    assert result["checkpoint_step"] == 1
    assert result["consumed_pair_presentations"] == 7
    assert result["status"] == "pass"
    assert result["all_gates_passed"] is True
    assert result["candidate_designation"] == "candidate"
    assert result["candidate_authorized"] is True
    assert result["training_continuation_authorized"] is False
    assert result["next_checkpoint_step"] is None
    assert result["hard32_authorized"] is False
    assert result["full170_authorized"] is False
    assert result["test_authorized"] is False
    assert result["other_benchmarks_authorized"] is False
    assert result["comparison"]["v10"] == gate.V10_DIAGNOSTIC_BASELINE
    assert result["comparison"]["v11"] == {
        name: gate.GATE_REQUIREMENTS[name]
        for name in gate.V10_DIAGNOSTIC_BASELINE
    }


@pytest.mark.parametrize(
    "field",
    tuple(
        name
        for name in gate.GATE_REQUIREMENTS
        if name != "zero_reset_control_is_row_invariant"
    ),
)
def test_v11_each_numeric_requirement_is_mandatory(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold = gate.GATE_REQUIREMENTS[field]
    assert isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
    below = threshold - (0.000001 if isinstance(threshold, float) else 1)
    monkeypatch.setattr(
        gate.v10,
        "build_v10_gate",
        lambda **_kwargs: _diagnostic(values={field: below}),
    )

    result = gate.build_v11_gate(
        records_by_condition={},
        pairing={},
        checkpoint_step=1,
    )

    assert result["status"] == "fail"
    assert result["all_gates_passed"] is False
    assert result["candidate_designation"] == "rejected"
    assert result["candidate_authorized"] is False
    assert result["training_continuation_authorized"] is False
    assert result["next_checkpoint_step"] is None
    assert list(result["gates"].values()).count(False) == 1


def test_v11_zero_reset_invariance_is_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.v10,
        "build_v10_gate",
        lambda **_kwargs: _diagnostic(zero_invariant=False),
    )

    result = gate.build_v11_gate(
        records_by_condition={},
        pairing={},
        checkpoint_step=1,
    )

    assert result["status"] == "fail"
    assert result["gates"]["zero_reset_control_is_row_invariant"] is False
    assert result["candidate_authorized"] is False
    assert result["training_continuation_authorized"] is False


def test_v11_gate_rejects_every_checkpoint_except_one() -> None:
    for step in (0, 2, 7):
        with pytest.raises(
            gate.V11EvaluationContractError,
            match="only checkpoint-1",
        ):
            gate.build_v11_gate(
                records_by_condition={},
                pairing={},
                checkpoint_step=step,
            )


def test_v11_continuation_is_forbidden_after_both_pass_and_fail() -> None:
    for status in ("pass", "fail"):
        with pytest.raises(
            gate.V11EvaluationContractError,
            match="neither pass nor fail authorizes continuation",
        ):
            gate.validate_continuation_authorization(
                {"status": status, "candidate_authorized": status == "pass"}
            )


def test_v11_candidate_receipt_never_contains_training_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        gate,
        "_gate_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        gate,
        "_artifact_binding",
        lambda path, **_kwargs: {"path": str(path), "sha256": "a" * 64},
    )
    monkeypatch.setattr(gate, "evaluator_code_binding", lambda: {})
    monkeypatch.setattr(gate.v10.v9, "utc_now", lambda: "2026-07-30T00:00:00Z")
    checkpoint = {
        "memory_dir": "/ssd/v11/checkpoint-1",
        "global_step": 1,
        "training_provenance": {"schema": "provenance"},
    }
    inputs = {
        "artifacts": {"train32": {"sha256": "b" * 64}},
        "v10_diagnostic_baseline": {"role": "diagnostic_only"},
    }
    passed_gate = {
        "status": "pass",
        "all_gates_passed": True,
        "candidate_authorized": True,
        "training_continuation_authorized": False,
    }

    receipt = gate.build_gate_receipt(
        output_dir=tmp_path,
        fingerprint="f" * 64,
        input_contract=inputs,
        checkpoint=checkpoint,
        gate=passed_gate,
        ssd_root=tmp_path,
    )

    assert receipt["status"] == "pass"
    assert receipt["candidate_designation"] == {
        "kind": gate.CANDIDATE_DESIGNATION_KIND,
        "designated": True,
        "checkpoint_binding": checkpoint,
    }
    assert receipt["training_authorization"] == {
        "authorized": False,
        "next_checkpoint_step": None,
        "policy": "forbidden_one_cycle_only_regardless_of_gate_status",
        "hard32_authorized": False,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    assert receipt["receipt_sha256"] == gate.self_hash_payload(
        receipt,
        hash_field="receipt_sha256",
    )


def test_v11_receipt_replay_rejects_objective_drift_before_live_access() -> None:
    payload = {
        "schema": gate.GATE_RECEIPT_SCHEMA,
        "contract": gate.GATE_CONTRACT,
        "objective": {**gate.V11_OBJECTIVE, "suffix_repair_weight": 0.75},
        "requirements": gate.GATE_REQUIREMENTS,
    }
    payload["receipt_sha256"] = gate.self_hash_payload(
        payload,
        hash_field="receipt_sha256",
    )

    with pytest.raises(
        gate.V11EvaluationContractError,
        match="receipt objective differs",
    ):
        gate.validate_gate_receipt_for_checkpoint(
            payload,
            memory_dir="/must/not/be/opened",
        )


def test_v11_fingerprint_binds_v10_diagnostic_and_candidate_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "validate_base_model_path",
        lambda _path: gate.launch.PINNED_BASE_MODEL,
    )
    monkeypatch.setattr(
        gate.launch,
        "validate_base_model_contract",
        lambda **_kwargs: {
            "path": str(gate.launch.PINNED_BASE_MODEL),
            "weights": {"weights": "pinned"},
            "prompt_artifacts": {"prompts": "pinned"},
        },
    )
    monkeypatch.setattr(
        gate.v10.v9,
        "runtime_package_versions",
        lambda: {"torch": "test"},
    )
    monkeypatch.setattr(gate, "evaluator_code_binding", lambda: {"gate": "code"})
    diagnostic = {
        "role": "frozen_diagnostic_only_never_warm_start",
        "checkpoint": "/ssd/v10/checkpoint-1",
        "metrics": dict(gate.V10_DIAGNOSTIC_BASELINE),
        "base_model_identity": {
            "path": str(gate.launch.PINNED_BASE_MODEL),
            "weights": {"weights": "pinned"},
            "prompt_artifacts": {"prompts": "pinned"},
        },
    }
    inputs = {
        "artifacts": {"train32": {"sha256": "a" * 64}},
        "v10_diagnostic_baseline": diagnostic,
    }
    checkpoint = {
        "memory_dir": "/ssd/v11/checkpoint-1",
        "global_step": 1,
        "architecture": {"target_layers": list(range(42))},
        "training_provenance": {"schema": "provenance"},
    }

    payload = gate.build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
    )

    assert payload["evaluation_scope"] == "Train32_records_Value14_gate_only"
    assert payload["v10_diagnostic_baseline"] == diagnostic
    assert payload["checkpoint"] == checkpoint
    assert payload["objective"] == gate.V11_OBJECTIVE
    assert payload["requirements"] == gate.GATE_REQUIREMENTS
    assert payload["hard32_access"] == gate.HARD32_ACCESS_POLICY
    assert payload["other_benchmarks_authorized"] is False
    drifted = copy.deepcopy(payload)
    drifted["training_provenance"]["git_commit"] = "f" * 64
    assert gate.v10.v9.fingerprint_payload_sha256(drifted) != (
        gate.v10.v9.fingerprint_payload_sha256(payload)
    )


def test_v11_manifest_rejects_payload_or_hard32_drift() -> None:
    payload = {
        "training_sources": {"train32": {"sha256": "a" * 64}},
        "v10_diagnostic_baseline": {"checkpoint": "/ssd/v10/checkpoint-1"},
        "requirements": dict(gate.GATE_REQUIREMENTS),
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    fingerprint = gate.v10.v9.fingerprint_payload_sha256(payload)
    manifest = {
        "schema": gate.GATE_MANIFEST_SCHEMA,
        "created_at": "2026-07-30T00:00:00Z",
        "fingerprint": fingerprint,
        "fingerprint_payload": payload,
        "hard32_access": gate.HARD32_ACCESS_POLICY,
    }
    assert gate.validate_existing_manifest(
        manifest,
        expected_fingerprint=fingerprint,
        expected_fingerprint_payload=payload,
    ) == manifest

    drifted = copy.deepcopy(manifest)
    drifted["fingerprint_payload"]["v10_diagnostic_baseline"] = {
        "checkpoint": "/ssd/v10/other"
    }
    with pytest.raises(
        gate.V11EvaluationContractError,
        match="fingerprint payload differs",
    ):
        gate.validate_existing_manifest(
            drifted,
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=payload,
        )

    drifted = copy.deepcopy(manifest)
    drifted["hard32_access"] = "authorized"
    with pytest.raises(gate.V11EvaluationContractError, match="Hard32"):
        gate.validate_existing_manifest(
            drifted,
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=payload,
        )


def test_v11_paths_reject_hard32_before_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = gate.launch.v11_gates_root_for(tmp_path) / "hard32-copy" / "gate"
    original_resolve = Path.resolve

    def guarded_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if any("hard32" in part.lower() for part in path.parts):
            raise AssertionError("protected gate path was resolved")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(
        gate.V11EvaluationContractError,
        match="locked V11 gates root",
    ):
        gate._gate_path(
            protected,
            description="protected gate",
            ssd_root=tmp_path,
        )


def test_v11_gate_paths_are_exactly_scoped_and_hard32_never_authorized(
    tmp_path: Path,
) -> None:
    run_root = gate.launch.v11_run_root_for(tmp_path)
    gate_root = gate.launch.v11_gates_root_for(tmp_path)
    assert gate._run_path(
        run_root / "run" / "trainer" / "checkpoint-1",
        description="checkpoint",
        ssd_root=tmp_path,
    ) == run_root / "run" / "trainer" / "checkpoint-1"
    assert gate._gate_path(
        gate_root / "candidate" / "gate_receipt.json",
        description="receipt",
        ssd_root=tmp_path,
    ) == gate_root / "candidate" / "gate_receipt.json"
    with pytest.raises(gate.V11EvaluationContractError, match="locked V11 run root"):
        gate._run_path(
            tmp_path / "other" / "checkpoint-1",
            description="checkpoint",
            ssd_root=tmp_path,
        )
    with pytest.raises(gate.V11EvaluationContractError, match="locked V11 gates root"):
        gate._gate_path(
            run_root / "not-gates" / "gate_receipt.json",
            description="receipt",
            ssd_root=tmp_path,
        )

    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "holdout.jsonl" not in source
    assert gate.HARD32_ACCESS_POLICY == (
        "forbidden_not_resolved_opened_or_hashed"
    )


def test_v11_evaluator_fingerprint_binds_transitive_metric_and_contract_code() -> None:
    bindings = gate.evaluator_code_binding()

    assert {
        "v11_gate",
        "v10_gate_metrics",
        "v9_gate_metrics",
        "v8_gate_metrics",
        "train32_metric_runtime",
        "state_runtime",
        "v11_launch_contract",
        "v11_warm_start",
        "v10_launch_contract",
        "v10_warm_start",
        "v9_launch_contract",
        "v9_warm_start",
    } <= set(bindings)
    assert all(binding["sha256"] for binding in bindings.values())


def test_v11_gate_cli_requires_exact_training_receipt_chain() -> None:
    args = gate._parse_args(
        [
            "--base-model",
            str(gate.launch.PINNED_BASE_MODEL),
            "--memory-dir",
            "/ssd/v11/checkpoint-1",
            "--output-dir",
            "/ssd/v11/gates/candidate",
            "--launch-receipt",
            "/ssd/v11/logs/run.launch.json",
            "--completion-receipt",
            "/ssd/v11/logs/run.completion.json",
        ]
    )
    assert args.launch_receipt.name == "run.launch.json"
    assert args.completion_receipt.name == "run.completion.json"
    with pytest.raises(SystemExit):
        gate._parse_args(
            [
                "--base-model",
                str(gate.launch.PINNED_BASE_MODEL),
                "--memory-dir",
                "/ssd/v11/checkpoint-1",
                "--output-dir",
                "/ssd/v11/gates/candidate",
            ]
        )
