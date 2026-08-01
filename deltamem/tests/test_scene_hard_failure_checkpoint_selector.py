from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    run_scene_state_eval as state_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (
    select_scene_hard_failure_checkpoint as selector,
)


def _source_binding(source_manifest: Path | None = None) -> dict[str, Any]:
    return {
        "source_lock_path": "/train/source_lock.json",
        "source_lock_file_sha256": "1" * 64,
        "source_lock_sha256": "2" * 64,
        "source_manifest_path": str(source_manifest or Path("/train/source_manifest.json")),
        "source_manifest_file_sha256": "3" * 64,
        "source_manifest_sha256": "4" * 64,
        "train_file_sha256": "5" * 64,
        "pair_manifest_file_sha256": "6" * 64,
        "pair_manifest_sha256": "7" * 64,
        "entries_sha256": "8" * 64,
        "protected_evaluation_accessed": False,
    }


def _checkpoint(run_root: Path, step: int) -> dict[str, Any]:
    return {
        "path": str(run_root / "trainer" / f"checkpoint-{step}"),
        "global_step": step,
        "artifacts": {
            "delta_mem_adapter.pt": {
                "path": str(
                    run_root
                    / "trainer"
                    / f"checkpoint-{step}"
                    / "delta_mem_adapter.pt"
                ),
                "bytes": 7,
                "sha256": f"{step:064x}",
            },
            "delta_mem_config.json": {
                "path": str(
                    run_root
                    / "trainer"
                    / f"checkpoint-{step}"
                    / "delta_mem_config.json"
                ),
                "bytes": 3,
                "sha256": f"{step + 1:064x}",
            },
        },
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": {
            suffix: 42
            for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
        },
        "full_trainable_family_coverage": True,
        "checkpoint_audit_receipt_sha256": "9" * 64,
    }


def _gate_report(
    *,
    passed: bool,
    passed_gate_count: int,
    state_f1: float,
    normal_f1: float,
) -> dict[str, Any]:
    return {
        "schema": selector.focused_gate.SCHEMA,
        "status": "diagnostic_pass" if passed else "diagnostic_fail",
        "stage": selector.STAGE,
        "task": selector.TASK,
        "rows": 32,
        "source_indices": list(range(32)),
        "all_gates_passed": passed,
        "gates": {
            f"gate_{index}": {"passed": index < passed_gate_count}
            for index in range(5)
        },
        "condition_scores": {
            "state_only": {"primary_metric": state_f1},
            "normal_full": {"primary_metric": normal_f1},
        },
    }


def _evidence(
    run_root: Path,
    *,
    step: int,
    passed: bool,
    passed_gate_count: int = 0,
    state_f1: float = 0.0,
    normal_f1: float = 0.0,
) -> selector.EndpointEvidence:
    report = _gate_report(
        passed=passed,
        passed_gate_count=passed_gate_count,
        state_f1=state_f1,
        normal_f1=normal_f1,
    )
    return selector.EndpointEvidence(
        step=step,
        checkpoint=_checkpoint(run_root, step),
        evaluation={"fingerprint": f"{step + 2:064x}"},
        gate={
            "focused_gate_path": str(
                run_root / "selection" / f"checkpoint-{step}" / selector.GATE_FILENAME
            ),
            "file_sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
            "evaluation_fingerprint": f"{step + 2:064x}",
        },
        report=report,
        fallback_rank=selector._fallback_rank(report, step=step),
    )


def _endpoint_specs(run_root: Path) -> list[selector.EndpointSpec]:
    return [
        selector.EndpointSpec(
            step,
            run_root / "selection" / f"checkpoint-{step}",
            run_root
            / "selection"
            / f"checkpoint-{step}"
            / selector.GATE_FILENAME,
        )
        for step in selector.ENDPOINT_STEPS
    ]


def test_selector_stops_at_first_passing_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[int] = []
    monkeypatch.setattr(selector, "validate_source_binding", _source_binding)

    def validate_checkpoint(
        root: Path, *, step: int, source: dict[str, Any]
    ) -> dict[str, Any]:
        assert root == run_root
        assert source["protected_evaluation_accessed"] is False
        calls.append(step)
        return _checkpoint(run_root, step)

    def load_endpoint(
        spec: selector.EndpointSpec,
        *,
        checkpoint: dict[str, Any],
        source: dict[str, Any],
    ) -> selector.EndpointEvidence:
        assert checkpoint["global_step"] == spec.step
        return _evidence(
            run_root,
            step=spec.step,
            passed=spec.step == 32,
            passed_gate_count=5 if spec.step == 32 else 3,
            state_f1=1.0 if spec.step == 32 else 0.25,
            normal_f1=1.0 if spec.step == 32 else 0.25,
        )

    monkeypatch.setattr(selector, "validate_checkpoint", validate_checkpoint)
    monkeypatch.setattr(selector, "load_endpoint_evidence", load_endpoint)

    receipt = selector.select_checkpoint(
        run_root=run_root,
        endpoint_specs=_endpoint_specs(run_root),
    )

    assert calls == [16, 32]
    assert receipt["status"] == "pass"
    assert receipt["evaluated_endpoint_steps"] == [16, 32]
    assert receipt["selected_checkpoint"]["global_step"] == 32
    assert receipt["authorization"] == {
        "hard32_authorized": True,
        "selected_checkpoint_only": True,
        "full_validation": False,
        "test": False,
        "other_benchmarks": False,
    }
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")
    assert recorded == selector.canonical_sha256(unsigned)


def test_selector_uses_deterministic_unauthorized_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(selector, "validate_source_binding", _source_binding)
    monkeypatch.setattr(
        selector,
        "validate_checkpoint",
        lambda root, *, step, source: _checkpoint(root, step),
    )
    ranks = {
        16: (3, 0.4, 0.5),
        32: (4, 0.3, 0.8),
        48: (4, 0.7, 0.2),
        64: (4, 0.7, 0.2),
    }
    monkeypatch.setattr(
        selector,
        "load_endpoint_evidence",
        lambda spec, *, checkpoint, source: _evidence(
            run_root,
            step=spec.step,
            passed=False,
            passed_gate_count=ranks[spec.step][0],
            state_f1=ranks[spec.step][1],
            normal_f1=ranks[spec.step][2],
        ),
    )

    receipt = selector.select_checkpoint(
        run_root=run_root,
        endpoint_specs=_endpoint_specs(run_root),
    )

    assert receipt["status"] == "fail"
    assert receipt["evaluated_endpoint_steps"] == [16, 32, 48, 64]
    assert receipt["selected_checkpoint"]["global_step"] == 48
    assert receipt["selection_reason"] == (
        "diagnostic_fallback_no_train32_endpoint_passed"
    )
    assert receipt["authorization"]["hard32_authorized"] is False
    assert receipt["hard32_accessed"] is False


@pytest.mark.parametrize(
    "steps",
    [
        (16, 32, 48),
        (16, 32, 48, 65),
        (16, 48, 32, 64),
        (16, 32, 32, 64),
    ],
)
def test_preflight_requires_exact_endpoint_schedule(
    tmp_path: Path,
    steps: tuple[int, ...],
) -> None:
    specs = [
        selector.EndpointSpec(
            step,
            tmp_path / f"train32-{step}",
            tmp_path / f"train32-{step}" / selector.GATE_FILENAME,
        )
        for step in steps
    ]
    with pytest.raises(selector.SelectionError, match="exactly 16,32,48,64"):
        selector.preflight_endpoint_specs(tmp_path / "run", specs)


@pytest.mark.parametrize(
    "protected",
    [
        Path("hard32") / "checkpoint-16",
        Path("holdout") / "checkpoint-16",
        Path("validation") / "checkpoint-16",
        Path("test.jsonl"),
        Path("full170-results"),
    ],
)
def test_preflight_rejects_held_out_paths_before_access(
    tmp_path: Path,
    protected: Path,
) -> None:
    specs = _endpoint_specs(tmp_path / "run")
    specs[0] = selector.EndpointSpec(
        16,
        tmp_path / protected,
        tmp_path / protected / selector.GATE_FILENAME,
    )
    with pytest.raises(selector.SelectionError, match="forbids protected path"):
        selector.preflight_endpoint_specs(tmp_path / "run", specs)


def test_output_path_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "receipt.json"
    linked.symlink_to(target)

    with pytest.raises(selector.SelectionError, match="symlink output"):
        selector.require_output_path(linked, description="selection receipt")


def _full_coverage_audit(step: int = 16) -> dict[str, Any]:
    coverage = {
        suffix: 42
        for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    return {
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": coverage,
        "full_trainable_family_coverage": True,
        "optimizer_contains_only_declared_trainable_adapter_state_count": True,
        "base_model_parameter_values_not_materialized_in_adapter_checkpoint": True,
        "nontrainable_adapter_tensors_unchanged": True,
        "optimizer_update": {
            "optimizer_parameter_state_count": 1134,
            "declared_trainable_adapter_tensor_count": 1134,
            "all_optimizer_parameter_states_at_checkpoint_step": True,
        },
        "adapter_change": {
            "changed_trainable_tensor_count": 1134,
            "changed_nontrainable_tensor_count": 0,
            "trainable_tensor_family_count": 27,
            "target_layer_count": 42,
            "expected_trainable_tensor_count": 1134,
            "trainable_family_layer_coverage": coverage,
            "missing_trainable_family_layers": {},
            "full_trainable_family_coverage": True,
            "full_trainable_family_coverage_required": step == 64,
        },
    }


def test_selector_requires_exact_27_family_by_42_layer_audit() -> None:
    audit = _full_coverage_audit()

    coverage = selector.validate_full_trainable_family_coverage(audit, step=16)

    assert coverage == {
        suffix: 42
        for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    assert len(coverage) == 27


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda audit: audit.update(full_trainable_family_coverage=False),
            "top-level audit",
        ),
        (
            lambda audit: audit["trainable_family_layer_coverage"].update(
                {selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES[0]: 41}
            ),
            "top-level audit",
        ),
        (
            lambda audit: audit["adapter_change"].update(
                changed_trainable_tensor_count=1133
            ),
            "adapter-change evidence",
        ),
        (
            lambda audit: audit["adapter_change"].update(
                missing_trainable_family_layers={"delta_q_proj": [41]}
            ),
            "adapter-change evidence",
        ),
        (
            lambda audit: audit["optimizer_update"].update(
                optimizer_parameter_state_count=1133
            ),
            "optimizer coverage",
        ),
    ],
)
def test_selector_rejects_incomplete_27_by_42_audit(
    mutation: Any,
    message: str,
) -> None:
    audit = _full_coverage_audit()
    mutation(audit)

    with pytest.raises(selector.SelectionError, match=message):
        selector.validate_full_trainable_family_coverage(audit, step=16)


def _write_checkpoint_fixture(
    run_root: Path,
    *,
    step: int,
    source_lock_sha256: str,
    full_coverage: bool = True,
) -> Path:
    checkpoint = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in selector.REQUIRED_CHECKPOINT_ARTIFACTS:
        path = checkpoint / name
        if name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_bytes(b"artifact")
    audit = _full_coverage_audit(step)
    if not full_coverage:
        audit["full_trainable_family_coverage"] = False
    audit.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "run_root": str(run_root),
            "checkpoint": str(checkpoint),
            "checkpoint_optimizer_step": step,
            "run_mode": selector.train_contract.PRODUCTION_RUN_MODE,
            "objective_version": selector.train_contract.OBJECTIVE_VERSION,
            "source_lock_sha256": source_lock_sha256,
            "row_audit_complete": True,
        }
    )
    audit["receipt_sha256"] = selector.canonical_sha256(audit)
    (checkpoint / selector.run_audit.AUDIT_FILENAME).write_text(
        json.dumps(audit, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


def test_validate_checkpoint_binds_v2_audit_and_full_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    source_lock_sha256 = "2" * 64
    checkpoint = _write_checkpoint_fixture(
        run_root,
        step=16,
        source_lock_sha256=source_lock_sha256,
    )
    monkeypatch.setattr(selector.run_audit, "_validate_protocol", lambda *a, **k: None)
    monkeypatch.setattr(
        selector.run_audit, "_validate_trainer_state", lambda *a, **k: None
    )
    monkeypatch.setattr(selector.run_audit, "_validate_row_audit", lambda *a, **k: None)

    validated = selector.validate_checkpoint(
        run_root,
        step=16,
        source={"source_lock_sha256": source_lock_sha256},
    )

    assert validated["path"] == str(checkpoint)
    assert validated["trainable_tensor_family_count"] == 27
    assert validated["target_layer_count"] == 42
    assert validated["full_trainable_family_coverage"] is True
    audit_binding = validated["artifacts"][selector.run_audit.AUDIT_FILENAME]
    assert audit_binding["path"] == str(
        checkpoint / selector.run_audit.AUDIT_FILENAME
    )
    assert audit_binding["sha256"] == selector.sha256_file(
        checkpoint / selector.run_audit.AUDIT_FILENAME
    )


def test_validate_checkpoint_rejects_v2_audit_without_full_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    source_lock_sha256 = "2" * 64
    _write_checkpoint_fixture(
        run_root,
        step=16,
        source_lock_sha256=source_lock_sha256,
        full_coverage=False,
    )
    monkeypatch.setattr(selector.run_audit, "_validate_protocol", lambda *a, **k: None)
    monkeypatch.setattr(
        selector.run_audit, "_validate_trainer_state", lambda *a, **k: None
    )
    monkeypatch.setattr(selector.run_audit, "_validate_row_audit", lambda *a, **k: None)

    with pytest.raises(selector.SelectionError, match="top-level audit"):
        selector.validate_checkpoint(
            run_root,
            step=16,
            source={"source_lock_sha256": source_lock_sha256},
        )


def _evaluation_contract(
    *,
    checkpoint: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": state_eval.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
        "task": selector.TASK,
        "split": "train",
        "rows": 32,
        "conditions": list(state_eval.SCENE_FOCUSED_CONDITIONS),
        "checkpoint": {
            "memory_dir": checkpoint["path"],
            "adapter_sha256": checkpoint["artifacts"]["delta_mem_adapter.pt"][
                "sha256"
            ],
            "config_sha256": checkpoint["artifacts"]["delta_mem_config.json"][
                "sha256"
            ],
        },
        "train_source": {
            "source_manifest": {
                "path": source["source_manifest_path"],
                "file_sha256": source["source_manifest_file_sha256"],
                "manifest_sha256": source["source_manifest_sha256"],
            },
            "dataset": {
                "sha256": source["train_file_sha256"],
                "split": "train",
                "rows": 32,
            },
            "pair_manifest": {
                "file_sha256": source["pair_manifest_file_sha256"],
                "manifest_sha256": source["pair_manifest_sha256"],
                "entries_sha256": source["entries_sha256"],
            },
            "protected_evaluation_accessed": False,
        },
    }


def test_endpoint_evidence_recomputes_and_binds_gate_and_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    results_dir = tmp_path / "train32-results"
    results_dir.mkdir()
    source_manifest = tmp_path / "train-source.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    source = _source_binding(source_manifest)
    checkpoint = _checkpoint(run_root, 16)
    contract = _evaluation_contract(checkpoint=checkpoint, source=source)
    fingerprint = "f" * 64
    report = _gate_report(
        passed=True,
        passed_gate_count=5,
        state_f1=1.0,
        normal_f1=1.0,
    )
    report["input"] = {
        "results_dir": str(results_dir),
        "evaluation_fingerprint": fingerprint,
        "evaluation_contract": contract,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(
            {"fingerprint": fingerprint, "evaluation_contract": contract},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (results_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    for condition in state_eval.SCENE_FOCUSED_CONDITIONS:
        (results_dir / f"{condition}.jsonl").write_text("{}\n", encoding="utf-8")
    gate_file = results_dir / selector.GATE_FILENAME
    gate_file.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        selector.focused_gate,
        "analyze_results_dir",
        lambda path, *, stage: report,
    )

    evidence = selector.load_endpoint_evidence(
        selector.EndpointSpec(16, results_dir, gate_file),
        checkpoint=checkpoint,
        source=source,
    )

    assert evidence.passed is True
    assert evidence.evaluation["fingerprint"] == fingerprint
    assert evidence.gate["file_sha256"] == selector.sha256_file(gate_file)
    assert set(evidence.evaluation["artifacts"]) == {
        "manifest.json",
        "summary.json",
        *(f"{condition}.jsonl" for condition in state_eval.SCENE_FOCUSED_CONDITIONS),
    }

    tampered = dict(report)
    tampered["status"] = "diagnostic_fail"
    gate_file.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="differs from recomputed"):
        selector.load_endpoint_evidence(
            selector.EndpointSpec(16, results_dir, gate_file),
            checkpoint=checkpoint,
            source=source,
        )


def test_passing_receipt_is_accepted_by_hard32_authorization_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_dir = tmp_path / "run" / "trainer" / "checkpoint-16"
    memory_dir.mkdir(parents=True)
    adapter = memory_dir / "delta_mem_adapter.pt"
    config = memory_dir / "delta_mem_config.json"
    audit_file = memory_dir / selector.run_audit.AUDIT_FILENAME
    adapter.write_bytes(b"adapter")
    config.write_text("{}\n", encoding="utf-8")
    audit_payload = _full_coverage_audit()
    audit_payload.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "checkpoint_optimizer_step": 16,
        }
    )
    audit_file.write_text(json.dumps(audit_payload, sort_keys=True), encoding="utf-8")
    checkpoint = {
        "path": str(memory_dir),
        "global_step": 16,
        "artifacts": {
            "delta_mem_adapter.pt": selector.artifact_binding(
                adapter, description="adapter"
            ),
            "delta_mem_config.json": selector.artifact_binding(
                config, description="config"
            ),
            selector.run_audit.AUDIT_FILENAME: selector.artifact_binding(
                audit_file,
                description="hard-failure checkpoint audit",
            ),
        },
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": audit_payload[
            "trainable_family_layer_coverage"
        ],
        "full_trainable_family_coverage": True,
        "checkpoint_audit_receipt_sha256": "9" * 64,
    }
    source_manifest = tmp_path / "train-source.json"
    source_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    source = _source_binding(source_manifest)
    source["source_manifest_file_sha256"] = selector.sha256_file(source_manifest)
    contract = _evaluation_contract(checkpoint=checkpoint, source=source)
    fingerprint = "f" * 64
    gate = _gate_report(
        passed=True,
        passed_gate_count=5,
        state_f1=1.0,
        normal_f1=1.0,
    )
    gate["input"] = {
        "results_dir": str(tmp_path / "train32-results"),
        "evaluation_fingerprint": fingerprint,
        "evaluation_contract": contract,
    }
    results_dir = tmp_path / "train32-results"
    results_dir.mkdir()
    gate_file = results_dir / selector.GATE_FILENAME
    gate_file.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    evidence = selector.EndpointEvidence(
        step=16,
        checkpoint=checkpoint,
        evaluation={"results_dir": str(results_dir), "fingerprint": fingerprint},
        gate={
            "focused_gate_path": str(gate_file),
            "file_sha256": selector.sha256_file(gate_file),
            "canonical_sha256": selector.canonical_sha256(gate),
            "evaluation_fingerprint": fingerprint,
        },
        report=gate,
        fallback_rank=selector._fallback_rank(gate, step=16),
    )
    receipt = selector.build_selection_receipt(
        source=source,
        evaluated=[evidence],
        selected=evidence,
        passed=True,
        created_at="2026-08-01T00:00:00+00:00",
    )
    receipt_path = tmp_path / "train-selection-receipt.json"
    selector.atomic_write_json(receipt_path, receipt)

    for name, value in {
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256": source[
            "source_manifest_file_sha256"
        ],
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256": source[
            "source_manifest_sha256"
        ],
        "SCENE_HARD_FAILURE_TRAIN_FILE_SHA256": source["train_file_sha256"],
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256": source[
            "pair_manifest_file_sha256"
        ],
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256": source[
            "pair_manifest_sha256"
        ],
        "SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256": source["entries_sha256"],
    }.items():
        monkeypatch.setattr(state_eval, name, value)

    authorization = state_eval.validate_focused_train_selection_authorization(
        receipt_path,
        memory_dir=memory_dir,
    )

    assert authorization["hard32_authorized"] is True
    assert authorization["selected_checkpoint"]["global_step"] == 16
    assert authorization["evaluation_fingerprint"] == fingerprint
    assert receipt["selected_checkpoint"]["artifacts"][
        selector.run_audit.AUDIT_FILENAME
    ]["sha256"] == selector.sha256_file(audit_file)


def test_source_binding_fails_closed_on_evaluator_lock_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_lock = tmp_path / "source-lock.json"
    source_manifest = tmp_path / "source-manifest.json"
    train_file = tmp_path / "train.jsonl"
    for path in (source_lock, source_manifest, train_file):
        path.write_text("{}\n", encoding="utf-8")
    source = _source_binding(source_manifest)
    validated_source = {
        "source_manifest": {
            "file_sha256": source["source_manifest_file_sha256"],
            "manifest_sha256": source["source_manifest_sha256"],
        },
        "dataset": {"sha256": source["train_file_sha256"]},
        "pair_manifest": {
            "file_sha256": source["pair_manifest_file_sha256"],
            "manifest_sha256": source["pair_manifest_sha256"],
            "entries_sha256": source["entries_sha256"],
        },
    }
    lock = {
        "lock_sha256": source["source_lock_sha256"],
        "training_artifacts": {
            "source_manifest.json": {
                "sha256": source["source_manifest_file_sha256"]
            },
            "train.jsonl": {"sha256": source["train_file_sha256"]},
            "pair_manifest.json": {
                "sha256": source["pair_manifest_file_sha256"]
            },
        },
        "protected_evaluation": {
            name: {"included": False, "path": None}
            for name in ("official_validation", "hard32", "official_test")
        },
    }
    monkeypatch.setattr(selector.train_contract, "SOURCE_LOCK", source_lock)
    monkeypatch.setattr(selector.train_contract, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(selector.train_contract, "TRAIN_FILE", train_file)
    monkeypatch.setattr(
        selector.train_contract, "validate_source_lock", lambda path: lock
    )
    monkeypatch.setattr(
        selector.state_eval,
        "validate_focused_train_source_manifest",
        lambda path, *, dataset_file: validated_source,
    )
    expected_names = {
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256": "0" * 64,
        "SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256": source[
            "source_manifest_sha256"
        ],
        "SCENE_HARD_FAILURE_TRAIN_FILE_SHA256": source["train_file_sha256"],
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256": source[
            "pair_manifest_file_sha256"
        ],
        "SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256": source[
            "pair_manifest_sha256"
        ],
        "SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256": source["entries_sha256"],
    }
    for name, value in expected_names.items():
        monkeypatch.setattr(selector.state_eval, name, value)

    with pytest.raises(selector.SelectionError, match="differs between"):
        selector.validate_source_binding()
