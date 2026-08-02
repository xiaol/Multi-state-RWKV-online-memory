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


def _checkpoint(
    run_root: Path,
    step: int,
    *,
    missing: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    missing = {} if missing is None else missing
    coverage = {
        suffix: 42 - len(missing.get(suffix, []))
        for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    changed_count = sum(coverage.values())
    full_coverage = not missing and changed_count == 1134
    adapter_binding = {
        "path": str(
            run_root / "trainer" / f"checkpoint-{step}" / "delta_mem_adapter.pt"
        ),
        "bytes": 7,
        "sha256": f"{step:064x}",
    }
    coverage_evidence = {
        "changed_trainable_tensor_count": changed_count,
        "expected_trainable_tensor_count": 1134,
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": coverage,
        "missing_trainable_family_layers": missing,
        "full_trainable_family_coverage": full_coverage,
        "full_trainable_family_coverage_required": step == 64,
    }
    return {
        "path": str(run_root / "trainer" / f"checkpoint-{step}"),
        "global_step": step,
        "artifacts": {
            "delta_mem_adapter.pt": adapter_binding,
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
        **coverage_evidence,
        "checkpoint_audit_receipt_sha256": "9" * 64,
        "current_adapter_validation": {
            "checkpoint_adapter": adapter_binding,
            "recomputed_adapter_change_canonical_sha256": f"{step + 3:064x}",
            **coverage_evidence,
            "frozen_adapter_tensors_unchanged": True,
        },
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
    checkpoint: dict[str, Any] | None = None,
) -> selector.EndpointEvidence:
    report = _gate_report(
        passed=passed,
        passed_gate_count=passed_gate_count,
        state_f1=state_f1,
        normal_f1=normal_f1,
    )
    return selector.EndpointEvidence(
        step=step,
        checkpoint=_checkpoint(run_root, step) if checkpoint is None else checkpoint,
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
    assert receipt["selected_endpoint_eligibility"] == {
        "benchmark_gate_passed": True,
        "full_coverage": True,
        "selection_eligible": True,
    }
    assert receipt["evaluated_endpoints"][-1]["selection_eligible"] is True
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


def test_selector_continues_gate_passes_until_first_full_coverage_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[int] = []
    partial = {
        16: {"hrm_rwkv7_core.x_a": [1], "hrm_rwkv7_core.x_w": list(range(14))},
        32: {"hrm_rwkv7_core.x_w": [1]},
    }
    monkeypatch.setattr(selector, "validate_source_binding", _source_binding)

    def validate_checkpoint(
        root: Path, *, step: int, source: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append(step)
        return _checkpoint(root, step, missing=partial.get(step))

    def load_endpoint(
        spec: selector.EndpointSpec,
        *,
        checkpoint: dict[str, Any],
        source: dict[str, Any],
    ) -> selector.EndpointEvidence:
        return _evidence(
            run_root,
            step=spec.step,
            passed=True,
            passed_gate_count=5,
            state_f1=1.0,
            normal_f1=1.0,
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(selector, "validate_checkpoint", validate_checkpoint)
    monkeypatch.setattr(selector, "load_endpoint_evidence", load_endpoint)

    receipt = selector.select_checkpoint(
        run_root=run_root,
        endpoint_specs=_endpoint_specs(run_root),
    )

    assert calls == [16, 32, 48]
    assert receipt["status"] == "pass"
    assert receipt["evaluated_endpoint_steps"] == [16, 32, 48]
    assert receipt["selected_checkpoint"]["global_step"] == 48
    assert [
        {
            "benchmark_gate_passed": item["benchmark_gate_passed"],
            "full_coverage": item["full_coverage"],
            "selection_eligible": item["selection_eligible"],
        }
        for item in receipt["evaluated_endpoints"]
    ] == [
        {
            "benchmark_gate_passed": True,
            "full_coverage": False,
            "selection_eligible": False,
        },
        {
            "benchmark_gate_passed": True,
            "full_coverage": False,
            "selection_eligible": False,
        },
        {
            "benchmark_gate_passed": True,
            "full_coverage": True,
            "selection_eligible": True,
        },
    ]
    assert receipt["evaluated_endpoints"][0]["checkpoint"][
        "missing_trainable_family_layers"
    ] == partial[16]
    assert receipt["evaluated_endpoints"][1]["checkpoint"][
        "changed_trainable_tensor_count"
    ] == 1133


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
        "diagnostic_fallback_no_selection_eligible_train32_endpoint"
    )
    assert receipt["authorization"]["hard32_authorized"] is False
    assert receipt["hard32_accessed"] is False


def test_selector_gate_only_pass_remains_unauthorized_after_four_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    missing = {"hrm_rwkv7_core.x_w": [1]}
    monkeypatch.setattr(selector, "validate_source_binding", _source_binding)
    monkeypatch.setattr(
        selector,
        "validate_checkpoint",
        lambda root, *, step, source: _checkpoint(
            root,
            step,
            missing=missing if step == 16 else None,
        ),
    )

    def load_endpoint(
        spec: selector.EndpointSpec,
        *,
        checkpoint: dict[str, Any],
        source: dict[str, Any],
    ) -> selector.EndpointEvidence:
        gate_passed = spec.step == 16
        return _evidence(
            run_root,
            step=spec.step,
            passed=gate_passed,
            passed_gate_count=5 if gate_passed else 0,
            state_f1=1.0 if gate_passed else 0.0,
            normal_f1=1.0 if gate_passed else 0.0,
            checkpoint=checkpoint,
        )

    monkeypatch.setattr(selector, "load_endpoint_evidence", load_endpoint)

    receipt = selector.select_checkpoint(
        run_root=run_root,
        endpoint_specs=_endpoint_specs(run_root),
    )

    assert receipt["status"] == "fail"
    assert receipt["evaluated_endpoint_steps"] == [16, 32, 48, 64]
    assert receipt["selected_checkpoint"]["global_step"] == 16
    assert receipt["selected_endpoint_eligibility"] == {
        "benchmark_gate_passed": True,
        "full_coverage": False,
        "selection_eligible": False,
    }
    assert receipt["authorization"]["hard32_authorized"] is False


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


def _coverage_audit(
    step: int = 16,
    *,
    missing: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    missing = {} if missing is None else missing
    coverage = {
        suffix: 42 - len(missing.get(suffix, []))
        for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    changed_count = sum(coverage.values())
    full_coverage = not missing and changed_count == 1134
    return {
        "trainable_tensor_family_count": 27,
        "target_layer_count": 42,
        "trainable_family_layer_coverage": dict(coverage),
        "full_trainable_family_coverage": full_coverage,
        "optimizer_contains_only_declared_trainable_adapter_state_count": True,
        "base_model_parameter_values_not_materialized_in_adapter_checkpoint": True,
        "nontrainable_adapter_tensors_unchanged": True,
        "optimizer_update": {
            "optimizer_parameter_state_count": 1134,
            "declared_trainable_adapter_tensor_count": 1134,
            "all_optimizer_parameter_states_at_checkpoint_step": True,
        },
        "adapter_change": {
            "changed_trainable_tensor_count": changed_count,
            "changed_nontrainable_tensor_count": 0,
            "trainable_tensor_family_count": 27,
            "target_layer_count": 42,
            "expected_trainable_tensor_count": 1134,
            "trainable_family_layer_coverage": dict(coverage),
            "missing_trainable_family_layers": missing,
            "full_trainable_family_coverage": full_coverage,
            "full_trainable_family_coverage_required": step == 64,
            "frozen_adapter_tensors_unchanged": True,
        },
    }


def _full_coverage_audit(step: int = 16) -> dict[str, Any]:
    return _coverage_audit(step)


def test_selector_requires_exact_27_family_by_42_layer_audit() -> None:
    audit = _full_coverage_audit()

    coverage = selector.validate_full_trainable_family_coverage(audit, step=16)

    assert coverage == {
        suffix: 42
        for suffix in selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    assert len(coverage) == 27


def test_selector_accepts_and_binds_exact_partial_endpoint_coverage() -> None:
    missing = {"hrm_rwkv7_core.x_w": [1]}
    audit = _coverage_audit(32, missing=missing)

    evidence = selector.validate_trainable_family_coverage(audit, step=32)

    assert evidence["changed_trainable_tensor_count"] == 1133
    assert evidence["missing_trainable_family_layers"] == missing
    assert evidence["trainable_family_layer_coverage"][
        "hrm_rwkv7_core.x_w"
    ] == 41
    assert evidence["full_trainable_family_coverage"] is False


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
            "adapter-change",
        ),
        (
            lambda audit: audit["adapter_change"].update(
                missing_trainable_family_layers={"delta_q_proj": [41]}
            ),
            "adapter-change",
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
        selector.validate_trainable_family_coverage(audit, step=16)


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
        audit = _coverage_audit(
            step,
            missing={"hrm_rwkv7_core.x_w": [1]},
        )
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
    audit = json.loads(
        (checkpoint / selector.run_audit.AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    current_adapter_validation = {
        "checkpoint_adapter": selector.artifact_binding(
            checkpoint / "delta_mem_adapter.pt",
            description="checkpoint adapter",
        ),
        **selector.validate_recomputed_adapter_change(
            audit["adapter_change"],
            step=16,
        ),
        "recomputed_adapter_change_canonical_sha256": selector.canonical_sha256(
            audit["adapter_change"]
        ),
        "frozen_adapter_tensors_unchanged": True,
    }
    monkeypatch.setattr(
        selector,
        "validate_current_adapter_change",
        lambda *a, **k: current_adapter_validation,
    )

    validated = selector.validate_checkpoint(
        run_root,
        step=16,
        source={"source_lock_sha256": source_lock_sha256},
    )

    assert validated["path"] == str(checkpoint)
    assert validated["trainable_tensor_family_count"] == 27
    assert validated["target_layer_count"] == 42
    assert validated["full_trainable_family_coverage"] is True
    assert validated["current_adapter_validation"] == current_adapter_validation
    audit_binding = validated["artifacts"][selector.run_audit.AUDIT_FILENAME]
    assert audit_binding["path"] == str(
        checkpoint / selector.run_audit.AUDIT_FILENAME
    )
    assert audit_binding["sha256"] == selector.sha256_file(
        checkpoint / selector.run_audit.AUDIT_FILENAME
    )


def test_validate_checkpoint_accepts_and_binds_partial_nonfinal_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    source_lock_sha256 = "2" * 64
    checkpoint = _write_checkpoint_fixture(
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
    audit = json.loads(
        (checkpoint / selector.run_audit.AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    current_adapter_validation = {
        "checkpoint_adapter": selector.artifact_binding(
            checkpoint / "delta_mem_adapter.pt",
            description="checkpoint adapter",
        ),
        **selector.validate_recomputed_adapter_change(
            audit["adapter_change"],
            step=16,
        ),
        "recomputed_adapter_change_canonical_sha256": selector.canonical_sha256(
            audit["adapter_change"]
        ),
        "frozen_adapter_tensors_unchanged": True,
    }
    monkeypatch.setattr(
        selector,
        "validate_current_adapter_change",
        lambda *a, **k: current_adapter_validation,
    )

    validated = selector.validate_checkpoint(
        run_root,
        step=16,
        source={"source_lock_sha256": source_lock_sha256},
    )

    assert validated["changed_trainable_tensor_count"] == 1133
    assert validated["full_trainable_family_coverage"] is False
    assert validated["missing_trainable_family_layers"] == {
        "hrm_rwkv7_core.x_w": [1]
    }
    assert validated["current_adapter_validation"] == current_adapter_validation


def test_selector_rejects_partial_final_checkpoint_coverage() -> None:
    audit = _coverage_audit(
        64,
        missing={"hrm_rwkv7_core.x_w": [1]},
    )

    with pytest.raises(selector.SelectionError, match="requires complete"):
        selector.validate_trainable_family_coverage(audit, step=64)


def _write_adapter_change_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    initial = run_root / "initial_adapter"
    checkpoint = run_root / "trainer" / "checkpoint-16"
    initial.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (initial / "initial_adapter_manifest.json").write_text(
        json.dumps({"topology": {"fixture": True}}),
        encoding="utf-8",
    )
    (initial / "delta_mem_adapter.pt").write_bytes(b"seed-adapter")
    (checkpoint / "delta_mem_adapter.pt").write_bytes(b"checkpoint-v1")
    return run_root, checkpoint


def _mock_adapter_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    change_for_checkpoint: Any,
) -> None:
    monkeypatch.setattr(
        selector.run_audit,
        "load_finite_adapter",
        lambda path: {"content": path.read_bytes()},
    )
    monkeypatch.setattr(
        selector.run_audit,
        "_validate_initial_adapter_topology",
        lambda topology, initial: ["fixture.trainable"],
    )

    def adapter_change(
        initial: dict[str, Any],
        checkpoint: dict[str, Any],
        *,
        trainable_names: list[str],
        checkpoint_step: int,
        smoke: bool,
    ) -> dict[str, Any]:
        assert initial == {"content": b"seed-adapter"}
        assert trainable_names == ["fixture.trainable"]
        assert checkpoint_step == 16
        assert smoke is False
        return change_for_checkpoint(checkpoint["content"])

    monkeypatch.setattr(selector.run_audit, "adapter_change_record", adapter_change)


def test_current_adapter_change_recomputes_and_binds_stable_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, checkpoint = _write_adapter_change_fixture(tmp_path)
    audited_change = _full_coverage_audit(16)["adapter_change"]
    audited_change["maximum_absolute_delta"] = 0.25
    _mock_adapter_recomputation(
        monkeypatch,
        change_for_checkpoint=lambda content: dict(audited_change),
    )

    evidence = selector.validate_current_adapter_change(
        run_root,
        checkpoint,
        step=16,
        audited_change=audited_change,
    )

    assert evidence["initial_adapter_manifest"]["sha256"] == selector.sha256_file(
        run_root / "initial_adapter" / "initial_adapter_manifest.json"
    )
    assert evidence["initial_adapter"]["sha256"] == selector.sha256_file(
        run_root / "initial_adapter" / "delta_mem_adapter.pt"
    )
    assert evidence["checkpoint_adapter"]["sha256"] == selector.sha256_file(
        checkpoint / "delta_mem_adapter.pt"
    )
    assert evidence["recomputed_adapter_change_canonical_sha256"] == (
        selector.canonical_sha256(audited_change)
    )
    assert evidence["full_trainable_family_coverage"] is True
    assert evidence["frozen_adapter_tensors_unchanged"] is True


def test_current_adapter_change_binds_exact_partial_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, checkpoint = _write_adapter_change_fixture(tmp_path)
    audited_change = _coverage_audit(
        16,
        missing={"hrm_rwkv7_core.x_w": [1]},
    )["adapter_change"]
    _mock_adapter_recomputation(
        monkeypatch,
        change_for_checkpoint=lambda content: dict(audited_change),
    )

    evidence = selector.validate_current_adapter_change(
        run_root,
        checkpoint,
        step=16,
        audited_change=audited_change,
    )

    assert evidence["changed_trainable_tensor_count"] == 1133
    assert evidence["full_trainable_family_coverage"] is False
    assert evidence["missing_trainable_family_layers"] == {
        "hrm_rwkv7_core.x_w": [1]
    }
    assert evidence["frozen_adapter_tensors_unchanged"] is True


def test_current_adapter_change_rejects_mutation_after_stale_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, checkpoint = _write_adapter_change_fixture(tmp_path)
    audited_change = _full_coverage_audit(16)["adapter_change"]
    audited_change["maximum_absolute_delta"] = 0.25
    checkpoint_change = dict(audited_change)
    checkpoint_change["maximum_absolute_delta"] = 0.5
    (checkpoint / "delta_mem_adapter.pt").write_bytes(b"checkpoint-v2")
    _mock_adapter_recomputation(
        monkeypatch,
        change_for_checkpoint=lambda content: (
            dict(audited_change)
            if content == b"checkpoint-v1"
            else dict(checkpoint_change)
        ),
    )

    with pytest.raises(selector.SelectionError, match="differs from audited evidence"):
        selector.validate_current_adapter_change(
            run_root,
            checkpoint,
            step=16,
            audited_change=audited_change,
        )


def test_current_adapter_change_rejects_mutation_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, checkpoint = _write_adapter_change_fixture(tmp_path)
    audited_change = _full_coverage_audit(16)["adapter_change"]
    checkpoint_adapter = checkpoint / "delta_mem_adapter.pt"

    def load_and_mutate(path: Path) -> dict[str, bytes]:
        content = path.read_bytes()
        if path == checkpoint_adapter:
            path.write_bytes(b"checkpoint-mutated-during-load")
        return {"content": content}

    monkeypatch.setattr(selector.run_audit, "load_finite_adapter", load_and_mutate)
    monkeypatch.setattr(
        selector.run_audit,
        "_validate_initial_adapter_topology",
        lambda topology, initial: ["fixture.trainable"],
    )
    monkeypatch.setattr(
        selector.run_audit,
        "adapter_change_record",
        lambda *a, **k: dict(audited_change),
    )

    with pytest.raises(selector.SelectionError, match="changed during current-byte"):
        selector.validate_current_adapter_change(
            run_root,
            checkpoint,
            step=16,
            audited_change=audited_change,
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


def _write_bound_endpoint_evidence(
    results_dir: Path,
    *,
    step: int,
    checkpoint: dict[str, Any],
    source: dict[str, Any],
    report: dict[str, Any],
) -> selector.EndpointEvidence:
    results_dir.mkdir()
    contract = _evaluation_contract(checkpoint=checkpoint, source=source)
    candidate_lineage = {
        "lineage_kind": state_eval.SCENE_HARD_FAILURE_TRAIN_CHECKPOINT_LINEAGE_KIND,
        "memory_dir": checkpoint["path"],
        "global_step": step,
    }
    candidate_lineage_record_binding = {
        "lineage_kind": candidate_lineage["lineage_kind"],
        "lineage_sha256": state_eval.fingerprint_payload_sha256(candidate_lineage),
        "memory_dir": checkpoint["path"],
        "global_step": step,
    }
    fingerprint_payload = {
        "schema_version": 1,
        "evaluation_contract": contract,
        "candidate_lineage": candidate_lineage,
        "candidate_lineage_record_binding": candidate_lineage_record_binding,
    }
    fingerprint = state_eval.fingerprint_payload_sha256(fingerprint_payload)
    report = json.loads(json.dumps(report))
    report["input"] = {
        "results_dir": str(results_dir.resolve()),
        "evaluation_fingerprint": fingerprint,
        "evaluation_contract": contract,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "fingerprint_payload": fingerprint_payload,
                "evaluation_contract": contract,
                "candidate_lineage": candidate_lineage,
                "candidate_lineage_record_binding": (
                    candidate_lineage_record_binding
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (results_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (results_dir / "progress.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "completed": 32 * len(state_eval.SCENE_FOCUSED_CONDITIONS),
                "expected": 32 * len(state_eval.SCENE_FOCUSED_CONDITIONS),
                "complete": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for condition in state_eval.SCENE_FOCUSED_CONDITIONS:
        (results_dir / f"{condition}.jsonl").write_text(
            "{}\n" * state_eval.SCENE_HARD_FAILURE_ROWS,
            encoding="utf-8",
        )
    gate_file = results_dir / selector.GATE_FILENAME
    gate_file.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    artifact_names = (
        "manifest.json",
        "summary.json",
        "progress.json",
        *(f"{condition}.jsonl" for condition in state_eval.SCENE_FOCUSED_CONDITIONS),
    )
    evaluation = {
        "results_dir": str(results_dir.resolve()),
        "fingerprint": fingerprint,
        "contract_canonical_sha256": selector.canonical_sha256(contract),
        "artifacts": {
            name: selector.artifact_binding(
                results_dir / name,
                description=f"checkpoint-{step} evaluation {name}",
            )
            for name in artifact_names
        },
    }
    gate = {
        "focused_gate_path": str(gate_file.resolve()),
        "file_sha256": selector.sha256_file(gate_file),
        "canonical_sha256": selector.canonical_sha256(report),
        "evaluation_fingerprint": fingerprint,
    }
    return selector.EndpointEvidence(
        step=step,
        checkpoint=checkpoint,
        evaluation=evaluation,
        gate=gate,
        report=report,
        fallback_rank=selector._fallback_rank(report, step=step),
    )


def _write_authorization_checkpoint(
    run_root: Path,
    *,
    step: int,
    initial_manifest: Path,
    initial_adapter: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir = run_root / "trainer" / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    adapter = checkpoint_dir / "delta_mem_adapter.pt"
    config = checkpoint_dir / "delta_mem_config.json"
    audit_file = checkpoint_dir / selector.run_audit.AUDIT_FILENAME
    adapter.write_bytes(f"adapter-{step}".encode("ascii"))
    config.write_text("{}\n", encoding="utf-8")
    audit = _full_coverage_audit(step)
    audit.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "run_root": str(run_root),
            "checkpoint": str(checkpoint_dir),
            "checkpoint_optimizer_step": step,
        }
    )
    audit["receipt_sha256"] = selector.canonical_sha256(audit)
    audit_file.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    coverage = selector.validate_recomputed_adapter_change(
        audit["adapter_change"],
        step=step,
    )
    adapter_binding = selector.artifact_binding(adapter, description="adapter")
    checkpoint = {
        "path": str(checkpoint_dir),
        "global_step": step,
        "artifacts": {
            "delta_mem_adapter.pt": adapter_binding,
            "delta_mem_config.json": selector.artifact_binding(
                config,
                description="config",
            ),
            selector.run_audit.AUDIT_FILENAME: selector.artifact_binding(
                audit_file,
                description="hard-failure checkpoint audit",
            ),
        },
        **coverage,
        "checkpoint_audit_receipt_sha256": audit["receipt_sha256"],
        "current_adapter_validation": {
            "initial_adapter_manifest": selector.artifact_binding(
                initial_manifest,
                description="initial adapter manifest",
            ),
            "initial_adapter": selector.artifact_binding(
                initial_adapter,
                description="initial adapter",
            ),
            "checkpoint_adapter": adapter_binding,
            "recomputed_adapter_change_canonical_sha256": (
                selector.canonical_sha256(audit["adapter_change"])
            ),
            **coverage,
            "frozen_adapter_tensors_unchanged": True,
        },
    }
    return checkpoint, audit


def _install_authorization_endpoint_validation_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition_protocols = state_eval.resolved_condition_protocols(
        list(state_eval.SCENE_FOCUSED_CONDITIONS),
        donor_rule=state_eval.DONOR_RULE_LENGTH_MATCHED,
    )

    def canonical_focused_train_evidence(
        *,
        checkpoint_dir: Path,
        checkpoint_artifact_sha256: dict[str, str],
        selector_source: dict[str, Any],
        fingerprint_payload: dict[str, Any],
    ) -> dict[str, Any]:
        del checkpoint_dir, checkpoint_artifact_sha256, selector_source
        return {
            "selected_by_index": {},
            "donors": {},
            "shuffled": {},
            "condition_protocols": condition_protocols,
            "evaluation_contract": fingerprint_payload["evaluation_contract"],
            "candidate_lineage": fingerprint_payload["candidate_lineage"],
            "candidate_lineage_record_binding": fingerprint_payload[
                "candidate_lineage_record_binding"
            ],
            "fingerprint_payload": fingerprint_payload,
        }

    monkeypatch.setattr(
        state_eval,
        "_canonical_focused_train_evidence",
        canonical_focused_train_evidence,
    )
    monkeypatch.setattr(
        state_eval,
        "validate_resume_records",
        lambda records, **kwargs: list(range(state_eval.SCENE_HARD_FAILURE_ROWS)),
    )


def _authorization_test_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    Path,
    dict[Path, dict[str, Any]],
]:
    run_root = tmp_path / "run"
    initial_dir = run_root / "initial_adapter"
    initial_dir.mkdir(parents=True)
    initial_manifest = initial_dir / "initial_adapter_manifest.json"
    initial_adapter = initial_dir / "delta_mem_adapter.pt"
    initial_manifest.write_text("{}\n", encoding="utf-8")
    initial_adapter.write_bytes(b"initial-adapter")
    source_manifest = tmp_path / "train-source.json"
    source_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    source = _source_binding(source_manifest)
    source["source_manifest_file_sha256"] = selector.sha256_file(source_manifest)
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
    monkeypatch.setattr(
        state_eval,
        "_recompute_focused_adapter_change",
        lambda checkpoint_adapter_path, **kwargs: dict(
            json.loads(
                (
                    checkpoint_adapter_path.parent
                    / selector.run_audit.AUDIT_FILENAME
                ).read_text(encoding="utf-8")
            )["adapter_change"]
        ),
    )
    recomputed: dict[Path, dict[str, Any]] = {}
    monkeypatch.setattr(
        selector.focused_gate,
        "analyze_results_dir",
        lambda path, *, stage: json.loads(json.dumps(recomputed[path.resolve()])),
    )
    _install_authorization_endpoint_validation_mocks(monkeypatch)
    return run_root, source, initial_manifest, initial_adapter, recomputed


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
    progress = {
        "fingerprint": fingerprint,
        "completed": 32 * len(state_eval.SCENE_FOCUSED_CONDITIONS),
        "expected": 32 * len(state_eval.SCENE_FOCUSED_CONDITIONS),
        "complete": True,
    }
    (results_dir / "progress.json").write_text(
        json.dumps(progress, sort_keys=True), encoding="utf-8"
    )
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

    assert evidence.benchmark_gate_passed is True
    assert evidence.full_coverage is True
    assert evidence.selection_eligible is True
    assert evidence.evaluation["fingerprint"] == fingerprint
    assert evidence.gate["file_sha256"] == selector.sha256_file(gate_file)
    assert set(evidence.evaluation["artifacts"]) == {
        "manifest.json",
        "summary.json",
        "progress.json",
        *(f"{condition}.jsonl" for condition in state_eval.SCENE_FOCUSED_CONDITIONS),
    }

    (results_dir / "progress.json").unlink()
    with pytest.raises(selector.SelectionError, match="evaluation progress"):
        selector.load_endpoint_evidence(
            selector.EndpointSpec(16, results_dir, gate_file),
            checkpoint=checkpoint,
            source=source,
        )

    invalid_progress = dict(progress)
    invalid_progress["completed"] -= 1
    (results_dir / "progress.json").write_text(
        json.dumps(invalid_progress, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(
        selector.SelectionError,
        match="progress is incomplete or differs",
    ):
        selector.load_endpoint_evidence(
            selector.EndpointSpec(16, results_dir, gate_file),
            checkpoint=checkpoint,
            source=source,
        )
    (results_dir / "progress.json").write_text(
        json.dumps(progress, sort_keys=True), encoding="utf-8"
    )

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
    initial_dir = tmp_path / "run" / "initial_adapter"
    initial_dir.mkdir()
    initial_manifest = initial_dir / "initial_adapter_manifest.json"
    initial_adapter = initial_dir / "delta_mem_adapter.pt"
    initial_manifest.write_text("{}\n", encoding="utf-8")
    initial_adapter.write_bytes(b"initial-adapter")
    adapter = memory_dir / "delta_mem_adapter.pt"
    config = memory_dir / "delta_mem_config.json"
    audit_file = memory_dir / selector.run_audit.AUDIT_FILENAME
    adapter.write_bytes(b"adapter")
    config.write_text("{}\n", encoding="utf-8")
    audit_payload = _full_coverage_audit()
    audit_payload.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "run_root": str(tmp_path / "run"),
            "checkpoint": str(memory_dir),
            "checkpoint_optimizer_step": 16,
        }
    )
    audit_payload["receipt_sha256"] = selector.canonical_sha256(audit_payload)
    audit_file.write_text(json.dumps(audit_payload, sort_keys=True), encoding="utf-8")
    coverage_evidence = selector.validate_recomputed_adapter_change(
        audit_payload["adapter_change"],
        step=16,
    )
    adapter_binding = selector.artifact_binding(adapter, description="adapter")
    checkpoint = {
        "path": str(memory_dir),
        "global_step": 16,
        "artifacts": {
            "delta_mem_adapter.pt": adapter_binding,
            "delta_mem_config.json": selector.artifact_binding(
                config, description="config"
            ),
            selector.run_audit.AUDIT_FILENAME: selector.artifact_binding(
                audit_file,
                description="hard-failure checkpoint audit",
            ),
        },
        **coverage_evidence,
        "checkpoint_audit_receipt_sha256": audit_payload["receipt_sha256"],
        "current_adapter_validation": {
            "initial_adapter_manifest": selector.artifact_binding(
                initial_manifest,
                description="initial adapter manifest",
            ),
            "initial_adapter": selector.artifact_binding(
                initial_adapter,
                description="initial adapter",
            ),
            "checkpoint_adapter": adapter_binding,
            "recomputed_adapter_change_canonical_sha256": (
                selector.canonical_sha256(audit_payload["adapter_change"])
            ),
            **coverage_evidence,
            "frozen_adapter_tensors_unchanged": True,
        },
    }
    source_manifest = tmp_path / "train-source.json"
    source_manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    source = _source_binding(source_manifest)
    source["source_manifest_file_sha256"] = selector.sha256_file(source_manifest)
    results_dir = tmp_path / "train32-results"
    evidence = _write_bound_endpoint_evidence(
        results_dir,
        step=16,
        checkpoint=checkpoint,
        source=source,
        report=_gate_report(
            passed=True,
            passed_gate_count=5,
            state_f1=1.0,
            normal_f1=1.0,
        ),
    )
    recomputed_gates = {results_dir.resolve(): evidence.report}
    monkeypatch.setattr(
        selector.focused_gate,
        "analyze_results_dir",
        lambda path, *, stage: json.loads(
            json.dumps(recomputed_gates[path.resolve()])
        ),
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
    monkeypatch.setattr(
        state_eval,
        "_recompute_focused_adapter_change",
        lambda checkpoint_adapter_path, **kwargs: dict(
            json.loads(
                (checkpoint_adapter_path.parent / selector.run_audit.AUDIT_FILENAME)
                .read_text(encoding="utf-8")
            )["adapter_change"]
        ),
    )
    _install_authorization_endpoint_validation_mocks(monkeypatch)

    authorization = state_eval.validate_focused_train_selection_authorization(
        receipt_path,
        memory_dir=memory_dir,
    )

    assert authorization["hard32_authorized"] is True
    assert authorization["selected_checkpoint"]["global_step"] == 16
    assert authorization["evaluation_fingerprint"] == evidence.evaluation[
        "fingerprint"
    ]
    assert receipt["selected_checkpoint"]["artifacts"][
        selector.run_audit.AUDIT_FILENAME
    ]["sha256"] == selector.sha256_file(audit_file)

    forged_full = json.loads(json.dumps(receipt))
    forged_audit = _coverage_audit(
        16,
        missing={"hrm_rwkv7_core.x_w": [1]},
    )
    forged_audit.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "run_root": str(tmp_path / "run"),
            "checkpoint": str(memory_dir),
            "checkpoint_optimizer_step": 16,
        }
    )
    forged_audit["receipt_sha256"] = selector.canonical_sha256(forged_audit)
    audit_file.write_text(json.dumps(forged_audit, sort_keys=True), encoding="utf-8")
    forged_audit_binding = selector.artifact_binding(
        audit_file,
        description="hard-failure checkpoint audit",
    )
    for candidate in (
        forged_full["selected_checkpoint"],
        forged_full["evaluated_endpoints"][-1]["checkpoint"],
    ):
        candidate["artifacts"][selector.run_audit.AUDIT_FILENAME] = (
            forged_audit_binding
        )
        candidate["checkpoint_audit_receipt_sha256"] = forged_audit[
            "receipt_sha256"
        ]
    forged_full.pop("receipt_sha256")
    forged_full["receipt_sha256"] = selector.canonical_sha256(forged_full)
    selector.atomic_write_json(receipt_path, forged_full)
    with pytest.raises(ValueError, match="current-byte coverage binding differs"):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=memory_dir,
        )

    audit_file.write_text(json.dumps(audit_payload, sort_keys=True), encoding="utf-8")
    wrong_families = json.loads(json.dumps(receipt))
    expected_suffix = selector.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES[0]
    for family_map in (
        wrong_families["selected_checkpoint"][
            "trainable_family_layer_coverage"
        ],
        wrong_families["selected_checkpoint"]["current_adapter_validation"][
            "trainable_family_layer_coverage"
        ],
        wrong_families["evaluated_endpoints"][-1]["checkpoint"][
            "trainable_family_layer_coverage"
        ],
        wrong_families["evaluated_endpoints"][-1]["checkpoint"][
            "current_adapter_validation"
        ]["trainable_family_layer_coverage"],
        wrong_families["evaluated_endpoints"][-1]["coverage_evidence"][
            "trainable_family_layer_coverage"
        ],
    ):
        family_map["invented_trainable_family"] = family_map.pop(expected_suffix)
    wrong_families.pop("receipt_sha256")
    wrong_families["receipt_sha256"] = selector.canonical_sha256(wrong_families)
    selector.atomic_write_json(receipt_path, wrong_families)
    with pytest.raises(ValueError, match="current-byte coverage binding differs"):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=memory_dir,
        )

    memory_dir_32 = tmp_path / "run" / "trainer" / "checkpoint-32"
    memory_dir_32.mkdir()
    adapter_32 = memory_dir_32 / "delta_mem_adapter.pt"
    config_32 = memory_dir_32 / "delta_mem_config.json"
    audit_file_32 = memory_dir_32 / selector.run_audit.AUDIT_FILENAME
    adapter_32.write_bytes(b"adapter-32")
    config_32.write_text("{}\n", encoding="utf-8")
    audit_payload_32 = _full_coverage_audit(32)
    audit_payload_32.update(
        {
            "schema": selector.run_audit.AUDIT_SCHEMA,
            "run_root": str(tmp_path / "run"),
            "checkpoint": str(memory_dir_32),
            "checkpoint_optimizer_step": 32,
        }
    )
    audit_payload_32["receipt_sha256"] = selector.canonical_sha256(
        audit_payload_32
    )
    audit_file_32.write_text(
        json.dumps(audit_payload_32, sort_keys=True),
        encoding="utf-8",
    )
    coverage_evidence_32 = selector.validate_recomputed_adapter_change(
        audit_payload_32["adapter_change"],
        step=32,
    )
    adapter_binding_32 = selector.artifact_binding(
        adapter_32,
        description="adapter-32",
    )
    checkpoint_32 = {
        "path": str(memory_dir_32),
        "global_step": 32,
        "artifacts": {
            "delta_mem_adapter.pt": adapter_binding_32,
            "delta_mem_config.json": selector.artifact_binding(
                config_32,
                description="config-32",
            ),
            selector.run_audit.AUDIT_FILENAME: selector.artifact_binding(
                audit_file_32,
                description="hard-failure checkpoint-32 audit",
            ),
        },
        **coverage_evidence_32,
        "checkpoint_audit_receipt_sha256": audit_payload_32["receipt_sha256"],
        "current_adapter_validation": {
            "initial_adapter_manifest": selector.artifact_binding(
                initial_manifest,
                description="initial adapter manifest",
            ),
            "initial_adapter": selector.artifact_binding(
                initial_adapter,
                description="initial adapter",
            ),
            "checkpoint_adapter": adapter_binding_32,
            "recomputed_adapter_change_canonical_sha256": (
                selector.canonical_sha256(audit_payload_32["adapter_change"])
            ),
            **coverage_evidence_32,
            "frozen_adapter_tensors_unchanged": True,
        },
    }
    results_dir_32 = tmp_path / "train32-results-32"
    evidence_32 = _write_bound_endpoint_evidence(
        results_dir_32,
        step=32,
        checkpoint=checkpoint_32,
        source=source,
        report=_gate_report(
            passed=True,
            passed_gate_count=5,
            state_f1=1.0,
            normal_f1=1.0,
        ),
    )
    recomputed_gates[results_dir_32.resolve()] = evidence_32.report
    skipped_earlier = json.loads(json.dumps(receipt))
    skipped_earlier["evaluated_endpoint_steps"] = [16, 32]
    skipped_earlier["evaluated_endpoints"][0]["benchmark_gate_passed"] = False
    skipped_earlier["evaluated_endpoints"][0]["selection_eligible"] = False
    skipped_earlier["evaluated_endpoints"].append(
        selector._endpoint_receipt_record(evidence_32)
    )
    skipped_earlier["selected_checkpoint"] = checkpoint_32
    skipped_earlier["selected_endpoint_eligibility"] = {
        "benchmark_gate_passed": True,
        "full_coverage": True,
        "selection_eligible": True,
    }
    skipped_earlier["evaluation"] = evidence_32.evaluation
    skipped_earlier["gate"] = evidence_32.gate
    skipped_earlier.pop("receipt_sha256")
    skipped_earlier["receipt_sha256"] = selector.canonical_sha256(skipped_earlier)
    selector.atomic_write_json(receipt_path, skipped_earlier)
    with pytest.raises(ValueError, match="endpoint eligibility differs"):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=memory_dir_32,
        )

    partial = json.loads(json.dumps(receipt))
    partial.pop("receipt_sha256")
    partial_checkpoint = partial["selected_checkpoint"]
    partial_endpoint = partial["evaluated_endpoints"][-1]
    partial_endpoint_checkpoint = partial_endpoint["checkpoint"]
    missing = {"hrm_rwkv7_core.x_w": [1]}
    for candidate in (partial_checkpoint, partial_endpoint_checkpoint):
        candidate["changed_trainable_tensor_count"] = 1133
        candidate["trainable_family_layer_coverage"][
            "hrm_rwkv7_core.x_w"
        ] = 41
        candidate["missing_trainable_family_layers"] = missing
        candidate["full_trainable_family_coverage"] = False
        current = candidate["current_adapter_validation"]
        current["changed_trainable_tensor_count"] = 1133
        current["trainable_family_layer_coverage"][
            "hrm_rwkv7_core.x_w"
        ] = 41
        current["missing_trainable_family_layers"] = missing
        current["full_trainable_family_coverage"] = False
    partial_coverage = partial_endpoint["coverage_evidence"]
    partial_coverage["changed_trainable_tensor_count"] = 1133
    partial_coverage["trainable_family_layer_coverage"][
        "hrm_rwkv7_core.x_w"
    ] = 41
    partial_coverage["missing_trainable_family_layers"] = missing
    partial_coverage["full_trainable_family_coverage"] = False
    partial_endpoint["full_coverage"] = False
    partial_endpoint["selection_eligible"] = False
    partial["selected_endpoint_eligibility"] = {
        "benchmark_gate_passed": True,
        "full_coverage": False,
        "selection_eligible": False,
    }
    partial["receipt_sha256"] = selector.canonical_sha256(partial)
    selector.atomic_write_json(receipt_path, partial)
    with pytest.raises(ValueError, match="current-byte coverage binding differs"):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=memory_dir,
        )

    legacy = json.loads(json.dumps(receipt))
    legacy.pop("receipt_sha256")
    legacy["schema"] = "rwkv_ms_scene_hard_failure_train_overfit_selection.v1"
    legacy["receipt_sha256"] = selector.canonical_sha256(legacy)
    selector.atomic_write_json(receipt_path, legacy)
    with pytest.raises(ValueError, match="schema differs"):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=memory_dir,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("redirected", "gate is not bound to its results directory"),
        ("fabricated", "gate differs from recomputed Train32 results"),
    ],
)
def test_authorization_recomputes_earlier_gate_for_first_eligible_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    (
        run_root,
        source,
        initial_manifest,
        initial_adapter,
        recomputed,
    ) = _authorization_test_context(tmp_path, monkeypatch)
    checkpoint_16, _ = _write_authorization_checkpoint(
        run_root,
        step=16,
        initial_manifest=initial_manifest,
        initial_adapter=initial_adapter,
    )
    checkpoint_32, _ = _write_authorization_checkpoint(
        run_root,
        step=32,
        initial_manifest=initial_manifest,
        initial_adapter=initial_adapter,
    )
    results_16 = tmp_path / "train32-results-16"
    results_32 = tmp_path / "train32-results-32"
    evidence_16 = _write_bound_endpoint_evidence(
        results_16,
        step=16,
        checkpoint=checkpoint_16,
        source=source,
        report=_gate_report(
            passed=False,
            passed_gate_count=4,
            state_f1=0.5,
            normal_f1=0.5,
        ),
    )
    evidence_32 = _write_bound_endpoint_evidence(
        results_32,
        step=32,
        checkpoint=checkpoint_32,
        source=source,
        report=_gate_report(
            passed=True,
            passed_gate_count=5,
            state_f1=1.0,
            normal_f1=1.0,
        ),
    )
    recomputed[results_16.resolve()] = evidence_16.report
    recomputed[results_32.resolve()] = evidence_32.report
    receipt = selector.build_selection_receipt(
        source=source,
        evaluated=[evidence_16, evidence_32],
        selected=evidence_32,
        passed=True,
        created_at="2026-08-01T00:00:00+00:00",
    )
    receipt_path = tmp_path / "selection-receipt.json"
    selector.atomic_write_json(receipt_path, receipt)
    authorization = state_eval.validate_focused_train_selection_authorization(
        receipt_path,
        memory_dir=Path(checkpoint_32["path"]),
    )
    assert authorization["selected_checkpoint"]["global_step"] == 32

    if tamper == "redirected":
        forged = json.loads(json.dumps(receipt))
        forged["evaluated_endpoints"][0]["gate"] = evidence_32.gate
        forged.pop("receipt_sha256")
        forged["receipt_sha256"] = selector.canonical_sha256(forged)
        selector.atomic_write_json(receipt_path, forged)
    else:
        fabricated = json.loads(json.dumps(evidence_16.report))
        fabricated["all_gates_passed"] = True
        fabricated["status"] = "diagnostic_pass"
        recomputed[results_16.resolve()] = fabricated

    with pytest.raises(ValueError, match=message):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=Path(checkpoint_32["path"]),
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("gate", "gate differs from recomputed Train32 results"),
        ("result", "current bytes differ"),
    ],
)
def test_authorization_rejects_selected_gate_or_result_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    (
        run_root,
        source,
        initial_manifest,
        initial_adapter,
        recomputed,
    ) = _authorization_test_context(tmp_path, monkeypatch)
    checkpoint, _ = _write_authorization_checkpoint(
        run_root,
        step=16,
        initial_manifest=initial_manifest,
        initial_adapter=initial_adapter,
    )
    results_dir = tmp_path / "train32-results-16"
    evidence = _write_bound_endpoint_evidence(
        results_dir,
        step=16,
        checkpoint=checkpoint,
        source=source,
        report=_gate_report(
            passed=True,
            passed_gate_count=5,
            state_f1=1.0,
            normal_f1=1.0,
        ),
    )
    recomputed[results_dir.resolve()] = evidence.report
    receipt = selector.build_selection_receipt(
        source=source,
        evaluated=[evidence],
        selected=evidence,
        passed=True,
        created_at="2026-08-01T00:00:00+00:00",
    )
    receipt_path = tmp_path / "selection-receipt.json"
    selector.atomic_write_json(receipt_path, receipt)
    assert state_eval.validate_focused_train_selection_authorization(
        receipt_path,
        memory_dir=Path(checkpoint["path"]),
    )["hard32_authorized"] is True

    if tamper == "gate":
        gate_path = results_dir / selector.GATE_FILENAME
        mutated = json.loads(gate_path.read_text(encoding="utf-8"))
        mutated["status"] = "diagnostic_fail"
        gate_path.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    else:
        with (results_dir / "summary.json").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")

    with pytest.raises(ValueError, match=message):
        state_eval.validate_focused_train_selection_authorization(
            receipt_path,
            memory_dir=Path(checkpoint["path"]),
        )


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
