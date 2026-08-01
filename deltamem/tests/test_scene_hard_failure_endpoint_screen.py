from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pytest

from experiments.rethinking_rwkv_ms_gemma import (
    run_scene_hard_failure_endpoint_screen as driver,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _production_run(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "scene_failure_runs"
    run_root = parent / "production_run_step64"
    log_dir = parent / "logs"
    log_dir.mkdir(parents=True)
    audits: list[dict[str, Any]] = []
    for step in range(1, driver.train_contract.TOTAL_OPTIMIZER_STEPS + 1):
        audit = (
            run_root
            / "trainer"
            / f"checkpoint-{step}"
            / driver.run_audit.AUDIT_FILENAME
        )
        _write_json(audit, {"checkpoint_optimizer_step": step})
        audits.append(
            {
                "step": step,
                "path": str(audit.absolute()),
                "sha256": driver.selector.sha256_file(audit),
            }
        )
    log = log_dir / f"{run_root.name}.log"
    log.write_text("completed production training\n", encoding="utf-8")
    receipt = log_dir / f"{run_root.name}.completion.json"
    payload: dict[str, Any] = {
        "schema": driver.COMPLETION_SCHEMA,
        "completed_at": "2026-08-01T00:00:00+00:00",
        "run_mode": "production",
        "run_root": str(run_root.absolute()),
        "global_step": driver.train_contract.TOTAL_OPTIMIZER_STEPS,
        "checkpoint_steps": list(
            range(1, driver.train_contract.TOTAL_OPTIMIZER_STEPS + 1)
        ),
        "checkpoint_audits": audits,
        "log": {
            "path": str(log.absolute()),
            "sha256": driver.selector.sha256_file(log),
        },
        "training_complete": True,
        "evaluation_accessed": False,
    }
    payload["receipt_sha256"] = driver.selector.canonical_sha256(payload)
    _write_json(receipt, payload)
    return run_root.absolute(), receipt.absolute()


def _gate_report(*, passed: bool, step: int) -> dict[str, Any]:
    return {
        "schema": driver.focused_gate.SCHEMA,
        "status": "diagnostic_pass" if passed else "diagnostic_fail",
        "stage": driver.selector.STAGE,
        "task": driver.selector.TASK,
        "rows": driver.state_eval.SCENE_HARD_FAILURE_ROWS,
        "source_indices": list(range(driver.state_eval.SCENE_HARD_FAILURE_ROWS)),
        "all_gates_passed": passed,
        "unit_step": step,
    }


class FakeCommands:
    def __init__(self, reports: Mapping[int, dict[str, Any]]) -> None:
        self.reports = dict(reports)
        self.calls: list[list[str]] = []

    @staticmethod
    def _option(command: Sequence[str], name: str) -> str:
        index = list(command).index(name)
        return str(command[index + 1])

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> int:
        argv = list(command)
        self.calls.append(argv)
        assert cwd == driver.PROJECT_ROOT
        assert environment["PYTHONPATH"].split(":", 1)[0] == str(
            driver.PROJECT_ROOT
        )
        script = Path(argv[1]).name
        if script == "run_scene_state_eval.py":
            output_dir = Path(self._option(argv, "--output-dir"))
            output_dir.mkdir(parents=True, exist_ok=False)
            return 0
        assert script == "run_scene_focused_recovery_gate.py"
        output_file = Path(self._option(argv, "--output-file"))
        step = int(output_file.parent.name.removeprefix("checkpoint-"))
        _write_json(output_file, self.reports[step])
        return 0 if self.reports[step]["all_gates_passed"] else 1


def _selector_receipt(
    *,
    run_root: Path,
    evaluated_steps: Sequence[int],
    passed: bool,
    benchmark_passed_steps: set[int] | None = None,
    missing_by_step: Mapping[int, dict[str, list[int]]] | None = None,
) -> dict[str, Any]:
    benchmark_passed_steps = (
        ({evaluated_steps[-1]} if passed else set())
        if benchmark_passed_steps is None
        else benchmark_passed_steps
    )
    missing_by_step = {} if missing_by_step is None else missing_by_step
    checkpoints = {
        step: _checkpoint(root=run_root, step=step, missing=missing_by_step.get(step))
        for step in evaluated_steps
    }
    endpoint_records = []
    for step in evaluated_steps:
        checkpoint = checkpoints[step]
        coverage = driver.selector.validate_endpoint_checkpoint_coverage(
            checkpoint,
            step=step,
        )
        benchmark_gate_passed = step in benchmark_passed_steps
        full_coverage = coverage["full_trainable_family_coverage"] is True
        endpoint_records.append(
            {
                "global_step": step,
                "checkpoint": checkpoint,
                "benchmark_gate_passed": benchmark_gate_passed,
                "full_coverage": full_coverage,
                "selection_eligible": benchmark_gate_passed and full_coverage,
                "coverage_evidence": coverage,
            }
        )
    selected = endpoint_records[-1]
    payload: dict[str, Any] = {
        "schema": driver.selector.SCHEMA,
        "status": "pass" if passed else "fail",
        "stage": driver.selector.STAGE,
        "task": driver.selector.TASK,
        "selection_policy": driver.selector.SELECTION_POLICY,
        "endpoint_schedule": list(driver.ENDPOINT_STEPS),
        "evaluated_endpoint_steps": list(evaluated_steps),
        "evaluated_endpoints": endpoint_records,
        "selected_checkpoint": checkpoints[evaluated_steps[-1]],
        "selected_endpoint_eligibility": {
            name: selected[name]
            for name in (
                "benchmark_gate_passed",
                "full_coverage",
                "selection_eligible",
            )
        },
        "selection_reason": (
            "first_train32_gate_pass_with_full_current_adapter_coverage"
            if passed
            else "diagnostic_fallback_no_selection_eligible_train32_endpoint"
        ),
        "authorization": {
            "hard32_authorized": passed,
            "selected_checkpoint_only": passed,
            "full_validation": False,
            "test": False,
            "other_benchmarks": False,
        },
        "held_out_access": driver.selector.HARD32_ACCESS_POLICY,
        "hard32_accessed": False,
        "full_validation_accessed": False,
        "test_accessed": False,
        "other_benchmarks_accessed": False,
    }
    payload["receipt_sha256"] = driver.selector.canonical_sha256(payload)
    return payload


def _source_validator() -> dict[str, Any]:
    return {"protected_evaluation_accessed": False, "source_lock_sha256": "a" * 64}


def _checkpoint(
    root: Path,
    step: int,
    *,
    missing: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    missing = {} if missing is None else missing
    coverage = {
        suffix: 42 - len(missing.get(suffix, []))
        for suffix in driver.run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    changed_count = sum(coverage.values())
    full_coverage = not missing and changed_count == 1134
    adapter = {
        "path": str(root / "trainer" / f"checkpoint-{step}" / "delta_mem_adapter.pt"),
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
        "path": str(root / "trainer" / f"checkpoint-{step}"),
        "global_step": step,
        "artifacts": {"delta_mem_adapter.pt": adapter},
        **coverage_evidence,
        "current_adapter_validation": {
            "checkpoint_adapter": adapter,
            "recomputed_adapter_change_canonical_sha256": f"{step + 1:064x}",
            **coverage_evidence,
            "frozen_adapter_tensors_unchanged": True,
        },
    }


def _checkpoint_validator(
    root: Path,
    step: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    assert source["protected_evaluation_accessed"] is False
    return _checkpoint(root, step)


def test_screen_stops_generation_at_first_passing_endpoint(tmp_path: Path) -> None:
    run_root, completion = _production_run(tmp_path)
    reports = {
        16: _gate_report(passed=False, step=16),
        32: _gate_report(passed=True, step=32),
    }
    commands = FakeCommands(reports)
    selector_calls: list[list[int]] = []

    def select(
        root: Path,
        specs: Sequence[driver.selector.EndpointSpec],
    ) -> dict[str, Any]:
        assert root == run_root
        selector_calls.append([spec.step for spec in specs])
        return _selector_receipt(
            run_root=root,
            evaluated_steps=[16, 32],
            passed=True,
        )

    receipt = driver.screen_endpoints(
        run_root=run_root,
        completion_receipt=completion,
        environment={"PATH": "/usr/bin"},
        command_runner=commands,
        gate_analyzer=lambda results: reports[
            int(results.name.removeprefix("checkpoint-"))
        ],
        selector_runner=select,
        source_validator=_source_validator,
        checkpoint_validator=_checkpoint_validator,
    )

    assert selector_calls == [[16, 32, 48, 64]]
    assert [Path(call[1]).name for call in commands.calls] == [
        "run_scene_state_eval.py",
        "run_scene_focused_recovery_gate.py",
        "run_scene_state_eval.py",
        "run_scene_focused_recovery_gate.py",
    ]
    evaluation_calls = commands.calls[::2]
    assert [FakeCommands._option(call, "--memory-dir") for call in evaluation_calls] == [
        str(run_root / "trainer" / "checkpoint-16"),
        str(run_root / "trainer" / "checkpoint-32"),
    ]
    assert all(
        FakeCommands._option(call, "--conditions") == ",".join(driver.CONDITIONS)
        for call in evaluation_calls
    )
    assert all(
        FakeCommands._option(call, "--evaluation-contract")
        == driver.EVALUATION_CONTRACT
        for call in evaluation_calls
    )
    assert receipt["status"] == "pass"
    assert receipt["evaluated_endpoint_steps"] == [16, 32]
    assert receipt["production_completion_receipt"]["global_step"] == 64
    assert receipt["endpoint_screen"]["hard32_accessed"] is False
    receipt_path = (
        run_root / driver.SCREEN_DIRNAME / driver.SELECTION_RECEIPT_FILENAME
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")
    assert recorded == driver.selector.canonical_sha256(unsigned)


def test_screen_continues_gate_only_partial_until_first_eligible_endpoint(
    tmp_path: Path,
) -> None:
    run_root, completion = _production_run(tmp_path)
    reports = {
        step: _gate_report(passed=True, step=step)
        for step in (16, 32, 48)
    }
    commands = FakeCommands(reports)
    missing = {
        16: {
            "hrm_rwkv7_core.x_a": [1],
            "hrm_rwkv7_core.x_w": list(range(14)),
        },
        32: {"hrm_rwkv7_core.x_w": [1]},
    }

    receipt = driver.screen_endpoints(
        run_root=run_root,
        completion_receipt=completion,
        environment={},
        command_runner=commands,
        gate_analyzer=lambda results: reports[
            int(results.name.removeprefix("checkpoint-"))
        ],
        selector_runner=lambda root, specs: _selector_receipt(
            run_root=root,
            evaluated_steps=[16, 32, 48],
            passed=True,
            benchmark_passed_steps={16, 32, 48},
            missing_by_step=missing,
        ),
        source_validator=_source_validator,
        checkpoint_validator=lambda root, step, source: _checkpoint(
            root,
            step,
            missing=missing.get(step),
        ),
    )

    assert len(commands.calls) == 6
    assert receipt["evaluated_endpoint_steps"] == [16, 32, 48]
    decisions = receipt["endpoint_screen"]["endpoint_decisions"]
    assert [decision["selection_eligible"] for decision in decisions] == [
        False,
        False,
        True,
    ]
    assert [
        decision["coverage_evidence"]["changed_trainable_tensor_count"]
        for decision in decisions
    ] == [1119, 1133, 1134]
    assert decisions[0]["benchmark_gate_passed"] is True
    assert decisions[0]["full_coverage"] is False
    assert decisions[2]["coverage_evidence"][
        "recomputed_adapter_change_canonical_sha256"
    ] == f"{49:064x}"
    protocol = receipt["endpoint_screen"]["protocol"]
    assert protocol["file_sha256"] == driver.selector.sha256_file(
        run_root / driver.SCREEN_DIRNAME / driver.PROTOCOL_FILENAME
    )


def test_screen_finishes_four_failures_and_writes_unauthorized_receipt(
    tmp_path: Path,
) -> None:
    run_root, completion = _production_run(tmp_path)
    reports = {
        step: _gate_report(passed=step == 16, step=step)
        for step in driver.ENDPOINT_STEPS
    }
    commands = FakeCommands(reports)
    missing = {16: {"hrm_rwkv7_core.x_w": [1]}}

    receipt = driver.screen_endpoints(
        run_root=run_root,
        completion_receipt=completion,
        environment={},
        command_runner=commands,
        gate_analyzer=lambda results: reports[
            int(results.name.removeprefix("checkpoint-"))
        ],
        selector_runner=lambda root, specs: _selector_receipt(
            run_root=root,
            evaluated_steps=driver.ENDPOINT_STEPS,
            passed=False,
            benchmark_passed_steps={16},
            missing_by_step=missing,
        ),
        source_validator=_source_validator,
        checkpoint_validator=lambda root, step, source: _checkpoint(
            root,
            step,
            missing=missing.get(step),
        ),
    )

    assert len(commands.calls) == 8
    assert receipt["status"] == "fail"
    assert receipt["evaluated_endpoint_steps"] == [16, 32, 48, 64]
    assert receipt["authorization"]["hard32_authorized"] is False
    assert receipt["endpoint_screen"]["endpoint_decisions"][0] == {
        "global_step": 16,
        "benchmark_gate_passed": True,
        "full_coverage": False,
        "selection_eligible": False,
        "coverage_evidence": driver.selector.validate_endpoint_checkpoint_coverage(
            _checkpoint(run_root, 16, missing=missing[16]),
            step=16,
        ),
    }
    assert receipt["endpoint_screen"]["evaluated_endpoint_steps"] == [
        16,
        32,
        48,
        64,
    ]


def test_screen_rejects_gate_recomputation_drift(tmp_path: Path) -> None:
    run_root, completion = _production_run(tmp_path)
    recorded = _gate_report(passed=False, step=16)
    commands = FakeCommands({16: recorded})

    with pytest.raises(driver.EndpointScreenError, match="independent recomputation"):
        driver.screen_endpoints(
            run_root=run_root,
            completion_receipt=completion,
            environment={},
            command_runner=commands,
            gate_analyzer=lambda results: _gate_report(passed=True, step=16),
            selector_runner=lambda root, specs: pytest.fail("selector must not run"),
            source_validator=_source_validator,
            checkpoint_validator=_checkpoint_validator,
        )


@pytest.mark.parametrize(
    ("field", "value", "rehash", "message"),
    [
        ("schema", "legacy.v1", True, "static contract"),
        ("selection_policy", "gate_only", True, "static contract"),
        ("status", "unknown", True, "static contract"),
        ("receipt_sha256", "0" * 64, False, "receipt_sha256 differs"),
    ],
)
def test_screen_rejects_malformed_selector_receipt(
    tmp_path: Path,
    field: str,
    value: str,
    rehash: bool,
    message: str,
) -> None:
    run_root, completion = _production_run(tmp_path)
    report = _gate_report(passed=True, step=16)
    commands = FakeCommands({16: report})

    def malformed_selector(
        root: Path,
        specs: Sequence[driver.selector.EndpointSpec],
    ) -> dict[str, Any]:
        receipt = _selector_receipt(
            run_root=root,
            evaluated_steps=[16],
            passed=True,
        )
        if field != "receipt_sha256":
            receipt.pop("receipt_sha256")
        receipt[field] = value
        if rehash:
            receipt["receipt_sha256"] = driver.selector.canonical_sha256(receipt)
        return receipt

    with pytest.raises(
        (driver.EndpointScreenError, driver.selector.SelectionError),
        match=message,
    ):
        driver.screen_endpoints(
            run_root=run_root,
            completion_receipt=completion,
            environment={},
            command_runner=commands,
            gate_analyzer=lambda results: report,
            selector_runner=malformed_selector,
            source_validator=_source_validator,
            checkpoint_validator=_checkpoint_validator,
        )


@pytest.mark.parametrize("mutation_target", ["selected", "endpoint"])
def test_screen_rejects_selector_checkpoint_substitution(
    tmp_path: Path,
    mutation_target: str,
) -> None:
    run_root, completion = _production_run(tmp_path)
    report = _gate_report(passed=True, step=16)
    commands = FakeCommands({16: report})

    def substituted_selector(
        root: Path,
        specs: Sequence[driver.selector.EndpointSpec],
    ) -> dict[str, Any]:
        receipt = _selector_receipt(
            run_root=root,
            evaluated_steps=[16],
            passed=True,
        )
        receipt.pop("receipt_sha256")
        if mutation_target == "selected":
            target = dict(receipt["selected_checkpoint"])
            receipt["selected_checkpoint"] = target
        else:
            target = receipt["evaluated_endpoints"][0]["checkpoint"]
        target["path"] = str(root / "trainer" / "substituted-checkpoint-16")
        receipt["receipt_sha256"] = driver.selector.canonical_sha256(receipt)
        return receipt

    with pytest.raises(
        driver.EndpointScreenError,
        match=(
            "selected endpoint eligibility differs"
            if mutation_target == "selected"
            else "endpoint eligibility differs"
        ),
    ):
        driver.screen_endpoints(
            run_root=run_root,
            completion_receipt=completion,
            environment={},
            command_runner=commands,
            gate_analyzer=lambda results: report,
            selector_runner=substituted_selector,
            source_validator=_source_validator,
            checkpoint_validator=_checkpoint_validator,
        )


def test_completion_preflight_rejects_tampered_audit(tmp_path: Path) -> None:
    run_root, completion = _production_run(tmp_path)
    audit = (
        run_root
        / "trainer"
        / "checkpoint-16"
        / driver.run_audit.AUDIT_FILENAME
    )
    audit.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(driver.EndpointScreenError, match="checkpoint-16 audit SHA-256"):
        driver.validate_production_completion(
            run_root=run_root,
            completion_receipt=completion,
        )


def test_preflight_rejects_protected_environment_and_path() -> None:
    with pytest.raises(driver.EndpointScreenError, match="environment variables"):
        driver.validate_environment({"HARD" + "32_TOKEN": "present"})
    with pytest.raises(driver.selector.SelectionError, match="protected path"):
        driver.selector.reject_protected_path(
            Path("/ssd") / ("hard" + "32") / "screen",
            description="unit path",
        )


def test_completion_preflight_rejects_protected_run_root(tmp_path: Path) -> None:
    run_root, completion = _production_run(tmp_path / ("hard" + "32"))

    with pytest.raises(driver.selector.SelectionError, match="protected path"):
        driver.validate_production_completion(
            run_root=run_root,
            completion_receipt=completion,
        )


def test_static_contract_rejects_condition_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "CONDITIONS", driver.FROZEN_CONDITIONS[:-1])

    with pytest.raises(driver.EndpointScreenError, match="seven-condition"):
        driver.validate_static_contract()


def test_screen_requires_fresh_output(tmp_path: Path) -> None:
    run_root, completion = _production_run(tmp_path)
    (run_root / driver.SCREEN_DIRNAME).mkdir()

    with pytest.raises(driver.EndpointScreenError, match="already exists"):
        driver.screen_endpoints(
            run_root=run_root,
            completion_receipt=completion,
            environment={},
            command_runner=lambda *args: pytest.fail("command must not run"),
            source_validator=_source_validator,
            checkpoint_validator=_checkpoint_validator,
        )


def test_external_command_runner_forwards_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=7)

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    status = driver.run_external_command(
        ["python", "worker.py"],
        driver.PROJECT_ROOT,
        {"PATH": "/usr/bin"},
    )

    assert status == 7
    assert observed["args"] == (["python", "worker.py"],)
    assert "shell" not in observed["kwargs"]
    assert observed["kwargs"]["check"] is False
    assert observed["kwargs"]["cwd"] == driver.PROJECT_ROOT
