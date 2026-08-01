#!/usr/bin/env python3
"""Screen the four hard-scene Train32 endpoints and select one checkpoint.

This driver is deliberately train-only. It accepts a completed production run
and its canonical completion receipt, then evaluates checkpoints 16, 32, 48,
and 64 in order under the exact focused Train32 contract. Generation stops at
the first endpoint that both passes the focused gate and has complete 27-family
by 42-layer current-adapter coverage. A gate-only pass with partial coverage
continues to the next endpoint. If no endpoint satisfies both requirements, all
four endpoints are screened and the selector writes an unauthorized diagnostic
receipt.

The model, dataset, source manifest, conditions, donor rule, runtime profile,
endpoint schedule, and output layout are compiled in. Validation, test,
holdout, Hard32, and broader benchmark paths are never accepted by this CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_focused_recovery_gate as focused_gate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_state_eval as state_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_hard_failure_run_audit as run_audit,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_hard_failure_train_contract as train_contract,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    select_scene_hard_failure_checkpoint as selector,
)


SCHEMA = "rwkv_ms_scene_hard_failure_endpoint_screen.v2"
PROTOCOL_SCHEMA = "rwkv_ms_scene_hard_failure_endpoint_screen_protocol.v2"
COMPLETION_SCHEMA = "rwkv_ms_scene_hard_failure_completion.v1"
SCREEN_DIRNAME = "train32_endpoint_screen"
PROTOCOL_FILENAME = "screening_protocol.json"
SELECTION_RECEIPT_FILENAME = "train32_checkpoint_selection_receipt.json"
ENDPOINT_STEPS = selector.ENDPOINT_STEPS
CONDITIONS = state_eval.SCENE_FOCUSED_CONDITIONS
EVALUATION_CONTRACT = state_eval.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
DONOR_RULE = state_eval.DONOR_RULE_LENGTH_MATCHED
FROZEN_ENDPOINT_STEPS = (16, 32, 48, 64)
FROZEN_SELECTION_POLICY = (
    "first_train32_gate_pass_with_full_current_adapter_coverage_v2"
)
FROZEN_CONDITIONS = (
    "base_full",
    "no_write_full",
    "normal_full",
    "state_only",
    "state_only_donor",
    "state_only_shuffled",
    "state_only_no_write",
)
PROTECTED_ENV_PREFIXES = ("HARD32", "VALIDATION", "TEST", "BENCHMARK")
FORBIDDEN_DISTRIBUTED_ENV = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


class EndpointScreenError(ValueError):
    """Raised when endpoint screening would differ from the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointScreenError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_static_contract() -> None:
    _require(
        tuple(ENDPOINT_STEPS) == FROZEN_ENDPOINT_STEPS
        and tuple(train_contract.GENERATION_ENDPOINT_STEPS)
        == FROZEN_ENDPOINT_STEPS,
        "endpoint schedule differs from frozen 16,32,48,64 contract",
    )
    _require(
        tuple(CONDITIONS) == FROZEN_CONDITIONS,
        "focused condition order differs from the frozen seven-condition contract",
    )
    _require(
        EVALUATION_CONTRACT
        == "scene_hard_failure_curriculum_train32_overfit_v1"
        and DONOR_RULE == "length_matched_label_distinct_symmetric_pair_v1"
        and selector.SCHEMA
        == "rwkv_ms_scene_hard_failure_train_overfit_selection.v2"
        and selector.SELECTION_POLICY == FROZEN_SELECTION_POLICY
        and train_contract.TOTAL_OPTIMIZER_STEPS == 64
        and tuple(train_contract.TARGET_LAYERS) == tuple(range(42)),
        "Train32 endpoint screen static contract differs",
    )


def validate_environment(environment: Mapping[str, str]) -> None:
    forbidden = sorted(
        name
        for name in environment
        if name in FORBIDDEN_DISTRIBUTED_ENV
        or name.upper().startswith(PROTECTED_ENV_PREFIXES)
    )
    _require(
        not forbidden,
        "endpoint screening forbids protected or distributed environment variables: "
        + ",".join(forbidden),
    )


def _artifact_binding(path: Path, *, description: str) -> dict[str, Any]:
    return selector.artifact_binding(path, description=description)


def _reject_symlink_components(path: Path, *, description: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        _require(
            not current.is_symlink(),
            f"endpoint screening forbids symlink component for {description}: "
            f"{current}",
        )


def validate_production_completion(
    *,
    run_root: Path | str,
    completion_receipt: Path | str,
) -> tuple[Path, dict[str, Any]]:
    """Validate the completed 64-step production run without held-out access."""

    root = selector.reject_protected_path(
        run_root,
        description="production run root",
    )
    root = selector.require_directory(root, description="production run root")
    receipt_candidate = selector.reject_protected_path(
        completion_receipt,
        description="production completion receipt",
    )
    receipt_path = selector.require_regular_file(
        receipt_candidate,
        description="production completion receipt",
    )
    expected_receipt = root.parent / "logs" / f"{root.name}.completion.json"
    _require(
        receipt_path == expected_receipt,
        f"production completion receipt path differs: {receipt_path}",
    )
    payload = selector.load_json_object(
        receipt_path,
        description="production completion receipt",
    )
    receipt_sha256 = selector.validate_self_hash(
        payload,
        field="receipt_sha256",
        description="production completion receipt",
    )
    expected_steps = list(range(1, train_contract.TOTAL_OPTIMIZER_STEPS + 1))
    _require(
        payload.get("schema") == COMPLETION_SCHEMA
        and payload.get("run_mode") == "production"
        and payload.get("run_root") == str(root)
        and payload.get("global_step") == train_contract.TOTAL_OPTIMIZER_STEPS
        and payload.get("checkpoint_steps") == expected_steps
        and payload.get("training_complete") is True
        and payload.get("evaluation_accessed") is False,
        "production completion receipt contract differs",
    )

    audits = payload.get("checkpoint_audits")
    _require(
        isinstance(audits, list) and len(audits) == len(expected_steps),
        "production completion receipt checkpoint audits are incomplete",
    )
    audit_bindings: list[dict[str, Any]] = []
    for step, record in zip(expected_steps, audits, strict=True):
        expected_path = (
            root
            / "trainer"
            / f"checkpoint-{step}"
            / run_audit.AUDIT_FILENAME
        )
        _require(
            isinstance(record, Mapping)
            and record.get("step") == step
            and record.get("path") == str(expected_path),
            f"production completion checkpoint-{step} audit binding differs",
        )
        binding = _artifact_binding(
            expected_path,
            description=f"production checkpoint-{step} audit",
        )
        _require(
            record.get("sha256") == binding["sha256"],
            f"production checkpoint-{step} audit SHA-256 differs",
        )
        audit_bindings.append({"step": step, **binding})

    log = payload.get("log")
    expected_log = root.parent / "logs" / f"{root.name}.log"
    _require(
        isinstance(log, Mapping)
        and log.get("path") == str(expected_log),
        "production completion log binding differs",
    )
    log_binding = _artifact_binding(expected_log, description="production training log")
    _require(
        log.get("sha256") == log_binding["sha256"],
        "production training log SHA-256 differs",
    )
    return root, {
        "path": str(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "file_sha256": selector.sha256_file(receipt_path),
        "receipt_sha256": receipt_sha256,
        "run_root": str(root),
        "global_step": train_contract.TOTAL_OPTIMIZER_STEPS,
        "checkpoint_steps": expected_steps,
        "endpoint_steps": list(ENDPOINT_STEPS),
        "checkpoint_audits_sha256": selector.canonical_sha256(audit_bindings),
        "log": log_binding,
        "training_complete": True,
        "evaluation_accessed": False,
    }


@dataclass(frozen=True)
class ScreenPaths:
    root: Path
    protocol: Path
    selection_receipt: Path

    def results_dir(self, step: int) -> Path:
        return self.root / f"checkpoint-{step}"

    def gate_file(self, step: int) -> Path:
        return self.results_dir(step) / selector.GATE_FILENAME


def create_screen_paths(run_root: Path) -> ScreenPaths:
    screen_root = selector.reject_protected_path(
        run_root / SCREEN_DIRNAME,
        description="Train32 endpoint screen root",
    )
    _reject_symlink_components(
        screen_root.parent,
        description="Train32 endpoint screen parent",
    )
    _require(
        not screen_root.exists() and not screen_root.is_symlink(),
        f"fresh Train32 endpoint screen output already exists: {screen_root}",
    )
    screen_root.mkdir(parents=False, exist_ok=False)
    return ScreenPaths(
        root=screen_root,
        protocol=screen_root / PROTOCOL_FILENAME,
        selection_receipt=screen_root / SELECTION_RECEIPT_FILENAME,
    )


def evaluation_command(*, run_root: Path, paths: ScreenPaths, step: int) -> list[str]:
    _require(step in ENDPOINT_STEPS, f"unsupported endpoint step: {step}")
    return [
        sys.executable,
        str(SCRIPT_DIR / "run_scene_state_eval.py"),
        "--base-model",
        str(train_contract.PINNED_BASE_MODEL),
        "--memory-dir",
        str(run_root / "trainer" / f"checkpoint-{step}"),
        "--dataset-file",
        str(train_contract.TRAIN_FILE),
        "--output-dir",
        str(paths.results_dir(step)),
        "--row-indices",
        ",".join(str(index) for index in range(state_eval.SCENE_HARD_FAILURE_ROWS)),
        "--conditions",
        ",".join(CONDITIONS),
        "--donor-rule",
        DONOR_RULE,
        "--max-new-tokens",
        str(state_eval.DEFAULT_MAX_NEW_TOKENS),
        "--delta-mem-root",
        str(PROJECT_ROOT),
        "--normal-fusion-profile",
        "native",
        "--expected-memory-layer-count",
        str(len(train_contract.TARGET_LAYERS)),
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--evaluation-contract",
        EVALUATION_CONTRACT,
        "--focused-source-manifest",
        str(train_contract.SOURCE_MANIFEST),
    ]


def gate_command(*, paths: ScreenPaths, step: int) -> list[str]:
    _require(step in ENDPOINT_STEPS, f"unsupported endpoint step: {step}")
    return [
        sys.executable,
        str(SCRIPT_DIR / "run_scene_focused_recovery_gate.py"),
        "--stage",
        selector.STAGE,
        "--results-dir",
        str(paths.results_dir(step)),
        "--output-file",
        str(paths.gate_file(step)),
    ]


def _command_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result = dict(environment)
    existing = result.get("PYTHONPATH")
    result["PYTHONPATH"] = str(PROJECT_ROOT) + (f":{existing}" if existing else "")
    return result


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], int]
GateAnalyzer = Callable[[Path], dict[str, Any]]
SelectorRunner = Callable[[Path, Sequence[selector.EndpointSpec]], dict[str, Any]]
SourceValidator = Callable[[], dict[str, Any]]
CheckpointValidator = Callable[
    [Path, int, Mapping[str, Any]],
    dict[str, Any],
]


def run_external_command(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
    )
    return int(completed.returncode)


def recompute_gate(results_dir: Path) -> dict[str, Any]:
    return focused_gate.analyze_results_dir(results_dir, stage=selector.STAGE)


def run_selector(
    run_root: Path,
    endpoint_specs: Sequence[selector.EndpointSpec],
) -> dict[str, Any]:
    return selector.select_checkpoint(
        run_root=run_root,
        endpoint_specs=endpoint_specs,
    )


def validate_selector_receipt(receipt: Any) -> dict[str, Any]:
    """Fail closed before rebinding selector evidence into the driver receipt."""

    _require(isinstance(receipt, Mapping), "selector receipt must be a JSON object")
    validated = dict(receipt)
    selector.validate_self_hash(
        validated,
        field="receipt_sha256",
        description="Train32 checkpoint selection receipt",
    )
    status = validated.get("status")
    _require(
        validated.get("schema") == selector.SCHEMA
        and validated.get("stage") == selector.STAGE
        and validated.get("task") == selector.TASK
        and validated.get("selection_policy") == selector.SELECTION_POLICY
        and validated.get("endpoint_schedule") == list(ENDPOINT_STEPS)
        and status in {"pass", "fail"},
        "selector receipt static contract or status differs",
    )
    expected_reason = (
        "first_train32_gate_pass_with_full_current_adapter_coverage"
        if status == "pass"
        else "diagnostic_fallback_no_selection_eligible_train32_endpoint"
    )
    _require(
        validated.get("selection_reason") == expected_reason
        and validated.get("held_out_access") == selector.HARD32_ACCESS_POLICY
        and validated.get("hard32_accessed") is False
        and validated.get("full_validation_accessed") is False
        and validated.get("test_accessed") is False
        and validated.get("other_benchmarks_accessed") is False,
        "selector receipt reason or protected-access evidence differs",
    )
    records = validated.get("evaluated_endpoints")
    selected = validated.get("selected_checkpoint")
    selected_eligibility = validated.get("selected_endpoint_eligibility")
    _require(
        isinstance(records, list)
        and records
        and all(isinstance(record, Mapping) for record in records)
        and isinstance(selected, Mapping)
        and isinstance(selected_eligibility, Mapping),
        "selector receipt endpoint evidence is incomplete",
    )
    eligible = [
        record for record in records if record.get("selection_eligible") is True
    ]
    selected_step = selected.get("global_step")
    selected_records = [
        record for record in records if record.get("global_step") == selected_step
    ]
    selected_record_checkpoint = (
        selected_records[0].get("checkpoint") if len(selected_records) == 1 else None
    )
    _require(
        len(selected_records) == 1
        and isinstance(selected_record_checkpoint, Mapping)
        and dict(selected) == dict(selected_record_checkpoint)
        and dict(selected_eligibility)
        == {
            name: selected_records[0].get(name)
            for name in (
                "benchmark_gate_passed",
                "full_coverage",
                "selection_eligible",
            )
        },
        "selector selected endpoint eligibility differs",
    )
    if status == "pass":
        _require(
            eligible == [records[-1]]
            and selected_records[0] is records[-1]
            and selected_eligibility.get("benchmark_gate_passed") is True
            and selected_eligibility.get("full_coverage") is True,
            "selector passing endpoint is not the first eligible endpoint",
        )
    else:
        _require(
            not eligible and len(records) == len(ENDPOINT_STEPS),
            "selector failed receipt contains an eligible or missing endpoint",
        )
    return validated


def validate_source() -> dict[str, Any]:
    return selector.validate_source_binding()


def validate_endpoint_checkpoint(
    run_root: Path,
    step: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return selector.validate_checkpoint(run_root, step=step, source=source)


def _protocol_payload(
    *,
    run_root: Path,
    paths: ScreenPaths,
    completion: Mapping[str, Any],
    source: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "created_at": utc_now(),
        "run_root": str(run_root),
        "screen_root": str(paths.root),
        "production_completion_receipt": dict(completion),
        "train_source": dict(source),
        "endpoint_checkpoints": [dict(checkpoint) for checkpoint in checkpoints],
        "endpoint_steps": list(ENDPOINT_STEPS),
        "selection_policy": selector.SELECTION_POLICY,
        "evaluation_contract": EVALUATION_CONTRACT,
        "conditions": list(CONDITIONS),
        "donor_rule": DONOR_RULE,
        "dataset": {
            "path": str(train_contract.TRAIN_FILE),
            "sha256": train_contract.FILE_SHA256["train.jsonl"],
            "split": "train",
            "rows": state_eval.SCENE_HARD_FAILURE_ROWS,
        },
        "source_manifest": {
            "path": str(train_contract.SOURCE_MANIFEST),
            "sha256": train_contract.FILE_SHA256["source_manifest.json"],
        },
        "protected_evaluation": {
            "hard32_accessed": False,
            "validation_accessed": False,
            "test_accessed": False,
            "other_benchmarks_accessed": False,
        },
    }
    payload["protocol_sha256"] = selector.canonical_sha256(payload)
    return payload


def _bind_driver_receipt(
    receipt: Mapping[str, Any],
    *,
    paths: ScreenPaths,
    completion: Mapping[str, Any],
    protocol: Mapping[str, Any],
    evaluated_steps: Sequence[int],
    endpoint_decisions: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bound = dict(receipt)
    bound.pop("receipt_sha256", None)
    bound["production_completion_receipt"] = dict(completion)
    bound["endpoint_screen"] = {
        "schema": SCHEMA,
        "screen_root": str(paths.root),
        "protocol": {
            "path": str(paths.protocol),
            "file_sha256": selector.sha256_file(paths.protocol),
            "protocol_sha256": protocol["protocol_sha256"],
        },
        "evaluated_endpoint_steps": list(evaluated_steps),
        "endpoint_decisions": [dict(decision) for decision in endpoint_decisions],
        "commands": [dict(record) for record in commands],
        "conditions": list(CONDITIONS),
        "hard32_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "other_benchmarks_accessed": False,
    }
    bound["receipt_sha256"] = selector.canonical_sha256(bound)
    return bound


def screen_endpoints(
    *,
    run_root: Path | str,
    completion_receipt: Path | str,
    environment: Mapping[str, str] | None = None,
    command_runner: CommandRunner = run_external_command,
    gate_analyzer: GateAnalyzer = recompute_gate,
    selector_runner: SelectorRunner = run_selector,
    source_validator: SourceValidator = validate_source,
    checkpoint_validator: CheckpointValidator = validate_endpoint_checkpoint,
) -> dict[str, Any]:
    runtime_environment = dict(os.environ if environment is None else environment)
    validate_static_contract()
    validate_environment(runtime_environment)
    root, completion = validate_production_completion(
        run_root=run_root,
        completion_receipt=completion_receipt,
    )
    source = source_validator()
    checkpoints = [
        checkpoint_validator(root, step, source) for step in ENDPOINT_STEPS
    ]
    _require(
        [checkpoint.get("global_step") for checkpoint in checkpoints]
        == list(ENDPOINT_STEPS),
        "preflight checkpoint endpoint sequence differs",
    )
    coverage_by_step = {
        int(checkpoint["global_step"]): selector.validate_endpoint_checkpoint_coverage(
            checkpoint,
            step=int(checkpoint["global_step"]),
        )
        for checkpoint in checkpoints
    }
    checkpoints_by_step = {
        int(checkpoint["global_step"]): checkpoint for checkpoint in checkpoints
    }
    paths = create_screen_paths(root)
    protocol = _protocol_payload(
        run_root=root,
        paths=paths,
        completion=completion,
        source=source,
        checkpoints=checkpoints,
    )
    selector.atomic_write_json(paths.protocol, protocol)

    command_environment = _command_environment(runtime_environment)
    command_records: list[dict[str, Any]] = []
    evaluated_steps: list[int] = []
    endpoint_decisions: list[dict[str, Any]] = []
    last_selection_eligible = False
    for step in ENDPOINT_STEPS:
        evaluate = evaluation_command(run_root=root, paths=paths, step=step)
        evaluate_status = command_runner(evaluate, PROJECT_ROOT, command_environment)
        command_records.append(
            {
                "step": step,
                "kind": "train32_generation",
                "argv": evaluate,
                "returncode": evaluate_status,
            }
        )
        _require(
            evaluate_status == 0,
            f"checkpoint-{step} Train32 generation failed with status "
            f"{evaluate_status}",
        )

        gate = gate_command(paths=paths, step=step)
        gate_status = command_runner(gate, PROJECT_ROOT, command_environment)
        command_records.append(
            {
                "step": step,
                "kind": "focused_gate",
                "argv": gate,
                "returncode": gate_status,
            }
        )
        _require(
            gate_status in {0, 1},
            f"checkpoint-{step} focused gate failed with status {gate_status}",
        )
        recorded = selector.load_json_object(
            paths.gate_file(step),
            description=f"checkpoint-{step} focused gate",
        )
        recomputed = gate_analyzer(paths.results_dir(step))
        _require(
            recorded == recomputed,
            f"checkpoint-{step} focused gate differs from independent recomputation",
        )
        benchmark_gate_passed = recomputed.get("all_gates_passed") is True
        _require(
            gate_status == (0 if benchmark_gate_passed else 1),
            f"checkpoint-{step} focused gate exit status differs from report",
        )
        checkpoint = checkpoints_by_step[step]
        coverage_evidence = coverage_by_step[step]
        full_coverage = coverage_evidence["full_trainable_family_coverage"] is True
        last_selection_eligible = benchmark_gate_passed and full_coverage
        decision = {
            "global_step": step,
            "benchmark_gate_passed": benchmark_gate_passed,
            "full_coverage": full_coverage,
            "selection_eligible": last_selection_eligible,
            "coverage_evidence": coverage_evidence,
        }
        endpoint_decisions.append(decision)
        command_records[-1].update(decision)
        evaluated_steps.append(step)
        if last_selection_eligible:
            break

    specs = [
        selector.EndpointSpec(
            step=step,
            results_dir=paths.results_dir(step),
            gate_file=paths.gate_file(step),
        )
        for step in ENDPOINT_STEPS
    ]
    receipt = validate_selector_receipt(selector_runner(root, specs))
    _require(
        receipt.get("evaluated_endpoint_steps") == evaluated_steps,
        "selector evaluated endpoint sequence differs from the screening driver",
    )
    selector_endpoints = receipt.get("evaluated_endpoints")
    _require(
        isinstance(selector_endpoints, list)
        and len(selector_endpoints) == len(endpoint_decisions)
        and all(
            isinstance(record, Mapping)
            and {
                "global_step": record.get("global_step"),
                "benchmark_gate_passed": record.get("benchmark_gate_passed"),
                "full_coverage": record.get("full_coverage"),
                "selection_eligible": record.get("selection_eligible"),
                "coverage_evidence": record.get("coverage_evidence"),
            }
            == decision
            and record.get("checkpoint")
            == checkpoints_by_step[decision["global_step"]]
            for record, decision in zip(selector_endpoints, endpoint_decisions)
        ),
        "selector endpoint eligibility differs from the screening driver",
    )
    passed = receipt.get("status") == "pass"
    _require(
        passed == last_selection_eligible,
        "selector status differs from endpoint stop policy",
    )
    if passed:
        selected_checkpoint = receipt.get("selected_checkpoint")
        _require(
            isinstance(selected_checkpoint, Mapping)
            and selected_checkpoint.get("global_step") == evaluated_steps[-1]
            and selected_checkpoint
            == checkpoints_by_step[evaluated_steps[-1]]
            and selected_checkpoint.get("full_trainable_family_coverage") is True
            and receipt.get("selected_endpoint_eligibility")
            == {
                "benchmark_gate_passed": True,
                "full_coverage": True,
                "selection_eligible": True,
            }
            and receipt.get("authorization")
            == {
                "hard32_authorized": True,
                "selected_checkpoint_only": True,
                "full_validation": False,
                "test": False,
                "other_benchmarks": False,
            },
            "passing selector receipt does not authorize only the selected checkpoint",
        )
    else:
        _require(
            evaluated_steps == list(ENDPOINT_STEPS)
            and receipt.get("authorization")
            == {
                "hard32_authorized": False,
                "selected_checkpoint_only": False,
                "full_validation": False,
                "test": False,
                "other_benchmarks": False,
            },
            "failed selector receipt must be a four-endpoint unauthorized fallback",
        )

    bound = _bind_driver_receipt(
        receipt,
        paths=paths,
        completion=completion,
        protocol=protocol,
        evaluated_steps=evaluated_steps,
        endpoint_decisions=endpoint_decisions,
        commands=command_records,
    )
    selector.atomic_write_json(paths.selection_receipt, bound)
    return bound


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = screen_endpoints(
            run_root=args.run_root,
            completion_receipt=args.completion_receipt,
        )
    except (EndpointScreenError, selector.SelectionError, ValueError, OSError) as exc:
        print(f"ERROR: scene_hard_failure_endpoint_screen_failed: {exc}")
        return 2
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
