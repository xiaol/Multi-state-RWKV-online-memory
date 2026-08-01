#!/usr/bin/env python3
"""Select one hard-scene checkpoint from Train32 evidence only.

The selector accepts exactly the four predeclared generation endpoints.  For
each endpoint, in order, it validates the saved checkpoint, recomputes the
focused recovery gate from the existing Train32 generation bundle, and checks
that the recorded gate is byte-bound evidence for that evaluation.  Evaluation
stops at the first passing endpoint.

If none of the four endpoints passes, a deterministic diagnostic fallback is
recorded, but the receipt does not authorize Hard32.  This module never accepts
or resolves validation, test, holdout, Hard32, or full-validation paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


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
SCHEMA = "rwkv_ms_scene_hard_failure_train_overfit_selection.v1"
TASK = "scene-v4-current"
STAGE = "train_overfit"
ENDPOINT_STEPS = (16, 32, 48, 64)
SELECTION_POLICY = "first_passing_train32_generation_endpoint_v1"
FALLBACK_POLICY = (
    "highest_passed_gate_count_then_state_only_strict_f1_then_normal_full_"
    "strict_f1_then_earliest_endpoint_v1"
)
GATE_FILENAME = "focused_recovery_gate.json"
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed"
SHA256_RE = re.compile(r"[0-9a-f]{64}")

REQUIRED_CHECKPOINT_ARTIFACTS = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "training_protocol.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    train_contract.ROW_OBJECTIVE_AUDIT_FILENAME,
    run_audit.AUDIT_FILENAME,
)

_PROTECTED_EXACT_COMPONENTS = frozenset(
    {"val", "validation", "test", "holdout", "hard32", "full170"}
)
_PROTECTED_FRAGMENTS = ("hard32", "holdout", "full170")


class SelectionError(ValueError):
    """Raised when train-only checkpoint selection evidence differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def reject_protected_path(path: Path | str, *, description: str) -> Path:
    candidate = _lexical_absolute(path)
    for part in candidate.parts:
        lowered = part.casefold()
        stem = Path(lowered).stem
        if (
            lowered in _PROTECTED_EXACT_COMPONENTS
            or stem in _PROTECTED_EXACT_COMPONENTS
            or any(fragment in lowered for fragment in _PROTECTED_FRAGMENTS)
        ):
            raise SelectionError(
                f"train-only selector forbids protected path for {description}: "
                f"{candidate}"
            )
    return candidate


def _reject_symlink_components(path: Path, *, description: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        _require(
            not current.is_symlink(),
            f"train-only selector forbids symlink component for {description}: "
            f"{current}",
        )


def require_directory(path: Path | str, *, description: str) -> Path:
    candidate = reject_protected_path(path, description=description)
    _reject_symlink_components(candidate, description=description)
    _require(
        candidate.is_dir() and not candidate.is_symlink(),
        f"missing train-only {description}: {candidate}",
    )
    return candidate


def require_regular_file(path: Path | str, *, description: str) -> Path:
    candidate = reject_protected_path(path, description=description)
    _reject_symlink_components(candidate, description=description)
    _require(
        candidate.is_file()
        and not candidate.is_symlink()
        and candidate.stat().st_size > 0,
        f"missing, empty, or linked train-only {description}: {candidate}",
    )
    return candidate


def require_output_path(path: Path | str, *, description: str) -> Path:
    candidate = reject_protected_path(path, description=description)
    _reject_symlink_components(candidate.parent, description=f"{description} parent")
    _require(
        candidate.parent.is_dir() and not candidate.parent.is_symlink(),
        f"missing train-only {description} parent: {candidate.parent}",
    )
    _require(
        not candidate.is_symlink(),
        f"train-only selector forbids symlink output for {description}: {candidate}",
    )
    return candidate


def load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    locked = require_regular_file(path, description=description)
    try:
        payload = json.loads(locked.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"invalid JSON in {description}: {locked}") from exc
    _require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    _require(
        isinstance(recorded, str)
        and SHA256_RE.fullmatch(recorded) is not None
        and recorded == canonical_sha256(unsigned),
        f"{description} {field} differs",
    )
    return recorded


def artifact_binding(path: Path, *, description: str) -> dict[str, Any]:
    locked = require_regular_file(path, description=description)
    return {
        "path": str(locked),
        "bytes": locked.stat().st_size,
        "sha256": sha256_file(locked),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class EndpointSpec:
    step: int
    results_dir: Path
    gate_file: Path


@dataclass(frozen=True)
class EndpointEvidence:
    step: int
    checkpoint: dict[str, Any]
    evaluation: dict[str, Any]
    gate: dict[str, Any]
    report: dict[str, Any]
    fallback_rank: tuple[int, float, float, int]

    @property
    def passed(self) -> bool:
        return self.report.get("all_gates_passed") is True


def preflight_endpoint_specs(
    run_root: Path | str,
    endpoint_specs: Sequence[EndpointSpec],
) -> tuple[Path, list[EndpointSpec]]:
    """Validate all endpoint paths lexically before reading any candidate."""

    root = reject_protected_path(run_root, description="run root")
    _require(
        len(endpoint_specs) == len(ENDPOINT_STEPS)
        and [spec.step for spec in endpoint_specs] == list(ENDPOINT_STEPS),
        "endpoint schedule must be exactly 16,32,48,64 in that order",
    )
    normalized: list[EndpointSpec] = []
    for spec in endpoint_specs:
        results_dir = reject_protected_path(
            spec.results_dir,
            description=f"checkpoint-{spec.step} Train32 results",
        )
        gate_file = reject_protected_path(
            spec.gate_file,
            description=f"checkpoint-{spec.step} focused gate",
        )
        _require(
            gate_file == results_dir / GATE_FILENAME,
            f"checkpoint-{spec.step} focused gate must be {GATE_FILENAME} in its "
            "Train32 results directory",
        )
        checkpoint = root / "trainer" / f"checkpoint-{spec.step}"
        reject_protected_path(
            checkpoint,
            description=f"checkpoint-{spec.step} artifact directory",
        )
        normalized.append(EndpointSpec(spec.step, results_dir, gate_file))
    return root, normalized


def validate_source_binding() -> dict[str, Any]:
    """Bind the source lock and exact Train32 source without held-out access."""

    source_lock_path = require_regular_file(
        train_contract.SOURCE_LOCK,
        description="source lock",
    )
    source_manifest_path = require_regular_file(
        train_contract.SOURCE_MANIFEST,
        description="source manifest",
    )
    train_file = require_regular_file(
        train_contract.TRAIN_FILE,
        description="Train32 dataset",
    )
    lock = train_contract.validate_source_lock(source_lock_path)
    source = state_eval.validate_focused_train_source_manifest(
        source_manifest_path,
        dataset_file=train_file,
    )
    source_manifest = source.get("source_manifest")
    dataset = source.get("dataset")
    pair_manifest = source.get("pair_manifest")
    _require(
        isinstance(source_manifest, Mapping)
        and isinstance(dataset, Mapping)
        and isinstance(pair_manifest, Mapping),
        "validated Train32 source binding is incomplete",
    )
    locked_artifacts = lock.get("training_artifacts")
    _require(
        isinstance(locked_artifacts, Mapping),
        "source lock training artifacts are missing",
    )
    expected = {
        "source_manifest_file_sha256": source_manifest.get("file_sha256"),
        "source_manifest_sha256": source_manifest.get("manifest_sha256"),
        "train_file_sha256": dataset.get("sha256"),
        "pair_manifest_file_sha256": pair_manifest.get("file_sha256"),
        "pair_manifest_sha256": pair_manifest.get("manifest_sha256"),
        "entries_sha256": pair_manifest.get("entries_sha256"),
    }
    evaluator_expected = {
        "source_manifest_file_sha256": (
            state_eval.SCENE_HARD_FAILURE_SOURCE_MANIFEST_FILE_SHA256
        ),
        "source_manifest_sha256": state_eval.SCENE_HARD_FAILURE_SOURCE_MANIFEST_SHA256,
        "train_file_sha256": state_eval.SCENE_HARD_FAILURE_TRAIN_FILE_SHA256,
        "pair_manifest_file_sha256": (
            state_eval.SCENE_HARD_FAILURE_PAIR_MANIFEST_FILE_SHA256
        ),
        "pair_manifest_sha256": state_eval.SCENE_HARD_FAILURE_PAIR_MANIFEST_SHA256,
        "entries_sha256": state_eval.SCENE_HARD_FAILURE_PAIR_ENTRIES_SHA256,
    }
    _require(
        expected == evaluator_expected,
        "Train32 source identity differs between the source lock and protected "
        "evaluator",
    )
    for filename, field in (
        ("source_manifest.json", "source_manifest_file_sha256"),
        ("train.jsonl", "train_file_sha256"),
        ("pair_manifest.json", "pair_manifest_file_sha256"),
    ):
        locked = locked_artifacts.get(filename)
        _require(
            isinstance(locked, Mapping) and locked.get("sha256") == expected[field],
            f"source-lock {filename} differs from evaluator source identity",
        )
    protected = lock.get("protected_evaluation")
    _require(
        isinstance(protected, Mapping)
        and all(
            isinstance(protected.get(name), Mapping)
            and protected[name].get("included") is False
            and protected[name].get("path") is None
            for name in ("official_validation", "hard32", "official_test")
        ),
        "source lock includes protected evaluation data",
    )
    return {
        "source_lock_path": str(source_lock_path),
        "source_lock_file_sha256": sha256_file(source_lock_path),
        "source_lock_sha256": lock["lock_sha256"],
        "source_manifest_path": str(source_manifest_path),
        **expected,
        "protected_evaluation_accessed": False,
    }


def validate_full_trainable_family_coverage(
    audit: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, int]:
    """Require all 27 trainable tensor families to change in all 42 layers."""

    expected_family_count = len(run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES)
    expected_layer_count = len(train_contract.TARGET_LAYERS)
    expected_tensor_count = expected_family_count * expected_layer_count
    expected_coverage = {
        suffix: expected_layer_count
        for suffix in run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    _require(
        expected_family_count == 27
        and expected_layer_count == 42
        and expected_tensor_count == 1134,
        "selector trainable adapter topology constants differ from 27 families x "
        "42 layers",
    )
    change = audit.get("adapter_change")
    _require(
        isinstance(change, Mapping),
        f"checkpoint-{step} adapter-change evidence is missing",
    )
    _require(
        audit.get("trainable_tensor_family_count") == expected_family_count
        and audit.get("target_layer_count") == expected_layer_count
        and audit.get("trainable_family_layer_coverage") == expected_coverage
        and audit.get("full_trainable_family_coverage") is True,
        f"checkpoint-{step} top-level audit does not prove complete 27-family x "
        "42-layer coverage",
    )
    recomputed_coverage = validate_recomputed_adapter_change(change, step=step)
    _require(
        recomputed_coverage == expected_coverage,
        f"checkpoint-{step} adapter-change coverage differs",
    )
    _require(
        audit.get("optimizer_contains_only_declared_trainable_adapter_state_count")
        is True
        and audit.get(
            "base_model_parameter_values_not_materialized_in_adapter_checkpoint"
        )
        is True
        and audit.get("nontrainable_adapter_tensors_unchanged") is True,
        f"checkpoint-{step} trainable/frozen parameter isolation evidence differs",
    )
    optimizer = audit.get("optimizer_update")
    _require(
        isinstance(optimizer, Mapping)
        and optimizer.get("optimizer_parameter_state_count") == expected_tensor_count
        and optimizer.get("declared_trainable_adapter_tensor_count")
        == expected_tensor_count
        and optimizer.get("all_optimizer_parameter_states_at_checkpoint_step") is True,
        f"checkpoint-{step} optimizer coverage does not bind all 1,134 trainable "
        "adapter tensors",
    )
    return expected_coverage


def validate_recomputed_adapter_change(
    change: Mapping[str, Any],
    *,
    step: int,
) -> dict[str, int]:
    """Require complete current-byte change evidence for every trainable tensor."""

    expected_family_count = len(run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES)
    expected_layer_count = len(train_contract.TARGET_LAYERS)
    expected_tensor_count = expected_family_count * expected_layer_count
    expected_coverage = {
        suffix: expected_layer_count
        for suffix in run_audit.TRAINABLE_ADAPTER_TENSOR_SUFFIXES
    }
    _require(
        change.get("trainable_tensor_family_count") == expected_family_count
        and change.get("target_layer_count") == expected_layer_count
        and change.get("expected_trainable_tensor_count") == expected_tensor_count
        and change.get("changed_trainable_tensor_count") == expected_tensor_count
        and change.get("changed_nontrainable_tensor_count") == 0
        and change.get("trainable_family_layer_coverage") == expected_coverage
        and change.get("missing_trainable_family_layers") == {}
        and change.get("full_trainable_family_coverage") is True,
        f"checkpoint-{step} adapter-change evidence does not prove complete "
        "27-family x 42-layer coverage",
    )
    _require(
        change.get("full_trainable_family_coverage_required")
        is (step == train_contract.TOTAL_OPTIMIZER_STEPS),
        f"checkpoint-{step} audit coverage requirement flag differs",
    )
    return expected_coverage


def validate_current_adapter_change(
    run_root: Path,
    checkpoint: Path,
    *,
    step: int,
    audited_change: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute adapter changes from stable current bytes and bind the evidence."""

    initial_dir = require_directory(
        run_root / "initial_adapter",
        description="initial adapter directory",
    )
    evidence_paths = {
        "initial_adapter_manifest": initial_dir / "initial_adapter_manifest.json",
        "initial_adapter": initial_dir / "delta_mem_adapter.pt",
        "checkpoint_adapter": checkpoint / "delta_mem_adapter.pt",
    }
    before = {
        name: artifact_binding(path, description=name.replace("_", " "))
        for name, path in evidence_paths.items()
    }
    try:
        initial_manifest = load_json_object(
            evidence_paths["initial_adapter_manifest"],
            description="initial adapter manifest",
        )
        initial_adapter = run_audit.load_finite_adapter(
            evidence_paths["initial_adapter"]
        )
        checkpoint_adapter = run_audit.load_finite_adapter(
            evidence_paths["checkpoint_adapter"]
        )
        trainable_names = run_audit._validate_initial_adapter_topology(
            initial_manifest.get("topology"),
            initial_adapter,
        )
        recomputed_change = run_audit.adapter_change_record(
            initial_adapter,
            checkpoint_adapter,
            trainable_names=trainable_names,
            checkpoint_step=step,
            smoke=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SelectionError(
            f"checkpoint-{step} current adapter change validation failed: {exc}"
        ) from exc

    after = {
        name: artifact_binding(path, description=name.replace("_", " "))
        for name, path in evidence_paths.items()
    }
    _require(
        after == before,
        f"checkpoint-{step} adapter evidence changed during current-byte validation",
    )
    coverage = validate_recomputed_adapter_change(recomputed_change, step=step)
    _require(
        recomputed_change == dict(audited_change),
        f"checkpoint-{step} current adapter change differs from audited evidence",
    )
    return {
        **before,
        "recomputed_adapter_change_canonical_sha256": canonical_sha256(
            recomputed_change
        ),
        "trainable_tensor_family_count": len(coverage),
        "target_layer_count": len(train_contract.TARGET_LAYERS),
        "full_trainable_family_coverage": True,
        "frozen_adapter_tensors_unchanged": True,
    }


def validate_checkpoint(
    run_root: Path,
    *,
    step: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    _require(step in ENDPOINT_STEPS, f"checkpoint-{step} is not a generation endpoint")
    checkpoint = require_directory(
        run_root / "trainer" / f"checkpoint-{step}",
        description=f"checkpoint-{step}",
    )
    artifacts = {
        name: artifact_binding(
            checkpoint / name,
            description=f"checkpoint-{step} {name}",
        )
        for name in REQUIRED_CHECKPOINT_ARTIFACTS
    }

    protocol = load_json_object(
        checkpoint / "training_protocol.json",
        description=f"checkpoint-{step} training protocol",
    )
    trainer_state = load_json_object(
        checkpoint / "trainer_state.json",
        description=f"checkpoint-{step} trainer state",
    )
    row_audit = load_json_object(
        checkpoint / train_contract.ROW_OBJECTIVE_AUDIT_FILENAME,
        description=f"checkpoint-{step} row objective audit",
    )
    run_audit._validate_protocol(protocol, smoke=False)
    run_audit._validate_trainer_state(trainer_state, step=step)
    run_audit._validate_row_audit(row_audit, step=step, smoke=False)

    audit = load_json_object(
        checkpoint / run_audit.AUDIT_FILENAME,
        description=f"checkpoint-{step} checkpoint audit",
    )
    validate_self_hash(
        audit,
        field="receipt_sha256",
        description=f"checkpoint-{step} checkpoint audit",
    )
    _require(
        audit.get("schema") == run_audit.AUDIT_SCHEMA
        and audit.get("run_root") == str(run_root)
        and audit.get("checkpoint") == str(checkpoint)
        and audit.get("checkpoint_optimizer_step") == step
        and audit.get("run_mode") == train_contract.PRODUCTION_RUN_MODE
        and audit.get("objective_version") == train_contract.OBJECTIVE_VERSION
        and audit.get("source_lock_sha256") == source.get("source_lock_sha256")
        and audit.get("nontrainable_adapter_tensors_unchanged") is True
        and audit.get("row_audit_complete") is True,
        f"checkpoint-{step} production audit binding differs",
    )
    family_coverage = validate_full_trainable_family_coverage(audit, step=step)
    current_adapter_validation = validate_current_adapter_change(
        run_root,
        checkpoint,
        step=step,
        audited_change=audit["adapter_change"],
    )
    _require(
        current_adapter_validation["checkpoint_adapter"]
        == artifacts["delta_mem_adapter.pt"],
        f"checkpoint-{step} adapter binding changed during checkpoint validation",
    )
    return {
        "path": str(checkpoint),
        "global_step": step,
        "artifacts": artifacts,
        "trainable_tensor_family_count": len(family_coverage),
        "target_layer_count": len(train_contract.TARGET_LAYERS),
        "trainable_family_layer_coverage": family_coverage,
        "full_trainable_family_coverage": True,
        "checkpoint_audit_receipt_sha256": audit["receipt_sha256"],
        "current_adapter_validation": current_adapter_validation,
    }


def _finite_metric(value: Any, *, description: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{description} must be finite",
    )
    return float(value)


def _fallback_rank(report: Mapping[str, Any], *, step: int) -> tuple[int, float, float, int]:
    gates = report.get("gates")
    scores = report.get("condition_scores")
    _require(
        isinstance(gates, Mapping) and isinstance(scores, Mapping),
        f"checkpoint-{step} gate diagnostics are incomplete",
    )
    passed_gates = sum(
        1
        for gate in gates.values()
        if isinstance(gate, Mapping) and gate.get("passed") is True
    )
    state_score = scores.get("state_only")
    normal_score = scores.get("normal_full")
    _require(
        isinstance(state_score, Mapping) and isinstance(normal_score, Mapping),
        f"checkpoint-{step} focused condition scores are incomplete",
    )
    return (
        passed_gates,
        _finite_metric(
            state_score.get("primary_metric"),
            description=f"checkpoint-{step} state-only strict F1",
        ),
        _finite_metric(
            normal_score.get("primary_metric"),
            description=f"checkpoint-{step} normal-full strict F1",
        ),
        -step,
    )


def _validate_evaluation_contract(
    contract: Any,
    *,
    step: int,
    checkpoint: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(contract, Mapping), "focused evaluation contract is missing")
    artifacts = checkpoint["artifacts"]
    expected_checkpoint = {
        "memory_dir": checkpoint["path"],
        "adapter_sha256": artifacts["delta_mem_adapter.pt"]["sha256"],
        "config_sha256": artifacts["delta_mem_config.json"]["sha256"],
    }
    _require(
        contract.get("name") == state_eval.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
        and contract.get("task") == TASK
        and contract.get("split") == "train"
        and contract.get("rows") == state_eval.SCENE_HARD_FAILURE_ROWS
        and contract.get("conditions") == list(state_eval.SCENE_FOCUSED_CONDITIONS)
        and contract.get("checkpoint") == expected_checkpoint,
        f"checkpoint-{step} focused Train32 evaluation contract differs",
    )
    train_source = contract.get("train_source")
    _require(isinstance(train_source, Mapping), "evaluation Train32 source is missing")
    source_manifest = train_source.get("source_manifest")
    dataset = train_source.get("dataset")
    pair_manifest = train_source.get("pair_manifest")
    _require(
        isinstance(source_manifest, Mapping)
        and source_manifest.get("path") == source.get("source_manifest_path")
        and source_manifest.get("file_sha256")
        == source.get("source_manifest_file_sha256")
        and source_manifest.get("manifest_sha256")
        == source.get("source_manifest_sha256")
        and isinstance(dataset, Mapping)
        and dataset.get("sha256") == source.get("train_file_sha256")
        and dataset.get("split") == "train"
        and dataset.get("rows") == state_eval.SCENE_HARD_FAILURE_ROWS
        and isinstance(pair_manifest, Mapping)
        and pair_manifest.get("file_sha256")
        == source.get("pair_manifest_file_sha256")
        and pair_manifest.get("manifest_sha256")
        == source.get("pair_manifest_sha256")
        and pair_manifest.get("entries_sha256") == source.get("entries_sha256")
        and train_source.get("protected_evaluation_accessed") is False,
        f"checkpoint-{step} evaluation source binding differs",
    )
    return dict(contract)


def load_endpoint_evidence(
    spec: EndpointSpec,
    *,
    checkpoint: dict[str, Any],
    source: Mapping[str, Any],
) -> EndpointEvidence:
    results_dir = require_directory(
        spec.results_dir,
        description=f"checkpoint-{spec.step} Train32 results",
    )
    gate_file = require_regular_file(
        spec.gate_file,
        description=f"checkpoint-{spec.step} focused gate",
    )
    _require(
        gate_file == results_dir / GATE_FILENAME,
        f"checkpoint-{spec.step} gate path differs after validation",
    )

    try:
        recomputed = focused_gate.analyze_results_dir(results_dir, stage=STAGE)
    except (ValueError, OSError) as exc:
        raise SelectionError(
            f"checkpoint-{spec.step} Train32 gate recomputation failed: {exc}"
        ) from exc
    recorded = load_json_object(
        gate_file,
        description=f"checkpoint-{spec.step} recorded focused gate",
    )
    _require(
        recorded == recomputed,
        f"checkpoint-{spec.step} recorded gate differs from recomputed Train32 gate",
    )
    expected_status = (
        "diagnostic_pass" if recomputed.get("all_gates_passed") is True
        else "diagnostic_fail"
    )
    _require(
        recomputed.get("schema") == focused_gate.SCHEMA
        and recomputed.get("status") == expected_status
        and recomputed.get("stage") == STAGE
        and recomputed.get("task") == TASK
        and recomputed.get("rows") == state_eval.SCENE_HARD_FAILURE_ROWS
        and recomputed.get("source_indices")
        == list(range(state_eval.SCENE_HARD_FAILURE_ROWS)),
        f"checkpoint-{spec.step} gate is not exact Train32 evidence",
    )
    gate_input = recomputed.get("input")
    _require(
        isinstance(gate_input, Mapping)
        and gate_input.get("results_dir") == str(results_dir),
        f"checkpoint-{spec.step} gate input directory differs",
    )
    fingerprint = gate_input.get("evaluation_fingerprint")
    _require(
        isinstance(fingerprint, str) and SHA256_RE.fullmatch(fingerprint) is not None,
        f"checkpoint-{spec.step} evaluation fingerprint is invalid",
    )
    contract = _validate_evaluation_contract(
        gate_input.get("evaluation_contract"),
        step=spec.step,
        checkpoint=checkpoint,
        source=source,
    )
    manifest = load_json_object(
        results_dir / "manifest.json",
        description=f"checkpoint-{spec.step} evaluation manifest",
    )
    _require(
        manifest.get("fingerprint") == fingerprint
        and manifest.get("evaluation_contract") == contract,
        f"checkpoint-{spec.step} manifest fingerprint or contract differs",
    )
    progress = load_json_object(
        results_dir / "progress.json",
        description=f"checkpoint-{spec.step} evaluation progress",
    )
    expected_records = state_eval.SCENE_HARD_FAILURE_ROWS * len(
        state_eval.SCENE_FOCUSED_CONDITIONS
    )
    _require(
        progress.get("fingerprint") == fingerprint
        and progress.get("completed") == expected_records
        and progress.get("expected") == expected_records
        and progress.get("complete") is True,
        f"checkpoint-{spec.step} evaluation progress is incomplete or differs",
    )
    evaluation_artifacts = {
        name: artifact_binding(
            results_dir / name,
            description=f"checkpoint-{spec.step} evaluation {name}",
        )
        for name in (
            "manifest.json",
            "summary.json",
            "progress.json",
            *(f"{condition}.jsonl" for condition in state_eval.SCENE_FOCUSED_CONDITIONS),
        )
    }
    evaluation = {
        "results_dir": str(results_dir),
        "fingerprint": fingerprint,
        "contract_canonical_sha256": canonical_sha256(contract),
        "artifacts": evaluation_artifacts,
    }
    gate_binding = {
        "focused_gate_path": str(gate_file),
        "file_sha256": sha256_file(gate_file),
        "canonical_sha256": canonical_sha256(recorded),
        "evaluation_fingerprint": fingerprint,
    }
    return EndpointEvidence(
        step=spec.step,
        checkpoint=checkpoint,
        evaluation=evaluation,
        gate=gate_binding,
        report=recorded,
        fallback_rank=_fallback_rank(recorded, step=spec.step),
    )


def _endpoint_receipt_record(evidence: EndpointEvidence) -> dict[str, Any]:
    return {
        "global_step": evidence.step,
        "gate_passed": evidence.passed,
        "fallback_rank": {
            "passed_gate_count": evidence.fallback_rank[0],
            "state_only_strict_f1": evidence.fallback_rank[1],
            "normal_full_strict_f1": evidence.fallback_rank[2],
            "earliest_endpoint_tiebreak": evidence.step,
        },
        "checkpoint": evidence.checkpoint,
        "evaluation": evidence.evaluation,
        "gate": evidence.gate,
    }


def build_selection_receipt(
    *,
    source: Mapping[str, Any],
    evaluated: Sequence[EndpointEvidence],
    selected: EndpointEvidence,
    passed: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    _require(bool(evaluated), "selector evaluated no Train32 endpoint")
    _require(selected in evaluated, "selected endpoint was not evaluated")
    if passed:
        _require(
            selected.passed
            and selected is next(item for item in evaluated if item.passed),
            "passing selection is not the first passing endpoint",
        )
    else:
        _require(
            len(evaluated) == len(ENDPOINT_STEPS)
            and not any(item.passed for item in evaluated)
            and selected == max(evaluated, key=lambda item: item.fallback_rank),
            "diagnostic fallback selection differs from the frozen policy",
        )
    authorization = (
        {
            "hard32_authorized": True,
            "selected_checkpoint_only": True,
            "full_validation": False,
            "test": False,
            "other_benchmarks": False,
        }
        if passed
        else {
            "hard32_authorized": False,
            "selected_checkpoint_only": False,
            "full_validation": False,
            "test": False,
            "other_benchmarks": False,
        }
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": (
            datetime.now(timezone.utc).isoformat() if created_at is None else created_at
        ),
        "status": "pass" if passed else "fail",
        "stage": STAGE,
        "task": TASK,
        "selection_policy": SELECTION_POLICY,
        "fallback_policy": FALLBACK_POLICY,
        "endpoint_schedule": list(ENDPOINT_STEPS),
        "evaluated_endpoint_steps": [item.step for item in evaluated],
        "evaluated_endpoints": [
            _endpoint_receipt_record(item) for item in evaluated
        ],
        "selected_checkpoint": selected.checkpoint,
        "selection_reason": (
            "first_passing_train32_generation_endpoint"
            if passed
            else "diagnostic_fallback_no_train32_endpoint_passed"
        ),
        "source": dict(source),
        "evaluation": selected.evaluation,
        "gate": selected.gate,
        "authorization": authorization,
        "held_out_access": HARD32_ACCESS_POLICY,
        "hard32_accessed": False,
        "full_validation_accessed": False,
        "test_accessed": False,
        "other_benchmarks_accessed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def select_checkpoint(
    *,
    run_root: Path | str,
    endpoint_specs: Sequence[EndpointSpec],
) -> dict[str, Any]:
    root, specs = preflight_endpoint_specs(run_root, endpoint_specs)
    root = require_directory(root, description="run root")
    source_binding = validate_source_binding()
    evaluated: list[EndpointEvidence] = []
    selected: EndpointEvidence | None = None
    for spec in specs:
        checkpoint = validate_checkpoint(
            root,
            step=spec.step,
            source=source_binding,
        )
        evidence = load_endpoint_evidence(
            spec,
            checkpoint=checkpoint,
            source=source_binding,
        )
        evaluated.append(evidence)
        if evidence.passed:
            selected = evidence
            break
    passed = selected is not None
    if selected is None:
        _require(
            len(evaluated) == len(ENDPOINT_STEPS),
            "all endpoints must be evaluated before diagnostic fallback",
        )
        selected = max(evaluated, key=lambda item: item.fallback_rank)
    return build_selection_receipt(
        source=source_binding,
        evaluated=evaluated,
        selected=selected,
        passed=passed,
    )


def parse_endpoint_spec(values: Sequence[str]) -> EndpointSpec:
    step_raw, results_dir, gate_file = values
    try:
        step = int(step_raw)
    except ValueError as exc:
        raise SelectionError(f"invalid endpoint step: {step_raw}") from exc
    return EndpointSpec(step, Path(results_dir), Path(gate_file))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        action="append",
        nargs=3,
        metavar=("STEP", "TRAIN32_RESULTS_DIR", "FOCUSED_GATE_FILE"),
        required=True,
        help="repeat exactly for steps 16, 32, 48, and 64 in that order",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = [parse_endpoint_spec(values) for values in args.endpoint]
        receipt_path = require_output_path(
            args.receipt,
            description="selection receipt",
        )
        if receipt_path.exists() and not args.overwrite:
            raise SelectionError(f"selection receipt already exists: {receipt_path}")
        receipt = select_checkpoint(
            run_root=args.run_root,
            endpoint_specs=specs,
        )
        atomic_write_json(receipt_path, receipt)
    except (SelectionError, ValueError, OSError) as exc:
        print(f"ERROR: scene_hard_failure_checkpoint_selection_failed: {exc}")
        return 2
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
