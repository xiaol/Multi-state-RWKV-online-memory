#!/usr/bin/env python3
"""Run the Train32-only Scene-Memory V10 cycle progression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_memory_v9_gate as v9,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v10_launch_contract as launch,
)


GATE_CONTRACT = "scene_memory_v10_train32_cycle_progression_gate"
GATE_RECORD_SCHEMA = "rwkv_ms_scene_memory_v10_train32_gate_record.v1"
GATE_SUMMARY_SCHEMA = "rwkv_ms_scene_memory_v10_train32_gate_summary.v1"
GATE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v10_train32_gate_manifest.v1"
GATE_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v10_train32_gate_receipt.v1"
CONTINUATION_AUTHORIZATION_KIND = "scene_memory_v10_train32_progression_receipt"
CONDITIONS = v9.CONDITIONS
VALUE14_ORDINALS = v9.VALUE14_ORDINALS
VALUE14_SET = frozenset(VALUE14_ORDINALS)
CHECKPOINT_STEPS = launch.CHECKPOINT_STEPS
FIRST_GATE_STEP = CHECKPOINT_STEPS[0]
FINAL_GATE_STEP = CHECKPOINT_STEPS[-1]
HARD32_ACCESS_POLICY = launch.HARD32_ACCESS_POLICY
GATE_DEVICE = "cuda:0"
GATE_DTYPE = "bfloat16"
GATE_ATTN_IMPLEMENTATION = "sdpa"
GATE_NORMAL_FUSION_PROFILE = "native"
GATE_EXPECTED_MEMORY_LAYER_COUNT = 42
GATE_MAX_NEW_TOKENS = v9.DEFAULT_MAX_NEW_TOKENS

V8_CHECKPOINT56_BASELINE = dict(v9.V8_CHECKPOINT56_BASELINE)
PROGRESSION_REQUIREMENTS = dict(v9.PROGRESSION_REQUIREMENTS)
V10_OBJECTIVE = {
    **v9.V9_OBJECTIVE,
    "training_objective_version": launch.OBJECTIVE_VERSION,
    "progression_basis": "strict_generation_and_bidirectional_identity_switch_cycle_v1",
    "optimization_unit": "one_complete_seven_pair_cycle_per_optimizer_update",
    "selected_full_vocab_ce_in_total": False,
    "generated_prefix_correction_mode": launch.GENERATED_PREFIX_MODE,
    "cycle_retention_mode": launch.CYCLE_RETENTION_MODE,
    "generated_rollout_decoding": (
        "greedy_use_cache_true_exact_system_only_prompt_v1"
    ),
    "generated_replay_state_gradient": True,
    "generated_replay_read_path_gradient": True,
}


class V10EvaluationContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V10EvaluationContractError(message)


canonical_sha256 = v9.canonical_sha256
self_hash_payload = v9.self_hash_payload
atomic_write_json = v9.atomic_write_json
atomic_write_canonical_json = v9.atomic_write_canonical_json
atomic_write_jsonl = v9.atomic_write_jsonl
_artifact_binding = v9._artifact_binding
_record_with_self_hash = v9._record_with_self_hash


def _ssd_path(path: Path | str, *, description: str, ssd_root: Path) -> Path:
    try:
        return launch.require_ssd(path, description=description, ssd_root=ssd_root)
    except Exception as exc:
        raise V10EvaluationContractError(
            f"V10 {description} must stay on the SSD outside protected paths: {exc}"
        ) from exc


def _run_path(path: Path | str, *, description: str, ssd_root: Path) -> Path:
    try:
        return launch.require_v10_run_path(
            path,
            description=description,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V10EvaluationContractError(
            f"V10 {description} must stay under the locked V10 run root: {exc}"
        ) from exc


def _gate_path(path: Path | str, *, description: str, ssd_root: Path) -> Path:
    try:
        return launch.require_v10_gate_path(
            path,
            description=description,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V10EvaluationContractError(
            f"V10 {description} must stay under the locked V10 gates root: {exc}"
        ) from exc


def validate_base_model_path(
    path: Path | str,
    *,
    ssd_root: Path = launch.SSD_ROOT,
    pinned_base_model: Path = launch.PINNED_BASE_MODEL,
) -> Path:
    """Validate the pinned base model before any model artifact is opened."""

    try:
        _ssd_path(
            pinned_base_model,
            description="V10 pinned base model",
            ssd_root=ssd_root,
        )
        resolved = launch.require_exact_path(
            path,
            pinned_base_model,
            description="v10_base_model",
        )
    except Exception as exc:
        raise V10EvaluationContractError(
            "V10 base model must be the exact canonical pinned Gemma path: "
            f"{exc}"
        ) from exc
    _require(
        resolved.is_dir() and not resolved.is_symlink(),
        "V10 pinned base model must be a directory and not a symlink",
    )
    return resolved


def _regular_file(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path | None = None,
) -> Path:
    raw = launch._lexically_guard_path(path, description=description)
    resolved = raw.resolve() if ssd_root is None else _gate_path(
        raw,
        description=description,
        ssd_root=ssd_root,
    )
    _require(resolved.is_file(), f"V10 {description} is missing: {resolved}")
    return resolved


def _load_json(path: Path | str, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V10EvaluationContractError(f"V10 {description} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"V10 {description} must be an object")
    return payload


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, description=description)
    records: list[dict[str, Any]] = []
    for row_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"V10 {description} contains a blank row")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V10EvaluationContractError(
                f"V10 {description} row {row_number} is invalid JSON"
            ) from exc
        _require(isinstance(payload, dict), f"V10 {description} row must be an object")
        records.append(payload)
    return records


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    recorded = payload.get(field)
    _require(isinstance(recorded, str), f"V10 {field} is missing")
    _require(recorded == self_hash_payload(payload, hash_field=field), f"V10 {field} differs")
    return recorded


def validate_v10_train_inputs(
    *,
    data_root: Path = launch.DATA_ROOT,
    source_lock_path: Path = launch.SOURCE_LOCK,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Validate only reused V9 Train32 data and the V10 cycle view."""

    try:
        launch_data = launch.validate_data_contract(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
        base = v9.validate_v9_train_inputs(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V10EvaluationContractError(f"V10 input contract failed: {exc}") from exc
    _require(
        base["v9_schedule_entries_sha256"] == launch_data["schedule_entries_sha256"]
        and base["v9_schedule_manifest_sha256"] == launch_data["schedule_manifest_sha256"],
        "V10 reused V9 schedule identity differs",
    )
    historical_names = {
        "train32": "train32",
        "train32_rows": "train32_rows",
        "train32_pair_manifest": "pair_manifest",
        "train32_source_manifest": "source_manifest",
    }
    for artifact_name, pinned_name in historical_names.items():
        expected = launch.PINNED_HISTORICAL_TRAIN32_ARTIFACTS[pinned_name]
        _require(
            base["artifacts"].get(artifact_name)
            == {
                "path": str(expected["path"]),
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            },
            f"V10 pinned historical Train32 artifact differs: {artifact_name}",
        )
    result = dict(base)
    result.update(
        {
            "contract": GATE_CONTRACT,
            "launch_data": launch_data,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "presentation_checkpoint_steps": list(launch.PRESENTATION_CHECKPOINTS),
            "optimizer_cycles": launch_data["optimizer_cycles"],
            "hard32_access": HARD32_ACCESS_POLICY,
        }
    )
    return result


def _validate_v10_objective_protocol(
    protocol: Mapping[str, Any],
    *,
    input_contract: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": launch.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "train_sampler_mode": launch.FIXED_SAMPLER_MODE,
        "gradient_accumulation_steps": launch.GRADIENT_ACCUMULATION_STEPS,
        "scene_generation_objective_formula": launch.OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": launch.BACKWARD_MODE,
        "scene_generation_generated_prefix_correction_weight": launch.PREFIX_CORRECTION_WEIGHT,
        "scene_generation_generated_prefix_correction_mode": launch.GENERATED_PREFIX_MODE,
        "scene_generation_generated_rollout_decoding": (
            "greedy_use_cache_true_exact_system_only_prompt_v1"
        ),
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
        "scene_generation_first_error_top1_hinge_weight": 1.0,
        "scene_generation_all_target_top1_retention_weight": 1.0,
        "scene_generation_all_target_top1_retention_margin": 0.2,
        "scene_generation_selected_full_vocab_ce_in_total": False,
        "scene_generation_selected_full_vocab_ce_optimization_weight": 0.0,
        "scene_generation_cycle_retention_mode": launch.CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        "scene_generation_gradient_accumulation_pair_cycle": launch.GRADIENT_ACCUMULATION_STEPS,
        "scene_generation_pair_unit": "canonical_low_with_reciprocal_full_payload_v1",
        "scene_generation_pair_physical_batch_size": 1,
        "scene_generation_pair_directional_exposures": 2,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    _require(not mismatches, "V10 objective protocol differs: " + ", ".join(mismatches))
    artifacts = input_contract["artifacts"]
    source = protocol.get("scene_state_source_manifest")
    expected_source = {
        "path": artifacts["v9_source_manifest"]["path"],
        "file_sha256": artifacts["v9_source_manifest"]["sha256"],
        "schema": launch.v9.SOURCE_SCHEMA,
        "train_file": artifacts["train32"]["path"],
        "train_file_sha256": artifacts["train32"]["sha256"],
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": v9.v8.v7.EPISODE_CONTRACT,
    }
    _require(source == expected_source, "V10 objective source identity differs")


def validate_v10_checkpoint(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    warm = launch.validate_warm_start_contract(ssd_root=ssd_root) if warm_contract is None else dict(warm_contract)
    try:
        lineage = launch.validate_checkpoint_contract(
            Path(memory_dir),
            data=input_contract["launch_data"],
            warm=warm,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V10EvaluationContractError(f"V10 checkpoint contract failed: {exc}") from exc
    resolved = Path(str(lineage["checkpoint"]))
    step = int(lineage["checkpoint_step"])
    protocol = _load_json(resolved / "training_protocol.json", description="V10 training protocol")
    _validate_v10_objective_protocol(protocol, input_contract=input_contract)
    pairing = _load_json(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="V10 checkpoint pairing manifest",
    )
    try:
        pairing_sha256 = v9._validate_v9_pairing(
            pairing,
            protocol=protocol,
            input_contract=input_contract,
        )
    except Exception as exc:
        raise V10EvaluationContractError(f"V10 pairing contract failed: {exc}") from exc
    architecture = v9.memory_architecture_contract(resolved)
    _require(
        architecture.get("target_layers") == list(range(42))
        and architecture.get("delta_heads") == ["q", "o"]
        and architecture.get("rank") == 4
        and architecture.get("rwkv_ms_semantics_version") == 2
        and architecture.get("memory_backend") == "rwkv_ms",
        "V10 checkpoint architecture differs",
    )
    artifacts = {
        name.removesuffix(".json").removesuffix(".pt").removesuffix(".pth"): _artifact_binding(
            resolved / name,
            description=f"V10 checkpoint {name}",
        )
        for name in launch.REQUIRED_CHECKPOINT_ARTIFACTS
    }
    rng = [
        _artifact_binding(path, description=f"V10 checkpoint RNG {path.name}")
        for path in sorted(resolved.glob("rng_state*.pth"))
    ]
    lineage_path = resolved / str(lineage["lineage_filename"])
    return {
        "memory_dir": str(resolved),
        "global_step": step,
        "max_steps": step,
        "consumed_pair_presentations": launch.presentation_cursor(step),
        "artifacts": artifacts,
        "rng_state": rng,
        "training_protocol_canonical_sha256": lineage["training_protocol_sha256"],
        "pairing_manifest_sha256": pairing_sha256,
        "lineage": dict(lineage),
        "lineage_artifact": _artifact_binding(lineage_path, description="V10 checkpoint lineage"),
        "architecture": architecture,
        "objective": dict(V10_OBJECTIVE),
    }


def _mapped_previous_gate(previous: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if previous is None:
        return None
    mapped = dict(previous)
    mapped["contract"] = v9.GATE_CONTRACT
    mapped["checkpoint_step"] = int(previous["checkpoint_step"]) * 7
    if previous.get("next_checkpoint_step") is not None:
        mapped["next_checkpoint_step"] = int(previous["next_checkpoint_step"]) * 7
    return mapped


def build_v10_gate(
    *,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    pairing: Mapping[str, Any],
    checkpoint_step: int,
    previous_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(checkpoint_step in CHECKPOINT_STEPS, "V10 checkpoint step is not locked")
    if checkpoint_step == FIRST_GATE_STEP:
        _require(previous_gate is None, "V10 checkpoint-1 forbids a previous gate")
    else:
        _require(isinstance(previous_gate, Mapping), "V10 later checkpoint requires its previous gate")
        expected_previous = CHECKPOINT_STEPS[CHECKPOINT_STEPS.index(checkpoint_step) - 1]
        _require(
            previous_gate.get("contract") == GATE_CONTRACT
            and previous_gate.get("checkpoint_step") == expected_previous,
            "V10 previous gate is not the immediate predecessor",
        )
    diagnostic = v9.build_v9_gate(
        records_by_condition=records_by_condition,
        pairing=pairing,
        checkpoint_step=checkpoint_step * 7,
        previous_gate=_mapped_previous_gate(previous_gate),
    )
    result = dict(diagnostic)
    result.update(
        {
            "contract": GATE_CONTRACT,
            "checkpoint_step": checkpoint_step,
            "optimizer_unit_pair_presentations": launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
            "consumed_pair_presentations": launch.presentation_cursor(checkpoint_step),
            "hard32_access": HARD32_ACCESS_POLICY,
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        }
    )
    comparison = dict(result["comparison"])
    if comparison.get("checkpoint_step") is not None:
        comparison["checkpoint_step"] = int(comparison["checkpoint_step"]) // 7
    result["comparison"] = comparison
    if result.get("next_checkpoint_step") is not None:
        result["next_checkpoint_step"] = int(result["next_checkpoint_step"]) // 7
    result["training_continuation_authorized"] = bool(
        result["status"] == "pass" and result["next_checkpoint_step"] is not None
    )
    return result


def validate_resume_records(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    donor_by_ordinal: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    originals: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    for record in records:
        current = dict(record)
        _require(current.get("schema") == GATE_RECORD_SCHEMA, "V10 record schema differs")
        _validate_self_hash(current, field="record_sha256")
        originals.append(current)
        converted = dict(current)
        converted["schema"] = v9.GATE_RECORD_SCHEMA
        compatible.append(v9._record_with_self_hash(converted))
    validated = v9.validate_resume_records(
        compatible,
        condition=condition,
        fingerprint=fingerprint,
        rows=rows,
        donor_by_ordinal=donor_by_ordinal,
    )
    return {ordinal: originals[ordinal] for ordinal in validated}


def evaluator_code_binding() -> dict[str, Any]:
    paths = {
        "v10_gate": Path(__file__).resolve(),
        "v9_gate_metrics": Path(v9.__file__).resolve(),
        "v7_train32_runtime": Path(v9.v8.v7.__file__).resolve(),
        "state_runtime": SCRIPT_DIR / "run_scene_state_eval.py",
        "v10_launch_contract": Path(launch.__file__).resolve(),
    }
    return {name: _artifact_binding(path, description=f"V10 evaluator code {name}") for name, path in paths.items()}


def build_evaluation_fingerprint_payload(
    *,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_model: Path | str = launch.PINNED_BASE_MODEL,
) -> dict[str, Any]:
    """Reconstruct the complete gate identity from pinned live inputs."""

    pinned_model = validate_base_model_path(base_model)
    _require(
        checkpoint.get("architecture", {}).get("target_layers") == list(range(42)),
        "V10 fingerprint checkpoint layer identity differs",
    )
    return {
        "schema_version": 1,
        "contract": GATE_CONTRACT,
        "task": v9.TASK_NAME,
        "split": "train",
        "training_sources": input_contract["artifacts"],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "checkpoint": dict(checkpoint),
        "base_model": str(pinned_model),
        "base_model_weights": v9.base_model_weight_identity(pinned_model),
        "base_model_prompt_artifacts": v9.base_model_prompt_identity(pinned_model),
        "expected_memory_layer_count": GATE_EXPECTED_MEMORY_LAYER_COUNT,
        "runtime": {
            "conditions": list(CONDITIONS),
            "semantic_selected_token_ordinals": list(VALUE14_ORDINALS),
            "max_new_tokens": GATE_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache_generation": True,
            "prime_use_cache": False,
            "device": GATE_DEVICE,
            "dtype": GATE_DTYPE,
            "attn_implementation": GATE_ATTN_IMPLEMENTATION,
            "normal_fusion_profile": GATE_NORMAL_FUSION_PROFILE,
            "packages": v9.runtime_package_versions(),
        },
        "objective": dict(V10_OBJECTIVE),
        "code": evaluator_code_binding(),
        "hard32_access": HARD32_ACCESS_POLICY,
    }


def _verify_artifact_binding(
    binding: Mapping[str, Any],
    *,
    description: str,
    ssd_root: Path | None = None,
    path_scope: str = "gate",
) -> Path:
    _require(isinstance(binding, Mapping), f"V10 {description} binding is missing")
    raw_path = binding.get("path")
    _require(isinstance(raw_path, str) and raw_path, f"V10 {description} path differs")
    guarded = launch._lexically_guard_path(raw_path, description=description)
    if ssd_root is None:
        expected = guarded.resolve()
    else:
        _require(path_scope in {"gate", "run"}, "V10 artifact path scope differs")
        resolver = _gate_path if path_scope == "gate" else _run_path
        expected = resolver(guarded, description=description, ssd_root=ssd_root)
    try:
        actual = v9._verify_artifact_binding(binding, description=description)
    except Exception as exc:
        raise V10EvaluationContractError(f"V10 {description} differs: {exc}") from exc
    _require(actual == expected, f"V10 {description} resolved path differs")
    return actual


def _previous_checkpoint_from_lineage(
    checkpoint: Mapping[str, Any],
    *,
    ssd_root: Path,
) -> Path | None:
    step = int(checkpoint["global_step"])
    if step == FIRST_GATE_STEP:
        return None
    lineage_path = _verify_artifact_binding(
        checkpoint["lineage_artifact"],
        description="current V10 lineage",
        ssd_root=ssd_root,
        path_scope="run",
    )
    lineage = _load_json(lineage_path, description="current V10 lineage")
    raw = lineage.get("source_checkpoint")
    _require(isinstance(raw, str) and raw, "V10 continuation source is missing")
    previous = _run_path(
        raw,
        description="V10 continuation source",
        ssd_root=ssd_root,
    )
    expected = CHECKPOINT_STEPS[CHECKPOINT_STEPS.index(step) - 1]
    _require(previous.name == f"checkpoint-{expected}", "V10 continuation source step differs")
    return previous


def validate_previous_gate_receipt(
    previous_receipt: Path | str | None,
    *,
    checkpoint: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    warm_contract: Mapping[str, Any] | None,
    ssd_root: Path,
) -> dict[str, Any] | None:
    previous_checkpoint = _previous_checkpoint_from_lineage(
        checkpoint,
        ssd_root=ssd_root,
    )
    if previous_checkpoint is None:
        _require(previous_receipt is None, "V10 checkpoint-1 forbids a previous receipt")
        return None
    _require(previous_receipt is not None, "V10 later checkpoint requires previous receipt")
    validated = validate_gate_receipt_for_checkpoint(
        previous_receipt,
        memory_dir=previous_checkpoint,
        input_contract=input_contract,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    expected_step = CHECKPOINT_STEPS[CHECKPOINT_STEPS.index(int(checkpoint["global_step"])) - 1]
    _require(
        validated["checkpoint"]["global_step"] == expected_step
        and validated["gate"]["next_checkpoint_step"] == checkpoint["global_step"],
        "V10 previous receipt does not bind immediate lineage",
    )
    return validated


def build_gate_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
    previous_receipt_path: Path | None,
    previous_receipt: Mapping[str, Any] | None,
    ssd_root: Path,
) -> dict[str, Any]:
    output_dir = _gate_path(
        output_dir,
        description="V10 gate output directory",
        ssd_root=ssd_root,
    )
    outputs = {
        "manifest": _artifact_binding(output_dir / "manifest.json", description="V10 gate manifest"),
        "summary": _artifact_binding(output_dir / "summary.json", description="V10 gate summary"),
        "conditions": {
            condition: _artifact_binding(output_dir / f"{condition}.jsonl", description=f"V10 {condition}")
            for condition in CONDITIONS
        },
    }
    previous_binding = None
    if previous_receipt_path is not None:
        _require(isinstance(previous_receipt, Mapping), "V10 predecessor payload missing")
        path = _regular_file(previous_receipt_path, description="previous V10 receipt", ssd_root=ssd_root)
        previous_binding = {
            "artifact": _artifact_binding(path, description="previous V10 receipt"),
            "receipt_sha256": previous_receipt["receipt_sha256"],
            "checkpoint": previous_receipt["checkpoint"],
        }
    passed = gate.get("status") == "pass" and gate.get("all_gates_passed") is True
    receipt: dict[str, Any] = {
        "schema": GATE_RECEIPT_SCHEMA,
        "created_at": v9.utc_now(),
        "status": "pass" if passed else "fail",
        "contract": GATE_CONTRACT,
        "task": v9.TASK_NAME,
        "evaluation_fingerprint": fingerprint,
        "objective": dict(V10_OBJECTIVE),
        "training_sources": dict(input_contract["artifacts"]),
        "checkpoint": dict(checkpoint),
        "previous_gate_receipt": previous_binding,
        "outputs": outputs,
        "code": evaluator_code_binding(),
        "gate": dict(gate),
        "training_authorization": {
            "authorization_kind": CONTINUATION_AUTHORIZATION_KIND,
            "authorized": bool(gate.get("training_continuation_authorized")),
            "checkpoint_binding": dict(checkpoint),
            "next_checkpoint_step": gate.get("next_checkpoint_step"),
            "hard32_access": HARD32_ACCESS_POLICY,
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }
    receipt["receipt_sha256"] = self_hash_payload(receipt, hash_field="receipt_sha256")
    return receipt


def _validate_embedded_previous_gate_receipt_binding(
    embedded_previous: Mapping[str, Any] | None,
    *,
    validated_previous: Mapping[str, Any] | None,
    ssd_root: Path,
) -> None:
    expected = None
    if validated_previous is not None:
        previous_path = _regular_file(
            validated_previous["receipt_path"],
            description="validated previous V10 receipt",
            ssd_root=ssd_root,
        )
        expected = {
            "artifact": _artifact_binding(
                previous_path,
                description="validated previous V10 receipt",
            ),
            "receipt_sha256": validated_previous["receipt_sha256"],
            "checkpoint": validated_previous["checkpoint"],
        }
    _require(
        embedded_previous == expected,
        "V10 embedded previous receipt binding differs from validated predecessor",
    )


def validate_gate_receipt_for_checkpoint(
    receipt: Path | str | Mapping[str, Any],
    *,
    memory_dir: Path | str,
    input_contract: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
    else:
        receipt_path = _regular_file(receipt, description="V10 gate receipt", ssd_root=ssd_root)
        payload = _load_json(receipt_path, description="V10 gate receipt")
    _require(payload.get("schema") == GATE_RECEIPT_SCHEMA, "V10 receipt schema differs")
    _validate_self_hash(payload, field="receipt_sha256")
    _require(payload.get("contract") == GATE_CONTRACT, "V10 receipt contract differs")
    _require(payload.get("objective") == V10_OBJECTIVE, "V10 receipt objective differs")
    _require(payload.get("status") == "pass", "V10 continuation requires passing receipt")
    inputs = validate_v10_train_inputs(ssd_root=ssd_root) if input_contract is None else dict(input_contract)
    checkpoint = validate_v10_checkpoint(
        memory_dir,
        input_contract=inputs,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    _require(payload.get("checkpoint") == checkpoint, "V10 receipt checkpoint differs")
    _require(payload.get("training_sources") == inputs["artifacts"], "V10 receipt sources differ")
    expected_fingerprint_payload = build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
    )
    expected_fingerprint = v9.fingerprint_payload_sha256(
        expected_fingerprint_payload
    )
    _require(
        payload.get("evaluation_fingerprint") == expected_fingerprint,
        "V10 receipt evaluation fingerprint differs from live inputs",
    )
    embedded_previous = payload.get("previous_gate_receipt")
    embedded_previous_path = None
    if embedded_previous is not None:
        _require(
            isinstance(embedded_previous, Mapping),
            "V10 embedded previous receipt binding must be an object",
        )
        embedded_artifact = embedded_previous.get("artifact")
        _require(
            isinstance(embedded_artifact, Mapping)
            and isinstance(embedded_artifact.get("path"), str)
            and bool(embedded_artifact.get("path")),
            "V10 embedded previous receipt artifact differs",
        )
        embedded_previous_path = embedded_artifact["path"]
    previous = validate_previous_gate_receipt(
        embedded_previous_path,
        checkpoint=checkpoint,
        input_contract=inputs,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    _validate_embedded_previous_gate_receipt_binding(
        embedded_previous,
        validated_previous=previous,
        ssd_root=ssd_root,
    )
    outputs = payload.get("outputs")
    _require(isinstance(outputs, Mapping), "V10 receipt outputs missing")
    manifest_path = _verify_artifact_binding(outputs["manifest"], description="V10 receipt manifest", ssd_root=ssd_root)
    summary_path = _verify_artifact_binding(outputs["summary"], description="V10 receipt summary", ssd_root=ssd_root)
    manifest = _load_json(manifest_path, description="V10 receipt manifest")
    summary = _load_json(summary_path, description="V10 receipt summary")
    manifest = validate_existing_manifest(
        manifest,
        expected_fingerprint=expected_fingerprint,
        expected_fingerprint_payload=expected_fingerprint_payload,
        require_postload=True,
    )
    _require(summary.get("schema") == GATE_SUMMARY_SCHEMA, "V10 summary schema differs")
    _validate_self_hash(summary, field="summary_sha256")
    fingerprint = expected_fingerprint
    _require(manifest.get("fingerprint") == fingerprint and summary.get("fingerprint") == fingerprint, "V10 fingerprints differ")
    records: dict[str, list[dict[str, Any]]] = {}
    bindings = outputs.get("conditions")
    _require(isinstance(bindings, Mapping) and set(bindings) == set(CONDITIONS), "V10 condition bindings differ")
    for condition in CONDITIONS:
        path = _verify_artifact_binding(bindings[condition], description=f"V10 {condition}", ssd_root=ssd_root)
        rows = _read_jsonl(path, description=f"V10 {condition}")
        indexed = validate_resume_records(
            rows,
            condition=condition,
            fingerprint=fingerprint,
            rows=inputs["rows"],
            donor_by_ordinal=inputs["pairing"]["donor_by_ordinal"],
        )
        _require(len(indexed) == 32, f"V10 {condition} output incomplete")
        records[condition] = [indexed[index] for index in range(32)]
    recomputed = build_v10_gate(
        records_by_condition=records,
        pairing=inputs["pairing"],
        checkpoint_step=checkpoint["global_step"],
        previous_gate=None if previous is None else previous["gate"],
    )
    _require(recomputed == payload.get("gate") == summary.get("gate"), "V10 gate does not reproduce")
    _require(payload.get("code") == evaluator_code_binding(), "V10 evaluator code differs")
    authorization = payload.get("training_authorization")
    _require(
        isinstance(authorization, Mapping)
        and authorization.get("authorization_kind") == CONTINUATION_AUTHORIZATION_KIND
        and authorization.get("checkpoint_binding") == checkpoint
        and authorization.get("hard32_authorized") is False,
        "V10 training authorization differs",
    )
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = v9.sha256_file(receipt_path)
    return result


def validate_existing_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_fingerprint: str,
    expected_fingerprint_payload: Mapping[str, Any],
    require_postload: bool = False,
) -> dict[str, Any]:
    preload_keys = {
        "schema",
        "created_at",
        "fingerprint",
        "fingerprint_payload",
        "hard32_access",
    }
    postload_keys = preload_keys | {
        "runtime_prefixes",
        "runtime_fusion_profile",
    }
    _require(
        set(manifest) in {frozenset(preload_keys), frozenset(postload_keys)},
        "V10 existing manifest fields differ",
    )
    if require_postload:
        _require(
            set(manifest) == postload_keys,
            "V10 completed manifest requires post-load runtime identity",
        )
    _require(
        manifest.get("schema") == GATE_MANIFEST_SCHEMA,
        "V10 existing manifest schema differs",
    )
    _require(
        isinstance(manifest.get("created_at"), str) and bool(manifest["created_at"]),
        "V10 existing manifest creation time differs",
    )
    expected_payload = dict(expected_fingerprint_payload)
    _require(
        manifest.get("fingerprint_payload") == expected_payload,
        "V10 existing manifest fingerprint payload differs",
    )
    _require(
        manifest.get("fingerprint") == expected_fingerprint
        and expected_fingerprint
        == v9.fingerprint_payload_sha256(expected_payload),
        "V10 existing manifest fingerprint differs",
    )
    _require(
        manifest.get("hard32_access") == HARD32_ACCESS_POLICY
        and expected_payload.get("hard32_access") == HARD32_ACCESS_POLICY,
        "V10 existing manifest Hard32 identity differs",
    )
    for field in ("training_sources", "runtime", "code"):
        _require(
            manifest["fingerprint_payload"].get(field) == expected_payload.get(field),
            f"V10 existing manifest {field} identity differs",
        )
    if "runtime_prefixes" in manifest:
        _require(
            manifest.get("runtime_prefixes") is not None
            and manifest.get("runtime_fusion_profile") is not None,
            "V10 existing post-load runtime identity differs",
        )
    return dict(manifest)


def bind_or_validate_manifest_runtime(
    manifest: Mapping[str, Any],
    *,
    runtime_prefixes: Any,
    runtime_fusion_profile: Any,
) -> dict[str, Any]:
    bound = dict(manifest)
    has_prefixes = "runtime_prefixes" in bound
    has_profile = "runtime_fusion_profile" in bound
    _require(
        has_prefixes == has_profile,
        "V10 manifest post-load runtime fields are incomplete",
    )
    if has_prefixes:
        _require(
            bound["runtime_prefixes"] == runtime_prefixes,
            "V10 runtime prefixes differ",
        )
        _require(
            bound["runtime_fusion_profile"] == runtime_fusion_profile,
            "V10 runtime fusion profile differs",
        )
        return bound
    bound["runtime_prefixes"] = runtime_prefixes
    bound["runtime_fusion_profile"] = runtime_fusion_profile
    return bound


def validate_continuation_authorization(
    receipt: Path | str,
    *,
    source_checkpoint: Path | str,
    target_step: int,
    input_contract: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    _require(target_step in CHECKPOINT_STEPS, "V10 target step is not locked")
    path = _regular_file(receipt, description="V10 continuation receipt", ssd_root=ssd_root)
    validated = validate_gate_receipt_for_checkpoint(
        path,
        memory_dir=source_checkpoint,
        input_contract=input_contract,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    checkpoint = validated["checkpoint"]
    source_step = int(checkpoint["global_step"])
    source_index = CHECKPOINT_STEPS.index(source_step)
    _require(
        source_index + 1 < len(CHECKPOINT_STEPS)
        and CHECKPOINT_STEPS[source_index + 1] == target_step,
        "V10 receipt does not bind immediate target",
    )
    gate = validated["gate"]
    authorization = validated["training_authorization"]
    _require(
        gate.get("training_continuation_authorized") is True
        and gate.get("next_checkpoint_step") == target_step
        and authorization.get("authorized") is True
        and authorization.get("next_checkpoint_step") == target_step,
        "V10 receipt does not authorize continuation",
    )
    return {
        "authorization_kind": CONTINUATION_AUTHORIZATION_KIND,
        "gate_receipt": str(path),
        "gate_receipt_file_sha256": v9.sha256_file(path),
        "gate_receipt_sha256": validated["receipt_sha256"],
        "source_checkpoint": checkpoint["memory_dir"],
        "source_step": source_step,
        "target_step": target_step,
        "hard32_access": HARD32_ACCESS_POLICY,
        "hard32_authorized": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-gate-receipt", type=Path)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=v9.DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--normal-fusion-profile", default="native", choices=("native",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(args.max_new_tokens == GATE_MAX_NEW_TOKENS, "V10 gate requires 128 tokens")
    _require(
        args.expected_memory_layer_count == GATE_EXPECTED_MEMORY_LAYER_COUNT,
        "V10 gate requires all 42 layers",
    )
    _require(args.device == GATE_DEVICE, "V10 gate device differs")
    _require(args.dtype == GATE_DTYPE, "V10 gate dtype differs")
    _require(
        args.attn_implementation == GATE_ATTN_IMPLEMENTATION,
        "V10 gate attention implementation differs",
    )
    _require(
        args.normal_fusion_profile == GATE_NORMAL_FUSION_PROFILE,
        "V10 gate fusion profile differs",
    )
    args.delta_mem_root = str(Path(args.delta_mem_root).expanduser().resolve())
    _require(Path(args.delta_mem_root) == PROJECT_ROOT, "V10 gate requires this checkout")
    base_model = validate_base_model_path(args.base_model)
    args.base_model = str(base_model)
    output_dir = _gate_path(
        args.output_dir,
        description="V10 gate output directory",
        ssd_root=launch.SSD_ROOT,
    )
    args.output_dir = output_dir
    inputs = validate_v10_train_inputs()
    warm = launch.validate_warm_start_contract()
    checkpoint = validate_v10_checkpoint(
        args.memory_dir,
        input_contract=inputs,
        warm_contract=warm,
    )
    previous = validate_previous_gate_receipt(
        args.previous_gate_receipt,
        checkpoint=checkpoint,
        input_contract=inputs,
        warm_contract=warm,
        ssd_root=launch.SSD_ROOT,
    )
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_pass",
            "model_loaded": False,
            "output_created": False,
            "hard32_access": HARD32_ACCESS_POLICY,
            "checkpoint": checkpoint,
            "previous_gate_receipt": None if previous is None else previous["receipt_sha256"],
            "training_sources": inputs["artifacts"],
            "base_model": str(base_model),
        }, indent=2, sort_keys=True))
        return 0
    rows = inputs["rows"]
    donors = inputs["pairing"]["donor_by_ordinal"]
    memory_dir = Path(str(checkpoint["memory_dir"]))
    args.memory_dir = memory_dir
    expected_layers = v9.resolved_memory_layer_count(memory_dir, args.expected_memory_layer_count)
    _require(
        expected_layers == GATE_EXPECTED_MEMORY_LAYER_COUNT,
        "V10 resolved memory layer count differs",
    )
    fingerprint_payload = build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
        base_model=base_model,
    )
    fingerprint = v9.fingerprint_payload_sha256(fingerprint_payload)
    manifest = {
        "schema": GATE_MANIFEST_SCHEMA,
        "created_at": v9.utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    paths = {condition: output_dir / f"{condition}.jsonl" for condition in CONDITIONS}
    paths.update({"manifest": output_dir / "manifest.json", "summary": output_dir / "summary.json", "receipt": output_dir / "gate_receipt.json"})
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    if paths["manifest"].exists():
        existing_manifest = _load_json(paths["manifest"], description="existing V10 manifest")
        manifest = validate_existing_manifest(
            existing_manifest,
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=fingerprint_payload,
        )
    else:
        atomic_write_json(paths["manifest"], manifest)
    completed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        existing = _read_jsonl(paths[condition], description=f"V10 {condition}") if paths[condition].exists() else []
        completed[condition] = validate_resume_records(
            existing,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donors,
        )
    if any(len(completed[condition]) < 32 for condition in CONDITIONS):
        model, tokenizer, runtime_profile = v9.load_adapter_model(args, expected_layers)
        prefixes = v9.v8.v7.validate_runtime_prefixes(tokenizer, rows=rows)
        manifest = bind_or_validate_manifest_runtime(
            manifest,
            runtime_prefixes=prefixes,
            runtime_fusion_profile=runtime_profile,
        )
        atomic_write_json(paths["manifest"], manifest)
        try:
            for condition in CONDITIONS:
                for ordinal, sample in enumerate(rows):
                    if ordinal in completed[condition]:
                        continue
                    donor_ordinal = donors[ordinal]
                    donor_sample = rows[donor_ordinal]
                    result = v9.evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_sample,
                        condition=condition,
                        max_new_tokens=v9.DEFAULT_MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=ordinal in VALUE14_SET,
                    )
                    record = _record_with_self_hash({
                        "schema": GATE_RECORD_SCHEMA,
                        "status": "ok",
                        "completed_at": v9.utc_now(),
                        "fingerprint": fingerprint,
                        "condition": condition,
                        "task": v9.TASK_NAME,
                        "task_kind": "scene",
                        "split": "train",
                        "train_row_ordinal": ordinal,
                        "source_index": sample["source_index"],
                        "row_sha256": sample["row_sha256"],
                        "gold": sample["gold"],
                        "donor_train_row_ordinal": donor_ordinal,
                        **result,
                        "donor_source_index": donor_sample["source_index"],
                        "donor_row_sha256": donor_sample["row_sha256"],
                    })
                    completed[condition][ordinal] = record
                    atomic_write_jsonl(paths[condition], [completed[condition][index] for index in sorted(completed[condition])])
        finally:
            del model
            del tokenizer
            v9.clear_model_memory()
    ordered = {condition: [completed[condition][index] for index in range(32)] for condition in CONDITIONS}
    gate = build_v10_gate(
        records_by_condition=ordered,
        pairing=inputs["pairing"],
        checkpoint_step=checkpoint["global_step"],
        previous_gate=None if previous is None else previous["gate"],
    )
    summaries = {condition: v9.summarize_records(records) for condition, records in ordered.items()}
    summary: dict[str, Any] = {
        "schema": GATE_SUMMARY_SCHEMA,
        "created_at": v9.utc_now(),
        "fingerprint": fingerprint,
        "complete": True,
        "contract": GATE_CONTRACT,
        "task": v9.TASK_NAME,
        "split": "train",
        "conditions": summaries,
        "comparisons": v9.build_comparisons(summaries),
        "gate": gate,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    summary["summary_sha256"] = self_hash_payload(summary, hash_field="summary_sha256")
    atomic_write_json(paths["summary"], summary)
    receipt = build_gate_receipt(
        output_dir=output_dir,
        fingerprint=fingerprint,
        input_contract=inputs,
        checkpoint=checkpoint,
        gate=gate,
        previous_receipt_path=args.previous_gate_receipt,
        previous_receipt=previous,
        ssd_root=launch.SSD_ROOT,
    )
    atomic_write_canonical_json(paths["receipt"], receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
