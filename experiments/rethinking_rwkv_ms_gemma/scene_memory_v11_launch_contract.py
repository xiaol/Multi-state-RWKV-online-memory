#!/usr/bin/env python3
"""Fail-closed launch contract for one-cycle Scene Memory V11 training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v10_launch_contract as v10,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v11_warm_start import (
    ABLATION_LINEAGE_FILENAME,
    CONTINUATION_LINEAGE_FILENAME,
    DEFAULT_LOCK_PATH as WARM_START_LOCK,
    RECEIPT_SCHEMA as WARM_START_RECEIPT_SCHEMA,
    SOURCE_IMPORT_POLICY,
    WARM_START_LINEAGE_FILENAME,
    WARM_START_MODE,
    load_v11_warm_start_lock,
    prepare_v11_v8_checkpoint56_warm_start,
)


SSD_ROOT = v10.SSD_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = v10.DATA_ROOT
SOURCE_LOCK = v10.SOURCE_LOCK
PINNED_BASE_MODEL = v10.PINNED_BASE_MODEL
PINNED_WARM_START_CHECKPOINT = v10.PINNED_WARM_START_CHECKPOINT
SOURCE_LOCK_FILE_SHA256 = v10.SOURCE_LOCK_FILE_SHA256
WARM_START_LOCK_FILE_SHA256 = v10.WARM_START_LOCK_FILE_SHA256
PINNED_DATA_ARTIFACTS = v10.PINNED_DATA_ARTIFACTS
PINNED_HISTORICAL_TRAIN32_ARTIFACTS = v10.PINNED_HISTORICAL_TRAIN32_ARTIFACTS
REQUIRED_CHECKPOINT_ARTIFACTS = v10.REQUIRED_CHECKPOINT_ARTIFACTS

V11_RUN_ROOT = (
    SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v11"
)
V11_GATES_ROOT = V11_RUN_ROOT / "gates"
PINNED_V10_DIAGNOSTIC_GATE_DIR = (
    SSD_ROOT
    / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v10"
    / "gates/value14_cycle1_20260730_1002"
)
PINNED_V10_DIAGNOSTIC_SUMMARY = (
    PINNED_V10_DIAGNOSTIC_GATE_DIR / "summary.json"
)
PINNED_V10_DIAGNOSTIC_RECEIPT = (
    PINNED_V10_DIAGNOSTIC_GATE_DIR / "gate_receipt.json"
)
PINNED_V10_DIAGNOSTIC_MANIFEST = (
    PINNED_V10_DIAGNOSTIC_GATE_DIR / "manifest.json"
)
PINNED_V10_DIAGNOSTIC_CHECKPOINT = (
    SSD_ROOT
    / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v10"
    / "scene_memory_v10_production_value14_cycle1_20260730_095119_step1"
    / "trainer/checkpoint-1"
)
PINNED_V10_DIAGNOSTIC_SUMMARY_FILE_SHA256 = (
    "23c102b1fad00601617bc3c3671094720b5d5b352cc3a1a34d09a5c6f36492ba"
)
PINNED_V10_DIAGNOSTIC_RECEIPT_FILE_SHA256 = (
    "4aa86ce2ea41095780de4f11a58b4f9ba26df24c90715ecc6f2485cf2df201b3"
)
PINNED_V10_DIAGNOSTIC_MANIFEST_FILE_SHA256 = (
    "dc89864301fcaea8e52de399c418f40de064072076748d31bbb0c6d7dba05bff"
)
PINNED_V10_DIAGNOSTIC_FINGERPRINT = (
    "5550b2d5dde29069b05d6cddcadc6718e8c3ba50f0778d081409bc4f3875e6e1"
)
PINNED_V10_DIAGNOSTIC_SUMMARY_SHA256 = (
    "86bb4cbdbbc54285f905450921ad8fcc81363a5aa3af084c3bc5434e6a2b5e0e"
)
PINNED_V10_DIAGNOSTIC_RECEIPT_SHA256 = (
    "07ff547258d723029eda4e7b24554c13f134472cd8318f308fc875cd2c00b3cc"
)
V10_DIAGNOSTIC_BASELINE_METRICS = {
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

OBJECTIVE_VERSION = "scene_state_generation_ce_symmetric_cycle_suffix_repair_v5"
PAIRING_OBJECTIVE_VERSION = v10.PAIRING_OBJECTIVE_VERSION
OBJECTIVE_SCHEMA_VERSION = 14
FIXED_SAMPLER_MODE = "explicit_ordered_v11_canonical_seven_pair_cycle_v1"
PAIR_PHYSICAL_BATCH_SIZE = 1
PAIR_LOGICAL_BATCH_SIZE = 2
PAIR_DIRECTIONAL_EXPOSURES = 2
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 7
TOTAL_PAIR_PRESENTATIONS = 7
TOTAL_OPTIMIZER_STEPS = 1
CHECKPOINT_STEPS = (1,)
PRESENTATION_CHECKPOINTS = (7,)
CONTINUATION_POLICY = "forbidden"
GRADIENT_ACCUMULATION_STEPS = 7
SAVE_STEPS = 1
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0
WARMUP_RATIO = 0.0
SUFFIX_REPAIR_WEIGHT = 0.5
GENERATED_MAX_CORRECTION_EVENTS = 1
GENERATED_ROLLOUT_EXTRA_TOKENS = 4
GENERATED_ROLLOUT_MAX_TOKENS = 24
MAX_LENGTH = 256
MAX_WRITE_LENGTH = 2048
TEACHER_MAX_LENGTH = MAX_LENGTH + MAX_WRITE_LENGTH
FIRST_CYCLE_PAIRS = (
    (3, 24),
    (19, 28),
    (20, 31),
    (10, 23),
    (1, 14),
    (5, 9),
    (22, 26),
)
FIRST_CYCLE_PAIRS_SHA256 = (
    "accb8683615c4ffb3e5e09b52a6e2a97ac60dfd2dcc8fd594965370884e38ca1"
)
SUFFIX_REPAIR_MODE = (
    "first_raw_token_divergence_common_prefix_weighted_gold_suffix_ce_"
    "first_generated_wrong_unlikelihood_v5"
)
SUFFIX_REPAIR_DIVERGENCE = (
    "first_raw_token_divergence_including_length_mismatch_v1"
)
SUFFIX_REPAIR_GOLD_WEIGHTING = "schema_2_decision_4_termination_1_v1"
OBJECTIVE_FORMULA = (
    "symmetric_pair_mean(weighted_full_gold_ce(schema=2,decision=4,termination=1) "
    "+ first_error_top1_hinge(0.2) + "
    "all_target_top1_retention_hinge(0.2) + "
    "selected_top_competitor_hinge(0.2) + "
    "selected_correct_vs_detached_zero_nll_hinge(0.2) + 0.5 * "
    "first_divergence_suffix_repair(weighted_gold_suffix_ce(schema=2,decision=4,"
    "termination=1) + first_generated_wrong_unlikelihood)); "
    "selected_full_vocab_ce=telemetry_only"
)
BACKWARD_MODE = (
    "sequential_pair_zero_probe_full_gold_first_error_all_target_retention_"
    "then_first_divergence_gold_suffix_replay_v6"
)
CYCLE_RETENTION_MODE = v10.CYCLE_RETENTION_MODE
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed"
TRAINING_CONTINUATION_POLICY = "forbidden_one_cycle_only_regardless_of_gate_status"
LAUNCH_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v11_attached_launch.v1"
COMPLETION_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v11_attached_completion.v1"
CRITICAL_TRAINING_FILES = (
    "deltamem/train/delta_sft_experimental.py",
    "deltamem/train/scene_state_generation_alignment.py",
    "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v9_data.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_data_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_source_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_v8_checkpoint56_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v11_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v11_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v11_gate.py",
    "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v11.sh",
)

_V11_RUN_RELATIVE = V11_RUN_ROOT.relative_to(SSD_ROOT)
_V11_GATES_RELATIVE = V11_GATES_ROOT.relative_to(SSD_ROOT)


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


canonical_sha256 = v10.canonical_sha256
sha256_file = v10.sha256_file
require_sha256 = v10.require_sha256
require_exact_path = v10.require_exact_path
require_under_root = v10.require_under_root
require_ssd = v10.require_ssd
_lexically_guard_path = v10._lexically_guard_path
_regular_file = v10._regular_file
_load_object = v10._load_object


def v11_run_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V11_RUN_RELATIVE


def v11_gates_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V11_GATES_RELATIVE


def require_v11_run_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v11_run_root_for(ssd_root),
        description=description,
    )


def require_v11_gate_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v11_gates_root_for(ssd_root),
        description=description,
    )


def presentation_cursor(global_step: int) -> int:
    require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step in (0, 1),
        "global_step_outside_v11_one_cycle_schedule",
    )
    return global_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP


def _finite_integral_metric(value: Any, *, description: str) -> int:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value).is_integer(),
        f"{description}_must_be_finite_integral",
    )
    return int(value)


def validate_v11_cycle_pair_telemetry(
    trainer_state: Mapping[str, Any],
) -> dict[str, Any]:
    history = trainer_state.get("log_history")
    require(isinstance(history, list), "v11_trainer_log_history_missing")
    count_key = "delta/scene_generation_v11_cycle_pair_presentations"
    candidates = [
        (index, entry)
        for index, entry in enumerate(history)
        if isinstance(entry, Mapping) and count_key in entry
    ]
    require(
        len(candidates) == 1,
        "v11_cycle_pair_telemetry_requires_exactly_one_optimizer_log",
    )
    history_index, entry = candidates[0]
    presentations = _finite_integral_metric(
        entry.get(count_key),
        description="v11_cycle_pair_presentations",
    )
    require(presentations == 7, "v11_cycle_pair_presentations_differ")
    require(
        _finite_integral_metric(entry.get("step"), description="v11_cycle_pair_step")
        == 1,
        "v11_cycle_pair_telemetry_step_differs",
    )
    pairs: list[list[int]] = []
    telemetry_keys = [count_key]
    for index, expected_pair in enumerate(FIRST_CYCLE_PAIRS):
        low_key = f"delta/scene_generation_v11_cycle_pair_{index}_low_ordinal"
        high_key = f"delta/scene_generation_v11_cycle_pair_{index}_high_ordinal"
        low = _finite_integral_metric(
            entry.get(low_key),
            description=f"v11_cycle_pair_{index}_low_ordinal",
        )
        high = _finite_integral_metric(
            entry.get(high_key),
            description=f"v11_cycle_pair_{index}_high_ordinal",
        )
        require(
            (low, high) == expected_pair,
            f"v11_cycle_pair_{index}_identity_or_order_differs",
        )
        pairs.append([low, high])
        telemetry_keys.extend((low_key, high_key))
    pairs_sha256 = canonical_sha256(pairs)
    require(
        pairs_sha256 == FIRST_CYCLE_PAIRS_SHA256,
        "v11_cycle_pair_telemetry_hash_differs",
    )
    return {
        "schema": "rwkv_ms_scene_memory_v11_cycle_pair_telemetry.v1",
        "trainer_log_history_index": history_index,
        "optimizer_step": 1,
        "pair_presentations": presentations,
        "ordered_pairs": pairs,
        "ordered_pairs_sha256": pairs_sha256,
        "telemetry_keys": telemetry_keys,
    }


def _validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    description: str,
) -> None:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    require(recorded == expected, f"{description}_recorded_hash_differs")
    require(recorded == canonical_sha256(unsigned), f"{description}_self_hash_differs")


def validate_v10_diagnostic_baseline(
    *,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require_exact_path(ssd_root, SSD_ROOT, description="v11_ssd_root")
    summary_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_SUMMARY,
        PINNED_V10_DIAGNOSTIC_SUMMARY,
        description="v11_v10_diagnostic_summary",
    )
    receipt_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_RECEIPT,
        PINNED_V10_DIAGNOSTIC_RECEIPT,
        description="v11_v10_diagnostic_receipt",
    )
    manifest_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_MANIFEST,
        PINNED_V10_DIAGNOSTIC_MANIFEST,
        description="v11_v10_diagnostic_manifest",
    )
    _regular_file(summary_path, description="v11_v10_diagnostic_summary")
    _regular_file(receipt_path, description="v11_v10_diagnostic_receipt")
    _regular_file(manifest_path, description="v11_v10_diagnostic_manifest")
    require(
        sha256_file(summary_path) == PINNED_V10_DIAGNOSTIC_SUMMARY_FILE_SHA256,
        "v11_v10_diagnostic_summary_file_hash_differs",
    )
    require(
        sha256_file(receipt_path) == PINNED_V10_DIAGNOSTIC_RECEIPT_FILE_SHA256,
        "v11_v10_diagnostic_receipt_file_hash_differs",
    )
    require(
        sha256_file(manifest_path) == PINNED_V10_DIAGNOSTIC_MANIFEST_FILE_SHA256,
        "v11_v10_diagnostic_manifest_file_hash_differs",
    )
    summary = _load_object(summary_path, description="v11_v10_diagnostic_summary")
    receipt = _load_object(receipt_path, description="v11_v10_diagnostic_receipt")
    manifest = _load_object(manifest_path, description="v11_v10_diagnostic_manifest")
    _validate_self_hash(
        summary,
        field="summary_sha256",
        expected=PINNED_V10_DIAGNOSTIC_SUMMARY_SHA256,
        description="v11_v10_diagnostic_summary",
    )
    _validate_self_hash(
        receipt,
        field="receipt_sha256",
        expected=PINNED_V10_DIAGNOSTIC_RECEIPT_SHA256,
        description="v11_v10_diagnostic_receipt",
    )
    require(
        summary.get("schema")
        == "rwkv_ms_scene_memory_v10_train32_gate_summary.v1"
        and receipt.get("schema")
        == "rwkv_ms_scene_memory_v10_train32_gate_receipt.v1"
        and manifest.get("schema")
        == "rwkv_ms_scene_memory_v10_train32_gate_manifest.v1"
        and summary.get("fingerprint") == PINNED_V10_DIAGNOSTIC_FINGERPRINT
        and receipt.get("evaluation_fingerprint")
        == PINNED_V10_DIAGNOSTIC_FINGERPRINT,
        "v11_v10_diagnostic_schema_or_fingerprint_differs",
    )
    require(
        manifest.get("fingerprint") == PINNED_V10_DIAGNOSTIC_FINGERPRINT
        and manifest.get("hard32_access") == HARD32_ACCESS_POLICY,
        "v11_v10_diagnostic_manifest_identity_differs",
    )
    fingerprint_payload = manifest.get("fingerprint_payload")
    require(
        isinstance(fingerprint_payload, Mapping)
        and fingerprint_payload.get("base_model") == str(PINNED_BASE_MODEL),
        "v11_v10_diagnostic_base_model_path_differs",
    )
    base_model_weights = fingerprint_payload.get("base_model_weights")
    base_model_prompt_artifacts = fingerprint_payload.get(
        "base_model_prompt_artifacts"
    )
    require(
        isinstance(base_model_weights, Mapping)
        and isinstance(base_model_prompt_artifacts, Mapping),
        "v11_v10_diagnostic_base_model_identity_missing",
    )
    checkpoint = receipt.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "v11_v10_diagnostic_checkpoint_missing")
    require(
        checkpoint.get("memory_dir") == str(PINNED_V10_DIAGNOSTIC_CHECKPOINT)
        and checkpoint.get("global_step") == 1,
        "v11_v10_diagnostic_checkpoint_differs",
    )
    gate = summary.get("gate")
    require(isinstance(gate, Mapping), "v11_v10_diagnostic_gate_missing")
    require(
        gate.get("status") == "fail"
        and gate.get("training_continuation_authorized") is False
        and gate.get("hard32_authorized") is False
        and gate.get("full170_authorized") is False,
        "v11_v10_diagnostic_authorization_differs",
    )
    generation = gate.get("metrics", {}).get("value14_generation", {})
    identity = gate.get("metrics", {}).get("value14_selected_token_identity", {}).get(
        "overall", {}
    )
    actual_metrics = {
        "canonical_correct_outputs": generation.get("canonical_correct_outputs"),
        "correct_strict_exact_rows": generation.get("correct_strict_exact_rows"),
        "donor_identity_strict_exact_rows": generation.get(
            "donor_identity_strict_exact_rows"
        ),
        "correct_strict_micro_f1": generation.get("correct_strict_micro_f1"),
        "bidirectional_identity_switch_rows": identity.get(
            "bidirectional_identity_switch_rows"
        ),
        "correct_state_beats_donor_state_on_source_token_rows": identity.get(
            "correct_state_beats_donor_state_on_source_token_rows"
        ),
        "correct_state_prefers_source_token_rows": identity.get(
            "correct_state_prefers_source_token_rows"
        ),
        "donor_state_prefers_donor_token_rows": identity.get(
            "donor_state_prefers_donor_token_rows"
        ),
        "correct_state_beats_zero_on_source_token_rows": identity.get(
            "correct_state_beats_zero_on_source_token_rows"
        ),
    }
    require(
        actual_metrics == V10_DIAGNOSTIC_BASELINE_METRICS,
        "v11_v10_diagnostic_metrics_differ",
    )
    return {
        "role": "frozen_diagnostic_only_never_warm_start",
        "gate_dir": str(PINNED_V10_DIAGNOSTIC_GATE_DIR),
        "summary": {
            "path": str(summary_path),
            "file_sha256": PINNED_V10_DIAGNOSTIC_SUMMARY_FILE_SHA256,
            "summary_sha256": PINNED_V10_DIAGNOSTIC_SUMMARY_SHA256,
        },
        "receipt": {
            "path": str(receipt_path),
            "file_sha256": PINNED_V10_DIAGNOSTIC_RECEIPT_FILE_SHA256,
            "receipt_sha256": PINNED_V10_DIAGNOSTIC_RECEIPT_SHA256,
        },
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": PINNED_V10_DIAGNOSTIC_MANIFEST_FILE_SHA256,
        },
        "evaluation_fingerprint": PINNED_V10_DIAGNOSTIC_FINGERPRINT,
        "checkpoint": str(PINNED_V10_DIAGNOSTIC_CHECKPOINT),
        "base_model_identity": {
            "path": str(PINNED_BASE_MODEL),
            "weights": dict(base_model_weights),
            "prompt_artifacts": dict(base_model_prompt_artifacts),
        },
        "metrics": dict(V10_DIAGNOSTIC_BASELINE_METRICS),
    }


def validate_base_model_contract(
    *,
    base_model: Path = PINNED_BASE_MODEL,
    baseline: Mapping[str, Any] | None = None,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require_exact_path(ssd_root, SSD_ROOT, description="v11_ssd_root")
    resolved = require_exact_path(
        base_model,
        PINNED_BASE_MODEL,
        description="v11_base_model",
    )
    require(
        resolved.is_dir() and not resolved.is_symlink(),
        "v11_base_model_missing_or_symlink",
    )
    diagnostic = (
        validate_v10_diagnostic_baseline(ssd_root=ssd_root)
        if baseline is None
        else dict(baseline)
    )
    expected = diagnostic.get("base_model_identity")
    require(isinstance(expected, Mapping), "v11_v10_base_model_identity_missing")
    try:
        from experiments.rethinking_rwkv_ms_gemma import (
            run_scene_memory_v9_gate as v9_gate,
        )

        weights = v9_gate.base_model_weight_identity(resolved)
        prompt_artifacts = v9_gate.base_model_prompt_identity(resolved)
    except Exception as exc:
        raise LaunchContractError(f"v11_base_model_identity_failed: {exc}") from exc
    current = {
        "path": str(resolved),
        "weights": weights,
        "prompt_artifacts": prompt_artifacts,
    }
    require(current == expected, "v11_base_model_differs_from_pinned_v10_manifest")
    return current


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    try:
        data = v10.validate_data_contract(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise LaunchContractError(f"v11_reused_v10_data_contract_failed: {exc}") from exc
    entries = list(data["entries"])
    require(len(entries) >= TOTAL_PAIR_PRESENTATIONS, "v11_pair_schedule_too_short")
    first_pairs = tuple(
        tuple(entry["canonical_pair_ordinals"])
        for entry in entries[:TOTAL_PAIR_PRESENTATIONS]
    )
    require(first_pairs == FIRST_CYCLE_PAIRS, "v11_first_cycle_pair_order_differs")
    require(
        canonical_sha256([list(pair) for pair in first_pairs])
        == FIRST_CYCLE_PAIRS_SHA256,
        "v11_first_cycle_pair_hash_differs",
    )
    result = dict(data)
    result["source_checkpoint_steps"] = list(data["checkpoint_steps"])
    result["source_presentation_checkpoint_steps"] = list(
        data["source_presentation_checkpoint_steps"]
    )
    result["checkpoint_steps"] = [1]
    result["presentation_checkpoint_steps"] = [7]
    result["optimizer_cycles"] = [dict(data["optimizer_cycles"][0])]
    result["first_cycle_pairs"] = [list(pair) for pair in first_pairs]
    result["first_cycle_pairs_sha256"] = FIRST_CYCLE_PAIRS_SHA256
    return result


def validate_warm_start_contract(
    *,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    try:
        require_exact_path(ssd_root, SSD_ROOT, description="v11_ssd_root")
        lock_path = require_exact_path(
            warm_start_lock_path,
            WARM_START_LOCK,
            description="v11_warm_start_lock",
        )
        _regular_file(lock_path, description="v11_warm_start_lock")
        require(
            sha256_file(lock_path) == WARM_START_LOCK_FILE_SHA256,
            "v11_warm_start_lock_file_hash_differs",
        )
        lock = load_v11_warm_start_lock(lock_path)
        checkpoint = require_exact_path(
            Path(str(lock.get("source_checkpoint", ""))),
            PINNED_WARM_START_CHECKPOINT,
            description="v11_warm_start_checkpoint",
        )
        context = prepare_v11_v8_checkpoint56_warm_start(
            checkpoint,
            lock_path=lock_path,
        )
    except Exception as exc:
        raise LaunchContractError(f"v11_warm_start_contract_failed: {exc}") from exc
    require(
        context.checkpoint != PINNED_V10_DIAGNOSTIC_CHECKPOINT,
        "v11_must_never_warm_start_from_v10",
    )
    return {
        "warm_start_checkpoint": str(context.checkpoint),
        "warm_start_lock": str(context.lock_path),
        "warm_start_lock_file_sha256": WARM_START_LOCK_FILE_SHA256,
        "warm_start_lock_sha256": context.lock["lock_sha256"],
        "warm_start_mode": WARM_START_MODE,
        "lock": context.lock,
    }


def _validate_checkpoint_protocol(
    protocol: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
) -> None:
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, Mapping), "v11_protocol_train_schedule_missing")
    schedule_expected = {
        "checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
        "optimizer_checkpoint_steps": list(CHECKPOINT_STEPS),
        "microbatch_cycle_size": PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        "continuation_policy": CONTINUATION_POLICY,
    }
    schedule_mismatches = [
        name
        for name, value in schedule_expected.items()
        if schedule.get(name) != value
    ]
    if "resume_schedule_cursor_formula" in schedule:
        schedule_mismatches.append("resume_schedule_cursor_formula")
    require(
        not schedule_mismatches,
        "v11_protocol_schedule_differs fields=" + ",".join(schedule_mismatches),
    )
    compatible = dict(protocol)
    compatible_schedule = dict(schedule)
    compatible_schedule.update(
        {
            "checkpoint_steps": list(v10.PRESENTATION_CHECKPOINTS),
            "optimizer_checkpoint_steps": list(v10.CHECKPOINT_STEPS),
            "resume_schedule_cursor_formula": "global_step_times_7_v1",
        }
    )
    compatible_schedule.pop("continuation_policy", None)
    compatible["train_schedule"] = compatible_schedule
    compatible.update(
        {
            "schema_version": v10.OBJECTIVE_SCHEMA_VERSION,
            "memory_objective_version": v10.OBJECTIVE_VERSION,
            "train_sampler_mode": v10.FIXED_SAMPLER_MODE,
            "scene_generation_objective_formula": v10.OBJECTIVE_FORMULA,
            "scene_generation_backward_mode": v10.BACKWARD_MODE,
            "scene_generation_generated_prefix_correction_mode": (
                v10.GENERATED_PREFIX_MODE
            ),
            "scene_generation_generated_unlikelihood_max_wrong_tokens": (
                v10.GENERATED_MAX_CORRECTION_EVENTS
            ),
            "scene_generation_generated_prefix_max_correction_events": (
                v10.GENERATED_MAX_CORRECTION_EVENTS
            ),
        }
    )
    try:
        v10._validate_checkpoint_protocol(
            compatible,
            checkpoint_step=1,
            data=data,
        )
    except Exception as exc:
        raise LaunchContractError(f"v11_reused_v10_protocol_failed: {exc}") from exc
    expected = {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": OBJECTIVE_VERSION,
        "max_steps": 1,
        "max_grad_norm": MAX_GRAD_NORM,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "scene_generation_objective_formula": OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": BACKWARD_MODE,
        "scene_generation_generated_prefix_correction_weight": SUFFIX_REPAIR_WEIGHT,
        "scene_generation_generated_prefix_correction_mode": SUFFIX_REPAIR_MODE,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": (
            GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_prefix_max_correction_events": (
            GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_rollout_extra_tokens": (
            GENERATED_ROLLOUT_EXTRA_TOKENS
        ),
        "scene_generation_generated_rollout_max_tokens": (
            GENERATED_ROLLOUT_MAX_TOKENS
        ),
        "scene_generation_suffix_repair_mode": SUFFIX_REPAIR_MODE,
        "scene_generation_suffix_repair_weight": SUFFIX_REPAIR_WEIGHT,
        "scene_generation_suffix_repair_divergence": SUFFIX_REPAIR_DIVERGENCE,
        "scene_generation_suffix_repair_gold_weighting": (
            SUFFIX_REPAIR_GOLD_WEIGHTING
        ),
        "scene_generation_suffix_repair_first_wrong_unlikelihood": True,
        "scene_generation_suffix_repair_premature_termination_suppression": True,
        "scene_generation_suffix_repair_exact_rollout_loss": 0.0,
        "scene_generation_cycle_retention_mode": CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": 7,
        "scene_generation_gradient_accumulation_pair_cycle": 7,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    require(not mismatches, "v11_protocol_differs fields=" + ",".join(mismatches))


def _validate_pairing_manifest(pairing: Mapping[str, Any]) -> str:
    return v10._validate_pairing_manifest(pairing)


def _validate_warm_lineage(
    lineage: Mapping[str, Any],
    *,
    protocol_sha256: str,
    config_sha256: str,
    pairing_sha256: str,
    warm: Mapping[str, Any],
) -> str:
    unsigned = dict(lineage)
    recorded = unsigned.pop("receipt_sha256", None)
    require(
        lineage.get("schema") == WARM_START_RECEIPT_SCHEMA
        and lineage.get("schema_version") == 1
        and lineage.get("mode") == WARM_START_MODE,
        "v11_warm_lineage_schema_or_mode_differs",
    )
    require(
        require_sha256(recorded, description="v11_warm_receipt_hash")
        == canonical_sha256(unsigned),
        "v11_warm_lineage_self_hash_differs",
    )
    source_lock = lineage.get("source_lock")
    require(isinstance(source_lock, Mapping), "v11_warm_source_lock_missing")
    require(
        lineage.get("source_checkpoint") == warm["warm_start_checkpoint"]
        and source_lock.get("path") == warm["warm_start_lock"]
        and source_lock.get("lock_sha256") == warm["warm_start_lock_sha256"]
        and lineage.get("source_state_imports") == SOURCE_IMPORT_POLICY
        and lineage.get("post_load_bit_equal") is True,
        "v11_warm_source_binding_differs",
    )
    require(
        lineage.get("target_fresh_start")
        == {
            "initial_global_step": 0,
            "optimizer_implementation": "adamw_torch_fused",
            "optimizer_created_after_adapter_load": True,
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "trainer_state": "fresh",
            "rng_state": "fresh_from_v11_seed",
        },
        "v11_warm_fresh_start_differs",
    )
    evidence = {
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "pre_train_global_step": 0,
        "fresh_optimizer_created": True,
        "fresh_optimizer_state_entries_before_train": 0,
        "fresh_scheduler_created_before_train": False,
        "target_delta_config_sha256": config_sha256,
        "target_training_protocol_sha256": protocol_sha256,
        "target_scene_state_pairing_manifest_sha256": pairing_sha256,
    }
    mismatches = [name for name, value in evidence.items() if lineage.get(name) != value]
    require(not mismatches, "v11_warm_target_evidence_differs fields=" + ",".join(mismatches))
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        isinstance(optimizer_class, str) and optimizer_class.endswith(".AdamW"),
        "v11_warm_optimizer_class_differs",
    )
    return str(recorded)


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved = require_v11_run_path(
        checkpoint,
        description="v11_checkpoint",
        ssd_root=ssd_root,
    )
    require(resolved.is_dir(), "v11_checkpoint_missing")
    require(resolved.name == "checkpoint-1", "v11_only_checkpoint_1_is_allowed")
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / filename, description=f"v11_{filename}")
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            for path in rng_files
        ),
        "v11_rng_state_missing",
    )
    trainer_state = _load_object(
        resolved / "trainer_state.json",
        description="v11_trainer_state",
    )
    protocol = _load_object(
        resolved / "training_protocol.json",
        description="v11_protocol",
    )
    config = _load_object(
        resolved / "delta_mem_config.json",
        description="v11_config",
    )
    pairing = _load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="v11_pairing",
    )
    require(
        trainer_state.get("global_step") == 1
        and trainer_state.get("max_steps") == 1
        and protocol.get("max_steps") == 1,
        "v11_checkpoint_not_completed_one_cycle",
    )
    cycle_pair_telemetry = validate_v11_cycle_pair_telemetry(trainer_state)
    _validate_checkpoint_protocol(protocol, data=data)
    try:
        v10.v9._validate_checkpoint_config(config)
    except Exception as exc:
        raise LaunchContractError(f"v11_delta_config_failed: {exc}") from exc
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing)
    lineage_names = (
        WARM_START_LINEAGE_FILENAME,
        CONTINUATION_LINEAGE_FILENAME,
        ABLATION_LINEAGE_FILENAME,
    )
    present = [name for name in lineage_names if (resolved / name).is_file()]
    require(
        present == [WARM_START_LINEAGE_FILENAME],
        "v11_requires_exactly_one_fresh_warm_start_lineage",
    )
    lineage_path = resolved / WARM_START_LINEAGE_FILENAME
    lineage = _load_object(lineage_path, description="v11_lineage")
    root_receipt = _validate_warm_lineage(
        lineage,
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        warm=warm,
    )
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": 1,
        "consumed_pair_presentations": cycle_pair_telemetry[
            "pair_presentations"
        ],
        "cycle_pair_telemetry": cycle_pair_telemetry,
        "lineage_filename": WARM_START_LINEAGE_FILENAME,
        "lineage_file_sha256": sha256_file(lineage_path),
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt,
    }


def artifact_binding(path: Path, *, description: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _regular_file(resolved, description=description)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def critical_training_code_bindings(
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    resolved_root = project_root.expanduser().resolve()
    require(
        resolved_root == PROJECT_ROOT,
        "v11_critical_code_requires_exact_project_root",
    )
    bindings: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_TRAINING_FILES:
        path = resolved_root / relative
        require(
            path.is_file() and not path.is_symlink(),
            f"v11_critical_file_missing_or_symlink path={relative}",
        )
        bindings[relative] = artifact_binding(
            path,
            description=f"v11_critical_{relative}",
        )
    return bindings


def _validate_receipt_self_hash(
    payload: Mapping[str, Any],
    *,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop("receipt_sha256", None)
    require(
        isinstance(recorded, str)
        and require_sha256(recorded, description=f"{description}_hash")
        == canonical_sha256(unsigned),
        f"{description}_self_hash_differs",
    )
    return recorded


def _validate_exact_artifact_binding(
    binding: Mapping[str, Any],
    *,
    expected_path: Path,
    description: str,
) -> dict[str, Any]:
    require(isinstance(binding, Mapping), f"{description}_binding_missing")
    resolved = expected_path.expanduser().resolve()
    expected = artifact_binding(resolved, description=description)
    require(dict(binding) == expected, f"{description}_binding_differs")
    return expected


def _git_head(project_root: Path = PROJECT_ROOT) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchContractError("v11_git_head_unavailable") from exc
    return require_sha256(value, description="v11_git_head")


def validate_launch_receipt(
    launch_receipt: Path,
    *,
    checkpoint: Path,
    baseline: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v11_run_path(
        checkpoint,
        description="v11_provenance_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v11_run_path(
        launch_receipt,
        description="v11_launch_receipt",
        ssd_root=ssd_root,
    )
    _regular_file(path, description="v11_launch_receipt")
    require(
        path.parent == v11_run_root_for(ssd_root) / "logs",
        "v11_launch_receipt_must_be_under_logs",
    )
    payload = _load_object(path, description="v11_launch_receipt")
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v11_launch_receipt",
    )
    require(
        payload.get("schema") == LAUNCH_RECEIPT_SCHEMA
        and payload.get("attached_foreground_execution") is True
        and payload.get("launch_mode") == "warm_start"
        and payload.get("source_step") == 0
        and payload.get("target_step") == 1
        and payload.get("resume_checkpoint") is None,
        "v11_launch_receipt_mode_or_horizon_differs",
    )
    expected_output = resolved_checkpoint.parents[1]
    require(
        payload.get("trainer_output") == str(expected_output)
        and payload.get("checkpoint") == str(resolved_checkpoint),
        "v11_launch_receipt_checkpoint_binding_differs",
    )
    require(
        payload.get("objective") == OBJECTIVE_VERSION
        and payload.get("gradient_accumulation_steps") == 7
        and payload.get("max_grad_norm") == 1.0
        and payload.get("first_cycle_pairs")
        == [list(pair) for pair in FIRST_CYCLE_PAIRS]
        and payload.get("first_cycle_pairs_sha256") == FIRST_CYCLE_PAIRS_SHA256,
        "v11_launch_receipt_objective_or_cycle_differs",
    )
    require(
        payload.get("warm_start_checkpoint") == str(PINNED_WARM_START_CHECKPOINT)
        and payload.get("v10_diagnostic_baseline") == baseline
        and payload.get("base_model_identity") == base_model_identity,
        "v11_launch_receipt_source_or_model_identity_differs",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v11_launch_receipt_authorization_differs",
    )
    require(
        payload.get("tracked_worktree_clean") is True
        and payload.get("git_commit") == _git_head(project_root),
        "v11_launch_receipt_git_identity_differs",
    )
    expected_code = critical_training_code_bindings(project_root=project_root)
    require(
        payload.get("critical_files") == expected_code,
        "v11_launch_receipt_critical_file_hashes_differ",
    )
    return {
        "artifact": artifact_binding(path, description="v11_launch_receipt"),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
    }


def validate_completion_receipt(
    completion_receipt: Path,
    *,
    checkpoint: Path,
    checkpoint_contract: Mapping[str, Any],
    launch: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v11_run_path(
        checkpoint,
        description="v11_completion_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v11_run_path(
        completion_receipt,
        description="v11_completion_receipt",
        ssd_root=ssd_root,
    )
    _regular_file(path, description="v11_completion_receipt")
    require(
        path.parent == v11_run_root_for(ssd_root) / "logs",
        "v11_completion_receipt_must_be_under_logs",
    )
    payload = _load_object(path, description="v11_completion_receipt")
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v11_completion_receipt",
    )
    require(
        payload.get("schema") == COMPLETION_RECEIPT_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("optimizer_step") == 1
        and payload.get("consumed_pair_presentations") == 7
        and payload.get("checkpoint") == str(resolved_checkpoint),
        "v11_completion_receipt_horizon_differs",
    )
    _validate_exact_artifact_binding(
        payload.get("launch_receipt", {}),
        expected_path=Path(str(launch["artifact"]["path"])),
        description="v11_completion_launch_receipt",
    )
    require(
        payload.get("launch_receipt_sha256") == launch["receipt_sha256"],
        "v11_completion_launch_receipt_self_hash_differs",
    )
    expected_checkpoint_artifacts = {
        filename: artifact_binding(
            resolved_checkpoint / filename,
            description=f"v11_completion_checkpoint_{filename}",
        )
        for filename in REQUIRED_CHECKPOINT_ARTIFACTS
    }
    expected_rng = {
        rng_path.name: artifact_binding(
            rng_path,
            description=f"v11_completion_checkpoint_{rng_path.name}",
        )
        for rng_path in sorted(resolved_checkpoint.glob("rng_state*.pth"))
    }
    require(
        payload.get("checkpoint_artifacts") == expected_checkpoint_artifacts
        and payload.get("rng_state_artifacts") == expected_rng,
        "v11_completion_checkpoint_artifacts_differ",
    )
    training_summary_binding = payload.get("training_summary")
    require(
        isinstance(training_summary_binding, Mapping),
        "v11_completion_training_summary_missing",
    )
    expected_summary_path = resolved_checkpoint.parents[1] / "training_summary.json"
    summary_artifact = _validate_exact_artifact_binding(
        training_summary_binding,
        expected_path=expected_summary_path,
        description="v11_completion_training_summary",
    )
    summary = _load_object(
        expected_summary_path,
        description="v11_completion_training_summary",
    )
    require(
        summary.get("memory_objective_version") == OBJECTIVE_VERSION
        and summary.get("warm_start_mode") == WARM_START_MODE
        and summary.get("training_protocol_sha256")
        == checkpoint_contract["training_protocol_sha256"],
        "v11_completion_training_summary_identity_differs",
    )
    log_binding = payload.get("log")
    require(isinstance(log_binding, Mapping), "v11_completion_log_missing")
    log_path = Path(str(log_binding.get("path", "")))
    require(
        log_path.parent == v11_run_root_for(ssd_root) / "logs",
        "v11_completion_log_path_differs",
    )
    log_artifact = _validate_exact_artifact_binding(
        log_binding,
        expected_path=log_path,
        description="v11_completion_log",
    )
    require(
        payload.get("cycle_pair_telemetry")
        == checkpoint_contract["cycle_pair_telemetry"],
        "v11_completion_cycle_pair_telemetry_differs",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v11_completion_authorization_differs",
    )
    return {
        "artifact": artifact_binding(path, description="v11_completion_receipt"),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
        "training_summary": summary_artifact,
        "log": log_artifact,
    }


def validate_training_provenance(
    *,
    checkpoint: Path,
    checkpoint_contract: Mapping[str, Any],
    launch_receipt: Path,
    completion_receipt: Path,
    baseline: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    launch = validate_launch_receipt(
        launch_receipt,
        checkpoint=checkpoint,
        baseline=baseline,
        base_model_identity=base_model_identity,
        ssd_root=ssd_root,
        project_root=project_root,
    )
    completion = validate_completion_receipt(
        completion_receipt,
        checkpoint=checkpoint,
        checkpoint_contract=checkpoint_contract,
        launch=launch,
        ssd_root=ssd_root,
    )
    return {
        "schema": "rwkv_ms_scene_memory_v11_training_provenance.v1",
        "launch_receipt": {
            "artifact": launch["artifact"],
            "receipt_sha256": launch["receipt_sha256"],
        },
        "completion_receipt": {
            "artifact": completion["artifact"],
            "receipt_sha256": completion["receipt_sha256"],
        },
        "git_commit": launch["payload"]["git_commit"],
        "critical_files": launch["payload"]["critical_files"],
        "base_model_identity": launch["payload"]["base_model_identity"],
        "v10_diagnostic_baseline": launch["payload"][
            "v10_diagnostic_baseline"
        ],
        "cycle_pair_telemetry": checkpoint_contract["cycle_pair_telemetry"],
        "training_summary": completion["training_summary"],
        "log": completion["log"],
    }


def validate_resume_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise LaunchContractError("V11 forbids resume and any second training cycle")


def validate_launch_contract(
    *,
    target_step: int,
    resume_checkpoint: Path | None = None,
    gate_receipt: Path | None = None,
    smoke: bool = False,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    base_model_path: Path = PINNED_BASE_MODEL,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require(target_step == 1, "v11_target_step_must_be_one")
    require(resume_checkpoint is None, "v11_resume_is_forbidden")
    require(gate_receipt is None, "v11_gate_receipt_cannot_authorize_training")
    data = validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    warm = validate_warm_start_contract(
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    baseline = validate_v10_diagnostic_baseline(ssd_root=ssd_root)
    base_model_identity = validate_base_model_contract(
        base_model=base_model_path,
        baseline=baseline,
        ssd_root=ssd_root,
    )
    require(
        warm["warm_start_checkpoint"] != baseline["checkpoint"],
        "v11_v10_diagnostic_must_never_be_warm_start",
    )
    first = data["entries"][0]
    public_data = {key: value for key, value in data.items() if key != "entries"}
    public_warm = {key: value for key, value in warm.items() if key != "lock"}
    return {
        **public_data,
        **public_warm,
        "launch_mode": "warm_start_smoke" if smoke else "warm_start",
        "source_step": 0,
        "target_step": 1,
        "resume_checkpoint": None,
        "resume_schedule_cursor": 0,
        "next_pair_low_ordinal": first["canonical_pair_ordinals"][0],
        "next_pair_high_ordinal": first["canonical_pair_ordinals"][1],
        "next_schedule_entry_sha256": first["entry_sha256"],
        "total_pair_presentations": 7,
        "total_optimizer_steps": 1,
        "checkpoint_steps": [1],
        "presentation_checkpoint_steps": [7],
        "objective_version": OBJECTIVE_VERSION,
        "pairing_objective_version": PAIRING_OBJECTIVE_VERSION,
        "pair_physical_batch_size": 1,
        "pair_logical_batch_size": 2,
        "pair_directional_exposures": 2,
        "pair_presentations_per_optimizer_step": 7,
        "gradient_accumulation_steps": 7,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "save_steps": 1,
        "hard32_access": HARD32_ACCESS_POLICY,
        "training_continuation_policy": TRAINING_CONTINUATION_POLICY,
        "v10_diagnostic_baseline": baseline,
        "base_model_identity": base_model_identity,
        "critical_files": critical_training_code_bindings(),
    }


def _tsv(result: Mapping[str, Any]) -> str:
    fields = (
        result["train_file"],
        result["train_file_sha256"],
        result["source_manifest"],
        result["source_manifest_file_sha256"],
        result["schedule"],
        result["schedule_file_sha256"],
        result["warm_start_checkpoint"],
        result["launch_mode"],
        result["source_step"],
        result["target_step"],
        result["resume_schedule_cursor"],
        result["next_pair_low_ordinal"],
        result["next_pair_high_ordinal"],
        result["next_schedule_entry_sha256"],
        result["save_steps"],
        result["v10_diagnostic_baseline"]["summary"]["file_sha256"],
    )
    rendered = tuple(str(value) for value in fields)
    require(
        all("\t" not in value and "\n" not in value for value in rendered),
        "v11_launch_contract_tsv_control_character",
    )
    return "\t".join(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--gate-receipt", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    parser.add_argument("--warm-start-lock", type=Path, default=WARM_START_LOCK)
    parser.add_argument("--base-model", type=Path, default=PINNED_BASE_MODEL)
    parser.add_argument("--ssd-root", type=Path, default=SSD_ROOT)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_launch_contract(
            target_step=args.target_step,
            resume_checkpoint=args.resume_checkpoint,
            gate_receipt=args.gate_receipt,
            smoke=args.smoke,
            data_root=args.data_root,
            source_lock_path=args.source_lock,
            warm_start_lock_path=args.warm_start_lock,
            base_model_path=args.base_model,
            ssd_root=args.ssd_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(_tsv(result) if args.format == "tsv" else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
