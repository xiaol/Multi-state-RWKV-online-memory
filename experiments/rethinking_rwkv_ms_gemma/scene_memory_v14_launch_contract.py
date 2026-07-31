#!/usr/bin/env python3
"""Validate the fresh, adapter-only Scene Memory V14 four-cycle launch."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v13_launch_contract as v13,
)
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v14_warm_start as warm


SSD_ROOT = v13.SSD_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = v13.DATA_ROOT
SOURCE_LOCK = v13.SOURCE_LOCK
WARM_START_LOCK = Path(warm.DEFAULT_LOCK_PATH)
PINNED_BASE_MODEL = v13.PINNED_BASE_MODEL
PINNED_HISTORICAL_TRAIN32_ARTIFACTS = v13.PINNED_HISTORICAL_TRAIN32_ARTIFACTS
PINNED_DATA_ARTIFACTS = v13.PINNED_DATA_ARTIFACTS
PINNED_V10_DIAGNOSTIC_SUMMARY = v13.PINNED_V10_DIAGNOSTIC_SUMMARY
V10_DIAGNOSTIC_BASELINE_METRICS = v13.V10_DIAGNOSTIC_BASELINE_METRICS
PINNED_WARM_START_CHECKPOINT = Path(warm.PINNED_SOURCE_CHECKPOINT)
PINNED_WARM_START_ADAPTER_SHA256 = warm.PINNED_ADAPTER_SHA256

V14_RUN_ROOT = (
    SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v14"
)
V14_GATES_ROOT = V14_RUN_ROOT / "gates"
_V14_RUN_RELATIVE = V14_RUN_ROOT.relative_to(SSD_ROOT)
_V14_GATES_RELATIVE = V14_GATES_ROOT.relative_to(SSD_ROOT)

OBJECTIVE_VERSION = "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14"
OBJECTIVE_SCHEMA_VERSION = 17
PAIRING_OBJECTIVE_VERSION = v13.PAIRING_OBJECTIVE_VERSION
FIXED_SAMPLER_MODE = "explicit_ordered_v14_four_canonical_seven_pair_cycles_v1"
PAIR_PHYSICAL_BATCH_SIZE = v13.PAIR_PHYSICAL_BATCH_SIZE
PAIR_LOGICAL_BATCH_SIZE = v13.PAIR_LOGICAL_BATCH_SIZE
PAIR_DIRECTIONAL_EXPOSURES = v13.PAIR_DIRECTIONAL_EXPOSURES
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 7
TOTAL_PAIR_PRESENTATIONS = 28
TOTAL_OPTIMIZER_STEPS = 4
CHECKPOINT_STEPS = (1, 2, 3, 4)
PRESENTATION_CHECKPOINTS = (7, 14, 21, 28)
ONE_PAIR_SMOKE_CHECKPOINT_STEPS = (1,)
ONE_PAIR_SMOKE_PRESENTATION_CHECKPOINTS = (1,)
CONTINUATION_POLICY = "forbidden"
TRAINING_CONTINUATION_POLICY = "forbidden_fresh_four_cycle_run_is_terminal"
GRADIENT_ACCUMULATION_STEPS = 7
SAVE_STEPS = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
OPTIMIZER_IMPLEMENTATION = "adamw_torch_fused"
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0
WARMUP_RATIO = 0.0
LOGGING_STEPS = 1
SEED = 42
DATA_SEED = 42
PREFIX_CORRECTION_WEIGHT = 0.0
SEMANTIC_MARGIN = 1.0
GENERATED_MAX_CORRECTION_EVENTS = 0
GENERATED_ROLLOUT_EXTRA_TOKENS = 4
GENERATED_ROLLOUT_MAX_TOKENS = 24
MAX_LENGTH = 256
MAX_WRITE_LENGTH = 2048
TEACHER_MAX_LENGTH = MAX_LENGTH + MAX_WRITE_LENGTH

FIRST_CYCLE_PAIRS = v13.FIRST_CYCLE_PAIRS
SECOND_CYCLE_PAIRS = v13.SECOND_CYCLE_PAIRS
THIRD_CYCLE_PAIRS = v13.THIRD_CYCLE_PAIRS
FOURTH_CYCLE_PAIRS = v13.FOURTH_CYCLE_PAIRS
FOUR_CYCLE_PAIRS = v13.FOUR_CYCLE_PAIRS
FIRST_CYCLE_PAIRS_SHA256 = v13.FIRST_CYCLE_PAIRS_SHA256
FOUR_CYCLE_PAIRS_SHA256 = v13.FOUR_CYCLE_PAIRS_SHA256
PAIR_PREFIX_SHA256_BY_CHECKPOINT = dict(v13.PAIR_PREFIX_SHA256_BY_CHECKPOINT)

ONE_PAIR_SMOKE_PAIR = FOUR_CYCLE_PAIRS[0]
ONE_PAIR_SMOKE_RUN_MODE = "one_pair_real_backward_optimizer_step_smoke_v1"
ONE_PAIR_SMOKE_SAMPLER_MODE = "explicit_ordered_v14_first_canonical_pair_smoke_v1"
ONE_PAIR_SMOKE_SCHEDULE_SELECTION_MODE = (
    "first_canonical_pair_only_from_verified_v13_four_cycle_schedule_v1"
)
PAIR_CURRICULUM_BINDING_SCHEMA = (
    "rwkv_ms_scene_memory_v9_pair_curriculum_binding.v1"
)

CACHED_PREFIX_MODE = (
    "cached_actual_greedy_prefix_failed_repair_cached_gold_prefix_all_decision_"
    "retention_v1"
)
FAILED_DECISION_ALIGNMENT_MODE = v13.FAILED_DECISION_ALIGNMENT_MODE
DECISION_MASK_MODE = v13.DENSE_DECISION_MASK_MODE
DECISION_TOKEN_OVERLAP_POLICY = v13.DENSE_DECISION_TOKEN_OVERLAP_POLICY
FAILED_REPLAY_MODE = "use_cache_true_logits_to_keep_1_actual_greedy_prefix_v2"
EXACT_REPLAY_MODE = "use_cache_true_logits_to_keep_1_gold_prefix_v1"
REPLAY_LOGITS_TO_KEEP = 1
EXACT_RETENTION_SCOPE = "all_boundary_decision_mask_tokens_cached_gold_prefix_v1"
EXACT_RETENTION_HINGE_MODE = "cached_gold_vs_detached_top_competitor_hinge_v1"
PARSED_EXACTNESS_MODE = v13.PARSED_EXACTNESS_MODE
PARSED_EXACTNESS = v13.PARSED_EXACTNESS
OBJECTIVE_FORMULA = (
    "symmetric_pair_mean(if(parsed_boundary_exact,"
    "all_boundary_decision_cached_gold_prefix_retention_hinge(1.0),"
    "cached_actual_greedy_prefix_semantic_ce+actual_greedy_competitor_hinge(1.0))); "
    "use_cache=true; logits_to_keep=1; full_answer_schema_footer_and_"
    "chat_termination_ce=0; selected_pair_and_zero_hinges=telemetry_only; "
    "raw_token_exact=telemetry_only"
)
BACKWARD_MODE = (
    "sequential_pair_zero_probe_telemetry_no_grad_then_cached_prefix_only_"
    "backward_v10"
)
CYCLE_RETENTION_MODE = v13.CYCLE_RETENTION_MODE
ROW_OBJECTIVE_AUDIT_FILENAME = "scene_memory_v14_row_objective.json"
ROW_OBJECTIVE_AUDIT_SCHEMA = "rwkv_ms_scene_memory_v14_row_objective.v1"
HARD32_ACCESS_POLICY = (
    "hard32_forbidden_until_value14_gate_passes_pinned_train32_source_allowed_v1"
)
LAUNCH_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v14_attached_launch.v1"
COMPLETION_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v14_attached_completion.v1"
ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v14_one_pair_smoke_launch.v1"
)
ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v14_one_pair_smoke_completion.v1"
)

REQUIRED_CHECKPOINT_ARTIFACTS = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_protocol.json",
    "scene_state_identity_pairing_manifest.json",
    ROW_OBJECTIVE_AUDIT_FILENAME,
    warm.WARM_START_LINEAGE_FILENAME,
)
CRITICAL_TRAINING_FILES = (
    "deltamem/scene_boundary.py",
    "deltamem/train/cached_prefix_replay.py",
    "deltamem/train/delta_sft_experimental.py",
    "deltamem/train/scene_state_generation_alignment.py",
    "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v9_data.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_data_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_source_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_v13_checkpoint4_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v14_gate.py",
    "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v14.sh",
)


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


canonical_sha256 = v13.canonical_sha256
sha256_file = v13.sha256_file
require_sha256 = v13.require_sha256
require_exact_path = v13.require_exact_path
require_under_root = v13.require_under_root
require_ssd = v13.require_ssd
_lexically_guard_path = v13._lexically_guard_path
_regular_file = v13._regular_file
_load_object = v13._load_object
artifact_binding = v13.artifact_binding


def guard_v14_training_data_path(path: Path | str, *, description: str) -> Path:
    try:
        return v13.guard_v13_training_data_path(path, description=description)
    except Exception as exc:
        raise LaunchContractError(str(exc).replace("v13", "v14")) from exc


def v14_run_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V14_RUN_RELATIVE


def v14_gates_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V14_GATES_RELATIVE


def require_v14_run_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v14_run_root_for(ssd_root),
        description=description,
    )


def require_v14_gate_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v14_gates_root_for(ssd_root),
        description=description,
    )


def presentation_cursor(global_step: int) -> int:
    require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step in (0, *CHECKPOINT_STEPS),
        "global_step_outside_v14_four_cycle_schedule",
    )
    return global_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP


def _finite_integral_metric(value: Any, *, description: str) -> int:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and int(value) == float(value),
        f"{description}_must_be_finite_integral",
    )
    return int(value)


def validate_v14_cycle_pair_telemetry(
    trainer_state: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    require(checkpoint_step in CHECKPOINT_STEPS, "v14_checkpoint_step_differs")
    history = trainer_state.get("log_history")
    require(isinstance(history, list), "v14_log_history_missing")
    by_step: dict[int, Mapping[str, Any]] = {}
    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        if "delta/scene_generation_v14_cycle_index" not in entry:
            continue
        step = _finite_integral_metric(entry.get("step"), description="v14_log_step")
        require(step not in by_step, "v14_cycle_telemetry_duplicate_step")
        by_step[step] = entry
    expected_steps = set(range(1, checkpoint_step + 1))
    require(set(by_step) == expected_steps, "v14_cycle_telemetry_steps_differ")
    cycles: list[dict[str, Any]] = []
    ordered_pairs: list[list[int]] = []
    for step in range(1, checkpoint_step + 1):
        entry = by_step[step]
        require(
            _finite_integral_metric(
                entry.get("delta/scene_generation_v14_cycle_index"),
                description="v14_cycle_index",
            )
            == step,
            "v14_cycle_index_differs",
        )
        require(
            _finite_integral_metric(
                entry.get("delta/scene_generation_v14_cycle_pair_presentations"),
                description="v14_cycle_pair_presentations",
            )
            == PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
            "v14_cycle_pair_presentations_differ",
        )
        observed: list[tuple[int, int]] = []
        for index in range(PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP):
            low = _finite_integral_metric(
                entry.get(
                    f"delta/scene_generation_v14_cycle_pair_{index}_low_ordinal"
                ),
                description=f"v14_cycle_pair_{index}_low",
            )
            high = _finite_integral_metric(
                entry.get(
                    f"delta/scene_generation_v14_cycle_pair_{index}_high_ordinal"
                ),
                description=f"v14_cycle_pair_{index}_high",
            )
            observed.append((low, high))
        expected = FOUR_CYCLE_PAIRS[(step - 1) * 7 : step * 7]
        require(tuple(observed) == tuple(expected), "v14_cycle_pair_order_differs")
        ordered_pairs.extend([list(pair) for pair in observed])
        cycles.append({"optimizer_step": step, "ordered_pairs": [list(pair) for pair in observed]})
    prefix = FOUR_CYCLE_PAIRS[: checkpoint_step * 7]
    require(
        canonical_sha256(ordered_pairs)
        == PAIR_PREFIX_SHA256_BY_CHECKPOINT[checkpoint_step],
        "v14_cycle_pair_prefix_hash_differs",
    )
    return {
        "optimizer_step": checkpoint_step,
        "pair_presentations": checkpoint_step * 7,
        "ordered_pairs": [list(pair) for pair in prefix],
        "ordered_pairs_sha256": PAIR_PREFIX_SHA256_BY_CHECKPOINT[checkpoint_step],
        "cycles": cycles,
    }


def _validate_one_pair_smoke_telemetry(
    trainer_state: Mapping[str, Any],
) -> dict[str, Any]:
    history = trainer_state.get("log_history")
    require(isinstance(history, list), "v14_smoke_log_history_missing")
    entries = [
        entry
        for entry in history
        if isinstance(entry, Mapping)
        and "delta/scene_generation_v14_cycle_index" in entry
    ]
    require(len(entries) == 1, "v14_smoke_cycle_telemetry_count_differs")
    entry = entries[0]
    require(
        _finite_integral_metric(entry.get("step"), description="v14_smoke_log_step")
        == 1
        and _finite_integral_metric(
            entry.get("delta/scene_generation_v14_cycle_index"),
            description="v14_smoke_cycle_index",
        )
        == 1
        and _finite_integral_metric(
            entry.get("delta/scene_generation_v14_cycle_pair_presentations"),
            description="v14_smoke_cycle_pair_presentations",
        )
        == 1,
        "v14_smoke_cycle_telemetry_horizon_differs",
    )
    pair_metric_prefix = "delta/scene_generation_v14_cycle_pair_"
    ordinal_metrics = {
        key: value
        for key, value in entry.items()
        if isinstance(key, str)
        and key.startswith(pair_metric_prefix)
        and key.endswith(("_low_ordinal", "_high_ordinal"))
    }
    low, high = ONE_PAIR_SMOKE_PAIR
    require(
        set(ordinal_metrics)
        == {
            f"{pair_metric_prefix}0_low_ordinal",
            f"{pair_metric_prefix}0_high_ordinal",
        }
        and _finite_integral_metric(
            ordinal_metrics[f"{pair_metric_prefix}0_low_ordinal"],
            description="v14_smoke_pair_low",
        )
        == low
        and _finite_integral_metric(
            ordinal_metrics[f"{pair_metric_prefix}0_high_ordinal"],
            description="v14_smoke_pair_high",
        )
        == high,
        "v14_smoke_cycle_pair_order_differs",
    )
    loss = _finite_number(entry.get("loss"), description="v14_smoke_loss")
    grad_norm = _finite_number(
        entry.get("grad_norm"),
        description="v14_smoke_grad_norm",
    )
    learning_rate = _finite_number(
        entry.get("learning_rate"),
        description="v14_smoke_learning_rate",
    )
    require(loss > 0.0, "v14_smoke_loss_must_be_positive")
    require(grad_norm > 0.0, "v14_smoke_grad_norm_must_be_positive")
    require(
        learning_rate == LEARNING_RATE,
        "v14_smoke_learning_rate_differs",
    )
    ordered_pairs = [list(ONE_PAIR_SMOKE_PAIR)]
    return {
        "optimizer_step": 1,
        "pair_presentations": 1,
        "ordered_pairs": ordered_pairs,
        "ordered_pairs_sha256": canonical_sha256(ordered_pairs),
        "cycles": [{"optimizer_step": 1, "ordered_pairs": ordered_pairs}],
        "loss": loss,
        "grad_norm": grad_norm,
        "learning_rate": learning_rate,
    }


def validate_v10_diagnostic_baseline(
    *, ssd_root: Path = SSD_ROOT
) -> dict[str, Any]:
    return v13.validate_v10_diagnostic_baseline(ssd_root=ssd_root)


def validate_base_model_contract(
    *,
    base_model: Path = PINNED_BASE_MODEL,
    baseline: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    return v13.validate_base_model_contract(
        base_model=base_model,
        baseline=baseline,
        ssd_root=ssd_root,
    )


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    result = v13.validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    require(
        result.get("four_cycle_pairs") == [list(pair) for pair in FOUR_CYCLE_PAIRS]
        and result.get("four_cycle_pairs_sha256") == FOUR_CYCLE_PAIRS_SHA256,
        "v14_data_four_cycle_identity_differs",
    )
    return result


def validate_warm_start_contract(
    *,
    checkpoint: Path = PINNED_WARM_START_CHECKPOINT,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require_exact_path(ssd_root, SSD_ROOT, description="v14_ssd_root")
    source = require_exact_path(
        checkpoint,
        PINNED_WARM_START_CHECKPOINT,
        description="v14_warm_start_checkpoint",
    )
    try:
        lock_path = require_exact_path(
            warm_start_lock_path,
            WARM_START_LOCK,
            description="v14_warm_start_lock",
        )
        context = warm.prepare_v14_v13_checkpoint4_warm_start(
            source,
            lock_path=lock_path,
        )
    except Exception as exc:
        raise LaunchContractError(f"v14_warm_start_contract_failed: {exc}") from exc
    adapter = source / "delta_mem_adapter.pt"
    _regular_file(adapter, description="v14_warm_start_adapter")
    require(
        sha256_file(adapter) == PINNED_WARM_START_ADAPTER_SHA256,
        "v14_warm_start_adapter_hash_differs",
    )
    require(
        context.source_trainer_state.get("global_step") == 4,
        "v14_warm_start_source_step_differs",
    )
    return {
        "warm_start_checkpoint": str(source),
        "warm_start_adapter_sha256": PINNED_WARM_START_ADAPTER_SHA256,
        "warm_start_lock": str(context.lock_path),
        "warm_start_lock_sha256": context.lock["lock_sha256"],
        "warm_start_mode": warm.WARM_START_MODE,
        "source_global_step": 4,
        "context": context,
    }


def _validate_checkpoint_protocol(
    protocol: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
) -> None:
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, Mapping), "v14_protocol_train_schedule_missing")
    expected = {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": OBJECTIVE_VERSION,
        "max_steps": TOTAL_OPTIMIZER_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "save_steps": SAVE_STEPS,
        "save_total_limit": len(CHECKPOINT_STEPS),
        "ignore_data_skip": False,
        "scene_generation_objective_formula": OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": BACKWARD_MODE,
        "scene_generation_generated_prefix_correction_weight": 0.0,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 0,
        "scene_generation_generated_prefix_max_correction_events": 0,
        "scene_generation_generated_rollout_extra_tokens": GENERATED_ROLLOUT_EXTRA_TOKENS,
        "scene_generation_generated_rollout_max_tokens": GENERATED_ROLLOUT_MAX_TOKENS,
        "scene_generation_generated_prefix_correction_mode": CACHED_PREFIX_MODE,
        "scene_generation_parsed_exactness": PARSED_EXACTNESS,
        "scene_generation_parsed_exactness_mode": PARSED_EXACTNESS_MODE,
        "scene_generation_raw_token_exactness_role": "telemetry_only",
        "scene_generation_failed_decision_alignment": FAILED_DECISION_ALIGNMENT_MODE,
        "scene_generation_failed_replay_mode": FAILED_REPLAY_MODE,
        "scene_generation_exact_replay_mode": EXACT_REPLAY_MODE,
        "scene_generation_cached_replay_use_cache": True,
        "scene_generation_cached_replay_logits_to_keep": REPLAY_LOGITS_TO_KEEP,
        "scene_generation_cached_replay_model_mode": (
            "train_grad_enabled_activation_checkpointing_zero_dropout_v1"
        ),
        "scene_generation_failed_replay_top1_parity_scope": (
            "every_actual_greedy_prefix_token_through_selected_cursor_v1"
        ),
        "scene_generation_cached_prefix_semantic_mode": CACHED_PREFIX_MODE,
        "scene_generation_cached_decision_token_overlap_policy": (
            DECISION_TOKEN_OVERLAP_POLICY
        ),
        "scene_generation_exact_retention_scope": EXACT_RETENTION_SCOPE,
        "scene_generation_exact_retention_hinge_weight": 1.0,
        "scene_generation_exact_retention_hinge_mode": EXACT_RETENTION_HINGE_MODE,
        "scene_generation_exact_retention_margin": SEMANTIC_MARGIN,
        "scene_generation_failed_semantic_repair_ce_weight": 1.0,
        "scene_generation_failed_semantic_repair_hinge_weight": 1.0,
        "scene_generation_failed_semantic_repair_margin": SEMANTIC_MARGIN,
        "scene_generation_teacher_forced_full_forward_mode": (
            "no_grad_telemetry_only_v1"
        ),
        "scene_generation_full_answer_ce_optimization_weight": 0.0,
        "scene_generation_schema_ce_optimization_weight": 0.0,
        "scene_generation_footer_ce_optimization_weight": 0.0,
        "scene_generation_termination_ce_optimization_weight": 0.0,
        "scene_generation_selected_full_vocab_ce_in_total": False,
        "scene_generation_selected_full_vocab_ce_optimization_weight": 0.0,
        "scene_generation_selected_pair_auxiliary_optimization_weight": 0.0,
        "scene_generation_zero_state_auxiliary_optimization_weight": 0.0,
        "scene_generation_row_objective_audit_filename": ROW_OBJECTIVE_AUDIT_FILENAME,
        "scene_generation_row_objective_audit_schema": ROW_OBJECTIVE_AUDIT_SCHEMA,
        "scene_generation_cycle_retention_mode": CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": 7,
        "scene_generation_gradient_accumulation_pair_cycle": 7,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    require(not mismatches, "v14_protocol_differs fields=" + ",".join(mismatches))
    schedule_expected = {
        "checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
        "optimizer_checkpoint_steps": list(CHECKPOINT_STEPS),
        "microbatch_cycle_size": 7,
        "continuation_policy": CONTINUATION_POLICY,
    }
    schedule_mismatches = [
        name for name, value in schedule_expected.items() if schedule.get(name) != value
    ]
    if "resume_schedule_cursor_formula" in schedule:
        schedule_mismatches.append("resume_schedule_cursor_formula")
    require(
        not schedule_mismatches,
        "v14_protocol_schedule_differs fields=" + ",".join(schedule_mismatches),
    )
    require(
        protocol.get("scene_state_source_manifest", {}).get("train_file")
        == data.get("train_file"),
        "v14_protocol_training_source_differs",
    )


def _validate_one_pair_smoke_checkpoint_protocol(
    protocol: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
) -> None:
    schedule = protocol.get("train_schedule")
    require(
        isinstance(schedule, Mapping),
        "v14_smoke_protocol_train_schedule_missing",
    )
    smoke_expected = {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": OBJECTIVE_VERSION,
        "max_steps": 1,
        "max_grad_norm": MAX_GRAD_NORM,
        "train_sampler_mode": ONE_PAIR_SMOKE_SAMPLER_MODE,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "optim": OPTIMIZER_IMPLEMENTATION,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 1,
        "seed": SEED,
        "data_seed": DATA_SEED,
        "ignore_data_skip": False,
        "scene_generation_v14_run_mode": ONE_PAIR_SMOKE_RUN_MODE,
        "scene_generation_v14_production_eligible": False,
        "scene_generation_cycle_pair_presentations": 1,
        "scene_generation_gradient_accumulation_pair_cycle": 1,
        "scene_generation_raw_token_exact_optimization_weight": 0.0,
        "scene_generation_schema_ce_optimization_scope": (
            "standalone_schema_mask_partition_only_v1"
        ),
        "scene_generation_failed_prefix_replay": FAILED_REPLAY_MODE,
    }
    mismatches = [
        name
        for name, expected in smoke_expected.items()
        if protocol.get(name) != expected
    ]
    require(
        not mismatches,
        "v14_smoke_protocol_differs fields=" + ",".join(mismatches),
    )
    schedule_expected = {
        "schema": PAIR_CURRICULUM_BINDING_SCHEMA,
        "checkpoint_steps": [1],
        "optimizer_checkpoint_steps": [1],
        "microbatch_cycle_size": 1,
        "continuation_policy": CONTINUATION_POLICY,
        "source_total_steps": TOTAL_PAIR_PRESENTATIONS,
        "source_checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
        "source_ordered_pairs_sha256": FOUR_CYCLE_PAIRS_SHA256,
        "schedule_selection_mode": ONE_PAIR_SMOKE_SCHEDULE_SELECTION_MODE,
        "active_ordered_pairs_sha256": canonical_sha256(
            [list(ONE_PAIR_SMOKE_PAIR)]
        ),
        "total_steps": 1,
        "pair_indices": [list(ONE_PAIR_SMOKE_PAIR)],
    }
    schedule_mismatches = [
        name
        for name, expected in schedule_expected.items()
        if schedule.get(name) != expected
    ]
    if "resume_schedule_cursor_formula" in schedule:
        schedule_mismatches.append("resume_schedule_cursor_formula")
    require(
        not schedule_mismatches,
        "v14_smoke_protocol_schedule_differs fields="
        + ",".join(schedule_mismatches),
    )

    # Reuse the production checker for every objective invariant that is
    # intentionally identical between the smoke and four-cycle launch.
    normalized = dict(protocol)
    normalized.update(
        {
            "max_steps": TOTAL_OPTIMIZER_STEPS,
            "train_sampler_mode": FIXED_SAMPLER_MODE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "save_total_limit": len(CHECKPOINT_STEPS),
            "scene_generation_cycle_pair_presentations": (
                PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
            ),
            "scene_generation_gradient_accumulation_pair_cycle": (
                PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
            ),
        }
    )
    normalized_schedule = dict(schedule)
    normalized_schedule.update(
        {
            "checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
            "optimizer_checkpoint_steps": list(CHECKPOINT_STEPS),
            "microbatch_cycle_size": PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        }
    )
    normalized_schedule.pop("resume_schedule_cursor_formula", None)
    normalized["train_schedule"] = normalized_schedule
    _validate_checkpoint_protocol(normalized, data=data)


_ROW_REQUIRED_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "pair_role",
        "row_ordinal",
        "paired_row_ordinal",
        "row_sha256",
        "paired_row_sha256",
        "parsed_boundary_exact",
        "raw_token_exact",
        "first_divergence",
        "rollout_token_count",
        "cached_branch_kind",
        "cached_branch_kind_code",
        "cached_replay_use_cache",
        "cached_replay_logits_to_keep",
        "cached_replay_token_count",
        "cached_replay_selected_cursor",
        "cached_decision_token_count",
        "cached_selected_decision_ordinal",
        "cached_selected_label_position",
        "cached_selected_gold_token_id",
        "cached_selected_competitor_id",
        "cached_competitor_is_actual_greedy",
        "cached_replay_top1_matches_actual",
        "cached_replay_top1_match_count",
        "cached_ce",
        "cached_failed_competitor_hinge",
        "cached_exact_retention_hinge",
        "cached_selected_gold_vs_competitor_margin",
        "cached_gold_top1_fraction",
        "cached_alignment_kind_code",
        "cached_selected_is_termination",
        "cached_branch_loss",
        "auxiliary_optimization_loss",
        "auxiliary_telemetry_loss",
        "selected_top_competitor_hinge_telemetry",
        "selected_correct_vs_zero_hinge_telemetry",
        "total_side_loss",
    }
)
_PAIR_REQUIRED_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "source_row_ordinal",
        "donor_row_ordinal",
        "source_row_sha256",
        "donor_row_sha256",
        "pair_mean_cached_branch_loss",
        "pair_mean_cached_exact_retention_hinge",
        "pair_mean_cached_failed_ce",
        "pair_mean_cached_failed_competitor_hinge",
        "pair_mean_auxiliary_optimization_loss",
        "pair_mean_selected_top_competitor_hinge_telemetry",
        "pair_mean_selected_correct_vs_zero_hinge_telemetry",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    }
)


def _finite_number(value: Any, *, description: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{description}_must_be_finite",
    )
    return float(value)


def _validate_v14_row_objective_audit(
    audit: Mapping[str, Any],
    *,
    checkpoint_step: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    require(audit.get("schema") == ROW_OBJECTIVE_AUDIT_SCHEMA, "v14_row_audit_schema_differs")
    require(audit.get("memory_objective_version") == OBJECTIVE_VERSION, "v14_row_audit_objective_differs")
    require(
        audit.get("checkpoint_optimizer_step") == checkpoint_step
        and audit.get("completed_pair_presentations") == checkpoint_step * 7,
        "v14_row_audit_horizon_differs",
    )
    phases = [f"cycle{index}_input" for index in range(1, checkpoint_step + 1)]
    require(audit.get("phases") == phases, "v14_row_audit_phases_differ")
    expected_pairs = FOUR_CYCLE_PAIRS[: checkpoint_step * 7]
    data_entries = data.get("entries")
    require(
        isinstance(data_entries, list)
        and len(data_entries) >= checkpoint_step * 7,
        "v14_row_audit_data_entries_missing",
    )
    require(
        audit.get("pair_schedule")
        == [
            {"source_row_ordinal": low, "donor_row_ordinal": high}
            for low, high in expected_pairs
        ],
        "v14_row_audit_pair_schedule_differs",
    )
    pairs = audit.get("pair_presentations")
    rows = audit.get("rows")
    require(isinstance(pairs, list) and len(pairs) == checkpoint_step * 7, "v14_row_audit_pair_count_differs")
    require(isinstance(rows, list) and len(rows) == 14, "v14_row_audit_row_count_differs")
    by_row: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "v14_row_audit_row_invalid")
        ordinal = row.get("row_ordinal")
        require(isinstance(ordinal, int) and not isinstance(ordinal, bool), "v14_row_audit_ordinal_invalid")
        require(ordinal not in by_row, "v14_row_audit_duplicate_row")
        by_row[ordinal] = row
    expected_row_order = [ordinal for pair in FIRST_CYCLE_PAIRS for ordinal in pair]
    require(list(by_row) == expected_row_order, "v14_row_audit_row_order_differs")
    for presentation, ((low, high), pair_observation) in enumerate(zip(expected_pairs, pairs, strict=True), 1):
        require(
            isinstance(pair_observation, Mapping)
            and set(pair_observation) == _PAIR_REQUIRED_FIELDS,
            "v14_row_audit_pair_fields_differ",
        )
        cycle = (presentation - 1) // 7 + 1
        require(
            pair_observation.get("phase") == f"cycle{cycle}_input"
            and pair_observation.get("cycle") == cycle
            and pair_observation.get("adapter_optimizer_step_before_update") == cycle - 1
            and pair_observation.get("presentation") == presentation
            and pair_observation.get("source_row_ordinal") == low
            and pair_observation.get("donor_row_ordinal") == high,
            "v14_row_audit_pair_identity_differs",
        )
        source_member, donor_member = data_entries[presentation - 1]["members"]
        require(
            source_member.get("train_row_ordinal") == low
            and donor_member.get("train_row_ordinal") == high
            and pair_observation.get("source_row_sha256")
            == source_member.get("row_sha256")
            and pair_observation.get("donor_row_sha256")
            == donor_member.get("row_sha256"),
            "v14_row_audit_pair_dataset_binding_differs",
        )
        cached = _finite_number(pair_observation.get("pair_mean_cached_branch_loss"), description="v14_pair_cached")
        recomputed_cached = sum(
            _finite_number(pair_observation.get(name), description=f"v14_pair_{name}")
            for name in (
                "pair_mean_cached_exact_retention_hinge",
                "pair_mean_cached_failed_ce",
                "pair_mean_cached_failed_competitor_hinge",
            )
        )
        auxiliary = _finite_number(pair_observation.get("pair_mean_auxiliary_optimization_loss"), description="v14_pair_auxiliary")
        total = recomputed_cached + auxiliary
        require(auxiliary == 0.0, "v14_pair_auxiliary_optimization_must_be_zero")
        for actual, expected in (
            (cached, recomputed_cached),
            (_finite_number(pair_observation.get("pair_mean_total_side_loss"), description="v14_pair_side"), total),
            (_finite_number(pair_observation.get("reported_objective_total_loss"), description="v14_pair_reported"), total),
            (_finite_number(pair_observation.get("recomputed_objective_total_loss"), description="v14_pair_recomputed"), total),
        ):
            require(math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6), "v14_row_audit_pair_arithmetic_differs")
        for role, ordinal, paired in (("source", low, high), ("donor", high, low)):
            observation = by_row[ordinal].get(f"cycle{cycle}_input")
            require(
                isinstance(observation, Mapping)
                and set(observation) == _ROW_REQUIRED_FIELDS,
                "v14_row_audit_row_fields_differ",
            )
            require(
                observation.get("pair_role") == role
                and observation.get("row_ordinal") == ordinal
                and observation.get("paired_row_ordinal") == paired
                and observation.get("presentation") == presentation,
                "v14_row_audit_row_identity_differs",
            )
            expected_member = source_member if role == "source" else donor_member
            expected_paired_member = donor_member if role == "source" else source_member
            require(
                observation.get("row_sha256") == expected_member.get("row_sha256")
                and observation.get("paired_row_sha256")
                == expected_paired_member.get("row_sha256"),
                "v14_row_audit_row_dataset_binding_differs",
            )
            parsed_exact = observation.get("parsed_boundary_exact") is True
            branch_code = observation.get("cached_branch_kind_code")
            require(branch_code in (0, 1) and parsed_exact == (branch_code == 0), "v14_row_audit_branch_differs")
            require(
                observation.get("cached_replay_use_cache") is True
                and observation.get("cached_replay_logits_to_keep") == 1
                and observation.get("cached_selected_is_termination") is False
                and observation.get("auxiliary_optimization_loss") == 0.0,
                "v14_row_audit_cached_contract_differs",
            )
            cached_ce = _finite_number(observation.get("cached_ce"), description="v14_row_cached_ce")
            failed_hinge = _finite_number(observation.get("cached_failed_competitor_hinge"), description="v14_row_failed_hinge")
            exact_hinge = _finite_number(observation.get("cached_exact_retention_hinge"), description="v14_row_exact_hinge")
            expected_loss = exact_hinge if parsed_exact else cached_ce + failed_hinge
            require(
                math.isclose(_finite_number(observation.get("cached_branch_loss"), description="v14_row_branch_loss"), expected_loss, rel_tol=1e-5, abs_tol=1e-6)
                and math.isclose(_finite_number(observation.get("total_side_loss"), description="v14_row_side_loss"), expected_loss, rel_tol=1e-5, abs_tol=1e-6),
                "v14_row_audit_row_arithmetic_differs",
            )
            if parsed_exact:
                require(
                    cached_ce == failed_hinge == 0.0
                    and observation.get("cached_alignment_kind_code") == -1
                    and observation.get("cached_competitor_is_actual_greedy") is False
                    and observation.get("cached_replay_top1_matches_actual") is False
                    and observation.get("cached_replay_top1_match_count") == 0,
                    "v14_exact_row_invented_failed_repair",
                )
            else:
                require(
                    exact_hinge == 0.0
                    and observation.get("cached_competitor_is_actual_greedy") is True
                    and observation.get("cached_replay_top1_matches_actual") is True
                    and observation.get("cached_replay_top1_match_count")
                    == observation.get("cached_replay_token_count")
                    and observation.get("cached_replay_token_count")
                    == observation.get("cached_replay_selected_cursor") + 1,
                    "v14_failed_row_cached_repair_incomplete",
                )
    return {
        "schema": ROW_OBJECTIVE_AUDIT_SCHEMA,
        "checkpoint_optimizer_step": checkpoint_step,
        "completed_pair_presentations": checkpoint_step * 7,
        "pair_schedule_sha256": canonical_sha256(audit["pair_schedule"]),
        "rows": 14,
        "pair_presentations": checkpoint_step * 7,
    }


def _validate_one_pair_smoke_row_objective_audit(
    audit: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        audit.get("schema") == ROW_OBJECTIVE_AUDIT_SCHEMA
        and audit.get("memory_objective_version") == OBJECTIVE_VERSION
        and audit.get("run_mode") == ONE_PAIR_SMOKE_RUN_MODE
        and audit.get("production_eligible") is False,
        "v14_smoke_row_audit_identity_differs",
    )
    require(
        audit.get("checkpoint_optimizer_step") == 1
        and audit.get("completed_pair_presentations") == 1
        and audit.get("phases") == ["smoke_input"],
        "v14_smoke_row_audit_horizon_differs",
    )
    low, high = ONE_PAIR_SMOKE_PAIR
    expected_schedule = [
        {"source_row_ordinal": low, "donor_row_ordinal": high}
    ]
    require(
        audit.get("pair_schedule") == expected_schedule,
        "v14_smoke_row_audit_pair_schedule_differs",
    )
    entries = data.get("entries")
    require(
        isinstance(entries, list) and bool(entries),
        "v14_smoke_row_audit_data_entry_missing",
    )
    first_entry = entries[0]
    require(
        isinstance(first_entry, Mapping)
        and first_entry.get("canonical_pair_ordinals") == [low, high],
        "v14_smoke_row_audit_data_pair_differs",
    )
    members = first_entry.get("members")
    require(
        isinstance(members, list)
        and len(members) == 2
        and all(isinstance(member, Mapping) for member in members),
        "v14_smoke_row_audit_data_members_missing",
    )
    source_member, donor_member = members
    require(
        source_member.get("train_row_ordinal") == low
        and donor_member.get("train_row_ordinal") == high,
        "v14_smoke_row_audit_data_member_ordinals_differ",
    )
    rows = audit.get("rows")
    require(
        isinstance(rows, list) and len(rows) == 2,
        "v14_smoke_row_audit_row_count_differs",
    )
    require(
        [row.get("row_ordinal") if isinstance(row, Mapping) else None for row in rows]
        == [low, high],
        "v14_smoke_row_audit_row_order_differs",
    )
    validated_rows: list[Mapping[str, Any]] = []
    for row, role, ordinal, paired, member, paired_member in (
        (rows[0], "source", low, high, source_member, donor_member),
        (rows[1], "donor", high, low, donor_member, source_member),
    ):
        require(
            isinstance(row, Mapping)
            and set(row) == {"row_ordinal", "smoke_input"},
            "v14_smoke_row_audit_row_container_differs",
        )
        observation = row.get("smoke_input")
        require(
            isinstance(observation, Mapping)
            and set(observation) == _ROW_REQUIRED_FIELDS,
            "v14_smoke_row_audit_row_fields_differ",
        )
        require(
            observation.get("phase") == "smoke_input"
            and observation.get("cycle") == 1
            and observation.get("adapter_optimizer_step_before_update") == 0
            and observation.get("presentation") == 1
            and observation.get("pair_role") == role
            and observation.get("row_ordinal") == ordinal
            and observation.get("paired_row_ordinal") == paired,
            "v14_smoke_row_audit_row_identity_differs",
        )
        require(
            observation.get("row_sha256") == member.get("row_sha256")
            and observation.get("paired_row_sha256")
            == paired_member.get("row_sha256"),
            "v14_smoke_row_audit_row_dataset_binding_differs",
        )
        for field in (
            "parsed_boundary_exact",
            "raw_token_exact",
            "cached_replay_use_cache",
            "cached_competitor_is_actual_greedy",
            "cached_replay_top1_matches_actual",
            "cached_selected_is_termination",
        ):
            require(
                isinstance(observation.get(field), bool),
                f"v14_smoke_row_audit_{field}_must_be_boolean",
            )
        parsed_exact = observation["parsed_boundary_exact"]
        branch_code = _finite_integral_metric(
            observation.get("cached_branch_kind_code"),
            description="v14_smoke_cached_branch_kind_code",
        )
        require(
            branch_code in (0, 1)
            and parsed_exact == (branch_code == 0)
            and observation.get("cached_branch_kind")
            == (
                "cached_gold_prefix"
                if parsed_exact
                else "cached_actual_greedy_prefix"
            ),
            "v14_smoke_row_audit_branch_differs",
        )
        integer_fields = {
            name: _finite_integral_metric(
                observation.get(name),
                description=f"v14_smoke_{name}",
            )
            for name in (
                "first_divergence",
                "rollout_token_count",
                "cached_replay_logits_to_keep",
                "cached_replay_token_count",
                "cached_replay_selected_cursor",
                "cached_decision_token_count",
                "cached_selected_decision_ordinal",
                "cached_selected_label_position",
                "cached_selected_gold_token_id",
                "cached_selected_competitor_id",
                "cached_replay_top1_match_count",
                "cached_alignment_kind_code",
            )
        }
        require(
            observation["cached_replay_use_cache"] is True
            and integer_fields["cached_replay_logits_to_keep"]
            == REPLAY_LOGITS_TO_KEEP
            and integer_fields["rollout_token_count"] > 0
            and integer_fields["cached_replay_token_count"] > 0
            and integer_fields["cached_replay_selected_cursor"] >= 0
            and integer_fields["cached_decision_token_count"] > 0
            and integer_fields["cached_selected_decision_ordinal"] >= 0
            and integer_fields["cached_selected_label_position"] > 0
            and integer_fields["cached_selected_gold_token_id"] >= 0
            and integer_fields["cached_selected_competitor_id"] >= 0
            and observation["cached_selected_is_termination"] is False,
            "v14_smoke_row_audit_cached_replay_incomplete",
        )
        numbers = {
            name: _finite_number(
                observation.get(name),
                description=f"v14_smoke_{name}",
            )
            for name in (
                "cached_ce",
                "cached_failed_competitor_hinge",
                "cached_exact_retention_hinge",
                "cached_selected_gold_vs_competitor_margin",
                "cached_gold_top1_fraction",
                "cached_branch_loss",
                "auxiliary_optimization_loss",
                "auxiliary_telemetry_loss",
                "selected_top_competitor_hinge_telemetry",
                "selected_correct_vs_zero_hinge_telemetry",
                "total_side_loss",
            )
        }
        require(
            all(
                numbers[name] >= 0.0
                for name in (
                    "cached_ce",
                    "cached_failed_competitor_hinge",
                    "cached_exact_retention_hinge",
                    "cached_branch_loss",
                    "auxiliary_optimization_loss",
                    "auxiliary_telemetry_loss",
                    "selected_top_competitor_hinge_telemetry",
                    "selected_correct_vs_zero_hinge_telemetry",
                    "total_side_loss",
                )
            )
            and 0.0 <= numbers["cached_gold_top1_fraction"] <= 1.0
            and numbers["auxiliary_optimization_loss"] == 0.0,
            "v14_smoke_row_audit_loss_domain_differs",
        )
        if parsed_exact:
            require(
                numbers["cached_ce"] == 0.0
                and numbers["cached_failed_competitor_hinge"] == 0.0
                and integer_fields["cached_alignment_kind_code"] == -1
                and observation["cached_competitor_is_actual_greedy"] is False
                and observation["cached_replay_top1_matches_actual"] is False
                and integer_fields["cached_replay_top1_match_count"] == 0,
                "v14_smoke_exact_row_invented_failed_repair",
            )
            expected_loss = numbers["cached_exact_retention_hinge"]
        else:
            require(
                numbers["cached_exact_retention_hinge"] == 0.0
                and integer_fields["cached_alignment_kind_code"] in (0, 1, 2)
                and observation["cached_competitor_is_actual_greedy"] is True
                and observation["cached_replay_top1_matches_actual"] is True
                and integer_fields["cached_replay_top1_match_count"]
                == integer_fields["cached_replay_token_count"]
                and integer_fields["cached_replay_token_count"]
                == integer_fields["cached_replay_selected_cursor"] + 1,
                "v14_smoke_failed_row_cached_repair_incomplete",
            )
            expected_loss = (
                numbers["cached_ce"]
                + numbers["cached_failed_competitor_hinge"]
            )
        require(
            math.isclose(
                numbers["cached_branch_loss"],
                expected_loss,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            and math.isclose(
                numbers["total_side_loss"],
                expected_loss,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ),
            "v14_smoke_row_audit_row_arithmetic_differs",
        )
        validated_rows.append(observation)

    pairs = audit.get("pair_presentations")
    require(
        isinstance(pairs, list) and len(pairs) == 1,
        "v14_smoke_row_audit_pair_count_differs",
    )
    pair = pairs[0]
    require(
        isinstance(pair, Mapping) and set(pair) == _PAIR_REQUIRED_FIELDS,
        "v14_smoke_row_audit_pair_fields_differ",
    )
    require(
        pair.get("phase") == "smoke_input"
        and pair.get("cycle") == 1
        and pair.get("adapter_optimizer_step_before_update") == 0
        and pair.get("presentation") == 1
        and pair.get("source_row_ordinal") == low
        and pair.get("donor_row_ordinal") == high
        and pair.get("source_row_sha256") == source_member.get("row_sha256")
        and pair.get("donor_row_sha256") == donor_member.get("row_sha256"),
        "v14_smoke_row_audit_pair_identity_differs",
    )
    pair_numbers = {
        name: _finite_number(
            pair.get(name),
            description=f"v14_smoke_pair_{name}",
        )
        for name in _PAIR_REQUIRED_FIELDS
        if name.startswith("pair_mean_")
        or name in {
            "reported_objective_total_loss",
            "recomputed_objective_total_loss",
        }
    }
    require(
        all(value >= 0.0 for value in pair_numbers.values())
        and pair_numbers["pair_mean_auxiliary_optimization_loss"] == 0.0,
        "v14_smoke_row_audit_pair_loss_domain_differs",
    )
    recomputed_cached = sum(
        pair_numbers[name]
        for name in (
            "pair_mean_cached_exact_retention_hinge",
            "pair_mean_cached_failed_ce",
            "pair_mean_cached_failed_competitor_hinge",
        )
    )
    recomputed_total = (
        recomputed_cached
        + pair_numbers["pair_mean_auxiliary_optimization_loss"]
    )
    require(
        all(
            math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
            for actual, expected in (
                (
                    pair_numbers["pair_mean_cached_branch_loss"],
                    recomputed_cached,
                ),
                (pair_numbers["pair_mean_total_side_loss"], recomputed_total),
                (pair_numbers["reported_objective_total_loss"], recomputed_total),
                (
                    pair_numbers["recomputed_objective_total_loss"],
                    recomputed_total,
                ),
            )
        ),
        "v14_smoke_row_audit_pair_arithmetic_differs",
    )
    pair_to_row = {
        "pair_mean_cached_branch_loss": "cached_branch_loss",
        "pair_mean_cached_exact_retention_hinge": (
            "cached_exact_retention_hinge"
        ),
        "pair_mean_cached_failed_ce": "cached_ce",
        "pair_mean_cached_failed_competitor_hinge": (
            "cached_failed_competitor_hinge"
        ),
        "pair_mean_auxiliary_optimization_loss": "auxiliary_optimization_loss",
        "pair_mean_selected_top_competitor_hinge_telemetry": (
            "selected_top_competitor_hinge_telemetry"
        ),
        "pair_mean_selected_correct_vs_zero_hinge_telemetry": (
            "selected_correct_vs_zero_hinge_telemetry"
        ),
        "pair_mean_total_side_loss": "total_side_loss",
    }
    require(
        all(
            math.isclose(
                pair_numbers[pair_name],
                sum(float(row[row_name]) for row in validated_rows) / 2.0,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            for pair_name, row_name in pair_to_row.items()
        ),
        "v14_smoke_row_audit_pair_row_means_differ",
    )
    return {
        "schema": ROW_OBJECTIVE_AUDIT_SCHEMA,
        "run_mode": ONE_PAIR_SMOKE_RUN_MODE,
        "production_eligible": False,
        "checkpoint_optimizer_step": 1,
        "completed_pair_presentations": 1,
        "pair_schedule_sha256": canonical_sha256(expected_schedule),
        "rows": 2,
        "pair_presentations": 1,
        "cached_replay_top1_parity_verified": all(
            row["parsed_boundary_exact"]
            or row["cached_replay_top1_matches_actual"]
            for row in validated_rows
        ),
    }


def _validate_pairing_manifest(pairing: Mapping[str, Any]) -> str:
    return v13._validate_pairing_manifest(pairing)


def _validate_warm_lineage(
    lineage: Mapping[str, Any],
    *,
    protocol_sha256: str,
    config_sha256: str,
    pairing_sha256: str,
    warm_contract: Mapping[str, Any],
) -> str:
    unsigned = dict(lineage)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    require(
        isinstance(receipt_sha256, str)
        and receipt_sha256 == canonical_sha256(unsigned),
        "v14_warm_lineage_self_hash_differs",
    )
    fresh = lineage.get("target_fresh_start")
    source_lock = lineage.get("source_lock")
    warm_context = warm_contract.get("context")
    warm_lock = getattr(warm_context, "lock", {})
    require(
        lineage.get("schema") == warm.RECEIPT_SCHEMA
        and lineage.get("schema_version") == 1
        and lineage.get("mode") == warm.WARM_START_MODE
        and lineage.get("source_checkpoint") == str(PINNED_WARM_START_CHECKPOINT)
        and lineage.get("source_global_step") == 4
        and lineage.get("trainer_resume_from_checkpoint") is None
        and lineage.get("target_initial_global_step") == 0,
        "v14_warm_lineage_identity_differs",
    )
    require(
        isinstance(source_lock, Mapping)
        and source_lock.get("path") == warm_contract.get("warm_start_lock")
        and source_lock.get("lock_sha256")
        == warm_contract.get("warm_start_lock_sha256")
        and lineage.get("source_state_imports") == warm.SOURCE_IMPORT_POLICY
        and lineage.get("source_artifacts")
        == warm_lock.get("artifacts")
        and lineage.get("loaded_source_artifacts") == ["delta_mem_adapter.pt"]
        and lineage.get("validated_not_imported_source_artifacts")
        == ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        and lineage.get("post_load_bit_equal") is True,
        "v14_warm_lineage_source_binding_differs",
    )
    require(
        isinstance(fresh, Mapping)
        and fresh.get("initial_global_step") == 0
        and fresh.get("optimizer_implementation") == OPTIMIZER_IMPLEMENTATION
        and fresh.get("optimizer_created_after_adapter_load") is True
        and fresh.get("optimizer_state") == "fresh"
        and fresh.get("scheduler_state") == "fresh"
        and fresh.get("trainer_state") == "fresh"
        and fresh.get("rng_state") == "fresh_from_v14_seed",
        "v14_warm_lineage_fresh_state_differs",
    )
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        lineage.get("pre_train_global_step") == 0
        and lineage.get("fresh_optimizer_created") is True
        and isinstance(optimizer_class, str)
        and optimizer_class.endswith(".AdamW")
        and lineage.get("fresh_optimizer_state_entries_before_train") == 0
        and lineage.get("fresh_scheduler_created_before_train") is False
        and lineage.get("fresh_adamw_creation_required_after_adapter_load") is True,
        "v14_warm_lineage_optimizer_evidence_differs",
    )
    source_artifacts = lineage.get("source_artifacts")
    require(
        isinstance(source_artifacts, Mapping)
        and source_artifacts.get("delta_mem_adapter.pt", {}).get("sha256")
        == PINNED_WARM_START_ADAPTER_SHA256,
        "v14_warm_lineage_adapter_hash_differs",
    )
    require(
        lineage.get("target_training_protocol_sha256") == protocol_sha256
        and lineage.get("target_delta_config_sha256") == config_sha256
        and lineage.get("target_scene_state_pairing_manifest_sha256")
        == pairing_sha256,
        "v14_warm_lineage_target_binding_differs",
    )
    require(
        warm_contract.get("warm_start_adapter_sha256")
        == PINNED_WARM_START_ADAPTER_SHA256,
        "v14_warm_contract_adapter_hash_differs",
    )
    return receipt_sha256


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved = require_v14_run_path(checkpoint, description="v14_checkpoint", ssd_root=ssd_root)
    require(resolved.is_dir(), "v14_checkpoint_missing")
    try:
        checkpoint_step = int(resolved.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise LaunchContractError("v14_checkpoint_name_differs") from exc
    require(
        resolved.name == f"checkpoint-{checkpoint_step}" and checkpoint_step in CHECKPOINT_STEPS,
        "v14_checkpoint_step_is_not_1_through_4",
    )
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / filename, description=f"v14_{filename}")
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in rng_files),
        "v14_rng_state_missing",
    )
    trainer_state = _load_object(resolved / "trainer_state.json", description="v14_trainer_state")
    protocol = _load_object(resolved / "training_protocol.json", description="v14_protocol")
    config = _load_object(resolved / "delta_mem_config.json", description="v14_config")
    pairing = _load_object(resolved / "scene_state_identity_pairing_manifest.json", description="v14_pairing")
    require(
        trainer_state.get("global_step") == checkpoint_step
        and trainer_state.get("max_steps") == TOTAL_OPTIMIZER_STEPS
        and protocol.get("max_steps") == TOTAL_OPTIMIZER_STEPS,
        "v14_checkpoint_horizon_differs",
    )
    telemetry = validate_v14_cycle_pair_telemetry(trainer_state, checkpoint_step=checkpoint_step)
    audit = _validate_v14_row_objective_audit(
        _load_object(resolved / ROW_OBJECTIVE_AUDIT_FILENAME, description="v14_row_objective_audit"),
        checkpoint_step=checkpoint_step,
        data=data,
    )
    _validate_checkpoint_protocol(protocol, data=data)
    try:
        v13.v10.v9._validate_checkpoint_config(config)
    except Exception as exc:
        raise LaunchContractError(f"v14_delta_config_failed: {exc}") from exc
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing)
    lineage_path = resolved / warm_module_lineage_filename()
    root_receipt = _validate_warm_lineage(
        _load_object(lineage_path, description="v14_lineage"),
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        warm_contract=warm,
    )
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": checkpoint_step,
        "consumed_pair_presentations": telemetry["pair_presentations"],
        "cycle_pair_telemetry": telemetry,
        "row_objective_audit": audit,
        "row_objective_audit_file_sha256": sha256_file(resolved / ROW_OBJECTIVE_AUDIT_FILENAME),
        "lineage_filename": warm_module_lineage_filename(),
        "lineage_file_sha256": sha256_file(lineage_path),
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt,
    }


def _validate_one_pair_smoke_adapter_update(
    checkpoint_adapter: Path,
    *,
    source_adapter: Path,
) -> dict[str, Any]:
    try:
        import torch

        source = torch.load(source_adapter, map_location="cpu", weights_only=True)
        candidate = torch.load(
            checkpoint_adapter,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise LaunchContractError(
            f"v14_smoke_adapter_update_load_failed: {exc}"
        ) from exc
    require(
        isinstance(source, Mapping)
        and bool(source)
        and isinstance(candidate, Mapping)
        and bool(candidate),
        "v14_smoke_adapter_state_missing",
    )
    require(
        list(candidate) == list(source),
        "v14_smoke_adapter_topology_differs",
    )
    changed_tensors = 0
    changed_elements = 0
    maximum_absolute_delta = 0.0
    for name, source_tensor in source.items():
        candidate_tensor = candidate[name]
        require(
            isinstance(name, str)
            and isinstance(source_tensor, torch.Tensor)
            and isinstance(candidate_tensor, torch.Tensor),
            "v14_smoke_adapter_entry_invalid",
        )
        require(
            source_tensor.shape == candidate_tensor.shape
            and source_tensor.dtype == candidate_tensor.dtype,
            f"v14_smoke_adapter_tensor_metadata_differs name={name}",
        )
        if source_tensor.is_floating_point() or source_tensor.is_complex():
            require(
                bool(torch.isfinite(source_tensor).all())
                and bool(torch.isfinite(candidate_tensor).all()),
                f"v14_smoke_adapter_tensor_nonfinite name={name}",
            )
        different = candidate_tensor != source_tensor
        tensor_changed_elements = int(torch.count_nonzero(different).item())
        if tensor_changed_elements == 0:
            continue
        changed_tensors += 1
        changed_elements += tensor_changed_elements
        if candidate_tensor.is_floating_point() or candidate_tensor.is_complex():
            maximum_absolute_delta = max(
                maximum_absolute_delta,
                float(
                    (candidate_tensor - source_tensor)
                    .detach()
                    .abs()
                    .max()
                    .item()
                ),
            )
    require(
        changed_tensors > 0
        and changed_elements > 0
        and math.isfinite(maximum_absolute_delta)
        and maximum_absolute_delta > 0.0,
        "v14_smoke_adapter_did_not_change",
    )
    return {
        "source_adapter_sha256": sha256_file(source_adapter),
        "checkpoint_adapter_sha256": sha256_file(checkpoint_adapter),
        "tensor_count": len(source),
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "maximum_absolute_delta": maximum_absolute_delta,
    }


def validate_one_pair_smoke_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved = require_v14_run_path(
        checkpoint,
        description="v14_smoke_checkpoint",
        ssd_root=ssd_root,
    )
    require(
        resolved.is_dir() and resolved.name == "checkpoint-1",
        "v14_smoke_checkpoint_must_be_checkpoint_1",
    )
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / filename, description=f"v14_smoke_{filename}")
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            for path in rng_files
        ),
        "v14_smoke_rng_state_missing",
    )
    trainer_state = _load_object(
        resolved / "trainer_state.json",
        description="v14_smoke_trainer_state",
    )
    protocol = _load_object(
        resolved / "training_protocol.json",
        description="v14_smoke_protocol",
    )
    config = _load_object(
        resolved / "delta_mem_config.json",
        description="v14_smoke_config",
    )
    pairing = _load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="v14_smoke_pairing",
    )
    require(
        trainer_state.get("global_step") == 1
        and trainer_state.get("max_steps") == 1
        and protocol.get("max_steps") == 1,
        "v14_smoke_checkpoint_horizon_differs",
    )
    telemetry = _validate_one_pair_smoke_telemetry(trainer_state)
    audit = _validate_one_pair_smoke_row_objective_audit(
        _load_object(
            resolved / ROW_OBJECTIVE_AUDIT_FILENAME,
            description="v14_smoke_row_objective_audit",
        ),
        data=data,
    )
    require(
        audit["cached_replay_top1_parity_verified"] is True,
        "v14_smoke_cached_replay_top1_parity_failed",
    )
    _validate_one_pair_smoke_checkpoint_protocol(protocol, data=data)
    try:
        v13.v10.v9._validate_checkpoint_config(config)
    except Exception as exc:
        raise LaunchContractError(f"v14_smoke_delta_config_failed: {exc}") from exc
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing)
    lineage_path = resolved / warm_module_lineage_filename()
    root_receipt = _validate_warm_lineage(
        _load_object(lineage_path, description="v14_smoke_lineage"),
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        warm_contract=warm,
    )
    source_checkpoint = Path(
        str(warm.get("warm_start_checkpoint", PINNED_WARM_START_CHECKPOINT))
    )
    source_adapter = source_checkpoint / "delta_mem_adapter.pt"
    _regular_file(source_adapter, description="v14_smoke_source_adapter")
    adapter_update = _validate_one_pair_smoke_adapter_update(
        resolved / "delta_mem_adapter.pt",
        source_adapter=source_adapter,
    )
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": 1,
        "consumed_pair_presentations": 1,
        "cycle_pair_telemetry": telemetry,
        "row_objective_audit": audit,
        "row_objective_audit_file_sha256": sha256_file(
            resolved / ROW_OBJECTIVE_AUDIT_FILENAME
        ),
        "lineage_filename": warm_module_lineage_filename(),
        "lineage_file_sha256": sha256_file(lineage_path),
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt,
        "adapter_update": adapter_update,
    }


def warm_module_lineage_filename() -> str:
    return warm.WARM_START_LINEAGE_FILENAME


def critical_training_code_bindings(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, Any]]:
    resolved = project_root.expanduser().resolve()
    require(resolved == PROJECT_ROOT, "v14_critical_code_requires_exact_project_root")
    return {
        relative: artifact_binding(resolved / relative, description=f"v14_critical_{relative}")
        for relative in CRITICAL_TRAINING_FILES
    }


require_git_object_id = v13.require_git_object_id
_git_head = v13._git_head
_require_git_ancestor = v13._require_git_ancestor
_resolve_git_commit = v13._resolve_git_commit
_validate_exact_artifact_binding = v13._validate_exact_artifact_binding


def critical_training_code_bindings_at_commit(
    commit: object,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    return v13._git_file_bindings_at_commit(
        commit,
        CRITICAL_TRAINING_FILES,
        project_root=project_root,
        description="v14_critical_code_at_commit",
    )


def _validate_receipt_self_hash(payload: Mapping[str, Any], *, description: str) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop("receipt_sha256", None)
    require(
        isinstance(recorded, str) and recorded == canonical_sha256(unsigned),
        f"{description}_self_hash_differs",
    )
    return recorded


def validate_launch_receipt(
    launch_receipt: Path,
    *,
    checkpoint: Path,
    baseline: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v14_run_path(checkpoint, description="v14_provenance_checkpoint", ssd_root=ssd_root)
    path = require_v14_run_path(launch_receipt, description="v14_launch_receipt", ssd_root=ssd_root)
    _regular_file(path, description="v14_launch_receipt")
    require(path.parent == v14_run_root_for(ssd_root) / "logs", "v14_launch_receipt_must_be_under_logs")
    payload = _load_object(path, description="v14_launch_receipt")
    receipt_sha256 = _validate_receipt_self_hash(payload, description="v14_launch_receipt")
    output = resolved_checkpoint.parents[1]
    expected_launch_path = (
        v14_run_root_for(ssd_root) / "logs" / f"{output.name}.launch.json"
    )
    expected_log_path = expected_launch_path.with_name(f"{output.name}.log")
    require(
        payload.get("schema") == LAUNCH_RECEIPT_SCHEMA
        and payload.get("attached_foreground_execution") is True
        and payload.get("launch_mode") == "warm_start"
        and payload.get("source_step") == 0
        and payload.get("source_checkpoint_step") == 4
        and payload.get("target_step") == 4
        and payload.get("resume_checkpoint") is None,
        "v14_launch_receipt_mode_or_horizon_differs",
    )
    require(
        payload.get("trainer_output") == str(output)
        and payload.get("checkpoints")
        == {f"checkpoint-{step}": str(output / f"trainer/checkpoint-{step}") for step in CHECKPOINT_STEPS}
        and str(resolved_checkpoint) in payload["checkpoints"].values()
        and payload.get("log_file") == str(expected_log_path)
        and path == expected_launch_path,
        "v14_launch_receipt_checkpoint_binding_differs",
    )
    require(
        payload.get("objective") == OBJECTIVE_VERSION
        and payload.get("objective_schema_version") == OBJECTIVE_SCHEMA_VERSION
        and payload.get("gradient_accumulation_steps")
        == GRADIENT_ACCUMULATION_STEPS
        and payload.get("max_grad_norm") == MAX_GRAD_NORM
        and payload.get("max_steps") == TOTAL_OPTIMIZER_STEPS
        and payload.get("learning_rate") == LEARNING_RATE
        and payload.get("optim") == OPTIMIZER_IMPLEMENTATION
        and payload.get("weight_decay") == WEIGHT_DECAY
        and payload.get("lr_scheduler_type") == "constant"
        and payload.get("warmup_steps") == WARMUP_STEPS
        and payload.get("warmup_ratio") == WARMUP_RATIO
        and payload.get("logging_steps") == LOGGING_STEPS
        and payload.get("save_steps") == SAVE_STEPS
        and payload.get("save_total_limit") == len(CHECKPOINT_STEPS)
        and payload.get("seed") == SEED
        and payload.get("data_seed") == DATA_SEED
        and payload.get("four_cycle_pairs") == [list(pair) for pair in FOUR_CYCLE_PAIRS]
        and payload.get("four_cycle_pairs_sha256") == FOUR_CYCLE_PAIRS_SHA256,
        "v14_launch_receipt_objective_or_cycle_differs",
    )
    warm_contract = validate_warm_start_contract(ssd_root=ssd_root)
    require(
        payload.get("warm_start_checkpoint") == str(PINNED_WARM_START_CHECKPOINT)
        and payload.get("warm_start_adapter_sha256") == PINNED_WARM_START_ADAPTER_SHA256
        and payload.get("warm_start_mode") == warm.WARM_START_MODE
        and payload.get("warm_start_lock")
        == warm_contract["warm_start_lock"]
        and payload.get("warm_start_lock_sha256")
        == warm_contract["warm_start_lock_sha256"]
        and payload.get("v10_diagnostic_baseline") == baseline
        and payload.get("base_model_identity") == base_model_identity,
        "v14_launch_receipt_source_or_model_identity_differs",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "value14_gate_only",
        "v14_launch_receipt_authorization_differs",
    )
    source_commit = _resolve_git_commit(
        payload.get("git_commit"),
        project_root=project_root,
        description="v14_launch_receipt_git_commit",
    )
    _require_git_ancestor(
        source_commit,
        _git_head(project_root),
        project_root=project_root,
        description="v14_launch_receipt_lineage",
    )
    require(
        payload.get("tracked_worktree_clean") is True,
        "v14_launch_receipt_git_identity_differs",
    )
    expected_code = critical_training_code_bindings_at_commit(
        source_commit,
        project_root=project_root,
    )
    require(
        payload.get("critical_files") == expected_code,
        "v14_launch_receipt_critical_file_hashes_differ",
    )
    return {
        "artifact": artifact_binding(path, description="v14_launch_receipt"),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
    }


def validate_one_pair_smoke_launch_receipt(
    launch_receipt: Path,
    *,
    checkpoint: Path,
    baseline: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v14_run_path(
        checkpoint,
        description="v14_smoke_provenance_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v14_run_path(
        launch_receipt,
        description="v14_smoke_launch_receipt",
        ssd_root=ssd_root,
    )
    _regular_file(path, description="v14_smoke_launch_receipt")
    require(
        path.parent == v14_run_root_for(ssd_root) / "logs",
        "v14_smoke_launch_receipt_must_be_under_logs",
    )
    payload = _load_object(path, description="v14_smoke_launch_receipt")
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v14_smoke_launch_receipt",
    )
    output = resolved_checkpoint.parents[1]
    run_name = payload.get("run_name")
    safe_run_name_characters = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    )
    require(
        isinstance(run_name, str)
        and bool(run_name)
        and run_name[0]
        in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        and all(character in safe_run_name_characters for character in run_name)
        and output.name == f"scene_memory_v14_smoke_{run_name}_step1",
        "v14_smoke_launch_run_name_differs",
    )
    expected_checkpoint = output / "trainer/checkpoint-1"
    expected_launch_path = (
        v14_run_root_for(ssd_root) / "logs" / f"{output.name}.launch.json"
    )
    expected_log_path = expected_launch_path.with_name(f"{output.name}.log")
    require(
        payload.get("schema") == ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA
        and payload.get("attached_foreground_execution") is True
        and payload.get("launch_mode") == "warm_start_smoke"
        and payload.get("run_mode") == ONE_PAIR_SMOKE_RUN_MODE
        and payload.get("production_eligible") is False
        and payload.get("source_step") == 0
        and payload.get("source_checkpoint_step") == 4
        and payload.get("target_step") == 1
        and payload.get("resume_checkpoint") is None,
        "v14_smoke_launch_receipt_mode_or_horizon_differs",
    )
    require(
        resolved_checkpoint == expected_checkpoint
        and payload.get("trainer_output") == str(output)
        and payload.get("checkpoints")
        == {"checkpoint-1": str(expected_checkpoint)}
        and payload.get("log_file") == str(expected_log_path)
        and path == expected_launch_path,
        "v14_smoke_launch_receipt_checkpoint_binding_differs",
    )
    require(
        payload.get("objective") == OBJECTIVE_VERSION
        and payload.get("objective_schema_version") == OBJECTIVE_SCHEMA_VERSION
        and payload.get("gradient_accumulation_steps") == 1
        and payload.get("max_grad_norm") == MAX_GRAD_NORM
        and payload.get("max_steps") == 1
        and payload.get("learning_rate") == LEARNING_RATE
        and payload.get("optim") == OPTIMIZER_IMPLEMENTATION
        and payload.get("weight_decay") == WEIGHT_DECAY
        and payload.get("lr_scheduler_type") == "constant"
        and payload.get("warmup_steps") == WARMUP_STEPS
        and payload.get("warmup_ratio") == WARMUP_RATIO
        and payload.get("logging_steps") == LOGGING_STEPS
        and payload.get("save_steps") == SAVE_STEPS
        and payload.get("save_total_limit") == 1
        and payload.get("seed") == SEED
        and payload.get("data_seed") == DATA_SEED
        and payload.get("total_pair_presentations") == 1
        and payload.get("scheduled_pairs") == [list(ONE_PAIR_SMOKE_PAIR)]
        and payload.get("four_cycle_pairs")
        == [list(pair) for pair in FOUR_CYCLE_PAIRS]
        and payload.get("four_cycle_pairs_sha256") == FOUR_CYCLE_PAIRS_SHA256,
        "v14_smoke_launch_receipt_objective_or_schedule_differs",
    )
    warm_contract = validate_warm_start_contract(ssd_root=ssd_root)
    require(
        payload.get("warm_start_checkpoint")
        == str(PINNED_WARM_START_CHECKPOINT)
        and payload.get("warm_start_adapter_sha256")
        == PINNED_WARM_START_ADAPTER_SHA256
        and payload.get("warm_start_mode") == warm.WARM_START_MODE
        and payload.get("warm_start_lock")
        == warm_contract["warm_start_lock"]
        and payload.get("warm_start_lock_sha256")
        == warm_contract["warm_start_lock_sha256"]
        and payload.get("v10_diagnostic_baseline") == baseline
        and payload.get("base_model_identity") == base_model_identity,
        "v14_smoke_launch_receipt_source_or_model_identity_differs",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v14_smoke_launch_receipt_authorization_differs",
    )
    source_commit = _resolve_git_commit(
        payload.get("git_commit"),
        project_root=project_root,
        description="v14_smoke_launch_receipt_git_commit",
    )
    _require_git_ancestor(
        source_commit,
        _git_head(project_root),
        project_root=project_root,
        description="v14_smoke_launch_receipt_lineage",
    )
    require(
        payload.get("tracked_worktree_clean") is True,
        "v14_smoke_launch_receipt_git_identity_differs",
    )
    require(
        payload.get("critical_files")
        == critical_training_code_bindings_at_commit(
            source_commit,
            project_root=project_root,
        ),
        "v14_smoke_launch_receipt_critical_file_hashes_differ",
    )
    return {
        "artifact": artifact_binding(
            path,
            description="v14_smoke_launch_receipt",
        ),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
    }


def _expected_completion_receipt_path(
    launch: Mapping[str, Any],
    *,
    ssd_root: Path,
) -> Path:
    launch_artifact = launch.get("artifact")
    require(
        isinstance(launch_artifact, Mapping),
        "v14_completion_launch_artifact_missing",
    )
    launch_path = require_v14_run_path(
        Path(str(launch_artifact.get("path", ""))),
        description="v14_completion_launch_receipt_path",
        ssd_root=ssd_root,
    )
    require(
        launch_path.parent == v14_run_root_for(ssd_root) / "logs"
        and launch_path.name.endswith(".launch.json"),
        "v14_completion_launch_receipt_path_differs",
    )
    return launch_path.with_name(
        launch_path.name.removesuffix(".launch.json") + ".completion.json"
    )


def validate_completion_receipt(
    completion_receipt: Path,
    *,
    checkpoint: Path,
    checkpoint_contract: Mapping[str, Any],
    launch: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v14_run_path(
        checkpoint,
        description="v14_completion_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v14_run_path(
        completion_receipt,
        description="v14_completion_receipt",
        ssd_root=ssd_root,
    )
    require(
        path == _expected_completion_receipt_path(launch, ssd_root=ssd_root),
        "v14_completion_receipt_path_differs",
    )
    _regular_file(path, description="v14_completion_receipt")
    payload = _load_object(path, description="v14_completion_receipt")
    receipt_sha256 = _validate_receipt_self_hash(payload, description="v14_completion_receipt")
    require(
        payload.get("schema") == COMPLETION_RECEIPT_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("optimizer_step") == 4
        and payload.get("consumed_pair_presentations") == 28,
        "v14_completion_receipt_horizon_differs",
    )
    launch_artifact = launch.get("artifact")
    require(
        isinstance(launch_artifact, Mapping),
        "v14_completion_launch_artifact_missing",
    )
    _validate_exact_artifact_binding(
        payload.get("launch_receipt", {}),
        expected_path=Path(str(launch_artifact.get("path", ""))),
        description="v14_completion_launch_receipt",
    )
    require(
        payload.get("launch_receipt_sha256") == launch.get("receipt_sha256"),
        "v14_completion_launch_binding_differs",
    )
    checkpoints = payload.get("checkpoints")
    require(
        isinstance(checkpoints, Mapping)
        and set(checkpoints) == {f"checkpoint-{step}" for step in CHECKPOINT_STEPS},
        "v14_completion_checkpoints_differ",
    )
    trainer_root = resolved_checkpoint.parent
    for step in CHECKPOINT_STEPS:
        name = f"checkpoint-{step}"
        bound_checkpoint = trainer_root / name
        entry = checkpoints.get(name)
        require(
            isinstance(entry, Mapping)
            and entry.get("path") == str(bound_checkpoint)
            and entry.get("optimizer_step") == step
            and entry.get("consumed_pair_presentations")
            == step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
            f"v14_completion_{name}_identity_differs",
        )
        expected_checkpoint_artifacts = {
            filename: artifact_binding(
                bound_checkpoint / filename,
                description=f"v14_completion_{name}_{filename}",
            )
            for filename in REQUIRED_CHECKPOINT_ARTIFACTS
        }
        expected_rng = {
            rng_path.name: artifact_binding(
                rng_path,
                description=f"v14_completion_{name}_{rng_path.name}",
            )
            for rng_path in sorted(bound_checkpoint.glob("rng_state*.pth"))
        }
        require(
            entry.get("checkpoint_artifacts") == expected_checkpoint_artifacts
            and entry.get("rng_state_artifacts") == expected_rng,
            f"v14_completion_{name}_artifacts_differ",
        )
        require(
            entry.get("cycle_pair_telemetry")
            == validate_v14_cycle_pair_telemetry(
                _load_object(
                    bound_checkpoint / "trainer_state.json",
                    description=f"v14_completion_{name}_trainer_state",
                ),
                checkpoint_step=step,
            )
            and entry.get("row_objective_audit_file_sha256")
            == sha256_file(bound_checkpoint / ROW_OBJECTIVE_AUDIT_FILENAME),
            f"v14_completion_{name}_telemetry_differs",
        )
    name = resolved_checkpoint.name
    entry = checkpoints[name]
    require(
        entry.get("path") == str(resolved_checkpoint)
        and entry.get("optimizer_step") == checkpoint_contract.get("checkpoint_step")
        and entry.get("consumed_pair_presentations") == checkpoint_contract.get("consumed_pair_presentations")
        and entry.get("cycle_pair_telemetry") == checkpoint_contract.get("cycle_pair_telemetry")
        and entry.get("row_objective_audit_file_sha256") == checkpoint_contract.get("row_objective_audit_file_sha256"),
        "v14_completion_checkpoint_binding_differs",
    )
    summary_binding = payload.get("training_summary")
    require(
        isinstance(summary_binding, Mapping),
        "v14_completion_training_summary_missing",
    )
    summary_path = resolved_checkpoint.parents[1] / "training_summary.json"
    summary_artifact = _validate_exact_artifact_binding(
        summary_binding,
        expected_path=summary_path,
        description="v14_completion_training_summary",
    )
    summary = _load_object(summary_path, description="v14_completion_training_summary")
    require(
        summary.get("memory_objective_version") == OBJECTIVE_VERSION
        and summary.get("warm_start_mode") == warm.WARM_START_MODE
        and summary.get("training_protocol_sha256")
        == checkpoint_contract.get("training_protocol_sha256"),
        "v14_completion_training_summary_identity_differs",
    )
    log_binding = payload.get("log")
    require(isinstance(log_binding, Mapping), "v14_completion_log_missing")
    launch_payload = launch.get("payload")
    require(
        isinstance(launch_payload, Mapping),
        "v14_completion_launch_payload_missing",
    )
    log_path = Path(str(launch_payload.get("log_file", "")))
    require(
        log_path.parent == v14_run_root_for(ssd_root) / "logs",
        "v14_completion_log_path_differs",
    )
    log_artifact = _validate_exact_artifact_binding(
        log_binding,
        expected_path=log_path,
        description="v14_completion_log",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "value14_gate_only",
        "v14_completion_authorization_differs",
    )
    return {
        "artifact": artifact_binding(path, description="v14_completion_receipt"),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
        "training_summary": summary_artifact,
        "log": log_artifact,
    }


def _validate_one_pair_smoke_cuda_memory(
    value: Any,
) -> dict[str, Any]:
    require(
        isinstance(value, Mapping) and value.get("device") == "cuda:0",
        "v14_smoke_cuda_memory_missing",
    )
    fields = (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "post_train_allocated_bytes",
        "post_train_reserved_bytes",
    )
    memory = {
        name: _finite_integral_metric(
            value.get(name),
            description=f"v14_smoke_cuda_{name}",
        )
        for name in fields
    }
    require(
        all(amount >= 0 for amount in memory.values())
        and memory["peak_allocated_bytes"] > 0
        and memory["peak_reserved_bytes"] > 0
        and memory["peak_allocated_bytes"]
        >= max(
            memory["baseline_allocated_bytes"],
            memory["post_train_allocated_bytes"],
        )
        and memory["peak_reserved_bytes"]
        >= max(
            memory["baseline_reserved_bytes"],
            memory["post_train_reserved_bytes"],
        ),
        "v14_smoke_cuda_memory_bounds_differ",
    )
    return {"device": "cuda:0", **memory}


def validate_one_pair_smoke_completion_receipt(
    completion_receipt: Path,
    *,
    checkpoint: Path,
    checkpoint_contract: Mapping[str, Any],
    launch: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v14_run_path(
        checkpoint,
        description="v14_smoke_completion_checkpoint",
        ssd_root=ssd_root,
    )
    require(
        resolved_checkpoint.name == "checkpoint-1",
        "v14_smoke_completion_checkpoint_differs",
    )
    path = require_v14_run_path(
        completion_receipt,
        description="v14_smoke_completion_receipt",
        ssd_root=ssd_root,
    )
    require(
        path == _expected_completion_receipt_path(launch, ssd_root=ssd_root),
        "v14_smoke_completion_receipt_path_differs",
    )
    _regular_file(path, description="v14_smoke_completion_receipt")
    payload = _load_object(path, description="v14_smoke_completion_receipt")
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v14_smoke_completion_receipt",
    )
    require(
        payload.get("schema") == ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("optimizer_step") == 1
        and payload.get("consumed_pair_presentations") == 1
        and payload.get("production_eligible") is False,
        "v14_smoke_completion_receipt_horizon_differs",
    )
    launch_artifact = launch.get("artifact")
    require(
        isinstance(launch_artifact, Mapping),
        "v14_smoke_completion_launch_artifact_missing",
    )
    _validate_exact_artifact_binding(
        payload.get("launch_receipt", {}),
        expected_path=Path(str(launch_artifact.get("path", ""))),
        description="v14_smoke_completion_launch_receipt",
    )
    require(
        payload.get("launch_receipt_sha256") == launch.get("receipt_sha256"),
        "v14_smoke_completion_launch_binding_differs",
    )
    checkpoints = payload.get("checkpoints")
    require(
        isinstance(checkpoints, Mapping) and set(checkpoints) == {"checkpoint-1"},
        "v14_smoke_completion_checkpoints_differ",
    )
    entry = checkpoints["checkpoint-1"]
    require(
        isinstance(entry, Mapping)
        and entry.get("path") == str(resolved_checkpoint)
        and entry.get("optimizer_step") == 1
        and entry.get("consumed_pair_presentations") == 1,
        "v14_smoke_completion_checkpoint_identity_differs",
    )
    expected_checkpoint_artifacts = {
        filename: artifact_binding(
            resolved_checkpoint / filename,
            description=f"v14_smoke_completion_checkpoint_1_{filename}",
        )
        for filename in REQUIRED_CHECKPOINT_ARTIFACTS
    }
    expected_rng = {
        rng_path.name: artifact_binding(
            rng_path,
            description=f"v14_smoke_completion_checkpoint_1_{rng_path.name}",
        )
        for rng_path in sorted(resolved_checkpoint.glob("rng_state*.pth"))
    }
    require(
        bool(expected_rng)
        and entry.get("checkpoint_artifacts") == expected_checkpoint_artifacts
        and entry.get("rng_state_artifacts") == expected_rng,
        "v14_smoke_completion_checkpoint_artifacts_differ",
    )
    telemetry = _validate_one_pair_smoke_telemetry(
        _load_object(
            resolved_checkpoint / "trainer_state.json",
            description="v14_smoke_completion_trainer_state",
        )
    )
    require(
        entry.get("cycle_pair_telemetry") == telemetry
        and entry.get("row_objective_audit_file_sha256")
        == sha256_file(resolved_checkpoint / ROW_OBJECTIVE_AUDIT_FILENAME),
        "v14_smoke_completion_checkpoint_telemetry_differs",
    )
    adapter_update = checkpoint_contract.get("adapter_update")
    changed_tensor_count = (
        adapter_update.get("changed_tensor_count")
        if isinstance(adapter_update, Mapping)
        else None
    )
    changed_element_count = (
        adapter_update.get("changed_element_count")
        if isinstance(adapter_update, Mapping)
        else None
    )
    require(
        checkpoint_contract.get("checkpoint") == str(resolved_checkpoint)
        and checkpoint_contract.get("checkpoint_step") == 1
        and checkpoint_contract.get("consumed_pair_presentations") == 1
        and checkpoint_contract.get("cycle_pair_telemetry") == telemetry
        and checkpoint_contract.get("row_objective_audit_file_sha256")
        == entry.get("row_objective_audit_file_sha256")
        and isinstance(adapter_update, Mapping)
        and isinstance(changed_tensor_count, int)
        and not isinstance(changed_tensor_count, bool)
        and changed_tensor_count > 0
        and isinstance(changed_element_count, int)
        and not isinstance(changed_element_count, bool)
        and changed_element_count > 0,
        "v14_smoke_completion_checkpoint_contract_differs",
    )
    summary_binding = payload.get("training_summary")
    require(
        isinstance(summary_binding, Mapping),
        "v14_smoke_completion_training_summary_missing",
    )
    summary_path = resolved_checkpoint.parents[1] / "training_summary.json"
    summary_artifact = _validate_exact_artifact_binding(
        summary_binding,
        expected_path=summary_path,
        description="v14_smoke_completion_training_summary",
    )
    summary = _load_object(
        summary_path,
        description="v14_smoke_completion_training_summary",
    )
    summary_schedule = summary.get("train_schedule")
    require(
        summary.get("memory_objective_version") == OBJECTIVE_VERSION
        and summary.get("warm_start_mode") == warm.WARM_START_MODE
        and summary.get("warm_start_from_checkpoint")
        == str(PINNED_WARM_START_CHECKPOINT)
        and summary.get("resume_from_checkpoint") is None
        and summary.get("training_protocol_sha256")
        == checkpoint_contract.get("training_protocol_sha256")
        and summary.get("train_sampler_mode") == ONE_PAIR_SMOKE_SAMPLER_MODE
        and summary.get("training_mode") == "episode"
        and summary.get("save_steps") == 1
        and summary.get("save_total_limit") == 1
        and isinstance(summary_schedule, Mapping)
        and summary_schedule.get("schedule_selection_mode")
        == ONE_PAIR_SMOKE_SCHEDULE_SELECTION_MODE
        and summary_schedule.get("active_ordered_pairs_sha256")
        == canonical_sha256([list(ONE_PAIR_SMOKE_PAIR)])
        and summary_schedule.get("total_steps") == 1,
        "v14_smoke_completion_training_summary_identity_differs",
    )
    cuda_memory = _validate_one_pair_smoke_cuda_memory(
        summary.get("cuda_memory")
    )
    log_binding = payload.get("log")
    launch_payload = launch.get("payload")
    require(
        isinstance(log_binding, Mapping)
        and isinstance(launch_payload, Mapping)
        and launch_payload.get("schema")
        == ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA
        and launch_payload.get("run_mode") == ONE_PAIR_SMOKE_RUN_MODE
        and launch_payload.get("production_eligible") is False,
        "v14_smoke_completion_log_or_launch_payload_missing",
    )
    log_path = Path(str(launch_payload.get("log_file", "")))
    require(
        log_path.parent == v14_run_root_for(ssd_root) / "logs",
        "v14_smoke_completion_log_path_differs",
    )
    log_artifact = _validate_exact_artifact_binding(
        log_binding,
        expected_path=log_path,
        description="v14_smoke_completion_log",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v14_smoke_completion_authorization_differs",
    )
    return {
        "artifact": artifact_binding(
            path,
            description="v14_smoke_completion_receipt",
        ),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
        "training_summary": summary_artifact,
        "log": log_artifact,
        "cuda_memory": cuda_memory,
        "adapter_update": dict(adapter_update),
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
) -> dict[str, Any]:
    launch = validate_launch_receipt(
        launch_receipt,
        checkpoint=checkpoint,
        baseline=baseline,
        base_model_identity=base_model_identity,
        ssd_root=ssd_root,
    )
    completion = validate_completion_receipt(
        completion_receipt,
        checkpoint=checkpoint,
        checkpoint_contract=checkpoint_contract,
        launch=launch,
        ssd_root=ssd_root,
    )
    return {"launch": launch, "completion": completion}


def validate_resume_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise LaunchContractError("V14 forbids resume; all cycles run in one fresh launch")


def validate_launch_contract(
    *,
    target_step: int,
    resume_checkpoint: Path | None = None,
    gate_receipt: Path | None = None,
    smoke: bool = False,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    warm_start_checkpoint: Path = PINNED_WARM_START_CHECKPOINT,
    base_model_path: Path = PINNED_BASE_MODEL,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    expected_target_step = 1 if smoke else TOTAL_OPTIMIZER_STEPS
    require(
        target_step == expected_target_step,
        "v14_smoke_target_step_must_be_one"
        if smoke
        else "v14_target_step_must_be_four",
    )
    require(resume_checkpoint is None, "v14_resume_is_forbidden")
    require(gate_receipt is None, "v14_gate_receipt_cannot_authorize_training")
    data = validate_data_contract(data_root=data_root, source_lock_path=source_lock_path, ssd_root=ssd_root)
    warm_contract = validate_warm_start_contract(
        checkpoint=warm_start_checkpoint,
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    baseline = validate_v10_diagnostic_baseline(ssd_root=ssd_root)
    base_model_identity = validate_base_model_contract(
        base_model=base_model_path,
        baseline=baseline,
        ssd_root=ssd_root,
    )
    first = data["entries"][0]
    checkpoint_steps = (
        ONE_PAIR_SMOKE_CHECKPOINT_STEPS if smoke else CHECKPOINT_STEPS
    )
    presentation_checkpoints = (
        ONE_PAIR_SMOKE_PRESENTATION_CHECKPOINTS
        if smoke
        else PRESENTATION_CHECKPOINTS
    )
    scheduled_pairs = (ONE_PAIR_SMOKE_PAIR,) if smoke else FOUR_CYCLE_PAIRS
    gradient_accumulation_steps = 1 if smoke else GRADIENT_ACCUMULATION_STEPS
    return {
        **{key: value for key, value in data.items() if key != "entries"},
        **{key: value for key, value in warm_contract.items() if key != "context"},
        "launch_mode": "warm_start_smoke" if smoke else "warm_start",
        "run_mode": ONE_PAIR_SMOKE_RUN_MODE if smoke else "production_four_canonical_seven_pair_cycles_v1",
        "production_eligible": not smoke,
        "source_step": 0,
        "source_checkpoint_step": 4,
        "target_step": expected_target_step,
        "resume_checkpoint": None,
        "resume_schedule_cursor": 0,
        "next_pair_low_ordinal": first["canonical_pair_ordinals"][0],
        "next_pair_high_ordinal": first["canonical_pair_ordinals"][1],
        "next_schedule_entry_sha256": first["entry_sha256"],
        "total_pair_presentations": 1 if smoke else TOTAL_PAIR_PRESENTATIONS,
        "total_optimizer_steps": expected_target_step,
        "checkpoint_steps": list(checkpoint_steps),
        "presentation_checkpoint_steps": list(presentation_checkpoints),
        "scheduled_pairs": [list(pair) for pair in scheduled_pairs],
        "train_sampler_mode": (
            ONE_PAIR_SMOKE_SAMPLER_MODE if smoke else FIXED_SAMPLER_MODE
        ),
        "objective_version": OBJECTIVE_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "pairing_objective_version": PAIRING_OBJECTIVE_VERSION,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": expected_target_step,
        "learning_rate": LEARNING_RATE,
        "save_steps": 1,
        "save_total_limit": 1 if smoke else len(CHECKPOINT_STEPS),
        "four_cycle_pairs": [list(pair) for pair in FOUR_CYCLE_PAIRS],
        "four_cycle_pairs_sha256": FOUR_CYCLE_PAIRS_SHA256,
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
        result["warm_start_adapter_sha256"],
        result["launch_mode"],
        result["source_step"],
        result["target_step"],
        result["resume_schedule_cursor"],
        result["next_pair_low_ordinal"],
        result["next_pair_high_ordinal"],
        result["next_schedule_entry_sha256"],
        result["save_steps"],
        result["gradient_accumulation_steps"],
        result["max_steps"],
        result["save_total_limit"],
        result["total_pair_presentations"],
        result["run_mode"],
        result["production_eligible"],
        result["v10_diagnostic_baseline"]["summary"]["file_sha256"],
    )
    rendered = tuple(str(value) for value in fields)
    require(all("\t" not in value and "\n" not in value for value in rendered), "v14_launch_contract_tsv_control_character")
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
    parser.add_argument("--warm-start-checkpoint", type=Path, default=PINNED_WARM_START_CHECKPOINT)
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
            warm_start_checkpoint=args.warm_start_checkpoint,
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
