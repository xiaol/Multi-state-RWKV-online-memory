#!/usr/bin/env python3
"""Fail-closed launch contract for Scene Memory V10 cycle training.

V10 reuses the frozen V9 Value14 pair presentation schedule byte-for-byte, but
changes the optimization unit.  Seven consecutive pair presentations (one
complete pass over all canonical pairs) are accumulated before each optimizer
update.  Optimizer checkpoints 1, 2, 3, and 4 therefore bind presentation
cursors 7, 14, 21, and 28 respectively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v9_launch_contract as v9,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v9_data import (
    CANONICAL_VALUE14_PAIRS,
    CURRICULUM_BINDING_SCHEMA,
    TOTAL_PAIR_STEPS,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v10_warm_start import (
    ABLATION_LINEAGE_FILENAME,
    CONTINUATION_LINEAGE_FILENAME,
    DEFAULT_LOCK_PATH as WARM_START_LOCK,
    RECEIPT_SCHEMA as WARM_START_RECEIPT_SCHEMA,
    SOURCE_IMPORT_POLICY,
    WARM_START_LINEAGE_FILENAME,
    WARM_START_MODE,
    load_v10_warm_start_lock,
    prepare_v10_v8_checkpoint56_warm_start,
)


SSD_ROOT = v9.SSD_ROOT
DATA_ROOT = v9.DATA_ROOT
SOURCE_LOCK = v9.SOURCE_LOCK
PINNED_BASE_MODEL = SSD_ROOT / "models/gemma/gemma-4-E4B-it"
V10_RUN_ROOT = (
    SSD_ROOT
    / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v10"
)
V10_GATES_ROOT = V10_RUN_ROOT / "gates"
PINNED_WARM_START_CHECKPOINT = (
    SSD_ROOT
    / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v8"
    / "scene_memory_v8_production_v8_value56_20260729_1931_step56"
    / "trainer/checkpoint-56"
)

SOURCE_LOCK_FILE_SHA256 = (
    "b2b6945f1d400480e65c719a4e6fdc268114f80bf7e36e6dc6ff4b50a5628f39"
)
WARM_START_LOCK_FILE_SHA256 = (
    "806c8cee7ab22dde2ec14dd2bb5d282b4d037f6ea25cc6db1e009ff3a655bd76"
)
PINNED_DATA_ARTIFACTS = {
    "bundle_manifest": {
        "path": DATA_ROOT / "manifest.json",
        "sha256": "0a32cd2148111f259c0d6083bb3a52391353ea45a25e80d00ded4bd126e29e61",
    },
    "pair_schedule": {
        "path": DATA_ROOT / "pair_schedule.jsonl",
        "sha256": "55fd3804709aaf0b949b7b56142fc1363f9a9b86f970b6d11db5df2bb9ddfe5d",
    },
    "pair_schedule_manifest": {
        "path": DATA_ROOT / "pair_schedule_manifest.json",
        "sha256": "dabb56fedbfb2343210684b98db6249ecde261c227d209685e13d00284e7b494",
    },
    "source_manifest": {
        "path": DATA_ROOT / "source_manifest.json",
        "sha256": "9610e68086da515ac9e689a00231f40142f8468cc67b211301f24f08e4823285",
    },
}
_PINNED_HISTORICAL_TRAIN32_ROOT = (
    SSD_ROOT
    / "delta_mem_data/scene_failure_state"
    / "scene_memory_v7_fixed_hard32_aligned_train32_v1"
)
PINNED_HISTORICAL_TRAIN32_ARTIFACTS = {
    "pair_manifest": {
        "path": _PINNED_HISTORICAL_TRAIN32_ROOT / "train32_pair_manifest.json",
        "bytes": 70558,
        "sha256": "13555da56823d9597bf061d51ec6575db25cde49c044cab378f2050373fd78b6",
    },
    "source_manifest": {
        "path": _PINNED_HISTORICAL_TRAIN32_ROOT / "train32_source_manifest.json",
        "bytes": 1899,
        "sha256": "57626d3629e055a5f7900ed2a8526d357890d63730aa55d56ee72bd56a05017b",
    },
    "train32": {
        "path": _PINNED_HISTORICAL_TRAIN32_ROOT / "train32.jsonl",
        "bytes": 60598,
        "sha256": "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9",
    },
    "train32_rows": {
        "path": _PINNED_HISTORICAL_TRAIN32_ROOT / "train32_rows.jsonl",
        "bytes": 104770,
        "sha256": "af80b1938196319e6595a0e6d0e2f2c9a6009963a82fea520d608e060a4fe957",
    },
}

OBJECTIVE_VERSION = "scene_state_generation_ce_symmetric_cycle_retention_v4"
PAIRING_OBJECTIVE_VERSION = v9.PAIRING_OBJECTIVE_VERSION
OBJECTIVE_SCHEMA_VERSION = 13
FIXED_SAMPLER_MODE = "explicit_ordered_v10_canonical_seven_pair_cycle_v1"

PAIR_PHYSICAL_BATCH_SIZE = 1
PAIR_LOGICAL_BATCH_SIZE = 2
PAIR_DIRECTIONAL_EXPOSURES = 2
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 7
TOTAL_PAIR_PRESENTATIONS = TOTAL_PAIR_STEPS
TOTAL_OPTIMIZER_STEPS = TOTAL_PAIR_PRESENTATIONS // PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
CHECKPOINT_STEPS = (1, 2, 3, 4)
PRESENTATION_CHECKPOINTS = (7, 14, 21, 28)
GRADIENT_ACCUMULATION_STEPS = PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
SAVE_STEPS = 1
LEARNING_RATE = 2e-4
WARMUP_STEPS = 0
WARMUP_RATIO = 0.0
PREFIX_CORRECTION_WEIGHT = 0.5
GENERATED_MAX_CORRECTION_EVENTS = 4
GENERATED_ROLLOUT_EXTRA_TOKENS = 4
GENERATED_ROLLOUT_MAX_TOKENS = 24
MAX_LENGTH = 256
MAX_WRITE_LENGTH = 2048
TEACHER_MAX_LENGTH = MAX_LENGTH + MAX_WRITE_LENGTH
GENERATED_PREFIX_MODE = (
    "levenshtein_raw_generated_prefix_per_event_mean_gold_ce_safe_wrong_"
    "unlikelihood_v4"
)

OBJECTIVE_FORMULA = (
    "symmetric_pair_mean(weighted_full_gold_ce(schema=2,decision=4,termination=1) "
    "+ first_error_top1_hinge(0.2) + "
    "all_target_top1_retention_hinge(0.2) + "
    "selected_top_competitor_hinge(0.2) + "
    "selected_correct_vs_detached_zero_nll_hinge(0.2) + 0.5 * "
    "generated_prefix_per_event_mean(aligned_gold_ce + safe_wrong_unlikelihood)); "
    "selected_full_vocab_ce=telemetry_only"
)
BACKWARD_MODE = (
    "sequential_pair_zero_probe_full_gold_first_error_all_target_retention_"
    "then_per_event_mean_aligned_replay_v5"
)
CYCLE_RETENTION_MODE = (
    "teacher_forced_all_target_top1_margin_detached_competitor_v1"
)
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed"

CONTINUATION_LINEAGE_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_ARTIFACTS = v9.REQUIRED_CHECKPOINT_ARTIFACTS
_FORBIDDEN_PATH_SUBSTRINGS = ("hard32", "eval", "validation", "holdout")
_V10_RUN_RELATIVE = V10_RUN_ROOT.relative_to(SSD_ROOT)
_V10_GATES_RELATIVE = V10_GATES_ROOT.relative_to(SSD_ROOT)


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


canonical_sha256 = v9.canonical_sha256
sha256_file = v9.sha256_file
require_sha256 = v9.require_sha256


def _require_no_symlink_components(path: Path, *, description: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(
            not current.is_symlink(),
            f"{description}_symlink_component_forbidden path={current}",
        )


def _lexically_guard_path(
    path: Path | str,
    *,
    description: str,
    protected_exact_allowlist: Sequence[Path | str] = (),
) -> Path:
    requested = Path(path)
    require(
        ".." not in requested.parts,
        f"{description}_parent_alias_forbidden path={requested}",
    )
    expanded = requested.expanduser()
    require(expanded.is_absolute(), f"{description}_must_be_absolute path={expanded}")
    allowed = {str(Path(item).expanduser()) for item in protected_exact_allowlist}
    protected = any(
        marker in part.lower()
        for part in expanded.parts
        for marker in _FORBIDDEN_PATH_SUBSTRINGS
    )
    require(
        not protected or str(expanded) in allowed,
        f"{description}_forbidden_hard32_or_evaluation_path path={expanded}",
    )
    _require_no_symlink_components(expanded, description=description)
    return expanded


def require_exact_path(
    path: Path | str,
    expected: Path | str,
    *,
    description: str,
    protected_exact_allowlist: Sequence[Path | str] = (),
) -> Path:
    guarded = _lexically_guard_path(
        path,
        description=description,
        protected_exact_allowlist=protected_exact_allowlist,
    )
    pinned = _lexically_guard_path(
        expected,
        description=f"{description}_pinned",
        protected_exact_allowlist=protected_exact_allowlist,
    )
    require(
        str(guarded) == str(pinned),
        f"{description}_must_equal_pinned_path path={guarded} expected={pinned}",
    )
    resolved = guarded.resolve()
    expected_resolved = pinned.resolve()
    require(
        resolved == expected_resolved
        and str(resolved) == str(guarded)
        and str(expected_resolved) == str(pinned),
        f"{description}_must_be_canonical path={guarded}",
    )
    return resolved


def require_under_root(
    path: Path | str,
    *,
    root: Path | str,
    description: str,
) -> Path:
    guarded = _lexically_guard_path(path, description=description)
    guarded_root = _lexically_guard_path(root, description=f"{description}_root")
    resolved = guarded.resolve()
    resolved_root = guarded_root.resolve()
    require(
        str(resolved) == str(guarded)
        and str(resolved_root) == str(guarded_root),
        f"{description}_must_be_canonical path={guarded}",
    )
    require(
        resolved == resolved_root or resolved_root in resolved.parents,
        f"{description}_outside_locked_root path={resolved} root={resolved_root}",
    )
    return resolved


def v10_run_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V10_RUN_RELATIVE


def v10_gates_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V10_GATES_RELATIVE


def require_v10_run_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v10_run_root_for(ssd_root),
        description=description,
    )


def require_v10_gate_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v10_gates_root_for(ssd_root),
        description=description,
    )


def require_ssd(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    """Reject protected names before resolving, opening, or hashing a path."""

    return require_under_root(path, root=ssd_root, description=description)


def _regular_file(path: Path, *, description: str) -> Path:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"{description}_missing_empty_or_symlink",
    )
    return path


def _load_object(path: Path, *, description: str) -> dict[str, Any]:
    _regular_file(path, description=description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"{description}_invalid_json") from exc
    require(isinstance(payload, dict), f"{description}_must_be_object")
    return payload


def presentation_cursor(global_step: int) -> int:
    require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and 0 <= global_step <= TOTAL_OPTIMIZER_STEPS,
        "global_step_outside_v10_optimizer_schedule",
    )
    return global_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require_exact_path(ssd_root, SSD_ROOT, description="v10_ssd_root")
    guarded_root = require_exact_path(
        data_root,
        DATA_ROOT,
        description="v10_data_root",
    )
    guarded_lock = require_exact_path(
        source_lock_path,
        SOURCE_LOCK,
        description="v10_source_lock",
    )
    _regular_file(guarded_lock, description="v10_source_lock")
    require(
        sha256_file(guarded_lock) == SOURCE_LOCK_FILE_SHA256,
        "v10_source_lock_file_hash_differs",
    )
    source_lock = _load_object(guarded_lock, description="v10_source_lock")
    locked_artifacts = source_lock.get("artifacts")
    require(isinstance(locked_artifacts, Mapping), "v10_source_lock_artifacts_missing")
    for name, expected in PINNED_DATA_ARTIFACTS.items():
        expected_binding = {
            "path": str(expected["path"]),
            "sha256": expected["sha256"],
        }
        require(
            locked_artifacts.get(name) == expected_binding,
            f"v10_source_lock_{name}_differs",
        )
        require_exact_path(
            expected_binding["path"],
            expected["path"],
            description=f"v10_locked_{name}",
        )
    locked_inputs = source_lock.get("inputs")
    require(isinstance(locked_inputs, Mapping), "v10_source_lock_inputs_missing")
    historical_paths = [
        Path(str(binding["path"]))
        for binding in PINNED_HISTORICAL_TRAIN32_ARTIFACTS.values()
    ]
    for name, expected in PINNED_HISTORICAL_TRAIN32_ARTIFACTS.items():
        expected_binding = {
            "bytes": expected["bytes"],
            "path": str(expected["path"]),
            "sha256": expected["sha256"],
        }
        require(
            locked_inputs.get(name) == expected_binding,
            f"v10_historical_train32_{name}_binding_differs",
        )
        require_exact_path(
            expected_binding["path"],
            expected["path"],
            description=f"v10_historical_train32_{name}",
            protected_exact_allowlist=historical_paths,
        )
    require(
        source_lock.get("excluded_artifacts")
        == {
            "hard32": {
                "included": False,
                "name": "Hard32",
                "path": None,
                "policy": HARD32_ACCESS_POLICY,
                "sha256": None,
            }
        },
        "v10_hard32_exclusion_binding_differs",
    )
    try:
        data = v9.validate_data_contract(
            data_root=guarded_root,
            source_lock_path=guarded_lock,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise LaunchContractError(f"v10_reused_v9_data_contract_failed: {exc}") from exc
    require(data.get("data_root") == str(DATA_ROOT), "v10_data_root_identity_differs")
    train32 = PINNED_HISTORICAL_TRAIN32_ARTIFACTS["train32"]
    require(
        data.get("train_file") == str(train32["path"])
        and data.get("train_file_sha256") == train32["sha256"],
        "v10_train32_identity_differs",
    )
    data_fields = {
        "bundle_manifest": "bundle_manifest",
        "schedule": "pair_schedule",
        "schedule_manifest": "pair_schedule_manifest",
        "source_manifest": "source_manifest",
    }
    for field, artifact_name in data_fields.items():
        expected = PINNED_DATA_ARTIFACTS[artifact_name]
        require(
            data.get(field) == str(expected["path"])
            and data.get(f"{field}_file_sha256") == expected["sha256"],
            f"v10_{field}_identity_differs",
        )
    entries = list(data["entries"])
    require(len(entries) == TOTAL_PAIR_PRESENTATIONS, "v10_pair_presentations_differ")
    canonical = {tuple(pair) for pair in CANONICAL_VALUE14_PAIRS}
    cycles: list[dict[str, Any]] = []
    for optimizer_step in CHECKPOINT_STEPS:
        start = presentation_cursor(optimizer_step - 1)
        stop = presentation_cursor(optimizer_step)
        current = entries[start:stop]
        pairs = [tuple(entry["canonical_pair_ordinals"]) for entry in current]
        require(
            len(current) == PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
            and len(set(pairs)) == PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
            and set(pairs) == canonical,
            f"v10_optimizer_cycle_not_complete step={optimizer_step}",
        )
        cycles.append(
            {
                "optimizer_step": optimizer_step,
                "presentation_start": start,
                "presentation_stop": stop,
                "pairs_sha256": canonical_sha256([list(pair) for pair in pairs]),
            }
        )
    result = dict(data)
    result["source_presentation_checkpoint_steps"] = list(data["checkpoint_steps"])
    result["checkpoint_steps"] = list(CHECKPOINT_STEPS)
    result["presentation_checkpoint_steps"] = list(PRESENTATION_CHECKPOINTS)
    result["optimizer_cycles"] = cycles
    return result


def validate_warm_start_contract(
    *,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    try:
        require_exact_path(ssd_root, SSD_ROOT, description="v10_ssd_root")
        lock_path = require_exact_path(
            warm_start_lock_path,
            WARM_START_LOCK,
            description="v10_warm_start_lock",
        )
        _regular_file(lock_path, description="v10_warm_start_lock")
        require(
            sha256_file(lock_path) == WARM_START_LOCK_FILE_SHA256,
            "v10_warm_start_lock_file_hash_differs",
        )
        lock = load_v10_warm_start_lock(lock_path)
        checkpoint = require_exact_path(
            Path(str(lock.get("source_checkpoint", ""))),
            PINNED_WARM_START_CHECKPOINT,
            description="v10_warm_start_checkpoint",
        )
        context = prepare_v10_v8_checkpoint56_warm_start(
            checkpoint,
            lock_path=lock_path,
        )
    except Exception as exc:
        raise LaunchContractError(f"v10_warm_start_contract_failed: {exc}") from exc
    return {
        "warm_start_checkpoint": str(context.checkpoint),
        "warm_start_lock": str(context.lock_path),
        "warm_start_lock_file_sha256": WARM_START_LOCK_FILE_SHA256,
        "warm_start_lock_sha256": context.lock["lock_sha256"],
        "warm_start_mode": WARM_START_MODE,
        "lock": context.lock,
    }


def _expected_schedule_protocol(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CURRICULUM_BINDING_SCHEMA,
        "source_manifest_path": data["source_manifest"],
        "source_manifest_file_sha256": data["source_manifest_file_sha256"],
        "schedule_path": data["schedule"],
        "schedule_file_sha256": data["schedule_file_sha256"],
        "schedule_entries_sha256": data["schedule_entries_sha256"],
        "schedule_manifest_path": data["schedule_manifest"],
        "schedule_manifest_file_sha256": data["schedule_manifest_file_sha256"],
        "schedule_manifest_sha256": data["schedule_manifest_sha256"],
        "ordered_pairs_sha256": data["ordered_pairs_sha256"],
        "canonical_value14_pairs": [list(pair) for pair in CANONICAL_VALUE14_PAIRS],
        "total_steps": TOTAL_PAIR_PRESENTATIONS,
        "checkpoint_steps": data["source_presentation_checkpoint_steps"],
        "pair_indices": [
            list(entry["canonical_pair_ordinals"]) for entry in data["entries"]
        ],
        "optimizer_checkpoint_steps": list(CHECKPOINT_STEPS),
        "microbatch_cycle_size": PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        "resume_schedule_cursor_formula": "global_step_times_7_v1",
    }


def _validate_checkpoint_protocol(
    protocol: Mapping[str, Any],
    *,
    checkpoint_step: int,
    data: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": OBJECTIVE_VERSION,
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": data["train_file"],
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "max_length": MAX_LENGTH,
        "max_write_length": MAX_WRITE_LENGTH,
        "teacher_max_length": TEACHER_MAX_LENGTH,
        "per_device_train_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "save_steps": SAVE_STEPS,
        "logging_steps": 1,
        "eval_steps": 1000,
        "save_total_limit": 1,
        "num_train_epochs": 1.0,
        "max_steps": checkpoint_step,
        "validation_split_ratio": 0.0,
        "load_best_model_at_end": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "frozen_mlp_activation_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "dtype": "bfloat16",
        "bf16": True,
        "tf32": True,
        "train_sampler_seed": None,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "ignore_data_skip": False,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_contrast_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "memory_causal_weight": 0.0,
        "memory_anchor_weight": 0.0,
        "memory_recover_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "scene_generation_objective_formula": OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": BACKWARD_MODE,
        "scene_generation_generated_unlikelihood_weight": 0.0,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": (
            GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_prefix_correction_weight": PREFIX_CORRECTION_WEIGHT,
        "scene_generation_generated_prefix_correction_mode": GENERATED_PREFIX_MODE,
        "scene_generation_generated_prefix_max_correction_events": (
            GENERATED_MAX_CORRECTION_EVENTS
        ),
        "scene_generation_generated_rollout_extra_tokens": (
            GENERATED_ROLLOUT_EXTRA_TOKENS
        ),
        "scene_generation_generated_rollout_max_tokens": (
            GENERATED_ROLLOUT_MAX_TOKENS
        ),
        "scene_generation_generated_rollout_decoding": (
            "greedy_use_cache_true_exact_system_only_prompt_v1"
        ),
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
        "scene_generation_pair_physical_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "scene_generation_pair_directional_exposures": PAIR_DIRECTIONAL_EXPOSURES,
        "scene_generation_first_error_top1_hinge_weight": 1.0,
        "scene_generation_all_target_top1_retention_weight": 1.0,
        "scene_generation_all_target_top1_retention_margin": 0.2,
        "scene_generation_selected_full_vocab_ce_in_total": False,
        "scene_generation_selected_full_vocab_ce_optimization_weight": 0.0,
        "scene_generation_cycle_retention_mode": CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        "scene_generation_gradient_accumulation_pair_cycle": GRADIENT_ACCUMULATION_STEPS,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    require(not mismatches, "resume_v10_protocol_differs fields=" + ",".join(mismatches))
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, Mapping), "resume_v10_protocol_schedule_missing")
    expected_schedule = _expected_schedule_protocol(data)
    schedule_mismatches = [
        name for name, value in expected_schedule.items() if schedule.get(name) != value
    ]
    require(
        not schedule_mismatches,
        "resume_v10_protocol_schedule_differs fields=" + ",".join(schedule_mismatches),
    )


def _validate_pairing_manifest(pairing: Mapping[str, Any]) -> str:
    unsigned = dict(pairing)
    recorded = unsigned.pop("manifest_sha256", None)
    require(
        require_sha256(recorded, description="resume_v10_pairing_hash")
        == canonical_sha256(unsigned),
        "resume_v10_pairing_self_hash_differs",
    )
    require(
        pairing.get("objective_version") == PAIRING_OBJECTIVE_VERSION,
        "resume_v10_pairing_materialization_objective_differs",
    )
    return str(recorded)


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
        "resume_v10_warm_lineage_schema_or_mode_differs",
    )
    require(
        require_sha256(recorded, description="resume_v10_warm_receipt_hash")
        == canonical_sha256(unsigned),
        "resume_v10_warm_lineage_self_hash_differs",
    )
    source_lock = lineage.get("source_lock")
    require(isinstance(source_lock, Mapping), "resume_v10_warm_source_lock_missing")
    require(
        lineage.get("source_checkpoint") == warm["warm_start_checkpoint"]
        and source_lock.get("path") == warm["warm_start_lock"]
        and source_lock.get("lock_sha256") == warm["warm_start_lock_sha256"]
        and lineage.get("source_state_imports") == SOURCE_IMPORT_POLICY
        and lineage.get("post_load_bit_equal") is True,
        "resume_v10_warm_source_binding_differs",
    )
    expected_fresh = {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v10_seed",
    }
    require(
        lineage.get("target_fresh_start") == expected_fresh,
        "resume_v10_warm_fresh_start_differs",
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
    require(
        not mismatches,
        "resume_v10_warm_target_evidence_differs fields=" + ",".join(mismatches),
    )
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        isinstance(optimizer_class, str) and optimizer_class.endswith(".AdamW"),
        "resume_v10_warm_optimizer_class_differs",
    )
    return str(recorded)


def _validate_checkpoint_lineage(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path,
    visited: set[Path] | None = None,
) -> dict[str, Any]:
    resolved = require_v10_run_path(
        checkpoint,
        description="v10_resume_checkpoint",
        ssd_root=ssd_root,
    )
    require(resolved.is_dir(), "resume_v10_checkpoint_missing")
    active = set() if visited is None else visited
    require(resolved not in active, "resume_v10_lineage_cycle")
    active.add(resolved)
    suffix = resolved.name.removeprefix("checkpoint-")
    require(
        resolved.name.startswith("checkpoint-") and suffix.isdigit(),
        "resume_v10_checkpoint_must_be_checkpoint_n",
    )
    step = int(suffix)
    require(step in CHECKPOINT_STEPS, "resume_source_not_locked_v10_endpoint")
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / filename, description=f"resume_v10_{filename}")
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in rng_files),
        "resume_v10_rng_state_missing",
    )
    trainer_state = _load_object(resolved / "trainer_state.json", description="resume_v10_trainer_state")
    protocol = _load_object(resolved / "training_protocol.json", description="resume_v10_protocol")
    config = _load_object(resolved / "delta_mem_config.json", description="resume_v10_config")
    pairing = _load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="resume_v10_pairing",
    )
    require(
        trainer_state.get("global_step") == step
        and trainer_state.get("max_steps") == step
        and protocol.get("max_steps") == step,
        "resume_v10_checkpoint_not_completed_horizon",
    )
    _validate_checkpoint_protocol(protocol, checkpoint_step=step, data=data)
    try:
        v9._validate_checkpoint_config(config)
    except Exception as exc:
        raise LaunchContractError(f"resume_v10_delta_config_failed: {exc}") from exc
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing)
    expected_lineage = (
        WARM_START_LINEAGE_FILENAME
        if step == CHECKPOINT_STEPS[0]
        else CONTINUATION_LINEAGE_FILENAME
    )
    lineage_names = (
        WARM_START_LINEAGE_FILENAME,
        CONTINUATION_LINEAGE_FILENAME,
        ABLATION_LINEAGE_FILENAME,
    )
    present = [name for name in lineage_names if (resolved / name).is_file()]
    require(present == [expected_lineage], "resume_v10_single_lineage_file_differs")
    lineage_path = resolved / expected_lineage
    lineage = _load_object(lineage_path, description="resume_v10_lineage")
    if step == CHECKPOINT_STEPS[0]:
        root_receipt = _validate_warm_lineage(
            lineage,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            pairing_sha256=pairing_sha256,
            warm=warm,
        )
    else:
        unsigned = dict(lineage)
        recorded = unsigned.pop("manifest_sha256", None)
        require(
            lineage.get("schema_version") == CONTINUATION_LINEAGE_SCHEMA_VERSION
            and lineage.get("mode") == "extend",
            "resume_v10_continuation_schema_or_mode_differs",
        )
        require(
            require_sha256(recorded, description="resume_v10_continuation_hash")
            == canonical_sha256(unsigned),
            "resume_v10_continuation_self_hash_differs",
        )
        expected_source_step = CHECKPOINT_STEPS[CHECKPOINT_STEPS.index(step) - 1]
        require(
            lineage.get("source_global_step") == expected_source_step
            and lineage.get("target_max_steps") == step
            and lineage.get("source_schedule_cursor")
            == presentation_cursor(expected_source_step)
            and lineage.get("target_schedule_cursor") == presentation_cursor(step)
            and lineage.get("target_training_protocol_sha256") == protocol_sha256,
            "resume_v10_continuation_horizon_or_protocol_differs",
        )
        raw_source = lineage.get("source_checkpoint")
        source = require_v10_run_path(
            Path(str(raw_source)),
            description="v10_continuation_source",
            ssd_root=ssd_root,
        )
        require(raw_source == str(source), "resume_v10_continuation_source_not_canonical")
        prior = _validate_checkpoint_lineage(
            source,
            data=data,
            warm=warm,
            ssd_root=ssd_root,
            visited=active,
        )
        require(
            prior["checkpoint_step"] == expected_source_step
            and lineage.get("source_training_protocol_sha256") == prior["training_protocol_sha256"]
            and lineage.get("source_lineage_filename") == prior["lineage_filename"]
            and lineage.get("source_lineage_file_sha256") == prior["lineage_file_sha256"]
            and lineage.get("root_warm_start_receipt_sha256") == prior["root_warm_start_receipt_sha256"],
            "resume_v10_continuation_source_lineage_differs",
        )
        root_receipt = prior["root_warm_start_receipt_sha256"]
    active.remove(resolved)
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": step,
        "consumed_pair_presentations": presentation_cursor(step),
        "lineage_filename": expected_lineage,
        "lineage_file_sha256": sha256_file(lineage_path),
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt,
    }


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    return _validate_checkpoint_lineage(
        checkpoint,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )


def validate_resume_contract(
    *,
    resume_checkpoint: Path,
    target_step: int,
    gate_receipt: Path | None,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require(gate_receipt is not None, "resume_requires_explicit_v10_gate_receipt")
    lineage = validate_checkpoint_contract(
        resume_checkpoint,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )
    source_step = int(lineage["checkpoint_step"])
    source_index = CHECKPOINT_STEPS.index(source_step)
    require(source_index + 1 < len(CHECKPOINT_STEPS), "resume_final_v10_checkpoint_has_no_successor")
    require(
        target_step == CHECKPOINT_STEPS[source_index + 1],
        "resume_target_is_not_next_locked_v10_endpoint",
    )
    resolved_receipt = require_v10_gate_path(
        gate_receipt,
        description="v10_progression_gate_receipt",
        ssd_root=ssd_root,
    )
    try:
        from experiments.rethinking_rwkv_ms_gemma.run_scene_memory_v10_gate import (
            validate_continuation_authorization,
        )

        authorization = validate_continuation_authorization(
            resolved_receipt,
            source_checkpoint=lineage["checkpoint"],
            target_step=target_step,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise LaunchContractError(f"v10_progression_gate_authorization_failed: {exc}") from exc
    require(
        authorization.get("source_checkpoint") == lineage["checkpoint"]
        and authorization.get("source_step") == source_step
        and authorization.get("target_step") == target_step
        and authorization.get("hard32_authorized") is False,
        "v10_progression_gate_binding_differs",
    )
    cursor = presentation_cursor(source_step)
    entry = data["entries"][cursor]
    return {
        "launch_mode": "resume",
        "source_step": source_step,
        "target_step": target_step,
        "resume_checkpoint": lineage["checkpoint"],
        "resume_schedule_cursor": cursor,
        "next_pair_low_ordinal": entry["canonical_pair_ordinals"][0],
        "next_pair_high_ordinal": entry["canonical_pair_ordinals"][1],
        "next_schedule_entry_sha256": entry["entry_sha256"],
        "root_warm_start_receipt_sha256": lineage["root_warm_start_receipt_sha256"],
        "gate_authorization_kind": authorization["authorization_kind"],
        "gate_receipt": authorization["gate_receipt"],
        "gate_receipt_file_sha256": authorization["gate_receipt_file_sha256"],
        "gate_receipt_sha256": authorization["gate_receipt_sha256"],
    }


def validate_launch_contract(
    *,
    target_step: int,
    resume_checkpoint: Path | None = None,
    gate_receipt: Path | None = None,
    smoke: bool = False,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require(target_step in CHECKPOINT_STEPS, "target_step_not_locked_v10_endpoint")
    if smoke:
        require(target_step == 1, "v10_smoke_must_target_complete_cycle_step1")
        require(resume_checkpoint is None, "v10_smoke_forbids_resume")
        require(gate_receipt is None, "v10_smoke_forbids_gate_receipt")
    data = validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    warm = validate_warm_start_contract(
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    if resume_checkpoint is None:
        require(gate_receipt is None, "fresh_v10_launch_forbids_gate_receipt")
        require(target_step == CHECKPOINT_STEPS[0], "fresh_v10_launch_must_target_step1")
        first = data["entries"][0]
        cursor = {
            "launch_mode": "warm_start_smoke" if smoke else "warm_start",
            "source_step": 0,
            "target_step": target_step,
            "resume_checkpoint": None,
            "resume_schedule_cursor": 0,
            "next_pair_low_ordinal": first["canonical_pair_ordinals"][0],
            "next_pair_high_ordinal": first["canonical_pair_ordinals"][1],
            "next_schedule_entry_sha256": first["entry_sha256"],
            "root_warm_start_receipt_sha256": None,
            "gate_authorization_kind": "not_required_fresh_start",
            "gate_receipt": "not_required_fresh_start",
            "gate_receipt_file_sha256": "not_required_fresh_start",
            "gate_receipt_sha256": "not_required_fresh_start",
        }
    else:
        cursor = validate_resume_contract(
            resume_checkpoint=resume_checkpoint,
            target_step=target_step,
            gate_receipt=gate_receipt,
            data=data,
            warm=warm,
            ssd_root=ssd_root,
        )
    public_data = {key: value for key, value in data.items() if key != "entries"}
    public_warm = {key: value for key, value in warm.items() if key != "lock"}
    return {
        **public_data,
        **public_warm,
        **cursor,
        "total_pair_presentations": TOTAL_PAIR_PRESENTATIONS,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "presentation_checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
        "objective_version": OBJECTIVE_VERSION,
        "pairing_objective_version": PAIRING_OBJECTIVE_VERSION,
        "pair_physical_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "pair_logical_batch_size": PAIR_LOGICAL_BATCH_SIZE,
        "pair_directional_exposures": PAIR_DIRECTIONAL_EXPOSURES,
        "pair_presentations_per_optimizer_step": PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "save_steps": SAVE_STEPS,
        "hard32_access": HARD32_ACCESS_POLICY,
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
        result["gate_authorization_kind"],
        result["gate_receipt"],
        result["gate_receipt_file_sha256"],
        result["gate_receipt_sha256"],
    )
    rendered = tuple(str(value) for value in fields)
    require(
        all("\t" not in value and "\n" not in value for value in rendered),
        "v10_launch_contract_tsv_control_character",
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
            ssd_root=args.ssd_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(_tsv(result) if args.format == "tsv" else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
