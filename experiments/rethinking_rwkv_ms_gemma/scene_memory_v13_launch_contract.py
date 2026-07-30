#!/usr/bin/env python3
"""Fail-closed launch contract for fresh four-cycle Scene Memory V13 training."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v10_launch_contract as v10,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v13_warm_start import (
    ABLATION_LINEAGE_FILENAME,
    CONTINUATION_LINEAGE_FILENAME,
    DEFAULT_LOCK_PATH as WARM_START_LOCK,
    RECEIPT_SCHEMA as WARM_START_RECEIPT_SCHEMA,
    SOURCE_IMPORT_POLICY,
    WARM_START_LINEAGE_FILENAME,
    WARM_START_MODE,
    load_v13_warm_start_lock,
    prepare_v13_v8_checkpoint56_warm_start,
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
ROW_OBJECTIVE_AUDIT_FILENAME = "scene_memory_v13_row_objective.json"
ROW_OBJECTIVE_AUDIT_SCHEMA = "rwkv_ms_scene_memory_v13_row_objective.v1"
REQUIRED_CHECKPOINT_ARTIFACTS = (
    *v10.REQUIRED_CHECKPOINT_ARTIFACTS,
    ROW_OBJECTIVE_AUDIT_FILENAME,
)

V13_RUN_ROOT = (
    SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v13"
)
V13_GATES_ROOT = V13_RUN_ROOT / "gates"
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
PINNED_V10_DIAGNOSTIC_HARD32_ACCESS = (
    "forbidden_not_resolved_opened_or_hashed"
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

OBJECTIVE_VERSION = (
    "scene_state_generation_ce_symmetric_dense_boundary_v13"
)
PAIRING_OBJECTIVE_VERSION = v10.PAIRING_OBJECTIVE_VERSION
OBJECTIVE_SCHEMA_VERSION = 16
FIXED_SAMPLER_MODE = "explicit_ordered_v13_four_canonical_seven_pair_cycles_v1"
PAIR_PHYSICAL_BATCH_SIZE = 1
PAIR_LOGICAL_BATCH_SIZE = 2
PAIR_DIRECTIONAL_EXPOSURES = 2
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 7
TOTAL_PAIR_PRESENTATIONS = 28
TOTAL_OPTIMIZER_STEPS = 4
CHECKPOINT_STEPS = (1, 2, 3, 4)
PRESENTATION_CHECKPOINTS = (7, 14, 21, 28)
CONTINUATION_POLICY = "forbidden"
GRADIENT_ACCUMULATION_STEPS = 7
SAVE_STEPS = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0
WARMUP_RATIO = 0.0
PREFIX_CORRECTION_WEIGHT = 0.0
SEMANTIC_MARGIN = 1.0
GENERATED_MAX_CORRECTION_EVENTS = 0
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
SECOND_CYCLE_PAIRS = (
    (19, 28),
    (22, 26),
    (5, 9),
    (3, 24),
    (20, 31),
    (10, 23),
    (1, 14),
)
THIRD_CYCLE_PAIRS = (
    (1, 14),
    (19, 28),
    (22, 26),
    (20, 31),
    (10, 23),
    (5, 9),
    (3, 24),
)
FOURTH_CYCLE_PAIRS = (
    (22, 26),
    (19, 28),
    (10, 23),
    (3, 24),
    (20, 31),
    (5, 9),
    (1, 14),
)
FOUR_CYCLE_PAIRS = (
    *FIRST_CYCLE_PAIRS,
    *SECOND_CYCLE_PAIRS,
    *THIRD_CYCLE_PAIRS,
    *FOURTH_CYCLE_PAIRS,
)
FOUR_CYCLE_PAIRS_SHA256 = (
    "d710fec2abee4e7b8b5ecf7f75f005d7d872a17b6c0aea2f193dcb29e5f3a55d"
)
PAIR_PREFIX_SHA256_BY_CHECKPOINT = {
    1: FIRST_CYCLE_PAIRS_SHA256,
    2: "72d0e0584fcdc19c08829ade6e06aad393ef3c0ef757f76cc96b53f337c8ab1c",
    3: "8fbff7d9e4c2374f40cfc97be14e4876eff9dda47ac5bd5b38a8f3f9061ab8f5",
    4: FOUR_CYCLE_PAIRS_SHA256,
}
DENSE_SEMANTIC_MODE = (
    "all_boundary_decision_ce_full_vocab_top1_retention_failed_first_"
    "semantic_actual_greedy_decision_only_repair_v2"
)
FAILED_DECISION_ALIGNMENT_MODE = (
    "first_boundary_semantic_effect_edit_mapped_to_decision_mask_only_v1"
)
DENSE_DECISION_MASK_MODE = "top_level_boundaries_json_decision_char_overlap_v1"
DENSE_DECISION_SCOPE = "all_boundary_decision_mask_tokens_v1"
DENSE_DECISION_TOKEN_OVERLAP_POLICY = (
    "supervise_whole_token_if_any_character_overlaps_boundary_decision_char_v1"
)
DENSE_TOP1_RETENTION_HINGE_MODE = (
    "dense_gold_vs_detached_top_competitor_hinge_v1"
)
SCHEMA_CE_SCOPE = "standalone_schema_mask_partition_only_v1"
PARSED_EXACTNESS_MODE = "parsed_json_literal_boundary_equality_v1"
PARSED_EXACTNESS = "benchmark_literal_boundary_set_equality_v1"
FAILED_PREFIX_REPLAY = (
    "detached_actual_greedy_prefix_reprime_history_with_gradient_v1"
)
FAILED_REPLAY_MODE = "detached_actual_greedy_prefix_differentiable_replay_v1"
OBJECTIVE_FORMULA = (
    "symmetric_pair_mean(all_boundary_decision_full_vocab_ce + "
    "all_boundary_decision_full_vocab_top1_retention_hinge(1.0) + "
    "selected_top_competitor_hinge(0.2) + "
    "selected_correct_vs_detached_zero_nll_hinge(0.2) + "
    "if(parsed_boundary_failed,first_actual_prefix_semantic_ce+"
    "actual_greedy_competitor_hinge(1.0),0)); "
    "standalone_schema_partition_footer_and_chat_termination_ce=0; "
    "decision_overlap_tokens_are_supervised_as_whole_tokens"
)
BACKWARD_MODE = (
    "sequential_pair_zero_probe_dense_boundary_teacher_then_failed_first_"
    "semantic_actual_prefix_replay_v9"
)
CYCLE_RETENTION_MODE = v10.CYCLE_RETENTION_MODE
HARD32_ACCESS_POLICY = (
    "hard32_benchmark_artifact_forbidden_pinned_train32_source_allowed_v1"
)
TRAINING_CONTINUATION_POLICY = "forbidden_fresh_four_cycle_run_is_terminal"
LAUNCH_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v13_attached_launch.v1"
COMPLETION_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v13_attached_completion.v1"
RECOVERED_COMPLETION_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v13_recovered_completion.v1"
)
COMPLETION_RECOVERY_SCHEMA = "rwkv_ms_scene_memory_v13_completion_recovery.v1"
COMPLETION_RECOVERY_REASON = "git_object_id_validator_length_bug_v1"
RECOVERY_VALIDATOR_FILES = (
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_warm_start.py",
)
CRITICAL_TRAINING_FILES = (
    "deltamem/scene_boundary.py",
    "deltamem/train/delta_sft_experimental.py",
    "deltamem/train/scene_state_generation_alignment.py",
    "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v9_data.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_data_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_source_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_v8_checkpoint56_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v9_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v10_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v13_gate.py",
    "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v13.sh",
)

_V13_RUN_RELATIVE = V13_RUN_ROOT.relative_to(SSD_ROOT)
_V13_GATES_RELATIVE = V13_GATES_ROOT.relative_to(SSD_ROOT)


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

_FORBIDDEN_TRAINING_DATA_COMPONENTS = frozenset(
    {
        "eval",
        "evaluation",
        "full170",
        "hard32",
        "holdout",
        "test",
        "tests",
        "val",
        "validation",
    }
)


def guard_v13_training_data_path(
    path: Path | str,
    *,
    description: str,
) -> Path:
    """Reject protected split names before resolution or filesystem access."""

    raw = Path(path).expanduser()
    normalized = Path(os.path.normpath(str(raw)))
    # The pinned training split lives in a historically named directory that
    # mentions Hard32. It is Train32, not the separate Hard32 benchmark artifact.
    historical_train32 = Path(
        str(PINNED_HISTORICAL_TRAIN32_ARTIFACTS["train32"]["path"])
    )
    if normalized == historical_train32:
        return normalized
    forbidden = []
    for component in normalized.parts:
        lowered = component.lower()
        stem = Path(lowered).stem
        if (
            lowered in _FORBIDDEN_TRAINING_DATA_COMPONENTS
            or stem in _FORBIDDEN_TRAINING_DATA_COMPONENTS
            or lowered.startswith("hard32")
            or stem.startswith("hard32")
            or lowered.startswith("full170")
            or stem.startswith("full170")
        ):
            forbidden.append(component)
    require(
        not forbidden,
        f"{description}_protected_split_path_forbidden path={normalized}",
    )
    return normalized


def v13_run_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V13_RUN_RELATIVE


def v13_gates_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root) / _V13_GATES_RELATIVE


def require_v13_run_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v13_run_root_for(ssd_root),
        description=description,
    )


def require_v13_gate_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=v13_gates_root_for(ssd_root),
        description=description,
    )


def presentation_cursor(global_step: int) -> int:
    require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step in (0, *CHECKPOINT_STEPS),
        "global_step_outside_v13_four_cycle_schedule",
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


def validate_v13_cycle_pair_telemetry(
    trainer_state: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> dict[str, Any]:
    require(
        checkpoint_step in CHECKPOINT_STEPS,
        "v13_cycle_pair_checkpoint_step_differs",
    )
    history = trainer_state.get("log_history")
    require(isinstance(history, list), "v13_trainer_log_history_missing")
    count_key = "delta/scene_generation_v13_cycle_pair_presentations"
    candidates = [
        (index, entry)
        for index, entry in enumerate(history)
        if isinstance(entry, Mapping) and count_key in entry
    ]
    require(
        len(candidates) == checkpoint_step,
        "v13_cycle_pair_telemetry_optimizer_log_count_differs",
    )
    cycles: list[dict[str, Any]] = []
    pairs: list[list[int]] = []
    for cycle_index, (history_index, entry) in enumerate(candidates, start=1):
        presentations = _finite_integral_metric(
            entry.get(count_key),
            description=f"v13_cycle_{cycle_index}_pair_presentations",
        )
        require(presentations == 7, "v13_cycle_pair_presentations_differ")
        require(
            _finite_integral_metric(
                entry.get("step"),
                description=f"v13_cycle_{cycle_index}_step",
            )
            == cycle_index,
            "v13_cycle_pair_telemetry_step_differs",
        )
        require(
            _finite_integral_metric(
                entry.get("delta/scene_generation_v13_cycle_index"),
                description=f"v13_cycle_{cycle_index}_index",
            )
            == cycle_index,
            "v13_cycle_pair_telemetry_cycle_index_differs",
        )
        expected_cycle = FOUR_CYCLE_PAIRS[
            (cycle_index - 1) * 7 : cycle_index * 7
        ]
        cycle_pairs: list[list[int]] = []
        telemetry_keys = [
            count_key,
            "delta/scene_generation_v13_cycle_index",
        ]
        for pair_index, expected_pair in enumerate(expected_cycle):
            low_key = (
                f"delta/scene_generation_v13_cycle_pair_{pair_index}_low_ordinal"
            )
            high_key = (
                f"delta/scene_generation_v13_cycle_pair_{pair_index}_high_ordinal"
            )
            low = _finite_integral_metric(
                entry.get(low_key),
                description=f"v13_cycle_{cycle_index}_pair_{pair_index}_low_ordinal",
            )
            high = _finite_integral_metric(
                entry.get(high_key),
                description=f"v13_cycle_{cycle_index}_pair_{pair_index}_high_ordinal",
            )
            require(
                (low, high) == expected_pair,
                f"v13_cycle_{cycle_index}_pair_{pair_index}_order_differs",
            )
            cycle_pairs.append([low, high])
            telemetry_keys.extend((low_key, high_key))
        pairs.extend(cycle_pairs)
        cycles.append(
            {
                "cycle_index": cycle_index,
                "trainer_log_history_index": history_index,
                "optimizer_step": cycle_index,
                "pair_presentations": presentations,
                "ordered_pairs": cycle_pairs,
                "ordered_pairs_sha256": canonical_sha256(cycle_pairs),
                "telemetry_keys": telemetry_keys,
            }
        )
    pairs_sha256 = canonical_sha256(pairs)
    require(
        pairs_sha256 == PAIR_PREFIX_SHA256_BY_CHECKPOINT[checkpoint_step],
        "v13_cycle_pair_telemetry_hash_differs",
    )
    return {
        "schema": "rwkv_ms_scene_memory_v13_cycle_pair_telemetry.v1",
        "optimizer_step": checkpoint_step,
        "pair_presentations": checkpoint_step * 7,
        "ordered_pairs": pairs,
        "ordered_pairs_sha256": pairs_sha256,
        "cycles": cycles,
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
    require_exact_path(ssd_root, SSD_ROOT, description="v13_ssd_root")
    summary_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_SUMMARY,
        PINNED_V10_DIAGNOSTIC_SUMMARY,
        description="v13_v10_diagnostic_summary",
    )
    receipt_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_RECEIPT,
        PINNED_V10_DIAGNOSTIC_RECEIPT,
        description="v13_v10_diagnostic_receipt",
    )
    manifest_path = require_exact_path(
        PINNED_V10_DIAGNOSTIC_MANIFEST,
        PINNED_V10_DIAGNOSTIC_MANIFEST,
        description="v13_v10_diagnostic_manifest",
    )
    _regular_file(summary_path, description="v13_v10_diagnostic_summary")
    _regular_file(receipt_path, description="v13_v10_diagnostic_receipt")
    _regular_file(manifest_path, description="v13_v10_diagnostic_manifest")
    require(
        sha256_file(summary_path) == PINNED_V10_DIAGNOSTIC_SUMMARY_FILE_SHA256,
        "v13_v10_diagnostic_summary_file_hash_differs",
    )
    require(
        sha256_file(receipt_path) == PINNED_V10_DIAGNOSTIC_RECEIPT_FILE_SHA256,
        "v13_v10_diagnostic_receipt_file_hash_differs",
    )
    require(
        sha256_file(manifest_path) == PINNED_V10_DIAGNOSTIC_MANIFEST_FILE_SHA256,
        "v13_v10_diagnostic_manifest_file_hash_differs",
    )
    summary = _load_object(summary_path, description="v13_v10_diagnostic_summary")
    receipt = _load_object(receipt_path, description="v13_v10_diagnostic_receipt")
    manifest = _load_object(manifest_path, description="v13_v10_diagnostic_manifest")
    _validate_self_hash(
        summary,
        field="summary_sha256",
        expected=PINNED_V10_DIAGNOSTIC_SUMMARY_SHA256,
        description="v13_v10_diagnostic_summary",
    )
    _validate_self_hash(
        receipt,
        field="receipt_sha256",
        expected=PINNED_V10_DIAGNOSTIC_RECEIPT_SHA256,
        description="v13_v10_diagnostic_receipt",
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
        "v13_v10_diagnostic_schema_or_fingerprint_differs",
    )
    require(
        manifest.get("fingerprint") == PINNED_V10_DIAGNOSTIC_FINGERPRINT
        and manifest.get("hard32_access") == PINNED_V10_DIAGNOSTIC_HARD32_ACCESS,
        "v13_v10_diagnostic_manifest_identity_differs",
    )
    fingerprint_payload = manifest.get("fingerprint_payload")
    require(
        isinstance(fingerprint_payload, Mapping)
        and fingerprint_payload.get("base_model") == str(PINNED_BASE_MODEL),
        "v13_v10_diagnostic_base_model_path_differs",
    )
    base_model_weights = fingerprint_payload.get("base_model_weights")
    base_model_prompt_artifacts = fingerprint_payload.get(
        "base_model_prompt_artifacts"
    )
    require(
        isinstance(base_model_weights, Mapping)
        and isinstance(base_model_prompt_artifacts, Mapping),
        "v13_v10_diagnostic_base_model_identity_missing",
    )
    checkpoint = receipt.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "v13_v10_diagnostic_checkpoint_missing")
    require(
        checkpoint.get("memory_dir") == str(PINNED_V10_DIAGNOSTIC_CHECKPOINT)
        and checkpoint.get("global_step") == 1,
        "v13_v10_diagnostic_checkpoint_differs",
    )
    gate = summary.get("gate")
    require(isinstance(gate, Mapping), "v13_v10_diagnostic_gate_missing")
    require(
        gate.get("status") == "fail"
        and gate.get("training_continuation_authorized") is False
        and gate.get("hard32_authorized") is False
        and gate.get("full170_authorized") is False,
        "v13_v10_diagnostic_authorization_differs",
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
        "v13_v10_diagnostic_metrics_differ",
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
    require_exact_path(ssd_root, SSD_ROOT, description="v13_ssd_root")
    resolved = require_exact_path(
        base_model,
        PINNED_BASE_MODEL,
        description="v13_base_model",
    )
    require(
        resolved.is_dir() and not resolved.is_symlink(),
        "v13_base_model_missing_or_symlink",
    )
    diagnostic = (
        validate_v10_diagnostic_baseline(ssd_root=ssd_root)
        if baseline is None
        else dict(baseline)
    )
    expected = diagnostic.get("base_model_identity")
    require(isinstance(expected, Mapping), "v13_v10_base_model_identity_missing")
    try:
        from experiments.rethinking_rwkv_ms_gemma import (
            run_scene_memory_v9_gate as v9_gate,
        )

        weights = v9_gate.base_model_weight_identity(resolved)
        prompt_artifacts = v9_gate.base_model_prompt_identity(resolved)
    except Exception as exc:
        raise LaunchContractError(f"v13_base_model_identity_failed: {exc}") from exc
    current = {
        "path": str(resolved),
        "weights": weights,
        "prompt_artifacts": prompt_artifacts,
    }
    require(current == expected, "v13_base_model_differs_from_pinned_v10_manifest")
    return current


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    guard_v13_training_data_path(data_root, description="v13_data_root")
    guard_v13_training_data_path(
        source_lock_path,
        description="v13_source_lock",
    )
    try:
        data = v10.validate_data_contract(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise LaunchContractError(f"v13_reused_v10_data_contract_failed: {exc}") from exc
    entries = list(data["entries"])
    require(len(entries) >= TOTAL_PAIR_PRESENTATIONS, "v13_pair_schedule_too_short")
    scheduled_pairs = tuple(
        tuple(entry["canonical_pair_ordinals"])
        for entry in entries[:TOTAL_PAIR_PRESENTATIONS]
    )
    require(
        scheduled_pairs == FOUR_CYCLE_PAIRS,
        "v13_four_cycle_pair_order_differs",
    )
    require(
        canonical_sha256([list(pair) for pair in scheduled_pairs])
        == FOUR_CYCLE_PAIRS_SHA256,
        "v13_four_cycle_pair_hash_differs",
    )
    result = dict(data)
    result["source_checkpoint_steps"] = list(data["checkpoint_steps"])
    result["source_presentation_checkpoint_steps"] = list(
        data["source_presentation_checkpoint_steps"]
    )
    result["checkpoint_steps"] = list(CHECKPOINT_STEPS)
    result["presentation_checkpoint_steps"] = list(PRESENTATION_CHECKPOINTS)
    result["optimizer_cycles"] = [dict(cycle) for cycle in data["optimizer_cycles"]]
    result["first_cycle_pairs"] = [list(pair) for pair in FIRST_CYCLE_PAIRS]
    result["first_cycle_pairs_sha256"] = FIRST_CYCLE_PAIRS_SHA256
    result["second_cycle_pairs"] = [list(pair) for pair in SECOND_CYCLE_PAIRS]
    result["third_cycle_pairs"] = [list(pair) for pair in THIRD_CYCLE_PAIRS]
    result["fourth_cycle_pairs"] = [list(pair) for pair in FOURTH_CYCLE_PAIRS]
    result["four_cycle_pairs"] = [list(pair) for pair in FOUR_CYCLE_PAIRS]
    result["four_cycle_pairs_sha256"] = FOUR_CYCLE_PAIRS_SHA256
    return result


def validate_warm_start_contract(
    *,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    try:
        require_exact_path(ssd_root, SSD_ROOT, description="v13_ssd_root")
        lock_path = require_exact_path(
            warm_start_lock_path,
            WARM_START_LOCK,
            description="v13_warm_start_lock",
        )
        _regular_file(lock_path, description="v13_warm_start_lock")
        require(
            sha256_file(lock_path) == WARM_START_LOCK_FILE_SHA256,
            "v13_warm_start_lock_file_hash_differs",
        )
        lock = load_v13_warm_start_lock(lock_path)
        checkpoint = require_exact_path(
            Path(str(lock.get("source_checkpoint", ""))),
            PINNED_WARM_START_CHECKPOINT,
            description="v13_warm_start_checkpoint",
        )
        context = prepare_v13_v8_checkpoint56_warm_start(
            checkpoint,
            lock_path=lock_path,
        )
    except Exception as exc:
        raise LaunchContractError(f"v13_warm_start_contract_failed: {exc}") from exc
    require(
        context.checkpoint != PINNED_V10_DIAGNOSTIC_CHECKPOINT,
        "v13_must_never_warm_start_from_v10",
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
    require(isinstance(schedule, Mapping), "v13_protocol_train_schedule_missing")
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
        "v13_protocol_schedule_differs fields=" + ",".join(schedule_mismatches),
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
            "max_steps": 1,
            "learning_rate": v10.LEARNING_RATE,
            "save_total_limit": 1,
            "train_sampler_mode": v10.FIXED_SAMPLER_MODE,
            "scene_generation_objective_formula": v10.OBJECTIVE_FORMULA,
            "scene_generation_backward_mode": v10.BACKWARD_MODE,
            "scene_generation_generated_prefix_correction_weight": (
                v10.PREFIX_CORRECTION_WEIGHT
            ),
            "scene_generation_generated_prefix_correction_mode": (
                v10.GENERATED_PREFIX_MODE
            ),
            "scene_generation_generated_unlikelihood_max_wrong_tokens": (
                v10.GENERATED_MAX_CORRECTION_EVENTS
            ),
            "scene_generation_generated_prefix_max_correction_events": (
                v10.GENERATED_MAX_CORRECTION_EVENTS
            ),
            "scene_generation_first_error_top1_hinge_weight": 1.0,
            "scene_generation_all_target_top1_retention_weight": 1.0,
            "scene_generation_all_target_top1_retention_margin": 0.2,
        }
    )
    try:
        v10._validate_checkpoint_protocol(
            compatible,
            checkpoint_step=1,
            data=data,
        )
    except Exception as exc:
        raise LaunchContractError(f"v13_reused_v10_protocol_failed: {exc}") from exc
    expected = {
        "schema_version": OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": OBJECTIVE_VERSION,
        "max_steps": TOTAL_OPTIMIZER_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "save_steps": 1,
        "save_total_limit": len(CHECKPOINT_STEPS),
        "ignore_data_skip": False,
        "scene_generation_objective_formula": OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": BACKWARD_MODE,
        "scene_generation_generated_prefix_correction_weight": (
            PREFIX_CORRECTION_WEIGHT
        ),
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
        "scene_generation_generated_prefix_correction_mode": DENSE_SEMANTIC_MODE,
        "scene_generation_parsed_exactness": PARSED_EXACTNESS,
        "scene_generation_parsed_exactness_mode": PARSED_EXACTNESS_MODE,
        "scene_generation_raw_token_exactness_role": "telemetry_only",
        "scene_generation_failed_replay_mode": FAILED_REPLAY_MODE,
        "scene_generation_failed_prefix_replay": FAILED_PREFIX_REPLAY,
        "scene_generation_failed_decision_alignment": (
            FAILED_DECISION_ALIGNMENT_MODE
        ),
        "scene_generation_decision_mask_mode": DENSE_DECISION_MASK_MODE,
        "scene_generation_dense_semantic_mode": DENSE_SEMANTIC_MODE,
        "scene_generation_dense_decision_scope": DENSE_DECISION_SCOPE,
        "scene_generation_dense_decision_token_overlap_policy": (
            DENSE_DECISION_TOKEN_OVERLAP_POLICY
        ),
        "scene_generation_dense_decision_ce_weight": 1.0,
        "scene_generation_dense_top1_retention_hinge_weight": 1.0,
        "scene_generation_dense_top1_retention_hinge_mode": (
            DENSE_TOP1_RETENTION_HINGE_MODE
        ),
        "scene_generation_dense_top1_retention_margin": SEMANTIC_MARGIN,
        "scene_generation_failed_semantic_repair_ce_weight": 1.0,
        "scene_generation_failed_semantic_repair_hinge_weight": 1.0,
        "scene_generation_failed_semantic_repair_margin": SEMANTIC_MARGIN,
        "scene_generation_raw_token_exact_optimization_weight": 0.0,
        "scene_generation_full_answer_ce_optimization_weight": 0.0,
        "scene_generation_schema_ce_optimization_weight": 0.0,
        "scene_generation_schema_ce_optimization_scope": SCHEMA_CE_SCOPE,
        "scene_generation_footer_ce_optimization_weight": 0.0,
        "scene_generation_termination_ce_optimization_weight": 0.0,
        "scene_generation_selected_full_vocab_ce_in_total": False,
        "scene_generation_selected_full_vocab_ce_optimization_weight": 0.0,
        "scene_generation_row_objective_audit_filename": (
            ROW_OBJECTIVE_AUDIT_FILENAME
        ),
        "scene_generation_row_objective_audit_schema": ROW_OBJECTIVE_AUDIT_SCHEMA,
        "scene_generation_cycle_retention_mode": CYCLE_RETENTION_MODE,
        "scene_generation_cycle_pair_presentations": 7,
        "scene_generation_gradient_accumulation_pair_cycle": 7,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    require(not mismatches, "v13_protocol_differs fields=" + ",".join(mismatches))


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
        "v13_warm_lineage_schema_or_mode_differs",
    )
    require(
        require_sha256(recorded, description="v13_warm_receipt_hash")
        == canonical_sha256(unsigned),
        "v13_warm_lineage_self_hash_differs",
    )
    source_lock = lineage.get("source_lock")
    require(isinstance(source_lock, Mapping), "v13_warm_source_lock_missing")
    require(
        lineage.get("source_checkpoint") == warm["warm_start_checkpoint"]
        and source_lock.get("path") == warm["warm_start_lock"]
        and source_lock.get("lock_sha256") == warm["warm_start_lock_sha256"]
        and lineage.get("source_state_imports") == SOURCE_IMPORT_POLICY
        and lineage.get("post_load_bit_equal") is True,
        "v13_warm_source_binding_differs",
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
            "rng_state": "fresh_from_v13_seed",
        },
        "v13_warm_fresh_start_differs",
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
    require(not mismatches, "v13_warm_target_evidence_differs fields=" + ",".join(mismatches))
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        isinstance(optimizer_class, str) and optimizer_class.endswith(".AdamW"),
        "v13_warm_optimizer_class_differs",
    )
    return str(recorded)


_V13_ROW_AUDIT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "memory_objective_version",
        "checkpoint_optimizer_step",
        "completed_pair_presentations",
        "phases",
        "pair_schedule",
        "pair_presentations",
        "rows",
    }
)
_V13_ROW_AUDIT_OBSERVATION_FIELDS = frozenset(
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
        "dense_decision_token_count",
        "dense_decision_ce",
        "dense_decision_top1_retention_hinge",
        "dense_decision_gold_vs_top_competitor_margin",
        "dense_decision_top1_fraction",
        "dense_top1_margin",
        "selected_top_competitor_hinge",
        "selected_correct_vs_zero_hinge",
        "selected_zero_minus_correct_nll",
        "selected_top1",
        "failed_semantic_repair_applied",
        "failed_semantic_repair_ce",
        "failed_semantic_repair_competitor_hinge",
        "failed_semantic_repair_loss",
        "failed_semantic_gold_vs_competitor_margin",
        "failed_semantic_decision_ordinal",
        "failed_semantic_label_position",
        "failed_semantic_gold_token_id",
        "failed_semantic_competitor_id",
        "failed_semantic_competitor_is_actual_greedy",
        "failed_semantic_replay_generated_cursor",
        "failed_semantic_alignment_kind_code",
        "failed_semantic_is_termination",
        "dense_teacher_loss",
        "total_side_loss",
    }
)
_V13_PAIR_PRESENTATION_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "source_row_ordinal",
        "donor_row_ordinal",
        "source_row_sha256",
        "donor_row_sha256",
        "pair_mean_dense_decision_ce",
        "pair_mean_dense_decision_top1_retention_hinge",
        "pair_mean_selected_top_competitor_hinge",
        "pair_mean_selected_correct_vs_zero_hinge",
        "pair_mean_failed_semantic_repair_applied_fraction",
        "pair_mean_failed_semantic_ce",
        "pair_mean_failed_semantic_competitor_hinge",
        "pair_mean_failed_semantic_repair_loss",
        "pair_mean_dense_teacher_loss",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    }
)


def _v13_row_audit_sha256(value: Any, *, description: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{description}_must_be_lowercase_sha256",
    )
    return value


def _v13_row_audit_expected_observations(
    data: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    presentation_count = checkpoint_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    entries = data.get("entries")
    require(
        isinstance(entries, list) and len(entries) >= presentation_count,
        "v13_row_objective_audit_data_entries_differ",
    )
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for presentation_index, pair in enumerate(FOUR_CYCLE_PAIRS[:presentation_count]):
        entry = entries[presentation_index]
        require(
            isinstance(entry, Mapping)
            and entry.get("canonical_pair_ordinals") == list(pair),
            "v13_row_objective_audit_data_pair_differs",
        )
        members = entry.get("members")
        require(
            isinstance(members, list)
            and len(members) == PAIR_LOGICAL_BATCH_SIZE
            and all(isinstance(member, Mapping) for member in members),
            "v13_row_objective_audit_data_members_differ",
        )
        member_hashes: list[tuple[str, str]] = []
        for member_index, member in enumerate(members):
            row_ordinal = pair[member_index]
            paired_row_ordinal = pair[1 - member_index]
            require(
                type(member.get("train_row_ordinal")) is int
                and member.get("train_row_ordinal") == row_ordinal
                and type(member.get("donor_train_row_ordinal")) is int
                and member.get("donor_train_row_ordinal") == paired_row_ordinal,
                "v13_row_objective_audit_data_member_ordinals_differ",
            )
            row_sha256 = _v13_row_audit_sha256(
                member.get("row_sha256"),
                description="v13_row_objective_audit_data_row_hash",
            )
            paired_row_sha256 = _v13_row_audit_sha256(
                member.get("donor_row_sha256"),
                description="v13_row_objective_audit_data_paired_hash",
            )
            member_hashes.append((row_sha256, paired_row_sha256))
            phase_index = presentation_index // PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
            phase = f"cycle{phase_index + 1}_input"
            identity = (phase, row_ordinal)
            require(
                identity not in expected,
                "v13_row_objective_audit_data_row_phase_duplicate",
            )
            expected[identity] = {
                "phase": phase,
                "cycle": phase_index + 1,
                "adapter_optimizer_step_before_update": phase_index,
                "presentation": presentation_index + 1,
                "pair_role": "source" if member_index == 0 else "donor",
                "row_ordinal": row_ordinal,
                "paired_row_ordinal": paired_row_ordinal,
                "row_sha256": row_sha256,
                "paired_row_sha256": paired_row_sha256,
            }
        require(
            member_hashes[0] == tuple(reversed(member_hashes[1])),
            "v13_row_objective_audit_data_member_hashes_not_reciprocal",
        )
    return expected


def _validate_v13_row_objective_observation(
    observation: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
) -> None:
    require(
        set(observation) == _V13_ROW_AUDIT_OBSERVATION_FIELDS,
        "v13_row_objective_audit_observation_fields_differ",
    )
    integer_fields = (
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "row_ordinal",
        "paired_row_ordinal",
        "first_divergence",
        "rollout_token_count",
        "dense_decision_token_count",
        "failed_semantic_decision_ordinal",
        "failed_semantic_label_position",
        "failed_semantic_gold_token_id",
        "failed_semantic_competitor_id",
        "failed_semantic_replay_generated_cursor",
        "failed_semantic_alignment_kind_code",
    )
    require(
        all(type(observation[field]) is int for field in integer_fields),
        "v13_row_objective_audit_integer_field_type_differs",
    )
    boolean_fields = (
        "parsed_boundary_exact",
        "raw_token_exact",
        "selected_top1",
        "failed_semantic_repair_applied",
        "failed_semantic_competitor_is_actual_greedy",
        "failed_semantic_is_termination",
    )
    require(
        all(type(observation[field]) is bool for field in boolean_fields),
        "v13_row_objective_audit_boolean_field_type_differs",
    )
    float_fields = (
        "dense_decision_ce",
        "dense_decision_top1_retention_hinge",
        "dense_decision_gold_vs_top_competitor_margin",
        "dense_decision_top1_fraction",
        "dense_top1_margin",
        "selected_top_competitor_hinge",
        "selected_correct_vs_zero_hinge",
        "selected_zero_minus_correct_nll",
        "failed_semantic_repair_ce",
        "failed_semantic_repair_competitor_hinge",
        "failed_semantic_repair_loss",
        "failed_semantic_gold_vs_competitor_margin",
        "dense_teacher_loss",
        "total_side_loss",
    )
    require(
        all(
            type(observation[field]) is float
            and math.isfinite(observation[field])
            for field in float_fields
        ),
        "v13_row_objective_audit_float_field_type_or_value_differs",
    )
    require(
        type(observation["phase"]) is str
        and type(observation["pair_role"]) is str
        and all(
            observation[field] == value
            for field, value in expected_identity.items()
        ),
        "v13_row_objective_audit_observation_identity_differs",
    )
    row_sha256 = _v13_row_audit_sha256(
        observation["row_sha256"],
        description="v13_row_objective_audit_row_hash",
    )
    paired_row_sha256 = _v13_row_audit_sha256(
        observation["paired_row_sha256"],
        description="v13_row_objective_audit_paired_hash",
    )
    require(
        row_sha256 != paired_row_sha256,
        "v13_row_objective_audit_row_hashes_not_distinct",
    )

    parsed_exact = observation["parsed_boundary_exact"]
    raw_exact = observation["raw_token_exact"]
    first_divergence = observation["first_divergence"]
    rollout_count = observation["rollout_token_count"]
    dense_token_count = observation["dense_decision_token_count"]
    require(
        1 <= rollout_count <= GENERATED_ROLLOUT_MAX_TOKENS
        and 0 <= first_divergence <= rollout_count
        and dense_token_count > 0,
        "v13_row_objective_audit_integer_domain_differs",
    )
    require(
        not raw_exact or (parsed_exact and first_divergence == rollout_count),
        "v13_row_objective_audit_raw_exactness_differs",
    )
    dense_ce = observation["dense_decision_ce"]
    dense_hinge = observation["dense_decision_top1_retention_hinge"]
    dense_top1 = observation["dense_decision_top1_fraction"]
    selected_hinge = observation["selected_top_competitor_hinge"]
    zero_hinge = observation["selected_correct_vs_zero_hinge"]
    zero_gap = observation["selected_zero_minus_correct_nll"]
    repair_applied = observation["failed_semantic_repair_applied"]
    repair_ce = observation["failed_semantic_repair_ce"]
    repair_hinge = observation["failed_semantic_repair_competitor_hinge"]
    repair_loss = observation["failed_semantic_repair_loss"]
    repair_margin = observation["failed_semantic_gold_vs_competitor_margin"]
    dense_teacher = observation["dense_teacher_loss"]
    total_side = observation["total_side_loss"]
    require(
        dense_ce >= 0.0
        and dense_hinge >= 0.0
        and 0.0 <= dense_top1 <= 1.0
        and observation["dense_top1_margin"] == SEMANTIC_MARGIN
        and selected_hinge >= 0.0
        and zero_hinge >= 0.0
        and repair_ce >= 0.0
        and repair_hinge >= 0.0
        and repair_loss >= 0.0
        and dense_teacher >= 0.0
        and total_side >= 0.0
        and math.isclose(
            zero_hinge,
            max(0.0, 0.2 - zero_gap),
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
        and math.isclose(
            dense_teacher,
            dense_ce + dense_hinge + selected_hinge + zero_hinge,
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
        and math.isclose(
            total_side,
            dense_teacher + repair_loss,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ),
        "v13_row_objective_audit_loss_arithmetic_differs",
    )
    require(
        repair_applied is (not parsed_exact)
        and observation["failed_semantic_is_termination"] is False,
        "v13_row_objective_audit_repair_branch_differs",
    )
    repair_sentinels = (
        observation["failed_semantic_decision_ordinal"],
        observation["failed_semantic_label_position"],
        observation["failed_semantic_gold_token_id"],
        observation["failed_semantic_competitor_id"],
        observation["failed_semantic_replay_generated_cursor"],
        observation["failed_semantic_alignment_kind_code"],
    )
    if parsed_exact:
        require(
            repair_sentinels == (-1, -1, -1, -1, -1, -1)
            and repair_ce == 0.0
            and repair_hinge == 0.0
            and repair_loss == 0.0
            and repair_margin == 0.0
            and observation[
                "failed_semantic_competitor_is_actual_greedy"
            ] is False,
            "v13_row_objective_audit_exact_branch_differs",
        )
    else:
        require(
            repair_sentinels[0] >= 0
            and repair_sentinels[1] > 0
            and repair_sentinels[2] >= 0
            and repair_sentinels[3] >= 0
            and repair_sentinels[2] != repair_sentinels[3]
            and 0 <= repair_sentinels[4] < rollout_count
            and repair_sentinels[5] in {0, 1, 2}
            and observation[
                "failed_semantic_competitor_is_actual_greedy"
            ] is True,
            "v13_row_objective_audit_failed_branch_differs",
        )
        require(
            math.isclose(
                repair_hinge,
                max(0.0, SEMANTIC_MARGIN - repair_margin),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            and math.isclose(
                repair_loss,
                repair_ce + repair_hinge,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ),
            "v13_row_objective_audit_failed_loss_arithmetic_differs",
        )


def _validate_v13_pair_presentation(
    presentation: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    source_observation: Mapping[str, Any],
    donor_observation: Mapping[str, Any],
) -> None:
    require(
        set(presentation) == _V13_PAIR_PRESENTATION_FIELDS,
        "v13_row_objective_audit_pair_fields_differ",
    )
    integer_fields = (
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "source_row_ordinal",
        "donor_row_ordinal",
    )
    float_fields = tuple(
        field
        for field in _V13_PAIR_PRESENTATION_FIELDS
        if field.startswith("pair_mean_")
        or field in {
            "reported_objective_total_loss",
            "recomputed_objective_total_loss",
        }
    )
    require(
        type(presentation.get("phase")) is str
        and all(type(presentation.get(field)) is int for field in integer_fields)
        and all(
            type(presentation.get(field)) is float
            and math.isfinite(presentation[field])
            for field in float_fields
        )
        and all(
            presentation.get(field) == value
            for field, value in expected_identity.items()
        ),
        "v13_row_objective_audit_pair_identity_or_type_differs",
    )
    source_hash = _v13_row_audit_sha256(
        presentation.get("source_row_sha256"),
        description="v13_row_objective_audit_pair_source_hash",
    )
    donor_hash = _v13_row_audit_sha256(
        presentation.get("donor_row_sha256"),
        description="v13_row_objective_audit_pair_donor_hash",
    )
    require(
        source_hash != donor_hash
        and source_observation["row_sha256"] == source_hash
        and donor_observation["row_sha256"] == donor_hash
        and source_observation["paired_row_sha256"] == donor_hash
        and donor_observation["paired_row_sha256"] == source_hash,
        "v13_row_objective_audit_pair_hash_binding_differs",
    )

    row_mean_fields = {
        "pair_mean_dense_decision_ce": "dense_decision_ce",
        "pair_mean_dense_decision_top1_retention_hinge": (
            "dense_decision_top1_retention_hinge"
        ),
        "pair_mean_selected_top_competitor_hinge": (
            "selected_top_competitor_hinge"
        ),
        "pair_mean_selected_correct_vs_zero_hinge": (
            "selected_correct_vs_zero_hinge"
        ),
        "pair_mean_failed_semantic_ce": "failed_semantic_repair_ce",
        "pair_mean_failed_semantic_competitor_hinge": (
            "failed_semantic_repair_competitor_hinge"
        ),
        "pair_mean_failed_semantic_repair_loss": (
            "failed_semantic_repair_loss"
        ),
        "pair_mean_dense_teacher_loss": "dense_teacher_loss",
        "pair_mean_total_side_loss": "total_side_loss",
    }
    for pair_field, row_field in row_mean_fields.items():
        expected_mean = 0.5 * (
            float(source_observation[row_field])
            + float(donor_observation[row_field])
        )
        require(
            math.isclose(
                presentation[pair_field],
                expected_mean,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ),
            f"v13_row_objective_audit_pair_arithmetic_differs field={pair_field}",
        )
    repair_fraction = 0.5 * (
        int(source_observation["failed_semantic_repair_applied"])
        + int(donor_observation["failed_semantic_repair_applied"])
    )
    require(
        0.0
        <= presentation["pair_mean_failed_semantic_repair_applied_fraction"]
        <= 1.0
        and math.isclose(
            presentation["pair_mean_failed_semantic_repair_applied_fraction"],
            repair_fraction,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ),
        "v13_row_objective_audit_pair_repair_fraction_differs",
    )

    teacher = (
        presentation["pair_mean_dense_decision_ce"]
        + presentation["pair_mean_dense_decision_top1_retention_hinge"]
        + presentation["pair_mean_selected_top_competitor_hinge"]
        + presentation["pair_mean_selected_correct_vs_zero_hinge"]
    )
    repair = (
        presentation["pair_mean_failed_semantic_ce"]
        + presentation["pair_mean_failed_semantic_competitor_hinge"]
    )
    total = teacher + repair
    arithmetic = (
        (presentation["pair_mean_dense_teacher_loss"], teacher),
        (presentation["pair_mean_failed_semantic_repair_loss"], repair),
        (presentation["pair_mean_total_side_loss"], total),
        (presentation["reported_objective_total_loss"], total),
        (presentation["recomputed_objective_total_loss"], total),
    )
    require(
        all(actual >= 0.0 for actual, _ in arithmetic)
        and all(
            math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
            for actual, expected in arithmetic
        ),
        "v13_row_objective_audit_pair_loss_arithmetic_differs",
    )


def _validate_v13_row_objective_audit(
    payload: Mapping[str, Any],
    *,
    checkpoint_step: int,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        type(checkpoint_step) is int
        and checkpoint_step in CHECKPOINT_STEPS
        and set(payload) == _V13_ROW_AUDIT_TOP_LEVEL_FIELDS,
        "v13_row_objective_audit_top_level_fields_differ",
    )
    expected_phases = [
        f"cycle{cycle_index}_input"
        for cycle_index in range(1, checkpoint_step + 1)
    ]
    require(
        payload.get("schema") == ROW_OBJECTIVE_AUDIT_SCHEMA
        and payload.get("memory_objective_version") == OBJECTIVE_VERSION
        and type(payload.get("checkpoint_optimizer_step")) is int
        and payload.get("checkpoint_optimizer_step") == checkpoint_step
        and type(payload.get("completed_pair_presentations")) is int
        and payload.get("completed_pair_presentations")
        == checkpoint_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
        and payload.get("phases") == expected_phases,
        "v13_row_objective_audit_identity_differs",
    )
    expected_schedule = [
        {"source_row_ordinal": low, "donor_row_ordinal": high}
        for low, high in FOUR_CYCLE_PAIRS[: checkpoint_step * 7]
    ]
    require(
        payload.get("pair_schedule") == expected_schedule,
        "v13_row_objective_audit_schedule_differs",
    )
    for actual, expected in zip(payload["pair_schedule"], expected_schedule):
        require(
            isinstance(actual, Mapping)
            and set(actual) == {"source_row_ordinal", "donor_row_ordinal"}
            and type(actual["source_row_ordinal"]) is int
            and type(actual["donor_row_ordinal"]) is int
            and dict(actual) == expected,
            "v13_row_objective_audit_schedule_entry_differs",
        )
    rows = payload.get("rows")
    expected_row_order = [
        ordinal for pair in FIRST_CYCLE_PAIRS for ordinal in pair
    ]
    require(
        isinstance(rows, list)
        and len(rows) == len(expected_row_order)
        and [
            row.get("row_ordinal") if isinstance(row, Mapping) else None
            for row in rows
        ]
        == expected_row_order,
        "v13_row_objective_audit_rows_differ",
    )
    expected_observations = _v13_row_audit_expected_observations(
        data,
        checkpoint_step=checkpoint_step,
    )
    expected_ordinals = frozenset(expected_row_order)
    observed_ordinals: set[int] = set()
    observed_by_identity: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "v13_row_objective_audit_row_invalid")
        require(
            set(row) == {"row_ordinal", *expected_phases},
            "v13_row_objective_audit_row_fields_differ",
        )
        ordinal = row.get("row_ordinal")
        require(
            type(ordinal) is int
            and ordinal not in observed_ordinals,
            "v13_row_objective_audit_row_ordinal_differs",
        )
        observed_ordinals.add(ordinal)
        for phase_index, phase in enumerate(expected_phases):
            observation = row.get(phase)
            require(
                isinstance(observation, Mapping),
                "v13_row_objective_audit_phase_differs",
            )
            expected_identity = expected_observations.get((phase, ordinal))
            require(
                expected_identity is not None
                and expected_identity["adapter_optimizer_step_before_update"]
                == phase_index,
                "v13_row_objective_audit_expected_phase_missing",
            )
            _validate_v13_row_objective_observation(
                observation,
                expected_identity=expected_identity,
            )
            observed_by_identity[(phase, ordinal)] = observation
    require(
        observed_ordinals == expected_ordinals,
        "v13_row_objective_audit_ordinal_set_differs",
    )
    pair_presentations = payload.get("pair_presentations")
    presentation_count = checkpoint_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    require(
        isinstance(pair_presentations, list)
        and len(pair_presentations) == presentation_count,
            "v13_row_objective_audit_pair_presentations_count_differs",
    )
    for presentation_index, expected_pair in enumerate(
        FOUR_CYCLE_PAIRS[:presentation_count]
    ):
        pair_presentation = pair_presentations[presentation_index]
        require(
            isinstance(pair_presentation, Mapping),
                "v13_row_objective_audit_pair_presentation_invalid",
        )
        cycle_index = (
            presentation_index // PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
        ) + 1
        phase = f"cycle{cycle_index}_input"
        source_observation = observed_by_identity.get(
            (phase, expected_pair[0])
        )
        donor_observation = observed_by_identity.get(
            (phase, expected_pair[1])
        )
        require(
            source_observation is not None and donor_observation is not None,
                "v13_row_objective_audit_pair_rows_missing",
        )
        _validate_v13_pair_presentation(
            pair_presentation,
            expected_identity={
                "phase": phase,
                "cycle": cycle_index,
                "adapter_optimizer_step_before_update": cycle_index - 1,
                "presentation": presentation_index + 1,
                "source_row_ordinal": expected_pair[0],
                "donor_row_ordinal": expected_pair[1],
                "source_row_sha256": source_observation["row_sha256"],
                "donor_row_sha256": donor_observation["row_sha256"],
            },
            source_observation=source_observation,
            donor_observation=donor_observation,
        )
    return dict(payload)


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved = require_v13_run_path(
        checkpoint,
        description="v13_checkpoint",
        ssd_root=ssd_root,
    )
    require(resolved.is_dir(), "v13_checkpoint_missing")
    try:
        checkpoint_step = int(resolved.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise LaunchContractError("v13_checkpoint_name_differs") from exc
    require(
        resolved.name == f"checkpoint-{checkpoint_step}"
        and checkpoint_step in CHECKPOINT_STEPS,
        "v13_checkpoint_step_is_not_1_through_4",
    )
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / filename, description=f"v13_{filename}")
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            for path in rng_files
        ),
        "v13_rng_state_missing",
    )
    trainer_state = _load_object(
        resolved / "trainer_state.json",
        description="v13_trainer_state",
    )
    protocol = _load_object(
        resolved / "training_protocol.json",
        description="v13_protocol",
    )
    config = _load_object(
        resolved / "delta_mem_config.json",
        description="v13_config",
    )
    pairing = _load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="v13_pairing",
    )
    require(
        trainer_state.get("global_step") == checkpoint_step
        and trainer_state.get("max_steps") == TOTAL_OPTIMIZER_STEPS
        and protocol.get("max_steps") == TOTAL_OPTIMIZER_STEPS,
        "v13_checkpoint_horizon_differs",
    )
    cycle_pair_telemetry = validate_v13_cycle_pair_telemetry(
        trainer_state,
        checkpoint_step=checkpoint_step,
    )
    audit = _validate_v13_row_objective_audit(
        _load_object(
            resolved / ROW_OBJECTIVE_AUDIT_FILENAME,
            description="v13_row_objective_audit",
        ),
        checkpoint_step=checkpoint_step,
        data=data,
    )
    _validate_checkpoint_protocol(protocol, data=data)
    try:
        v10.v9._validate_checkpoint_config(config)
    except Exception as exc:
        raise LaunchContractError(f"v13_delta_config_failed: {exc}") from exc
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
        "v13_requires_exactly_one_fresh_warm_start_lineage",
    )
    lineage_path = resolved / WARM_START_LINEAGE_FILENAME
    lineage = _load_object(lineage_path, description="v13_lineage")
    root_receipt = _validate_warm_lineage(
        lineage,
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        warm=warm,
    )
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": checkpoint_step,
        "consumed_pair_presentations": cycle_pair_telemetry[
            "pair_presentations"
        ],
        "cycle_pair_telemetry": cycle_pair_telemetry,
        "row_objective_audit": audit,
        "row_objective_audit_file_sha256": sha256_file(
            resolved / ROW_OBJECTIVE_AUDIT_FILENAME
        ),
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
        "v13_critical_code_requires_exact_project_root",
    )
    bindings: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_TRAINING_FILES:
        path = resolved_root / relative
        require(
            path.is_file() and not path.is_symlink(),
            f"v13_critical_file_missing_or_symlink path={relative}",
        )
        bindings[relative] = artifact_binding(
            path,
            description=f"v13_critical_{relative}",
        )
    return bindings


def require_git_object_id(value: object, *, description: str) -> str:
    require(
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value),
        f"{description}_invalid_git_object_id",
    )
    return value


def _resolved_project_root(project_root: Path, *, description: str) -> Path:
    resolved = project_root.expanduser().resolve()
    require(
        resolved == PROJECT_ROOT,
        f"{description}_requires_exact_project_root",
    )
    return resolved


def _git_output(
    project_root: Path,
    arguments: Sequence[str],
    *,
    description: str,
) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchContractError(f"{description}_git_command_failed") from exc


def _resolve_git_commit(
    object_id: object,
    *,
    project_root: Path = PROJECT_ROOT,
    description: str,
) -> str:
    candidate = require_git_object_id(object_id, description=description)
    resolved_root = _resolved_project_root(project_root, description=description)
    resolved_raw = _git_output(
        resolved_root,
        ["rev-parse", "--verify", f"{candidate}^{{commit}}"],
        description=description,
    )
    try:
        resolved = resolved_raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LaunchContractError(f"{description}_non_ascii_git_output") from exc
    resolved = require_git_object_id(resolved, description=description)
    require(resolved == candidate, f"{description}_does_not_name_commit_exactly")
    return resolved


def _require_git_ancestor(
    ancestor: object,
    descendant: object,
    *,
    project_root: Path = PROJECT_ROOT,
    description: str,
) -> None:
    resolved_root = _resolved_project_root(project_root, description=description)
    ancestor_commit = _resolve_git_commit(
        ancestor,
        project_root=resolved_root,
        description=f"{description}_ancestor",
    )
    descendant_commit = _resolve_git_commit(
        descendant,
        project_root=resolved_root,
        description=f"{description}_descendant",
    )
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_root),
                "merge-base",
                "--is-ancestor",
                ancestor_commit,
                descendant_commit,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise LaunchContractError(f"{description}_git_command_failed") from exc
    require(result.returncode in (0, 1), f"{description}_git_command_failed")
    require(result.returncode == 0, f"{description}_source_not_ancestor")


def _git_file_bindings_at_commit(
    commit: object,
    relative_files: Sequence[str],
    *,
    project_root: Path = PROJECT_ROOT,
    description: str,
) -> dict[str, dict[str, Any]]:
    resolved_root = _resolved_project_root(project_root, description=description)
    resolved_commit = _resolve_git_commit(
        commit,
        project_root=resolved_root,
        description=f"{description}_commit",
    )
    bindings: dict[str, dict[str, Any]] = {}
    for relative in relative_files:
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.as_posix() == relative,
            f"{description}_invalid_relative_path path={relative}",
        )
        tree_output = _git_output(
            resolved_root,
            ["ls-tree", "-z", "--full-tree", resolved_commit, "--", relative],
            description=f"{description}_{relative}",
        )
        records = [record for record in tree_output.split(b"\0") if record]
        require(
            len(records) == 1 and b"\t" in records[0],
            f"{description}_tree_entry_missing path={relative}",
        )
        metadata, recorded_path = records[0].split(b"\t", 1)
        try:
            mode, object_type, blob_object_id = metadata.decode("ascii").split()
            decoded_path = recorded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise LaunchContractError(
                f"{description}_tree_entry_invalid path={relative}"
            ) from exc
        require(
            decoded_path == relative
            and mode in ("100644", "100755")
            and object_type == "blob",
            f"{description}_tree_entry_not_regular_file path={relative}",
        )
        blob_object_id = require_git_object_id(
            blob_object_id,
            description=f"{description}_{relative}_blob",
        )
        blob = _git_output(
            resolved_root,
            ["cat-file", "blob", blob_object_id],
            description=f"{description}_{relative}_blob",
        )
        bindings[relative] = {
            "path": str(resolved_root / relative),
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
    return bindings


def critical_training_code_bindings_at_commit(
    commit: object,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    return _git_file_bindings_at_commit(
        commit,
        CRITICAL_TRAINING_FILES,
        project_root=project_root,
        description="v13_critical_code_at_commit",
    )


def _tracked_worktree_clean(project_root: Path = PROJECT_ROOT) -> bool:
    resolved_root = _resolved_project_root(
        project_root,
        description="v13_tracked_worktree",
    )
    output = _git_output(
        resolved_root,
        ["status", "--porcelain", "--untracked-files=no"],
        description="v13_tracked_worktree",
    )
    return not output.strip()


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
    resolved_root = _resolved_project_root(
        project_root,
        description="v13_git_head",
    )
    try:
        value = subprocess.check_output(
            ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchContractError("v13_git_head_unavailable") from exc
    return _resolve_git_commit(
        value,
        project_root=resolved_root,
        description="v13_git_head",
    )


def validate_launch_receipt(
    launch_receipt: Path,
    *,
    checkpoint: Path,
    baseline: Mapping[str, Any],
    base_model_identity: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    resolved_checkpoint = require_v13_run_path(
        checkpoint,
        description="v13_provenance_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v13_run_path(
        launch_receipt,
        description="v13_launch_receipt",
        ssd_root=ssd_root,
    )
    _regular_file(path, description="v13_launch_receipt")
    require(
        path.parent == v13_run_root_for(ssd_root) / "logs",
        "v13_launch_receipt_must_be_under_logs",
    )
    payload = _load_object(path, description="v13_launch_receipt")
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v13_launch_receipt",
    )
    require(
        payload.get("schema") == LAUNCH_RECEIPT_SCHEMA
        and payload.get("attached_foreground_execution") is True
        and payload.get("launch_mode") == "warm_start"
        and payload.get("source_step") == 0
        and payload.get("target_step") == TOTAL_OPTIMIZER_STEPS
        and payload.get("resume_checkpoint") is None,
        "v13_launch_receipt_mode_or_horizon_differs",
    )
    expected_output = resolved_checkpoint.parents[1]
    expected_launch_path = (
        v13_run_root_for(ssd_root)
        / "logs"
        / f"{expected_output.name}.launch.json"
    )
    expected_log_path = expected_launch_path.with_name(
        f"{expected_output.name}.log"
    )
    require(
        payload.get("trainer_output") == str(expected_output)
        and payload.get("checkpoints")
        == {
            f"checkpoint-{step}": str(
                expected_output / f"trainer/checkpoint-{step}"
            )
            for step in CHECKPOINT_STEPS
        }
        and str(resolved_checkpoint) in payload["checkpoints"].values()
        and payload.get("log_file") == str(expected_log_path)
        and path == expected_launch_path,
        "v13_launch_receipt_checkpoint_binding_differs",
    )
    require(
        payload.get("objective") == OBJECTIVE_VERSION
        and payload.get("gradient_accumulation_steps") == 7
        and payload.get("max_grad_norm") == 1.0
        and payload.get("max_steps") == TOTAL_OPTIMIZER_STEPS
        and payload.get("learning_rate") == LEARNING_RATE
        and payload.get("lr_scheduler_type") == "constant"
        and payload.get("warmup_steps") == 0
        and payload.get("save_total_limit") == len(CHECKPOINT_STEPS)
        and payload.get("four_cycle_pairs")
        == [list(pair) for pair in FOUR_CYCLE_PAIRS]
        and payload.get("four_cycle_pairs_sha256") == FOUR_CYCLE_PAIRS_SHA256,
        "v13_launch_receipt_objective_or_cycle_differs",
    )
    require(
        payload.get("warm_start_checkpoint") == str(PINNED_WARM_START_CHECKPOINT)
        and payload.get("v10_diagnostic_baseline") == baseline
        and payload.get("base_model_identity") == base_model_identity,
        "v13_launch_receipt_source_or_model_identity_differs",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v13_launch_receipt_authorization_differs",
    )
    source_commit = _resolve_git_commit(
        payload.get("git_commit"),
        project_root=project_root,
        description="v13_launch_receipt_git_commit",
    )
    _require_git_ancestor(
        source_commit,
        _git_head(project_root),
        project_root=project_root,
        description="v13_launch_receipt_lineage",
    )
    require(
        payload.get("tracked_worktree_clean") is True,
        "v13_launch_receipt_git_identity_differs",
    )
    expected_code = critical_training_code_bindings_at_commit(
        source_commit,
        project_root=project_root,
    )
    require(
        payload.get("critical_files") == expected_code,
        "v13_launch_receipt_critical_file_hashes_differ",
    )
    return {
        "artifact": artifact_binding(path, description="v13_launch_receipt"),
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
        "v13_completion_launch_artifact_missing",
    )
    launch_path = require_v13_run_path(
        Path(str(launch_artifact.get("path", ""))),
        description="v13_completion_launch_receipt_path",
        ssd_root=ssd_root,
    )
    require(
        launch_path.parent == v13_run_root_for(ssd_root) / "logs"
        and launch_path.name.endswith(".launch.json"),
        "v13_completion_launch_receipt_path_differs",
    )
    return launch_path.with_name(
        launch_path.name.removesuffix(".launch.json") + ".completion.json"
    )


def _validate_completion_recovery(
    payload: Mapping[str, Any],
    *,
    launch: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any] | None:
    schema = payload.get("schema")
    recovery = payload.get("recovery")
    if schema == COMPLETION_RECEIPT_SCHEMA:
        require(
            "recovery" not in payload and "recovered_at" not in payload,
            "v13_attached_completion_has_recovery_metadata",
        )
        return None
    require(
        schema == RECOVERED_COMPLETION_RECEIPT_SCHEMA
        and isinstance(recovery, Mapping),
        "v13_completion_recovery_schema_differs",
    )
    expected_fields = {
        "schema",
        "reason",
        "source_git_commit",
        "validator_git_commit",
        "validator_files",
    }
    require(
        set(recovery) == expected_fields
        and recovery.get("schema") == COMPLETION_RECOVERY_SCHEMA
        and recovery.get("reason") == COMPLETION_RECOVERY_REASON,
        "v13_completion_recovery_metadata_differs",
    )
    recovered_at = payload.get("recovered_at")
    require(
        isinstance(recovered_at, str) and "completed_at" not in payload,
        "v13_completion_recovery_time_invalid",
    )
    _recovery_timestamp(recovered_at)
    launch_payload = launch.get("payload")
    require(
        isinstance(launch_payload, Mapping),
        "v13_completion_recovery_launch_payload_missing",
    )
    source_commit = _resolve_git_commit(
        recovery.get("source_git_commit"),
        project_root=project_root,
        description="v13_completion_recovery_source_commit",
    )
    require(
        source_commit == launch_payload.get("git_commit"),
        "v13_completion_recovery_source_commit_differs",
    )
    validator_commit = _resolve_git_commit(
        recovery.get("validator_git_commit"),
        project_root=project_root,
        description="v13_completion_recovery_validator_commit",
    )
    _require_git_ancestor(
        source_commit,
        validator_commit,
        project_root=project_root,
        description="v13_completion_recovery_lineage",
    )
    _require_git_ancestor(
        validator_commit,
        _git_head(project_root),
        project_root=project_root,
        description="v13_completion_recovery_validator_lineage",
    )
    expected_validator_files = _git_file_bindings_at_commit(
        validator_commit,
        RECOVERY_VALIDATOR_FILES,
        project_root=project_root,
        description="v13_completion_recovery_validator_files",
    )
    require(
        recovery.get("validator_files") == expected_validator_files,
        "v13_completion_recovery_validator_files_differ",
    )
    return dict(recovery)


def validate_completion_receipt(
    completion_receipt: Path,
    *,
    checkpoint: Path,
    checkpoint_contract: Mapping[str, Any],
    launch: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
    _payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_checkpoint = require_v13_run_path(
        checkpoint,
        description="v13_completion_checkpoint",
        ssd_root=ssd_root,
    )
    path = require_v13_run_path(
        completion_receipt,
        description="v13_completion_receipt",
        ssd_root=ssd_root,
    )
    require(
        path == _expected_completion_receipt_path(launch, ssd_root=ssd_root),
        "v13_completion_receipt_path_differs",
    )
    if _payload is None:
        _regular_file(path, description="v13_completion_receipt")
        payload = _load_object(path, description="v13_completion_receipt")
    else:
        require(
            not path.exists() and isinstance(_payload, Mapping),
            "v13_completion_preview_requires_absent_receipt",
        )
        payload = dict(_payload)
    receipt_sha256 = _validate_receipt_self_hash(
        payload,
        description="v13_completion_receipt",
    )
    require(
        payload.get("status") == "completed"
        and payload.get("optimizer_step") == TOTAL_OPTIMIZER_STEPS
        and payload.get("consumed_pair_presentations") == TOTAL_PAIR_PRESENTATIONS,
        "v13_completion_receipt_horizon_differs",
    )
    recovery = _validate_completion_recovery(
        payload,
        launch=launch,
        project_root=project_root,
    )
    _validate_exact_artifact_binding(
        payload.get("launch_receipt", {}),
        expected_path=Path(str(launch["artifact"]["path"])),
        description="v13_completion_launch_receipt",
    )
    require(
        payload.get("launch_receipt_sha256") == launch["receipt_sha256"],
        "v13_completion_launch_receipt_self_hash_differs",
    )
    checkpoint_receipts = payload.get("checkpoints")
    require(
        isinstance(checkpoint_receipts, Mapping)
        and set(checkpoint_receipts)
        == {f"checkpoint-{step}" for step in CHECKPOINT_STEPS},
        "v13_completion_checkpoint_set_differs",
    )
    trainer_root = resolved_checkpoint.parent
    for step in CHECKPOINT_STEPS:
        name = f"checkpoint-{step}"
        bound_checkpoint = trainer_root / name
        receipt = checkpoint_receipts.get(name)
        require(
            isinstance(receipt, Mapping)
            and receipt.get("path") == str(bound_checkpoint)
            and receipt.get("optimizer_step") == step
            and receipt.get("consumed_pair_presentations") == step * 7,
            f"v13_completion_{name}_identity_differs",
        )
        expected_checkpoint_artifacts = {
            filename: artifact_binding(
                bound_checkpoint / filename,
                description=f"v13_completion_{name}_{filename}",
            )
            for filename in REQUIRED_CHECKPOINT_ARTIFACTS
        }
        expected_rng = {
            rng_path.name: artifact_binding(
                rng_path,
                description=f"v13_completion_{name}_{rng_path.name}",
            )
            for rng_path in sorted(bound_checkpoint.glob("rng_state*.pth"))
        }
        require(
            receipt.get("checkpoint_artifacts") == expected_checkpoint_artifacts
            and receipt.get("rng_state_artifacts") == expected_rng,
            f"v13_completion_{name}_artifacts_differ",
        )
    selected_receipt = checkpoint_receipts[resolved_checkpoint.name]
    require(
        selected_receipt.get("cycle_pair_telemetry")
        == checkpoint_contract["cycle_pair_telemetry"]
        and selected_receipt.get("row_objective_audit_file_sha256")
        == checkpoint_contract["row_objective_audit_file_sha256"],
        "v13_completion_selected_checkpoint_contract_differs",
    )
    training_summary_binding = payload.get("training_summary")
    require(
        isinstance(training_summary_binding, Mapping),
        "v13_completion_training_summary_missing",
    )
    expected_summary_path = resolved_checkpoint.parents[1] / "training_summary.json"
    summary_artifact = _validate_exact_artifact_binding(
        training_summary_binding,
        expected_path=expected_summary_path,
        description="v13_completion_training_summary",
    )
    summary = _load_object(
        expected_summary_path,
        description="v13_completion_training_summary",
    )
    require(
        summary.get("memory_objective_version") == OBJECTIVE_VERSION
        and summary.get("warm_start_mode") == WARM_START_MODE
        and summary.get("training_protocol_sha256")
        == checkpoint_contract["training_protocol_sha256"],
        "v13_completion_training_summary_identity_differs",
    )
    log_binding = payload.get("log")
    require(isinstance(log_binding, Mapping), "v13_completion_log_missing")
    launch_payload = launch.get("payload")
    require(
        isinstance(launch_payload, Mapping),
        "v13_completion_launch_payload_missing",
    )
    log_path = Path(str(launch_payload.get("log_file", "")))
    require(
        log_path.parent == v13_run_root_for(ssd_root) / "logs",
        "v13_completion_log_path_differs",
    )
    log_artifact = _validate_exact_artifact_binding(
        log_binding,
        expected_path=log_path,
        description="v13_completion_log",
    )
    require(
        payload.get("training_continuation") == TRAINING_CONTINUATION_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY
        and payload.get("evaluation_access") == "forbidden",
        "v13_completion_authorization_differs",
    )
    return {
        "artifact": (
            artifact_binding(path, description="v13_completion_receipt")
            if _payload is None
            else None
        ),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
        "training_summary": summary_artifact,
        "log": log_artifact,
        "recovery": recovery,
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
        project_root=project_root,
    )
    return {
        "schema": "rwkv_ms_scene_memory_v13_training_provenance.v1",
        "launch_receipt": {
            "artifact": launch["artifact"],
            "receipt_sha256": launch["receipt_sha256"],
        },
        "completion_receipt": {
            "artifact": completion["artifact"],
            "receipt_sha256": completion["receipt_sha256"],
        },
        "completion_recovery": completion["recovery"],
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


def _recovery_timestamp(value: object) -> str:
    require(isinstance(value, str), "v13_completion_recovery_time_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LaunchContractError("v13_completion_recovery_time_invalid") from exc
    require(
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0,
        "v13_completion_recovery_time_not_utc",
    )
    return value


def _path_matches_file_identity(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _unlink_if_matching_file_identity(
    path: Path,
    identity: tuple[int, int],
) -> bool:
    if not _path_matches_file_identity(path, identity):
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


def _atomic_create_json(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[int, int]:
    require(not path.exists(), "v13_completion_recovery_receipt_already_exists")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    linked = False
    identity: tuple[int, int] | None = None
    try:
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_fd = -1
        temporary_stat = temporary_path.stat()
        identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.link(temporary_path, path, follow_symlinks=False)
        linked = True
        _fsync_directory(path.parent)
        return identity
    except FileExistsError as exc:
        raise LaunchContractError(
            "v13_completion_recovery_receipt_already_exists"
        ) from exc
    except Exception:
        if linked and identity is not None:
            _unlink_if_matching_file_identity(path, identity)
        raise
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        temporary_path.unlink(missing_ok=True)


def recover_completion_receipt(
    *,
    completion_receipt: Path,
    launch_receipt: Path,
    log: Path,
    training_summary: Path,
    checkpoint: Path,
    recovered_at: str,
    expected_receipt_sha256: str | None = None,
    write: bool = False,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    raise LaunchContractError(
        "V13 completion recovery is unsupported; rerun the fresh four-cycle launch"
    )
    recovery_time = _recovery_timestamp(recovered_at)
    completion_path = require_v13_run_path(
        completion_receipt,
        description="v13_completion_recovery_receipt",
        ssd_root=ssd_root,
    )
    require(
        completion_path.parent == v13_run_root_for(ssd_root) / "logs"
        and not completion_path.exists(),
        "v13_completion_recovery_receipt_path_invalid_or_exists",
    )
    resolved_checkpoint = require_v13_run_path(
        checkpoint,
        description="v13_completion_recovery_checkpoint",
        ssd_root=ssd_root,
    )
    data = validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    warm = validate_warm_start_contract(
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    checkpoint_contract = validate_checkpoint_contract(
        resolved_checkpoint,
        data=data,
        warm=warm,
    )
    baseline = validate_v10_diagnostic_baseline(ssd_root=ssd_root)
    launch = validate_launch_receipt(
        launch_receipt,
        checkpoint=resolved_checkpoint,
        baseline=baseline,
        base_model_identity=baseline["base_model_identity"],
        ssd_root=ssd_root,
        project_root=project_root,
    )
    require(
        completion_path
        == _expected_completion_receipt_path(launch, ssd_root=ssd_root),
        "v13_completion_recovery_receipt_name_differs",
    )
    expected_log = Path(str(launch["payload"]["log_file"]))
    expected_summary = resolved_checkpoint.parents[1] / "training_summary.json"
    resolved_log = require_v13_run_path(
        log,
        description="v13_completion_recovery_log",
        ssd_root=ssd_root,
    )
    resolved_summary = require_v13_run_path(
        training_summary,
        description="v13_completion_recovery_training_summary",
        ssd_root=ssd_root,
    )
    require(
        resolved_log == expected_log and resolved_summary == expected_summary,
        "v13_completion_recovery_log_or_summary_path_differs",
    )
    require(
        _tracked_worktree_clean(project_root),
        "v13_completion_recovery_requires_clean_tracked_worktree",
    )
    source_commit = _resolve_git_commit(
        launch["payload"]["git_commit"],
        project_root=project_root,
        description="v13_completion_recovery_source_commit",
    )
    validator_commit = _git_head(project_root)
    _require_git_ancestor(
        source_commit,
        validator_commit,
        project_root=project_root,
        description="v13_completion_recovery_lineage",
    )
    validator_files = _git_file_bindings_at_commit(
        validator_commit,
        RECOVERY_VALIDATOR_FILES,
        project_root=project_root,
        description="v13_completion_recovery_validator_files",
    )
    rng = sorted(resolved_checkpoint.glob("rng_state*.pth"))
    payload: dict[str, Any] = {
        "schema": RECOVERED_COMPLETION_RECEIPT_SCHEMA,
        "recovered_at": recovery_time,
        "status": "completed",
        "optimizer_step": 1,
        "consumed_pair_presentations": checkpoint_contract[
            "consumed_pair_presentations"
        ],
        "launch_receipt": launch["artifact"],
        "launch_receipt_sha256": launch["receipt_sha256"],
        "log": artifact_binding(expected_log, description="v13_completion_log"),
        "training_summary": artifact_binding(
            expected_summary,
            description="v13_completion_training_summary",
        ),
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_artifacts": {
            name: artifact_binding(
                resolved_checkpoint / name,
                description=f"v13_completion_checkpoint_{name}",
            )
            for name in REQUIRED_CHECKPOINT_ARTIFACTS
        },
        "rng_state_artifacts": {
            path.name: artifact_binding(
                path,
                description=f"v13_completion_checkpoint_{path.name}",
            )
            for path in rng
        },
        "cycle_pair_telemetry": checkpoint_contract["cycle_pair_telemetry"],
        "training_continuation": TRAINING_CONTINUATION_POLICY,
        "hard32_access": HARD32_ACCESS_POLICY,
        "evaluation_access": "forbidden",
        "recovery": {
            "schema": COMPLETION_RECOVERY_SCHEMA,
            "reason": COMPLETION_RECOVERY_REASON,
            "source_git_commit": source_commit,
            "validator_git_commit": validator_commit,
            "validator_files": validator_files,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    preview = validate_completion_receipt(
        completion_path,
        checkpoint=resolved_checkpoint,
        checkpoint_contract=checkpoint_contract,
        launch=launch,
        ssd_root=ssd_root,
        project_root=project_root,
        _payload=payload,
    )
    receipt_sha256 = payload["receipt_sha256"]
    if expected_receipt_sha256 is not None:
        require(
            require_sha256(
                expected_receipt_sha256,
                description="v13_completion_recovery_expected_hash",
            )
            == receipt_sha256,
            "v13_completion_recovery_expected_hash_differs",
        )
    if not write:
        return {
            "written": False,
            "completion_receipt": str(completion_path),
            "receipt_sha256": receipt_sha256,
            "payload": payload,
            "source_git_commit": source_commit,
            "validator_git_commit": validator_commit,
            "checkpoint_contract": checkpoint_contract,
            "preview_validation": preview,
        }
    require(
        expected_receipt_sha256 is not None,
        "v13_completion_recovery_write_requires_preview_hash",
    )
    created_identity = _atomic_create_json(completion_path, payload)
    try:
        provenance = validate_training_provenance(
            checkpoint=resolved_checkpoint,
            checkpoint_contract=checkpoint_contract,
            launch_receipt=launch_receipt,
            completion_receipt=completion_path,
            baseline=baseline,
            base_model_identity=baseline["base_model_identity"],
            ssd_root=ssd_root,
            project_root=project_root,
        )
        require(
            _path_matches_file_identity(completion_path, created_identity),
            "v13_completion_recovery_receipt_replaced_after_create",
        )
    except Exception:
        _unlink_if_matching_file_identity(completion_path, created_identity)
        raise
    return {
        "written": True,
        "completion_receipt": str(completion_path),
        "receipt_sha256": receipt_sha256,
        "payload": payload,
        "source_git_commit": source_commit,
        "validator_git_commit": validator_commit,
        "checkpoint_contract": checkpoint_contract,
        "training_provenance": provenance,
    }


def validate_resume_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise LaunchContractError("V13 forbids resume; all cycles run in one fresh launch")


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
    require(
        target_step == TOTAL_OPTIMIZER_STEPS,
        "v13_target_step_must_be_four",
    )
    require(resume_checkpoint is None, "v13_resume_is_forbidden")
    require(gate_receipt is None, "v13_gate_receipt_cannot_authorize_training")
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
        "v13_v10_diagnostic_must_never_be_warm_start",
    )
    first = data["entries"][0]
    public_data = {key: value for key, value in data.items() if key != "entries"}
    public_warm = {key: value for key, value in warm.items() if key != "lock"}
    return {
        **public_data,
        **public_warm,
        "launch_mode": "warm_start_smoke" if smoke else "warm_start",
        "source_step": 0,
        "target_step": TOTAL_OPTIMIZER_STEPS,
        "resume_checkpoint": None,
        "resume_schedule_cursor": 0,
        "next_pair_low_ordinal": first["canonical_pair_ordinals"][0],
        "next_pair_high_ordinal": first["canonical_pair_ordinals"][1],
        "next_schedule_entry_sha256": first["entry_sha256"],
        "total_pair_presentations": TOTAL_PAIR_PRESENTATIONS,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "presentation_checkpoint_steps": list(PRESENTATION_CHECKPOINTS),
        "objective_version": OBJECTIVE_VERSION,
        "pairing_objective_version": PAIRING_OBJECTIVE_VERSION,
        "pair_physical_batch_size": 1,
        "pair_logical_batch_size": 2,
        "pair_directional_exposures": 2,
        "pair_presentations_per_optimizer_step": 7,
        "gradient_accumulation_steps": 7,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "max_grad_norm": MAX_GRAD_NORM,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "save_steps": 1,
        "save_total_limit": len(CHECKPOINT_STEPS),
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
        "v13_launch_contract_tsv_control_character",
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
