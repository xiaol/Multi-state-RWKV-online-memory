#!/usr/bin/env python3
"""Run the protected Scene-Memory V8 Train32 value/causal gate.

The gate is deliberately train-derived.  It validates only the frozen Train32,
pairing, source, and curriculum artifacts before evaluating one completed V8
checkpoint.  A passing, checkpoint-bound receipt can subsequently authorize
only the frozen ``scene-v4-current`` Hard32 evaluation; this module never opens
or hashes Hard32 itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_train32_eval as v7,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v8_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    PAIR_TARGET_DECISION_MASK_MODE,
    PAIR_TARGET_DECISION_NLL_NORMALIZATION,
    TASK_NAME,
    base_model_prompt_identity,
    base_model_weight_identity,
    build_comparisons,
    clear_model_memory,
    evaluate_condition,
    extract_json,
    fingerprint_payload_sha256,
    is_canonical_scene_prediction,
    load_adapter_model,
    memory_architecture_contract,
    recovered_scene_score,
    resolved_memory_layer_count,
    runtime_package_versions,
    score_prediction,
    sha256_file,
    strict_gold_boundaries,
    summarize_records,
    utc_now,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v8_warm_start import (  # noqa: E402
    DEFAULT_LOCK_PATH as WARM_START_LOCK,
    RECEIPT_SCHEMA as WARM_START_RECEIPT_SCHEMA,
    WARM_START_MODE,
    load_v8_warm_start_lock,
)


BENCHMARK_LOCK = SCRIPT_DIR / "scene_memory_v8_benchmark_lock.json"
BENCHMARK_LOCK_SCHEMA = "rwkv_ms_scene_memory_v8_benchmark_selection_lock.v1"
GATE_CONTRACT = "scene_memory_v8_train32_value_causal_gate"
GATE_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v8_train32_gate_receipt.v1"
GATE_RECORD_SCHEMA = "rwkv_ms_scene_memory_v8_train32_gate_record.v1"
GATE_SUMMARY_SCHEMA = "rwkv_ms_scene_memory_v8_train32_gate_summary.v1"
GATE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v8_train32_gate_manifest.v1"
HARD32_AUTHORIZATION_KIND = "scene_memory_v8_train32_gate_receipt"
HARD32_AUTHORIZATION_SCOPE = (
    "fixed_scene_v4_current_hard32_only_no_full170_no_test_no_other_benchmarks"
)
CONDITIONS = ("state_only", "state_only_donor", "state_only_no_write")
LOCKED_CHECKPOINT_STEPS = (14, 28, 42, 56, 80, 104, 128, 152)
FIRST_GATE_STEP = 56
VALUE14_ORDINALS = (1, 3, 5, 9, 10, 14, 19, 20, 22, 23, 24, 26, 28, 31)
VALUE14_SET = frozenset(VALUE14_ORDINALS)
TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
VALUE_STRATA = ("same_cardinality_value", "cross_cardinality_value")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

V8_OBJECTIVE = {
    "training_objective_version": (
        "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
    ),
    "evaluation_criterion": (
        "canonical_greedy_generation_plus_first_pair_distinguishing_token_identity"
    ),
    "aggregate_full_answer_ce_authorizes": False,
    "pair_target": "first_pair_distinguishing_semantic_token_v1",
    "causal_control": "same_checkpoint_adapter_active_no_write_state_reset",
    "donor_control": "predeclared_symmetric_train32_donor_state",
}
# The pairing manifest describes how source/donor targets were materialized.
# Generated-prefix unlikelihood extends the training loss without changing that
# pairing contract, so its version remains the base generation objective.
V8_PAIRING_OBJECTIVE_VERSION = "scene_state_generation_ce_v1"
ROOT_VALUE14_BASELINE = {
    "strict_exact_rows": 2,
    "strict_micro_f1": 0.17647058823529413,
    "tp": 3,
    "fp": 12,
    "fn": 16,
    "donor_current_strict_exact_rows": 1,
    "zero_strict_exact_rows": 0,
}
VALUE14_GATE_REQUIREMENTS = {
    "canonical_correct_outputs": 14,
    "correct_strict_exact_rows": 11,
    "correct_exact_gain_over_v7_root": 9,
    "donor_identity_strict_exact_rows": 11,
    "correct_beats_donor_current_rows": 11,
    "correct_minus_zero_strict_exact_rows": 9,
    "correct_vs_donor_selected_token_positive_rows": 11,
    "donor_vs_correct_selected_token_positive_rows": 11,
    "bidirectional_identity_switch_rows": 10,
    "correct_vs_zero_selected_token_positive_rows": 11,
    "same_cardinality_correct_exact_rows": 8,
    "same_cardinality_donor_identity_exact_rows": 8,
    "same_cardinality_identity_switch_rows": 8,
    "cross_cardinality_correct_exact_rows": 3,
    "cross_cardinality_donor_identity_exact_rows": 3,
    "cross_cardinality_identity_switch_rows": 3,
}


class V8EvaluationContractError(ValueError):
    """Raised when a protected V8 evaluation binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V8EvaluationContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hash_payload(
    payload: Mapping[str, Any], *, hash_field: str
) -> str:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    return canonical_sha256(unsigned)


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    recorded = payload.get(field)
    _require(
        isinstance(recorded, str) and SHA256_RE.fullmatch(recorded) is not None,
        f"V8 {field} is missing or invalid",
    )
    _require(
        recorded == self_hash_payload(payload, hash_field=field),
        f"V8 {field} differs",
    )
    return recorded


def _regular_file(path: Path | str, *, description: str) -> Path:
    expanded = Path(path).expanduser()
    _require(not expanded.is_symlink(), f"V8 forbids a symlink for {description}")
    resolved = expanded.resolve()
    _require(resolved.is_file(), f"V8 {description} is missing: {resolved}")
    return resolved


def _load_json(path: Path | str, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V8EvaluationContractError(
            f"V8 {description} is invalid JSON: {resolved}"
        ) from exc
    _require(isinstance(payload, dict), f"V8 {description} must be an object")
    return payload


def _artifact_binding(path: Path | str, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_artifact_binding(
    binding: Mapping[str, Any], *, description: str
) -> Path:
    _require(isinstance(binding, Mapping), f"V8 {description} binding is missing")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    size = binding.get("bytes")
    _require(isinstance(raw_path, str) and raw_path, f"V8 {description} path differs")
    _require(
        isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
        f"V8 {description} SHA-256 differs",
    )
    path = _regular_file(raw_path, description=description)
    _require(path.stat().st_size == size, f"V8 {description} byte size differs")
    _require(sha256_file(path) == digest, f"V8 {description} artifact differs")
    return path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_canonical_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, canonical_json_bytes(payload).decode("utf-8") + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, description=description)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            _require(bool(raw.strip()), f"V8 blank row in {description}:{line_number}")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise V8EvaluationContractError(
                    f"V8 invalid JSONL in {description}:{line_number}"
                ) from exc
            _require(isinstance(payload, dict), f"V8 non-object in {description}")
            records.append(payload)
    return records


def validate_benchmark_lock(
    path: Path | str = BENCHMARK_LOCK,
) -> dict[str, Any]:
    """Validate task selection without touching the protected benchmark files."""

    lock_path = _regular_file(path, description="benchmark selection lock")
    lock = _load_json(lock_path, description="benchmark selection lock")
    _require(lock.get("schema") == BENCHMARK_LOCK_SCHEMA, "V8 benchmark schema differs")
    _validate_self_hash(lock, field="lock_sha256")
    benchmark = lock.get("benchmark")
    protected = lock.get("protected_benchmark")
    pairing = lock.get("training_pairing")
    baseline = lock.get("root_checkpoint_train32_baseline")
    _require(
        benchmark
        == {
            "memory_relevant_task_family": "scene_boundary_detection",
            "selected_task": TASK_NAME,
            "selection_rule": (
                "lowest_base_strict_f1_among_schema_compatible_scene_tasks"
            ),
            "task_kind": "scene",
        },
        "V8 selected benchmark differs",
    )
    _require(isinstance(protected, dict), "V8 protected benchmark binding is missing")
    _require(protected.get("rows") == 32, "V8 protected benchmark row count differs")
    _require(protected.get("split") == "val", "V8 protected split differs")
    _require(
        protected.get("authorization_scope") == HARD32_AUTHORIZATION_SCOPE
        and protected.get("full170_authorized") is False
        and protected.get("test_authorized") is False
        and protected.get("other_benchmarks_authorized") is False,
        "V8 protected benchmark scope differs",
    )
    for field in (
        "holdout_sha256",
        "official_val_sha256",
        "pair_manifest_sha256",
        "selection_sha256",
    ):
        _require(
            isinstance(protected.get(field), str)
            and SHA256_RE.fullmatch(str(protected[field])) is not None,
            f"V8 protected benchmark {field} differs",
        )
    _require(
        isinstance(pairing, dict)
        and pairing.get("first_authorizable_checkpoint_step") == FIRST_GATE_STEP
        and pairing.get("locked_checkpoint_steps") == list(LOCKED_CHECKPOINT_STEPS)
        and pairing.get("value14_steps_before_first_gate") == 56,
        "V8 benchmark training pairing differs",
    )
    _require(
        isinstance(baseline, dict)
        and baseline.get("value14", {}).get("ordinals") == list(VALUE14_ORDINALS)
        and baseline.get("value14", {}).get("state_only_strict_exact_rows")
        == ROOT_VALUE14_BASELINE["strict_exact_rows"],
        "V8 root Value14 baseline differs",
    )
    return {
        "path": str(lock_path),
        "file_sha256": sha256_file(lock_path),
        "lock_sha256": lock["lock_sha256"],
        "selected_task": TASK_NAME,
        "authorization_scope": HARD32_AUTHORIZATION_SCOPE,
        "protected_benchmark": dict(protected),
        "root_checkpoint_train32_baseline": dict(baseline),
    }


def validate_v8_train_inputs(
    *,
    source_lock_path: Path | str = launch.SOURCE_LOCK,
    benchmark_lock_path: Path | str = BENCHMARK_LOCK,
    ssd_root: Path | str = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Validate V8 Train32/curriculum inputs without Hard32 filesystem access."""

    data = launch.validate_data_contract(
        source_lock_path=Path(source_lock_path),
        ssd_root=Path(ssd_root),
    )
    benchmark = validate_benchmark_lock(benchmark_lock_path)
    v8_source_path = Path(str(data["source_manifest"]))
    v8_source = _load_json(v8_source_path, description="V8 source manifest")
    train_partition = v8_source.get("partitions", {}).get("train")
    parent_source = v8_source.get("parent_v7_source_manifest")
    _require(isinstance(train_partition, dict), "V8 Train32 partition is missing")
    _require(isinstance(parent_source, dict), "V8 parent V7 source is missing")
    row_binding = train_partition.get("row_manifest")
    _require(isinstance(row_binding, dict), "V8 Train32 row binding is missing")
    rows_path = Path(str(row_binding.get("path")))
    parent_source_path = Path(str(parent_source.get("path")))
    expected_hashes = {
        "dataset": launch.TRAIN32_SHA256,
        "row_manifest": launch.TRAIN32_ROWS_SHA256,
        "pair_manifest": launch.TRAIN32_PAIR_SHA256,
        "source_manifest": str(parent_source.get("sha256")),
    }
    input_contract = v7.validate_v7_contract(
        contract="scene_v7_train32_overfit",
        dataset_file=Path(str(data["train_file"])),
        row_manifest_file=rows_path,
        pair_manifest_file=Path(str(data["pair_manifest"])),
        source_manifest_file=parent_source_path,
        expected_dataset_sha256=expected_hashes["dataset"],
        expected_row_manifest_sha256=expected_hashes["row_manifest"],
        expected_pair_manifest_sha256=expected_hashes["pair_manifest"],
        expected_source_manifest_sha256=expected_hashes["source_manifest"],
        source_lock_file=v7.DEFAULT_SOURCE_LOCK,
    )
    source_lock = _load_json(source_lock_path, description="V8 source lock")
    locked_artifacts = source_lock.get("artifacts")
    _require(isinstance(locked_artifacts, dict), "V8 source-lock artifacts are missing")
    artifacts = {
        "train32": _artifact_binding(data["train_file"], description="Train32"),
        "train32_rows": _artifact_binding(rows_path, description="Train32 rows"),
        "train32_pair_manifest": _artifact_binding(
            data["pair_manifest"], description="Train32 pair manifest"
        ),
        "train32_source_manifest": _artifact_binding(
            parent_source_path, description="Train32 source manifest"
        ),
        "v7_source_lock": _artifact_binding(
            v7.DEFAULT_SOURCE_LOCK, description="V7 source lock"
        ),
        "v8_source_lock": _artifact_binding(
            source_lock_path, description="V8 source lock"
        ),
        "v8_bundle_manifest": _artifact_binding(
            locked_artifacts["bundle_manifest"]["path"],
            description="V8 bundle manifest",
        ),
        "v8_source_manifest": _artifact_binding(
            data["source_manifest"], description="V8 source manifest"
        ),
        "v8_schedule": _artifact_binding(
            data["schedule"], description="V8 schedule"
        ),
        "v8_schedule_manifest": _artifact_binding(
            data["schedule_manifest"], description="V8 schedule manifest"
        ),
    }
    expected_artifact_hashes = {
        "train32": launch.TRAIN32_SHA256,
        "train32_rows": launch.TRAIN32_ROWS_SHA256,
        "train32_pair_manifest": launch.TRAIN32_PAIR_SHA256,
        "v8_source_lock": launch.SOURCE_LOCK_FILE_SHA256,
        "v8_bundle_manifest": launch.BUNDLE_MANIFEST_FILE_SHA256,
        "v8_source_manifest": launch.SOURCE_MANIFEST_FILE_SHA256,
        "v8_schedule": launch.SCHEDULE_FILE_SHA256,
        "v8_schedule_manifest": launch.SCHEDULE_MANIFEST_FILE_SHA256,
    }
    for name, expected in expected_artifact_hashes.items():
        _require(artifacts[name]["sha256"] == expected, f"V8 {name} hash differs")
    pair_entries = input_contract["pairing"]["directed_pairs"]
    value_ordinals = tuple(
        int(entry["train_row_ordinal"])
        for entry in pair_entries
        if entry.get("target_stratum") in VALUE_STRATA
    )
    _require(value_ordinals == VALUE14_ORDINALS, "V8 Value14 pairing differs")
    return {
        "contract": GATE_CONTRACT,
        "rows": input_contract["rows"],
        "pairing": input_contract["pairing"],
        "v7_input_contract": input_contract,
        "artifacts": artifacts,
        "benchmark_lock": benchmark,
        "v8_source_manifest_sha256": v8_source["manifest_sha256"],
        "v8_schedule_entries_sha256": launch.SCHEDULE_ENTRIES_SHA256,
        "value14_ordinals": list(VALUE14_ORDINALS),
        "checkpoint_steps": list(LOCKED_CHECKPOINT_STEPS),
    }


def _canonical_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(protocol))


def _validate_v8_protocol(
    protocol: Mapping[str, Any],
    *,
    step: int,
    input_contract: Mapping[str, Any],
) -> str:
    artifacts = input_contract["artifacts"]
    expected = {
        "schema_version": 11,
        "memory_objective_version": V8_OBJECTIVE["training_objective_version"],
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": artifacts["train32"]["path"],
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.0,
        "warmup_steps": 4,
        "save_steps": 14,
        "num_train_epochs": 1.0,
        "train_sampler_seed": None,
        "train_sampler_mode": "explicit_ordered_train_row_ordinal_v1",
        "max_steps": step,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.5,
        "scene_generation_generated_unlikelihood_mode": (
            "greedy_correct_state_edit_aligned_wrong_tokens_v2"
        ),
        "scene_generation_generated_unlikelihood_scope": (
            "same_and_cross_cardinality_value_rows_v1"
        ),
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_generation_generated_rollout_extra_tokens": 4,
        "scene_generation_generated_rollout_max_tokens": 24,
        "scene_generation_generated_rollout_decoding": (
            "greedy_use_cache_true_exact_system_only_prompt_v1"
        ),
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    _require(not mismatches, "V8 training protocol differs: " + ", ".join(mismatches))
    formula = str(protocol.get("scene_generation_objective_formula", ""))
    _require(
        formula.endswith(" + 0.5 * correct_state_generated_prefix_unlikelihood")
        and "first_wrong_gold_prefix_top1_hinge(0.2)" in formula
        and "correct_source_vs_donor_two_token_ce" in formula
        and "donor_donor_vs_source_two_token_ce" in formula,
        "V8 training objective formula differs",
    )
    source_identity = protocol.get("scene_state_source_manifest")
    expected_source = {
        "path": artifacts["v8_source_manifest"]["path"],
        "file_sha256": artifacts["v8_source_manifest"]["sha256"],
        "schema": launch.SOURCE_SCHEMA,
        "train_file": artifacts["train32"]["path"],
        "train_file_sha256": artifacts["train32"]["sha256"],
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": v7.EPISODE_CONTRACT,
    }
    _require(source_identity == expected_source, "V8 protocol source identity differs")
    schedule = protocol.get("train_schedule")
    _require(isinstance(schedule, Mapping), "V8 protocol schedule is missing")
    schedule_expected = {
        "schema": launch.CURRICULUM_SCHEMA,
        "schedule_path": artifacts["v8_schedule"]["path"],
        "schedule_file_sha256": artifacts["v8_schedule"]["sha256"],
        "schedule_entries_sha256": launch.SCHEDULE_ENTRIES_SHA256,
        "schedule_manifest_path": artifacts["v8_schedule_manifest"]["path"],
        "schedule_manifest_file_sha256": artifacts["v8_schedule_manifest"]["sha256"],
        "schedule_manifest_sha256": launch.SCHEDULE_MANIFEST_CANONICAL_SHA256,
        "ordered_train_row_ordinals_sha256": launch.SCHEDULE_ORDINALS_SHA256,
        "total_steps": 152,
        "checkpoint_steps": list(LOCKED_CHECKPOINT_STEPS),
        "value14_ordinals": list(VALUE14_ORDINALS),
    }
    schedule_mismatches = [
        name for name, value in schedule_expected.items() if schedule.get(name) != value
    ]
    _require(
        not schedule_mismatches,
        "V8 protocol schedule differs: " + ", ".join(schedule_mismatches),
    )
    return _canonical_protocol_sha256(protocol)


def _validate_pairing_checkpoint(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    pairing = _load_json(path, description="checkpoint pairing manifest")
    _validate_self_hash(pairing, field="manifest_sha256")
    _require(pairing.get("schema_version") == 2, "V8 checkpoint pairing schema differs")
    _require(
        pairing.get("objective_version") == V8_PAIRING_OBJECTIVE_VERSION,
        "V8 checkpoint pairing objective differs",
    )
    _require(set(pairing.get("splits", {})) == {"train"}, "V8 pairing splits differ")
    train = pairing["splits"]["train"]
    _require(isinstance(train, dict), "V8 checkpoint train pairing is missing")
    _validate_self_hash(train, field="manifest_sha256")
    expected_train = {
        "sample_count": 32,
        "source_pair_manifest_path": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["path"],
        "source_pair_manifest_file_sha256": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["sha256"],
        "source_pair_manifest_sha256": input_contract["pairing"][
            "manifest_sha256"
        ],
        "source_entries_sha256": input_contract["pairing"]["entries_sha256"],
        "target_stratum_row_counts": {
            "presence": 18,
            "same_cardinality_value": 10,
            "cross_cardinality_value": 4,
        },
    }
    for name, value in expected_train.items():
        _require(train.get(name) == value, f"V8 checkpoint pairing {name} differs")
    protocol_pairing = protocol.get("scene_state_identity_pairing")
    _require(isinstance(protocol_pairing, Mapping), "V8 protocol pairing is missing")
    _require(
        protocol_pairing.get("manifest_sha256") == pairing["manifest_sha256"]
        and protocol_pairing.get("target_stratum_row_counts")
        == expected_train["target_stratum_row_counts"],
        "V8 protocol/checkpoint pairing differs",
    )
    return pairing


def _checkpoint_step(checkpoint: Path) -> int:
    suffix = checkpoint.name.removeprefix("checkpoint-")
    _require(
        checkpoint.name.startswith("checkpoint-") and suffix.isdigit(),
        "V8 checkpoint directory must be checkpoint-N",
    )
    return int(suffix)


def _validate_v8_config(config: Mapping[str, Any]) -> None:
    expected = {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "target_layers": list(range(42)),
        "delta_heads": ["q", "o"],
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "memory_fusion_residual_scale_max": 1.0,
        "trainable_delta_scale": True,
        "delta_scale_init": 0.1,
        "delta_scale_max": 0.5,
        "delta_scale_granularity": "head",
        "delta_scale_parameterization": "alpha_over_rank",
        "output_init": "base_slice_fixed",
        "base_slice_ref_width": 8,
        "online_gain": 0.2,
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_semantics_version": 2,
    }
    mismatches = [name for name, value in expected.items() if config.get(name) != value]
    _require(not mismatches, "V8 checkpoint config differs: " + ", ".join(mismatches))


def _validate_root_warm_start_lineage(
    checkpoint: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    target_protocol_sha256: str,
) -> dict[str, Any]:
    _require(
        manifest.get("schema") == WARM_START_RECEIPT_SCHEMA,
        "V8 root warm-start receipt schema differs",
    )
    _require(manifest.get("schema_version") == 1, "V8 warm lineage schema differs")
    _require(manifest.get("mode") == WARM_START_MODE, "V8 warm lineage mode differs")
    receipt_sha256 = _validate_self_hash(manifest, field="receipt_sha256")
    warm_lock = load_v8_warm_start_lock(WARM_START_LOCK)
    expected_source = Path(str(warm_lock.get("source_checkpoint"))).expanduser().resolve()
    _require(
        Path(str(manifest.get("source_checkpoint"))).expanduser().resolve()
        == expected_source,
        "V8 warm lineage source checkpoint differs",
    )
    source_lock = manifest.get("source_lock")
    _require(
        source_lock
        == {
            "path": str(WARM_START_LOCK.resolve()),
            "lock_sha256": warm_lock["lock_sha256"],
        },
        "V8 warm lineage source lock differs",
    )
    _require(
        manifest.get("source_artifacts") == warm_lock["artifacts"],
        "V8 warm lineage source artifacts differ",
    )
    for filename, binding in warm_lock["artifacts"].items():
        _require(
            isinstance(binding, Mapping),
            f"V8 warm lineage source binding is invalid: {filename}",
        )
        source_artifact = expected_source / filename
        _require(
            source_artifact.is_file()
            and not source_artifact.is_symlink()
            and source_artifact.stat().st_size == binding.get("bytes")
            and sha256_file(source_artifact) == binding.get("sha256"),
            f"V8 pinned V7 source artifact differs: {filename}",
        )
    evidence = {
        "source_global_step": 256,
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "pre_train_global_step": 0,
        "fresh_optimizer_created": True,
        "fresh_optimizer_state_entries_before_train": 0,
        "fresh_scheduler_created_before_train": False,
        "target_training_protocol_sha256": target_protocol_sha256,
    }
    mismatches = [name for name, value in evidence.items() if manifest.get(name) != value]
    _require(
        not mismatches,
        "V8 warm lineage fresh-state evidence differs: " + ", ".join(mismatches),
    )
    target_fresh = manifest.get("target_fresh_start")
    _require(
        isinstance(target_fresh, Mapping)
        and target_fresh.get("initial_global_step") == 0
        and target_fresh.get("optimizer_state") == "fresh"
        and target_fresh.get("scheduler_state") == "fresh"
        and target_fresh.get("trainer_state") == "fresh",
        "V8 warm lineage target fresh-start evidence differs",
    )
    return {
        "root_warm_start_receipt_sha256": receipt_sha256,
        "chain": [
            {
                "checkpoint": str(checkpoint),
                "step": 14,
                "lineage_filename": manifest_path.name,
                "lineage_file_sha256": sha256_file(manifest_path),
                "lineage_payload_sha256": receipt_sha256,
                "training_protocol_sha256": target_protocol_sha256,
            }
        ],
    }


def _validate_lineage_manifest(
    checkpoint: Path,
    *,
    step: int,
    target_protocol_sha256: str,
    input_contract: Mapping[str, Any],
    seen: set[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    warm_path = checkpoint / "warm_start_lineage_manifest.json"
    continuation_path = checkpoint / "continuation_manifest.json"
    present = [path for path in (warm_path, continuation_path) if path.is_file()]
    _require(len(present) == 1, "V8 checkpoint lineage manifest is missing or ambiguous")
    manifest_path = present[0]
    manifest = _load_json(manifest_path, description="checkpoint lineage manifest")
    if step == 14:
        _require(
            manifest_path.name == "warm_start_lineage_manifest.json",
            "V8 step14 requires warm-start lineage",
        )
        summary = _validate_root_warm_start_lineage(
            checkpoint,
            manifest_path,
            manifest,
            target_protocol_sha256=target_protocol_sha256,
        )
        return dict(manifest), summary

    _require(
        manifest_path.name == "continuation_manifest.json",
        "V8 resumed checkpoint requires continuation lineage",
    )
    _require(manifest.get("schema_version") == 1, "V8 continuation schema differs")
    _require(manifest.get("mode") == "extend", "V8 continuation mode differs")
    manifest_sha256 = _validate_self_hash(manifest, field="manifest_sha256")
    step_position = LOCKED_CHECKPOINT_STEPS.index(step)
    source_step = LOCKED_CHECKPOINT_STEPS[step_position - 1]
    source_raw = manifest.get("source_checkpoint")
    _require(isinstance(source_raw, str) and source_raw, "V8 lineage source is missing")
    source_checkpoint = Path(source_raw).expanduser()
    _require(not source_checkpoint.is_symlink(), "V8 lineage source cannot be a symlink")
    source_checkpoint = source_checkpoint.resolve()
    _require(
        source_checkpoint.is_dir() and _checkpoint_step(source_checkpoint) == source_step,
        "V8 lineage does not identify the previous locked endpoint",
    )
    source_binding = _validate_v8_checkpoint_internal(
        source_checkpoint,
        input_contract=input_contract,
        require_gate_step=False,
        seen=seen,
    )
    source_lineage = source_binding["lineage"]
    source_lineage_filename = source_lineage["chain"][-1]["lineage_filename"]
    source_lineage_path = source_checkpoint / source_lineage_filename
    expected = {
        "source_global_step": source_step,
        "source_effective_max_steps": source_step,
        "source_max_steps": source_step,
        "target_max_steps": step,
        "source_lineage_filename": source_lineage_filename,
        "source_lineage_file_sha256": sha256_file(source_lineage_path),
        "source_training_protocol_sha256": source_binding["training_protocol"][
            "canonical_sha256"
        ],
        "target_training_protocol_sha256": target_protocol_sha256,
        "root_warm_start_receipt_sha256": source_lineage[
            "root_warm_start_receipt_sha256"
        ],
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    _require(
        not mismatches,
        "V8 continuation lineage differs: " + ", ".join(mismatches),
    )
    chain = [
        *source_lineage["chain"],
        {
            "checkpoint": str(checkpoint),
            "step": step,
            "lineage_filename": manifest_path.name,
            "lineage_file_sha256": sha256_file(manifest_path),
            "lineage_payload_sha256": manifest_sha256,
            "training_protocol_sha256": target_protocol_sha256,
        },
    ]
    return dict(manifest), {
        "root_warm_start_receipt_sha256": source_lineage[
            "root_warm_start_receipt_sha256"
        ],
        "chain": chain,
    }


def _validate_v8_checkpoint_internal(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
    require_gate_step: bool,
    seen: set[Path],
) -> dict[str, Any]:
    requested = Path(memory_dir).expanduser()
    _require(not requested.is_symlink(), "V8 checkpoint directory cannot be a symlink")
    checkpoint = requested.resolve()
    _require(checkpoint.is_dir(), f"V8 checkpoint is missing: {checkpoint}")
    _require(checkpoint not in seen, "V8 checkpoint lineage contains a cycle")
    seen.add(checkpoint)
    try:
        step = _checkpoint_step(checkpoint)
        _require(step in LOCKED_CHECKPOINT_STEPS, "V8 checkpoint step is not locked")
        if require_gate_step:
            _require(step >= FIRST_GATE_STEP, "V8 Train32 gate is unavailable before step56")
        paths = {
            "adapter": checkpoint / "delta_mem_adapter.pt",
            "config": checkpoint / "delta_mem_config.json",
            "trainer_state": checkpoint / "trainer_state.json",
            "training_protocol": checkpoint / "training_protocol.json",
            "pairing": checkpoint / "scene_state_identity_pairing_manifest.json",
            "optimizer": checkpoint / "optimizer.pt",
            "scheduler": checkpoint / "scheduler.pt",
        }
        artifacts = {
            name: _artifact_binding(path, description=f"checkpoint {name}")
            for name, path in paths.items()
        }
        rng_paths = sorted(
            path for path in checkpoint.glob("rng_state*.pth") if path.is_file()
        )
        _require(bool(rng_paths), "V8 checkpoint RNG state is missing")
        rng = [
            _artifact_binding(path, description=f"checkpoint RNG state {path.name}")
            for path in rng_paths
        ]
        trainer_state = _load_json(paths["trainer_state"], description="trainer state")
        _require(
            trainer_state.get("global_step") == step
            and trainer_state.get("max_steps") == step,
            "V8 checkpoint is not a completed locked horizon",
        )
        protocol = _load_json(paths["training_protocol"], description="training protocol")
        protocol_sha256 = _validate_v8_protocol(
            protocol,
            step=step,
            input_contract=input_contract,
        )
        config = _load_json(paths["config"], description="Delta-Mem config")
        _validate_v8_config(config)
        architecture = memory_architecture_contract(checkpoint)
        _require(
            architecture.get("target_layers") == list(range(42))
            and architecture.get("delta_heads") == ["q", "o"]
            and architecture.get("rank") == 4
            and architecture.get("rwkv_ms_semantics_version") == 2
            and architecture.get("memory_backend") == "rwkv_ms",
            "V8 checkpoint architecture differs",
        )
        pairing = _validate_pairing_checkpoint(
            paths["pairing"],
            protocol=protocol,
            input_contract=input_contract,
        )
        lineage_payload, lineage = _validate_lineage_manifest(
            checkpoint,
            step=step,
            target_protocol_sha256=protocol_sha256,
            input_contract=input_contract,
            seen=seen,
        )
        lineage_path = checkpoint / lineage["chain"][-1]["lineage_filename"]
        return {
            "memory_dir": str(checkpoint),
            "global_step": step,
            "max_steps": step,
            "artifacts": artifacts,
            "rng_state": rng,
            "training_protocol": {
                **artifacts["training_protocol"],
                "canonical_sha256": protocol_sha256,
            },
            "lineage_artifact": _artifact_binding(
                lineage_path, description="checkpoint lineage"
            ),
            "lineage_payload": lineage_payload,
            "lineage": lineage,
            "pairing_manifest_sha256": pairing["manifest_sha256"],
            "architecture": architecture,
            "objective": dict(V8_OBJECTIVE),
        }
    finally:
        seen.remove(checkpoint)


def validate_v8_checkpoint(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one gate-eligible V8 checkpoint and its complete lineage chain."""

    return _validate_v8_checkpoint_internal(
        memory_dir,
        input_contract=input_contract,
        require_gate_step=True,
        seen=set(),
    )


def _strict_score(record: Mapping[str, Any]) -> dict[str, Any]:
    expected = score_prediction("scene", record.get("parsed_json"), record.get("gold"))
    recorded = record.get("score_strict")
    if recorded is not None:
        _require(recorded == expected, "V8 record strict score differs")
    return expected


def _strict_exact(score: Mapping[str, Any]) -> bool:
    return (
        bool(score.get("schema_valid"))
        and int(score.get("fp", 0)) == 0
        and int(score.get("fn", 0)) == 0
    )


def _strictly_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_score = _strict_score(left)
    right_score = _strict_score(right)
    return (
        int(_strict_exact(left_score)),
        float(left_score["sample_f1"]),
        -int(left_score["fp"]),
        -int(left_score["fn"]),
    ) > (
        int(_strict_exact(right_score)),
        float(right_score["sample_f1"]),
        -int(right_score["fp"]),
        -int(right_score["fn"]),
    )


def _micro_f1(scores: Iterable[Mapping[str, Any]]) -> float:
    rows = list(scores)
    true_positive = sum(int(score["tp"]) for score in rows)
    false_positive = sum(int(score["fp"]) for score in rows)
    false_negative = sum(int(score["fn"]) for score in rows)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _pair_target_report(record: Mapping[str, Any]) -> Mapping[str, Any]:
    semantic = record.get("semantic_decision_nll")
    _require(isinstance(semantic, Mapping), "V8 Value14 semantic evidence is missing")
    pair = semantic.get("pair_target")
    _require(isinstance(pair, Mapping), "V8 Value14 pair-target evidence is missing")
    _require(
        pair.get("mask_mode") == PAIR_TARGET_DECISION_MASK_MODE
        and pair.get("normalization") == PAIR_TARGET_DECISION_NLL_NORMALIZATION
        and pair.get("token_count") == 1,
        "V8 Value14 pair-target protocol differs",
    )
    for field in (
        "mean_nll",
        "alternative_target_mean_nll",
        "selected_over_alternative_logprob_margin",
    ):
        value = pair.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"V8 Value14 {field} is invalid",
        )
    return pair


def build_value14_selected_token_evidence(
    *,
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    pairing: Mapping[str, Any],
) -> dict[str, Any]:
    pair_by_ordinal = {
        int(entry["train_row_ordinal"]): entry
        for entry in pairing["directed_pairs"]
    }
    rows: list[dict[str, Any]] = []
    for ordinal in VALUE14_ORDINALS:
        reports = {
            condition: _pair_target_report(indexed[condition][ordinal])
            for condition in CONDITIONS
        }
        identity_fields = (
            "selected_target_positions",
            "selected_target_token_ids",
            "donor_target_token_ids",
            "first_differing_semantic_ordinal",
            "causal_prefix_sha256",
            "donor_source_index",
            "donor_row_sha256",
            "read_rendered_sha256",
        )
        for field in identity_fields:
            values = {
                json.dumps(report.get(field), sort_keys=True)
                for report in reports.values()
            }
            _require(len(values) == 1, f"V8 Value14 {field} differs across states")
        source_margin = float(
            reports["state_only"]["selected_over_alternative_logprob_margin"]
        )
        donor_source_margin = float(
            reports["state_only_donor"][
                "selected_over_alternative_logprob_margin"
            ]
        )
        zero_source_margin = float(
            reports["state_only_no_write"][
                "selected_over_alternative_logprob_margin"
            ]
        )
        correct_target_nll = float(reports["state_only"]["mean_nll"])
        donor_state_correct_target_nll = float(
            reports["state_only_donor"]["mean_nll"]
        )
        zero_correct_target_nll = float(
            reports["state_only_no_write"]["mean_nll"]
        )
        donor_identity_margin = -donor_source_margin
        row = {
            "train_row_ordinal": ordinal,
            "official_source_index": indexed["state_only"][ordinal]["source_index"],
            "donor_train_row_ordinal": pair_by_ordinal[ordinal][
                "donor_train_row_ordinal"
            ],
            "target_stratum": pair_by_ordinal[ordinal]["target_stratum"],
            "source_selected_over_donor_token_margin": source_margin,
            "donor_selected_over_source_token_margin": donor_identity_margin,
            "identity_switch_margin": source_margin + donor_identity_margin,
            "donor_state_minus_correct_state_current_target_nll": (
                donor_state_correct_target_nll - correct_target_nll
            ),
            "zero_state_minus_correct_state_current_target_nll": (
                zero_correct_target_nll - correct_target_nll
            ),
            "zero_source_over_donor_token_margin_diagnostic": zero_source_margin,
            "correct_state_prefers_source_token": source_margin > 0.0,
            "donor_state_prefers_donor_token": donor_identity_margin > 0.0,
            "bidirectional_identity_switch": (
                source_margin > 0.0 and donor_identity_margin > 0.0
            ),
            "correct_state_beats_donor_state_on_source_token": (
                donor_state_correct_target_nll - correct_target_nll > 0.0
            ),
            "correct_state_beats_zero_on_source_token": (
                zero_correct_target_nll - correct_target_nll > 0.0
            ),
            "pair_target": {
                field: reports["state_only"][field]
                for field in (
                    "selected_target_positions",
                    "selected_target_token_ids",
                    "donor_target_token_ids",
                    "first_differing_semantic_ordinal",
                    "causal_prefix_sha256",
                    "donor_source_index",
                    "donor_row_sha256",
                )
            },
        }
        rows.append(row)

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def mean(field: str) -> float:
            return sum(float(row[field]) for row in selected) / len(selected)

        return {
            "rows": len(selected),
            "ordinals": [int(row["train_row_ordinal"]) for row in selected],
            "correct_state_prefers_source_token_rows": sum(
                bool(row["correct_state_prefers_source_token"]) for row in selected
            ),
            "donor_state_prefers_donor_token_rows": sum(
                bool(row["donor_state_prefers_donor_token"]) for row in selected
            ),
            "bidirectional_identity_switch_rows": sum(
                bool(row["bidirectional_identity_switch"]) for row in selected
            ),
            "correct_state_beats_donor_state_on_source_token_rows": sum(
                bool(row["correct_state_beats_donor_state_on_source_token"])
                for row in selected
            ),
            "correct_state_beats_zero_on_source_token_rows": sum(
                bool(row["correct_state_beats_zero_on_source_token"])
                for row in selected
            ),
            "mean_source_selected_over_donor_token_margin": mean(
                "source_selected_over_donor_token_margin"
            ),
            "mean_donor_selected_over_source_token_margin": mean(
                "donor_selected_over_source_token_margin"
            ),
            "mean_identity_switch_margin": mean("identity_switch_margin"),
            "mean_zero_minus_correct_current_target_nll": mean(
                "zero_state_minus_correct_state_current_target_nll"
            ),
        }

    by_stratum = {
        stratum: summarize([row for row in rows if row["target_stratum"] == stratum])
        for stratum in VALUE_STRATA
    }
    _require(by_stratum["same_cardinality_value"]["rows"] == 10, "V8 same-value rows differ")
    _require(by_stratum["cross_cardinality_value"]["rows"] == 4, "V8 cross-value rows differ")
    return {
        "criterion": "first_pair_distinguishing_token_logits_only_v1",
        "full_answer_ce_used": False,
        "gap_sign": "positive_means_state_prefers_its_bound_identity",
        "overall": summarize(rows),
        "by_stratum": by_stratum,
        "rows": rows,
    }


def build_v8_gate(
    *,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    pairing: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the locked V8 generation/value gate over Train32 and Value14."""

    _require(set(records_by_condition) == set(CONDITIONS), "V8 gate conditions differ")
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        records = list(records_by_condition[condition])
        _require(len(records) == 32, f"V8 gate requires 32 {condition} rows")
        condition_rows: dict[int, Mapping[str, Any]] = {}
        for position, record in enumerate(records):
            ordinal = record.get("train_row_ordinal")
            _require(ordinal == position, "V8 gate rows must retain Train32 order")
            _require(record.get("condition") == condition, "V8 gate condition differs")
            _require(ordinal not in condition_rows, "V8 gate ordinal is duplicated")
            _strict_score(record)
            condition_rows[int(ordinal)] = record
        indexed[condition] = condition_rows
    correct = indexed["state_only"]
    donor = indexed["state_only_donor"]
    zero = indexed["state_only_no_write"]
    donor_by_ordinal = {
        int(entry["train_row_ordinal"]): int(entry["donor_train_row_ordinal"])
        for entry in pairing["directed_pairs"]
    }
    _require(
        all(donor_by_ordinal.get(donor_ordinal) == ordinal for ordinal, donor_ordinal in donor_by_ordinal.items()),
        "V8 donor mapping is not symmetric",
    )

    value_scores = {ordinal: _strict_score(correct[ordinal]) for ordinal in VALUE14_ORDINALS}
    value_donor_current_scores = {
        ordinal: _strict_score(donor[ordinal]) for ordinal in VALUE14_ORDINALS
    }
    value_zero_scores = {ordinal: _strict_score(zero[ordinal]) for ordinal in VALUE14_ORDINALS}
    donor_identity_scores = {
        ordinal: score_prediction(
            "scene",
            donor[ordinal].get("parsed_json"),
            correct[donor_by_ordinal[ordinal]]["gold"],
        )
        for ordinal in VALUE14_ORDINALS
    }
    value_pair_entries = {
        int(entry["train_row_ordinal"]): entry
        for entry in pairing["directed_pairs"]
        if int(entry["train_row_ordinal"]) in VALUE14_SET
    }

    def generation_summary(ordinals: Sequence[int]) -> dict[str, Any]:
        scores = [value_scores[ordinal] for ordinal in ordinals]
        donor_current = [value_donor_current_scores[ordinal] for ordinal in ordinals]
        zero_scores = [value_zero_scores[ordinal] for ordinal in ordinals]
        donor_identity = [donor_identity_scores[ordinal] for ordinal in ordinals]
        exact = sum(_strict_exact(score) for score in scores)
        return {
            "rows": len(ordinals),
            "ordinals": list(ordinals),
            "canonical_correct_outputs": sum(
                is_canonical_scene_prediction(correct[ordinal].get("parsed_json"))
                for ordinal in ordinals
            ),
            "correct_strict_exact_rows": exact,
            "correct_strict_micro_f1": _micro_f1(scores),
            "donor_current_strict_exact_rows": sum(
                _strict_exact(score) for score in donor_current
            ),
            "donor_current_strict_micro_f1": _micro_f1(donor_current),
            "donor_identity_strict_exact_rows": sum(
                _strict_exact(score) for score in donor_identity
            ),
            "donor_identity_strict_micro_f1": _micro_f1(donor_identity),
            "zero_strict_exact_rows": sum(_strict_exact(score) for score in zero_scores),
            "zero_strict_micro_f1": _micro_f1(zero_scores),
            "correct_minus_zero_strict_exact_rows": (
                exact - sum(_strict_exact(score) for score in zero_scores)
            ),
            "correct_beats_donor_current_rows": sum(
                _strictly_better(correct[ordinal], donor[ordinal])
                for ordinal in ordinals
            ),
            "correct_beats_zero_rows": sum(
                _strictly_better(correct[ordinal], zero[ordinal])
                for ordinal in ordinals
            ),
            "tp": sum(int(score["tp"]) for score in scores),
            "fp": sum(int(score["fp"]) for score in scores),
            "fn": sum(int(score["fn"]) for score in scores),
        }

    value_generation = generation_summary(VALUE14_ORDINALS)
    strata_ordinals = {
        stratum: tuple(
            ordinal
            for ordinal in VALUE14_ORDINALS
            if value_pair_entries[ordinal]["target_stratum"] == stratum
        )
        for stratum in VALUE_STRATA
    }
    by_stratum_generation = {
        stratum: generation_summary(ordinals)
        for stratum, ordinals in strata_ordinals.items()
    }
    selected = build_value14_selected_token_evidence(
        indexed=indexed,
        pairing=pairing,
    )
    selected_overall = selected["overall"]
    selected_same = selected["by_stratum"]["same_cardinality_value"]
    selected_cross = selected["by_stratum"]["cross_cardinality_value"]
    same_generation = by_stratum_generation["same_cardinality_value"]
    cross_generation = by_stratum_generation["cross_cardinality_value"]
    zero_raw = [str(zero[index].get("raw_generation")) for index in range(32)]

    requirements = VALUE14_GATE_REQUIREMENTS
    gates = {
        "value14_all_correct_outputs_canonical": (
            value_generation["canonical_correct_outputs"]
            >= requirements["canonical_correct_outputs"]
        ),
        "value14_correct_identity_generation": (
            value_generation["correct_strict_exact_rows"]
            >= requirements["correct_strict_exact_rows"]
        ),
        "value14_material_gain_over_v7_root": (
            value_generation["correct_strict_exact_rows"]
            - ROOT_VALUE14_BASELINE["strict_exact_rows"]
            >= requirements["correct_exact_gain_over_v7_root"]
        ),
        "value14_donor_identity_generation": (
            value_generation["donor_identity_strict_exact_rows"]
            >= requirements["donor_identity_strict_exact_rows"]
        ),
        "value14_correct_beats_donor_current": (
            value_generation["correct_beats_donor_current_rows"]
            >= requirements["correct_beats_donor_current_rows"]
        ),
        "value14_correct_state_is_causal": (
            value_generation["correct_minus_zero_strict_exact_rows"]
            >= requirements["correct_minus_zero_strict_exact_rows"]
        ),
        "value14_correct_selected_token_identity": (
            selected_overall["correct_state_prefers_source_token_rows"]
            >= requirements["correct_vs_donor_selected_token_positive_rows"]
        ),
        "value14_donor_selected_token_identity": (
            selected_overall["donor_state_prefers_donor_token_rows"]
            >= requirements["donor_vs_correct_selected_token_positive_rows"]
        ),
        "value14_bidirectional_selected_token_switch": (
            selected_overall["bidirectional_identity_switch_rows"]
            >= requirements["bidirectional_identity_switch_rows"]
        ),
        "value14_selected_token_causal_vs_zero": (
            selected_overall["correct_state_beats_zero_on_source_token_rows"]
            >= requirements["correct_vs_zero_selected_token_positive_rows"]
        ),
        "same_cardinality_correct_identity_generation": (
            same_generation["correct_strict_exact_rows"]
            >= requirements["same_cardinality_correct_exact_rows"]
        ),
        "same_cardinality_donor_identity_generation": (
            same_generation["donor_identity_strict_exact_rows"]
            >= requirements["same_cardinality_donor_identity_exact_rows"]
        ),
        "same_cardinality_selected_token_switch": (
            selected_same["bidirectional_identity_switch_rows"]
            >= requirements["same_cardinality_identity_switch_rows"]
        ),
        "cross_cardinality_correct_identity_generation": (
            cross_generation["correct_strict_exact_rows"]
            >= requirements["cross_cardinality_correct_exact_rows"]
        ),
        "cross_cardinality_donor_identity_generation": (
            cross_generation["donor_identity_strict_exact_rows"]
            >= requirements["cross_cardinality_donor_identity_exact_rows"]
        ),
        "cross_cardinality_selected_token_switch": (
            selected_cross["bidirectional_identity_switch_rows"]
            >= requirements["cross_cardinality_identity_switch_rows"]
        ),
        "zero_reset_control_is_row_invariant": len(set(zero_raw)) == 1,
    }
    passed = all(gates.values())
    all_generation = {
        condition: summarize_records(list(records_by_condition[condition]))
        for condition in CONDITIONS
    }
    return {
        "status": "pass" if passed else "fail",
        "all_gates_passed": passed,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "first_authorizable_checkpoint_step": FIRST_GATE_STEP,
        "criterion": V8_OBJECTIVE["evaluation_criterion"],
        "full_answer_ce_used_for_gate": False,
        "requirements": dict(requirements),
        "root_baseline": dict(ROOT_VALUE14_BASELINE),
        "metrics": {
            "value14_generation": value_generation,
            "value14_generation_by_stratum": by_stratum_generation,
            "value14_selected_token_identity": selected,
            "all_train32_diagnostic": all_generation,
            "correct_exact_gain_over_v7_root": (
                value_generation["correct_strict_exact_rows"]
                - ROOT_VALUE14_BASELINE["strict_exact_rows"]
            ),
        },
        "gates": gates,
        "hard32_authorized": passed,
        "authorization_scope": HARD32_AUTHORIZATION_SCOPE,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }


def _record_with_self_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("record_sha256", None)
    result["record_sha256"] = canonical_sha256(result)
    return result


def validate_resume_records(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    donor_by_ordinal: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    _require(condition in CONDITIONS, "V8 resume condition differs")
    validated: dict[int, dict[str, Any]] = {}
    for position, raw_record in enumerate(records):
        record = dict(raw_record)
        _require(record.get("schema") == GATE_RECORD_SCHEMA, "V8 record schema differs")
        _validate_self_hash(record, field="record_sha256")
        _require(record.get("fingerprint") == fingerprint, "V8 record fingerprint differs")
        _require(record.get("condition") == condition, "V8 record condition differs")
        _require(record.get("split") == "train", "V8 record split differs")
        _require(record.get("train_row_ordinal") == position, "V8 records are not a prefix")
        sample = rows[position]
        expected_donor = donor_by_ordinal[position]
        _require(record.get("source_index") == sample["source_index"], "V8 source differs")
        _require(record.get("row_sha256") == sample["row_sha256"], "V8 row hash differs")
        _require(record.get("gold") == sample["gold"], "V8 record gold differs")
        _require(
            record.get("donor_train_row_ordinal") == expected_donor,
            "V8 donor ordinal differs",
        )
        _require(
            record.get("donor_source_index") == rows[expected_donor]["source_index"],
            "V8 donor source differs",
        )
        raw_generation = record.get("raw_generation")
        _require(isinstance(raw_generation, str), "V8 raw generation differs")
        _require(extract_json(raw_generation) == record.get("parsed_json"), "V8 parsed JSON differs")
        _strict_score(record)
        _require(
            record.get("score_recovered")
            == recovered_scene_score(record.get("parsed_json"), sample["gold"]),
            "V8 recovery score differs",
        )
        semantic = record.get("semantic_decision_nll")
        if position in VALUE14_SET:
            _pair_target_report(record)
        else:
            _require(semantic is None, "V8 semantic evidence is restricted to Value14")
        validated[position] = record
    return validated


def evaluator_code_binding() -> dict[str, Any]:
    paths = {
        "v8_gate": Path(__file__).resolve(),
        "v7_train32_runtime": Path(v7.__file__).resolve(),
        "state_runtime": SCRIPT_DIR / "run_scene_state_eval.py",
        "v8_launch_contract": Path(launch.__file__).resolve(),
    }
    return {
        name: _artifact_binding(path, description=f"evaluator code {name}")
        for name, path in paths.items()
    }


def build_gate_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    outputs = {
        "manifest": _artifact_binding(
            output_dir / "manifest.json", description="V8 gate manifest"
        ),
        "summary": _artifact_binding(
            output_dir / "summary.json", description="V8 gate summary"
        ),
        "conditions": {
            condition: _artifact_binding(
                output_dir / f"{condition}.jsonl",
                description=f"V8 gate {condition} output",
            )
            for condition in CONDITIONS
        },
    }
    passed = gate.get("status") == "pass" and gate.get("all_gates_passed") is True
    receipt: dict[str, Any] = {
        "schema": GATE_RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "status": "pass" if passed else "fail",
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "evaluation_fingerprint": fingerprint,
        "objective": dict(V8_OBJECTIVE),
        "training_sources": dict(input_contract["artifacts"]),
        "v8_source_manifest_sha256": input_contract[
            "v8_source_manifest_sha256"
        ],
        "v8_schedule_entries_sha256": input_contract[
            "v8_schedule_entries_sha256"
        ],
        "benchmark_selection_lock": dict(input_contract["benchmark_lock"]),
        "checkpoint": dict(checkpoint),
        "outputs": outputs,
        "code": evaluator_code_binding(),
        "gate": dict(gate),
        "hard32_authorization": {
            "authorization_kind": HARD32_AUTHORIZATION_KIND,
            "authorized": passed,
            "scope": HARD32_AUTHORIZATION_SCOPE,
            "checkpoint_binding": dict(checkpoint),
            "protected_benchmark_binding": input_contract["benchmark_lock"][
                "protected_benchmark"
            ],
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }
    receipt["receipt_sha256"] = self_hash_payload(
        receipt,
        hash_field="receipt_sha256",
    )
    return receipt


def _validate_receipt_outputs(
    payload: Mapping[str, Any],
    *,
    input_contract: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    outputs = payload.get("outputs")
    _require(isinstance(outputs, Mapping), "V8 receipt outputs are missing")
    manifest_path = _verify_artifact_binding(
        outputs.get("manifest", {}), description="receipt manifest"
    )
    summary_path = _verify_artifact_binding(
        outputs.get("summary", {}), description="receipt summary"
    )
    manifest = _load_json(manifest_path, description="receipt manifest")
    _require(manifest.get("schema") == GATE_MANIFEST_SCHEMA, "V8 manifest schema differs")
    _require(
        manifest.get("fingerprint") == payload.get("evaluation_fingerprint"),
        "V8 manifest fingerprint differs",
    )
    summary = _load_json(summary_path, description="receipt summary")
    _require(summary.get("schema") == GATE_SUMMARY_SCHEMA, "V8 summary schema differs")
    _validate_self_hash(summary, field="summary_sha256")
    _require(
        summary.get("fingerprint") == payload.get("evaluation_fingerprint"),
        "V8 summary fingerprint differs",
    )
    condition_bindings = outputs.get("conditions")
    _require(
        isinstance(condition_bindings, Mapping)
        and set(condition_bindings) == set(CONDITIONS),
        "V8 receipt condition outputs differ",
    )
    rows = input_contract["rows"]
    donor_by_ordinal = input_contract["pairing"]["donor_by_ordinal"]
    records: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = _verify_artifact_binding(
            condition_bindings[condition],
            description=f"receipt {condition} output",
        )
        condition_records = _read_jsonl(path, description=f"V8 {condition} output")
        validated = validate_resume_records(
            condition_records,
            condition=condition,
            fingerprint=str(payload["evaluation_fingerprint"]),
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
        )
        _require(len(validated) == 32, f"V8 receipt {condition} output is incomplete")
        records[condition] = [validated[index] for index in range(32)]
    recomputed = build_v8_gate(
        records_by_condition=records,
        pairing=input_contract["pairing"],
    )
    _require(recomputed == payload.get("gate"), "V8 receipt gate does not reproduce")
    _require(recomputed == summary.get("gate"), "V8 summary gate does not reproduce")
    return records


def validate_gate_receipt_for_checkpoint(
    receipt: Path | str | Mapping[str, Any],
    *,
    memory_dir: Path | str,
    input_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a passing V8 receipt before any protected-data access."""

    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
    else:
        receipt_path = _regular_file(receipt, description="V8 gate receipt")
        payload = _load_json(receipt_path, description="V8 gate receipt")
    _require(payload.get("schema") == GATE_RECEIPT_SCHEMA, "V8 receipt schema differs")
    _validate_self_hash(payload, field="receipt_sha256")
    _require(payload.get("contract") == GATE_CONTRACT, "V8 receipt contract differs")
    _require(payload.get("task") == TASK_NAME, "V8 receipt task differs")
    _require(payload.get("objective") == V8_OBJECTIVE, "V8 receipt objective differs")
    _require(payload.get("status") == "pass", "V8 Hard32 requires a passing receipt")
    gate = payload.get("gate")
    _require(
        isinstance(gate, Mapping)
        and gate.get("status") == "pass"
        and gate.get("all_gates_passed") is True
        and gate.get("hard32_authorized") is True,
        "V8 receipt gate did not pass",
    )
    current_inputs = (
        validate_v8_train_inputs()
        if input_contract is None
        else dict(input_contract)
    )
    _require(
        payload.get("training_sources") == current_inputs["artifacts"],
        "V8 receipt training-source bindings differ",
    )
    _require(
        payload.get("v8_source_manifest_sha256")
        == current_inputs["v8_source_manifest_sha256"]
        and payload.get("v8_schedule_entries_sha256")
        == current_inputs["v8_schedule_entries_sha256"],
        "V8 receipt source or schedule hash differs",
    )
    _require(
        payload.get("benchmark_selection_lock") == current_inputs["benchmark_lock"],
        "V8 receipt benchmark-selection lock differs",
    )
    current_checkpoint = validate_v8_checkpoint(
        memory_dir,
        input_contract=current_inputs,
    )
    _require(
        payload.get("checkpoint") == current_checkpoint,
        "V8 receipt checkpoint binding differs",
    )
    _require(payload.get("code") == evaluator_code_binding(), "V8 evaluator code differs")
    authorization = payload.get("hard32_authorization")
    expected_authorization = {
        "authorization_kind": HARD32_AUTHORIZATION_KIND,
        "authorized": True,
        "scope": HARD32_AUTHORIZATION_SCOPE,
        "checkpoint_binding": current_checkpoint,
        "protected_benchmark_binding": current_inputs["benchmark_lock"][
            "protected_benchmark"
        ],
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    _require(authorization == expected_authorization, "V8 Hard32 authorization differs")
    _validate_receipt_outputs(payload, input_contract=current_inputs)
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = sha256_file(receipt_path)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--normal-fusion-profile", default="native", choices=("native",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate V8 data/checkpoint lineage without loading a model or writing output.",
    )
    return parser.parse_args(argv)


def _manifest_is_valid(
    manifest: Mapping[str, Any], *, expected_fingerprint: str
) -> dict[str, Any]:
    _require(manifest.get("schema") == GATE_MANIFEST_SCHEMA, "V8 manifest schema differs")
    payload = manifest.get("fingerprint_payload")
    _require(isinstance(payload, dict), "V8 manifest fingerprint payload is missing")
    _require(
        fingerprint_payload_sha256(payload) == manifest.get("fingerprint"),
        "V8 manifest self-fingerprint differs",
    )
    _require(manifest.get("fingerprint") == expected_fingerprint, "V8 manifest differs")
    return dict(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(
        args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS,
        "V8 gate requires max_new_tokens=128",
    )
    _require(args.expected_memory_layer_count == 42, "V8 gate requires all 42 layers")
    args.delta_mem_root = str(Path(args.delta_mem_root).expanduser().resolve())
    _require(Path(args.delta_mem_root) == PROJECT_ROOT, "V8 gate requires this checkout")
    input_contract = validate_v8_train_inputs()
    checkpoint = validate_v8_checkpoint(
        args.memory_dir,
        input_contract=input_contract,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "model_loaded": False,
                    "output_created": False,
                    "hard32_access": "not_resolved_not_opened_not_hashed",
                    "checkpoint": checkpoint,
                    "training_sources": input_contract["artifacts"],
                    "benchmark_selection_lock": input_contract["benchmark_lock"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    rows = input_contract["rows"]
    donor_by_ordinal = input_contract["pairing"]["donor_by_ordinal"]
    memory_dir = args.memory_dir.expanduser().resolve()
    args.memory_dir = memory_dir
    output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = output_dir
    base_model = Path(args.base_model).expanduser().resolve()
    args.base_model = str(base_model)
    expected_layers = resolved_memory_layer_count(
        memory_dir,
        args.expected_memory_layer_count,
    )
    fingerprint_payload = {
        "schema_version": 1,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "split": "train",
        "training_sources": input_contract["artifacts"],
        "benchmark_selection_lock": input_contract["benchmark_lock"],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "rows": [
            {
                "train_row_ordinal": row["train_row_ordinal"],
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "donor_train_row_ordinal": donor_by_ordinal[
                    row["train_row_ordinal"]
                ],
            }
            for row in rows
        ],
        "checkpoint": checkpoint,
        "base_model": str(base_model),
        "base_model_weights": base_model_weight_identity(base_model),
        "base_model_prompt_artifacts": base_model_prompt_identity(base_model),
        "expected_memory_layer_count": expected_layers,
        "runtime": {
            "conditions": list(CONDITIONS),
            "semantic_selected_token_ordinals": list(VALUE14_ORDINALS),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache_generation": True,
            "prime_use_cache": False,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "normal_fusion_profile": "native",
            "packages": runtime_package_versions(),
        },
        "objective": dict(V8_OBJECTIVE),
        "code": evaluator_code_binding(),
    }
    fingerprint = fingerprint_payload_sha256(fingerprint_payload)
    manifest = {
        "schema": GATE_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "hard32_access": "not_resolved_not_opened_not_hashed",
    }
    output_paths = {
        condition: output_dir / f"{condition}.jsonl" for condition in CONDITIONS
    }
    output_paths.update(
        {
            "manifest": output_dir / "manifest.json",
            "summary": output_dir / "summary.json",
            "receipt": output_dir / "gate_receipt.json",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
    manifest_path = output_paths["manifest"]
    if manifest_path.exists():
        manifest = _manifest_is_valid(
            _load_json(manifest_path, description="existing V8 manifest"),
            expected_fingerprint=fingerprint,
        )
    else:
        atomic_write_json(manifest_path, manifest)

    completed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = output_paths[condition]
        records = _read_jsonl(path, description=f"V8 {condition} output") if path.exists() else []
        completed[condition] = validate_resume_records(
            records,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
        )

    if any(len(completed[condition]) < 32 for condition in CONDITIONS):
        model, tokenizer, runtime_profile = load_adapter_model(args, expected_layers)
        runtime_prefixes = v7.validate_runtime_prefixes(tokenizer, rows=rows)
        if "runtime_prefixes" in manifest:
            _require(manifest["runtime_prefixes"] == runtime_prefixes, "V8 prefixes differ")
        else:
            manifest["runtime_prefixes"] = runtime_prefixes
        if "runtime_fusion_profile" in manifest:
            _require(
                manifest["runtime_fusion_profile"] == runtime_profile,
                "V8 runtime profile differs",
            )
        else:
            manifest["runtime_fusion_profile"] = runtime_profile
        atomic_write_json(manifest_path, manifest)
        try:
            for condition in CONDITIONS:
                for ordinal, sample in enumerate(rows):
                    if ordinal in completed[condition]:
                        continue
                    donor_ordinal = donor_by_ordinal[ordinal]
                    donor_sample = rows[donor_ordinal]
                    result = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_sample,
                        condition=condition,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=ordinal in VALUE14_SET,
                    )
                    record = _record_with_self_hash(
                        {
                            "schema": GATE_RECORD_SCHEMA,
                            "status": "ok",
                            "completed_at": utc_now(),
                            "fingerprint": fingerprint,
                            "condition": condition,
                            "task": TASK_NAME,
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
                        }
                    )
                    completed[condition][ordinal] = record
                    atomic_write_jsonl(
                        output_paths[condition],
                        [completed[condition][index] for index in sorted(completed[condition])],
                    )
        finally:
            del model
            del tokenizer
            clear_model_memory()

    ordered_records = {
        condition: [completed[condition][index] for index in range(32)]
        for condition in CONDITIONS
    }
    gate = build_v8_gate(
        records_by_condition=ordered_records,
        pairing=input_contract["pairing"],
    )
    summaries = {
        condition: summarize_records(records)
        for condition, records in ordered_records.items()
    }
    summary: dict[str, Any] = {
        "schema": GATE_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "complete": True,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "split": "train",
        "conditions": summaries,
        "comparisons": build_comparisons(summaries),
        "gate": gate,
        "hard32_access": "not_resolved_not_opened_not_hashed",
    }
    summary["summary_sha256"] = self_hash_payload(
        summary,
        hash_field="summary_sha256",
    )
    atomic_write_json(output_paths["summary"], summary)
    receipt = build_gate_receipt(
        output_dir=output_dir,
        fingerprint=fingerprint,
        input_contract=input_contract,
        checkpoint=checkpoint,
        gate=gate,
    )
    atomic_write_canonical_json(output_paths["receipt"], receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
