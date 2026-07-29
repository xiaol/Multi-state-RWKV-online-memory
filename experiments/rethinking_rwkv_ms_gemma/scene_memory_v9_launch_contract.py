#!/usr/bin/env python3
"""Fail-closed launch contract for Scene Memory V9 pair training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v9_data import (
    CANONICAL_VALUE14_PAIRS,
    CHECKPOINT_STEPS,
    CURRICULUM_BINDING_SCHEMA,
    DEFAULT_OUTPUT_DIR as DATA_ROOT,
    PAIR_SCHEDULE_ENTRY_SCHEMA,
    SOURCE_SCHEMA,
    TOTAL_PAIR_STEPS,
    TRAIN32_SHA256,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v9_data_contract import (
    validate_bundle,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v9_warm_start import (
    CONTINUATION_LINEAGE_FILENAME,
    DEFAULT_LOCK_PATH as WARM_START_LOCK,
    RECEIPT_SCHEMA as WARM_START_RECEIPT_SCHEMA,
    SOURCE_IMPORT_POLICY,
    WARM_START_LINEAGE_FILENAME,
    WARM_START_MODE,
    load_v9_warm_start_lock,
    prepare_v9_v8_checkpoint56_warm_start,
)


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v9_source_lock.json")

OBJECTIVE_VERSION = (
    "scene_state_generation_ce_symmetric_full_gold_selected_vocab_aligned_prefix_v3"
)
PAIRING_OBJECTIVE_VERSION = "scene_state_generation_ce_v1"
OBJECTIVE_SCHEMA_VERSION = 12
PAIR_PHYSICAL_BATCH_SIZE = 1
PAIR_LOGICAL_BATCH_SIZE = 2
PAIR_DIRECTIONAL_EXPOSURES = 2
SAVE_STEPS = 7
WARMUP_STEPS = 4
WARMUP_RATIO = 0.0
LEARNING_RATE = 2e-4
PREFIX_CORRECTION_WEIGHT = 0.5
FIXED_SAMPLER_MODE = "explicit_ordered_v9_canonical_low_pair_v1"

CONTINUATION_LINEAGE_SCHEMA_VERSION = 1
ABLATION_LINEAGE_FILENAME = "ablation_lineage_manifest.json"
REQUIRED_CHECKPOINT_ARTIFACTS = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_protocol.json",
    "scene_state_identity_pairing_manifest.json",
)


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


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


def load_object(path: Path, *, description: str) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"{description}_missing_or_symlink",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"{description}_invalid_json") from exc
    require(isinstance(payload, dict), f"{description}_must_be_object")
    return payload


def require_sha256(value: Any, *, description: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{description}_invalid",
    )
    return value


def require_ssd(
    path: Path,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    resolved = path.expanduser().resolve()
    root = ssd_root.expanduser().resolve()
    require(
        resolved == root or root in resolved.parents,
        f"{description}_must_stay_on_2t_ssd path={resolved}",
    )
    forbidden_parts = {"hard32", "eval", "evaluation", "validation"}
    require(
        not forbidden_parts.intersection(part.lower() for part in resolved.parts),
        f"{description}_forbidden_hard32_or_evaluation_path path={resolved}",
    )
    return resolved


def _require_regular_artifact(path: Path, *, description: str) -> Path:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"{description}_missing_empty_or_symlink",
    )
    return path


def _validate_schedule_entries(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == TOTAL_PAIR_STEPS, "v9_schedule_row_count_differs")
    require(all(line.strip() for line in lines), "v9_schedule_contains_blank_rows")
    entries: list[dict[str, Any]] = []
    for schedule_index, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchContractError(
                f"v9_schedule_invalid_json row={schedule_index}"
            ) from exc
        require(isinstance(entry, dict), f"v9_schedule_entry_not_object row={schedule_index}")
        unsigned = dict(entry)
        entry_sha256 = unsigned.pop("entry_sha256", None)
        require(
            entry.get("schema") == PAIR_SCHEDULE_ENTRY_SCHEMA
            and entry_sha256 == canonical_sha256(unsigned),
            f"v9_schedule_entry_identity_differs row={schedule_index}",
        )
        pair = entry.get("canonical_pair_ordinals")
        members = entry.get("members")
        require(
            entry.get("schedule_index") == schedule_index
            and entry.get("step") == schedule_index + 1
            and entry.get("pair_batch_size") == PAIR_LOGICAL_BATCH_SIZE
            and isinstance(pair, list)
            and len(pair) == 2
            and isinstance(members, list)
            and len(members) == PAIR_LOGICAL_BATCH_SIZE,
            f"v9_schedule_pair_indexing_differs row={schedule_index}",
        )
        require(
            [member.get("train_row_ordinal") for member in members] == pair
            and [member.get("donor_train_row_ordinal") for member in members]
            == list(reversed(pair)),
            f"v9_schedule_reciprocal_pair_differs row={schedule_index}",
        )
        entries.append(entry)
    return entries


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    resolved_root = require_ssd(
        data_root,
        description="v9_data_root",
        ssd_root=ssd_root,
    )
    try:
        validated = validate_bundle(
            resolved_root,
            source_lock_path=source_lock_path.expanduser().resolve(),
        )
    except Exception as exc:
        raise LaunchContractError(f"v9_data_contract_failed: {exc}") from exc
    require(validated.get("status") == "pass", "v9_data_contract_status_differs")
    require(validated.get("pair_steps") == TOTAL_PAIR_STEPS, "v9_pair_steps_differ")
    require(
        validated.get("directed_presentations")
        == TOTAL_PAIR_STEPS * PAIR_DIRECTIONAL_EXPOSURES,
        "v9_directed_presentations_differ",
    )
    require(
        tuple(validated.get("checkpoint_steps", ())) == tuple(CHECKPOINT_STEPS),
        "v9_checkpoint_steps_differ",
    )
    require(validated.get("hard32_rows_in_schedule") == 0, "v9_schedule_contains_hard32")
    exclusion = validated.get("hard32_exclusion")
    require(
        isinstance(exclusion, dict)
        and exclusion.get("included") is False
        and exclusion.get("path") is None
        and exclusion.get("sha256") is None,
        "v9_hard32_exclusion_differs",
    )

    artifacts = validated.get("artifacts")
    require(isinstance(artifacts, dict), "v9_validated_artifacts_missing")
    resolved_artifacts: dict[str, Path] = {}
    for name in (
        "bundle_manifest",
        "pair_schedule",
        "pair_schedule_manifest",
        "source_manifest",
    ):
        record = artifacts.get(name)
        require(isinstance(record, dict), f"v9_artifact_binding_missing key={name}")
        path = require_ssd(
            Path(str(record.get("path", ""))),
            description=f"v9_{name}",
            ssd_root=ssd_root,
        )
        _require_regular_artifact(path, description=f"v9_{name}")
        require(sha256_file(path) == record.get("sha256"), f"v9_{name}_hash_differs")
        resolved_artifacts[name] = path

    source_manifest = load_object(
        resolved_artifacts["source_manifest"],
        description="v9_source_manifest",
    )
    require(source_manifest.get("schema") == SOURCE_SCHEMA, "v9_source_schema_differs")
    inputs = source_manifest.get("inputs")
    require(isinstance(inputs, dict), "v9_source_inputs_missing")
    train_record = inputs.get("train32")
    require(isinstance(train_record, dict), "v9_train32_binding_missing")
    train_file = require_ssd(
        Path(str(train_record.get("path", ""))),
        description="v9_train32",
        ssd_root=ssd_root,
    )
    _require_regular_artifact(train_file, description="v9_train32")
    require(
        train_record.get("sha256") == TRAIN32_SHA256
        and sha256_file(train_file) == TRAIN32_SHA256,
        "v9_train32_hash_differs",
    )
    entries = _validate_schedule_entries(resolved_artifacts["pair_schedule"])
    require(
        canonical_sha256(entries) == validated.get("pair_schedule_entries_sha256"),
        "v9_schedule_entries_hash_differs",
    )
    return {
        "data_root": str(resolved_root),
        "train_file": str(train_file),
        "train_file_sha256": TRAIN32_SHA256,
        "source_manifest": str(resolved_artifacts["source_manifest"]),
        "source_manifest_file_sha256": artifacts["source_manifest"]["sha256"],
        "source_manifest_sha256": validated["source_manifest_sha256"],
        "schedule": str(resolved_artifacts["pair_schedule"]),
        "schedule_file_sha256": artifacts["pair_schedule"]["sha256"],
        "schedule_entries_sha256": validated["pair_schedule_entries_sha256"],
        "ordered_pairs_sha256": validated["ordered_pairs_sha256"],
        "schedule_manifest": str(resolved_artifacts["pair_schedule_manifest"]),
        "schedule_manifest_file_sha256": artifacts["pair_schedule_manifest"]["sha256"],
        "schedule_manifest_sha256": validated["pair_schedule_manifest_sha256"],
        "bundle_manifest": str(resolved_artifacts["bundle_manifest"]),
        "bundle_manifest_file_sha256": artifacts["bundle_manifest"]["sha256"],
        "bundle_manifest_sha256": validated["bundle_manifest_sha256"],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "entries": entries,
    }


def validate_warm_start_contract(
    *,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    lock_path = warm_start_lock_path.expanduser().resolve()
    try:
        lock = load_v9_warm_start_lock(lock_path)
        checkpoint = require_ssd(
            Path(str(lock.get("source_checkpoint", ""))),
            description="v9_warm_start_checkpoint",
            ssd_root=ssd_root,
        )
        context = prepare_v9_v8_checkpoint56_warm_start(
            checkpoint,
            lock_path=lock_path,
        )
    except Exception as exc:
        raise LaunchContractError(f"v9_warm_start_contract_failed: {exc}") from exc
    return {
        "warm_start_checkpoint": str(context.checkpoint),
        "warm_start_lock": str(context.lock_path),
        "warm_start_lock_file_sha256": sha256_file(context.lock_path),
        "warm_start_lock_sha256": context.lock["lock_sha256"],
        "warm_start_mode": WARM_START_MODE,
        "lock": context.lock,
    }


def _expected_schedule_protocol(data: Mapping[str, Any]) -> dict[str, Any]:
    pair_indices = [
        list(entry["canonical_pair_ordinals"])
        for entry in data["entries"]
    ]
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
        "canonical_value14_pairs": [
            list(pair) for pair in CANONICAL_VALUE14_PAIRS
        ],
        "total_steps": TOTAL_PAIR_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "pair_indices": pair_indices,
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
        "per_device_train_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "save_steps": SAVE_STEPS,
        "num_train_epochs": 1.0,
        "max_steps": checkpoint_step,
        "train_sampler_seed": None,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.0,
        "scene_generation_generated_prefix_correction_weight": (
            PREFIX_CORRECTION_WEIGHT
        ),
        "scene_generation_pair_physical_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "scene_generation_pair_directional_exposures": (
            PAIR_DIRECTIONAL_EXPOSURES
        ),
    }
    mismatches = [key for key, value in expected.items() if protocol.get(key) != value]
    require(not mismatches, "resume_protocol_differs fields=" + ",".join(mismatches))
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, dict), "resume_protocol_schedule_missing")
    expected_schedule = _expected_schedule_protocol(data)
    schedule_mismatches = [
        key for key, value in expected_schedule.items() if schedule.get(key) != value
    ]
    require(
        not schedule_mismatches,
        "resume_protocol_schedule_differs fields=" + ",".join(schedule_mismatches),
    )


def _validate_checkpoint_config(config: Mapping[str, Any]) -> None:
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
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    require(not mismatches, "resume_delta_config_differs fields=" + ",".join(mismatches))


def _validate_pairing_manifest(pairing: Mapping[str, Any]) -> str:
    unsigned = dict(pairing)
    manifest_sha256 = unsigned.pop("manifest_sha256", None)
    require(
        require_sha256(manifest_sha256, description="resume_pairing_manifest_hash")
        == canonical_sha256(unsigned),
        "resume_pairing_manifest_self_hash_differs",
    )
    require(
        pairing.get("objective_version") == PAIRING_OBJECTIVE_VERSION,
        "resume_pairing_materialization_objective_differs",
    )
    return manifest_sha256


def _validate_warm_lineage(
    lineage: Mapping[str, Any],
    *,
    checkpoint: Path,
    protocol_sha256: str,
    config_sha256: str,
    pairing_sha256: str,
    warm: Mapping[str, Any],
) -> str:
    unsigned = dict(lineage)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    require(
        lineage.get("schema") == WARM_START_RECEIPT_SCHEMA
        and lineage.get("schema_version") == 1
        and lineage.get("mode") == WARM_START_MODE,
        "resume_v9_warm_lineage_schema_or_mode_differs",
    )
    require(
        require_sha256(receipt_sha256, description="resume_v9_warm_receipt_hash")
        == canonical_sha256(unsigned),
        "resume_v9_warm_lineage_self_hash_differs",
    )
    source_lock = lineage.get("source_lock")
    require(isinstance(source_lock, dict), "resume_v9_warm_source_lock_missing")
    require(
        lineage.get("source_checkpoint") == warm["warm_start_checkpoint"]
        and source_lock.get("path") == warm["warm_start_lock"]
        and source_lock.get("lock_sha256") == warm["warm_start_lock_sha256"]
        and lineage.get("source_state_imports") == SOURCE_IMPORT_POLICY
        and lineage.get("post_load_bit_equal") is True,
        "resume_v9_warm_source_binding_differs",
    )
    expected_fresh = {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v9_seed",
    }
    require(
        lineage.get("target_fresh_start") == expected_fresh,
        "resume_v9_warm_fresh_start_differs",
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
    mismatches = [key for key, value in evidence.items() if lineage.get(key) != value]
    require(
        not mismatches,
        "resume_v9_warm_target_evidence_differs fields=" + ",".join(mismatches),
    )
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        isinstance(optimizer_class, str) and optimizer_class.endswith(".AdamW"),
        "resume_v9_warm_optimizer_class_differs",
    )
    return receipt_sha256


def _validate_checkpoint_lineage(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path,
    visited: set[Path] | None = None,
) -> dict[str, Any]:
    requested = checkpoint.expanduser()
    require(not requested.is_symlink(), "resume_checkpoint_must_not_be_symlink")
    resolved = require_ssd(
        requested,
        description="v9_resume_checkpoint",
        ssd_root=ssd_root,
    )
    require(resolved.is_dir(), "resume_checkpoint_missing")
    active_visited = set() if visited is None else visited
    require(resolved not in active_visited, "resume_v9_lineage_cycle")
    active_visited.add(resolved)
    suffix = resolved.name.removeprefix("checkpoint-")
    require(
        resolved.name.startswith("checkpoint-") and suffix.isdigit(),
        "resume_checkpoint_must_be_checkpoint_n",
    )
    checkpoint_step = int(suffix)
    require(checkpoint_step in CHECKPOINT_STEPS, "resume_source_not_locked_v9_endpoint")
    for filename in REQUIRED_CHECKPOINT_ARTIFACTS:
        _require_regular_artifact(
            resolved / filename,
            description=f"resume_{filename}",
        )
    rng_files = sorted(resolved.glob("rng_state*.pth"))
    require(
        bool(rng_files)
        and all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in rng_files),
        "resume_rng_state_missing_empty_or_symlink",
    )

    trainer_state = load_object(resolved / "trainer_state.json", description="resume_trainer_state")
    protocol = load_object(resolved / "training_protocol.json", description="resume_training_protocol")
    config = load_object(resolved / "delta_mem_config.json", description="resume_delta_config")
    pairing = load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="resume_scene_state_pairing",
    )
    require(
        trainer_state.get("global_step") == checkpoint_step
        and trainer_state.get("max_steps") == checkpoint_step
        and protocol.get("max_steps") == checkpoint_step,
        "resume_checkpoint_not_completed_horizon",
    )
    _validate_checkpoint_protocol(protocol, checkpoint_step=checkpoint_step, data=data)
    _validate_checkpoint_config(config)
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing)

    expected_lineage_filename = (
        WARM_START_LINEAGE_FILENAME
        if checkpoint_step == CHECKPOINT_STEPS[0]
        else CONTINUATION_LINEAGE_FILENAME
    )
    lineage_names = (
        WARM_START_LINEAGE_FILENAME,
        CONTINUATION_LINEAGE_FILENAME,
        ABLATION_LINEAGE_FILENAME,
    )
    present = [name for name in lineage_names if (resolved / name).is_file()]
    require(
        present == [expected_lineage_filename],
        "resume_v9_expected_single_lineage_file_differs",
    )
    lineage_path = resolved / expected_lineage_filename
    require(not lineage_path.is_symlink(), "resume_v9_lineage_must_not_be_symlink")
    lineage = load_object(lineage_path, description="resume_v9_lineage")
    if checkpoint_step == CHECKPOINT_STEPS[0]:
        root_receipt_sha256 = _validate_warm_lineage(
            lineage,
            checkpoint=resolved,
            protocol_sha256=protocol_sha256,
            config_sha256=config_sha256,
            pairing_sha256=pairing_sha256,
            warm=warm,
        )
    else:
        unsigned = dict(lineage)
        manifest_sha256 = unsigned.pop("manifest_sha256", None)
        require(
            lineage.get("schema_version") == CONTINUATION_LINEAGE_SCHEMA_VERSION
            and lineage.get("mode") == "extend",
            "resume_v9_continuation_schema_or_mode_differs",
        )
        require(
            require_sha256(
                manifest_sha256,
                description="resume_v9_continuation_hash",
            )
            == canonical_sha256(unsigned),
            "resume_v9_continuation_self_hash_differs",
        )
        source_step = lineage.get("source_global_step")
        source_index = CHECKPOINT_STEPS.index(checkpoint_step) - 1
        expected_source_step = CHECKPOINT_STEPS[source_index]
        require(
            source_step == expected_source_step
            and lineage.get("target_max_steps") == checkpoint_step
            and lineage.get("target_training_protocol_sha256") == protocol_sha256,
            "resume_v9_continuation_horizon_or_protocol_differs",
        )
        source_checkpoint_raw = lineage.get("source_checkpoint")
        source_checkpoint = Path(str(source_checkpoint_raw)).expanduser().resolve()
        require(
            source_checkpoint_raw == str(source_checkpoint),
            "resume_v9_continuation_source_not_canonical",
        )
        source_lineage = _validate_checkpoint_lineage(
            source_checkpoint,
            data=data,
            warm=warm,
            ssd_root=ssd_root,
            visited=active_visited,
        )
        require(
            source_lineage["checkpoint_step"] == expected_source_step
            and lineage.get("source_training_protocol_sha256")
            == source_lineage["training_protocol_sha256"]
            and lineage.get("source_lineage_filename")
            == source_lineage["lineage_filename"]
            and lineage.get("source_lineage_file_sha256")
            == source_lineage["lineage_file_sha256"]
            and lineage.get("root_warm_start_receipt_sha256")
            == source_lineage["root_warm_start_receipt_sha256"],
            "resume_v9_continuation_source_lineage_differs",
        )
        root_receipt_sha256 = source_lineage["root_warm_start_receipt_sha256"]
    active_visited.remove(resolved)
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": checkpoint_step,
        "lineage_filename": expected_lineage_filename,
        "lineage_file_sha256": sha256_file(lineage_path),
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt_sha256,
    }


def validate_resume_contract(
    *,
    resume_checkpoint: Path,
    target_step: int,
    gate_receipt: Path | None,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    require(gate_receipt is not None, "resume_requires_explicit_v9_gate_receipt")
    lineage = _validate_checkpoint_lineage(
        resume_checkpoint,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )
    source_step = lineage["checkpoint_step"]
    source_index = CHECKPOINT_STEPS.index(source_step)
    require(source_index + 1 < len(CHECKPOINT_STEPS), "resume_final_v9_checkpoint_has_no_successor")
    require(
        target_step == CHECKPOINT_STEPS[source_index + 1],
        "resume_target_is_not_next_locked_v9_endpoint",
    )
    resolved_gate_receipt = require_ssd(
        gate_receipt,
        description="v9_progression_gate_receipt",
        ssd_root=ssd_root,
    )
    try:
        from experiments.rethinking_rwkv_ms_gemma.run_scene_memory_v9_gate import (
            validate_continuation_authorization,
        )

        gate_authorization = validate_continuation_authorization(
            resolved_gate_receipt,
            source_checkpoint=lineage["checkpoint"],
            target_step=target_step,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise LaunchContractError(
            f"v9_progression_gate_authorization_failed: {exc}"
        ) from exc
    require(
        gate_authorization.get("source_checkpoint") == lineage["checkpoint"]
        and gate_authorization.get("source_step") == source_step
        and gate_authorization.get("target_step") == target_step
        and gate_authorization.get("hard32_authorized") is False,
        "v9_progression_gate_binding_differs",
    )
    entry = data["entries"][source_step]
    return {
        "launch_mode": "resume",
        "source_step": source_step,
        "target_step": target_step,
        "resume_checkpoint": lineage["checkpoint"],
        "resume_schedule_cursor": source_step,
        "next_pair_low_ordinal": entry["canonical_pair_ordinals"][0],
        "next_pair_high_ordinal": entry["canonical_pair_ordinals"][1],
        "next_schedule_entry_sha256": entry["entry_sha256"],
        "root_warm_start_receipt_sha256": lineage[
            "root_warm_start_receipt_sha256"
        ],
        "gate_authorization_kind": gate_authorization["authorization_kind"],
        "gate_receipt": gate_authorization["gate_receipt"],
        "gate_receipt_file_sha256": gate_authorization[
            "gate_receipt_file_sha256"
        ],
        "gate_receipt_sha256": gate_authorization["gate_receipt_sha256"],
    }


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    """Validate a completed V9 endpoint without requiring a successor."""
    return _validate_checkpoint_lineage(
        checkpoint,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )


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
    if smoke:
        require(target_step == 1, "smoke_launch_must_target_step1")
        require(resume_checkpoint is None, "smoke_launch_forbids_resume")
        require(gate_receipt is None, "smoke_launch_forbids_gate_receipt")
    else:
        require(target_step in CHECKPOINT_STEPS, "target_step_not_locked_v9_endpoint")
    data = validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    warm = validate_warm_start_contract(
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    if resume_checkpoint is not None:
        cursor = validate_resume_contract(
            resume_checkpoint=resume_checkpoint,
            target_step=target_step,
            gate_receipt=gate_receipt,
            data=data,
            warm=warm,
            ssd_root=ssd_root,
        )
    else:
        require(gate_receipt is None, "fresh_launch_forbids_gate_receipt")
        require(
            smoke or target_step == CHECKPOINT_STEPS[0],
            "fresh_launch_must_target_step7",
        )
        entry = data["entries"][0]
        cursor = {
            "launch_mode": "warm_start_smoke" if smoke else "warm_start",
            "source_step": 0,
            "target_step": target_step,
            "resume_checkpoint": None,
            "resume_schedule_cursor": 0,
            "next_pair_low_ordinal": entry["canonical_pair_ordinals"][0],
            "next_pair_high_ordinal": entry["canonical_pair_ordinals"][1],
            "next_schedule_entry_sha256": entry["entry_sha256"],
            "root_warm_start_receipt_sha256": None,
            "gate_authorization_kind": "not_required_fresh_start",
            "gate_receipt": "not_required_fresh_start",
            "gate_receipt_file_sha256": "not_required_fresh_start",
            "gate_receipt_sha256": "not_required_fresh_start",
        }
    public_data = {key: value for key, value in data.items() if key != "entries"}
    public_warm = {key: value for key, value in warm.items() if key != "lock"}
    return {
        **public_data,
        **public_warm,
        **cursor,
        "total_pair_steps": TOTAL_PAIR_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "objective_version": OBJECTIVE_VERSION,
        "pairing_objective_version": PAIRING_OBJECTIVE_VERSION,
        "pair_physical_batch_size": PAIR_PHYSICAL_BATCH_SIZE,
        "pair_logical_batch_size": PAIR_LOGICAL_BATCH_SIZE,
        "pair_directional_exposures": PAIR_DIRECTIONAL_EXPOSURES,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "save_steps": 1 if smoke else SAVE_STEPS,
        "hard32_access": "forbidden_not_resolved_opened_or_hashed",
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
    rendered = tuple(str(field) for field in fields)
    require(
        all("\t" not in field and "\n" not in field for field in rendered),
        "v9_launch_contract_tsv_control_character",
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
    if args.format == "tsv":
        print(_tsv(result))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
