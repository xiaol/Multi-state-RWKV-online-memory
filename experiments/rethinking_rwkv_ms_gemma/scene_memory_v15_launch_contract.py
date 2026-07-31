#!/usr/bin/env python3
"""Fail-closed launch contract for Scene Memory V15 all-Train32 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_scene_memory_v15_data as data_prep,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v14_warm_start as warm,
)
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v15_data_contract as data_contract,
)


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_BASE_MODEL = SSD_ROOT / "models/gemma/gemma-4-E4B-it"
DATA_ROOT = (
    SSD_ROOT
    / "delta_mem_data/scene_failure_state/scene_memory_v15/all32_pair64_v1"
)
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v15_source_lock.json")
V7_TRAIN32_ROOT = (
    SSD_ROOT
    / "delta_mem_data/scene_failure_state/"
    "scene_memory_v7_fixed_hard32_aligned_train32_v1"
)
PINNED_TRAIN_FILE = V7_TRAIN32_ROOT / "train32.jsonl"
PINNED_TRAIN_FILE_SHA256 = (
    "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9"
)
SOURCE_MANIFEST = DATA_ROOT / "source_manifest.json"
PAIR_SCHEDULE = DATA_ROOT / "pair_schedule.jsonl"
PAIR_SCHEDULE_MANIFEST = DATA_ROOT / "pair_schedule_manifest.json"
PINNED_PAIR_SCHEDULE_SHA256 = (
    "d59e239e4f29f7981175783dafb8a8e4f34c9c95399000e73f2c3dbd26347421"
)
WARM_START_LOCK = Path(warm.DEFAULT_LOCK_PATH)
PINNED_WARM_START_CHECKPOINT = Path(warm.PINNED_SOURCE_CHECKPOINT)
PINNED_WARM_START_ADAPTER_SHA256 = warm.PINNED_ADAPTER_SHA256

RUN_ROOT = SSD_ROOT / "delta_mem_outputs/novel_rwkv_ms_memory/scene_memory_v15"
CACHE_ROOT = SSD_ROOT / "delta_mem_cache/scene_memory_v15"
_RUN_ROOT_RELATIVE = RUN_ROOT.relative_to(SSD_ROOT)
_CACHE_ROOT_RELATIVE = CACHE_ROOT.relative_to(SSD_ROOT)

OBJECTIVE_VERSION = "scene_state_generation_ce_symmetric_cached_prefix_identity_v15"
OBJECTIVE_SCHEMA_VERSION = 18
ROW_OBJECTIVE_AUDIT_FILENAME = "scene_memory_v15_row_objective.json"
ROW_OBJECTIVE_AUDIT_SCHEMA = "rwkv_ms_scene_memory_v15_row_objective.v1"
PRODUCTION_RUN_MODE = "production_four_all32_pair_cycles_v15"
FIXED_SAMPLER_MODE = "explicit_ordered_v15_full_pair_cycle_v1"
ONE_PAIR_SMOKE_RUN_MODE = "one_pair_real_backward_optimizer_step_smoke_v15"
ONE_PAIR_SMOKE_SAMPLER_MODE = (
    "explicit_ordered_v15_first_materialized_pair_smoke_v1"
)
ONE_PAIR_SMOKE_FLAG = "--scene-state-v15-one-pair-smoke"

FULL_PAIR_CYCLES = data_prep.FULL_PAIR_CYCLES
FOUR_CYCLE_PAIRS = tuple(pair for cycle in FULL_PAIR_CYCLES for pair in cycle)
FOUR_CYCLE_PAIRS_SHA256 = data_prep.PAIR_PREFIX_SHA256_BY_CHECKPOINT["4"]
PAIR_PREFIX_SHA256_BY_CHECKPOINT = {
    int(step): digest
    for step, digest in data_prep.PAIR_PREFIX_SHA256_BY_CHECKPOINT.items()
}
ONE_PAIR_SMOKE_PAIR = FOUR_CYCLE_PAIRS[0]
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 16
TOTAL_PAIR_PRESENTATIONS = 64
TOTAL_OPTIMIZER_STEPS = 4
CHECKPOINT_STEPS = (1, 2, 3, 4)
PRESENTATION_CHECKPOINTS = (16, 32, 48, 64)
GRADIENT_ACCUMULATION_STEPS = 16
SAVE_STEPS = 1
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
OPTIMIZER_IMPLEMENTATION = "adamw_torch_fused"
PAIRING_OBJECTIVE_VERSION = "scene_state_generation_ce_v1"
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0
WARMUP_RATIO = 0.0
LOGGING_STEPS = 1
SEED = 42
DATA_SEED = 42
MAX_LENGTH = 256
MAX_WRITE_LENGTH = 2048
TARGET_LAYERS = tuple(range(42))
DELTA_HEADS = ("q", "o")
CONTINUATION_POLICY = "forbidden_fresh_four_cycle_run_is_terminal"
EVALUATION_ACCESS_POLICY = (
    "forbidden_during_training_no_validation_hard32_full170_or_other_benchmark_v1"
)
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed_during_training"
WARM_START_REUSE_POLICY = (
    "reuse_v14_verified_v13_checkpoint4_adapter_only_loader_and_lock_v1"
)
WARM_START_LINEAGE_FILENAME = warm.WARM_START_LINEAGE_FILENAME

LAUNCH_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v15_attached_launch.v1"
COMPLETION_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v15_attached_completion.v1"
ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v15_one_pair_smoke_launch.v1"
)
ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA = (
    "rwkv_ms_scene_memory_v15_one_pair_smoke_completion.v1"
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
    "experiments/rethinking_rwkv_ms_gemma/prepare_scene_memory_v15_data.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_data_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_source_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_v13_checkpoint4_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_warm_start.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v15_launch_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v15.sh",
)

_PINNED_BASE_PROMPT_ARTIFACTS = {
    "chat_template.jinja": "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    "config.json": "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
    "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
}
_PINNED_BASE_WEIGHT_BYTES = 15_992_595_884
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FORBIDDEN_PATH_COMPONENTS = frozenset(
    {
        "benchmark",
        "benchmarks",
        "eval",
        "evaluation",
        "full170",
        "holdout",
        "test",
        "tests",
        "val",
        "validation",
    }
)


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"cannot_read_{description} path={path}") from exc
    require(isinstance(payload, dict), f"{description}_must_be_json_object")
    return payload


def _load_v15_pair_schedule() -> tuple[dict[str, Any], ...]:
    path = _regular_file(PAIR_SCHEDULE, description="v15_pair_schedule")
    require(
        sha256_file(path) == PINNED_PAIR_SCHEDULE_SHA256,
        "v15_pair_schedule_sha256_differs",
    )
    entries: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                require(
                    isinstance(entry, dict),
                    f"v15_pair_schedule_entry_must_be_object line={line_number}",
                )
                entries.append(entry)
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(
            f"cannot_read_v15_pair_schedule path={path}"
        ) from exc
    require(
        len(entries) == TOTAL_PAIR_PRESENTATIONS,
        "v15_pair_schedule_entry_count_differs",
    )
    for index, (entry, pair) in enumerate(zip(entries, FOUR_CYCLE_PAIRS)):
        members = entry.get("members")
        require(
            entry.get("schedule_index") == index
            and entry.get("presentation") == index + 1
            and entry.get("canonical_pair_ordinals") == list(pair)
            and isinstance(members, list)
            and len(members) == 2
            and isinstance(members[0], Mapping)
            and isinstance(members[1], Mapping)
            and members[0].get("train_row_ordinal") == pair[0]
            and members[1].get("train_row_ordinal") == pair[1]
            and isinstance(members[0].get("row_sha256"), str)
            and isinstance(members[1].get("row_sha256"), str),
            f"v15_pair_schedule_entry_binding_differs index={index}",
        )
    return tuple(entries)


def _regular_file(path: Path, *, description: str) -> Path:
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
        f"{description}_missing_empty_or_symlink path={path}",
    )
    return path


def artifact_binding(path: Path, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path.expanduser().resolve(), description=description)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _has_forbidden_component(path: Path) -> bool:
    for component in path.parts:
        lowered = component.lower()
        stem = Path(lowered).stem
        if (
            lowered in _FORBIDDEN_PATH_COMPONENTS
            or stem in _FORBIDDEN_PATH_COMPONENTS
            or lowered.startswith("hard32")
            or stem.startswith("hard32")
            or lowered.startswith("full170")
            or stem.startswith("full170")
        ):
            return True
    return False


def guard_training_path(
    path: Path | str,
    *,
    description: str,
    allowed_protected_paths: Sequence[Path] = (),
) -> Path:
    raw = Path(path).expanduser()
    require(raw.is_absolute(), f"{description}_must_be_absolute path={raw}")
    require(".." not in raw.parts, f"{description}_parent_alias_forbidden path={raw}")
    normalized = Path(os.path.normpath(str(raw)))
    allowed = {Path(item).expanduser() for item in allowed_protected_paths}
    require(
        normalized in allowed or not _has_forbidden_component(normalized),
        f"{description}_benchmark_or_protected_split_forbidden path={normalized}",
    )
    return normalized


def _require_no_symlink_components(path: Path, *, description: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.exists() or current.is_symlink():
            require(
                not current.is_symlink(),
                f"{description}_symlink_component_forbidden path={current}",
            )


def require_under_root(
    path: Path | str,
    *,
    root: Path,
    description: str,
) -> Path:
    raw = guard_training_path(path, description=description)
    _require_no_symlink_components(raw, description=description)
    resolved = raw.resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    require(
        resolved == resolved_root or resolved_root in resolved.parents,
        f"{description}_outside_locked_root path={resolved} root={resolved_root}",
    )
    return resolved


def require_v15_run_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path = SSD_ROOT,
) -> Path:
    return require_under_root(
        path,
        root=run_root_for(ssd_root),
        description=description,
    )


def require_exact_path(
    path: Path | str,
    expected: Path,
    *,
    description: str,
    allowed_protected: bool = False,
) -> Path:
    allowed = (expected,) if allowed_protected else ()
    raw = guard_training_path(
        path,
        description=description,
        allowed_protected_paths=allowed,
    )
    _require_no_symlink_components(raw, description=description)
    actual = raw.resolve(strict=False)
    pinned = expected.expanduser().resolve(strict=False)
    require(actual == pinned, f"{description}_differs actual={actual} expected={pinned}")
    return actual


def run_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root).expanduser().resolve(strict=False) / _RUN_ROOT_RELATIVE


def cache_root_for(ssd_root: Path = SSD_ROOT) -> Path:
    return Path(ssd_root).expanduser().resolve(strict=False) / _CACHE_ROOT_RELATIVE


def presentation_cursor(global_step: int) -> int:
    require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step in (0, *CHECKPOINT_STEPS),
        "global_step_outside_v15_schedule",
    )
    return global_step * PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP


def validate_base_model_contract(
    base_model: Path = PINNED_BASE_MODEL,
) -> dict[str, Any]:
    root = require_exact_path(
        base_model,
        PINNED_BASE_MODEL,
        description="v15_base_model",
    )
    require(root.is_dir() and not root.is_symlink(), "v15_base_model_missing_or_symlink")
    prompt_bindings: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in _PINNED_BASE_PROMPT_ARTIFACTS.items():
        binding = artifact_binding(root / name, description=f"v15_base_model_{name}")
        require(
            binding["sha256"] == expected_sha256,
            f"v15_base_model_{name}_sha256_differs",
        )
        prompt_bindings[name] = binding
    weights = _regular_file(
        root / "model.safetensors",
        description="v15_base_model_weights",
    )
    require(
        weights.stat().st_size == _PINNED_BASE_WEIGHT_BYTES,
        "v15_base_model_weight_size_differs",
    )
    return {
        "path": str(root),
        "weight_layout": "unsharded_safetensors",
        "weight_file": "model.safetensors",
        "weight_bytes": weights.stat().st_size,
        "prompt_artifacts": prompt_bindings,
    }


def validate_data_contract(
    *,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
) -> dict[str, Any]:
    root = require_exact_path(data_root, DATA_ROOT, description="v15_data_root")
    lock = require_exact_path(
        source_lock_path,
        SOURCE_LOCK,
        description="v15_repo_source_lock",
    )
    try:
        validated = data_contract.validate_bundle(
            root,
            source_lock_path=lock,
            v7_root=V7_TRAIN32_ROOT,
        )
    except Exception as exc:
        raise LaunchContractError(f"v15_data_contract_failed: {exc}") from exc
    source_manifest = _load_object(
        root / "source_manifest.json",
        description="v15_source_manifest",
    )
    train_record = source_manifest.get("partitions", {}).get("train", {}).get("data")
    require(isinstance(train_record, Mapping), "v15_train_partition_binding_missing")
    train_file = require_exact_path(
        Path(str(train_record.get("path", ""))),
        PINNED_TRAIN_FILE,
        description="v15_train_file",
        allowed_protected=True,
    )
    _regular_file(train_file, description="v15_train_file")
    require(
        train_record.get("sha256") == PINNED_TRAIN_FILE_SHA256
        and sha256_file(train_file) == PINNED_TRAIN_FILE_SHA256,
        "v15_train_file_sha256_differs",
    )
    artifacts = validated.get("artifacts")
    require(isinstance(artifacts, Mapping), "v15_validated_artifacts_missing")
    expected_artifacts = {
        "source_manifest": SOURCE_MANIFEST,
        "pair_schedule": PAIR_SCHEDULE,
        "pair_schedule_manifest": PAIR_SCHEDULE_MANIFEST,
    }
    for name, expected in expected_artifacts.items():
        record = artifacts.get(name)
        require(isinstance(record, Mapping), f"v15_{name}_binding_missing")
        require(
            Path(str(record.get("path", ""))).resolve() == expected.resolve(),
            f"v15_{name}_path_differs",
        )
    require(
        validated.get("scheduled_train_rows") == 32
        and validated.get("base_failure_rows") == 32
        and validated.get("pair_presentations") == TOTAL_PAIR_PRESENTATIONS
        and validated.get("pairs_per_cycle")
        == PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
        and validated.get("pair_cycles") == TOTAL_OPTIMIZER_STEPS
        and validated.get("presentation_checkpoints")
        == list(PRESENTATION_CHECKPOINTS)
        and validated.get("hard32_rows_in_schedule") == 0,
        "v15_data_schedule_cardinality_or_leakage_differs",
    )
    require(
        validated.get("pair_prefix_sha256_by_checkpoint")
        == {str(step): digest for step, digest in PAIR_PREFIX_SHA256_BY_CHECKPOINT.items()},
        "v15_data_schedule_prefix_hashes_differ",
    )
    return {
        **validated,
        "train_file": str(train_file),
        "train_file_sha256": PINNED_TRAIN_FILE_SHA256,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": artifacts["source_manifest"]["sha256"],
        "schedule": str(PAIR_SCHEDULE),
        "schedule_sha256": artifacts["pair_schedule"]["sha256"],
        "schedule_manifest": str(PAIR_SCHEDULE_MANIFEST),
        "schedule_manifest_sha256": artifacts["pair_schedule_manifest"]["sha256"],
        "source_lock_path": str(lock),
        "source_lock_file_sha256": sha256_file(lock),
        "scheduled_pairs": [list(pair) for pair in FOUR_CYCLE_PAIRS],
    }


def validate_warm_start_contract(
    *,
    checkpoint: Path = PINNED_WARM_START_CHECKPOINT,
    warm_start_lock_path: Path = WARM_START_LOCK,
) -> dict[str, Any]:
    source = require_exact_path(
        checkpoint,
        PINNED_WARM_START_CHECKPOINT,
        description="v15_warm_start_checkpoint",
    )
    lock = require_exact_path(
        warm_start_lock_path,
        WARM_START_LOCK,
        description="v15_warm_start_lock",
    )
    try:
        context = warm.prepare_v14_v13_checkpoint4_warm_start(
            source,
            lock_path=lock,
        )
    except Exception as exc:
        raise LaunchContractError(f"v15_warm_start_contract_failed: {exc}") from exc
    require(
        context.source_trainer_state.get("global_step") == 4,
        "v15_warm_start_source_step_differs",
    )
    require(
        warm.SOURCE_IMPORT_POLICY
        == {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
            "global_step": False,
        },
        "v15_warm_start_import_policy_differs",
    )
    source_topology = context.lock.get("adapter_topology")
    require(
        isinstance(source_topology, Mapping),
        "v15_warm_start_adapter_topology_missing",
    )
    return {
        "warm_start_checkpoint": str(source),
        "warm_start_adapter_sha256": PINNED_WARM_START_ADAPTER_SHA256,
        "warm_start_lock": str(lock),
        "warm_start_lock_file_sha256": sha256_file(lock),
        "warm_start_lock_sha256": context.lock["lock_sha256"],
        "warm_start_mode": warm.WARM_START_MODE,
        "warm_start_reuse_policy": WARM_START_REUSE_POLICY,
        "source_global_step": 4,
        "lineage_source": {
            "source_epoch": context.source_trainer_state.get("epoch"),
            "source_protocol_objective_version": (
                context.source_training_protocol.get("memory_objective_version")
            ),
            "source_pairing_objective_version": (
                context.source_pairing_manifest.get("objective_version")
            ),
            "source_row_objective_audit_schema": (
                context.source_row_objective_audit.get("schema")
            ),
            "source_v13_warm_start_receipt_sha256": (
                context.source_warm_start_lineage.get("receipt_sha256")
            ),
            "source_artifacts": context.lock.get("artifacts"),
            "source_adapter_topology": dict(source_topology),
        },
    }


def validate_storage_contract(
    *,
    output_dir: Path,
    cache_root: Path,
    run_name: str,
    target_step: int,
    smoke: bool,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, str]:
    require(
        Path(ssd_root).expanduser().resolve(strict=False) == SSD_ROOT,
        "v15_ssd_root_differs",
    )
    require(bool(_SAFE_RUN_NAME.fullmatch(run_name)), "v15_run_name_is_unsafe")
    kind = "smoke" if smoke else "production"
    run_id = f"scene_memory_v15_{kind}_{run_name}_step{target_step}"
    run_root = run_root_for(ssd_root)
    resolved_output = require_under_root(
        output_dir,
        root=run_root,
        description="v15_output_dir",
    )
    require(
        resolved_output == run_root / run_id,
        "v15_output_dir_name_differs",
    )
    resolved_cache = require_under_root(
        cache_root,
        root=cache_root_for(ssd_root),
        description="v15_cache_root",
    )
    require(
        resolved_cache == cache_root_for(ssd_root),
        "v15_cache_root_differs",
    )
    require(not resolved_output.exists(), f"v15_fresh_output_collision path={resolved_output}")
    return {
        "run_id": run_id,
        "output_dir": str(resolved_output),
        "cache_root": str(resolved_cache),
        "log_dir": str(run_root / "logs"),
        "log_file": str(run_root / "logs" / f"{run_id}.log"),
        "launch_receipt": str(run_root / "logs" / f"{run_id}.launch.json"),
        "completion_receipt": str(
            run_root / "logs" / f"{run_id}.completion.json"
        ),
    }


def _file_binding_without_path(path: Path) -> dict[str, Any]:
    _regular_file(path, description=f"critical_file_{path.name}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def critical_training_code_bindings(
    *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, Any]]:
    root = project_root.expanduser().resolve()
    require(root == PROJECT_ROOT, "v15_critical_code_requires_exact_project_root")
    return {
        relative: _file_binding_without_path(root / relative)
        for relative in CRITICAL_TRAINING_FILES
    }


def _resolve_git_commit(commit: object, *, project_root: Path = PROJECT_ROOT) -> str:
    require(isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-fA-F]{40}", commit)), "v15_git_commit_invalid")
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=project_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, "v15_git_commit_missing")
    return result.stdout.strip()


def critical_training_code_bindings_at_commit(
    commit: object,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    source_commit = _resolve_git_commit(commit, project_root=project_root)
    bindings: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_TRAINING_FILES:
        result = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(result.returncode == 0, f"v15_critical_file_missing_at_commit path={relative}")
        bindings[relative] = {
            "bytes": len(result.stdout),
            "sha256": hashlib.sha256(result.stdout).hexdigest(),
        }
    return bindings


def validate_launch_contract(
    *,
    target_step: int,
    run_name: str,
    output_dir: Path,
    cache_root: Path = CACHE_ROOT,
    resume_checkpoint: Path | None = None,
    benchmark_paths: Sequence[Path] = (),
    smoke: bool = False,
    data_root: Path = DATA_ROOT,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    warm_start_checkpoint: Path = PINNED_WARM_START_CHECKPOINT,
    base_model_path: Path = PINNED_BASE_MODEL,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    expected_target = 1 if smoke else TOTAL_OPTIMIZER_STEPS
    require(
        target_step == expected_target,
        "v15_smoke_target_step_must_be_one"
        if smoke
        else "v15_target_step_must_be_four",
    )
    require(resume_checkpoint is None, "v15_resume_is_forbidden")
    require(not benchmark_paths, "v15_benchmark_or_validation_access_is_forbidden")
    storage = validate_storage_contract(
        output_dir=output_dir,
        cache_root=cache_root,
        run_name=run_name,
        target_step=target_step,
        smoke=smoke,
        ssd_root=ssd_root,
    )
    data = validate_data_contract(
        data_root=data_root,
        source_lock_path=source_lock_path,
    )
    warm_start = validate_warm_start_contract(
        checkpoint=warm_start_checkpoint,
        warm_start_lock_path=warm_start_lock_path,
    )
    base_model = validate_base_model_contract(base_model_path)
    scheduled_pairs = (ONE_PAIR_SMOKE_PAIR,) if smoke else FOUR_CYCLE_PAIRS
    gradient_accumulation = 1 if smoke else GRADIENT_ACCUMULATION_STEPS
    return {
        **storage,
        **{key: value for key, value in data.items() if key != "artifacts"},
        **warm_start,
        "base_model_identity": base_model,
        "launch_mode": "warm_start_smoke" if smoke else "warm_start",
        "run_mode": ONE_PAIR_SMOKE_RUN_MODE if smoke else PRODUCTION_RUN_MODE,
        "production_eligible": not smoke,
        "source_step": 0,
        "source_checkpoint_step": 4,
        "target_step": expected_target,
        "resume_checkpoint": None,
        "resume_schedule_cursor": 0,
        "objective_version": OBJECTIVE_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "train_sampler_mode": (
            ONE_PAIR_SMOKE_SAMPLER_MODE if smoke else FIXED_SAMPLER_MODE
        ),
        "gradient_accumulation_steps": gradient_accumulation,
        "max_steps": expected_target,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 1 if smoke else len(CHECKPOINT_STEPS),
        "total_pair_presentations": 1 if smoke else TOTAL_PAIR_PRESENTATIONS,
        "checkpoint_steps": [1] if smoke else list(CHECKPOINT_STEPS),
        "presentation_checkpoints": [1]
        if smoke
        else list(PRESENTATION_CHECKPOINTS),
        "scheduled_pairs": [list(pair) for pair in scheduled_pairs],
        "scheduled_pairs_sha256": canonical_sha256(
            [list(pair) for pair in scheduled_pairs]
        ),
        "full_schedule_pairs": [list(pair) for pair in FOUR_CYCLE_PAIRS],
        "full_schedule_pairs_sha256": FOUR_CYCLE_PAIRS_SHA256,
        "first_pair_low_ordinal": scheduled_pairs[0][0],
        "first_pair_high_ordinal": scheduled_pairs[0][1],
        "target_layers": list(TARGET_LAYERS),
        "delta_heads": list(DELTA_HEADS),
        "training_continuation": CONTINUATION_POLICY,
        "evaluation_access": EVALUATION_ACCESS_POLICY,
        "hard32_access": HARD32_ACCESS_POLICY,
    }


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    return canonical_sha256(unsigned)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_launch_receipt(
    launch: Mapping[str, Any],
    *,
    git_commit: str,
    critical_files: Mapping[str, Any],
) -> dict[str, Any]:
    smoke = not bool(launch["production_eligible"])
    output = Path(str(launch["output_dir"]))
    checkpoint_steps = (1,) if smoke else CHECKPOINT_STEPS
    payload: dict[str, Any] = {
        "schema": ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA if smoke else LAUNCH_RECEIPT_SCHEMA,
        "run_name": output.name.split("_smoke_", 1)[-1].removesuffix("_step1")
        if smoke
        else output.name.split("_production_", 1)[-1].removesuffix("_step4"),
        "attached_foreground_execution": True,
        "launch_mode": launch["launch_mode"],
        "run_mode": launch["run_mode"],
        "production_eligible": not smoke,
        "source_step": 0,
        "source_checkpoint_step": 4,
        "target_step": launch["target_step"],
        "resume_checkpoint": None,
        "trainer_output": str(output),
        "checkpoints": {
            f"checkpoint-{step}": str(output / f"trainer/checkpoint-{step}")
            for step in checkpoint_steps
        },
        "log_file": launch["log_file"],
        "objective": OBJECTIVE_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "train_file": launch["train_file"],
        "train_file_sha256": launch["train_file_sha256"],
        "source_manifest": launch["source_manifest"],
        "source_manifest_sha256": launch["source_manifest_sha256"],
        "source_lock": launch["source_lock_path"],
        "source_lock_file_sha256": launch["source_lock_file_sha256"],
        "schedule": launch["schedule"],
        "schedule_sha256": launch["schedule_sha256"],
        "scheduled_pairs": launch["scheduled_pairs"],
        "scheduled_pairs_sha256": launch["scheduled_pairs_sha256"],
        "full_schedule_pairs_sha256": FOUR_CYCLE_PAIRS_SHA256,
        "total_pair_presentations": launch["total_pair_presentations"],
        "gradient_accumulation_steps": launch["gradient_accumulation_steps"],
        "max_steps": launch["max_steps"],
        "save_steps": SAVE_STEPS,
        "save_total_limit": launch["save_total_limit"],
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": OPTIMIZER_IMPLEMENTATION,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": "constant",
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": LOGGING_STEPS,
        "seed": SEED,
        "data_seed": DATA_SEED,
        "target_layers": list(TARGET_LAYERS),
        "delta_heads": list(DELTA_HEADS),
        "warm_start_checkpoint": launch["warm_start_checkpoint"],
        "warm_start_adapter_sha256": launch["warm_start_adapter_sha256"],
        "warm_start_mode": launch["warm_start_mode"],
        "warm_start_lock": launch["warm_start_lock"],
        "warm_start_lock_sha256": launch["warm_start_lock_sha256"],
        "warm_start_reuse_policy": WARM_START_REUSE_POLICY,
        "base_model_identity": launch["base_model_identity"],
        "critical_files": dict(critical_files),
        "tracked_worktree_clean": True,
        "git_commit": git_commit,
        "training_continuation": CONTINUATION_POLICY,
        "evaluation_access": EVALUATION_ACCESS_POLICY,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    payload["receipt_sha256"] = _receipt_hash(payload)
    return payload


def write_launch_receipt(
    path: Path,
    launch: Mapping[str, Any],
    *,
    git_commit: str,
    critical_files: Mapping[str, Any],
) -> Path:
    return _write_json_exclusive(
        path,
        build_launch_receipt(
            launch,
            git_commit=git_commit,
            critical_files=critical_files,
        ),
    )


def validate_launch_receipt(
    path: Path,
    *,
    checkpoint: Path | None = None,
    base_model_identity: Mapping[str, Any] | None = None,
    smoke: bool | None = None,
    data: Mapping[str, Any] | None = None,
    warm_start: Mapping[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
    **_unused: Any,
) -> dict[str, Any]:
    payload = _load_object(path, description="v15_launch_receipt")
    require(payload.get("receipt_sha256") == _receipt_hash(payload), "v15_launch_receipt_self_hash_differs")
    schema = payload.get("schema")
    require(
        schema in (LAUNCH_RECEIPT_SCHEMA, ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA),
        "v15_launch_receipt_schema_differs",
    )
    receipt_smoke = schema == ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA
    if smoke is not None:
        require(
            receipt_smoke is smoke,
            "v15_launch_receipt_mode_differs_from_completion",
        )
    expected_schema = (
        ONE_PAIR_SMOKE_LAUNCH_RECEIPT_SCHEMA if receipt_smoke else LAUNCH_RECEIPT_SCHEMA
    )
    expected_steps = 1 if receipt_smoke else TOTAL_OPTIMIZER_STEPS
    expected_accumulation = 1 if receipt_smoke else GRADIENT_ACCUMULATION_STEPS
    expected_presentations = 1 if receipt_smoke else TOTAL_PAIR_PRESENTATIONS
    expected_pairs = (
        [list(ONE_PAIR_SMOKE_PAIR)]
        if receipt_smoke
        else [list(pair) for pair in FOUR_CYCLE_PAIRS]
    )
    expected_launch_mode = "warm_start_smoke" if receipt_smoke else "warm_start"
    expected_run_mode = ONE_PAIR_SMOKE_RUN_MODE if receipt_smoke else PRODUCTION_RUN_MODE
    require(
        payload.get("schema") == expected_schema
        and payload.get("objective") == OBJECTIVE_VERSION
        and payload.get("objective_schema_version") == OBJECTIVE_SCHEMA_VERSION
        and payload.get("launch_mode") == expected_launch_mode
        and payload.get("run_mode") == expected_run_mode
        and payload.get("production_eligible") is (not receipt_smoke)
        and payload.get("source_step") == 0
        and payload.get("source_checkpoint_step") == 4
        and payload.get("target_step") == expected_steps
        and payload.get("max_steps") == expected_steps
        and payload.get("gradient_accumulation_steps") == expected_accumulation
        and payload.get("total_pair_presentations") == expected_presentations
        and payload.get("scheduled_pairs") == expected_pairs
        and payload.get("scheduled_pairs_sha256") == canonical_sha256(expected_pairs)
        and payload.get("resume_checkpoint") is None,
        "v15_launch_receipt_schedule_or_objective_differs",
    )
    run_name = payload.get("run_name")
    require(
        isinstance(run_name, str) and bool(_SAFE_RUN_NAME.fullmatch(run_name)),
        "v15_launch_receipt_run_name_is_unsafe",
    )
    trainer_output = require_under_root(
        Path(str(payload.get("trainer_output", ""))),
        root=RUN_ROOT,
        description="v15_launch_receipt_trainer_output",
    )
    kind = "smoke" if receipt_smoke else "production"
    run_id = f"scene_memory_v15_{kind}_{run_name}_step{expected_steps}"
    require(
        trainer_output == RUN_ROOT.resolve(strict=False) / run_id,
        "v15_launch_receipt_trainer_output_differs",
    )
    expected_checkpoints = {
        f"checkpoint-{step}": str(trainer_output / f"trainer/checkpoint-{step}")
        for step in ((1,) if receipt_smoke else CHECKPOINT_STEPS)
    }
    checkpoints = payload.get("checkpoints")
    require(
        isinstance(checkpoints, Mapping)
        and list(checkpoints) == list(expected_checkpoints)
        and dict(checkpoints) == expected_checkpoints,
        "v15_launch_receipt_checkpoint_paths_differ",
    )
    expected_log = RUN_ROOT.resolve(strict=False) / "logs" / f"{run_id}.log"
    expected_receipt = RUN_ROOT.resolve(strict=False) / "logs" / f"{run_id}.launch.json"
    require(
        Path(str(payload.get("log_file", ""))).resolve(strict=False) == expected_log
        and path.expanduser().resolve(strict=False) == expected_receipt,
        "v15_launch_receipt_log_or_path_differs",
    )
    require(
        payload.get("target_layers") == list(TARGET_LAYERS)
        and payload.get("delta_heads") == list(DELTA_HEADS)
        and payload.get("warm_start_checkpoint") == str(PINNED_WARM_START_CHECKPOINT)
        and payload.get("warm_start_adapter_sha256") == PINNED_WARM_START_ADAPTER_SHA256
        and payload.get("warm_start_mode") == warm.WARM_START_MODE
        and payload.get("warm_start_reuse_policy") == WARM_START_REUSE_POLICY,
        "v15_launch_receipt_architecture_or_warm_start_differs",
    )
    if data is not None:
        expected_data = {
            "train_file": data.get("train_file"),
            "train_file_sha256": data.get("train_file_sha256"),
            "source_manifest": data.get("source_manifest"),
            "source_manifest_sha256": data.get("source_manifest_sha256"),
            "source_lock": data.get("source_lock_path"),
            "source_lock_file_sha256": data.get("source_lock_file_sha256"),
            "schedule": data.get("schedule"),
            "schedule_sha256": data.get("schedule_sha256"),
        }
        require(
            all(payload.get(key) == value for key, value in expected_data.items()),
            "v15_launch_receipt_live_data_binding_differs",
        )
    if warm_start is not None:
        expected_warm = {
            "warm_start_checkpoint": warm_start.get("warm_start_checkpoint"),
            "warm_start_adapter_sha256": warm_start.get(
                "warm_start_adapter_sha256"
            ),
            "warm_start_mode": warm_start.get("warm_start_mode"),
            "warm_start_lock": warm_start.get("warm_start_lock"),
            "warm_start_lock_sha256": warm_start.get("warm_start_lock_sha256"),
            "warm_start_reuse_policy": warm_start.get("warm_start_reuse_policy"),
        }
        require(
            all(payload.get(key) == value for key, value in expected_warm.items()),
            "v15_launch_receipt_live_warm_start_binding_differs",
        )
    require(
        payload.get("training_continuation") == CONTINUATION_POLICY
        and payload.get("evaluation_access") == EVALUATION_ACCESS_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY,
        "v15_launch_receipt_access_policy_differs",
    )
    expected_code = critical_training_code_bindings_at_commit(
        payload.get("git_commit"),
        project_root=project_root,
    )
    require(payload.get("critical_files") == expected_code, "v15_launch_receipt_critical_files_differ")
    if checkpoint is not None:
        resolved_checkpoint = require_under_root(
            checkpoint,
            root=RUN_ROOT,
            description="v15_launch_receipt_checkpoint",
        )
        require(
            str(resolved_checkpoint) in expected_checkpoints.values(),
            "v15_launch_receipt_checkpoint_binding_differs",
        )
    if base_model_identity is not None:
        require(
            payload.get("base_model_identity") == dict(base_model_identity),
            "v15_launch_receipt_base_model_identity_differs",
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "receipt_sha256": payload["receipt_sha256"],
        "payload": payload,
    }


_V15_PAIR_IDENTITY_AUDIT_FIELDS = (
    "pair_mean_pair_identity_hinge",
    "pair_mean_pair_identity_logit_margin",
    "pair_mean_pair_identity_own_beats_paired_fraction",
    "pair_mean_pair_identity_margin_satisfied_fraction",
)
_V15_ROW_IDENTITY_AUDIT_FIELDS = (
    "pair_identity_hinge",
    "pair_identity_logit_margin",
    "pair_identity_own_beats_paired_fraction",
    "pair_identity_margin_satisfied_fraction",
)


def _finite_audit_number(
    payload: Mapping[str, Any],
    field: str,
    *,
    description: str,
) -> float:
    value = payload.get(field)
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{description}_missing_or_nonfinite field={field}",
    )
    return float(value)


def _validate_checkpoint_row_audit(
    audit: Mapping[str, Any],
    *,
    step: int,
    smoke: bool,
) -> dict[str, Any]:
    expected_count = 1 if smoke else presentation_cursor(step)
    expected_entries = _load_v15_pair_schedule()[:expected_count]
    expected_pairs = [
        list(entry["canonical_pair_ordinals"]) for entry in expected_entries
    ]
    expected_phases = (
        ["smoke_input"]
        if smoke
        else [f"cycle{cycle}_input" for cycle in range(1, step + 1)]
    )
    expected_pair_schedule = [
        {
            "source_row_ordinal": pair[0],
            "donor_row_ordinal": pair[1],
        }
        for pair in expected_pairs
    ]
    require(
        audit.get("schema") == ROW_OBJECTIVE_AUDIT_SCHEMA
        and audit.get("memory_objective_version") == OBJECTIVE_VERSION
        and audit.get("run_mode")
        == (ONE_PAIR_SMOKE_RUN_MODE if smoke else PRODUCTION_RUN_MODE)
        and audit.get("production_eligible") is (not smoke)
        and audit.get("checkpoint_optimizer_step") == step
        and audit.get("completed_pair_presentations") == expected_count
        and audit.get("phases") == expected_phases,
        "v15_checkpoint_row_audit_differs",
    )
    require(
        audit.get("pair_schedule") == expected_pair_schedule,
        "v15_checkpoint_audit_pair_schedule_prefix_or_order_differs",
    )
    pair_prefix_sha256 = canonical_sha256(expected_pairs)
    expected_prefix_sha256 = (
        canonical_sha256([list(ONE_PAIR_SMOKE_PAIR)])
        if smoke
        else PAIR_PREFIX_SHA256_BY_CHECKPOINT[step]
    )
    require(
        pair_prefix_sha256 == expected_prefix_sha256,
        "v15_checkpoint_audit_pair_prefix_sha256_differs",
    )

    pair_presentations = audit.get("pair_presentations")
    require(
        isinstance(pair_presentations, list)
        and len(pair_presentations) == expected_count,
        "v15_checkpoint_audit_pair_presentations_count_differs",
    )
    expected_row_observations: dict[tuple[str, int], dict[str, Any]] = {}
    pair_binding_material: list[dict[str, Any]] = []
    for index, (presentation, entry) in enumerate(
        zip(pair_presentations, expected_entries)
    ):
        require(
            isinstance(presentation, Mapping),
            f"v15_checkpoint_pair_presentation_missing index={index}",
        )
        pair = list(entry["canonical_pair_ordinals"])
        members = entry["members"]
        cycle = int(entry["cycle_index"])
        phase = "smoke_input" if smoke else f"cycle{cycle}_input"
        expected_binding = {
            "phase": phase,
            "cycle": cycle,
            "adapter_optimizer_step_before_update": cycle - 1,
            "presentation": index + 1,
            "source_row_ordinal": pair[0],
            "donor_row_ordinal": pair[1],
            "source_row_sha256": members[0]["row_sha256"],
            "donor_row_sha256": members[1]["row_sha256"],
        }
        require(
            all(
                presentation.get(field) == value
                for field, value in expected_binding.items()
            ),
            f"v15_checkpoint_pair_presentation_order_or_hash_differs index={index}",
        )
        for field in _V15_PAIR_IDENTITY_AUDIT_FIELDS:
            _finite_audit_number(
                presentation,
                field,
                description=f"v15_checkpoint_pair_presentation_{index}",
            )
        pair_binding_material.append(expected_binding)
        for role, row_position, paired_position in (
            ("source", 0, 1),
            ("donor", 1, 0),
        ):
            row_ordinal = pair[row_position]
            paired_ordinal = pair[paired_position]
            key = (phase, row_ordinal)
            require(
                key not in expected_row_observations,
                f"v15_checkpoint_expected_row_phase_duplicated phase={phase} row={row_ordinal}",
            )
            expected_row_observations[key] = {
                "phase": phase,
                "cycle": cycle,
                "adapter_optimizer_step_before_update": cycle - 1,
                "presentation": index + 1,
                "pair_role": role,
                "row_ordinal": row_ordinal,
                "paired_row_ordinal": paired_ordinal,
                "row_sha256": members[row_position]["row_sha256"],
                "paired_row_sha256": members[paired_position]["row_sha256"],
            }

    cycle_size = 1 if smoke else PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    expected_row_order = [
        ordinal
        for entry in expected_entries[:cycle_size]
        for ordinal in entry["canonical_pair_ordinals"]
    ]
    rows = audit.get("rows")
    require(
        isinstance(rows, list)
        and [
            row.get("row_ordinal") if isinstance(row, Mapping) else None
            for row in rows
        ]
        == expected_row_order,
        "v15_checkpoint_audit_grouped_row_order_differs",
    )
    row_binding_material: list[dict[str, Any]] = []
    for row_payload in rows:
        require(
            isinstance(row_payload, Mapping),
            "v15_checkpoint_audit_grouped_row_missing",
        )
        row_ordinal = int(row_payload["row_ordinal"])
        require(
            set(row_payload) == {"row_ordinal", *expected_phases},
            f"v15_checkpoint_audit_grouped_row_phases_differ row={row_ordinal}",
        )
        for phase in expected_phases:
            observation = row_payload.get(phase)
            require(
                isinstance(observation, Mapping),
                f"v15_checkpoint_audit_row_phase_missing phase={phase} row={row_ordinal}",
            )
            expected = expected_row_observations[(phase, row_ordinal)]
            require(
                all(
                    observation.get(field) == value
                    for field, value in expected.items()
                ),
                f"v15_checkpoint_audit_row_phase_order_or_hash_differs phase={phase} row={row_ordinal}",
            )
            for field in _V15_ROW_IDENTITY_AUDIT_FIELDS:
                _finite_audit_number(
                    observation,
                    field,
                    description=(
                        "v15_checkpoint_audit_row_phase_"
                        f"{phase}_{row_ordinal}"
                    ),
                )
            row_binding_material.append(expected)
    require(
        len(row_binding_material) == expected_count * 2,
        "v15_checkpoint_audit_row_phase_coverage_differs",
    )
    return {
        "pair_prefix_sha256": pair_prefix_sha256,
        "pair_binding_sha256": canonical_sha256(pair_binding_material),
        "row_phase_binding_sha256": canonical_sha256(row_binding_material),
        "pair_presentations": expected_count,
        "row_phase_observations": expected_count * 2,
    }


def _expected_scene_state_source_manifest(
    data: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": data.get("source_manifest"),
        "file_sha256": data.get("source_manifest_sha256"),
        "schema": data_prep.SOURCE_SCHEMA,
        "train_file": data.get("train_file"),
        "train_file_sha256": data.get("train_file_sha256"),
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": {
            "episode_recent_messages": 0,
            "write_phase": "system + user",
            "read_supervision": "system + assistant",
        },
    }


def _validate_pairing_manifest(
    pairing: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> str:
    unsigned = dict(pairing)
    recorded = unsigned.pop("manifest_sha256", None)
    require(
        isinstance(recorded, str)
        and len(recorded) == 64
        and recorded == canonical_sha256(unsigned),
        "v15_checkpoint_pairing_manifest_self_hash_differs",
    )
    protocol_pairing = protocol.get("scene_state_identity_pairing")
    require(
        pairing.get("objective_version") == PAIRING_OBJECTIVE_VERSION
        and isinstance(protocol_pairing, Mapping)
        and protocol_pairing.get("manifest_sha256") == recorded,
        "v15_checkpoint_pairing_protocol_binding_differs",
    )
    return recorded


def _validate_warm_start_lineage(
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
        and len(receipt_sha256) == 64
        and receipt_sha256 == canonical_sha256(unsigned),
        "v15_warm_lineage_self_hash_differs",
    )
    source = warm_contract.get("lineage_source")
    source_lock = lineage.get("source_lock")
    topology = lineage.get("topology")
    require(
        isinstance(source, Mapping)
        and isinstance(source_lock, Mapping)
        and isinstance(topology, Mapping),
        "v15_warm_lineage_source_contract_missing",
    )
    require(
        lineage.get("schema") == warm.RECEIPT_SCHEMA
        and lineage.get("schema_version") == 1
        and lineage.get("mode") == warm.WARM_START_MODE
        and lineage.get("source_checkpoint")
        == warm_contract.get("warm_start_checkpoint")
        and lineage.get("source_global_step")
        == warm_contract.get("source_global_step")
        and lineage.get("source_epoch") == source.get("source_epoch")
        and lineage.get("source_protocol_objective_version")
        == source.get("source_protocol_objective_version")
        and lineage.get("source_pairing_objective_version")
        == source.get("source_pairing_objective_version")
        and lineage.get("source_row_objective_audit_schema")
        == source.get("source_row_objective_audit_schema")
        and lineage.get("source_v13_warm_start_receipt_sha256")
        == source.get("source_v13_warm_start_receipt_sha256"),
        "v15_warm_lineage_identity_differs",
    )
    require(
        source_lock.get("path") == warm_contract.get("warm_start_lock")
        and source_lock.get("lock_sha256")
        == warm_contract.get("warm_start_lock_sha256")
        and lineage.get("source_artifacts") == source.get("source_artifacts")
        and lineage.get("source_state_imports") == warm.SOURCE_IMPORT_POLICY
        and lineage.get("loaded_source_artifacts") == ["delta_mem_adapter.pt"]
        and lineage.get("validated_not_imported_source_artifacts")
        == ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
        and lineage.get("post_load_bit_equal") is True,
        "v15_warm_lineage_source_binding_differs",
    )
    source_topology = source.get("source_adapter_topology")
    require(
        isinstance(source_topology, Mapping)
        and topology.get("adapter_tensor_count")
        == source_topology.get("tensor_count")
        and topology.get("adapter_tensor_elements")
        == source_topology.get("tensor_elements")
        and topology.get("adapter_topology_sha256") == source_topology.get("sha256")
        and topology.get("ordered_dtypes_equal") is True
        and topology.get("ordered_parameter_names_equal") is True
        and topology.get("ordered_shapes_equal") is True
        and lineage.get("post_load_topology_sha256") == source_topology.get("sha256"),
        "v15_warm_lineage_topology_differs",
    )
    try:
        expected_fresh_start = warm.validate_v14_fresh_start_contract(
            warm.V14FreshStartContract(
                resume_from_checkpoint=None,
                initial_global_step=0,
                optimizer_created=False,
                scheduler_created=False,
                trainer_state_imported=False,
                rng_state_imported=False,
                optim=OPTIMIZER_IMPLEMENTATION,
            )
        )
    except Exception as exc:
        raise LaunchContractError(
            f"v15_warm_lineage_fresh_start_contract_failed: {exc}"
        ) from exc
    require(
        lineage.get("target_fresh_start") == expected_fresh_start,
        "v15_warm_lineage_fresh_state_differs",
    )
    optimizer_class = lineage.get("fresh_optimizer_class")
    require(
        lineage.get("trainer_resume_from_checkpoint") is None
        and lineage.get("target_initial_global_step") == 0
        and lineage.get("pre_train_global_step") == 0
        and lineage.get("fresh_optimizer_created") is True
        and isinstance(optimizer_class, str)
        and optimizer_class.endswith(".AdamW")
        and lineage.get("fresh_optimizer_state_entries_before_train") == 0
        and lineage.get("fresh_scheduler_created_before_train") is False
        and lineage.get("fresh_adamw_creation_required_after_adapter_load") is True,
        "v15_warm_lineage_optimizer_evidence_differs",
    )
    require(
        lineage.get("target_training_protocol_sha256") == protocol_sha256
        and lineage.get("target_delta_config_sha256") == config_sha256
        and lineage.get("target_scene_state_pairing_manifest_sha256")
        == pairing_sha256,
        "v15_warm_lineage_target_binding_differs",
    )
    return receipt_sha256


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    data: Mapping[str, Any] | None = None,
    warm: Mapping[str, Any] | None = None,
    smoke: bool | None = None,
) -> dict[str, Any]:
    if data is None:
        data = validate_data_contract()
    if warm is None:
        warm = validate_warm_start_contract()
    require(
        isinstance(data, Mapping) and isinstance(warm, Mapping),
        "v15_checkpoint_live_contracts_missing",
    )
    resolved = require_under_root(
        checkpoint,
        root=RUN_ROOT,
        description="v15_checkpoint",
    )
    match = re.fullmatch(r"checkpoint-([1-4])", resolved.name)
    require(match is not None, "v15_checkpoint_name_differs")
    step = int(match.group(1))
    if smoke is None:
        smoke = "_smoke_" in resolved.parents[1].name
    require(step == 1 or not smoke, "v15_smoke_checkpoint_step_differs")
    for name in REQUIRED_CHECKPOINT_ARTIFACTS:
        _regular_file(resolved / name, description=f"v15_checkpoint_{name}")
    trainer_state = _load_object(
        resolved / "trainer_state.json",
        description="v15_checkpoint_trainer_state",
    )
    protocol = _load_object(
        resolved / "training_protocol.json",
        description="v15_checkpoint_training_protocol",
    )
    config = _load_object(
        resolved / "delta_mem_config.json",
        description="v15_checkpoint_delta_config",
    )
    audit = _load_object(
        resolved / ROW_OBJECTIVE_AUDIT_FILENAME,
        description="v15_checkpoint_row_audit",
    )
    pairing = _load_object(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="v15_checkpoint_pairing_manifest",
    )
    lineage = _load_object(
        resolved / WARM_START_LINEAGE_FILENAME,
        description="v15_checkpoint_warm_start_lineage",
    )
    require(trainer_state.get("global_step") == step, "v15_checkpoint_global_step_differs")
    require(
        protocol.get("memory_objective_version") == OBJECTIVE_VERSION
        and protocol.get("schema_version") == OBJECTIVE_SCHEMA_VERSION
        and protocol.get("max_steps") == (1 if smoke else TOTAL_OPTIMIZER_STEPS)
        and protocol.get("gradient_accumulation_steps")
        == (1 if smoke else GRADIENT_ACCUMULATION_STEPS)
        and protocol.get("save_steps") == SAVE_STEPS,
        "v15_checkpoint_protocol_differs",
    )
    require(
        config.get("target_layers") == list(TARGET_LAYERS)
        and config.get("delta_heads") == list(DELTA_HEADS)
        and config.get("rank") == 4,
        "v15_checkpoint_adapter_topology_differs",
    )
    require(
        protocol.get("scene_state_source_manifest")
        == _expected_scene_state_source_manifest(data),
        "v15_checkpoint_data_binding_differs",
    )
    protocol_sha256 = canonical_sha256(protocol)
    config_sha256 = canonical_sha256(config)
    pairing_sha256 = _validate_pairing_manifest(pairing, protocol=protocol)
    lineage_sha256 = _validate_warm_start_lineage(
        lineage,
        protocol_sha256=protocol_sha256,
        config_sha256=config_sha256,
        pairing_sha256=pairing_sha256,
        warm_contract=warm,
    )
    audit_binding = _validate_checkpoint_row_audit(
        audit,
        step=step,
        smoke=smoke,
    )
    adapter_binding = artifact_binding(
        resolved / "delta_mem_adapter.pt",
        description=f"v15_checkpoint_{step}_adapter",
    )
    smoke_update: dict[str, Any] | None = None
    if smoke:
        history = trainer_state.get("log_history")
        require(isinstance(history, list), "v15_smoke_log_history_missing")
        candidates = [
            entry
            for entry in history
            if isinstance(entry, Mapping) and int(entry.get("step", -1)) == 1
        ]
        require(bool(candidates), "v15_smoke_optimizer_log_missing")
        entry = candidates[-1]

        def finite_metric(name: str) -> float:
            value = entry.get(name)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"v15_smoke_metric_missing_or_nonfinite name={name}",
            )
            return float(value)

        loss = finite_metric("loss")
        grad_norm = finite_metric("grad_norm")
        objective_total = finite_metric(
            "delta/scene_generation_v15_objective_total_loss"
        )
        identity_hinge = finite_metric(
            "delta/scene_generation_v15_pair_mean_pair_identity_hinge"
        )
        identity_margin = finite_metric(
            "delta/scene_generation_v15_pair_mean_pair_identity_logit_margin"
        )
        identity_own_beats = finite_metric(
            "delta/scene_generation_v15_pair_mean_pair_identity_own_beats_paired_fraction"
        )
        identity_margin_satisfied = finite_metric(
            "delta/scene_generation_v15_pair_mean_pair_identity_margin_satisfied_fraction"
        )
        cycle_presentations = finite_metric(
            "delta/scene_generation_v15_cycle_pair_presentations"
        )
        cycle_index = finite_metric("delta/scene_generation_v15_cycle_index")
        low = finite_metric(
            "delta/scene_generation_v15_cycle_pair_0_low_ordinal"
        )
        high = finite_metric(
            "delta/scene_generation_v15_cycle_pair_0_high_ordinal"
        )
        require(loss >= 0.0, "v15_smoke_loss_must_be_nonnegative")
        require(grad_norm > 0.0, "v15_smoke_grad_norm_must_be_positive")
        require(
            objective_total >= 0.0 and identity_hinge >= 0.0,
            "v15_smoke_objective_telemetry_is_invalid",
        )
        require(
            0.0 <= identity_own_beats <= 1.0
            and 0.0 <= identity_margin_satisfied <= 1.0,
            "v15_smoke_identity_fraction_telemetry_is_invalid",
        )
        require(
            cycle_presentations == 1.0
            and cycle_index == 1.0
            and (int(low), int(high)) == ONE_PAIR_SMOKE_PAIR,
            "v15_smoke_cycle_telemetry_differs",
        )
        pair_presentations = audit.get("pair_presentations")
        rows = audit.get("rows")
        require(
            isinstance(pair_presentations, list)
            and len(pair_presentations) == 1
            and isinstance(rows, list)
            and len(rows) == 2,
            "v15_smoke_audit_must_cover_one_pair",
        )
        pair_audit = pair_presentations[0]
        require(isinstance(pair_audit, Mapping), "v15_smoke_pair_audit_missing")
        require(
            math.isclose(
                float(pair_audit["pair_mean_pair_identity_hinge"]),
                identity_hinge,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(pair_audit["pair_mean_pair_identity_logit_margin"]),
                identity_margin,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(
                    pair_audit[
                        "pair_mean_pair_identity_own_beats_paired_fraction"
                    ]
                ),
                identity_own_beats,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(
                    pair_audit[
                        "pair_mean_pair_identity_margin_satisfied_fraction"
                    ]
                ),
                identity_margin_satisfied,
                rel_tol=1e-5,
                abs_tol=1e-6,
            ),
            "v15_smoke_identity_audit_differs_from_optimizer_log",
        )
        require(
            adapter_binding["sha256"] != PINNED_WARM_START_ADAPTER_SHA256,
            "v15_smoke_adapter_did_not_change_from_warm_start",
        )
        smoke_update = {
            "loss": loss,
            "grad_norm": grad_norm,
            "objective_total_loss": objective_total,
            "pair_identity_hinge": identity_hinge,
            "pair_identity_logit_margin": identity_margin,
            "pair_identity_own_beats_paired_fraction": identity_own_beats,
            "pair_identity_margin_satisfied_fraction": (
                identity_margin_satisfied
            ),
            "adapter_sha256": adapter_binding["sha256"],
            "warm_start_adapter_sha256": PINNED_WARM_START_ADAPTER_SHA256,
            "adapter_changed": True,
        }
    return {
        "path": str(resolved),
        "checkpoint_step": step,
        "consumed_pair_presentations": 1 if smoke else presentation_cursor(step),
        "training_protocol_sha256": protocol_sha256,
        "delta_config_sha256": config_sha256,
        "pairing_manifest_sha256": pairing_sha256,
        "warm_start_lineage_receipt_sha256": lineage_sha256,
        "scene_state_source_manifest": _expected_scene_state_source_manifest(data),
        "artifacts": {
            name: artifact_binding(
                resolved / name,
                description=f"v15_checkpoint_{step}_{name}",
            )
            for name in REQUIRED_CHECKPOINT_ARTIFACTS
        },
        "rng_state_artifacts": {
            item.name: artifact_binding(
                item,
                description=f"v15_checkpoint_{step}_{item.name}",
            )
            for item in sorted(resolved.glob("rng_state*.pth"))
        },
        "row_objective_audit_binding": audit_binding,
        "smoke_real_optimizer_update": smoke_update,
    }


def build_completion_receipt(
    *,
    launch_receipt: Path,
    training_summary: Path,
    log_file: Path,
    checkpoints: Sequence[Path],
    smoke: bool,
) -> dict[str, Any]:
    data = validate_data_contract()
    warm_contract = validate_warm_start_contract()
    launch = validate_launch_receipt(
        launch_receipt,
        smoke=smoke,
        data=data,
        warm_start=warm_contract,
    )
    launch_payload = launch["payload"]
    expected_steps = (1,) if smoke else CHECKPOINT_STEPS
    expected_checkpoint_records = launch_payload.get("checkpoints")
    require(
        isinstance(expected_checkpoint_records, Mapping),
        "v15_completion_launch_checkpoints_missing",
    )
    expected_checkpoint_paths = tuple(
        Path(str(expected_checkpoint_records[f"checkpoint-{step}"])).resolve(
            strict=False
        )
        for step in expected_steps
    )
    resolved_checkpoints = tuple(
        require_under_root(
            path,
            root=RUN_ROOT,
            description=f"v15_completion_checkpoint_{index}",
        )
        for index, path in enumerate(checkpoints, start=1)
    )
    require(
        resolved_checkpoints == expected_checkpoint_paths,
        "v15_completion_checkpoint_paths_or_order_differ_from_launch",
    )
    trainer_output = Path(str(launch_payload["trainer_output"])).resolve(strict=False)
    require(
        Path(training_summary).expanduser().resolve(strict=False)
        == trainer_output / "training_summary.json"
        and Path(log_file).expanduser().resolve(strict=False)
        == Path(str(launch_payload["log_file"])).resolve(strict=False),
        "v15_completion_summary_or_log_path_differs_from_launch",
    )
    checkpoint_contracts = [
        validate_checkpoint_contract(
            path,
            data=data,
            warm=warm_contract,
            smoke=smoke,
        )
        for path in resolved_checkpoints
    ]
    payload: dict[str, Any] = {
        "schema": ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA
        if smoke
        else COMPLETION_RECEIPT_SCHEMA,
        "status": "completed",
        "optimizer_step": 1 if smoke else TOTAL_OPTIMIZER_STEPS,
        "consumed_pair_presentations": 1 if smoke else TOTAL_PAIR_PRESENTATIONS,
        "launch_receipt": artifact_binding(
            launch_receipt,
            description="v15_completion_launch_receipt",
        ),
        "launch_receipt_sha256": launch["receipt_sha256"],
        "log": artifact_binding(log_file, description="v15_completion_log"),
        "training_summary": artifact_binding(
            training_summary,
            description="v15_completion_training_summary",
        ),
        "checkpoints": {
            Path(contract["path"]).name: contract
            for contract in checkpoint_contracts
        },
        "training_continuation": CONTINUATION_POLICY,
        "evaluation_access": EVALUATION_ACCESS_POLICY,
        "hard32_access": HARD32_ACCESS_POLICY,
        "production_eligible": not smoke,
    }
    payload["receipt_sha256"] = _receipt_hash(payload)
    return payload


def write_completion_receipt(
    path: Path,
    *,
    launch_receipt: Path,
    training_summary: Path,
    log_file: Path,
    checkpoints: Sequence[Path],
    smoke: bool,
) -> Path:
    return _write_json_exclusive(
        path,
        build_completion_receipt(
            launch_receipt=launch_receipt,
            training_summary=training_summary,
            log_file=log_file,
            checkpoints=checkpoints,
            smoke=smoke,
        ),
    )


def _validate_artifact_record(
    record: object,
    *,
    description: str,
) -> Path:
    require(isinstance(record, Mapping), f"{description}_binding_missing")
    raw_path = record.get("path")
    require(isinstance(raw_path, str), f"{description}_path_missing")
    path = Path(raw_path).expanduser().resolve(strict=False)
    require(
        artifact_binding(path, description=description) == dict(record),
        f"{description}_binding_differs",
    )
    return path


def validate_completion_receipt(
    path: Path,
    *,
    checkpoint: Path | None = None,
    checkpoint_contract: Mapping[str, Any] | None = None,
    launch: Mapping[str, Any] | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    payload = _load_object(path, description="v15_completion_receipt")
    require(payload.get("receipt_sha256") == _receipt_hash(payload), "v15_completion_receipt_self_hash_differs")
    schema = payload.get("schema")
    require(
        schema
        in (COMPLETION_RECEIPT_SCHEMA, ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA),
        "v15_completion_receipt_schema_differs",
    )
    smoke = schema == ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA
    require(
        payload.get("schema")
        == (ONE_PAIR_SMOKE_COMPLETION_RECEIPT_SCHEMA if smoke else COMPLETION_RECEIPT_SCHEMA)
        and payload.get("status") == "completed"
        and payload.get("optimizer_step") == (1 if smoke else TOTAL_OPTIMIZER_STEPS)
        and payload.get("consumed_pair_presentations")
        == (1 if smoke else TOTAL_PAIR_PRESENTATIONS)
        and payload.get("production_eligible") is (not smoke)
        and payload.get("training_continuation") == CONTINUATION_POLICY
        and payload.get("evaluation_access") == EVALUATION_ACCESS_POLICY
        and payload.get("hard32_access") == HARD32_ACCESS_POLICY,
        "v15_completion_receipt_contract_differs",
    )
    data = validate_data_contract()
    warm_contract = validate_warm_start_contract()
    launch_receipt_path = _validate_artifact_record(
        payload.get("launch_receipt"),
        description="v15_completion_launch_receipt",
    )
    validated_launch = validate_launch_receipt(
        launch_receipt_path,
        smoke=smoke,
        data=data,
        warm_start=warm_contract,
    )
    launch_payload = validated_launch["payload"]
    require(
        payload.get("launch_receipt_sha256")
        == validated_launch.get("receipt_sha256"),
        "v15_completion_receipt_launch_binding_differs",
    )
    expected_completion_path = launch_receipt_path.with_name(
        launch_receipt_path.name.removesuffix(".launch.json") + ".completion.json"
    )
    require(
        path.expanduser().resolve(strict=False) == expected_completion_path,
        "v15_completion_receipt_path_differs_from_launch",
    )
    log_path = _validate_artifact_record(
        payload.get("log"),
        description="v15_completion_log",
    )
    summary_path = _validate_artifact_record(
        payload.get("training_summary"),
        description="v15_completion_training_summary",
    )
    trainer_output = Path(str(launch_payload["trainer_output"])).resolve(strict=False)
    require(
        log_path == Path(str(launch_payload["log_file"])).resolve(strict=False)
        and summary_path == trainer_output / "training_summary.json",
        "v15_completion_receipt_summary_or_log_path_differs_from_launch",
    )
    expected_steps = (1,) if smoke else CHECKPOINT_STEPS
    launch_checkpoints = launch_payload.get("checkpoints")
    receipt_checkpoints = payload.get("checkpoints")
    expected_names = [f"checkpoint-{step}" for step in expected_steps]
    require(
        isinstance(launch_checkpoints, Mapping)
        and isinstance(receipt_checkpoints, Mapping)
        and list(launch_checkpoints) == expected_names
        and list(receipt_checkpoints) == expected_names,
        "v15_completion_receipt_checkpoint_order_differs",
    )
    validated_checkpoint_contracts: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        checkpoint_path = Path(str(launch_checkpoints[name])).resolve(strict=False)
        contract = validate_checkpoint_contract(
            checkpoint_path,
            data=data,
            warm=warm_contract,
            smoke=smoke,
        )
        require(
            receipt_checkpoints.get(name) == contract,
            f"v15_completion_receipt_checkpoint_contract_differs name={name}",
        )
        validated_checkpoint_contracts[name] = contract
    if checkpoint is not None:
        resolved_checkpoint = require_under_root(
            checkpoint,
            root=RUN_ROOT,
            description="v15_completion_receipt_checkpoint",
        )
        record = receipt_checkpoints.get(resolved_checkpoint.name)
        require(
            isinstance(record, Mapping)
            and record.get("path") == str(resolved_checkpoint),
            "v15_completion_receipt_checkpoint_binding_differs",
        )
    if checkpoint_contract is not None and checkpoint is not None:
        require(
            receipt_checkpoints.get(Path(checkpoint).name)
            == dict(checkpoint_contract),
            "v15_completion_receipt_checkpoint_contract_differs",
        )
    if launch is not None:
        launch_sha256 = launch.get("receipt_sha256")
        if launch_sha256 is None and isinstance(launch.get("payload"), Mapping):
            launch_sha256 = launch["payload"].get("receipt_sha256")
        require(
            payload.get("launch_receipt_sha256") == launch_sha256,
            "v15_completion_receipt_launch_binding_differs",
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "receipt_sha256": payload["receipt_sha256"],
        "payload": payload,
        "launch": validated_launch,
        "checkpoint_contracts": validated_checkpoint_contracts,
    }


def _tsv(result: Mapping[str, Any]) -> str:
    values = (
        result["train_file"],
        result["train_file_sha256"],
        result["source_manifest"],
        result["source_manifest_sha256"],
        result["schedule"],
        result["schedule_sha256"],
        result["warm_start_checkpoint"],
        result["warm_start_adapter_sha256"],
        result["warm_start_mode"],
        result["output_dir"],
        result["log_file"],
        result["launch_receipt"],
        result["completion_receipt"],
        result["run_mode"],
        result["gradient_accumulation_steps"],
        result["max_steps"],
        result["save_total_limit"],
        result["total_pair_presentations"],
        result["first_pair_low_ordinal"],
        result["first_pair_high_ordinal"],
        result["source_lock_file_sha256"],
    )
    rendered = tuple(str(item) for item in values)
    require(
        all("\t" not in item and "\n" not in item for item in rendered),
        "v15_launch_contract_tsv_control_character",
    )
    return "\t".join(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--benchmark-path", type=Path, action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    parser.add_argument("--warm-start-lock", type=Path, default=WARM_START_LOCK)
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=PINNED_WARM_START_CHECKPOINT,
    )
    parser.add_argument("--base-model", type=Path, default=PINNED_BASE_MODEL)
    parser.add_argument("--ssd-root", type=Path, default=SSD_ROOT)
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_launch_contract(
            target_step=args.target_step,
            run_name=args.run_name,
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            resume_checkpoint=args.resume_checkpoint,
            benchmark_paths=args.benchmark_path,
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
