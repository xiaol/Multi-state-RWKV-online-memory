#!/usr/bin/env python3
"""Fail-closed contract for fresh hard-scene RWKV-MS training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_scene_hard_failure_curriculum as data_prep,
)


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_BASE_MODEL = SSD_ROOT / "models/gemma/gemma-4-E4B-it"
DATA_ROOT = (
    SSD_ROOT
    / "delta_mem_data/scene_failure_state/"
    "scene_hard_failure_curriculum_base64_pairs16_v1"
)
TRAIN_FILE = DATA_ROOT / "train.jsonl"
TRAIN_ROWS = DATA_ROOT / "train_rows.jsonl"
PAIR_MANIFEST = DATA_ROOT / "pair_manifest.json"
PAIR_SCHEDULE = DATA_ROOT / "pair_schedule.jsonl"
PAIR_SCHEDULE_MANIFEST = DATA_ROOT / "pair_schedule_manifest.json"
SOURCE_MANIFEST = DATA_ROOT / "source_manifest.json"
BUNDLE_MANIFEST = DATA_ROOT / "manifest.json"
SOURCE_LOCK = Path(__file__).with_name("scene_hard_failure_source_lock.json")
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_hard_failure_source_lock.v1"

FILE_SHA256 = {
    "manifest.json": "7ee8df6a55cbfec43d5860591595dbf19f8cb3340af1909fcf54d3faba909873",
    "pair_manifest.json": "61a4597ef1e014d20b0aece500117d4068e39e36fa320868f7758c506f30f78f",
    "pair_schedule.jsonl": "bd12f021fc238f644972758047e7850cd22301be93b484dbf9f38f2203adb249",
    "pair_schedule_manifest.json": "2a36938ba37252001e1cd5d845a1feb9dfde8be42eb33b9c9c58f54b599bd9ec",
    "source_manifest.json": "a3b1e0a255f2e7440971e81337d9648a1dbf96da8a1211aa3e103f2d01f052d8",
    "train.jsonl": "254e4fe5c2c8e8e9f107ef72a5e08abb48c1c86a446f75e3b265034de1ada8df",
    "train_rows.jsonl": "d12dff5cc2d8d038b8a04aef002cb0dc3b8dc047f726d63eb063f9aba66200df",
}

BASE_MODEL_ARTIFACTS = {
    "chat_template.jinja": {
        "bytes": 17_336,
        "sha256": "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    },
    "config.json": {
        "bytes": 5_145,
        "sha256": "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
    },
    "generation_config.json": {
        "bytes": 208,
        "sha256": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    },
    "tokenizer.json": {
        "bytes": 32_169_626,
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    "tokenizer_config.json": {
        "bytes": 2_095,
        "sha256": "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
    },
}
BASE_MODEL_WEIGHT_FILE = "model.safetensors"
BASE_MODEL_WEIGHT_BYTES = 15_992_595_884

OBJECTIVE_VERSION = (
    "scene_state_generation_ce_symmetric_cached_prefix_identity_"
    "hard_failure_v1"
)
OBJECTIVE_SCHEMA_VERSION = 19
PRODUCTION_RUN_MODE = "production_64_single_pair_updates_hard_failure_v1"
ONE_PAIR_SMOKE_RUN_MODE = "one_pair_real_optimizer_update_hard_failure_v1"
FIXED_SAMPLER_MODE = "explicit_ordered_scene_hard_failure_pair_cycle_v1"
ONE_PAIR_SMOKE_SAMPLER_MODE = (
    "explicit_ordered_scene_hard_failure_first_pair_smoke_v1"
)
ONE_PAIR_SMOKE_FLAG = "--scene-state-hard-failure-one-pair-smoke"
ROW_OBJECTIVE_AUDIT_FILENAME = "scene_hard_failure_row_objective.json"
ROW_OBJECTIVE_AUDIT_SCHEMA = (
    "rwkv_ms_scene_hard_failure_row_objective.v1"
)

PAIR_CYCLES = 4
PAIR_COUNT = 16
PAIRS_PER_CYCLE = PAIR_COUNT
TOTAL_PAIR_PRESENTATIONS = PAIR_CYCLES * PAIRS_PER_CYCLE
PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP = 1
GRADIENT_ACCUMULATION_STEPS = 1
TOTAL_OPTIMIZER_STEPS = TOTAL_PAIR_PRESENTATIONS
CHECKPOINT_STEPS = tuple(range(1, TOTAL_OPTIMIZER_STEPS + 1))
GENERATION_ENDPOINT_STEPS = (16, 32, 48, 64)
SAVE_STEPS = 1
SAVE_TOTAL_LIMIT = 64

TARGET_LAYERS = tuple(range(42))
RANK = 4
ALPHA = 8
DELTA_HEADS = ("q", "o")
RWKV_MS_NUM_STATES = 4
RWKV_MS_SEMANTICS_VERSION = 2
RWKV_MS_CHUNK_SIZE = 128
RWKV_MS_BOUNDARY_MODE = "fixed_chunk"
STATE_RESET_PER_ROW = True
READ_SIDE_WRITES_ENABLED = False
MEMORY_FUSION_MODE = "add"
MEMORY_FUSION_PLACEMENT = "attention_output"
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0
SEED = 42
DATA_SEED = 42

PAIR_SCHEDULE_SHA256 = FILE_SHA256["pair_schedule.jsonl"]
PAIR_SCHEDULE_ENTRIES_SHA256 = (
    "c9f5b907c5cfca87f55718188359dfcbacca156e19662e4fe180b7558d78dc50"
)
ORDERED_PAIRS_SHA256 = (
    "5ade0812571c32eadf3656f6b3cc54bf194df85871b4c47a7103f4964335fb66"
)
CANONICAL_PAIRS_SHA256 = (
    "2efee1bf7282605d52a0682a7a43ab3195f0d30cb3de20e7f1d868fc5ce1a602"
)

CRITICAL_TRAINING_FILES = (
    "deltamem/scene_boundary.py",
    "deltamem/train/cached_prefix_replay.py",
    "deltamem/train/delta_sft_experimental.py",
    "deltamem/train/scene_state_generation_alignment.py",
    "experiments/rethinking_rwkv_ms_gemma/prepare_scene_hard_failure_curriculum.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_hard_failure_run_audit.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_hard_failure_train_contract.py",
    "experiments/rethinking_rwkv_ms_gemma/scene_hard_failure_source_lock.json",
    "experiments/rethinking_rwkv_ms_gemma/train_scene_hard_failure.sh",
)

_FRESH_START_FLAGS = frozenset(
    {
        "--warm-start-from-checkpoint",
        "--warm-start-mode",
        "--resume-from-checkpoint",
        "--resume-checkpoint",
        "--resume-mode",
    }
)
_FRESH_LINEAGE_FIELDS = (
    "source_checkpoint",
    "optimizer_state_imported",
    "scheduler_state_imported",
    "trainer_state_imported",
    "rng_state_imported",
)
_PROTECTED_COMPONENTS = frozenset(
    {
        "benchmark",
        "benchmarks",
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


class LaunchContractError(ValueError):
    """Raised when the hard-scene launch differs from its locked contract."""


def _require(condition: bool, message: str) -> None:
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


def _regular_file(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    _require(
        resolved.is_file() and not resolved.is_symlink() and resolved.stat().st_size > 0,
        f"{description}_missing_empty_or_symlink path={resolved}",
    )
    return resolved


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = _regular_file(path, description=description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LaunchContractError(f"{description}_invalid_json") from exc
    _require(isinstance(payload, dict), f"{description}_must_be_object")
    return payload


def _validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    _require(
        isinstance(recorded, str) and recorded == canonical_sha256(unsigned),
        f"{description}_{field}_differs",
    )
    return recorded


def _validate_locked_file_record(
    record: Any,
    *,
    expected_path: Path,
    expected_sha256: str | None,
    expected_bytes: int,
    description: str,
) -> None:
    _require(isinstance(record, Mapping), f"{description}_record_invalid")
    path = _regular_file(expected_path, description=description)
    _require(
        record.get("path") == str(path)
        and record.get("bytes") == expected_bytes
        and path.stat().st_size == expected_bytes,
        f"{description}_identity_differs",
    )
    if expected_sha256 is not None:
        _require(
            record.get("sha256") == expected_sha256
            and sha256_file(path) == expected_sha256,
            f"{description}_sha256_differs",
        )


def validate_source_lock(path: Path = SOURCE_LOCK) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    _require(resolved == SOURCE_LOCK.resolve(), "source_lock_path_differs")
    lock = _load_json(resolved, description="source_lock")
    _require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "source_lock_schema_differs")
    _validate_self_hash(lock, field="lock_sha256", description="source_lock")

    base_model = lock.get("base_model")
    _require(isinstance(base_model, Mapping), "source_lock_base_model_invalid")
    _require(
        base_model.get("path") == str(PINNED_BASE_MODEL.resolve()),
        "source_lock_base_model_path_differs",
    )
    prompt_artifacts = base_model.get("prompt_artifacts")
    _require(
        isinstance(prompt_artifacts, Mapping)
        and set(prompt_artifacts) == set(BASE_MODEL_ARTIFACTS),
        "source_lock_prompt_artifacts_differ",
    )
    for filename, identity in BASE_MODEL_ARTIFACTS.items():
        _validate_locked_file_record(
            prompt_artifacts.get(filename),
            expected_path=PINNED_BASE_MODEL / filename,
            expected_sha256=str(identity["sha256"]),
            expected_bytes=int(identity["bytes"]),
            description=f"base_model_{filename}",
        )
    weight = base_model.get("weight")
    _validate_locked_file_record(
        weight,
        expected_path=PINNED_BASE_MODEL / BASE_MODEL_WEIGHT_FILE,
        expected_sha256=None,
        expected_bytes=BASE_MODEL_WEIGHT_BYTES,
        description="base_model_weight",
    )
    _require(
        isinstance(weight, Mapping)
        and weight.get("relative_path") == BASE_MODEL_WEIGHT_FILE
        and weight.get("layout") == "unsharded_safetensors",
        "source_lock_base_model_weight_layout_differs",
    )

    training_artifacts = lock.get("training_artifacts")
    _require(
        isinstance(training_artifacts, Mapping)
        and set(training_artifacts) == set(FILE_SHA256),
        "source_lock_training_artifacts_differ",
    )
    for filename, expected_sha256 in FILE_SHA256.items():
        _validate_locked_file_record(
            training_artifacts.get(filename),
            expected_path=DATA_ROOT / filename,
            expected_sha256=expected_sha256,
            expected_bytes=(DATA_ROOT / filename).stat().st_size,
            description=f"training_artifact_{filename}",
        )

    training = lock.get("training_contract")
    expected_training = {
        "task": "scene-v4-current",
        "source_split": "train",
        "rows": 32,
        "objective_version": OBJECTIVE_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "fresh_adapter": True,
        "warm_start": False,
        "resume": False,
        "target_layers": list(TARGET_LAYERS),
        "rank": RANK,
        "alpha": ALPHA,
        "delta_heads": list(DELTA_HEADS),
        "rwkv_ms_num_states": RWKV_MS_NUM_STATES,
        "rwkv_ms_semantics_version": RWKV_MS_SEMANTICS_VERSION,
        "rwkv_ms_chunk_size": RWKV_MS_CHUNK_SIZE,
        "rwkv_ms_boundary_mode": RWKV_MS_BOUNDARY_MODE,
        "state_reset_per_row": STATE_RESET_PER_ROW,
        "read_side_writes_enabled": READ_SIDE_WRITES_ENABLED,
        "memory_fusion_mode": MEMORY_FUSION_MODE,
        "memory_fusion_placement": MEMORY_FUSION_PLACEMENT,
        "pair_cycles": PAIR_CYCLES,
        "pairs_per_cycle": PAIRS_PER_CYCLE,
        "pair_presentations": TOTAL_PAIR_PRESENTATIONS,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "generation_endpoint_steps": list(GENERATION_ENDPOINT_STEPS),
    }
    _require(training == expected_training, "source_lock_training_contract_differs")
    protected = lock.get("protected_evaluation")
    _require(isinstance(protected, Mapping), "source_lock_protected_evaluation_missing")
    for name in ("official_validation", "hard32", "official_test"):
        record = protected.get(name)
        _require(
            isinstance(record, Mapping)
            and record.get("included") is False
            and record.get("path") is None,
            f"source_lock_protected_evaluation_differs name={name}",
        )
    return lock


def reject_protected_path(path: Path | str, *, description: str) -> Path:
    candidate = Path(path).expanduser()
    lowered = {
        value
        for part in candidate.parts
        for value in (part.lower(), Path(part).stem.lower())
    }
    _require(
        not lowered.intersection(_PROTECTED_COMPONENTS),
        f"{description}_protected_split_forbidden path={candidate}",
    )
    return candidate


def assert_training_path_allowed(path: Path | str) -> Path:
    return reject_protected_path(path, description="training_path")


def validate_fresh_start_arguments(
    argv: Sequence[str],
    *,
    lineage: Mapping[str, Any] | None = None,
) -> None:
    for raw_argument in argv:
        argument = str(raw_argument)
        flag = argument.split("=", 1)[0]
        _require(
            flag not in _FRESH_START_FLAGS,
            f"fresh_start_forbids_argument flag={flag}",
        )
    if lineage is None:
        return
    source_checkpoint = lineage.get("source_checkpoint")
    _require(
        source_checkpoint is None,
        "fresh_start_forbids_lineage field=source_checkpoint",
    )
    for field in _FRESH_LINEAGE_FIELDS[1:]:
        _require(
            lineage.get(field, False) is False,
            f"fresh_start_forbids_lineage field={field}",
        )


def validate_critical_worktree(
    repo: Path | str = PROJECT_ROOT,
    critical_files: Sequence[str] = CRITICAL_TRAINING_FILES,
) -> None:
    resolved_repo = Path(repo).expanduser().resolve()
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *critical_files],
        cwd=resolved_repo,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(result.returncode == 0, "critical_worktree_git_status_failed")
    dirty = tuple(line for line in result.stdout.splitlines() if line.strip())
    _require(
        not dirty,
        "critical_worktree_differs: " + "; ".join(dirty),
    )


def load_pair_schedule(path: Path = PAIR_SCHEDULE) -> list[dict[str, Any]]:
    schedule_path = _regular_file(path, description="pair_schedule")
    lines = schedule_path.read_text(encoding="utf-8").splitlines()
    _require(len(lines) == TOTAL_PAIR_PRESENTATIONS, "pair_schedule_length_differs")
    _require(all(line.strip() for line in lines), "pair_schedule_contains_blank_rows")
    entries: list[dict[str, Any]] = []
    for schedule_index, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchContractError(
                f"pair_schedule_invalid_json row={schedule_index}"
            ) from exc
        _require(isinstance(entry, dict), "pair_schedule_entry_must_be_object")
        _validate_self_hash(
            entry,
            field="entry_sha256",
            description=f"pair_schedule_entry_{schedule_index}",
        )
        pair = entry.get("canonical_pair_ordinals")
        members = entry.get("members")
        expected_cycle = schedule_index // PAIRS_PER_CYCLE + 1
        expected_position = schedule_index % PAIRS_PER_CYCLE
        _require(
            entry.get("schema") == data_prep.PAIR_SCHEDULE_ENTRY_SCHEMA
            and entry.get("schedule_index") == schedule_index
            and entry.get("presentation") == schedule_index + 1
            and entry.get("optimizer_step") == schedule_index + 1
            and entry.get("cycle_index") == expected_cycle
            and entry.get("cycle_position") == expected_position
            and entry.get("pair_batch_size") == 2
            and isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in pair)
            and pair[0] < pair[1]
            and isinstance(members, list)
            and len(members) == 2
            and members[0].get("train_row_ordinal") == pair[0]
            and members[0].get("donor_train_row_ordinal") == pair[1]
            and members[1].get("train_row_ordinal") == pair[1]
            and members[1].get("donor_train_row_ordinal") == pair[0],
            f"pair_schedule_entry_contract_differs row={schedule_index}",
        )
        entries.append(entry)
    return entries


def validate_data_contract(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    root = data_root.expanduser().resolve()
    _require(root == DATA_ROOT.resolve(), "hard_failure_data_root_differs")
    for filename, expected_sha256 in FILE_SHA256.items():
        path = _regular_file(root / filename, description=filename)
        _require(
            sha256_file(path) == expected_sha256,
            f"hard_failure_artifact_sha256_differs file={filename}",
        )

    bundle = _load_json(root / "manifest.json", description="bundle_manifest")
    source = _load_json(root / "source_manifest.json", description="source_manifest")
    pair_manifest = _load_json(root / "pair_manifest.json", description="pair_manifest")
    schedule_manifest = _load_json(
        root / "pair_schedule_manifest.json",
        description="pair_schedule_manifest",
    )
    _validate_self_hash(bundle, field="manifest_sha256", description="bundle")
    _validate_self_hash(source, field="manifest_sha256", description="source")
    _validate_self_hash(
        pair_manifest,
        field="manifest_sha256",
        description="pair_manifest",
    )
    _validate_self_hash(
        schedule_manifest,
        field="manifest_sha256",
        description="pair_schedule_manifest",
    )
    _require(bundle.get("schema") == data_prep.SCHEMA, "bundle_schema_differs")
    _require(source.get("schema") == data_prep.SOURCE_SCHEMA, "source_schema_differs")
    _require(
        pair_manifest.get("schema") == data_prep.PAIRING_SCHEMA,
        "pair_manifest_schema_differs",
    )
    _require(
        schedule_manifest.get("schema") == data_prep.PAIR_SCHEDULE_MANIFEST_SCHEMA,
        "pair_schedule_manifest_schema_differs",
    )

    directed = pair_manifest.get("directed_pairs")
    _require(isinstance(directed, list) and len(directed) == 32, "directed_pairs_differ")
    donor_by_row: dict[int, int] = {}
    for entry in directed:
        _require(isinstance(entry, dict), "directed_pair_entry_invalid")
        _validate_self_hash(
            entry,
            field="entry_sha256",
            description="directed_pair_entry",
        )
        row = entry.get("train_row_ordinal")
        donor = entry.get("donor_train_row_ordinal")
        _require(
            isinstance(row, int)
            and not isinstance(row, bool)
            and isinstance(donor, int)
            and not isinstance(donor, bool)
            and row not in donor_by_row
            and row != donor,
            "directed_pair_ordinals_differ",
        )
        donor_by_row[row] = donor
    _require(sorted(donor_by_row) == list(range(32)), "directed_pair_coverage_differs")
    _require(
        all(donor_by_row.get(donor) == row for row, donor in donor_by_row.items()),
        "directed_pair_reciprocity_differs",
    )
    canonical_pairs = tuple(
        sorted({tuple(sorted((row, donor))) for row, donor in donor_by_row.items()})
    )
    _require(len(canonical_pairs) == PAIR_COUNT, "canonical_pair_count_differs")
    _require(
        canonical_sha256([list(pair) for pair in canonical_pairs])
        == CANONICAL_PAIRS_SHA256,
        "canonical_pairs_sha256_differs",
    )

    entries = load_pair_schedule(root / "pair_schedule.jsonl")
    scheduled_pairs = tuple(
        tuple(entry["canonical_pair_ordinals"]) for entry in entries
    )
    full_pair_cycles = tuple(
        scheduled_pairs[index : index + PAIRS_PER_CYCLE]
        for index in range(0, TOTAL_PAIR_PRESENTATIONS, PAIRS_PER_CYCLE)
    )
    canonical_set = set(canonical_pairs)
    _require(
        len(full_pair_cycles) == PAIR_CYCLES
        and all(len(set(cycle)) == PAIR_COUNT for cycle in full_pair_cycles)
        and all(set(cycle) == canonical_set for cycle in full_pair_cycles),
        "four_complete_pair_cycles_differ",
    )
    entries_sha256 = canonical_sha256(entries)
    ordered_pairs_sha256 = canonical_sha256(
        [list(pair) for pair in scheduled_pairs]
    )
    _require(entries_sha256 == PAIR_SCHEDULE_ENTRIES_SHA256, "entries_sha256_differs")
    _require(ordered_pairs_sha256 == ORDERED_PAIRS_SHA256, "ordered_pairs_sha256_differs")
    schedule_record = schedule_manifest.get("schedule")
    curriculum = schedule_manifest.get("curriculum")
    _require(
        isinstance(schedule_record, dict)
        and schedule_record.get("sha256") == PAIR_SCHEDULE_SHA256
        and schedule_record.get("rows") == TOTAL_PAIR_PRESENTATIONS
        and schedule_record.get("entries_sha256") == entries_sha256
        and schedule_record.get("ordered_pairs_sha256") == ordered_pairs_sha256,
        "schedule_manifest_binding_differs",
    )
    _require(
        isinstance(curriculum, dict)
        and curriculum.get("pair_cycles") == PAIR_CYCLES
        and curriculum.get("pairs_per_cycle") == PAIRS_PER_CYCLE
        and curriculum.get("pair_presentations") == TOTAL_PAIR_PRESENTATIONS
        and curriculum.get("optimizer_steps") == TOTAL_OPTIMIZER_STEPS
        and curriculum.get("gradient_accumulation_steps") == 1
        and curriculum.get("generation_endpoint_steps")
        == list(GENERATION_ENDPOINT_STEPS),
        "schedule_curriculum_binding_differs",
    )
    hard_curriculum = source.get("hard_failure_curriculum")
    train_schedule = (
        None
        if not isinstance(hard_curriculum, dict)
        else hard_curriculum.get("train_schedule")
    )
    _require(
        isinstance(train_schedule, dict)
        and train_schedule.get("schema") == data_prep.PAIR_CURRICULUM_BINDING_SCHEMA
        and train_schedule.get("pair_presentations") == TOTAL_PAIR_PRESENTATIONS
        and train_schedule.get("optimizer_steps") == TOTAL_OPTIMIZER_STEPS
        and train_schedule.get("gradient_accumulation_steps") == 1
        and train_schedule.get("generation_endpoint_steps")
        == list(GENERATION_ENDPOINT_STEPS),
        "source_train_schedule_binding_differs",
    )
    protected = (
        None
        if not isinstance(hard_curriculum, dict)
        else hard_curriculum.get("protected_evaluation")
    )
    _require(isinstance(protected, dict), "protected_evaluation_binding_missing")
    for name in ("official_validation", "hard32", "official_test"):
        record = protected.get(name)
        _require(
            isinstance(record, dict)
            and record.get("included") is False
            and record.get("path") is None,
            f"protected_evaluation_binding_differs name={name}",
        )

    endpoint_map = {
        step: step // PAIRS_PER_CYCLE for step in GENERATION_ENDPOINT_STEPS
    }
    return {
        "status": "pass",
        "data_root": str(root),
        "canonical_pairs": [list(pair) for pair in canonical_pairs],
        "canonical_pairs_sha256": CANONICAL_PAIRS_SHA256,
        "scheduled_ordinals": list(range(32)),
        "scheduled_pairs": [list(pair) for pair in scheduled_pairs],
        "full_pair_cycles": [
            [list(pair) for pair in cycle] for cycle in full_pair_cycles
        ],
        "generation_endpoint_by_step": endpoint_map,
        "schedule_entries_sha256": entries_sha256,
        "ordered_pairs_sha256": ordered_pairs_sha256,
        "expected_schedule_sha256": PAIR_SCHEDULE_SHA256,
        "actual_schedule_sha256": sha256_file(PAIR_SCHEDULE),
        "source_manifest_sha256": FILE_SHA256["source_manifest.json"],
        "train_file_sha256": FILE_SHA256["train.jsonl"],
    }


def validate_launch_contract(
    values: argparse.Namespace | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    supplied = {} if values is None else (
        vars(values) if isinstance(values, argparse.Namespace) else dict(values)
    )
    supplied.update(overrides)
    smoke = bool(supplied.get("smoke", supplied.get("one_pair_smoke", False)))
    expected = {
        "objective_version": OBJECTIVE_VERSION,
        "max_steps": 1 if smoke else TOTAL_OPTIMIZER_STEPS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 1 if smoke else SAVE_TOTAL_LIMIT,
        "target_layers": TARGET_LAYERS,
        "rank": RANK,
        "alpha": ALPHA,
        "delta_heads": DELTA_HEADS,
        "rwkv_ms_num_states": RWKV_MS_NUM_STATES,
        "rwkv_ms_semantics_version": RWKV_MS_SEMANTICS_VERSION,
        "rwkv_ms_chunk_size": RWKV_MS_CHUNK_SIZE,
        "rwkv_ms_boundary_mode": RWKV_MS_BOUNDARY_MODE,
        "state_reset_per_row": STATE_RESET_PER_ROW,
        "episode_read_write_enabled": READ_SIDE_WRITES_ENABLED,
        "memory_fusion_mode": MEMORY_FUSION_MODE,
        "memory_fusion_placement": MEMORY_FUSION_PLACEMENT,
        "per_device_train_batch_size": 1,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "validation_split_ratio": 0.0,
    }
    aliases = {
        "objective_version": "scene_state_generation_objective_version",
    }
    mismatches: list[str] = []
    for name, expected_value in expected.items():
        alias = aliases.get(name)
        if name in supplied:
            actual = supplied[name]
        elif alias is not None and alias in supplied:
            actual = supplied[alias]
        else:
            mismatches.append(name)
            continue
        if name == "target_layers" and not isinstance(actual, str):
            actual = tuple(actual)
        elif name == "delta_heads":
            if isinstance(actual, str):
                actual = tuple(
                    value.strip() for value in actual.split(",") if value.strip()
                )
            else:
                actual = tuple(actual)
        if actual != expected_value:
            mismatches.append(name)
    argv = supplied.get("argv", ())
    lineage = supplied.get("lineage")
    validate_fresh_start_arguments(argv, lineage=lineage)
    for field in ("train_file", "data_root", "output_dir", "cache_root"):
        if field in supplied and supplied[field] is not None:
            reject_protected_path(supplied[field], description=field)
    _require(not mismatches, "launch_contract_differs: " + ", ".join(mismatches))
    data = validate_data_contract()
    source_lock = validate_source_lock()
    return {
        "status": "pass",
        "run_mode": ONE_PAIR_SMOKE_RUN_MODE if smoke else PRODUCTION_RUN_MODE,
        "sampler_mode": (
            ONE_PAIR_SMOKE_SAMPLER_MODE if smoke else FIXED_SAMPLER_MODE
        ),
        "objective_version": OBJECTIVE_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "data_contract": data,
        "source_lock": {
            "path": str(SOURCE_LOCK.resolve()),
            "file_sha256": sha256_file(SOURCE_LOCK),
            "lock_sha256": source_lock["lock_sha256"],
        },
        **expected,
    }


__all__ = [
    "ALPHA",
    "CHECKPOINT_STEPS",
    "DATA_ROOT",
    "DELTA_HEADS",
    "FIXED_SAMPLER_MODE",
    "GENERATION_ENDPOINT_STEPS",
    "GRADIENT_ACCUMULATION_STEPS",
    "LaunchContractError",
    "OBJECTIVE_VERSION",
    "ONE_PAIR_SMOKE_RUN_MODE",
    "PAIR_COUNT",
    "PAIR_CYCLES",
    "PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP",
    "PAIRS_PER_CYCLE",
    "PRODUCTION_RUN_MODE",
    "RANK",
    "READ_SIDE_WRITES_ENABLED",
    "RWKV_MS_CHUNK_SIZE",
    "RWKV_MS_NUM_STATES",
    "RWKV_MS_SEMANTICS_VERSION",
    "SAVE_STEPS",
    "SAVE_TOTAL_LIMIT",
    "SOURCE_LOCK",
    "SOURCE_LOCK_SCHEMA",
    "STATE_RESET_PER_ROW",
    "TARGET_LAYERS",
    "TOTAL_OPTIMIZER_STEPS",
    "TOTAL_PAIR_PRESENTATIONS",
    "assert_training_path_allowed",
    "load_pair_schedule",
    "reject_protected_path",
    "validate_critical_worktree",
    "validate_data_contract",
    "validate_fresh_start_arguments",
    "validate_launch_contract",
    "validate_source_lock",
]
