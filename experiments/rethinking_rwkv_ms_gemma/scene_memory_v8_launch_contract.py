#!/usr/bin/env python3
"""Fail-closed production launch contract for Scene Memory V8 training blocks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SSD_ROOT = Path("/run/media/xiaol/B214449214445C0B")
SOURCE_LOCK = Path(__file__).with_name("scene_memory_v8_source_lock.json")
WARM_START_LOCK = Path(__file__).with_name(
    "scene_memory_v8_v7_checkpoint256_lock.json"
)
SOURCE_LOCK_FILE_SHA256 = (
    "921fa1a43bd88c5405f6469b1570ce57f86a210a57cd452463be36628c045d60"
)
SOURCE_LOCK_CANONICAL_SHA256 = (
    "7a8ee4b9aeb4f5201e23a3a6a07f63a2d0d7ba31040ffc1906e7363f2f08bf84"
)
WARM_START_LOCK_FILE_SHA256 = (
    "530d192a03b2be977df77df6c0f337cd4dd5331d84a91457dfea1199d09c4d3c"
)
WARM_START_LOCK_CANONICAL_SHA256 = (
    "e2d0ed1f65dca7ac419c3b5f87870a30e3b8c76b89801e082569bb53f7318e35"
)

TRAIN32_SHA256 = "20d395bc99a9e2e502d2ed86169664ea0162638c07a7b52a4b4d293c171dc7b9"
TRAIN32_ROWS_SHA256 = (
    "af80b1938196319e6595a0e6d0e2f2c9a6009963a82fea520d608e060a4fe957"
)
TRAIN32_PAIR_SHA256 = (
    "13555da56823d9597bf061d51ec6575db25cde49c044cab378f2050373fd78b6"
)
SCHEDULE_FILE_SHA256 = (
    "64fb83996bf7b211505022b94f4fa2e5ee0ab9f1fe87fad0bc53cd536326ea8a"
)
SCHEDULE_ENTRIES_SHA256 = (
    "979ca0c2dc253373eed6b4221cd6fa4c37f4a7a6e93173e8ce7f86f811e23df0"
)
SCHEDULE_ORDINALS_SHA256 = (
    "dfd2efa5f0fb8e5969fbb7f36689cc4d47d66166b40b2ee08c8d26d70f2d17f3"
)
SCHEDULE_MANIFEST_FILE_SHA256 = (
    "6096a1b93316186f92af0f25846669e8b5114aba540aad6f982c1b7e1341d251"
)
SCHEDULE_MANIFEST_CANONICAL_SHA256 = (
    "d1b2c865630428ca0f6193fde8e72b08542ecf9e51e56ab22c5941db14a542ea"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "13c6cd91b4ac64d25b74cbb3a8e5e46e65c2f6e2d71b5ae73b0bf285cb69f5be"
)
SOURCE_MANIFEST_CANONICAL_SHA256 = (
    "96edab2c7c02dde9c2294cca5101bdca33db96f1917893231c3a021c22c99a14"
)
BUNDLE_MANIFEST_FILE_SHA256 = (
    "cd47a5603c40bd431111b6c2f3b915d3a0031dcf626e93d0386d010e71839661"
)
BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "9d68f134f0db3e01f03c16f44948a467c9e30929ad9cdb9fc979e2b2760bdc2f"
)

CHECKPOINT_STEPS = (14, 28, 42, 56, 80, 104, 128, 152)
VALUE14_ORDINALS = (1, 3, 5, 9, 10, 14, 19, 20, 22, 23, 24, 26, 28, 31)
TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
BALANCED_QUOTAS = {
    "presence": 32,
    "same_cardinality_value": 32,
    "cross_cardinality_value": 32,
}
TOTAL_STEPS = 152
VALUE_STEPS = 56
SAVE_STEPS = 14
WARMUP_STEPS = 4
WARMUP_RATIO = 0.0
LEARNING_RATE = 2e-4
WARM_START_MODE = "scene_memory_v8_v7_checkpoint256_adapter_only"
FIXED_SAMPLER_MODE = "explicit_ordered_train_row_ordinal_v1"

SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_memory_v8_source_lock.v1"
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v8_source.v1"
SCHEDULE_ENTRY_SCHEMA = "rwkv_ms_scene_memory_v8_schedule_entry.v1"
SCHEDULE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v8_schedule_manifest.v1"
BUNDLE_SCHEMA = "rwkv_ms_scene_memory_v8_bundle.v1"
WARM_LOCK_SCHEMA = "rwkv_ms_scene_memory_v8_v7_warm_start_lock.v1"
CURRICULUM_SCHEMA = "rwkv_ms_scene_memory_v8_curriculum_binding.v1"
WARM_START_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v8_adapter_warm_start_receipt.v1"
WARM_START_LINEAGE_FILENAME = "warm_start_lineage_manifest.json"
CONTINUATION_LINEAGE_FILENAME = "continuation_manifest.json"
ABLATION_LINEAGE_FILENAME = "ablation_lineage_manifest.json"
WARM_START_LINEAGE_SCHEMA_VERSION = 1
CONTINUATION_LINEAGE_SCHEMA_VERSION = 1


class LaunchContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path, *, description: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{description}_missing_or_symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"{description}_invalid_json") from exc
    require(isinstance(payload, dict), f"{description}_must_be_object")
    return payload


def require_ssd(path: Path, *, description: str, ssd_root: Path = SSD_ROOT) -> Path:
    resolved = path.expanduser().resolve()
    root = ssd_root.expanduser().resolve()
    require(
        resolved == root or root in resolved.parents,
        f"{description}_must_stay_on_2t_ssd path={resolved}",
    )
    return resolved


def require_regular_hash(
    path: Path,
    expected_sha256: str,
    *,
    description: str,
) -> Path:
    require(path.is_file() and not path.is_symlink(), f"{description}_missing_or_symlink")
    require(sha256_file(path) == expected_sha256, f"{description}_hash_differs")
    return path


def validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str | None = None,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    actual = canonical_sha256(unsigned)
    require(recorded == actual, f"{description}_self_hash_differs")
    if expected is not None:
        require(actual == expected, f"{description}_canonical_hash_differs")
    return actual


def require_sha256(value: Any, *, description: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{description}_invalid",
    )
    return value


def _artifact_binding(
    artifacts: Mapping[str, Any],
    name: str,
    *,
    expected_sha256: str,
    ssd_root: Path,
) -> Path:
    binding = artifacts.get(name)
    require(isinstance(binding, dict), f"source_lock_artifact_missing key={name}")
    path = require_ssd(
        Path(str(binding.get("path", ""))),
        description=f"source_lock_{name}",
        ssd_root=ssd_root,
    )
    require(binding.get("sha256") == expected_sha256, f"source_lock_{name}_hash_differs")
    return require_regular_hash(
        path,
        expected_sha256,
        description=f"locked_{name}",
    )


def _validate_schedule(schedule_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    raw_lines = schedule_path.read_text(encoding="utf-8").splitlines()
    require(len(raw_lines) == TOTAL_STEPS, "schedule_row_count_differs")
    require(all(line.strip() for line in raw_lines), "schedule_contains_blank_rows")
    entries: list[dict[str, Any]] = []
    ordinals: list[int] = []
    for schedule_index, line in enumerate(raw_lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchContractError(
                f"schedule_invalid_json row={schedule_index}"
            ) from exc
        require(isinstance(entry, dict), f"schedule_entry_not_object row={schedule_index}")
        require(
            entry.get("schema") == SCHEDULE_ENTRY_SCHEMA,
            f"schedule_entry_schema_differs row={schedule_index}",
        )
        validate_self_hash(
            entry,
            field="entry_sha256",
            description=f"schedule_entry_{schedule_index}",
        )
        ordinal = entry.get("train_row_ordinal")
        require(
            entry.get("schedule_index") == schedule_index
            and entry.get("step") == schedule_index + 1
            and isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and 0 <= ordinal < 32,
            f"schedule_indexing_differs row={schedule_index}",
        )
        expected_phase = "value14" if schedule_index < VALUE_STEPS else "balanced"
        require(entry.get("phase") == expected_phase, f"schedule_phase_differs row={schedule_index}")
        require(
            entry.get("target_stratum") in TARGET_STRATA,
            f"schedule_stratum_differs row={schedule_index}",
        )
        entries.append(entry)
        ordinals.append(ordinal)
    require(canonical_sha256(entries) == SCHEDULE_ENTRIES_SHA256, "schedule_entries_hash_differs")
    require(canonical_sha256(ordinals) == SCHEDULE_ORDINALS_SHA256, "schedule_ordinals_hash_differs")
    for pass_index in range(4):
        start = pass_index * len(VALUE14_ORDINALS)
        current = ordinals[start : start + len(VALUE14_ORDINALS)]
        require(
            len(current) == len(set(current)) == len(VALUE14_ORDINALS)
            and set(current) == set(VALUE14_ORDINALS),
            f"value14_pass_differs pass={pass_index}",
        )
    balanced = entries[VALUE_STEPS:]
    counts = Counter(str(entry["target_stratum"]) for entry in balanced)
    require(
        {stratum: counts[stratum] for stratum in TARGET_STRATA} == BALANCED_QUOTAS,
        "balanced_schedule_quotas_differ",
    )
    for round_index in range(32):
        current = balanced[round_index * 3 : round_index * 3 + 3]
        require(
            {entry["target_stratum"] for entry in current} == set(TARGET_STRATA),
            f"balanced_round_differs round={round_index}",
        )
    return entries, ordinals


def validate_data_contract(
    *,
    source_lock_path: Path = SOURCE_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    source_lock_path = source_lock_path.expanduser().resolve()
    require_regular_hash(
        source_lock_path,
        SOURCE_LOCK_FILE_SHA256,
        description="source_lock_file",
    )
    lock = load_object(source_lock_path, description="source_lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "source_lock_schema_differs")
    validate_self_hash(
        lock,
        field="lock_sha256",
        expected=SOURCE_LOCK_CANONICAL_SHA256,
        description="source_lock",
    )
    require(lock.get("parent_train32_sha256") == TRAIN32_SHA256, "source_lock_train32_differs")
    curriculum = lock.get("curriculum")
    require(isinstance(curriculum, dict), "source_lock_curriculum_missing")
    require(
        curriculum
        == {
            "balanced_quotas": BALANCED_QUOTAS,
            "balanced_steps": 96,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "total_steps": TOTAL_STEPS,
            "value14_ordinals": list(VALUE14_ORDINALS),
            "value14_passes": 4,
            "value14_steps": VALUE_STEPS,
        },
        "source_lock_curriculum_differs",
    )
    forbidden = lock.get("fixed_hard32")
    require(isinstance(forbidden, dict), "source_lock_hard32_binding_missing")
    require(
        forbidden.get("role") == "protected_evaluation_only_not_scheduled",
        "hard32_role_differs",
    )
    forbidden_raw = Path(str(forbidden.get("path", ""))).expanduser()
    require(forbidden_raw.is_absolute(), "hard32_forbidden_path_must_be_absolute")
    forbidden_path = Path(os.path.normpath(str(forbidden_raw)))

    artifacts = lock.get("artifacts")
    require(isinstance(artifacts, dict), "source_lock_artifacts_missing")
    bundle_path = _artifact_binding(
        artifacts,
        "bundle_manifest",
        expected_sha256=BUNDLE_MANIFEST_FILE_SHA256,
        ssd_root=ssd_root,
    )
    schedule_path = _artifact_binding(
        artifacts,
        "schedule",
        expected_sha256=SCHEDULE_FILE_SHA256,
        ssd_root=ssd_root,
    )
    schedule_manifest_path = _artifact_binding(
        artifacts,
        "schedule_manifest",
        expected_sha256=SCHEDULE_MANIFEST_FILE_SHA256,
        ssd_root=ssd_root,
    )
    source_manifest_path = _artifact_binding(
        artifacts,
        "source_manifest",
        expected_sha256=SOURCE_MANIFEST_FILE_SHA256,
        ssd_root=ssd_root,
    )
    require(
        forbidden_path
        not in {bundle_path, schedule_path, schedule_manifest_path, source_manifest_path},
        "hard32_artifact_entered_training_contract",
    )

    entries, ordinals = _validate_schedule(schedule_path)
    schedule_manifest = load_object(
        schedule_manifest_path,
        description="schedule_manifest",
    )
    require(
        schedule_manifest.get("schema") == SCHEDULE_MANIFEST_SCHEMA,
        "schedule_manifest_schema_differs",
    )
    validate_self_hash(
        schedule_manifest,
        field="manifest_sha256",
        expected=SCHEDULE_MANIFEST_CANONICAL_SHA256,
        description="schedule_manifest",
    )
    schedule_binding = schedule_manifest.get("schedule")
    schedule_curriculum = schedule_manifest.get("curriculum")
    require(isinstance(schedule_binding, dict), "schedule_manifest_binding_missing")
    require(isinstance(schedule_curriculum, dict), "schedule_manifest_curriculum_missing")
    require(
        schedule_binding.get("path") == str(schedule_path)
        and schedule_binding.get("sha256") == SCHEDULE_FILE_SHA256
        and schedule_binding.get("rows") == TOTAL_STEPS
        and schedule_binding.get("entries_sha256") == SCHEDULE_ENTRIES_SHA256
        and schedule_binding.get("ordered_train_row_ordinals_sha256")
        == SCHEDULE_ORDINALS_SHA256,
        "schedule_manifest_schedule_differs",
    )
    require(
        schedule_curriculum.get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and schedule_curriculum.get("total_steps") == TOTAL_STEPS
        and schedule_curriculum.get("entries_sha256") == SCHEDULE_ENTRIES_SHA256,
        "schedule_manifest_curriculum_differs",
    )

    source = load_object(source_manifest_path, description="source_manifest")
    require(source.get("schema") == SOURCE_SCHEMA, "source_manifest_schema_differs")
    validate_self_hash(
        source,
        field="manifest_sha256",
        expected=SOURCE_MANIFEST_CANONICAL_SHA256,
        description="source_manifest",
    )
    source_contract = source.get("contract")
    require(isinstance(source_contract, dict), "source_manifest_contract_missing")
    require(
        source_contract.get("source_split") == "train"
        and source_contract.get("val_rows") == 0
        and source_contract.get("test_rows") == 0
        and source_contract.get("hard32_rows") == 0,
        "source_manifest_split_or_hard32_contract_differs",
    )
    train_partition = source.get("partitions", {}).get("train", {})
    train_binding = train_partition.get("data", {})
    row_binding = train_partition.get("row_manifest", {})
    require(train_partition.get("rows") == 32, "source_manifest_train_rows_differs")
    train_path = require_ssd(
        Path(str(train_binding.get("path", ""))),
        description="train32",
        ssd_root=ssd_root,
    )
    rows_path = require_ssd(
        Path(str(row_binding.get("path", ""))),
        description="train32_rows",
        ssd_root=ssd_root,
    )
    require(train_path != forbidden_path and rows_path != forbidden_path, "hard32_entered_train_partition")
    require(train_binding.get("sha256") == TRAIN32_SHA256, "source_manifest_train_hash_differs")
    require(row_binding.get("sha256") == TRAIN32_ROWS_SHA256, "source_manifest_rows_hash_differs")
    require_regular_hash(train_path, TRAIN32_SHA256, description="locked_train32")
    require_regular_hash(rows_path, TRAIN32_ROWS_SHA256, description="locked_train32_rows")
    require(
        len([line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        == 32,
        "locked_train32_row_count_differs",
    )
    pairing = source.get("v7_pairing")
    require(isinstance(pairing, dict), "source_manifest_pairing_missing")
    pair_record = pairing.get("pair_manifest")
    require(isinstance(pair_record, dict), "source_manifest_pair_manifest_missing")
    pair_path = require_ssd(
        Path(str(pair_record.get("path", ""))),
        description="train32_pair_manifest",
        ssd_root=ssd_root,
    )
    require(pair_path != forbidden_path, "hard32_entered_pairing_contract")
    require(pair_record.get("sha256") == TRAIN32_PAIR_SHA256, "pair_manifest_hash_differs")
    require_regular_hash(pair_path, TRAIN32_PAIR_SHA256, description="locked_pair_manifest")

    v8_curriculum = source.get("v8_curriculum")
    require(
        isinstance(v8_curriculum, dict)
        and v8_curriculum.get("schema") == CURRICULUM_SCHEMA,
        "source_manifest_v8_curriculum_differs",
    )
    require(
        v8_curriculum.get("parent_train32_sha256") == TRAIN32_SHA256
        and v8_curriculum.get("total_steps") == TOTAL_STEPS
        and v8_curriculum.get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and v8_curriculum.get("value14_ordinals") == list(VALUE14_ORDINALS),
        "source_manifest_curriculum_identity_differs",
    )
    require(
        v8_curriculum.get("schedule", {}).get("path") == str(schedule_path)
        and v8_curriculum.get("schedule", {}).get("sha256") == SCHEDULE_FILE_SHA256
        and v8_curriculum.get("schedule", {}).get("entries_sha256")
        == SCHEDULE_ENTRIES_SHA256
        and v8_curriculum.get("schedule_manifest", {}).get("path")
        == str(schedule_manifest_path)
        and v8_curriculum.get("schedule_manifest", {}).get("sha256")
        == SCHEDULE_MANIFEST_FILE_SHA256,
        "source_manifest_curriculum_artifacts_differ",
    )

    bundle = load_object(bundle_path, description="bundle_manifest")
    require(bundle.get("schema") == BUNDLE_SCHEMA, "bundle_manifest_schema_differs")
    validate_self_hash(
        bundle,
        field="manifest_sha256",
        expected=BUNDLE_MANIFEST_CANONICAL_SHA256,
        description="bundle_manifest",
    )
    leakage = bundle.get("leakage")
    require(
        isinstance(leakage, dict)
        and leakage.get("source_split") == "train"
        and leakage.get("val_rows_in_schedule") == 0
        and leakage.get("test_rows_in_schedule") == 0
        and leakage.get("hard32_rows_in_schedule") == 0,
        "bundle_leakage_contract_differs",
    )
    return {
        "train_file": str(train_path),
        "train_file_sha256": TRAIN32_SHA256,
        "source_manifest": str(source_manifest_path),
        "source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "schedule": str(schedule_path),
        "schedule_file_sha256": SCHEDULE_FILE_SHA256,
        "schedule_entries_sha256": SCHEDULE_ENTRIES_SHA256,
        "schedule_manifest": str(schedule_manifest_path),
        "schedule_manifest_file_sha256": SCHEDULE_MANIFEST_FILE_SHA256,
        "pair_manifest": str(pair_path),
        "pair_manifest_file_sha256": TRAIN32_PAIR_SHA256,
        "forbidden_hard32_path": str(forbidden_path),
        "ordinals": ordinals,
        "entries": entries,
    }


def validate_warm_start_contract(
    *,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    warm_start_lock_path = warm_start_lock_path.expanduser().resolve()
    require_regular_hash(
        warm_start_lock_path,
        WARM_START_LOCK_FILE_SHA256,
        description="warm_start_lock_file",
    )
    lock = load_object(warm_start_lock_path, description="warm_start_lock")
    require(lock.get("schema") == WARM_LOCK_SCHEMA, "warm_start_lock_schema_differs")
    validate_self_hash(
        lock,
        field="lock_sha256",
        expected=WARM_START_LOCK_CANONICAL_SHA256,
        description="warm_start_lock",
    )
    require(
        lock.get("source_state_imports")
        == {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
            "global_step": False,
        },
        "warm_start_import_policy_differs",
    )
    require(
        lock.get("target_fresh_start")
        == {
            "global_step": 0,
            "optimizer_created_after_adapter_load": True,
            "optimizer_family": "AdamW",
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "trainer_state": "fresh",
            "rng_state": "fresh_from_v8_seed",
        },
        "warm_start_fresh_state_policy_differs",
    )
    checkpoint = require_ssd(
        Path(str(lock.get("source_checkpoint", ""))),
        description="warm_start_checkpoint",
        ssd_root=ssd_root,
    )
    require(
        checkpoint.is_dir()
        and not checkpoint.is_symlink()
        and checkpoint.name == "checkpoint-256",
        "warm_start_checkpoint_invalid",
    )
    artifacts = lock.get("artifacts")
    require(isinstance(artifacts, dict), "warm_start_artifacts_missing")
    for filename, binding in artifacts.items():
        require(isinstance(binding, dict), f"warm_start_binding_invalid file={filename}")
        artifact = checkpoint / filename
        require(
            artifact.is_file() and not artifact.is_symlink(),
            f"warm_start_artifact_missing file={filename}",
        )
        require(
            artifact.stat().st_size == binding.get("bytes"),
            f"warm_start_artifact_size_differs file={filename}",
        )
        require(
            sha256_file(artifact) == binding.get("sha256"),
            f"warm_start_artifact_hash_differs file={filename}",
        )
    return {
        "warm_start_checkpoint": str(checkpoint),
        "warm_start_lock": str(warm_start_lock_path),
        "warm_start_lock_file_sha256": WARM_START_LOCK_FILE_SHA256,
        "warm_start_mode": WARM_START_MODE,
    }


def _require_protocol_fields(protocol: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 11,
        "memory_objective_version": (
            "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
        ),
        "memory_loss_mode": "scene_state_generation_ce",
        "train_file": data["train_file"],
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "save_steps": SAVE_STEPS,
        "num_train_epochs": 1.0,
        "train_sampler_seed": None,
        "train_sampler_mode": FIXED_SAMPLER_MODE,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_generated_unlikelihood_weight": 0.5,
        "scene_generation_generated_unlikelihood_max_wrong_tokens": 4,
        "scene_generation_generated_rollout_extra_tokens": 4,
        "scene_generation_generated_rollout_max_tokens": 24,
    }
    mismatches = [key for key, value in expected.items() if protocol.get(key) != value]
    require(not mismatches, "resume_protocol_differs fields=" + ",".join(mismatches))
    source_identity = protocol.get("scene_state_source_manifest")
    require(isinstance(source_identity, dict), "resume_protocol_source_manifest_missing")
    require(
        source_identity.get("path") == data["source_manifest"]
        and source_identity.get("file_sha256") == data["source_manifest_file_sha256"]
        and source_identity.get("train_file_sha256") == TRAIN32_SHA256
        and source_identity.get("train_rows") == 32
        and source_identity.get("train_source_split") == "train",
        "resume_protocol_source_manifest_differs",
    )
    schedule = protocol.get("train_schedule")
    require(isinstance(schedule, dict), "resume_protocol_schedule_missing")
    require(
        schedule.get("schema") == CURRICULUM_SCHEMA
        and schedule.get("schedule_path") == data["schedule"]
        and schedule.get("schedule_file_sha256") == SCHEDULE_FILE_SHA256
        and schedule.get("schedule_entries_sha256") == SCHEDULE_ENTRIES_SHA256
        and schedule.get("schedule_manifest_path") == data["schedule_manifest"]
        and schedule.get("schedule_manifest_file_sha256")
        == SCHEDULE_MANIFEST_FILE_SHA256
        and schedule.get("schedule_manifest_sha256")
        == SCHEDULE_MANIFEST_CANONICAL_SHA256
        and schedule.get("ordered_train_row_ordinals_sha256")
        == SCHEDULE_ORDINALS_SHA256
        and schedule.get("total_steps") == TOTAL_STEPS
        and schedule.get("checkpoint_steps") == list(CHECKPOINT_STEPS)
        and schedule.get("value14_ordinals") == list(VALUE14_ORDINALS),
        "resume_protocol_schedule_differs",
    )


def _require_only_lineage_file(checkpoint: Path, expected_filename: str) -> Path:
    expected_path = checkpoint / expected_filename
    require(
        expected_path.is_file() and not expected_path.is_symlink(),
        f"resume_lineage_missing_or_symlink filename={expected_filename}",
    )
    unexpected = [
        filename
        for filename in (
            WARM_START_LINEAGE_FILENAME,
            CONTINUATION_LINEAGE_FILENAME,
            ABLATION_LINEAGE_FILENAME,
        )
        if filename != expected_filename and os.path.lexists(checkpoint / filename)
    ]
    require(
        not unexpected,
        "resume_lineage_unexpected filenames=" + ",".join(unexpected),
    )
    return expected_path


def _load_checkpoint_horizon(
    *,
    checkpoint: Path,
    expected_step: int,
    data: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        f"{description}_invalid",
    )
    state = load_object(
        checkpoint / "trainer_state.json",
        description=f"{description}_trainer_state",
    )
    protocol = load_object(
        checkpoint / "training_protocol.json",
        description=f"{description}_training_protocol",
    )
    try:
        global_step = int(state["global_step"])
        effective_max_steps = int(state["max_steps"])
        protocol_max_steps = int(protocol["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchContractError(f"{description}_horizon_invalid") from exc
    require(
        checkpoint.name == f"checkpoint-{expected_step}"
        and global_step == effective_max_steps == protocol_max_steps == expected_step,
        f"{description}_is_not_completed_expected_horizon",
    )
    _require_protocol_fields(protocol, data)
    return protocol


def _validate_warm_start_lineage(
    *,
    checkpoint: Path,
    protocol: Mapping[str, Any],
    warm: Mapping[str, Any],
) -> dict[str, str]:
    lineage_path = _require_only_lineage_file(
        checkpoint,
        WARM_START_LINEAGE_FILENAME,
    )
    lineage = load_object(lineage_path, description="resume_warm_start_lineage")
    require(
        lineage.get("schema") == WARM_START_RECEIPT_SCHEMA
        and lineage.get("schema_version") == WARM_START_LINEAGE_SCHEMA_VERSION
        and lineage.get("mode") == WARM_START_MODE,
        "resume_warm_start_lineage_identity_differs",
    )
    root_receipt_sha256 = validate_self_hash(
        lineage,
        field="receipt_sha256",
        description="resume_warm_start_lineage",
    )
    require(
        lineage.get("source_checkpoint") == warm["warm_start_checkpoint"]
        and lineage.get("source_global_step") == 256,
        "resume_warm_start_source_checkpoint_differs",
    )
    source_lock = lineage.get("source_lock")
    require(
        isinstance(source_lock, dict)
        and source_lock.get("path") == warm["warm_start_lock"]
        and source_lock.get("lock_sha256") == WARM_START_LOCK_CANONICAL_SHA256,
        "resume_warm_start_source_lock_differs",
    )
    locked_warm_start = load_object(
        Path(str(warm["warm_start_lock"])),
        description="resume_warm_start_lock",
    )
    require(
        lineage.get("source_artifacts") == locked_warm_start.get("artifacts")
        and lineage.get("source_state_imports")
        == {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
            "global_step": False,
        }
        and lineage.get("post_load_bit_equal") is True,
        "resume_warm_start_import_evidence_differs",
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
            "rng_state": "fresh_from_v8_seed",
        }
        and lineage.get("trainer_resume_from_checkpoint") is None
        and lineage.get("target_initial_global_step") == 0
        and lineage.get("pre_train_global_step") == 0
        and lineage.get("fresh_optimizer_created") is True
        and lineage.get("fresh_optimizer_state_entries_before_train") == 0
        and lineage.get("fresh_scheduler_created_before_train") is False
        and isinstance(lineage.get("fresh_optimizer_class"), str)
        and str(lineage["fresh_optimizer_class"]).endswith(".AdamW"),
        "resume_warm_start_fresh_optimizer_evidence_differs",
    )
    config = load_object(
        checkpoint / "delta_mem_config.json",
        description="resume_warm_start_delta_config",
    )
    require(
        lineage.get("target_delta_config_sha256") == canonical_sha256(config),
        "resume_warm_start_target_delta_config_differs",
    )
    target_protocol_sha256 = require_sha256(
        lineage.get("target_training_protocol_sha256"),
        description="resume_warm_start_target_protocol_sha256",
    )
    require(
        target_protocol_sha256 == canonical_sha256(protocol),
        "resume_warm_start_target_protocol_differs",
    )
    pairing = load_object(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        description="resume_warm_start_pairing_manifest",
    )
    pairing_sha256 = require_sha256(
        pairing.get("manifest_sha256"),
        description="resume_warm_start_pairing_manifest_sha256",
    )
    require(
        lineage.get("target_scene_state_pairing_manifest_sha256") == pairing_sha256,
        "resume_warm_start_target_pairing_differs",
    )
    return {
        "root_warm_start_receipt_sha256": root_receipt_sha256,
        "lineage_filename": WARM_START_LINEAGE_FILENAME,
        "lineage_file_sha256": sha256_file(lineage_path),
    }


def _validate_continuation_lineage(
    *,
    checkpoint: Path,
    checkpoint_step: int,
    protocol: Mapping[str, Any],
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path,
) -> dict[str, str]:
    lineage_path = _require_only_lineage_file(
        checkpoint,
        CONTINUATION_LINEAGE_FILENAME,
    )
    lineage = load_object(lineage_path, description="resume_continuation_lineage")
    require(
        lineage.get("schema_version") == CONTINUATION_LINEAGE_SCHEMA_VERSION
        and lineage.get("mode") == "extend",
        "resume_continuation_lineage_identity_differs",
    )
    validate_self_hash(
        lineage,
        field="manifest_sha256",
        description="resume_continuation_lineage",
    )

    checkpoint_position = CHECKPOINT_STEPS.index(checkpoint_step)
    require(checkpoint_position > 0, "resume_continuation_source_position_invalid")
    expected_source_step = CHECKPOINT_STEPS[checkpoint_position - 1]
    source_checkpoint_value = lineage.get("source_checkpoint")
    require(
        isinstance(source_checkpoint_value, str)
        and Path(source_checkpoint_value).expanduser().is_absolute(),
        "resume_continuation_source_checkpoint_invalid",
    )
    source_checkpoint = require_ssd(
        Path(source_checkpoint_value),
        description="resume_continuation_source_checkpoint",
        ssd_root=ssd_root,
    )
    require(
        source_checkpoint_value == str(source_checkpoint),
        "resume_continuation_source_checkpoint_not_canonical",
    )
    source_protocol = _load_checkpoint_horizon(
        checkpoint=source_checkpoint,
        expected_step=expected_source_step,
        data=data,
        description="resume_continuation_source_checkpoint",
    )
    require(
        lineage.get("source_global_step") == expected_source_step
        and lineage.get("source_effective_max_steps") == expected_source_step
        and lineage.get("source_max_steps") == expected_source_step
        and lineage.get("target_max_steps") == checkpoint_step
        and lineage.get("source_num_train_epochs")
        == float(source_protocol["num_train_epochs"])
        and lineage.get("target_num_train_epochs")
        == float(protocol["num_train_epochs"])
        and lineage.get("lr_scheduler_type") == protocol["lr_scheduler_type"]
        and lineage.get("warmup_steps") == protocol["warmup_steps"],
        "resume_continuation_horizon_differs",
    )

    expected_source_lineage_filename = (
        WARM_START_LINEAGE_FILENAME
        if expected_source_step == CHECKPOINT_STEPS[0]
        else CONTINUATION_LINEAGE_FILENAME
    )
    require(
        lineage.get("source_lineage_filename")
        == expected_source_lineage_filename,
        "resume_continuation_source_lineage_filename_differs",
    )
    source_lineage_path = _require_only_lineage_file(
        source_checkpoint,
        expected_source_lineage_filename,
    )
    require(
        lineage.get("source_lineage_file_sha256")
        == sha256_file(source_lineage_path),
        "resume_continuation_source_lineage_file_hash_differs",
    )
    require(
        lineage.get("source_training_protocol_sha256")
        == canonical_sha256(source_protocol),
        "resume_continuation_source_protocol_differs",
    )
    require(
        lineage.get("target_training_protocol_sha256")
        == canonical_sha256(protocol),
        "resume_continuation_target_protocol_differs",
    )
    source_lineage = _validate_checkpoint_lineage(
        checkpoint=source_checkpoint,
        checkpoint_step=expected_source_step,
        protocol=source_protocol,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )
    root_receipt_sha256 = require_sha256(
        lineage.get("root_warm_start_receipt_sha256"),
        description="resume_continuation_root_warm_start_receipt_sha256",
    )
    require(
        root_receipt_sha256
        == source_lineage["root_warm_start_receipt_sha256"],
        "resume_continuation_root_warm_start_receipt_differs",
    )
    return {
        "root_warm_start_receipt_sha256": root_receipt_sha256,
        "lineage_filename": CONTINUATION_LINEAGE_FILENAME,
        "lineage_file_sha256": sha256_file(lineage_path),
    }


def _validate_checkpoint_lineage(
    *,
    checkpoint: Path,
    checkpoint_step: int,
    protocol: Mapping[str, Any],
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path,
) -> dict[str, str]:
    if checkpoint_step == CHECKPOINT_STEPS[0]:
        return _validate_warm_start_lineage(
            checkpoint=checkpoint,
            protocol=protocol,
            warm=warm,
        )
    return _validate_continuation_lineage(
        checkpoint=checkpoint,
        checkpoint_step=checkpoint_step,
        protocol=protocol,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )


def validate_resume_contract(
    *,
    resume_checkpoint: Path,
    target_step: int,
    data: Mapping[str, Any],
    warm: Mapping[str, Any],
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    checkpoint = require_ssd(
        resume_checkpoint,
        description="resume_checkpoint",
        ssd_root=ssd_root,
    )
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), "resume_checkpoint_invalid")
    required = (
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_protocol.json",
        "scene_state_identity_pairing_manifest.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    require(not missing, "resume_checkpoint_incomplete missing=" + ",".join(missing))
    state = load_object(checkpoint / "trainer_state.json", description="resume_trainer_state")
    protocol = load_object(
        checkpoint / "training_protocol.json",
        description="resume_training_protocol",
    )
    try:
        source_step = int(state["global_step"])
        source_max_steps = int(state["max_steps"])
        protocol_max_steps = int(protocol["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LaunchContractError("resume_checkpoint_horizon_invalid") from exc
    require(
        checkpoint.name == f"checkpoint-{source_step}"
        and source_step == source_max_steps == protocol_max_steps,
        "resume_checkpoint_is_not_completed_horizon",
    )
    require(source_step in CHECKPOINT_STEPS, "resume_source_step_not_locked_checkpoint")
    source_position = CHECKPOINT_STEPS.index(source_step)
    require(source_position + 1 < len(CHECKPOINT_STEPS), "resume_source_is_final_checkpoint")
    require(
        target_step == CHECKPOINT_STEPS[source_position + 1],
        "resume_target_is_not_next_locked_checkpoint",
    )
    _require_protocol_fields(protocol, data)
    config = load_object(
        checkpoint / "delta_mem_config.json",
        description="resume_delta_config",
    )
    expected_config = {
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
    config_mismatches = [
        key for key, value in expected_config.items() if config.get(key) != value
    ]
    require(
        not config_mismatches,
        "resume_delta_config_differs fields=" + ",".join(config_mismatches),
    )
    lineage = _validate_checkpoint_lineage(
        checkpoint=checkpoint,
        checkpoint_step=source_step,
        protocol=protocol,
        data=data,
        warm=warm,
        ssd_root=ssd_root,
    )
    next_entry = data["entries"][source_step]
    require(
        next_entry["schedule_index"] == source_step
        and next_entry["step"] == source_step + 1,
        "resume_schedule_cursor_differs",
    )
    return {
        "launch_mode": "resume",
        "source_step": source_step,
        "target_step": target_step,
        "resume_checkpoint": str(checkpoint),
        "resume_schedule_cursor": source_step,
        "next_schedule_ordinal": next_entry["train_row_ordinal"],
        "next_schedule_entry_sha256": next_entry["entry_sha256"],
        "root_warm_start_receipt_sha256": lineage[
            "root_warm_start_receipt_sha256"
        ],
        "source_lineage_filename": lineage["lineage_filename"],
        "source_lineage_file_sha256": lineage["lineage_file_sha256"],
    }


def validate_launch_contract(
    *,
    target_step: int,
    resume_checkpoint: Path | None = None,
    smoke: bool = False,
    source_lock_path: Path = SOURCE_LOCK,
    warm_start_lock_path: Path = WARM_START_LOCK,
    ssd_root: Path = SSD_ROOT,
) -> dict[str, Any]:
    if smoke:
        require(target_step == 1, "smoke_launch_must_target_step1")
        require(resume_checkpoint is None, "smoke_launch_forbids_resume")
    else:
        require(target_step in CHECKPOINT_STEPS, "target_step_not_locked_checkpoint")
    data = validate_data_contract(
        source_lock_path=source_lock_path,
        ssd_root=ssd_root,
    )
    warm = validate_warm_start_contract(
        warm_start_lock_path=warm_start_lock_path,
        ssd_root=ssd_root,
    )
    if smoke:
        cursor = {
            "launch_mode": "warm_start_smoke",
            "source_step": 0,
            "target_step": 1,
            "resume_checkpoint": None,
            "resume_schedule_cursor": 0,
            "next_schedule_ordinal": data["entries"][0]["train_row_ordinal"],
            "next_schedule_entry_sha256": data["entries"][0]["entry_sha256"],
        }
    elif resume_checkpoint is None:
        require(target_step == CHECKPOINT_STEPS[0], "fresh_launch_must_target_step14")
        cursor = {
            "launch_mode": "warm_start",
            "source_step": 0,
            "target_step": target_step,
            "resume_checkpoint": None,
            "resume_schedule_cursor": 0,
            "next_schedule_ordinal": data["entries"][0]["train_row_ordinal"],
            "next_schedule_entry_sha256": data["entries"][0]["entry_sha256"],
        }
    else:
        cursor = validate_resume_contract(
            resume_checkpoint=resume_checkpoint,
            target_step=target_step,
            data=data,
            warm=warm,
            ssd_root=ssd_root,
        )
    result = {
        key: value
        for key, value in data.items()
        if key not in {"entries", "ordinals", "forbidden_hard32_path"}
    }
    result.update(warm)
    result.update(cursor)
    result.update(
        {
            "total_steps": TOTAL_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "save_steps": 1 if smoke else SAVE_STEPS,
            "hard32_access": "forbidden_not_opened_by_launch_contract",
        }
    )
    return result


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
        result["next_schedule_ordinal"],
        result["next_schedule_entry_sha256"],
        result["save_steps"],
    )
    rendered = tuple(str(field) for field in fields)
    require(
        all("\t" not in field and "\n" not in field for field in rendered),
        "launch_contract_tsv_control_character",
    )
    return "\t".join(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
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
            smoke=args.smoke,
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
