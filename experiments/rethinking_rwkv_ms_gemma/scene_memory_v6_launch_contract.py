#!/usr/bin/env python3
"""Validate and materialize the scene V6 identity-proof launch contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import sys
from typing import Iterable, Mapping, Sequence


SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_v6_identity_source_lock.v1"
LAUNCH_SCHEMA = "rwkv_ms_scene_v6_identity_launch.v1"
DATA_SCHEMA = "rwkv_ms_scene_v6_identity_data.v1"
TOKENIZATION_LOCK_SCHEMA = "rwkv_ms_scene_v6_identity_tokenization_lock.v1"
PREPARE_RECEIPT_SCHEMA = "rwkv_ms_scene_v6_identity_prepare.v1"
EXPERIMENT = "scene_memory_v6_identity_proof"

EXPECTED_REPO = Path("/home/xiaol/X/Multi-state-RWKV-online-memory")
EXPECTED_MODEL = Path("/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it")
EXPECTED_PYTHON_BIN = Path("/home/xiaol/X/delta-Mem/.venv/bin/python")
EXPECTED_RUN_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory"
)
EXPECTED_PAIR_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "pairs_candidate64_failure32_holdout32_v1"
)
EXPECTED_PAIR_MANIFEST = EXPECTED_PAIR_ROOT / "manifest.json"
EXPECTED_PAIR_MANIFEST_SHA256 = (
    "2ceb291b9c21063164e30ca0b8b052798f8ba42d9a089a5abc78d1cb321dc008"
)
EXPECTED_TRAIN = EXPECTED_PAIR_ROOT / "train.jsonl"
EXPECTED_TRAIN_SHA256 = (
    "5f35f6ed41a2edaf88afee83626f17c34da38f5cb61cf4b6796a03eaae38f897"
)
EXPECTED_HARD32 = EXPECTED_PAIR_ROOT / "holdout.jsonl"
EXPECTED_HARD32_SHA256 = (
    "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
)
EXPECTED_HARD32_INDICES = EXPECTED_PAIR_ROOT / "holdout_source_indices.json"
EXPECTED_HARD32_INDICES_SHA256 = (
    "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
)
EXPECTED_TRAIN_ROW_MANIFEST = EXPECTED_PAIR_ROOT / "train_manifest.jsonl"
EXPECTED_TRAIN_ROW_MANIFEST_SHA256 = (
    "d112056a80b9dc13728b021646c0fbe3da5c3c41641fb28bb8c5448b1f8427fa"
)
EXPECTED_HARD32_ROW_MANIFEST = EXPECTED_PAIR_ROOT / "holdout_manifest.jsonl"
EXPECTED_HARD32_ROW_MANIFEST_SHA256 = (
    "6802d992805164342ea4ed16b9113814ee472ad363aa76eaf5298147e7a0d1cc"
)
EXPECTED_DATA_ARTIFACTS: Mapping[str, tuple[Path, str]] = {
    "pair_manifest": (EXPECTED_PAIR_MANIFEST, EXPECTED_PAIR_MANIFEST_SHA256),
    "train32": (EXPECTED_TRAIN, EXPECTED_TRAIN_SHA256),
    "hard32": (EXPECTED_HARD32, EXPECTED_HARD32_SHA256),
    "hard32_indices": (
        EXPECTED_HARD32_INDICES,
        EXPECTED_HARD32_INDICES_SHA256,
    ),
    "train32_row_manifest": (
        EXPECTED_TRAIN_ROW_MANIFEST,
        EXPECTED_TRAIN_ROW_MANIFEST_SHA256,
    ),
    "hard32_row_manifest": (
        EXPECTED_HARD32_ROW_MANIFEST,
        EXPECTED_HARD32_ROW_MANIFEST_SHA256,
    ),
}
EXPECTED_SOURCE_LOCK = (
    EXPECTED_REPO
    / "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_source_lock.json"
)
EXPECTED_TOKENIZATION_LOCK = (
    EXPECTED_REPO
    / "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_tokenized_cache_lock.json"
)
EXPECTED_HF_CACHE_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_cache/scene_memory_v6_identity/"
    "huggingface/datasets"
)
REVIEWED_INITIAL_ADAPTER_SHA256 = (
    "592f8c1d47bde674c30625e3c05277025f0dfd063bcf5b693c148f60d74354e1"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
)
EXPECTED_MODEL_WEIGHT = {
    "relative_path": "model.safetensors",
    "bytes": 15992595884,
    "sha256": "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503",
}
EXPECTED_TOKENIZER_ARTIFACTS: Mapping[str, Mapping[str, object]] = {
    "tokenizer_json": {
        "relative_path": "tokenizer.json",
        "bytes": 32169626,
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    "tokenizer_config": {
        "relative_path": "tokenizer_config.json",
        "bytes": 2095,
        "sha256": "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
    },
    "chat_template": {
        "relative_path": "chat_template.jinja",
        "bytes": 17336,
        "sha256": "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    },
}
EXPECTED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "python": "3.12.13",
    "torch": "2.12.1",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
}
EXPECTED_TARGET_LAYERS = ",".join(str(index) for index in range(42))
SCENE_STATE_IDENTITY_OBJECTIVE_VERSION = "scene_state_identity_ce_v2"
SCENE_STATE_IDENTITY_BACKWARD_MODE = (
    "sequential_replayed_donor_single_zero_diagnostic_exact_first_order_v2"
)
SCENE_STATE_IDENTITY_PAIRING_VERSION = (
    "nearest_write_token_length_label_distinct_symmetric_pair_v2"
)
SCENE_STATE_IDENTITY_PAIRING_REFINEMENT = (
    "maximize_nonempty_same_cardinality_within_nearest_length_budget_v1"
)
SCENE_STATE_IDENTITY_PAIRING_LENGTH_CONTROL = (
    "nearest_feasible_symmetric_absolute_write_token_delta_v1"
)
SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE = (
    "exact_input_ids_and_attention_before_pair_target_v1"
)
SCENE_STATE_IDENTITY_TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
EXPECTED_TARGET_STRATUM_ROW_COUNTS = {
    "presence": 24,
    "same_cardinality_value": 8,
    "cross_cardinality_value": 0,
}
EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM = {"0": 12, "1": 13, "2": 6, "3": 1}
EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX = 61
EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN = 19.5625
EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL = 313
EXPECTED_CAUSAL_PREFIX_TOKEN_COUNT_HISTOGRAM = {"60": 24, "61": 8}
EXPECTED_CAUSAL_PREFIX_SHA256_SET = {
    "48eb5978215603aec962f7cad646d21ec9119109888f49eac9ffa6ed0bcdbfe5",
    "e9bbdbd92eabd5a8db6f677dcc664b7b2cac771aa0a0c0b6e61bbec923b4097d",
}
EXPECTED_IDENTITY_PAIRS_SHA256 = (
    "f4fb3b9611c5996518490588297d83099c8aaccad6ced6bea1c9dfd51e1dbbc6"
)
EXPECTED_SOURCE_PATHS: Mapping[str, str] = {
    "launcher": "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v6.sh",
    "data_contract": "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_data_contract.py",
    "launch_contract": "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_launch_contract.py",
    "run_audit": "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_run_audit.py",
    "tokenization_lock": "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_tokenized_cache_lock.json",
    "pair_builder": "experiments/rethinking_rwkv_ms_gemma/prepare_scene_failure_pairs.py",
    "trainer_entrypoint": "deltamem/train/delta_sft.py",
    "trainer_implementation": "deltamem/train/delta_sft_experimental.py",
    "delta_entrypoint": "deltamem/core/delta.py",
    "delta_implementation": "deltamem/core/delta_impl.py",
    "rwkv_ms_core": "deltamem/core/hrm_rwkv7.py",
    "backbone_compatibility": "deltamem/core/backbone_compat.py",
    "affine_scan": "deltamem/kernels/affine_scan.py",
    "chat_templates": "deltamem/chat_templates.py",
    "model_loading": "deltamem/model_loading.py",
}

STAGES: Mapping[str, Mapping[str, object]] = {
    "prepare": {
        "prepare_only": True,
        "source_partition_rows": 32,
        "optimization_updates": 0,
        "source_rows_consumed": 0,
        "max_steps": 32,
        "save_steps": 16,
        "save_total_limit": 2,
        "warmup_ratio": "0.0625",
        "warmup_steps": 2,
        "checkpoint_steps": [],
        "purpose": "seeded step-zero identity-proof adapter and provenance preparation",
    },
    "smoke": {
        "prepare_only": False,
        "source_partition_rows": 32,
        "optimization_updates": 1,
        "source_rows_consumed": 1,
        "max_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
        "warmup_ratio": "0",
        "warmup_steps": 0,
        "checkpoint_steps": [1],
        "purpose": "one disposable update validating the three-branch objective",
    },
    "proof": {
        "prepare_only": False,
        "source_partition_rows": 32,
        "optimization_updates": 32,
        "source_rows_consumed": 32,
        "max_steps": 32,
        "save_steps": 16,
        "save_total_limit": 2,
        "warmup_ratio": "0.0625",
        "warmup_steps": 2,
        "checkpoint_steps": [16, 32],
        "purpose": "single fresh pass over frozen official-train failure32 for identity proof",
    },
}


class ContractError(ValueError):
    """Raised when a launch artifact differs from the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_json_object(path: Path, *, description: str) -> dict[str, object]:
    require(path.is_file() and not path.is_symlink(), f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{description} is invalid JSON: {path}") from exc
    require(isinstance(payload, dict), f"{description} must be an object: {path}")
    return payload


def _validate_checksum(payload: Mapping[str, object], *, field: str, description: str) -> None:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    require(recorded == canonical_sha256(unsigned), f"{description} checksum differs")


def current_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "transformers": metadata.version("transformers"),
        "tokenizers": metadata.version("tokenizers"),
        "datasets": metadata.version("datasets"),
        "accelerate": metadata.version("accelerate"),
        "safetensors": metadata.version("safetensors"),
    }


def expected_run_name(run_mode: str, run_attempt: str) -> str:
    require(run_mode in STAGES, f"unsupported run mode: {run_mode}")
    require(re.fullmatch(r"run[1-9][0-9]*", run_attempt) is not None, "run attempt must match runN")
    prefix = "scene_memory_v6_identityproof_all42_qo_r4_fail32"
    if run_mode == "prepare":
        return f"{prefix}_s32_{run_attempt}_prepare"
    if run_mode == "smoke":
        return f"{prefix}_smoke1_{run_attempt}"
    return f"{prefix}_s32_{run_attempt}"


def validate_external_paths(
    *,
    run_mode: str,
    run_attempt: str,
    python_bin: Path,
    output_dir: Path,
    initial_adapter_dir: Path,
    hf_cache_dir: Path,
) -> None:
    require(
        python_bin.resolve() == EXPECTED_PYTHON_BIN.resolve(),
        "training Python differs from lock",
    )
    expected_output = EXPECTED_RUN_ROOT / expected_run_name(run_mode, run_attempt)
    require(output_dir.resolve() == expected_output, "output directory differs from proof lineage")
    require(initial_adapter_dir.resolve() == expected_output / "initial_adapter", "initial adapter directory differs")
    require(hf_cache_dir.resolve() == EXPECTED_HF_CACHE_DIR, "HF cache directory differs")


def _require_locked_regular_file(path: Path, *, description: str) -> None:
    require(
        path.is_file() and not path.is_symlink(),
        f"{description} is missing or a symlink: {path}",
    )


def _verified_immutable_file(
    path: Path,
    *,
    description: str,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> tuple[int, str]:
    _require_locked_regular_file(path, description=description)
    size = path.stat().st_size
    require(
        expected_bytes is None or size == expected_bytes,
        f"{description} byte size differs from lock",
    )
    digest = sha256_file(path)
    require(digest == expected_sha256, f"{description} content differs from lock")
    return size, digest


def build_source_lock(repo: Path) -> dict[str, object]:
    repo = repo.expanduser().resolve()
    require(repo == EXPECTED_REPO, "repository path differs from lock")
    require(repo.is_dir(), f"repository is missing: {repo}")
    require(
        Path(sys.executable).resolve() == EXPECTED_PYTHON_BIN.resolve(),
        "source-lock writer used the wrong Python",
    )
    runtime_versions = current_runtime_versions()
    require(
        runtime_versions == dict(EXPECTED_RUNTIME_VERSIONS),
        "source-lock writer runtime differs",
    )

    source_records: dict[str, object] = {}
    for label, relative_path in EXPECTED_SOURCE_PATHS.items():
        source_path = repo / relative_path
        _require_locked_regular_file(
            source_path,
            description=f"behavior source {label}",
        )
        require(
            source_path.resolve().is_relative_to(repo),
            f"behavior source escapes repository: {label}",
        )
        source_records[label] = {
            "relative_path": relative_path,
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }

    data_records: dict[str, object] = {}
    for label, (artifact_path, expected_sha256) in EXPECTED_DATA_ARTIFACTS.items():
        size, digest = _verified_immutable_file(
            artifact_path,
            description=f"data artifact {label}",
            expected_sha256=expected_sha256,
        )
        data_records[label] = {
            "path": str(artifact_path),
            "bytes": size,
            "sha256": digest,
        }

    config_path = EXPECTED_MODEL / "config.json"
    config_size, config_digest = _verified_immutable_file(
        config_path,
        description="model config",
        expected_sha256=EXPECTED_MODEL_CONFIG_SHA256,
    )
    weight_path = EXPECTED_MODEL / str(EXPECTED_MODEL_WEIGHT["relative_path"])
    _verified_immutable_file(
        weight_path,
        description="model weight",
        expected_sha256=str(EXPECTED_MODEL_WEIGHT["sha256"]),
        expected_bytes=int(EXPECTED_MODEL_WEIGHT["bytes"]),
    )
    tokenizer_records: dict[str, object] = {}
    for label, expected in EXPECTED_TOKENIZER_ARTIFACTS.items():
        artifact = EXPECTED_MODEL / str(expected["relative_path"])
        _verified_immutable_file(
            artifact,
            description=f"tokenizer artifact {label}",
            expected_sha256=str(expected["sha256"]),
            expected_bytes=int(expected["bytes"]),
        )
        tokenizer_records[label] = dict(expected)

    payload: dict[str, object] = {
        "schema": SOURCE_LOCK_SCHEMA,
        "experiment": EXPERIMENT,
        "repository": str(repo),
        "runtime_versions": runtime_versions,
        "sources": source_records,
        "data_artifacts": data_records,
        "model": {
            "path": str(EXPECTED_MODEL),
            "config": {
                "relative_path": "config.json",
                "bytes": config_size,
                "sha256": config_digest,
            },
            "weight": dict(EXPECTED_MODEL_WEIGHT),
            "tokenizer_artifacts": tokenizer_records,
        },
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    return payload


def write_json_atomic_replace(path: Path, payload: object) -> None:
    path = path.expanduser().absolute()
    require(not path.is_symlink(), f"output is a symlink: {path}")
    require(not path.exists() or path.is_file(), f"output is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_source_lock(repo: Path, path: Path) -> dict[str, object]:
    path = path.expanduser().absolute()
    require(not path.is_symlink(), f"source-lock output is a symlink: {path}")
    require(
        path == EXPECTED_SOURCE_LOCK.absolute(),
        "source-lock output path differs",
    )
    payload = build_source_lock(repo)
    write_json_atomic_replace(path, payload)
    saved = load_json_object(path, description="written source lock")
    require(saved == payload, "written source lock differs from generated payload")
    _validate_checksum(saved, field="lock_sha256", description="written source lock")
    return saved


def validate_source_lock(
    repo: Path,
    path: Path,
    *,
    verify_model_weight: bool,
    verify_runtime: bool,
) -> dict[str, object]:
    repo = repo.resolve()
    path = path.resolve()
    require(repo == EXPECTED_REPO, "repository path differs from lock")
    require(path == EXPECTED_SOURCE_LOCK, "source-lock path differs")
    lock = load_json_object(path, description="source lock")
    require(
        set(lock)
        == {
            "schema",
            "experiment",
            "repository",
            "runtime_versions",
            "sources",
            "data_artifacts",
            "model",
            "lock_sha256",
        },
        "source-lock field set differs",
    )
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "source-lock schema differs")
    require(lock.get("experiment") == EXPERIMENT, "source-lock experiment differs")
    require(lock.get("repository") == str(repo), "source-lock repository differs")
    require(lock.get("runtime_versions") == dict(EXPECTED_RUNTIME_VERSIONS), "source-lock runtime differs")
    _validate_checksum(lock, field="lock_sha256", description="source lock")

    source_records = lock.get("sources")
    require(isinstance(source_records, dict), "source-lock code records are missing")
    require(set(source_records) == set(EXPECTED_SOURCE_PATHS), "source-lock code set differs")
    for label, relative_path in EXPECTED_SOURCE_PATHS.items():
        record = source_records.get(label)
        require(isinstance(record, dict), f"source-lock record is missing: {label}")
        source_path = repo / relative_path
        _require_locked_regular_file(
            source_path,
            description=f"behavior source {label}",
        )
        require(
            record
            == {
                "relative_path": relative_path,
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            },
            f"behavior source differs from lock: {label}",
        )

    data_records = lock.get("data_artifacts")
    require(isinstance(data_records, dict), "source-lock data records are missing")
    require(
        set(data_records) == set(EXPECTED_DATA_ARTIFACTS),
        "source-lock data set differs",
    )
    for label, (artifact_path, expected_sha256) in EXPECTED_DATA_ARTIFACTS.items():
        record = data_records.get(label)
        require(isinstance(record, dict), f"source-lock data record is missing: {label}")
        _require_locked_regular_file(
            artifact_path,
            description=f"data artifact {label}",
        )
        require(
            record
            == {
                "path": str(artifact_path),
                "bytes": artifact_path.stat().st_size,
                "sha256": expected_sha256,
            }
            and expected_sha256 == sha256_file(artifact_path),
            f"data artifact differs from lock: {label}",
        )

    model = lock.get("model")
    require(
        isinstance(model, dict)
        and set(model)
        == {"path", "config", "weight", "tokenizer_artifacts"}
        and model.get("path") == str(EXPECTED_MODEL),
        "model lock differs",
    )
    config_path = EXPECTED_MODEL / "config.json"
    _require_locked_regular_file(config_path, description="model config")
    require(
        model.get("config")
        == {
            "relative_path": "config.json",
            "bytes": config_path.stat().st_size,
            "sha256": EXPECTED_MODEL_CONFIG_SHA256,
        }
        and EXPECTED_MODEL_CONFIG_SHA256 == sha256_file(config_path),
        "model config differs from lock",
    )
    weight_record = model.get("weight")
    require(isinstance(weight_record, dict) and weight_record == EXPECTED_MODEL_WEIGHT, "model weight record differs")
    weight_path = EXPECTED_MODEL / str(EXPECTED_MODEL_WEIGHT["relative_path"])
    _require_locked_regular_file(weight_path, description="model weight")
    require(
        weight_path.stat().st_size == EXPECTED_MODEL_WEIGHT["bytes"],
        "model weight byte size differs",
    )
    if verify_model_weight:
        require(
            sha256_file(weight_path) == EXPECTED_MODEL_WEIGHT["sha256"],
            "model weight content differs",
        )
    tokenizer_records = model.get("tokenizer_artifacts")
    require(
        isinstance(tokenizer_records, dict)
        and set(tokenizer_records) == set(EXPECTED_TOKENIZER_ARTIFACTS),
        "tokenizer locks differ",
    )
    for label, expected in EXPECTED_TOKENIZER_ARTIFACTS.items():
        record = tokenizer_records.get(label)
        artifact = EXPECTED_MODEL / str(expected["relative_path"])
        _require_locked_regular_file(
            artifact,
            description=f"tokenizer artifact {label}",
        )
        require(
            record == dict(expected)
            and artifact.stat().st_size == expected["bytes"]
            and sha256_file(artifact) == expected["sha256"],
            f"tokenizer artifact differs: {label}",
        )
    if verify_runtime:
        require(
            Path(sys.executable).resolve() == EXPECTED_PYTHON_BIN.resolve(),
            "source validation used the wrong Python",
        )
        require(current_runtime_versions() == dict(EXPECTED_RUNTIME_VERSIONS), "training runtime differs")
    return lock


def validate_tokenization_lock(path: Path) -> dict[str, object]:
    path = path.resolve()
    require(path == EXPECTED_TOKENIZATION_LOCK, "tokenization-lock path differs")
    lock = load_json_object(path, description="tokenization lock")
    require(lock.get("schema") == TOKENIZATION_LOCK_SCHEMA, "tokenization-lock schema differs")
    require(lock.get("experiment") == EXPERIMENT, "tokenization-lock experiment differs")
    require(lock.get("persisted_cache_enabled") is False, "identity proof forbids persisted tokenized cache reuse")
    require(lock.get("rebuild_each_fresh_run") is True, "identity proof must rebuild tokenization")
    require(lock.get("train_file") == str(EXPECTED_TRAIN), "tokenization train path differs")
    require(lock.get("train_file_sha256") == EXPECTED_TRAIN_SHA256, "tokenization train hash differs")
    require(lock.get("source_manifest_sha256") == EXPECTED_PAIR_MANIFEST_SHA256, "tokenization source manifest differs")
    require(
        lock.get("required_generated_columns")
        == [
            "scene_state_semantic_mask",
            "scene_state_boundary_count",
            "scene_state_identity_target_mask",
            "scene_state_identity_target_mask_sha256",
            "scene_state_identity_target_stratum",
            "scene_state_donor_write_input_ids",
            "scene_state_donor_write_attention_mask",
            "scene_state_donor_write_message_ids",
            "scene_state_donor_write_sentence_ids",
            "scene_state_donor_boundary_count",
            "scene_state_source_index",
            "scene_state_donor_index",
            "scene_state_source_row_sha256",
            "scene_state_donor_row_sha256",
            "scene_state_source_label_sha256",
            "scene_state_donor_label_sha256",
            "scene_state_source_write_sha256",
            "scene_state_donor_write_sha256",
        ],
        "tokenization required-column lock differs",
    )
    _validate_checksum(lock, field="lock_sha256", description="tokenization lock")
    return lock


def validate_data_manifest(path: Path) -> dict[str, object]:
    manifest = load_json_object(path, description="data-contract manifest")
    require(manifest.get("schema") == DATA_SCHEMA, "data-contract schema differs")
    require(manifest.get("experiment") == EXPERIMENT, "data-contract experiment differs")
    _validate_checksum(manifest, field="manifest_sha256", description="data-contract manifest")
    partition = manifest.get("training_partition")
    require(isinstance(partition, dict), "training partition is missing")
    require(
        partition.get("source_split") == "train"
        and partition.get("rows") == 32
        and partition.get("path") == str(EXPECTED_TRAIN)
        and partition.get("sha256") == EXPECTED_TRAIN_SHA256
        and partition.get("val_or_test_rows_emitted_for_training") == 0,
        "proof training partition differs",
    )
    hard32 = manifest.get("hard_evaluation_selection")
    require(isinstance(hard32, dict), "hard32 selection is missing")
    require(
        hard32.get("name") == "scene_v6_identity_hard32"
        and hard32.get("source_split") == "val"
        and hard32.get("rows") == 32
        and hard32.get("path") == str(EXPECTED_HARD32)
        and hard32.get("sha256") == EXPECTED_HARD32_SHA256
        and hard32.get("source_indices_file", {}).get("sha256")
        == EXPECTED_HARD32_INDICES_SHA256,
        "hard32 selection differs",
    )
    require(
        manifest.get("pair_manifest", {}).get("sha256") == EXPECTED_PAIR_MANIFEST_SHA256,
        "data-contract pair manifest differs",
    )
    selected_overlap = manifest.get("selected_slice_overlap_audit")
    selected_comparison = (
        selected_overlap.get("comparison")
        if isinstance(selected_overlap, dict)
        else None
    )
    require(
        isinstance(selected_overlap, dict)
        and selected_overlap.get("passage_disjoint") is True
        and isinstance(selected_comparison, dict)
        and selected_comparison.get("left_split") == "failure32_train"
        and selected_comparison.get("right_split") == "fixed_val32"
        and selected_comparison.get("exact_normalized_full_prompts_shared") == 0
        and selected_comparison.get("exact_normalized_paragraphs_shared") == 0
        and selected_comparison.get("left_rows_with_shared_paragraph") == 0
        and selected_comparison.get("right_rows_with_shared_paragraph") == 0,
        "selected failure32/fixed-val32 passage-overlap proof differs",
    )
    test_policy = manifest.get("test_policy")
    require(
        isinstance(test_policy, dict)
        and test_policy.get("rows_emitted_for_training") == 0
        and test_policy.get("rows_emitted_for_checkpoint_selection") == 0
        and test_policy.get("full_validation_before_hard32_pass") == "forbidden",
        "test/full-validation policy differs",
    )
    return manifest


def _identity_target_stratum(source_count: int, donor_count: int) -> str:
    if (source_count == 0) != (donor_count == 0):
        return "presence"
    if source_count == donor_count:
        return "same_cardinality_value"
    return "cross_cardinality_value"


def validate_identity_pairing_manifest(path: Path) -> dict[str, object]:
    pairing = load_json_object(path, description="identity pairing manifest")
    require(pairing.get("schema_version") == 2, "identity pairing schema differs")
    for field, expected in {
        "objective_version": SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
        "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "pairing_scope": "within_post_split_partition",
        "target_mode": "first_pair_distinguishing_semantic_token_v1",
        "causal_prefix_mode": SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "semantic_mask_mode": "top_level_boundaries_nonwhitespace_offset_overlap_v1",
        "semantic_loss_normalization": "selected_tokens_per_row_then_batch_mean_v1",
        "target_token_count": 32,
        "target_stratum_row_counts": EXPECTED_TARGET_STRATUM_ROW_COUNTS,
        "source_boundary_count_histogram": EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM,
        "write_token_count_delta_max": EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX,
        "write_token_count_delta_mean": EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN,
        "write_token_count_delta_total": EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL,
        "nearest_baseline_write_token_count_delta_max": (
            EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX
        ),
        "nearest_baseline_write_token_count_delta_total": (
            EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL
        ),
        "data_seed": 42,
    }.items():
        require(pairing.get(field) == expected, f"identity pairing differs: {field}")
    _validate_checksum(pairing, field="manifest_sha256", description="identity pairing manifest")
    splits = pairing.get("splits")
    require(isinstance(splits, dict) and set(splits) == {"train"}, "identity pairing split set differs")
    train = splits["train"]
    require(isinstance(train, dict), "identity train pairing is missing")
    for field, expected in {
        "split": "train",
        "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "target_mode": "first_pair_distinguishing_semantic_token_v1",
        "causal_prefix_mode": SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "sample_count": 32,
        "pair_count": 16,
        "target_token_count": 32,
        "target_stratum_row_counts": EXPECTED_TARGET_STRATUM_ROW_COUNTS,
        "source_boundary_count_histogram": EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM,
        "write_token_count_delta_max": EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX,
        "write_token_count_delta_mean": EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN,
        "write_token_count_delta_total": EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL,
        "nearest_baseline_write_token_count_delta_max": (
            EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX
        ),
        "nearest_baseline_write_token_count_delta_total": (
            EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL
        ),
        "pairs_sha256": EXPECTED_IDENTITY_PAIRS_SHA256,
    }.items():
        require(train.get(field) == expected, f"identity train pairing differs: {field}")
    _validate_checksum(train, field="manifest_sha256", description="identity train pairing")
    pairs = train.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 32, "identity pairing row audit differs")
    require(canonical_sha256(pairs) == EXPECTED_IDENTITY_PAIRS_SHA256, "identity pair rows differ")

    donor_map: dict[int, int] = {}
    stratum_counts = {stratum: 0 for stratum in SCENE_STATE_IDENTITY_TARGET_STRATA}
    boundary_histogram: dict[str, int] = {}
    deltas: list[int] = []
    causal_prefix_histogram: dict[str, int] = {}
    causal_prefix_hashes: set[str] = set()
    for expected_source, row in enumerate(pairs):
        require(isinstance(row, dict), "identity pair row is invalid")
        source = row.get("source_index")
        donor = row.get("donor_index")
        require(
            source == expected_source
            and isinstance(donor, int)
            and not isinstance(donor, bool)
            and 0 <= donor < len(pairs)
            and donor != source,
            "identity donor indices differ",
        )
        require(row.get("source_label_sha256") != row.get("donor_label_sha256"), "identity donor label is not distinct")
        require(row.get("target_mode") == pairing.get("target_mode"), "identity target mode differs")
        require(
            row.get("causal_prefix_mode") == SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
            "identity causal-prefix mode differs",
        )
        require(row.get("target_span_tokens") == 1, "identity target must contain one token")
        target_positions = row.get("target_label_positions")
        donor_target_positions = row.get("donor_target_label_positions")
        target_predictors = row.get("target_predictor_positions")
        donor_target_predictors = row.get("donor_target_predictor_positions")
        target_tokens = row.get("target_token_ids")
        donor_tokens = row.get("donor_token_ids")
        prefix_count = row.get("causal_prefix_token_count")
        prefix_sha = row.get("causal_prefix_sha256")
        require(
            isinstance(prefix_count, int)
            and not isinstance(prefix_count, bool)
            and prefix_count > 0
            and target_positions == [prefix_count]
            and donor_target_positions == [prefix_count]
            and target_predictors == [prefix_count - 1]
            and donor_target_predictors == [prefix_count - 1]
            and isinstance(target_tokens, list)
            and len(target_tokens) == 1
            and isinstance(donor_tokens, list)
            and len(donor_tokens) == 1
            and target_tokens != donor_tokens
            and isinstance(prefix_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", prefix_sha) is not None,
            "identity causal-prefix audit differs",
        )
        prefix_key = str(prefix_count)
        causal_prefix_histogram[prefix_key] = (
            causal_prefix_histogram.get(prefix_key, 0) + 1
        )
        causal_prefix_hashes.add(prefix_sha)
        source_count = row.get("source_boundary_count")
        donor_count = row.get("donor_boundary_count")
        require(
            isinstance(source_count, int)
            and not isinstance(source_count, bool)
            and source_count >= 0
            and isinstance(donor_count, int)
            and not isinstance(donor_count, bool)
            and donor_count >= 0,
            "identity boundary cardinality differs",
        )
        stratum = _identity_target_stratum(source_count, donor_count)
        require(row.get("target_stratum") == stratum, "identity target stratum differs")
        stratum_counts[stratum] += 1
        boundary_key = str(source_count)
        boundary_histogram[boundary_key] = boundary_histogram.get(boundary_key, 0) + 1
        source_tokens = row.get("source_write_token_count")
        donor_tokens = row.get("donor_write_token_count")
        delta = row.get("write_token_count_delta")
        require(
            isinstance(source_tokens, int)
            and not isinstance(source_tokens, bool)
            and source_tokens > 0
            and isinstance(donor_tokens, int)
            and not isinstance(donor_tokens, bool)
            and donor_tokens > 0
            and delta == abs(source_tokens - donor_tokens),
            "identity write-length delta differs",
        )
        deltas.append(delta)
        donor_map[source] = donor

    require(all(donor_map.get(donor) == source for source, donor in donor_map.items()), "identity donor map is not symmetric")
    for source, donor in donor_map.items():
        row = pairs[source]
        reverse = pairs[donor]
        for left, right in (
            ("source_row_sha256", "donor_row_sha256"),
            ("source_label_sha256", "donor_label_sha256"),
            ("source_write_sha256", "donor_write_sha256"),
            ("source_write_token_count", "donor_write_token_count"),
            ("source_boundary_count", "donor_boundary_count"),
        ):
            require(row.get(left) == reverse.get(right), f"identity symmetric pair differs: {left}")
        require(row.get("write_token_count_delta") == reverse.get("write_token_count_delta"), "identity symmetric write-length delta differs")
        for left, right in (
            ("target_label_positions", "donor_target_label_positions"),
            ("target_predictor_positions", "donor_target_predictor_positions"),
            ("target_token_ids", "donor_token_ids"),
        ):
            require(row.get(left) == reverse.get(right), f"identity symmetric target differs: {left}")
        require(
            row.get("first_differing_semantic_ordinal")
            == reverse.get("first_differing_semantic_ordinal")
            and row.get("causal_prefix_token_count")
            == reverse.get("causal_prefix_token_count")
            and row.get("causal_prefix_sha256")
            == reverse.get("causal_prefix_sha256"),
            "identity symmetric causal-prefix audit differs",
        )
    require(stratum_counts == EXPECTED_TARGET_STRATUM_ROW_COUNTS, "identity derived target strata differ")
    require(boundary_histogram == EXPECTED_SOURCE_BOUNDARY_COUNT_HISTOGRAM, "identity derived boundary histogram differs")
    require(
        causal_prefix_histogram == EXPECTED_CAUSAL_PREFIX_TOKEN_COUNT_HISTOGRAM
        and causal_prefix_hashes == EXPECTED_CAUSAL_PREFIX_SHA256_SET,
        "identity derived causal-prefix audit differs",
    )
    require(max(deltas) == EXPECTED_WRITE_TOKEN_COUNT_DELTA_MAX, "identity derived max write-length delta differs")
    require(
        sum(deltas) == 2 * EXPECTED_WRITE_TOKEN_COUNT_DELTA_TOTAL,
        "identity derived total write-length delta differs",
    )
    require(
        math.isclose(sum(deltas) / len(deltas), EXPECTED_WRITE_TOKEN_COUNT_DELTA_MEAN),
        "identity derived mean write-length delta differs",
    )
    return pairing


def _artifact_record_valid(
    record: object,
    expected_path: Path,
    *,
    sha_field: str = "file_sha256",
) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("path") == str(expected_path)
        and expected_path.is_file()
        and not expected_path.is_symlink()
        and expected_path.stat().st_size > 0
        and record.get(sha_field) == sha256_file(expected_path)
    )


def validate_prepare_authorization(path: Path) -> dict[str, object]:
    path = path.resolve()
    require(path.name == "prepare_receipt.json", "prepare receipt filename differs")
    require(path.parent.parent == EXPECTED_RUN_ROOT, "prepare receipt root differs")
    require(
        re.fullmatch(
            r"scene_memory_v6_identityproof_all42_qo_r4_fail32_s32_run[1-9][0-9]*_prepare",
            path.parent.name,
        ) is not None,
        "prepare receipt run name differs",
    )
    receipt = load_json_object(path, description="prepare receipt")
    _validate_checksum(receipt, field="receipt_sha256", description="prepare receipt")
    for field, expected in {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "experiment": EXPERIMENT,
        "fresh_run": True,
        "global_step": 0,
        "training_arguments_created": False,
        "trainer_created": False,
        "optimizer_created": False,
        "training_started": False,
    }.items():
        require(receipt.get(field) == expected, f"prepare receipt differs: {field}")
    root = path.parent
    initial = root / "initial_adapter"
    for field, artifact, sha_field in (
        ("initial_adapter_manifest", initial / "initial_adapter_manifest.json", "file_sha256"),
        ("adapter", initial / "delta_mem_adapter.pt", "file_sha256"),
        ("config", initial / "delta_mem_config.json", "file_sha256"),
        ("training_protocol", initial / "training_protocol.json", "file_sha256"),
        ("data_contract_manifest", root / "data_contract_manifest.json", "file_sha256"),
        ("launch_manifest", root / "launch_manifest.json", "file_sha256"),
        ("identity_pairing_manifest", root / "scene_state_identity_pairing_manifest.json", "file_sha256"),
    ):
        require(_artifact_record_valid(receipt.get(field), artifact, sha_field=sha_field), f"prepare artifact differs: {field}")
    require(
        receipt.get("adapter", {}).get("file_sha256")
        == REVIEWED_INITIAL_ADAPTER_SHA256,
        "seeded adapter differs",
    )
    require(receipt.get("source_lock", {}).get("file_sha256") == sha256_file(EXPECTED_SOURCE_LOCK), "prepare source lock differs")
    require(
        receipt.get("tokenization_lock", {}).get("file_sha256")
        == sha256_file(EXPECTED_TOKENIZATION_LOCK),
        "prepare tokenization lock differs",
    )
    validate_identity_pairing_manifest(
        root / "scene_state_identity_pairing_manifest.json"
    )
    require(not (root / "trainer").exists(), "prepare authorization contains a Trainer directory")
    prohibited = {"optimizer.pt", "scheduler.pt", "trainer_state.json"}
    require(not any(item.name in prohibited for item in root.rglob("*")), "prepare authorization contains training state")
    return receipt


def validate_smoke_authorization(path: Path) -> dict[str, object]:
    path = path.resolve()
    require(path.name == "run_audit_receipt.json", "smoke receipt filename differs")
    require(path.parent.parent == EXPECTED_RUN_ROOT, "smoke receipt root differs")
    require(
        re.fullmatch(
            r"scene_memory_v6_identityproof_all42_qo_r4_fail32_smoke1_run[1-9][0-9]*",
            path.parent.name,
        ) is not None,
        "smoke receipt run name differs",
    )
    from experiments.rethinking_rwkv_ms_gemma.scene_memory_v6_run_audit import (
        validate_existing_run_receipt,
    )

    receipt = validate_existing_run_receipt(
        path,
        expected_run_mode="smoke",
        expected_run_root=path.parent,
    )
    checkpoints = receipt.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 1, "smoke checkpoint evidence differs")
    checkpoint = checkpoints[0]
    require(isinstance(checkpoint, dict) and checkpoint.get("step") == 1, "smoke checkpoint step differs")
    history = checkpoint.get("history")
    require(
        isinstance(history, dict)
        and history.get("records") == 1
        and history.get("identity_metrics_finite") is True,
        "smoke identity-objective evidence differs",
    )
    change = checkpoint.get("adapter_change")
    require(
        isinstance(change, dict)
        and int(change.get("changed_trainable_tensor_count", 0)) > 0
        and change.get("changed_nontrainable_tensor_count") == 0
        and change.get("inactive_kv_projection_tensors_unchanged") == 168
        and change.get("inactive_kv_delta_scale_entries_unchanged") == 84,
        "smoke adapter/frozen evidence differs",
    )
    coverage = change.get("required_changed_layer_coverage")
    require(
        isinstance(coverage, dict)
        and len(coverage) == 22
        and all(value == 42 for value in coverage.values()),
        "smoke all-layer update coverage differs",
    )
    return receipt


def _option_values(command: Sequence[str]) -> tuple[dict[str, str], set[str]]:
    values: dict[str, str] = {}
    switches: set[str] = set()
    index = 3
    while index < len(command):
        flag = command[index]
        require(flag.startswith("--"), f"unexpected positional trainer argument: {flag}")
        require(flag not in values and flag not in switches, f"duplicate trainer option: {flag}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            values[flag] = command[index + 1]
            index += 2
        else:
            switches.add(flag)
            index += 1
    return values, switches


def validate_command(
    command: Sequence[str],
    *,
    python_bin: Path,
    run_mode: str,
    output_dir: Path,
    initial_adapter_dir: Path,
    hf_cache_dir: Path,
) -> None:
    require(run_mode in STAGES, f"unsupported run mode: {run_mode}")
    require(list(command[:3]) == [str(python_bin), "-m", "deltamem.train.delta_sft"], "trainer prefix differs")
    values, switches = _option_values(command)
    stage = STAGES[run_mode]
    required_values = {
        "--model-path": str(EXPECTED_MODEL),
        "--train-file": str(EXPECTED_TRAIN),
        "--output-dir": str(output_dir),
        "--initial-adapter-output-dir": str(initial_adapter_dir),
        "--hf-cache-dir": str(hf_cache_dir),
        "--device": "cuda:0",
        "--dtype": "bfloat16",
        "--attn-implementation": "sdpa",
        "--memory-backend": "rwkv_ms",
        "--rwkv-ms-num-states": "4",
        "--rwkv-ms-chunk-size": "128",
        "--rwkv-ms-boundary-mode": "fixed_chunk",
        "--rwkv-ms-erase-gate": "1.0",
        "--rwkv-ms-read-top-k": "0",
        "--rwkv-ms-output-init-scale": "0.02",
        "--rwkv-ms-semantics-version": "2",
        "--rank": "4",
        "--alpha": "8",
        "--num-state-heads": "1",
        "--beta-bias-init": "0.0",
        "--state-update-mode": "standard",
        "--output-init": "base_slice_fixed",
        "--base-slice-ref-width": "8",
        "--delta-heads": "q,o",
        "--memory-fusion-mode": "add",
        "--memory-fusion-placement": "attention_output",
        "--delta-scale-init": "0.1",
        "--delta-scale-max": "0.5",
        "--delta-scale-granularity": "head",
        "--delta-scale-parameterization": "alpha_over_rank",
        "--online-gain": "0.2",
        "--target-layers": EXPECTED_TARGET_LAYERS,
        "--memory-readout-mode": "delta",
        "--memory-write-source": "learned_hidden",
        "--memory-write-granularity": "token",
        "--training-mode": "episode",
        "--assistant-loss-mode": "final_assistant_only",
        "--episode-recent-messages": "0",
        "--max-length": "256",
        "--max-write-length": "1280",
        "--memory-loss-mode": "scene_state_identity_ce",
        "--scene-state-identity-margin": "0.5",
        "--scene-state-source-manifest": str(EXPECTED_PAIR_MANIFEST),
        "--expected-scene-state-source-manifest-sha256": EXPECTED_PAIR_MANIFEST_SHA256,
        "--scene-boundary-payload-ce-weight": "0",
        "--memory-base-kl-weight": "0",
        "--memory-contrast-weight": "0",
        "--memory-representation-weight": "0",
        "--memory-kl-weight": "0",
        "--memory-causal-weight": "0",
        "--memory-anchor-weight": "0",
        "--memory-recover-weight": "0",
        "--memory-dropout-no-memory-prob": "0",
        "--memory-dropout-state-only-prob": "0",
        "--write-sparsity-weight": "0",
        "--learning-rate": "5e-4",
        "--lr-scheduler-type": "constant_with_warmup",
        "--weight-decay": "0",
        "--max-steps": str(stage["max_steps"]),
        "--save-steps": str(stage["save_steps"]),
        "--save-total-limit": str(stage["save_total_limit"]),
        "--warmup-ratio": str(stage["warmup_ratio"]),
        "--validation-split-ratio": "0",
        "--seed": "42",
        "--data-seed": "42",
        "--train-sampler-seed": "42",
        "--per-device-train-batch-size": "1",
        "--gradient-accumulation-steps": "1",
        "--dataloader-num-workers": "0",
    }
    for flag, expected in required_values.items():
        require(values.get(flag) == expected, f"locked trainer option differs: {flag}")
    for forbidden in (
        "--resume-from-checkpoint",
        "--resume-mode",
        "--warm-start-from-checkpoint",
        "--warm-start-mode",
        "--tokenized-cache",
        "--tokenized-dataset-dir",
        "--tokenized-dataset-root",
        "--expected-tokenized-dataset-sha256",
    ):
        require(forbidden not in values and forbidden not in switches, f"identity proof forbids {forbidden}")
    required_switches = {
        "--bf16",
        "--couple-lambda",
        "--trainable-delta-scale",
        "--no-episode-read-write-enabled",
        "--no-load-best-model-at-end",
        "--frozen-mlp-activation-checkpointing",
        "--tf32",
        "--rankwise-gates",
        "--no-delta-o-rmsnorm",
        "--no-tokenized-cache",
        "--log-delta-debug-stats",
    }
    require(required_switches <= switches, "trainer command omits a required switch")
    require(("--prepare-only" in switches) is bool(stage["prepare_only"]), "prepare-only switch differs")


def write_json_exclusive(path: Path, payload: object) -> None:
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_launch_manifest(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    require(args.launch_manifest.resolve() == output_dir / "launch_manifest.json", "launch path differs")
    validate_external_paths(
        run_mode=args.run_mode,
        run_attempt=args.run_attempt,
        python_bin=args.python_bin,
        output_dir=output_dir,
        initial_adapter_dir=args.initial_adapter_dir,
        hf_cache_dir=args.hf_cache_dir,
    )
    source_lock = validate_source_lock(
        args.repo,
        args.source_lock,
        verify_model_weight=not args.skip_model_weight_hash,
        verify_runtime=True,
    )
    tokenization_lock = validate_tokenization_lock(args.tokenization_lock)
    data_manifest = validate_data_manifest(args.data_manifest)
    prepare = validate_prepare_authorization(args.prepare_receipt) if args.run_mode != "prepare" else None
    smoke = validate_smoke_authorization(args.smoke_receipt) if args.run_mode == "proof" else None
    validate_command(
        args.command,
        python_bin=args.python_bin,
        run_mode=args.run_mode,
        output_dir=output_dir,
        initial_adapter_dir=args.initial_adapter_dir.resolve(),
        hf_cache_dir=args.hf_cache_dir.resolve(),
    )
    payload: dict[str, object] = {
        "schema": LAUNCH_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "run_mode": args.run_mode,
        "run_attempt": args.run_attempt,
        "fresh_run": True,
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": None,
        "stage": dict(STAGES[args.run_mode]),
        "paths": {
            "repository": str(args.repo.resolve()),
            "model": str(EXPECTED_MODEL),
            "train_file": str(EXPECTED_TRAIN),
            "pair_manifest": str(EXPECTED_PAIR_MANIFEST),
            "output_dir": str(output_dir),
            "initial_adapter_dir": str(args.initial_adapter_dir.resolve()),
            "hf_cache_dir": str(args.hf_cache_dir.resolve()),
        },
        "topology": {
            "model_layers": 42,
            "target_layers": list(range(42)),
            "delta_heads": ["q", "o"],
            "rank": 4,
            "alpha": 8,
            "memory_backend": "rwkv_ms",
            "rwkv_ms_semantics_version": 2,
            "fusion": "direct_add_at_attention_output",
        },
        "objective": {
            "memory_loss_mode": "scene_state_identity_ce",
            "objective_version": SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
            "margin": 0.5,
            "margin_mode": "per_row_hinge_relu_v1",
            "objective_formula": (
                "full_correct_ce + correct_all_semantic_ce + "
                "mean(relu(margin - (donor_pair_semantic_ce - "
                "correct_pair_semantic_ce)))"
            ),
            "full_correct_ce_weight": 1.0,
            "correct_all_semantic_ce_weight": 1.0,
            "donor_margin_weight": 1.0,
            "zero_diagnostic_weight": 0.0,
            "zero_diagnostic_gradient": False,
            "read_time_positions_observable": False,
            "correct_all_semantic_scope": "all_semantic_tokens_v1",
            "pair_semantic_scope": "first_pair_distinguishing_semantic_token_v1",
            "donor_margin_scope": "first_pair_distinguishing_semantic_token_v1",
            "zero_diagnostic_scope": "all_semantic_tokens_v1",
            "source_manifest_path": str(EXPECTED_PAIR_MANIFEST),
            "source_manifest_sha256": EXPECTED_PAIR_MANIFEST_SHA256,
            "backward_mode": SCENE_STATE_IDENTITY_BACKWARD_MODE,
            "read_protocol": "state_only_same_read_correct_donor_zero_adapter_active_v1",
            "zero_protocol": "adapter_active_reset_state_writes_disabled_v1",
            "semantic_mask_mode": "top_level_boundaries_nonwhitespace_offset_overlap_v1",
            "semantic_loss_normalization": "selected_tokens_per_row_then_batch_mean_v1",
            "target_mode": "first_pair_distinguishing_semantic_token_v1",
            "causal_prefix_mode": SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
            "target_strata": list(SCENE_STATE_IDENTITY_TARGET_STRATA),
            "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
            "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
            "pairing_length_control": SCENE_STATE_IDENTITY_PAIRING_LENGTH_CONTROL,
            "pairing_limitation": (
                "nonzero_write_token_length_deltas_retained_without_truncation_"
                "or_hybrid_carriers_v1"
            ),
            "auxiliary_regularizers": "all_zero",
        },
        "data_contract": {
            "path": str(args.data_manifest.resolve()),
            "file_sha256": sha256_file(args.data_manifest),
            "manifest_sha256": data_manifest["manifest_sha256"],
            "pair_manifest": data_manifest["pair_manifest"],
            "training_partition": data_manifest["training_partition"],
            "hard_evaluation_selection": data_manifest["hard_evaluation_selection"],
            "test_policy": data_manifest["test_policy"],
            "overlap_audit": data_manifest["overlap_audit"],
            "selected_slice_overlap_audit": data_manifest[
                "selected_slice_overlap_audit"
            ],
        },
        "evaluation_policy": {
            "checkpoint_receipt_filename": "checkpoint_receipt.json",
            "checkpoint_receipt_required_at_steps": list(STAGES[args.run_mode]["checkpoint_steps"]),
            "checkpoint16_independent_of_final_training_summary": args.run_mode == "proof",
            "hard32_required_before_full170": True,
            "test_untouched_until_validation_authorization": True,
        },
        "source_lock": {
            "path": str(args.source_lock.resolve()),
            "file_sha256": sha256_file(args.source_lock),
            "payload_sha256": canonical_sha256(source_lock),
            "payload": source_lock,
            "model_weight_content_rehashed": not args.skip_model_weight_hash,
        },
        "tokenization_lock": {
            "path": str(args.tokenization_lock.resolve()),
            "file_sha256": sha256_file(args.tokenization_lock),
            "payload_sha256": canonical_sha256(tokenization_lock),
            "payload": tokenization_lock,
        },
        "prepare_authorization": None if prepare is None else {
            "receipt_path": str(args.prepare_receipt.resolve()),
            "receipt_file_sha256": sha256_file(args.prepare_receipt),
            "receipt_payload_sha256": canonical_sha256(prepare),
            "validated_before_training": True,
        },
        "smoke_authorization": None if smoke is None else {
            "receipt_path": str(args.smoke_receipt.resolve()),
            "receipt_file_sha256": sha256_file(args.smoke_receipt),
            "receipt_payload_sha256": canonical_sha256(smoke),
            "validated_before_proof": True,
        },
        "runtime_versions": current_runtime_versions(),
        "command": {
            "argv": list(args.command),
            "argv_sha256": canonical_sha256(list(args.command)),
            "shell": shlex.join(args.command),
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json_exclusive(args.launch_manifest, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    source = subparsers.add_parser("validate-source-lock")
    source.add_argument("--repo", type=Path, required=True)
    source.add_argument("--source-lock", type=Path, required=True)
    source.add_argument("--skip-model-weight-hash", action="store_true")
    source_writer = subparsers.add_parser("write-source-lock")
    source_writer.add_argument("--repo", type=Path, required=True)
    source_writer.add_argument("--source-lock", type=Path, required=True)
    tokenization = subparsers.add_parser("validate-tokenization-lock")
    tokenization.add_argument("--tokenization-lock", type=Path, required=True)
    prepare = subparsers.add_parser("validate-prepare-authorization")
    prepare.add_argument("--prepare-receipt", type=Path, required=True)
    smoke = subparsers.add_parser("validate-smoke-authorization")
    smoke.add_argument("--smoke-receipt", type=Path, required=True)
    write = subparsers.add_parser("write-launch-manifest")
    write.add_argument("--repo", type=Path, required=True)
    write.add_argument("--source-lock", type=Path, required=True)
    write.add_argument("--tokenization-lock", type=Path, required=True)
    write.add_argument("--prepare-receipt", type=Path, required=True)
    write.add_argument("--smoke-receipt", type=Path, required=True)
    write.add_argument("--data-manifest", type=Path, required=True)
    write.add_argument("--launch-manifest", type=Path, required=True)
    write.add_argument("--run-mode", choices=tuple(STAGES), required=True)
    write.add_argument("--run-attempt", required=True)
    write.add_argument("--python-bin", type=Path, required=True)
    write.add_argument("--output-dir", type=Path, required=True)
    write.add_argument("--initial-adapter-dir", type=Path, required=True)
    write.add_argument("--hf-cache-dir", type=Path, required=True)
    write.add_argument("--skip-model-weight-hash", action="store_true")
    write.add_argument("command", nargs=argparse.REMAINDER)
    validate = subparsers.add_parser("validate-command")
    validate.add_argument("--run-mode", choices=tuple(STAGES), required=True)
    validate.add_argument("--python-bin", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--initial-adapter-dir", type=Path, required=True)
    validate.add_argument("--hf-cache-dir", type=Path, required=True)
    validate.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "write-source-lock":
            payload = write_source_lock(args.repo, args.source_lock)
            print(
                f"source_lock={args.source_lock} "
                f"lock_sha256={payload['lock_sha256']}"
            )
        elif args.action == "validate-source-lock":
            validate_source_lock(
                args.repo,
                args.source_lock,
                verify_model_weight=not args.skip_model_weight_hash,
                verify_runtime=True,
            )
            print("source_lock=valid")
        elif args.action == "validate-tokenization-lock":
            validate_tokenization_lock(args.tokenization_lock)
            print("tokenization_lock=valid")
        elif args.action == "validate-prepare-authorization":
            validate_prepare_authorization(args.prepare_receipt)
            print("prepare_authorization=valid")
        elif args.action == "validate-smoke-authorization":
            validate_smoke_authorization(args.smoke_receipt)
            print("smoke_authorization=valid")
        elif args.action == "validate-command":
            require(bool(args.command), "trainer command is missing")
            validate_command(
                args.command,
                python_bin=args.python_bin,
                run_mode=args.run_mode,
                output_dir=args.output_dir,
                initial_adapter_dir=args.initial_adapter_dir,
                hf_cache_dir=args.hf_cache_dir,
            )
            print("trainer_command=valid")
        else:
            require(bool(args.command), "trainer command is missing")
            payload = write_launch_manifest(args)
            print(f"launch_manifest={args.launch_manifest} manifest_sha256={payload['manifest_sha256']}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
