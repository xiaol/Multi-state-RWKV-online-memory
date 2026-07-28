#!/usr/bin/env python3
"""Audit scene-memory V6 identity-proof preparation, checkpoints, and runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT = "scene_memory_v6_identity_proof"
LAUNCH_SCHEMA = "rwkv_ms_scene_v6_identity_launch.v1"
DATA_SCHEMA = "rwkv_ms_scene_v6_identity_data.v1"
SOURCE_LOCK_SCHEMA = "rwkv_ms_scene_v6_identity_source_lock.v1"
TOKENIZATION_LOCK_SCHEMA = "rwkv_ms_scene_v6_identity_tokenization_lock.v1"
PREPARE_RECEIPT_SCHEMA = "rwkv_ms_scene_v6_identity_prepare.v1"
CHECKPOINT_RECEIPT_SCHEMA = "rwkv_ms_scene_v6_identity_checkpoint.v1"
RUN_RECEIPT_SCHEMA = "rwkv_ms_scene_v6_identity_run.v1"
INITIAL_ADAPTER_SCHEMA = "deltamem.seeded_initial_adapter.v1"

EXPECTED_REPO = Path("/home/xiaol/X/Multi-state-RWKV-online-memory")
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
EXPECTED_TRAIN_MANIFEST = EXPECTED_PAIR_ROOT / "train_manifest.jsonl"
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "d112056a80b9dc13728b021646c0fbe3da5c3c41641fb28bb8c5448b1f8427fa"
)
EXPECTED_HARD32 = EXPECTED_PAIR_ROOT / "holdout.jsonl"
EXPECTED_HARD32_SHA256 = (
    "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
)
EXPECTED_HARD32_MANIFEST = EXPECTED_PAIR_ROOT / "holdout_manifest.jsonl"
EXPECTED_HARD32_MANIFEST_SHA256 = (
    "6802d992805164342ea4ed16b9113814ee472ad363aa76eaf5298147e7a0d1cc"
)
EXPECTED_HARD32_INDICES = EXPECTED_PAIR_ROOT / "holdout_source_indices.json"
EXPECTED_HARD32_INDICES_SHA256 = (
    "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
)
EXPECTED_HARD32_SOURCE_INDICES = (
    3, 6, 16, 21, 24, 30, 33, 47, 50, 56, 59, 63, 64, 66, 67, 70,
    71, 74, 75, 79, 87, 88, 102, 112, 113, 128, 132, 141, 144, 151,
    159, 166,
)
REVIEWED_INITIAL_ADAPTER_SHA256 = (
    "592f8c1d47bde674c30625e3c05277025f0dfd063bcf5b693c148f60d74354e1"
)
EXPECTED_TARGET_LAYERS = list(range(42))
EXPECTED_DELTA_HEADS = ["q", "o"]
AUDITOR_RELATIVE_PATH = (
    "experiments/rethinking_rwkv_ms_gemma/scene_memory_v6_run_audit.py"
)
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

OBJECTIVE_PROTOCOL: Mapping[str, object] = {
    "memory_loss_mode": "scene_state_identity_ce",
    "objective_version": SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
    "margin": 0.5,
    "margin_mode": "per_row_hinge_relu_v1",
    "objective_formula": (
        "full_correct_ce + correct_all_semantic_ce + "
        "mean(relu(margin - (donor_pair_semantic_ce - "
        "correct_pair_semantic_ce)))"
    ),
    "backward_mode": SCENE_STATE_IDENTITY_BACKWARD_MODE,
    "read_protocol": "state_only_same_read_correct_donor_zero_adapter_active_v1",
    "zero_protocol": "adapter_active_reset_state_writes_disabled_v1",
    "semantic_mask_mode": "top_level_boundaries_nonwhitespace_offset_overlap_v1",
    "semantic_loss_normalization": "selected_tokens_per_row_then_batch_mean_v1",
    "target_mode": "first_pair_distinguishing_semantic_token_v1",
    "causal_prefix_mode": SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
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
    "target_strata": list(SCENE_STATE_IDENTITY_TARGET_STRATA),
    "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
    "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
    "pairing_length_control": SCENE_STATE_IDENTITY_PAIRING_LENGTH_CONTROL,
    "pairing_limitation": (
        "nonzero_write_token_length_deltas_retained_without_truncation_"
        "or_hybrid_carriers_v1"
    ),
    "auxiliary_regularizers": "all_zero",
}

IDENTITY_METRICS = (
    "delta/scene_state_full_correct_ce",
    "delta/scene_state_correct_all_semantic_ce",
    "delta/scene_state_correct_pair_semantic_ce",
    "delta/scene_state_donor_pair_semantic_ce",
    "delta/scene_state_zero_all_semantic_ce",
    "delta/scene_state_donor_pair_gap",
    "delta/scene_state_zero_all_gap",
    "delta/scene_state_donor_margin_loss",
    "delta/scene_state_donor_positive_fraction",
    "delta/scene_state_zero_positive_fraction",
    "delta/scene_state_semantic_token_count",
    "delta/scene_state_semantic_row_count",
    "delta/scene_state_target_presence_row_count",
    "delta/scene_state_target_same_cardinality_value_row_count",
    "delta/scene_state_target_cross_cardinality_value_row_count",
)
REQUIRED_STEP_METRICS = ("loss", "grad_norm", "learning_rate", *IDENTITY_METRICS)

EXPECTED_ADAPTER_SUFFIXES = (
    "delta_scale_raw",
    "memory_q_proj",
    "memory_k_proj",
    "memory_v_proj",
    "delta_q_proj",
    "delta_k_proj",
    "delta_v_proj",
    "delta_o_proj",
    "beta_proj",
    "beta_bias",
    "hrm_rwkv7_core.x_r",
    "hrm_rwkv7_core.x_w",
    "hrm_rwkv7_core.x_k",
    "hrm_rwkv7_core.x_v",
    "hrm_rwkv7_core.x_a",
    "hrm_rwkv7_core.x_g",
    "hrm_rwkv7_core.w1",
    "hrm_rwkv7_core.w2",
    "hrm_rwkv7_core.w0",
    "hrm_rwkv7_core.a1",
    "hrm_rwkv7_core.a2",
    "hrm_rwkv7_core.a0",
    "hrm_rwkv7_core.g1",
    "hrm_rwkv7_core.g2",
    "hrm_rwkv7_core.k_k",
    "hrm_rwkv7_core.k_a",
    "hrm_rwkv7_core.receptance.weight",
    "hrm_rwkv7_core.key.weight",
    "hrm_rwkv7_core.value.weight",
    "hrm_rwkv7_core.output.weight",
    "hrm_rwkv7_core.ln_x.weight",
    "hrm_rwkv7_core.ln_x.bias",
)
EXPECTED_FROZEN_SUFFIXES = (
    "memory_q_proj",
    "memory_k_proj",
    "delta_k_proj",
    "delta_v_proj",
    "hrm_rwkv7_core.ln_x.bias",
)
EXPECTED_TRAINABLE_SUFFIXES = tuple(
    suffix for suffix in EXPECTED_ADAPTER_SUFFIXES if suffix not in EXPECTED_FROZEN_SUFFIXES
)
REQUIRED_CHANGED_SUFFIXES = (
    "delta_scale_raw",
    "memory_v_proj",
    "delta_q_proj",
    "delta_o_proj",
    "beta_proj",
    "beta_bias",
    "hrm_rwkv7_core.x_r",
    "hrm_rwkv7_core.x_k",
    "hrm_rwkv7_core.x_v",
    "hrm_rwkv7_core.w1",
    "hrm_rwkv7_core.w0",
    "hrm_rwkv7_core.a1",
    "hrm_rwkv7_core.a0",
    "hrm_rwkv7_core.g1",
    "hrm_rwkv7_core.g2",
    "hrm_rwkv7_core.k_k",
    "hrm_rwkv7_core.k_a",
    "hrm_rwkv7_core.receptance.weight",
    "hrm_rwkv7_core.key.weight",
    "hrm_rwkv7_core.value.weight",
    "hrm_rwkv7_core.output.weight",
    "hrm_rwkv7_core.ln_x.weight",
)
INACTIVE_KV_SUFFIXES = (
    "memory_q_proj",
    "memory_k_proj",
    "delta_k_proj",
    "delta_v_proj",
)


@dataclass(frozen=True)
class RunSpec:
    max_steps: int
    save_steps: int
    save_total_limit: int
    warmup_steps: int
    optimization_updates: int
    checkpoint_steps: tuple[int, ...]


RUN_SPECS: Mapping[str, RunSpec] = {
    "smoke": RunSpec(1, 1, 1, 0, 1, (1,)),
    "proof": RunSpec(32, 16, 2, 2, 32, (16, 32)),
}
PREPARE_SPEC = RunSpec(32, 16, 2, 2, 0, ())

REQUIRED_CHECKPOINT_FILES = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "training_protocol.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
)


class AuditError(ValueError):
    """Raised when evidence differs from the identity-proof contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


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
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant is forbidden: {value}")


def require_regular_file(path: Path, *, description: str) -> None:
    require(path.exists(), f"{description} is missing: {path}")
    require(path.is_file(), f"{description} is not a file: {path}")
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    require(path.stat().st_size > 0, f"{description} is empty: {path}")


def load_json_object(path: Path, *, description: str) -> dict[str, object]:
    require_regular_file(path, description=description)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid {description}: {path}") from exc
    require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def validate_checksum(
    payload: Mapping[str, object],
    *,
    field: str,
    description: str,
) -> str:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    require(
        isinstance(recorded, str) and recorded == canonical_sha256(unsigned),
        f"{description} checksum differs",
    )
    return recorded


def file_record(path: Path, *, json_payload: object | None = None) -> dict[str, object]:
    path = path.expanduser().resolve()
    require_regular_file(path, description="audited artifact")
    record: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }
    if json_payload is not None:
        record["payload_sha256"] = canonical_sha256(json_payload)
    return record


def _validate_file_record(record: object, *, description: str) -> None:
    require(isinstance(record, dict), f"{description} record is invalid")
    raw_path = record.get("path")
    require(isinstance(raw_path, str), f"{description} path is invalid")
    path = Path(raw_path)
    require(path.is_absolute() and str(path.resolve()) == raw_path, f"{description} path is not canonical")
    require_regular_file(path, description=description)
    require(record.get("bytes") == path.stat().st_size, f"{description} byte count differs")
    require(record.get("file_sha256") == sha256_file(path), f"{description} SHA differs")
    if "payload_sha256" in record:
        payload = load_json_object(path, description=description)
        require(
            record.get("payload_sha256") == canonical_sha256(payload),
            f"{description} payload SHA differs",
        )


def _require_finite_number(value: object, *, description: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{description} must be finite")
    return result


def _validate_finite_tree(value: object, *, description: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all().item()), f"{description} contains non-finite values")
        return
    if value is None or isinstance(value, (bool, str, bytes)):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"{description} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, description=f"{description}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tree(item, description=f"{description}[{index}]")
        return
    raise AuditError(f"{description} contains unsupported type {type(value).__name__}")


def _load_torch_object(path: Path, *, description: str) -> object:
    import torch

    require_regular_file(path, description=description)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AuditError(f"{description} is not safely readable: {path}") from exc


def load_finite_adapter(path: Path) -> dict[str, Any]:
    import torch

    payload = _load_torch_object(path, description="Delta-Mem adapter")
    require(isinstance(payload, dict) and payload, "adapter state dictionary is empty")
    for name, tensor in payload.items():
        require(isinstance(name, str), "adapter has a non-string key")
        require(isinstance(tensor, torch.Tensor), f"adapter entry is not a tensor: {name}")
        _validate_finite_tree(tensor, description=f"adapter.{name}")
    return payload


def adapter_change_record(
    initial: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    trainable_names: Sequence[str],
) -> dict[str, object]:
    import torch

    require(list(candidate) == list(initial), "checkpoint adapter topology/order differs from step zero")
    trainable = set(trainable_names)
    require(trainable <= set(initial), "declared trainable topology is invalid")
    changed: list[str] = []
    changed_trainable: list[str] = []
    changed_nontrainable: list[str] = []
    changed_layers: dict[str, set[int]] = {}
    maximum_absolute_delta = 0.0
    for name, initial_tensor in initial.items():
        candidate_tensor = candidate[name]
        require(
            initial_tensor.shape == candidate_tensor.shape
            and initial_tensor.dtype == candidate_tensor.dtype,
            f"checkpoint tensor metadata differs: {name}",
        )
        if torch.equal(initial_tensor, candidate_tensor):
            continue
        changed.append(name)
        if name in trainable:
            changed_trainable.append(name)
        else:
            changed_nontrainable.append(name)
        match = re.search(r"\.layers\.(\d+)\.self_attn\.(.+)$", name)
        require(match is not None, f"unexpected adapter tensor name: {name}")
        changed_layers.setdefault(match.group(2), set()).add(int(match.group(1)))
        if initial_tensor.is_floating_point() or initial_tensor.is_complex():
            maximum_absolute_delta = max(
                maximum_absolute_delta,
                float((candidate_tensor - initial_tensor).detach().abs().max().item()),
            )
    require(changed_trainable, "checkpoint changed no trainable adapter tensor")
    require(not changed_nontrainable, "checkpoint changed frozen adapter tensors")
    require(math.isfinite(maximum_absolute_delta) and maximum_absolute_delta > 0.0, "adapter delta is not positive and finite")
    expected_layers = set(EXPECTED_TARGET_LAYERS)
    coverage: dict[str, int] = {}
    for suffix in REQUIRED_CHANGED_SUFFIXES:
        layers = changed_layers.get(suffix, set())
        require(layers == expected_layers, f"checkpoint lacks all-layer update coverage for {suffix}")
        coverage[suffix] = len(layers)
    for layer in EXPECTED_TARGET_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn."
        initial_scale = initial[prefix + "delta_scale_raw"]
        candidate_scale = candidate[prefix + "delta_scale_raw"]
        require(initial_scale.shape == candidate_scale.shape == (4,), f"layer {layer} delta-scale shape differs")
        require(
            not torch.equal(initial_scale[0], candidate_scale[0])
            and not torch.equal(initial_scale[3], candidate_scale[3]),
            f"layer {layer} active Q/O delta-scale entries did not both change",
        )
        require(torch.equal(initial_scale[1:3], candidate_scale[1:3]), f"layer {layer} inactive K/V delta-scale entries changed")
        for suffix in INACTIVE_KV_SUFFIXES:
            name = prefix + suffix
            require(name not in trainable, f"inactive K/V tensor is trainable: {name}")
            require(torch.equal(initial[name], candidate[name]), f"inactive K/V tensor changed: {name}")
    return {
        "changed_tensor_count": len(changed),
        "changed_trainable_tensor_count": len(changed_trainable),
        "changed_nontrainable_tensor_count": 0,
        "maximum_absolute_delta": maximum_absolute_delta,
        "required_changed_layer_coverage": coverage,
        "inactive_kv_projection_tensors_unchanged": 42 * len(INACTIVE_KV_SUFFIXES),
        "inactive_kv_delta_scale_entries_unchanged": 42 * 2,
        "first_changed_trainable_tensors": changed_trainable[:8],
    }


def validate_adapter_change_evidence(change: object) -> None:
    require(isinstance(change, dict), "checkpoint adapter-change evidence is missing")
    require(
        isinstance(change.get("changed_trainable_tensor_count"), int)
        and int(change["changed_trainable_tensor_count"]) > 0
        and change.get("changed_nontrainable_tensor_count") == 0,
        "checkpoint adapter-change counts differ",
    )
    maximum_delta = _require_finite_number(
        change.get("maximum_absolute_delta"),
        description="checkpoint maximum adapter delta",
    )
    require(maximum_delta > 0.0, "checkpoint maximum adapter delta is not positive")
    coverage = change.get("required_changed_layer_coverage")
    require(
        isinstance(coverage, dict)
        and set(coverage) == set(REQUIRED_CHANGED_SUFFIXES)
        and all(coverage.get(suffix) == 42 for suffix in REQUIRED_CHANGED_SUFFIXES),
        "checkpoint all-layer update coverage differs",
    )
    require(
        change.get("inactive_kv_projection_tensors_unchanged")
        == 42 * len(INACTIVE_KV_SUFFIXES)
        and change.get("inactive_kv_delta_scale_entries_unchanged") == 42 * 2,
        "checkpoint inactive K/V evidence differs",
    )


def _expected_run_name_pattern(run_mode: str) -> str:
    prefix = r"scene_memory_v6_identityproof_all42_qo_r4_fail32"
    if run_mode == "prepare":
        return prefix + r"_s32_run[1-9][0-9]*_prepare"
    if run_mode == "smoke":
        return prefix + r"_smoke1_run[1-9][0-9]*"
    return prefix + r"_s32_run[1-9][0-9]*"


def _validate_run_root(run_root: Path, *, run_mode: str) -> Path:
    run_root = run_root.expanduser().resolve()
    require(run_root.parent == EXPECTED_RUN_ROOT, "run root is outside the locked output root")
    require(re.fullmatch(_expected_run_name_pattern(run_mode), run_root.name) is not None, "run name differs from identity-proof lineage")
    require(run_root.is_dir() and not run_root.is_symlink(), f"run root is invalid: {run_root}")
    return run_root


def validate_source_lock(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    lock = load_json_object(path, description="source lock")
    require(lock.get("schema") == SOURCE_LOCK_SCHEMA, "source-lock schema differs")
    require(lock.get("experiment") == EXPERIMENT, "source-lock experiment differs")
    require(lock.get("repository") == str(EXPECTED_REPO), "source-lock repository differs")
    validate_checksum(lock, field="lock_sha256", description="source lock")
    sources = lock.get("sources")
    require(isinstance(sources, dict), "source lock omits source records")
    auditor = sources.get("run_audit")
    require(
        isinstance(auditor, dict)
        and auditor.get("relative_path") == AUDITOR_RELATIVE_PATH
        and auditor.get("sha256") == sha256_file(Path(__file__).resolve()),
        "source lock does not bind this run auditor",
    )
    expected_data = {
        "pair_manifest": (EXPECTED_PAIR_MANIFEST, EXPECTED_PAIR_MANIFEST_SHA256),
        "train32": (EXPECTED_TRAIN, EXPECTED_TRAIN_SHA256),
        "train32_row_manifest": (EXPECTED_TRAIN_MANIFEST, EXPECTED_TRAIN_MANIFEST_SHA256),
        "hard32": (EXPECTED_HARD32, EXPECTED_HARD32_SHA256),
        "hard32_row_manifest": (EXPECTED_HARD32_MANIFEST, EXPECTED_HARD32_MANIFEST_SHA256),
        "hard32_indices": (EXPECTED_HARD32_INDICES, EXPECTED_HARD32_INDICES_SHA256),
    }
    records = lock.get("data_artifacts")
    require(isinstance(records, dict), "source lock omits data artifacts")
    for name, (expected_path, expected_sha) in expected_data.items():
        record = records.get(name)
        require(
            isinstance(record, dict)
            and record.get("path") == str(expected_path)
            and record.get("sha256") == expected_sha
            and sha256_file(expected_path) == expected_sha,
            f"source-lock data artifact differs: {name}",
        )
    return lock


def validate_data_contract(path: Path) -> dict[str, object]:
    data = load_json_object(path, description="data-contract manifest")
    require(data.get("schema") == DATA_SCHEMA, "data-contract schema differs")
    require(data.get("experiment") == EXPERIMENT, "data-contract experiment differs")
    validate_checksum(data, field="manifest_sha256", description="data-contract manifest")
    train = data.get("training_partition")
    require(
        isinstance(train, dict)
        and train.get("source_split") == "train"
        and train.get("rows") == 32
        and train.get("path") == str(EXPECTED_TRAIN)
        and train.get("sha256") == EXPECTED_TRAIN_SHA256
        and train.get("row_manifest", {}).get("path") == str(EXPECTED_TRAIN_MANIFEST)
        and train.get("row_manifest", {}).get("sha256") == EXPECTED_TRAIN_MANIFEST_SHA256
        and train.get("val_or_test_rows_emitted_for_training") == 0,
        "training partition differs",
    )
    hard = data.get("hard_evaluation_selection")
    require(
        isinstance(hard, dict)
        and hard.get("name") == "scene_v6_identity_hard32"
        and hard.get("source_split") == "val"
        and hard.get("rows") == 32
        and hard.get("path") == str(EXPECTED_HARD32)
        and hard.get("sha256") == EXPECTED_HARD32_SHA256
        and hard.get("source_indices") == list(EXPECTED_HARD32_SOURCE_INDICES)
        and hard.get("source_indices_file", {}).get("path") == str(EXPECTED_HARD32_INDICES)
        and hard.get("source_indices_file", {}).get("sha256") == EXPECTED_HARD32_INDICES_SHA256
        and hard.get("row_manifest", {}).get("path") == str(EXPECTED_HARD32_MANIFEST)
        and hard.get("row_manifest", {}).get("sha256") == EXPECTED_HARD32_MANIFEST_SHA256,
        "hard32 selection differs",
    )
    require(data.get("pair_manifest", {}).get("sha256") == EXPECTED_PAIR_MANIFEST_SHA256, "data-contract pair manifest differs")
    selected_overlap = data.get("selected_slice_overlap_audit")
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
    policy = data.get("test_policy")
    require(
        isinstance(policy, dict)
        and policy.get("rows_emitted_for_training") == 0
        and policy.get("rows_emitted_for_checkpoint_selection") == 0
        and policy.get("full_validation_before_hard32_pass") == "forbidden"
        and policy.get("test_before_validation_selection_receipt") == "forbidden",
        "test/full-validation policy differs",
    )
    return data


def validate_pair_manifest(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    require(path == EXPECTED_PAIR_MANIFEST, "pair manifest path differs")
    require(sha256_file(path) == EXPECTED_PAIR_MANIFEST_SHA256, "pair manifest SHA differs")
    manifest = load_json_object(path, description="pair manifest")
    require(manifest.get("schema") == "rwkv_ms_scene_failure_pairs.v1", "pair manifest schema differs")
    partitions = manifest.get("partitions")
    require(isinstance(partitions, dict), "pair manifest partitions are missing")
    train = partitions.get("train")
    hard = partitions.get("holdout")
    require(
        isinstance(train, dict)
        and train.get("rows") == 32
        and train.get("source_split") == "train"
        and train.get("data", {}).get("path") == str(EXPECTED_TRAIN)
        and train.get("data", {}).get("sha256") == EXPECTED_TRAIN_SHA256,
        "pair train partition differs",
    )
    require(
        isinstance(hard, dict)
        and hard.get("rows") == 32
        and hard.get("source_split") == "val"
        and hard.get("data", {}).get("path") == str(EXPECTED_HARD32)
        and hard.get("data", {}).get("sha256") == EXPECTED_HARD32_SHA256,
        "pair hard32 partition differs",
    )
    test = manifest.get("sources", {}).get("test", {})
    require(test.get("emitted_for_training") is False and test.get("emitted_for_holdout") is False, "pair manifest emitted test rows")
    return manifest


def _identity_target_stratum(source_count: int, donor_count: int) -> str:
    if (source_count == 0) != (donor_count == 0):
        return "presence"
    if source_count == donor_count:
        return "same_cardinality_value"
    return "cross_cardinality_value"


def validate_identity_pairing_manifest(path: Path) -> dict[str, object]:
    manifest = load_json_object(path, description="identity pairing manifest")
    require(manifest.get("schema_version") == 2, "identity pairing schema differs")
    expected = {
        "objective_version": OBJECTIVE_PROTOCOL["objective_version"],
        "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "pairing_scope": "within_post_split_partition",
        "target_mode": OBJECTIVE_PROTOCOL["target_mode"],
        "causal_prefix_mode": SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "semantic_mask_mode": OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "semantic_loss_normalization": OBJECTIVE_PROTOCOL["semantic_loss_normalization"],
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
    }
    for field, value in expected.items():
        require(manifest.get(field) == value, f"identity pairing differs: {field}")
    validate_checksum(manifest, field="manifest_sha256", description="identity pairing manifest")
    splits = manifest.get("splits")
    require(isinstance(splits, dict) and set(splits) == {"train"}, "identity pairing split set differs")
    train = splits["train"]
    require(isinstance(train, dict), "identity train pairing is missing")
    for field, value in {
        "split": "train",
        "pairing_version": SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": True,
        "target_mode": OBJECTIVE_PROTOCOL["target_mode"],
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
        require(train.get(field) == value, f"identity train pairing differs: {field}")
    validate_checksum(train, field="manifest_sha256", description="identity train pairing")
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
        require(row.get("target_mode") == OBJECTIVE_PROTOCOL["target_mode"], "identity target mode differs")
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
    return manifest


def _stage_spec(run_mode: str) -> RunSpec:
    if run_mode == "prepare":
        return PREPARE_SPEC
    require(run_mode in RUN_SPECS, f"unsupported run mode: {run_mode}")
    return RUN_SPECS[run_mode]


def validate_launch_manifest(
    path: Path,
    *,
    run_mode: str,
    run_root: Path,
    source_lock_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    launch = load_json_object(path, description="launch manifest")
    require(launch.get("schema") == LAUNCH_SCHEMA, "launch schema differs")
    require(launch.get("experiment") == EXPERIMENT, "launch experiment differs")
    require(launch.get("run_mode") == run_mode, "launch run mode differs")
    require(launch.get("fresh_run") is True, "launch is not fresh")
    require(launch.get("resume_from_checkpoint") is None, "launch resumed training")
    require(launch.get("warm_start_from_checkpoint") is None, "launch warm-started training")
    validate_checksum(launch, field="manifest_sha256", description="launch manifest")
    spec = _stage_spec(run_mode)
    stage = launch.get("stage")
    require(isinstance(stage, dict), "launch stage is missing")
    for field, expected in {
        "source_partition_rows": 32,
        "optimization_updates": spec.optimization_updates,
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "save_total_limit": spec.save_total_limit,
        "warmup_steps": spec.warmup_steps,
        "checkpoint_steps": list(spec.checkpoint_steps),
    }.items():
        require(stage.get(field) == expected, f"launch stage differs: {field}")
    paths = launch.get("paths")
    require(
        isinstance(paths, dict)
        and paths.get("output_dir") == str(run_root)
        and paths.get("initial_adapter_dir") == str(run_root / "initial_adapter")
        and paths.get("train_file") == str(EXPECTED_TRAIN)
        and paths.get("pair_manifest") == str(EXPECTED_PAIR_MANIFEST),
        "launch paths differ",
    )
    topology = launch.get("topology")
    for field, expected in {
        "model_layers": 42,
        "target_layers": EXPECTED_TARGET_LAYERS,
        "delta_heads": EXPECTED_DELTA_HEADS,
        "rank": 4,
        "alpha": 8,
        "memory_backend": "rwkv_ms",
        "rwkv_ms_semantics_version": 2,
        "fusion": "direct_add_at_attention_output",
    }.items():
        require(isinstance(topology, dict) and topology.get(field) == expected, f"launch topology differs: {field}")
    objective = launch.get("objective")
    require(isinstance(objective, dict), "launch objective is missing")
    for field, expected in OBJECTIVE_PROTOCOL.items():
        require(objective.get(field) == expected, f"launch objective differs: {field}")

    source_lock_path = source_lock_path.expanduser().resolve()
    source_lock = validate_source_lock(source_lock_path)
    source_record = launch.get("source_lock")
    require(
        isinstance(source_record, dict)
        and source_record.get("path") == str(source_lock_path)
        and source_record.get("file_sha256") == sha256_file(source_lock_path)
        and source_record.get("payload_sha256") == canonical_sha256(source_lock)
        and source_record.get("payload") == source_lock,
        "launch source-lock binding differs",
    )
    data_record = launch.get("data_contract")
    require(isinstance(data_record, dict), "launch data-contract binding is missing")
    data_path = Path(str(data_record.get("path", ""))).expanduser().resolve()
    require(data_path == run_root / "data_contract_manifest.json", "launch data-contract path differs")
    data = validate_data_contract(data_path)
    require(data_record.get("file_sha256") == sha256_file(data_path), "launch data-contract file SHA differs")
    require(data_record.get("manifest_sha256") == data.get("manifest_sha256"), "launch data-contract payload SHA differs")
    require(
        data_record.get("selected_slice_overlap_audit")
        == data.get("selected_slice_overlap_audit"),
        "launch selected-slice overlap binding differs",
    )
    token_record = launch.get("tokenization_lock")
    require(isinstance(token_record, dict), "launch tokenization lock is missing")
    token_path = Path(str(token_record.get("path", ""))).expanduser().resolve()
    token = load_json_object(token_path, description="tokenization lock")
    require(token.get("schema") == TOKENIZATION_LOCK_SCHEMA, "tokenization-lock schema differs")
    validate_checksum(token, field="lock_sha256", description="tokenization lock")
    require(
        token.get("persisted_cache_enabled") is False
        and token.get("rebuild_each_fresh_run") is True
        and token_record.get("file_sha256") == sha256_file(token_path)
        and token_record.get("payload_sha256") == canonical_sha256(token),
        "launch tokenization-lock binding differs",
    )
    pair = validate_pair_manifest(EXPECTED_PAIR_MANIFEST)
    return launch, data, source_lock, pair


def validate_delta_config(config: Mapping[str, object]) -> None:
    for field, expected in {
        "memory_backend": "rwkv_ms",
        "rwkv_ms_semantics_version": 2,
        "rank": 4,
        "alpha": 8,
        "delta_heads": EXPECTED_DELTA_HEADS,
        "target_layers": EXPECTED_TARGET_LAYERS,
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
    }.items():
        require(config.get(field) == expected, f"Delta-Mem config differs: {field}")


def validate_training_protocol(
    protocol: Mapping[str, object],
    *,
    run_mode: str,
    pairing: Mapping[str, object],
) -> None:
    spec = _stage_spec(run_mode)
    expected = {
        "train_file": str(EXPECTED_TRAIN),
        "tokenized_samples": 32,
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "max_length": 256,
        "max_write_length": 1280,
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_identity_ce",
        "memory_objective_version": OBJECTIVE_PROTOCOL["objective_version"],
        "scene_state_identity_margin": 0.5,
        "scene_state_margin_mode": "per_row_hinge_relu_v1",
        "scene_state_objective_formula": OBJECTIVE_PROTOCOL["objective_formula"],
        "scene_state_correct_all_semantic_scope": OBJECTIVE_PROTOCOL[
            "correct_all_semantic_scope"
        ],
        "scene_state_pair_semantic_scope": OBJECTIVE_PROTOCOL[
            "pair_semantic_scope"
        ],
        "scene_state_donor_margin_scope": OBJECTIVE_PROTOCOL["donor_margin_scope"],
        "scene_state_zero_diagnostic_scope": OBJECTIVE_PROTOCOL[
            "zero_diagnostic_scope"
        ],
        "scene_state_zero_diagnostic_gradient": False,
        "scene_state_read_time_positions_observable": False,
        "scene_state_pairing_length_control": SCENE_STATE_IDENTITY_PAIRING_LENGTH_CONTROL,
        "scene_state_pairing_refinement": SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "scene_state_identity_target_strata": list(
            SCENE_STATE_IDENTITY_TARGET_STRATA
        ),
        "scene_state_identity_backward_mode": OBJECTIVE_PROTOCOL["backward_mode"],
        "scene_state_identity_read_protocol": OBJECTIVE_PROTOCOL["read_protocol"],
        "scene_state_identity_zero_protocol": OBJECTIVE_PROTOCOL["zero_protocol"],
        "scene_state_semantic_mask_mode": OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "scene_state_semantic_loss_normalization": OBJECTIVE_PROTOCOL["semantic_loss_normalization"],
        "scene_state_identity_target_mode": OBJECTIVE_PROTOCOL["target_mode"],
        "scene_state_identity_causal_prefix_mode": (
            SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE
        ),
        "scene_state_full_correct_ce_weight": 1.0,
        "scene_state_correct_all_semantic_ce_weight": 1.0,
        "scene_state_donor_margin_weight": 1.0,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "validation_split_ratio": 0.0,
        "seed": 42,
        "data_seed": 42,
        "train_sampler_seed": 42,
        "train_sampler_mode": "torch_random_sampler_seed_equals_data_seed_v1",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": spec.warmup_steps,
        "max_steps": spec.max_steps,
        "save_steps": spec.save_steps,
        "frozen_mlp_activation_checkpointing": True,
    }
    for field, expected_value in expected.items():
        require(protocol.get(field) == expected_value, f"training protocol differs: {field}")
    require(
        "scene_state_zero_margin_weight" not in protocol,
        "training protocol contains obsolete zero-margin weight",
    )
    identity = protocol.get("scene_state_source_manifest")
    require(
        isinstance(identity, dict)
        and identity.get("path") == str(EXPECTED_PAIR_MANIFEST)
        and identity.get("file_sha256") == EXPECTED_PAIR_MANIFEST_SHA256
        and identity.get("train_file") == str(EXPECTED_TRAIN)
        and identity.get("train_file_sha256") == EXPECTED_TRAIN_SHA256
        and identity.get("train_rows") == 32
        and identity.get("train_source_split") == "train",
        "training protocol source-manifest identity differs",
    )
    train_pairing = pairing["splits"]["train"]
    pairing_summary = protocol.get("scene_state_identity_pairing")
    expected_pairing_summary = {
        key: pairing[key]
        for key in (
            "pairing_version",
            "pairing_refinement",
            "pairing_refinement_applied",
            "pairing_scope",
            "target_mode",
            "causal_prefix_mode",
            "semantic_mask_mode",
            "semantic_loss_normalization",
            "target_token_count",
            "target_stratum_row_counts",
            "source_boundary_count_histogram",
            "write_token_count_delta_max",
            "write_token_count_delta_mean",
            "write_token_count_delta_total",
            "nearest_baseline_write_token_count_delta_max",
            "nearest_baseline_write_token_count_delta_total",
            "data_seed",
            "tokenized_fingerprint",
            "tokenized_dataset_sha256",
            "manifest_sha256",
        )
    }
    expected_pairing_summary["splits"] = {
        "train": {
            key: train_pairing[key]
            for key in (
                "sample_count",
                "pair_count",
                "target_token_count",
                "causal_prefix_mode",
                "target_stratum_row_counts",
                "source_boundary_count_histogram",
                "write_token_count_delta_max",
                "write_token_count_delta_mean",
                "write_token_count_delta_total",
                "nearest_baseline_write_token_count_delta_max",
                "nearest_baseline_write_token_count_delta_total",
                "pairing_refinement_applied",
                "source_fingerprint",
                "paired_fingerprint",
                "pairs_sha256",
                "manifest_sha256",
            )
        }
    }
    require(
        pairing_summary == expected_pairing_summary,
        "training protocol pairing summary differs",
    )
    cache_identity = protocol.get("tokenized_cache_identity")
    require(
        isinstance(cache_identity, dict)
        and cache_identity.get("rows") == 32
        and cache_identity.get("persisted_files") == []
        and cache_identity.get("ordered_content_sha256") == protocol.get("tokenized_dataset_sha256"),
        "training protocol does not prove fresh in-memory tokenization",
    )
    require(protocol.get("expected_tokenized_dataset_sha256") is None, "training protocol reused a persisted tokenization lock")


def validate_initial_adapter(
    initial_dir: Path,
    *,
    run_mode: str,
    run_root: Path,
    launch: Mapping[str, object],
    pairing: Mapping[str, object],
    require_prepare_reference: bool,
) -> tuple[dict[str, object], dict[str, Any], dict[str, object], dict[str, object]]:
    manifest_path = initial_dir / "initial_adapter_manifest.json"
    adapter_path = initial_dir / "delta_mem_adapter.pt"
    config_path = initial_dir / "delta_mem_config.json"
    protocol_path = initial_dir / "training_protocol.json"
    manifest = load_json_object(manifest_path, description="initial-adapter manifest")
    validate_checksum(manifest, field="manifest_sha256", description="initial-adapter manifest")
    for field, expected in {
        "schema": INITIAL_ADAPTER_SCHEMA,
        "artifact_kind": "seeded_freshly_attached_delta_mem_adapter",
        "global_step": 0,
        "fresh_run": True,
        "training_started": False,
        "optimizer_created": False,
        "optimizer_state_included": False,
        "seed": 42,
        "data_seed": 42,
        "output_dir": str(initial_dir),
    }.items():
        require(manifest.get(field) == expected, f"initial-adapter manifest differs: {field}")
    dataset = manifest.get("dataset")
    require(
        isinstance(dataset, dict)
        and dataset.get("train_file") == str(EXPECTED_TRAIN)
        and dataset.get("train_file_sha256") == EXPECTED_TRAIN_SHA256
        and dataset.get("train_samples") == 32,
        "initial-adapter dataset differs",
    )
    cache_identity = dataset.get("tokenized_cache_identity")
    require(
        isinstance(cache_identity, dict)
        and cache_identity.get("rows") == 32
        and cache_identity.get("persisted_files") == []
        and cache_identity.get("ordered_content_sha256") == dataset.get("tokenized_dataset_sha256"),
        "initial adapter reused persisted tokenization",
    )
    identity_unsigned = dict(cache_identity)
    recorded_identity_sha = identity_unsigned.pop("identity_sha256", None)
    require(recorded_identity_sha == canonical_sha256(identity_unsigned), "tokenized identity checksum differs")
    columns = cache_identity.get("column_names")
    require(isinstance(columns, list) and "scene_state_semantic_mask" in columns, "semantic token mask is absent from tokenization")
    topology = manifest.get("topology")
    expected_replaced = [f"model.language_model.layers.{layer}.self_attn" for layer in EXPECTED_TARGET_LAYERS]
    expected_trainable = [
        f"model.language_model.layers.{layer}.self_attn.{suffix}"
        for layer in EXPECTED_TARGET_LAYERS
        for suffix in EXPECTED_TRAINABLE_SUFFIXES
    ]
    require(
        isinstance(topology, dict)
        and topology.get("replaced_modules") == expected_replaced
        and topology.get("trainable_names") == expected_trainable
        and topology.get("adapter_tensor_count") == 42 * len(EXPECTED_ADAPTER_SUFFIXES),
        "initial-adapter topology differs",
    )
    launch_record = manifest.get("launch_manifest")
    data_record = manifest.get("data_contract_manifest")
    require(
        isinstance(launch_record, dict)
        and launch_record.get("path") == str(run_root / "launch_manifest.json")
        and launch_record.get("sha256") == sha256_file(run_root / "launch_manifest.json"),
        "initial-adapter launch lineage differs",
    )
    require(
        isinstance(data_record, dict)
        and data_record.get("path") == str(run_root / "data_contract_manifest.json")
        and data_record.get("sha256") == sha256_file(run_root / "data_contract_manifest.json"),
        "initial-adapter data lineage differs",
    )
    config = load_json_object(config_path, description="initial Delta-Mem config")
    validate_delta_config(config)
    protocol = load_json_object(protocol_path, description="initial training protocol")
    validate_training_protocol(protocol, run_mode=run_mode, pairing=pairing)
    protocol_record = manifest.get("training_protocol")
    require(
        isinstance(protocol_record, dict)
        and protocol_record.get("canonical_sha256") == canonical_sha256(protocol)
        and protocol_record.get("file_sha256") == sha256_file(protocol_path),
        "initial training-protocol lineage differs",
    )
    adapter = load_finite_adapter(adapter_path)
    expected_names = {
        f"model.language_model.layers.{layer}.self_attn.{suffix}"
        for layer in EXPECTED_TARGET_LAYERS
        for suffix in EXPECTED_ADAPTER_SUFFIXES
    }
    require(set(adapter) == expected_names, "initial adapter tensor set differs")
    require(sha256_file(adapter_path) == REVIEWED_INITIAL_ADAPTER_SHA256, "initial adapter differs from the reviewed deterministic seed")
    if require_prepare_reference:
        authorization = launch.get("prepare_authorization")
        require(isinstance(authorization, dict), "launch prepare authorization is missing")
        receipt_path = Path(str(authorization.get("receipt_path", ""))).expanduser().resolve()
        receipt = load_json_object(receipt_path, description="prepare receipt")
        validate_checksum(receipt, field="receipt_sha256", description="prepare receipt")
        require(authorization.get("receipt_file_sha256") == sha256_file(receipt_path), "launch prepare receipt SHA differs")
        reference = receipt.get("adapter")
        require(
            isinstance(reference, dict)
            and reference.get("file_sha256") == REVIEWED_INITIAL_ADAPTER_SHA256,
            "prepare receipt does not bind the reviewed adapter",
        )
        require(sha256_file(Path(str(reference.get("path")))) == REVIEWED_INITIAL_ADAPTER_SHA256, "prepare reference adapter changed")
    return manifest, adapter, config, protocol


def validate_step_history(
    trainer_state: Mapping[str, object],
    *,
    checkpoint_step: int,
    spec: RunSpec,
) -> dict[str, object]:
    require(trainer_state.get("global_step") == checkpoint_step, "trainer global step differs")
    require(trainer_state.get("max_steps") == spec.max_steps, "trainer max steps differs")
    history = trainer_state.get("log_history")
    require(isinstance(history, list), "trainer log history is missing")
    metric_records: list[dict[str, object]] = []
    for record in history:
        if not isinstance(record, dict) or not isinstance(record.get("step"), int):
            continue
        if not all(metric in record for metric in REQUIRED_STEP_METRICS):
            continue
        selected = {"step": record["step"]}
        for metric in REQUIRED_STEP_METRICS:
            selected[metric] = _require_finite_number(
                record[metric],
                description=f"step {record['step']} {metric}",
            )
        metric_records.append(selected)
    require([record["step"] for record in metric_records] == list(range(1, checkpoint_step + 1)), "three-branch history is incomplete or reordered")
    positive_gradients = 0
    for record in metric_records:
        step = int(record["step"])
        require(float(record["loss"]) >= 0.0, f"step {step} loss is negative")
        require(float(record["grad_norm"]) >= 0.0, f"step {step} grad norm is negative")
        positive_gradients += int(float(record["grad_norm"]) > 0.0)
        for metric in IDENTITY_METRICS[:5]:
            require(float(record[metric]) >= 0.0, f"step {step} CE is negative: {metric}")
        require(
            float(record[IDENTITY_METRICS[7]]) >= 0.0,
            f"step {step} donor margin loss is negative",
        )
        for metric in IDENTITY_METRICS[8:10]:
            require(0.0 <= float(record[metric]) <= 1.0, f"step {step} fraction is invalid: {metric}")
        require(float(record[IDENTITY_METRICS[10]]) > 0.0, f"step {step} selected no semantic token")
        semantic_rows = float(record[IDENTITY_METRICS[11]])
        require(semantic_rows > 0.0, f"step {step} selected no semantic row")
        stratum_rows = [float(record[metric]) for metric in IDENTITY_METRICS[12:15]]
        require(
            all(value >= 0.0 for value in stratum_rows)
            and math.isclose(sum(stratum_rows), semantic_rows),
            f"step {step} target-stratum coverage differs",
        )
    require(positive_gradients > 0, "trainer recorded no positive gradient")
    return {
        "first_step": 1,
        "last_step": checkpoint_step,
        "records": checkpoint_step,
        "identity_metrics_finite": True,
        "metric_names": list(REQUIRED_STEP_METRICS),
        "metric_records": metric_records,
        "metric_records_sha256": canonical_sha256(metric_records),
        "first_loss": metric_records[0]["loss"],
        "last_loss": metric_records[-1]["loss"],
        "positive_gradient_records": positive_gradients,
    }


def _checkpoint_artifacts(checkpoint: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    for filename in REQUIRED_CHECKPOINT_FILES:
        require_regular_file(checkpoint / filename, description=f"checkpoint {filename}")
    rng_paths = sorted(checkpoint.glob("rng_state*.pth"))
    require(rng_paths, "checkpoint RNG state is missing")
    artifacts: dict[str, object] = {
        "adapter": file_record(checkpoint / "delta_mem_adapter.pt"),
        "config": file_record(
            checkpoint / "delta_mem_config.json",
            json_payload=load_json_object(checkpoint / "delta_mem_config.json", description="checkpoint config"),
        ),
        "protocol": file_record(
            checkpoint / "training_protocol.json",
            json_payload=load_json_object(checkpoint / "training_protocol.json", description="checkpoint protocol"),
        ),
        "trainer_state": file_record(
            checkpoint / "trainer_state.json",
            json_payload=load_json_object(checkpoint / "trainer_state.json", description="checkpoint trainer state"),
        ),
        "optimizer": file_record(checkpoint / "optimizer.pt"),
        "scheduler": file_record(checkpoint / "scheduler.pt"),
        "rng": [file_record(path) for path in rng_paths],
    }
    config = load_json_object(checkpoint / "delta_mem_config.json", description="checkpoint config")
    protocol = load_json_object(checkpoint / "training_protocol.json", description="checkpoint protocol")
    return artifacts, config, protocol


def write_json_atomic_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    require(not path.exists(), f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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


def audit_checkpoint(
    *,
    run_mode: str,
    run_root: Path,
    checkpoint_step: int,
    source_lock: Path,
    receipt: Path | None = None,
) -> dict[str, object]:
    require(run_mode in RUN_SPECS, f"unsupported checkpoint run mode: {run_mode}")
    spec = RUN_SPECS[run_mode]
    require(checkpoint_step in spec.checkpoint_steps, "checkpoint step is outside the locked schedule")
    run_root = _validate_run_root(run_root, run_mode=run_mode)
    checkpoint = run_root / "trainer" / f"checkpoint-{checkpoint_step}"
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"checkpoint directory is invalid: {checkpoint}")
    receipt = (checkpoint / "checkpoint_receipt.json") if receipt is None else receipt.expanduser().resolve()
    require(receipt == checkpoint / "checkpoint_receipt.json", "checkpoint receipt path differs")
    if receipt.exists():
        return validate_existing_checkpoint_receipt(
            receipt,
            expected_run_mode=run_mode,
            expected_run_root=run_root,
            expected_step=checkpoint_step,
        )

    launch_path = run_root / "launch_manifest.json"
    launch, data, source, pair = validate_launch_manifest(
        launch_path,
        run_mode=run_mode,
        run_root=run_root,
        source_lock_path=source_lock,
    )
    pairing_path = run_root / "scene_state_identity_pairing_manifest.json"
    pairing = validate_identity_pairing_manifest(pairing_path)
    initial_manifest, initial_adapter, initial_config, initial_protocol = validate_initial_adapter(
        run_root / "initial_adapter",
        run_mode=run_mode,
        run_root=run_root,
        launch=launch,
        pairing=pairing,
        require_prepare_reference=True,
    )
    artifacts, checkpoint_config, checkpoint_protocol = _checkpoint_artifacts(checkpoint)
    require(checkpoint_config == initial_config, "checkpoint config differs from step zero")
    require(checkpoint_protocol == initial_protocol, "checkpoint protocol differs from step zero")
    trainer_state = load_json_object(checkpoint / "trainer_state.json", description="checkpoint trainer state")
    history = validate_step_history(trainer_state, checkpoint_step=checkpoint_step, spec=spec)
    candidate = load_finite_adapter(checkpoint / "delta_mem_adapter.pt")
    trainable_names = initial_manifest.get("topology", {}).get("trainable_names")
    require(isinstance(trainable_names, list), "initial trainable-name inventory is missing")
    change = adapter_change_record(initial_adapter, candidate, trainable_names=trainable_names)
    optimizer = _load_torch_object(checkpoint / "optimizer.pt", description="optimizer state")
    require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict) and optimizer["state"], "optimizer state is empty")
    _validate_finite_tree(optimizer, description="optimizer")
    scheduler = _load_torch_object(checkpoint / "scheduler.pt", description="scheduler state")
    require(isinstance(scheduler, dict) and scheduler, "scheduler state is empty")
    _validate_finite_tree(scheduler, description="scheduler")

    hard = data["hard_evaluation_selection"]
    payload: dict[str, object] = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "run_mode": run_mode,
        "run_root": str(run_root),
        "checkpoint_step": checkpoint_step,
        "checkpoint_dir": str(checkpoint),
        "complete": True,
        "training_summary_required": False,
        "hard32_only": True,
        "full170_authorized": False,
        "test_forbidden": True,
        "auditor": file_record(Path(__file__).resolve()),
        "launch": file_record(launch_path, json_payload=launch),
        "data_contract": file_record(run_root / "data_contract_manifest.json", json_payload=data),
        "source_lock": file_record(source_lock, json_payload=source),
        "pair_manifest": file_record(EXPECTED_PAIR_MANIFEST, json_payload=pair),
        "train_partition": {
            **file_record(EXPECTED_TRAIN),
            "rows": 32,
            "source_split": "train",
            "row_manifest": file_record(EXPECTED_TRAIN_MANIFEST),
        },
        "hard32_selection": {
            "rows": 32,
            "source_split": "val",
            "test_rows": 0,
            "source_indices": list(EXPECTED_HARD32_SOURCE_INDICES),
            "holdout": file_record(EXPECTED_HARD32),
            "indices": file_record(EXPECTED_HARD32_INDICES, json_payload=load_json_object(EXPECTED_HARD32_INDICES, description="hard32 indices")),
            "row_manifest": file_record(EXPECTED_HARD32_MANIFEST),
            "selection_rule": hard["selection_rule"],
        },
        "identity_pairing_manifest": file_record(pairing_path, json_payload=pairing),
        "initial_adapter": {
            "manifest": file_record(run_root / "initial_adapter" / "initial_adapter_manifest.json", json_payload=initial_manifest),
            "adapter": file_record(run_root / "initial_adapter" / "delta_mem_adapter.pt"),
            "config": file_record(run_root / "initial_adapter" / "delta_mem_config.json", json_payload=initial_config),
            "protocol": file_record(run_root / "initial_adapter" / "training_protocol.json", json_payload=initial_protocol),
        },
        "checkpoint_artifacts": artifacts,
        "trainer_state": {
            "global_step": trainer_state["global_step"],
            "max_steps": trainer_state["max_steps"],
        },
        "objective": dict(OBJECTIVE_PROTOCOL),
        "history": history,
        "adapter_change": change,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json_atomic_exclusive(receipt, payload)
    saved = load_json_object(receipt, description="checkpoint receipt")
    require(saved == payload, "saved checkpoint receipt differs")
    return payload


def validate_existing_checkpoint_receipt(
    receipt_path: Path,
    *,
    expected_run_mode: str,
    expected_run_root: Path,
    expected_step: int,
) -> dict[str, object]:
    receipt_path = receipt_path.expanduser().resolve()
    expected_run_root = expected_run_root.expanduser().resolve()
    expected_checkpoint = expected_run_root / "trainer" / f"checkpoint-{expected_step}"
    require(receipt_path == expected_checkpoint / "checkpoint_receipt.json", "checkpoint receipt path differs")
    receipt = load_json_object(receipt_path, description="checkpoint receipt")
    validate_checksum(receipt, field="receipt_sha256", description="checkpoint receipt")
    for field, expected in {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "experiment": EXPERIMENT,
        "run_mode": expected_run_mode,
        "run_root": str(expected_run_root),
        "checkpoint_step": expected_step,
        "checkpoint_dir": str(expected_checkpoint),
        "complete": True,
        "training_summary_required": False,
        "hard32_only": True,
        "full170_authorized": False,
        "test_forbidden": True,
    }.items():
        require(receipt.get(field) == expected, f"checkpoint receipt differs: {field}")
    for field in ("auditor", "launch", "data_contract", "source_lock", "pair_manifest", "identity_pairing_manifest"):
        _validate_file_record(receipt.get(field), description=f"checkpoint {field}")
    train = receipt.get("train_partition")
    require(isinstance(train, dict) and train.get("rows") == 32 and train.get("source_split") == "train", "checkpoint train partition differs")
    _validate_file_record(train, description="checkpoint train partition")
    _validate_file_record(train.get("row_manifest"), description="checkpoint train row manifest")
    hard = receipt.get("hard32_selection")
    require(isinstance(hard, dict) and hard.get("rows") == 32 and hard.get("source_split") == "val" and hard.get("test_rows") == 0, "checkpoint hard32 selection differs")
    for field in ("holdout", "indices", "row_manifest"):
        _validate_file_record(hard.get(field), description=f"checkpoint hard32 {field}")
    initial = receipt.get("initial_adapter")
    require(isinstance(initial, dict), "checkpoint initial-adapter records are missing")
    for field in ("manifest", "adapter", "config", "protocol"):
        _validate_file_record(initial.get(field), description=f"checkpoint initial {field}")
    artifacts = receipt.get("checkpoint_artifacts")
    require(isinstance(artifacts, dict), "checkpoint artifact records are missing")
    for field in ("adapter", "config", "protocol", "trainer_state", "optimizer", "scheduler"):
        _validate_file_record(artifacts.get(field), description=f"checkpoint artifact {field}")
    rng = artifacts.get("rng")
    require(isinstance(rng, list) and rng, "checkpoint RNG records are missing")
    for index, record in enumerate(rng):
        _validate_file_record(record, description=f"checkpoint RNG {index}")
    trainer_state = load_json_object(
        Path(str(artifacts["trainer_state"]["path"])),
        description="checkpoint trainer state",
    )
    computed_history = validate_step_history(
        trainer_state,
        checkpoint_step=expected_step,
        spec=_stage_spec(expected_run_mode),
    )
    require(receipt.get("history") == computed_history, "checkpoint history evidence differs")
    initial_manifest = load_json_object(
        Path(str(initial["manifest"]["path"])),
        description="initial adapter manifest",
    )
    trainable_names = initial_manifest.get("topology", {}).get("trainable_names")
    require(isinstance(trainable_names, list), "initial trainable-name inventory is missing")
    computed_change = adapter_change_record(
        load_finite_adapter(Path(str(initial["adapter"]["path"]))),
        load_finite_adapter(Path(str(artifacts["adapter"]["path"]))),
        trainable_names=trainable_names,
    )
    validate_adapter_change_evidence(receipt.get("adapter_change"))
    require(
        receipt.get("adapter_change") == computed_change,
        "checkpoint adapter-change evidence differs",
    )
    require(receipt.get("objective") == dict(OBJECTIVE_PROTOCOL), "checkpoint objective differs")
    return receipt


def watch_checkpoints(
    *,
    run_mode: str,
    run_root: Path,
    source_lock: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> list[dict[str, object]]:
    require(run_mode in RUN_SPECS, f"unsupported watcher run mode: {run_mode}")
    require(timeout_seconds > 0.0 and poll_seconds > 0.0, "watcher timing must be positive")
    pending = list(RUN_SPECS[run_mode].checkpoint_steps)
    completed: list[dict[str, object]] = []
    last_errors: dict[int, str] = {}
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        for step in list(pending):
            try:
                completed.append(
                    audit_checkpoint(
                        run_mode=run_mode,
                        run_root=run_root,
                        checkpoint_step=step,
                        source_lock=source_lock,
                    )
                )
            except (AuditError, OSError) as exc:
                last_errors[step] = str(exc)
                continue
            pending.remove(step)
            last_errors.pop(step, None)
        if pending:
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    require(not pending, f"checkpoint watcher timed out: pending={pending} last_errors={last_errors}")
    return sorted(completed, key=lambda item: int(item["checkpoint_step"]))


def audit_prepare(
    *,
    run_root: Path,
    source_lock: Path,
    receipt: Path,
) -> dict[str, object]:
    run_root = _validate_run_root(run_root, run_mode="prepare")
    receipt = receipt.expanduser().resolve()
    require(receipt == run_root / "prepare_receipt.json", "prepare receipt path differs")
    launch_path = run_root / "launch_manifest.json"
    launch, data, source, _ = validate_launch_manifest(
        launch_path,
        run_mode="prepare",
        run_root=run_root,
        source_lock_path=source_lock,
    )
    pairing_path = run_root / "scene_state_identity_pairing_manifest.json"
    pairing = validate_identity_pairing_manifest(pairing_path)
    initial_manifest, _, config, protocol = validate_initial_adapter(
        run_root / "initial_adapter",
        run_mode="prepare",
        run_root=run_root,
        launch=launch,
        pairing=pairing,
        require_prepare_reference=False,
    )
    require(not (run_root / "trainer").exists(), "prepare-only run created a Trainer directory")
    prohibited = {"optimizer.pt", "scheduler.pt", "trainer_state.json"}
    require(not any(path.name in prohibited for path in run_root.rglob("*")), "prepare-only run contains training state")
    payload: dict[str, object] = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "run_mode": "prepare",
        "run_root": str(run_root),
        "fresh_run": True,
        "global_step": 0,
        "training_arguments_created": False,
        "trainer_created": False,
        "optimizer_created": False,
        "training_started": False,
        "auditor": file_record(Path(__file__).resolve()),
        "initial_adapter_manifest": file_record(run_root / "initial_adapter" / "initial_adapter_manifest.json", json_payload=initial_manifest),
        "adapter": file_record(run_root / "initial_adapter" / "delta_mem_adapter.pt"),
        "config": file_record(run_root / "initial_adapter" / "delta_mem_config.json", json_payload=config),
        "training_protocol": file_record(run_root / "initial_adapter" / "training_protocol.json", json_payload=protocol),
        "data_contract_manifest": file_record(run_root / "data_contract_manifest.json", json_payload=data),
        "launch_manifest": file_record(launch_path, json_payload=launch),
        "identity_pairing_manifest": file_record(pairing_path, json_payload=pairing),
        "source_lock": file_record(source_lock, json_payload=source),
        "tokenization_lock": file_record(Path(str(launch["tokenization_lock"]["path"])), json_payload=launch["tokenization_lock"]["payload"]),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json_atomic_exclusive(receipt, payload)
    return payload


def _find_active_trainers(run_root: Path) -> list[dict[str, object]]:
    active: list[dict[str, object]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return active
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = " ".join(
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if str(run_root) in command and "deltamem.train.delta_sft" in command:
            active.append({"pid": int(entry.name), "command": command})
    return sorted(active, key=lambda item: int(item["pid"]))


def validate_training_summary(
    path: Path,
    *,
    run_mode: str,
    run_root: Path,
    protocol: Mapping[str, object],
    pairing: Mapping[str, object],
) -> dict[str, object]:
    summary = load_json_object(path, description="training summary")
    expected = {
        "output_dir": str(run_root),
        "resume_from_checkpoint": None,
        "warm_start_from_checkpoint": None,
        "initial_adapter_output_dir": str(run_root / "initial_adapter"),
        "num_replaced_modules": 42,
        "num_trainable_tensors": 42 * len(EXPECTED_TRAINABLE_SUFFIXES),
        "num_checkpointed_frozen_mlps": 42,
        "tokenized_samples": 32,
        "train_samples": 32,
        "eval_samples": 0,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "train_sampler_seed": 42,
        "train_sampler_mode": "torch_random_sampler_seed_equals_data_seed_v1",
        "episode_recent_messages": 0,
        "max_write_length": 1280,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_identity_ce",
        "memory_objective_version": OBJECTIVE_PROTOCOL["objective_version"],
        "scene_boundary_payload_ce_weight": 0.0,
        "rwkv_ms_semantics_version": 2,
        "seed": 42,
        "data_seed": 42,
        "world_size": 1,
        "local_rank": -1,
        "tokenized_cache": False,
        "tokenized_cache_hit": False,
        "tokenized_cache_dir": None,
        "tokenized_dataset_source": "direct_map",
        "scene_state_identity_margin": 0.5,
        "scene_state_margin_mode": "per_row_hinge_relu_v1",
        "scene_state_identity_backward_mode": OBJECTIVE_PROTOCOL["backward_mode"],
        "scene_state_identity_read_protocol": OBJECTIVE_PROTOCOL["read_protocol"],
        "scene_state_identity_zero_protocol": OBJECTIVE_PROTOCOL["zero_protocol"],
        "scene_state_semantic_mask_mode": OBJECTIVE_PROTOCOL["semantic_mask_mode"],
        "scene_state_semantic_loss_normalization": OBJECTIVE_PROTOCOL[
            "semantic_loss_normalization"
        ],
        "scene_state_identity_target_mode": OBJECTIVE_PROTOCOL["target_mode"],
        "scene_state_full_correct_ce_weight": 1.0,
        "scene_state_correct_all_semantic_ce_weight": 1.0,
        "scene_state_donor_margin_weight": 1.0,
    }
    for field, expected_value in expected.items():
        require(summary.get(field) == expected_value, f"training summary differs: {field}")
    require(summary.get("training_protocol_sha256") == canonical_sha256(protocol), "summary protocol SHA differs")
    require(summary.get("scene_state_identity_pairing_manifest_sha256") == pairing.get("manifest_sha256"), "summary pairing SHA differs")
    _validate_finite_tree(summary.get("gate_stats"), description="summary.gate_stats")
    _validate_finite_tree(summary.get("output_ratio_stats"), description="summary.output_ratio_stats")
    return summary


def audit_run(
    *,
    run_mode: str,
    run_root: Path,
    log_file: Path,
    source_lock: Path,
    receipt: Path,
    trainer_exit_code: int,
    tee_exit_code: int,
    active_trainers: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    require(run_mode in RUN_SPECS, f"unsupported run mode: {run_mode}")
    require(trainer_exit_code == 0 and tee_exit_code == 0, "trainer or tee exit code is nonzero")
    run_root = _validate_run_root(run_root, run_mode=run_mode)
    receipt = receipt.expanduser().resolve()
    require(receipt == run_root / "run_audit_receipt.json", "run receipt path differs")
    log_file = log_file.expanduser().resolve()
    require_regular_file(log_file, description="training log")
    observed = list(active_trainers) if active_trainers is not None else _find_active_trainers(run_root)
    require(not observed, f"training process is still active: {observed}")
    launch, data, source, pair = validate_launch_manifest(
        run_root / "launch_manifest.json",
        run_mode=run_mode,
        run_root=run_root,
        source_lock_path=source_lock,
    )
    pairing_path = run_root / "scene_state_identity_pairing_manifest.json"
    pairing = validate_identity_pairing_manifest(pairing_path)
    initial_manifest, _, initial_config, initial_protocol = validate_initial_adapter(
        run_root / "initial_adapter",
        run_mode=run_mode,
        run_root=run_root,
        launch=launch,
        pairing=pairing,
        require_prepare_reference=True,
    )
    summary_path = run_root / "training_summary.json"
    summary = validate_training_summary(
        summary_path,
        run_mode=run_mode,
        run_root=run_root,
        protocol=initial_protocol,
        pairing=pairing,
    )
    spec = RUN_SPECS[run_mode]
    trainer_dir = run_root / "trainer"
    actual_steps = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in trainer_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    require(actual_steps == list(spec.checkpoint_steps), "completed run checkpoint set differs")
    checkpoint_records: list[dict[str, object]] = []
    for step in spec.checkpoint_steps:
        checkpoint_receipt_path = trainer_dir / f"checkpoint-{step}" / "checkpoint_receipt.json"
        checkpoint_receipt = validate_existing_checkpoint_receipt(
            checkpoint_receipt_path,
            expected_run_mode=run_mode,
            expected_run_root=run_root,
            expected_step=step,
        )
        checkpoint_records.append(
            {
                "step": step,
                "receipt": file_record(checkpoint_receipt_path, json_payload=checkpoint_receipt),
                "history": checkpoint_receipt["history"],
                "adapter_change": checkpoint_receipt["adapter_change"],
            }
        )
    final_checkpoint = trainer_dir / f"checkpoint-{spec.max_steps}"
    root_adapter = run_root / "delta_mem_adapter.pt"
    root_config = run_root / "delta_mem_config.json"
    root_protocol = run_root / "training_protocol.json"
    require(sha256_file(root_adapter) == sha256_file(final_checkpoint / "delta_mem_adapter.pt"), "root adapter differs from final checkpoint")
    require(load_json_object(root_config, description="root config") == initial_config, "root config differs from step zero")
    require(load_json_object(root_protocol, description="root protocol") == initial_protocol, "root protocol differs from step zero")
    payload: dict[str, object] = {
        "schema": RUN_RECEIPT_SCHEMA,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "run_mode": run_mode,
        "run_root": str(run_root),
        "complete": True,
        "trainer_exit_code": 0,
        "tee_exit_code": 0,
        "training_processes_active": [],
        "hard32_only": True,
        "full170_authorized": False,
        "test_forbidden": True,
        "auditor": file_record(Path(__file__).resolve()),
        "launch": file_record(run_root / "launch_manifest.json", json_payload=launch),
        "data_contract": file_record(run_root / "data_contract_manifest.json", json_payload=data),
        "source_lock": file_record(source_lock, json_payload=source),
        "pair_manifest": file_record(EXPECTED_PAIR_MANIFEST, json_payload=pair),
        "identity_pairing_manifest": file_record(pairing_path, json_payload=pairing),
        "log": file_record(log_file),
        "initial_adapter": {
            "manifest": file_record(run_root / "initial_adapter" / "initial_adapter_manifest.json", json_payload=initial_manifest),
            "adapter": file_record(run_root / "initial_adapter" / "delta_mem_adapter.pt"),
            "config": file_record(run_root / "initial_adapter" / "delta_mem_config.json", json_payload=initial_config),
            "protocol": file_record(run_root / "initial_adapter" / "training_protocol.json", json_payload=initial_protocol),
        },
        "checkpoints": checkpoint_records,
        "completed_artifacts": {
            "adapter": file_record(root_adapter),
            "config": file_record(root_config, json_payload=initial_config),
            "protocol": file_record(root_protocol, json_payload=initial_protocol),
            "training_summary": file_record(summary_path, json_payload=summary),
        },
        "objective": dict(OBJECTIVE_PROTOCOL),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json_atomic_exclusive(receipt, payload)
    return payload


def validate_existing_run_receipt(
    receipt_path: Path,
    *,
    expected_run_mode: str,
    expected_run_root: Path,
) -> dict[str, object]:
    receipt_path = receipt_path.expanduser().resolve()
    expected_run_root = expected_run_root.expanduser().resolve()
    require(receipt_path == expected_run_root / "run_audit_receipt.json", "run receipt path differs")
    receipt = load_json_object(receipt_path, description="run receipt")
    validate_checksum(receipt, field="receipt_sha256", description="run receipt")
    for field, expected in {
        "schema": RUN_RECEIPT_SCHEMA,
        "experiment": EXPERIMENT,
        "run_mode": expected_run_mode,
        "run_root": str(expected_run_root),
        "complete": True,
        "trainer_exit_code": 0,
        "tee_exit_code": 0,
        "training_processes_active": [],
        "hard32_only": True,
        "full170_authorized": False,
        "test_forbidden": True,
    }.items():
        require(receipt.get(field) == expected, f"run receipt differs: {field}")
    for field in ("auditor", "launch", "data_contract", "source_lock", "pair_manifest", "identity_pairing_manifest", "log"):
        _validate_file_record(receipt.get(field), description=f"run {field}")
    initial = receipt.get("initial_adapter")
    require(isinstance(initial, dict), "run initial-adapter records are missing")
    for field in ("manifest", "adapter", "config", "protocol"):
        _validate_file_record(initial.get(field), description=f"run initial {field}")
    checkpoints = receipt.get("checkpoints")
    spec = _stage_spec(expected_run_mode)
    require(
        isinstance(checkpoints, list)
        and [record.get("step") for record in checkpoints if isinstance(record, dict)]
        == list(spec.checkpoint_steps),
        "run checkpoint record set differs",
    )
    for record in checkpoints:
        require(isinstance(record, dict) and isinstance(record.get("step"), int), "run checkpoint record is invalid")
        _validate_file_record(record.get("receipt"), description="run checkpoint receipt")
        checkpoint_receipt = validate_existing_checkpoint_receipt(
            Path(str(record["receipt"]["path"])),
            expected_run_mode=expected_run_mode,
            expected_run_root=expected_run_root,
            expected_step=int(record["step"]),
        )
        require(
            record.get("history") == checkpoint_receipt.get("history")
            and record.get("adapter_change")
            == checkpoint_receipt.get("adapter_change"),
            "run checkpoint evidence copy differs",
        )
    completed = receipt.get("completed_artifacts")
    require(isinstance(completed, dict), "run completed artifacts are missing")
    for field in ("adapter", "config", "protocol", "training_summary"):
        _validate_file_record(completed.get(field), description=f"run completed {field}")
    require(receipt.get("objective") == dict(OBJECTIVE_PROTOCOL), "run objective differs")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("audit-prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--source-lock", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)

    checkpoint = subparsers.add_parser("audit-checkpoint")
    checkpoint.add_argument("--run-mode", choices=tuple(RUN_SPECS), required=True)
    checkpoint.add_argument("--run-root", type=Path, required=True)
    checkpoint.add_argument("--checkpoint-step", type=int, required=True)
    checkpoint.add_argument("--source-lock", type=Path, required=True)
    checkpoint.add_argument("--receipt", type=Path)

    watcher = subparsers.add_parser("watch-checkpoints")
    watcher.add_argument("--run-mode", choices=tuple(RUN_SPECS), required=True)
    watcher.add_argument("--run-root", type=Path, required=True)
    watcher.add_argument("--source-lock", type=Path, required=True)
    watcher.add_argument("--timeout-seconds", type=float, default=14400.0)
    watcher.add_argument("--poll-seconds", type=float, default=2.0)

    run = subparsers.add_parser("audit-run")
    run.add_argument("--run-mode", choices=tuple(RUN_SPECS), required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--log-file", type=Path, required=True)
    run.add_argument("--source-lock", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--trainer-exit-code", type=int, required=True)
    run.add_argument("--tee-exit-code", type=int, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "audit-prepare":
            payload = audit_prepare(
                run_root=args.run_root,
                source_lock=args.source_lock,
                receipt=args.receipt,
            )
            label = f"prepare_receipt={args.receipt}"
        elif args.action == "audit-checkpoint":
            payload = audit_checkpoint(
                run_mode=args.run_mode,
                run_root=args.run_root,
                checkpoint_step=args.checkpoint_step,
                source_lock=args.source_lock,
                receipt=args.receipt,
            )
            label = f"checkpoint_receipt_step={args.checkpoint_step}"
        elif args.action == "watch-checkpoints":
            payloads = watch_checkpoints(
                run_mode=args.run_mode,
                run_root=args.run_root,
                source_lock=args.source_lock,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
            print(f"checkpoint_watcher=complete steps={[item['checkpoint_step'] for item in payloads]}")
            return 0
        else:
            payload = audit_run(
                run_mode=args.run_mode,
                run_root=args.run_root,
                log_file=args.log_file,
                source_lock=args.source_lock,
                receipt=args.receipt,
                trainer_exit_code=args.trainer_exit_code,
                tee_exit_code=args.tee_exit_code,
            )
            label = f"run_receipt={args.receipt}"
    except (AuditError, OSError) as exc:
        print(f"ERROR: scene_memory_v6_run_audit_failed: {exc}", file=sys.stderr)
        return 2
    print(f"audit=valid {label} sha256={payload['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
