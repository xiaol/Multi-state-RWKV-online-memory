#!/usr/bin/env python3
"""Pinned adapter-only V14 warm start from completed V13 checkpoint-4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from deltamem.core.delta_impl import (
    get_delta_mem_state_dict,
    load_delta_mem_state_dict,
)
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v9_warm_start as v9


PINNED_SOURCE_CHECKPOINT = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory/"
    "scene_memory_v13/"
    "scene_memory_v13_production_value14_dense_20260731_070142_step4/"
    "trainer/checkpoint-4"
)
PINNED_ADAPTER_SHA256 = (
    "9e39522b179067ae6c76076aa1b5aa0b4ceaf40fd97daf6abc7edeef2a1783eb"
)
PINNED_CONFIG_SHA256 = (
    "86e1bbb1a258cb591c9167935317a2374cf9edc79e5a3a77b0f903dc38483931"
)
PINNED_TRAINER_STATE_SHA256 = (
    "ced5bc22624f073ef409fba8db1d6f78da564e765cde309de47b5d09b7e92888"
)
PINNED_PROTOCOL_SHA256 = (
    "00abbed2107253779cb07a403b9c7bebceca5d59b7a4705b32f8540d623039eb"
)
PINNED_PAIRING_SHA256 = (
    "52572b354dc6344a0ceab37251a6301dac7ed2f8c595760948eef042e3f6c11b"
)
PINNED_LINEAGE_SHA256 = (
    "837327373e502c28ad40fae5a5aabc46accd47394952825786864e41268adfd7"
)
LOCK_SCHEMA = (
    "rwkv_ms_scene_memory_v14_v13_checkpoint4_adapter_warm_start_lock.v1"
)
RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v14_adapter_warm_start_receipt.v1"
WARM_START_MODE = "scene_memory_v14_v13_checkpoint4_adapter_only"
DEFAULT_LOCK_PATH = Path(__file__).with_name(
    "scene_memory_v14_v13_checkpoint4_lock.json"
)
REQUIRED_SOURCE_ARTIFACTS = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "rng_state.pth",
    "scene_memory_v13_row_objective.json",
    "scene_state_identity_pairing_manifest.json",
    "scheduler.pt",
    "trainer_state.json",
    "training_protocol.json",
    "warm_start_lineage_manifest.json",
)
SOURCE_IMPORT_POLICY = {
    "adapter": True,
    "optimizer": False,
    "scheduler": False,
    "trainer_state": False,
    "rng": False,
    "global_step": False,
}
TARGET_FRESH_START_POLICY = {
    "global_step": 0,
    "optimizer_created_after_adapter_load": True,
    "optimizer_family": "AdamW",
    "optimizer_state": "fresh",
    "scheduler_state": "fresh",
    "trainer_state": "fresh",
    "rng_state": "fresh_from_v14_seed",
}
ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS = v9.ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS
WARM_START_LINEAGE_FILENAME = v9.WARM_START_LINEAGE_FILENAME


@dataclass(frozen=True)
class V14WarmStartContext:
    checkpoint: Path
    lock_path: Path
    lock: dict[str, Any]
    source_config: dict[str, Any]
    source_trainer_state: dict[str, Any]
    source_training_protocol: dict[str, Any]
    source_pairing_manifest: dict[str, Any]
    source_row_objective_audit: dict[str, Any]
    source_warm_start_lineage: dict[str, Any]


@dataclass(frozen=True)
class V14FreshStartContract:
    resume_from_checkpoint: str | Path | None
    initial_global_step: int
    optimizer_created: bool
    scheduler_created: bool
    trainer_state_imported: bool
    rng_state_imported: bool
    optim: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _require_regular_file(path: Path, *, description: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"{description} is missing, empty, or a symlink: {path}")


def _validate_expected_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    description: str,
) -> None:
    mismatches = [
        key
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Pinned V13 {description} differs for: " + ", ".join(mismatches)
        )


def _validate_artifact_binding(
    checkpoint: Path,
    filename: str,
    binding: Mapping[str, Any],
) -> None:
    path = checkpoint / filename
    _require_regular_file(path, description=f"Pinned V13 artifact {filename}")
    expected_bytes = binding.get("bytes")
    _require(
        isinstance(expected_bytes, int)
        and not isinstance(expected_bytes, bool)
        and expected_bytes > 0,
        f"Pinned V13 artifact {filename} byte binding is invalid",
    )
    _require(
        path.stat().st_size == expected_bytes,
        f"Pinned V13 artifact {filename} byte size differs",
    )
    expected_sha256 = binding.get("sha256")
    _require(
        isinstance(expected_sha256, str) and len(expected_sha256) == 64,
        f"Pinned V13 artifact {filename} SHA-256 binding is invalid",
    )
    _require(
        v9.sha256_file(path) == expected_sha256,
        f"Pinned V13 artifact {filename} SHA-256 differs",
    )


def _load_v14_warm_start_lock(
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(lock_path).expanduser().resolve()
    _require_regular_file(resolved, description="V14 warm-start lock")
    lock = _load_json_object(resolved, description="V14 warm-start lock")
    _require(lock.get("schema") == LOCK_SCHEMA, "V14 warm-start lock schema differs")
    unsigned_lock = dict(lock)
    recorded_hash = unsigned_lock.pop("lock_sha256", None)
    _require(
        recorded_hash == v9.canonical_sha256(unsigned_lock),
        "V14 warm-start lock self-hash differs",
    )
    _require(
        lock.get("source_checkpoint") == str(PINNED_SOURCE_CHECKPOINT),
        "V14 warm-start lock source checkpoint differs",
    )
    artifacts = lock.get("artifacts")
    _require(isinstance(artifacts, dict), "V14 source artifact bindings are missing")
    _require(
        tuple(artifacts) == REQUIRED_SOURCE_ARTIFACTS,
        "V14 source artifact names or order differ",
    )
    pinned_hashes = {
        "delta_mem_adapter.pt": PINNED_ADAPTER_SHA256,
        "delta_mem_config.json": PINNED_CONFIG_SHA256,
        "trainer_state.json": PINNED_TRAINER_STATE_SHA256,
        "training_protocol.json": PINNED_PROTOCOL_SHA256,
        "scene_state_identity_pairing_manifest.json": PINNED_PAIRING_SHA256,
        "warm_start_lineage_manifest.json": PINNED_LINEAGE_SHA256,
    }
    _require(
        all(
            isinstance(artifacts.get(name), dict)
            and artifacts[name].get("sha256") == expected
            for name, expected in pinned_hashes.items()
        ),
        "V14 pinned source hashes differ",
    )
    _require(
        lock.get("source_state_imports") == SOURCE_IMPORT_POLICY,
        "V14 source import policy differs",
    )
    _require(
        lock.get("target_fresh_start") == TARGET_FRESH_START_POLICY,
        "V14 target fresh-start policy differs",
    )
    return resolved, lock


def _validate_source_metadata(
    checkpoint: Path,
    lock: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config = _load_json_object(
        checkpoint / "delta_mem_config.json",
        description="pinned V13 Delta-Mem config",
    )
    trainer_state = _load_json_object(
        checkpoint / "trainer_state.json",
        description="pinned V13 trainer state",
    )
    protocol = _load_json_object(
        checkpoint / "training_protocol.json",
        description="pinned V13 training protocol",
    )
    pairing = _load_json_object(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        description="pinned V13 pairing manifest",
    )
    row_audit = _load_json_object(
        checkpoint / "scene_memory_v13_row_objective.json",
        description="pinned V13 row-objective audit",
    )
    lineage = _load_json_object(
        checkpoint / "warm_start_lineage_manifest.json",
        description="pinned V13 warm-start lineage",
    )

    for key, actual, description in (
        ("expected_delta_config", config, "Delta-Mem config"),
        ("expected_trainer_state", trainer_state, "trainer state"),
        ("expected_training_protocol", protocol, "training protocol"),
        ("expected_pairing_manifest", pairing, "pairing manifest"),
    ):
        expected = lock.get(key)
        _require(isinstance(expected, dict), f"V14 lock omits {key}")
        _validate_expected_fields(actual, expected, description=description)

    _require(
        config.get("target_layers") == list(range(42))
        and config.get("delta_heads") == ["q", "o"],
        "Pinned V13 adapter topology config differs",
    )
    pairing_unsigned = dict(pairing)
    pairing_hash = pairing_unsigned.pop("manifest_sha256", None)
    _require(
        pairing_hash == v9.canonical_sha256(pairing_unsigned),
        "Pinned V13 pairing-manifest self-hash differs",
    )
    protocol_pairing = protocol.get("scene_state_identity_pairing")
    _require(
        isinstance(protocol_pairing, dict)
        and protocol_pairing.get("manifest_sha256") == pairing_hash,
        "Pinned V13 protocol and pairing manifest differ",
    )

    expected_audit = lock.get("expected_row_objective_audit")
    _require(isinstance(expected_audit, dict), "V14 lock omits V13 row audit")
    derived_audit = {
        **row_audit,
        "row_count": len(row_audit.get("rows", [])),
        "pair_presentation_count": len(row_audit.get("pair_presentations", [])),
    }
    _validate_expected_fields(
        derived_audit,
        expected_audit,
        description="row-objective audit",
    )

    lineage_unsigned = dict(lineage)
    lineage_hash = lineage_unsigned.pop("receipt_sha256", None)
    _require(
        lineage_hash == v9.canonical_sha256(lineage_unsigned),
        "Pinned V13 warm-start lineage self-hash differs",
    )
    lineage_fresh_start = lineage.get("target_fresh_start")
    _require(
        lineage.get("schema")
        == "rwkv_ms_scene_memory_v13_adapter_warm_start_receipt.v1"
        and lineage.get("mode") == "scene_memory_v13_v8_checkpoint56_adapter_only"
        and lineage.get("source_state_imports") == SOURCE_IMPORT_POLICY
        and lineage.get("post_load_bit_equal") is True
        and lineage.get("trainer_resume_from_checkpoint") is None
        and lineage.get("target_initial_global_step") == 0
        and lineage.get("pre_train_global_step") == 0
        and lineage.get("fresh_optimizer_created") is True
        and lineage.get("fresh_optimizer_state_entries_before_train") == 0
        and lineage.get("fresh_scheduler_created_before_train") is False
        and isinstance(lineage_fresh_start, dict)
        and lineage_fresh_start.get("initial_global_step") == 0
        and lineage_fresh_start.get("optimizer_state") == "fresh"
        and lineage_fresh_start.get("scheduler_state") == "fresh"
        and lineage_fresh_start.get("trainer_state") == "fresh"
        and lineage_fresh_start.get("rng_state") == "fresh_from_v13_seed",
        "Pinned V13 warm-start lineage evidence differs",
    )
    _require(
        lineage.get("target_delta_config_sha256") == v9.canonical_sha256(config)
        and lineage.get("target_training_protocol_sha256")
        == v9.canonical_sha256(protocol)
        and lineage.get("target_scene_state_pairing_manifest_sha256") == pairing_hash,
        "Pinned V13 warm-start lineage target identities differ",
    )
    return config, trainer_state, protocol, pairing, row_audit, lineage


def prepare_v14_v13_checkpoint4_warm_start(
    source_checkpoint: str | Path | None,
    *,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> V14WarmStartContext:
    if source_checkpoint is None or not str(source_checkpoint).strip():
        raise ValueError("V14 warm start requires an explicit V13 checkpoint-4 path")
    if str(source_checkpoint).strip().lower() in {"latest", "last", "auto"}:
        raise ValueError("V14 warm start forbids implicit checkpoint selection")
    requested = Path(source_checkpoint).expanduser()
    if requested.is_symlink():
        raise ValueError("V14 warm-start checkpoint directory must not be a symlink")
    checkpoint = requested.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"V14 warm-start checkpoint does not exist: {checkpoint}")
    _require(
        checkpoint == PINNED_SOURCE_CHECKPOINT,
        "V14 warm start source is not the specifically pinned V13 checkpoint",
    )
    _require(checkpoint.name == "checkpoint-4", "V14 source must be checkpoint-4")

    resolved_lock_path, lock = _load_v14_warm_start_lock(lock_path)
    artifacts = lock["artifacts"]
    for filename in REQUIRED_SOURCE_ARTIFACTS:
        binding = artifacts[filename]
        _require(
            isinstance(binding, dict),
            f"V14 source binding is invalid: {filename}",
        )
        _validate_artifact_binding(checkpoint, filename, binding)
    config, trainer_state, protocol, pairing, row_audit, lineage = (
        _validate_source_metadata(checkpoint, lock)
    )
    return V14WarmStartContext(
        checkpoint=checkpoint,
        lock_path=resolved_lock_path,
        lock=lock,
        source_config=config,
        source_trainer_state=trainer_state,
        source_training_protocol=protocol,
        source_pairing_manifest=pairing,
        source_row_objective_audit=row_audit,
        source_warm_start_lineage=lineage,
    )


def validate_v14_fresh_start_contract(
    contract: V14FreshStartContract,
) -> dict[str, Any]:
    if contract.resume_from_checkpoint is not None:
        raise ValueError("V14 is fresh training and forbids checkpoint resume")
    if isinstance(contract.initial_global_step, bool) or contract.initial_global_step != 0:
        raise ValueError("V14 adapter warm start must begin at global step 0")
    imported = {
        "optimizer": contract.optimizer_created,
        "scheduler": contract.scheduler_created,
        "trainer_state": contract.trainer_state_imported,
        "rng": contract.rng_state_imported,
    }
    forbidden = [name for name, value in imported.items() if value]
    if forbidden:
        raise ValueError(
            "V14 adapter warm start forbids preloaded training state: "
            + ", ".join(forbidden)
        )
    if contract.optim not in ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS:
        raise ValueError(
            "V14 adapter warm start requires a fresh AdamW optimizer created after load"
        )
    return {
        "initial_global_step": 0,
        "optimizer_implementation": contract.optim,
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v14_seed",
    }


def _load_pinned_adapter_state(
    context: V14WarmStartContext,
) -> dict[str, torch.Tensor]:
    payload = torch.load(
        context.checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Pinned V13 adapter must contain a state dictionary")
    for name, tensor in payload.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Pinned V13 adapter entry is invalid: {name}")
    return payload


def apply_v14_v13_checkpoint4_adapter_only_warm_start(
    model: nn.Module,
    context: V14WarmStartContext,
    *,
    fresh_start: V14FreshStartContract,
) -> dict[str, Any]:
    fresh_start_receipt = validate_v14_fresh_start_contract(fresh_start)
    source_state = _load_pinned_adapter_state(context)
    target_state = get_delta_mem_state_dict(model)
    topology_receipt = v9.validate_ordered_adapter_topology(
        source_state,
        target_state,
    )
    pinned_topology = context.lock.get("adapter_topology")
    _require(isinstance(pinned_topology, dict), "Pinned V13 topology is missing")
    _require(
        topology_receipt["adapter_tensor_count"] == pinned_topology.get("tensor_count")
        and topology_receipt["adapter_tensor_elements"]
        == pinned_topology.get("tensor_elements")
        and topology_receipt["adapter_topology_sha256"]
        == pinned_topology.get("sha256"),
        "Pinned V13 adapter topology differs from its lock",
    )

    load_delta_mem_state_dict(model, source_state)
    loaded_state = get_delta_mem_state_dict(model)
    post_topology = v9.validate_ordered_adapter_topology(source_state, loaded_state)
    unequal = [
        name
        for name, source_tensor in source_state.items()
        if not torch.equal(loaded_state[name], source_tensor)
    ]
    if unequal:
        raise ValueError(
            "V14 warm-start adapter tensors are not bit-equal after loading: "
            + ", ".join(unequal[:8])
        )

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "mode": WARM_START_MODE,
        "source_checkpoint": str(context.checkpoint),
        "source_lock": {
            "path": str(context.lock_path),
            "lock_sha256": context.lock["lock_sha256"],
        },
        "source_artifacts": context.lock["artifacts"],
        "source_global_step": context.source_trainer_state["global_step"],
        "source_epoch": context.source_trainer_state["epoch"],
        "source_protocol_objective_version": context.source_training_protocol[
            "memory_objective_version"
        ],
        "source_pairing_objective_version": context.source_pairing_manifest[
            "objective_version"
        ],
        "source_row_objective_audit_schema": context.source_row_objective_audit[
            "schema"
        ],
        "source_v13_warm_start_receipt_sha256": context.source_warm_start_lineage[
            "receipt_sha256"
        ],
        "source_state_imports": context.lock["source_state_imports"],
        "loaded_source_artifacts": ["delta_mem_adapter.pt"],
        "validated_not_imported_source_artifacts": [
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ],
        "topology": topology_receipt,
        "post_load_topology_sha256": post_topology["adapter_topology_sha256"],
        "post_load_bit_equal": True,
        "target_fresh_start": fresh_start_receipt,
    }
    receipt["receipt_sha256"] = v9.canonical_sha256(receipt)
    return receipt


def write_v14_warm_start_receipt(
    receipt: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    payload = dict(receipt)
    recorded_sha256 = payload.pop("receipt_sha256", None)
    _require(
        recorded_sha256 == v9.canonical_sha256(payload),
        "V14 warm-start receipt self-hash differs",
    )
    payload["receipt_sha256"] = recorded_sha256
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


__all__ = (
    "ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS",
    "DEFAULT_LOCK_PATH",
    "LOCK_SCHEMA",
    "PINNED_ADAPTER_SHA256",
    "PINNED_SOURCE_CHECKPOINT",
    "RECEIPT_SCHEMA",
    "REQUIRED_SOURCE_ARTIFACTS",
    "SOURCE_IMPORT_POLICY",
    "TARGET_FRESH_START_POLICY",
    "V14FreshStartContract",
    "V14WarmStartContext",
    "WARM_START_MODE",
    "WARM_START_LINEAGE_FILENAME",
    "apply_v14_v13_checkpoint4_adapter_only_warm_start",
    "prepare_v14_v13_checkpoint4_warm_start",
    "validate_v14_fresh_start_contract",
    "write_v14_warm_start_receipt",
)
