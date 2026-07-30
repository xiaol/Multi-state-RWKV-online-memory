#!/usr/bin/env python3
"""V12-native wrapper for the pinned V8 checkpoint-56 adapter warm start."""

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


WARM_START_MODE = "scene_memory_v12_v8_checkpoint56_adapter_only"
RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v12_adapter_warm_start_receipt.v1"
LOCK_SCHEMA = v9.LOCK_SCHEMA
DEFAULT_LOCK_PATH = v9.DEFAULT_LOCK_PATH
SOURCE_IMPORT_POLICY = v9.SOURCE_IMPORT_POLICY
TARGET_FRESH_START_POLICY = {
    **v9.TARGET_FRESH_START_POLICY,
    "rng_state": "fresh_from_v12_seed",
}
ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS = v9.ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS
WARM_START_LINEAGE_FILENAME = v9.WARM_START_LINEAGE_FILENAME
CONTINUATION_LINEAGE_FILENAME = v9.CONTINUATION_LINEAGE_FILENAME
ABLATION_LINEAGE_FILENAME = v9.ABLATION_LINEAGE_FILENAME


@dataclass(frozen=True)
class V12WarmStartContext:
    checkpoint: Path
    lock_path: Path
    lock: dict[str, Any]
    source_config: dict[str, Any]
    source_trainer_state: dict[str, Any]
    source_training_protocol: dict[str, Any]
    source_pairing_manifest: dict[str, Any]
    continuation_lineage: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class V12FreshStartContract:
    resume_from_checkpoint: str | Path | None
    initial_global_step: int
    optimizer_created: bool
    scheduler_created: bool
    trainer_state_imported: bool
    rng_state_imported: bool
    optim: str


def load_v12_warm_start_lock(
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    return v9.load_v9_warm_start_lock(lock_path)


def prepare_v12_v8_checkpoint56_warm_start(
    source_checkpoint: str | Path | None,
    *,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> V12WarmStartContext:
    source = v9.prepare_v9_v8_checkpoint56_warm_start(
        source_checkpoint,
        lock_path=lock_path,
    )
    return V12WarmStartContext(
        checkpoint=source.checkpoint,
        lock_path=source.lock_path,
        lock=source.lock,
        source_config=source.source_config,
        source_trainer_state=source.source_trainer_state,
        source_training_protocol=source.source_training_protocol,
        source_pairing_manifest=source.source_pairing_manifest,
        continuation_lineage=source.continuation_lineage,
    )


def validate_v12_fresh_start_contract(
    contract: V12FreshStartContract,
) -> dict[str, Any]:
    if contract.resume_from_checkpoint is not None:
        raise ValueError("V12 is fresh two-cycle training and forbids checkpoint resume")
    if isinstance(contract.initial_global_step, bool) or contract.initial_global_step != 0:
        raise ValueError("V12 adapter warm start must begin at global step 0")
    imported = {
        "optimizer": contract.optimizer_created,
        "scheduler": contract.scheduler_created,
        "trainer_state": contract.trainer_state_imported,
        "rng": contract.rng_state_imported,
    }
    forbidden = [name for name, value in imported.items() if value]
    if forbidden:
        raise ValueError(
            "V12 adapter warm start forbids preloaded training state: "
            + ", ".join(forbidden)
        )
    if contract.optim not in ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS:
        raise ValueError(
            "V12 adapter warm start requires a fresh AdamW optimizer created after load"
        )
    return {
        "initial_global_step": 0,
        "optimizer_implementation": contract.optim,
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v12_seed",
    }


def _load_pinned_adapter_state(
    context: V12WarmStartContext,
) -> dict[str, torch.Tensor]:
    payload = torch.load(
        context.checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Pinned V8 adapter must contain a state dictionary")
    for name, tensor in payload.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Pinned V8 adapter entry is invalid: {name}")
    return payload


def apply_v12_v8_checkpoint56_adapter_only_warm_start(
    model: nn.Module,
    context: V12WarmStartContext,
    *,
    fresh_start: V12FreshStartContract,
) -> dict[str, Any]:
    fresh_start_receipt = validate_v12_fresh_start_contract(fresh_start)
    source_state = _load_pinned_adapter_state(context)
    target_state = get_delta_mem_state_dict(model)
    topology_receipt = v9.validate_ordered_adapter_topology(
        source_state,
        target_state,
    )

    pinned_topology = context.lock.get("adapter_topology")
    if not isinstance(pinned_topology, dict):
        raise ValueError("Pinned V8 topology is missing")
    if not (
        topology_receipt["adapter_tensor_count"] == pinned_topology.get("tensor_count")
        and topology_receipt["adapter_tensor_elements"]
        == pinned_topology.get("tensor_elements")
        and topology_receipt["adapter_topology_sha256"]
        == pinned_topology.get("sha256")
    ):
        raise ValueError("Pinned V8 adapter topology differs from its lock")

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
            "V12 warm-start adapter tensors are not bit-equal after loading: "
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
        "source_state_imports": context.lock["source_state_imports"],
        "loaded_source_artifacts": ["delta_mem_adapter.pt"],
        "validated_not_imported_source_artifacts": [
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        ],
        "continuation_lineage": list(context.continuation_lineage),
        "topology": topology_receipt,
        "post_load_topology_sha256": post_topology["adapter_topology_sha256"],
        "post_load_bit_equal": True,
        "target_fresh_start": fresh_start_receipt,
    }
    receipt["receipt_sha256"] = v9.canonical_sha256(receipt)
    return receipt


def write_v12_warm_start_receipt(
    receipt: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    payload = dict(receipt)
    recorded_sha256 = payload.pop("receipt_sha256", None)
    if recorded_sha256 != v9.canonical_sha256(payload):
        raise ValueError("V12 warm-start receipt self-hash differs")
    payload["receipt_sha256"] = recorded_sha256
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


__all__ = (
    "ABLATION_LINEAGE_FILENAME",
    "ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS",
    "CONTINUATION_LINEAGE_FILENAME",
    "DEFAULT_LOCK_PATH",
    "LOCK_SCHEMA",
    "RECEIPT_SCHEMA",
    "SOURCE_IMPORT_POLICY",
    "TARGET_FRESH_START_POLICY",
    "V12FreshStartContract",
    "V12WarmStartContext",
    "WARM_START_LINEAGE_FILENAME",
    "WARM_START_MODE",
    "apply_v12_v8_checkpoint56_adapter_only_warm_start",
    "load_v12_warm_start_lock",
    "prepare_v12_v8_checkpoint56_warm_start",
    "validate_v12_fresh_start_contract",
    "write_v12_warm_start_receipt",
)
