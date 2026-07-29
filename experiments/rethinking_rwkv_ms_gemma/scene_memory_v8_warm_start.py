#!/usr/bin/env python3
"""Pinned adapter-only warm start from the completed V7 Train32 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from deltamem.core.delta_impl import (
    get_delta_mem_state_dict,
    load_delta_mem_state_dict,
)


LOCK_SCHEMA = "rwkv_ms_scene_memory_v8_v7_warm_start_lock.v1"
RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v8_adapter_warm_start_receipt.v1"
WARM_START_MODE = "scene_memory_v8_v7_checkpoint256_adapter_only"
DEFAULT_LOCK_PATH = Path(__file__).with_name(
    "scene_memory_v8_v7_checkpoint256_lock.json"
)
REQUIRED_SOURCE_ARTIFACTS = (
    "continuation_manifest.json",
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "rng_state.pth",
    "scene_state_identity_pairing_manifest.json",
    "scheduler.pt",
    "trainer_state.json",
    "training_protocol.json",
)
ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS = (
    "adamw_torch",
    "adamw_torch_fused",
)


@dataclass(frozen=True)
class V8WarmStartContext:
    checkpoint: Path
    lock_path: Path
    lock: dict[str, Any]
    source_config: dict[str, Any]
    source_trainer_state: dict[str, Any]
    source_training_protocol: dict[str, Any]


@dataclass(frozen=True)
class V8FreshStartContract:
    resume_from_checkpoint: str | Path | None
    initial_global_step: int
    optimizer_created: bool
    scheduler_created: bool
    trainer_state_imported: bool
    rng_state_imported: bool
    optim: str


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        key for key, expected_value in expected.items() if actual.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Pinned V7 {description} differs for: " + ", ".join(mismatches)
        )


def load_v8_warm_start_lock(
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    resolved = Path(lock_path).expanduser().resolve()
    _require_regular_file(resolved, description="V8 warm-start lock")
    lock = _load_json_object(resolved, description="V8 warm-start lock")
    _require(lock.get("schema") == LOCK_SCHEMA, "V8 warm-start lock schema differs")
    unsigned = dict(lock)
    recorded_sha256 = unsigned.pop("lock_sha256", None)
    _require(
        recorded_sha256 == canonical_sha256(unsigned),
        "V8 warm-start lock self-hash differs",
    )
    artifacts = lock.get("artifacts")
    _require(isinstance(artifacts, dict), "V8 warm-start artifact bindings are missing")
    _require(
        tuple(artifacts) == REQUIRED_SOURCE_ARTIFACTS,
        "V8 warm-start artifact names or order differ",
    )
    import_policy = lock.get("source_state_imports")
    _require(
        import_policy
        == {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
            "global_step": False,
        },
        "V8 warm-start import policy differs",
    )
    _require(
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
        "V8 target fresh-start policy differs",
    )
    return lock


def prepare_v8_v7_checkpoint256_warm_start(
    source_checkpoint: str | Path | None,
    *,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
) -> V8WarmStartContext:
    if source_checkpoint is None or not str(source_checkpoint).strip():
        raise ValueError("V8 warm start requires an explicit V7 checkpoint-256 path")
    if str(source_checkpoint).strip().lower() in {"latest", "last", "auto"}:
        raise ValueError("V8 warm start forbids implicit checkpoint selection")

    requested = Path(source_checkpoint).expanduser()
    if requested.is_symlink():
        raise ValueError("V8 warm-start checkpoint directory must not be a symlink")
    checkpoint = requested.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"V8 warm-start checkpoint does not exist: {checkpoint}")

    resolved_lock_path = Path(lock_path).expanduser().resolve()
    lock = load_v8_warm_start_lock(resolved_lock_path)
    pinned_checkpoint = Path(str(lock.get("source_checkpoint"))).expanduser().resolve()
    _require(
        checkpoint == pinned_checkpoint,
        "V8 warm start source is not the specifically pinned V7 checkpoint",
    )
    _require(
        checkpoint.name == "checkpoint-256",
        "V8 warm start source must be checkpoint-256",
    )

    artifacts = lock["artifacts"]
    for filename in REQUIRED_SOURCE_ARTIFACTS:
        binding = artifacts.get(filename)
        _require(isinstance(binding, dict), f"V8 source binding is invalid: {filename}")
        artifact = checkpoint / filename
        _require_regular_file(artifact, description=f"V8 source artifact {filename}")
        _require(
            artifact.stat().st_size == binding.get("bytes"),
            f"Pinned V7 artifact byte size differs: {filename}",
        )
        _require(
            sha256_file(artifact) == binding.get("sha256"),
            f"Pinned V7 artifact SHA-256 differs: {filename}",
        )

    source_config = _load_json_object(
        checkpoint / "delta_mem_config.json",
        description="pinned V7 Delta-Mem config",
    )
    expected_config = lock.get("expected_delta_config")
    _require(isinstance(expected_config, dict), "Pinned V7 config expectations are missing")
    _validate_expected_fields(
        source_config,
        expected_config,
        description="Delta-Mem config",
    )
    _require(
        source_config.get("target_layers") == list(range(42)),
        "Pinned V7 adapter must target ordered layers 0-41",
    )
    _require(
        source_config.get("delta_heads") == ["q", "o"],
        "Pinned V7 adapter must use exactly Q+O delta heads",
    )

    trainer_state = _load_json_object(
        checkpoint / "trainer_state.json",
        description="pinned V7 trainer state",
    )
    expected_trainer_state = lock.get("expected_trainer_state")
    _require(
        isinstance(expected_trainer_state, dict),
        "Pinned V7 trainer-state expectations are missing",
    )
    _validate_expected_fields(
        trainer_state,
        expected_trainer_state,
        description="trainer state",
    )

    training_protocol = _load_json_object(
        checkpoint / "training_protocol.json",
        description="pinned V7 training protocol",
    )
    expected_protocol = lock.get("expected_training_protocol")
    _require(
        isinstance(expected_protocol, dict),
        "Pinned V7 training-protocol expectations are missing",
    )
    _validate_expected_fields(
        training_protocol,
        expected_protocol,
        description="training protocol",
    )

    pairing = _load_json_object(
        checkpoint / "scene_state_identity_pairing_manifest.json",
        description="pinned V7 pairing manifest",
    )
    _require(
        pairing.get("manifest_sha256") == lock.get("source_pairing_manifest_sha256"),
        "Pinned V7 pairing-manifest identity differs",
    )
    protocol_pairing = training_protocol.get("scene_state_identity_pairing")
    _require(
        isinstance(protocol_pairing, dict)
        and protocol_pairing.get("manifest_sha256") == pairing.get("manifest_sha256"),
        "Pinned V7 protocol and pairing manifest differ",
    )

    return V8WarmStartContext(
        checkpoint=checkpoint,
        lock_path=resolved_lock_path,
        lock=lock,
        source_config=source_config,
        source_trainer_state=trainer_state,
        source_training_protocol=training_protocol,
    )


def ordered_adapter_topology(
    state: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], str, int]:
    topology: list[dict[str, Any]] = []
    tensor_elements = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Adapter parameter names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Adapter entry is not a tensor: {name}")
        topology.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        tensor_elements += tensor.numel()
    _require(bool(topology), "Adapter state dictionary must not be empty")
    topology_sha256 = hashlib.sha256(
        json.dumps(
            topology,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return topology, topology_sha256, tensor_elements


def validate_ordered_adapter_topology(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    source_names = list(source_state)
    target_names = list(target_state)
    if source_names != target_names:
        raise ValueError(
            "V8 warm start requires identical ordered adapter parameter names"
        )
    for name in source_names:
        source_tensor = source_state[name]
        target_tensor = target_state[name]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(
            target_tensor, torch.Tensor
        ):
            raise ValueError(f"V8 warm-start adapter entry is not a tensor: {name}")
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"V8 warm-start adapter shape differs for {name}: "
                f"source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"V8 warm-start adapter dtype differs for {name}: "
                f"source={source_tensor.dtype} target={target_tensor.dtype}"
            )
    _, source_sha256, source_elements = ordered_adapter_topology(source_state)
    _, target_sha256, target_elements = ordered_adapter_topology(target_state)
    _require(
        source_sha256 == target_sha256 and source_elements == target_elements,
        "V8 warm-start adapter topology digest differs",
    )
    return {
        "ordered_parameter_names_equal": True,
        "ordered_shapes_equal": True,
        "ordered_dtypes_equal": True,
        "adapter_tensor_count": len(source_names),
        "adapter_tensor_elements": source_elements,
        "adapter_topology_sha256": source_sha256,
    }


def validate_v8_fresh_start_contract(
    contract: V8FreshStartContract,
) -> dict[str, Any]:
    if contract.resume_from_checkpoint is not None:
        raise ValueError("V8 adapter warm start cannot import checkpoint resume state")
    if isinstance(contract.initial_global_step, bool) or contract.initial_global_step != 0:
        raise ValueError("V8 adapter warm start must begin at global step 0")
    imported = {
        "optimizer": contract.optimizer_created,
        "scheduler": contract.scheduler_created,
        "trainer_state": contract.trainer_state_imported,
        "rng": contract.rng_state_imported,
    }
    forbidden = [name for name, value in imported.items() if value]
    if forbidden:
        raise ValueError(
            "V8 adapter warm start forbids preloaded training state: "
            + ", ".join(forbidden)
        )
    if contract.optim not in ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS:
        raise ValueError(
            "V8 adapter warm start requires a fresh AdamW optimizer created after load"
        )
    return {
        "initial_global_step": 0,
        "optimizer_implementation": contract.optim,
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v8_seed",
    }


def _load_pinned_adapter_state(context: V8WarmStartContext) -> dict[str, torch.Tensor]:
    adapter_path = context.checkpoint / "delta_mem_adapter.pt"
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Pinned V7 adapter must contain a state dictionary")
    for name, tensor in payload.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Pinned V7 adapter entry is invalid: {name}")
    return payload


def apply_v8_v7_checkpoint256_adapter_only_warm_start(
    model: nn.Module,
    context: V8WarmStartContext,
    *,
    fresh_start: V8FreshStartContract,
) -> dict[str, Any]:
    fresh_start_receipt = validate_v8_fresh_start_contract(fresh_start)
    source_state = _load_pinned_adapter_state(context)
    target_state = get_delta_mem_state_dict(model)
    topology_receipt = validate_ordered_adapter_topology(source_state, target_state)

    pinned_topology = context.lock.get("adapter_topology")
    _require(isinstance(pinned_topology, dict), "Pinned V7 topology is missing")
    _require(
        topology_receipt["adapter_tensor_count"] == pinned_topology.get("tensor_count")
        and topology_receipt["adapter_tensor_elements"]
        == pinned_topology.get("tensor_elements")
        and topology_receipt["adapter_topology_sha256"]
        == pinned_topology.get("sha256"),
        "Pinned V7 adapter topology differs from its lock",
    )

    load_delta_mem_state_dict(model, source_state)
    loaded_state = get_delta_mem_state_dict(model)
    post_topology = validate_ordered_adapter_topology(source_state, loaded_state)
    unequal = [
        name
        for name, source_tensor in source_state.items()
        if not torch.equal(loaded_state[name], source_tensor)
    ]
    if unequal:
        raise ValueError(
            "V8 warm-start adapter tensors are not bit-equal after loading: "
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
        "source_state_imports": context.lock["source_state_imports"],
        "topology": topology_receipt,
        "post_load_topology_sha256": post_topology["adapter_topology_sha256"],
        "post_load_bit_equal": True,
        "target_fresh_start": fresh_start_receipt,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_v8_warm_start_receipt(
    receipt: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    payload = dict(receipt)
    recorded_sha256 = payload.pop("receipt_sha256", None)
    _require(
        recorded_sha256 == canonical_sha256(payload),
        "V8 warm-start receipt self-hash differs",
    )
    payload["receipt_sha256"] = recorded_sha256
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


__all__ = [
    "ALLOWED_FRESH_ADAMW_IMPLEMENTATIONS",
    "DEFAULT_LOCK_PATH",
    "LOCK_SCHEMA",
    "RECEIPT_SCHEMA",
    "REQUIRED_SOURCE_ARTIFACTS",
    "V8FreshStartContract",
    "V8WarmStartContext",
    "WARM_START_MODE",
    "apply_v8_v7_checkpoint256_adapter_only_warm_start",
    "canonical_sha256",
    "load_v8_warm_start_lock",
    "ordered_adapter_topology",
    "prepare_v8_v7_checkpoint256_warm_start",
    "sha256_file",
    "validate_ordered_adapter_topology",
    "validate_v8_fresh_start_contract",
    "write_v8_warm_start_receipt",
]
