#!/usr/bin/env python3
"""Evaluate the locked projected-KV associative checkpoint trajectory.

This evaluator is synthetic-only. It consumes the four-row train canary and
never resolves protected, Hard32, validation, or test data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.train.delta_sft_experimental import _disable_training_cache
from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_associative_retrieval_canary as canary,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_associative_retrieval_gate0 as gate0,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_associative_retrieval_preflight as preflight,
)


RESULT_SCHEMA = "rwkv_ms_synthetic_associative_retrieval_trajectory_eval.v2"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
SEED = 42
CHECKPOINT_STEPS = (8, 16, 32, 64)
CONDITIONS = ("correct", "donor", "wrong_slot", "no_write")
DONOR_INDICES = (1, 0, 3, 2)
EXPECTED_QUERY_SLOTS = (0, 0, 1, 1)
TARGET_LAYERS = tuple(range(42))
CHECKPOINT_ARTIFACT_NAMES = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "rng_state.pth",
    "scene_state_identity_pairing_manifest.json",
    "scheduler.pt",
    "trainer_state.json",
    "training_protocol.json",
)
IDENTICAL_WRITE_ROW_PAIRS = ((0, 2), (1, 3))
QUERY_SEPARATION_ROW_PAIRS = IDENTICAL_WRITE_ROW_PAIRS
SAME_QUERY_ROW_PAIRS = ((0, 1), (2, 3))

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "local_artifacts/synthetic_associative_retrieval_canary_v2"
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE_DIR / "source_manifest.json"
DEFAULT_GATE0_RECEIPT = DEFAULT_SOURCE_DIR / "gate0_receipt.json"
DEFAULT_PREFLIGHT_RECEIPT = DEFAULT_SOURCE_DIR / "projected_kv_preflight_receipt.json"
DEFAULT_RUN_DIR = (
    SCRIPT_DIR
    / "local_artifacts/synthetic_associative_retrieval_runs"
    / "synthetic_associative_projected_kv_s2_k32_t16_u1_b4_lr2e4_seed42"
)


@dataclass(frozen=True)
class CheckpointBinding:
    step: int
    path: Path
    config: HFDeltaMemConfig
    artifacts_before: dict[str, dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{description} is not a regular file: {resolved}")
    return resolved


def _regular_directory(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{description} is not a regular directory: {resolved}")
    return resolved


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    regular = _regular_file(path, description=description)
    try:
        value = json.loads(regular.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {regular}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {regular}")
    return value


def _file_record(path: Path, *, description: str) -> dict[str, Any]:
    regular = _regular_file(path, description=description)
    return {
        "path": str(regular),
        "bytes": regular.stat().st_size,
        "sha256": canary.sha256_file(regular),
    }


def snapshot_artifacts(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash a named set of regular files for later immutability verification."""

    return {
        name: _file_record(path, description=f"checkpoint artifact {name}")
        for name, path in files.items()
    }


def verify_artifact_snapshot(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Rehash every snapshotted file and reject any path, size, or hash drift."""

    verified: dict[str, dict[str, Any]] = {}
    for name, expected in snapshot.items():
        path = Path(str(expected.get("path", "")))
        actual = _file_record(path, description=f"checkpoint artifact {name}")
        if actual != dict(expected):
            raise ValueError(f"Checkpoint artifact changed during evaluation: {name}")
        verified[name] = actual
    return verified


def write_json_atomic_fresh(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish JSON atomically without ever replacing an existing output."""

    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Trajectory output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise ValueError(f"Trajectory output must be fresh: {output}") from exc
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_path": str(source["manifest_path"]),
        "manifest_file_sha256": source["manifest_file_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "train_path": str(source["train_path"]),
        "train_sha256": source["train_sha256"],
        "rows_path": str(source["rows_path"]),
        "rows_sha256": source["rows_sha256"],
        "row_count": len(source["rows"]),
    }


def validate_pairing_manifest(
    pairing: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(pairing)
    declared_hash = unsigned.pop("manifest_sha256", None)
    actual_hash = canonical_sha256(unsigned)
    _require(
        declared_hash == actual_hash,
        "Pairing-manifest canonical SHA-256 differs",
    )
    _require(pairing.get("pairing_locked") is True, "Pairing manifest is not locked")
    _require(
        pairing.get("locked_donor_indices") == list(DONOR_INDICES),
        "Pairing donor indices differ",
    )
    _require(pairing.get("data_seed") == SEED, "Pairing data seed differs")
    _require(
        pairing.get("objective_version") == "scene_state_identity_ce_v2",
        "Pairing objective differs",
    )
    splits = pairing.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    _require(
        isinstance(splits, dict) and set(splits) == {"train"},
        "Pairing manifest must contain only train",
    )
    _require(isinstance(train, dict), "Pairing train split is absent")
    assert isinstance(train, dict)
    _require(train.get("sample_count") == 4, "Pairing train sample count differs")
    _require(train.get("pairing_locked") is True, "Pairing train split is not locked")
    _require(
        train.get("locked_donor_indices") == list(DONOR_INDICES),
        "Pairing train donor indices differ",
    )
    pairs = train.get("pairs")
    _require(
        isinstance(pairs, list) and len(pairs) == 4,
        "Pairing audit must contain four directional rows",
    )
    assert isinstance(pairs, list)
    row_records = source["row_records"]
    for source_index, (pair, donor_index) in enumerate(
        zip(pairs, DONOR_INDICES, strict=True)
    ):
        _require(
            isinstance(pair, dict),
            f"Pairing row {source_index} is not an object",
        )
        assert isinstance(pair, dict)
        target_metadata = row_records[source_index]["token_metadata"]
        donor_metadata = row_records[donor_index]["token_metadata"]
        expected_position = int(target_metadata["target_label_position"])
        _require(
            pair.get("source_index") == source_index,
            f"Pairing source index differs at row {source_index}",
        )
        _require(
            pair.get("donor_index") == donor_index,
            f"Pairing donor index differs at row {source_index}",
        )
        _require(
            pair.get("target_label_positions") == [expected_position],
            f"Pairing target position differs at row {source_index}",
        )
        _require(
            pair.get("target_token_ids")
            == [int(target_metadata["target_token_id"])],
            f"Pairing target token differs at row {source_index}",
        )
        _require(
            pair.get("donor_token_ids")
            == [int(donor_metadata["target_token_id"])],
            f"Pairing donor token differs at row {source_index}",
        )
    return {
        "validated": True,
        "manifest_sha256": actual_hash,
        "locked_donor_indices": list(DONOR_INDICES),
        "sample_count": 4,
    }


def validate_training_protocol(
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    pairing: Mapping[str, Any],
) -> dict[str, Any]:
    exact_fields = {
        "dataset_name": None,
        "dataset_split": "train",
        "train_samples": 4,
        "eval_samples": 0,
        "validation_split_ratio": 0.0,
        "seed": SEED,
        "data_seed": SEED,
        "train_sampler_seed": SEED,
        "max_steps": 64,
        "save_steps": 8,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0002,
        "memory_loss_mode": "scene_state_identity_ce",
        "memory_objective_version": "scene_state_identity_ce_v2",
        "memory_readout_mode": "projected_kv_slots",
        "memory_write_granularity": "message_mean",
        "memory_write_source": "learned_hidden",
        "rwkv_ms_semantics_version": 2,
        "rwkv_ms_write_mode": "recurrent",
        "projected_kv_key_dim": 32,
        "projected_kv_temperature": 16.0,
        "projected_kv_update_cosine_threshold": 1.0,
    }
    for name, expected in exact_fields.items():
        _require(
            protocol.get(name) == expected,
            f"Training protocol field differs: {name}",
        )
    _require(
        Path(str(protocol.get("train_file", ""))).expanduser().resolve()
        == source["train_path"],
        "Training protocol train file differs",
    )
    source_identity = protocol.get("scene_state_source_manifest")
    expected_source_identity = {
        "path": str(source["manifest_path"]),
        "file_sha256": source["manifest_file_sha256"],
        "train_file": str(source["train_path"]),
        "train_file_sha256": source["train_sha256"],
        "train_rows": 4,
        "train_source_split": "train",
        "schema": canary.SOURCE_SCHEMA,
        "identity_donor_indices": list(DONOR_INDICES),
        "episode_contract": dict(canary.EPISODE_CONTRACT),
    }
    _require(
        source_identity == expected_source_identity,
        "Training protocol source binding differs",
    )
    pairing_binding = protocol.get("scene_state_identity_pairing")
    _require(
        isinstance(pairing_binding, dict),
        "Training protocol pairing binding is absent",
    )
    assert isinstance(pairing_binding, dict)
    _require(
        pairing_binding.get("pairing_locked") is True,
        "Training protocol pairing is not locked",
    )
    _require(
        pairing_binding.get("locked_donor_indices") == list(DONOR_INDICES),
        "Training protocol donor indices differ",
    )
    _require(
        pairing_binding.get("manifest_sha256")
        == pairing.get("manifest_sha256"),
        "Training protocol pairing hash differs",
    )
    return {
        "validated": True,
        "canonical_sha256": canonical_sha256(protocol),
        "seed": SEED,
        "max_steps": 64,
        "synthetic_train_rows": 4,
        "protected_evaluation_accessed": False,
    }


def load_run_provenance(run_dir: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    run = _regular_directory(run_dir, description="associative training run")
    protocol_path = _regular_file(
        run / "training_protocol.json",
        description="run training protocol",
    )
    pairing_path = _regular_file(
        run / "scene_state_identity_pairing_manifest.json",
        description="run pairing manifest",
    )
    protocol = _json_object(protocol_path, description="run training protocol")
    pairing = _json_object(pairing_path, description="run pairing manifest")
    pairing_validation = validate_pairing_manifest(pairing, source)
    protocol_validation = validate_training_protocol(protocol, source, pairing)
    return {
        "path": run,
        "protocol_payload": protocol,
        "pairing_payload": pairing,
        "public": {
            "run_path": str(run),
            "training_protocol": _file_record(
                protocol_path,
                description="run training protocol",
            ),
            "training_protocol_validation": protocol_validation,
            "pairing_manifest": _file_record(
                pairing_path,
                description="run pairing manifest",
            ),
            "pairing_manifest_validation": pairing_validation,
        },
    }


def load_checkpoint_binding(
    run: Mapping[str, Any],
    *,
    step: int,
    expected_config: HFDeltaMemConfig,
    source: Mapping[str, Any],
) -> CheckpointBinding:
    checkpoint = _regular_directory(
        Path(run["path"]) / "trainer" / f"checkpoint-{step}",
        description=f"checkpoint {step}",
    )
    files = {name: checkpoint / name for name in CHECKPOINT_ARTIFACT_NAMES}
    artifacts = snapshot_artifacts(files)
    config = HFDeltaMemConfig.from_pretrained(checkpoint)
    _require(
        config == expected_config,
        f"Checkpoint {step} Delta-Mem config differs from the locked preflight config",
    )
    protocol = _json_object(
        files["training_protocol.json"],
        description=f"checkpoint {step} training protocol",
    )
    pairing = _json_object(
        files["scene_state_identity_pairing_manifest.json"],
        description=f"checkpoint {step} pairing manifest",
    )
    _require(
        protocol == run["protocol_payload"],
        f"Checkpoint {step} training protocol differs from run-level protocol",
    )
    _require(
        pairing == run["pairing_payload"],
        f"Checkpoint {step} pairing manifest differs from run-level manifest",
    )
    validate_pairing_manifest(pairing, source)
    validate_training_protocol(protocol, source, pairing)
    return CheckpointBinding(
        step=step,
        path=checkpoint,
        config=config,
        artifacts_before=artifacts,
    )


def verify_all_checkpoint_artifacts_unchanged(
    checkpoints: Sequence[CheckpointBinding],
) -> dict[int, dict[str, dict[str, Any]]]:
    verified: dict[int, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    for checkpoint in checkpoints:
        try:
            verified[checkpoint.step] = verify_artifact_snapshot(
                checkpoint.artifacts_before
            )
        except (OSError, ValueError) as exc:
            errors.append(f"step {checkpoint.step}: {exc}")
    if errors:
        raise ValueError(
            "Checkpoint immutability verification failed: " + "; ".join(errors)
        )
    return verified


def checkpoint_public_binding(
    checkpoint: CheckpointBinding,
    artifacts_after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        checkpoint.artifacts_before
        == {name: dict(value) for name, value in artifacts_after.items()},
        f"Checkpoint {checkpoint.step} artifact before/after binding differs",
    )
    artifact_tree_sha256 = canonical_sha256(
        [
            {
                "name": name,
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
            for name, record in sorted(checkpoint.artifacts_before.items())
        ]
    )
    return {
        "step": checkpoint.step,
        "path": str(checkpoint.path),
        "delta_config": checkpoint.config.to_dict(),
        "delta_config_sha256": canonical_sha256(checkpoint.config.to_dict()),
        "artifact_tree_sha256": artifact_tree_sha256,
        "artifact_immutability": {
            "verified": True,
            "before": checkpoint.artifacts_before,
            "after": {name: dict(value) for name, value in artifacts_after.items()},
        },
    }


def online_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    records = []
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        records.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return canonical_sha256(records)


def audit_identical_write_rows(
    state: Mapping[str, torch.Tensor],
    row_pairs: Sequence[tuple[int, int]] = IDENTICAL_WRITE_ROW_PAIRS,
) -> dict[str, Any]:
    """Verify that rows with byte-identical writes produced identical state."""

    mismatches: list[dict[str, Any]] = []
    for name, tensor in state.items():
        _require(
            tensor.ndim >= 1 and tensor.size(0) == 4,
            f"Online state does not have the four-row batch shape: {name}",
        )
        for left, right in row_pairs:
            if not torch.equal(tensor[left], tensor[right]):
                mismatch: dict[str, Any] = {
                    "name": name,
                    "row_pair": [left, right],
                    "dtype": str(tensor.dtype),
                }
                if tensor.dtype == torch.bool:
                    mismatch["different_element_count"] = int(
                        tensor[left].ne(tensor[right]).sum().item()
                    )
                else:
                    mismatch["maximum_absolute_delta"] = float(
                        (tensor[left].float() - tensor[right].float())
                        .abs()
                        .max()
                        .item()
                    )
                mismatches.append(mismatch)
    return {
        "passed": not mismatches,
        "row_pairs": [list(pair) for pair in row_pairs],
        "state_tensor_count": len(state),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def validate_donor_state_transform(
    source: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
    indices: Sequence[int] = DONOR_INDICES,
) -> None:
    _require(set(source) == set(donor), "Donor state tensor names differ")
    order = torch.tensor(tuple(indices), dtype=torch.long)
    for name, tensor in source.items():
        _require(
            tensor.ndim >= 1 and tensor.size(0) == len(indices),
            f"Donor source state batch shape differs: {name}",
        )
        expected = tensor.index_select(0, order)
        _require(
            torch.equal(donor[name], expected),
            f"Donor row permutation differs: {name}",
        )


def build_donor_state(
    state: Mapping[str, torch.Tensor],
    indices: Sequence[int] = DONOR_INDICES,
) -> dict[str, torch.Tensor]:
    transformed = preflight._permuted_state(dict(state), tuple(indices))
    validate_donor_state_transform(state, transformed, indices)
    return transformed


def validate_wrong_slot_state_transform(
    source: Mapping[str, torch.Tensor],
    wrong_slot: Mapping[str, torch.Tensor],
    *,
    expected_layer_count: int = len(TARGET_LAYERS),
) -> None:
    _require(set(source) == set(wrong_slot), "Wrong-slot state tensor names differ")
    value_names = sorted(
        name for name in source if name.endswith(".__projected_kv_values")
    )
    _require(
        len(value_names) == expected_layer_count,
        "Wrong-slot state must contain one value tensor per target layer",
    )
    for name, tensor in source.items():
        expected = tensor.flip(1) if name in value_names else tensor
        _require(
            torch.equal(wrong_slot[name], expected),
            f"Wrong-slot transform changed the wrong tensor: {name}",
        )


def build_wrong_slot_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    transformed = preflight._wrong_slot_state(dict(state))
    validate_wrong_slot_state_transform(state, transformed)
    return transformed


def audit_projected_kv_write_state(
    state: Mapping[str, torch.Tensor],
    modules: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    _require(
        len(modules) == len(TARGET_LAYERS),
        "Projected-KV write state does not cover 42 layers",
    )
    layers: list[dict[str, Any]] = []
    actual_layers = tuple(int(module.layer_idx) for _, module in modules)
    _require(actual_layers == TARGET_LAYERS, "Projected-KV module layers differ from 0-41")
    for name, module in modules:
        keys = state.get(f"{name}.__projected_kv_keys")
        values = state.get(f"{name}.__projected_kv_values")
        occupied = state.get(f"{name}.__projected_kv_occupied")
        surprise = state.get(f"{name}.__projected_kv_surprise")
        write_routes = module.last_write_routes
        _require(
            all(value is not None for value in (keys, values, occupied, surprise)),
            f"Projected-KV state is incomplete at {name}",
        )
        assert (
            keys is not None
            and values is not None
            and occupied is not None
            and surprise is not None
        )
        _require(
            tuple(keys.shape) == (4, 2, 32),
            f"Projected-KV key shape differs at {name}",
        )
        _require(
            values.ndim == 3 and tuple(values.shape[:2]) == (4, 2),
            f"Projected-KV value shape differs at {name}",
        )
        _require(
            tuple(occupied.shape) == (4, 2),
            f"Projected-KV occupancy shape differs at {name}",
        )
        _require(
            tuple(surprise.shape) == (4, 2),
            f"Projected-KV surprise shape differs at {name}",
        )
        _require(
            isinstance(write_routes, torch.Tensor)
            and tuple(write_routes.shape) == (4, 2, 2),
            f"Projected-KV write routes differ at {name}",
        )
        _require(
            bool(occupied.bool().all()),
            f"Projected-KV slots are not both occupied at {name}",
        )
        _require(
            bool(torch.isfinite(keys).all())
            and bool(torch.isfinite(values).all()),
            f"Projected-KV state is non-finite at {name}",
        )
        key_norms = keys.float().norm(dim=-1)
        _require(
            torch.allclose(
                key_norms,
                torch.ones_like(key_norms),
                rtol=1e-3,
                atol=1e-3,
            ),
            f"Projected-KV keys are not unit vectors at {name}",
        )
        assert isinstance(write_routes, torch.Tensor)
        expected_write_routes = torch.tensor(
            [[0, 1]] * 4,
            device=write_routes.device,
            dtype=torch.long,
        )
        _require(
            torch.equal(write_routes.argmax(dim=-1), expected_write_routes),
            f"Projected-KV writes did not initialize slots 0 then 1 at {name}",
        )
        layers.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "occupied_per_row": occupied.sum(dim=-1).tolist(),
                "write_slot_indices": (
                    write_routes.argmax(dim=-1).detach().cpu().tolist()
                ),
                "key_norm_min": float(key_norms.min().item()),
                "key_norm_max": float(key_norms.max().item()),
            }
        )
    return {
        "validated": True,
        "semantics_version": 2,
        "immutable_snapshot": True,
        "layer_count": len(layers),
        "layers": layers,
        "online_state_sha256": online_state_sha256(state),
    }


def _route_audit(
    modules: Sequence[tuple[str, Any]],
    positions: Sequence[int],
    *,
    expect_absent: bool,
) -> dict[str, Any]:
    _require(
        len(modules) == len(TARGET_LAYERS),
        "Read-route audit does not cover 42 layers",
    )
    actual_layers = tuple(int(module.layer_idx) for _, module in modules)
    _require(actual_layers == TARGET_LAYERS, "Read-route layers differ from 0-41")
    expected = torch.tensor(EXPECTED_QUERY_SLOTS, dtype=torch.long)
    predictor_positions = torch.tensor(tuple(positions), dtype=torch.long) - 1
    rows = torch.arange(4, dtype=torch.long)
    audits: list[dict[str, Any]] = []
    all_passed = True
    intended_slot_match_count = 0
    exact_intended_layer_count = 0
    query_separation_match_count = 0
    same_query_consistency_match_count = 0
    for name, module in modules:
        routes = module.last_read_routes
        layer_intended_match_count: int | None = None
        layer_query_separation_match_count: int | None = None
        layer_same_query_consistency_match_count: int | None = None
        if expect_absent:
            passed = routes is None
            selected_indices = None
        else:
            passed = False
            selected_indices = None
            if (
                isinstance(routes, torch.Tensor)
                and routes.ndim == 3
                and routes.size(0) == 4
                and routes.size(-1) == 2
                and int(predictor_positions.min().item()) >= 0
                and int(predictor_positions.max().item()) < routes.size(1)
            ):
                selected = routes[
                    rows.to(routes.device),
                    predictor_positions.to(routes.device),
                ]
                if bool(torch.isfinite(selected).all()):
                    selected_indices_tensor = selected.argmax(dim=-1).detach().cpu()
                    selected_indices = selected_indices_tensor.tolist()
                    layer_intended_match_count = int(
                        selected_indices_tensor.eq(expected).sum().item()
                    )
                    layer_query_separation_match_count = sum(
                        int(selected_indices_tensor[left] != selected_indices_tensor[right])
                        for left, right in QUERY_SEPARATION_ROW_PAIRS
                    )
                    layer_same_query_consistency_match_count = sum(
                        int(selected_indices_tensor[left] == selected_indices_tensor[right])
                        for left, right in SAME_QUERY_ROW_PAIRS
                    )
                    passed = torch.equal(selected_indices_tensor, expected)
                    intended_slot_match_count += layer_intended_match_count
                    query_separation_match_count += layer_query_separation_match_count
                    same_query_consistency_match_count += (
                        layer_same_query_consistency_match_count
                    )
                    exact_intended_layer_count += int(passed)
        all_passed = all_passed and passed
        audits.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "routes_present": routes is not None,
                "target_predictor_slot_indices": selected_indices,
                "intended_slot_match_count": layer_intended_match_count,
                "query_separation_match_count": (
                    layer_query_separation_match_count
                ),
                "same_query_consistency_match_count": (
                    layer_same_query_consistency_match_count
                ),
                "passed": passed,
            }
        )
    intended_slot_comparison_count = len(TARGET_LAYERS) * len(EXPECTED_QUERY_SLOTS)
    query_separation_comparison_count = (
        len(TARGET_LAYERS) * len(QUERY_SEPARATION_ROW_PAIRS)
    )
    same_query_consistency_comparison_count = (
        len(TARGET_LAYERS) * len(SAME_QUERY_ROW_PAIRS)
    )
    return {
        "expectation": "absent" if expect_absent else "exact_[0,0,1,1]",
        "expected_query_slot_indices": None if expect_absent else list(EXPECTED_QUERY_SLOTS),
        "passed": all_passed,
        "layer_count": len(audits),
        "intended_slot_match_count": (
            None if expect_absent else intended_slot_match_count
        ),
        "intended_slot_comparison_count": (
            None if expect_absent else intended_slot_comparison_count
        ),
        "intended_slot_match_fraction": (
            None
            if expect_absent
            else intended_slot_match_count / intended_slot_comparison_count
        ),
        "exact_intended_layer_count": (
            None if expect_absent else exact_intended_layer_count
        ),
        "query_separation_row_pairs": (
            None
            if expect_absent
            else [list(pair) for pair in QUERY_SEPARATION_ROW_PAIRS]
        ),
        "query_separation_match_count": (
            None if expect_absent else query_separation_match_count
        ),
        "query_separation_comparison_count": (
            None if expect_absent else query_separation_comparison_count
        ),
        "query_separation_match_fraction": (
            None
            if expect_absent
            else query_separation_match_count / query_separation_comparison_count
        ),
        "same_query_row_pairs": (
            None if expect_absent else [list(pair) for pair in SAME_QUERY_ROW_PAIRS]
        ),
        "same_query_consistency_match_count": (
            None if expect_absent else same_query_consistency_match_count
        ),
        "same_query_consistency_comparison_count": (
            None if expect_absent else same_query_consistency_comparison_count
        ),
        "same_query_consistency_match_fraction": (
            None
            if expect_absent
            else (
                same_query_consistency_match_count
                / same_query_consistency_comparison_count
            )
        ),
        "layers": audits,
    }


def audit_expected_read_routes(
    modules: Sequence[tuple[str, Any]],
    positions: Sequence[int],
) -> dict[str, Any]:
    return _route_audit(modules, positions, expect_absent=False)


def audit_no_write_absence(
    modules: Sequence[tuple[str, Any]],
    positions: Sequence[int],
) -> dict[str, Any]:
    route_audit = _route_audit(modules, positions, expect_absent=True)
    state_layers = []
    state_absent = True
    for name, module in modules:
        sidecars = {
            "keys": module.projected_kv_keys,
            "values": module.projected_kv_values,
            "occupied": module.projected_kv_occupied,
            "surprise": module.projected_kv_surprise,
        }
        passed = all(value is None for value in sidecars.values())
        state_absent = state_absent and passed
        state_layers.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "projected_kv_state_absent": passed,
            }
        )
    return {
        "passed": state_absent and route_audit["passed"],
        "projected_kv_state_absent": state_absent,
        "read_routes_absent": route_audit["passed"],
        "state_layers": state_layers,
        "route_audit": route_audit,
    }


def build_row_score_reports(
    target_logits: Sequence[float],
    donor_logits: Sequence[float],
    target_ids: Sequence[int],
    donor_ids: Sequence[int],
    target_texts: Sequence[str],
    donor_texts: Sequence[str],
) -> list[dict[str, Any]]:
    values = (
        target_logits,
        donor_logits,
        target_ids,
        donor_ids,
        target_texts,
        donor_texts,
    )
    _require(
        all(len(value) == 4 for value in values),
        "Condition score report requires exactly four rows",
    )
    reports: list[dict[str, Any]] = []
    for row_index, (
        target_logit,
        donor_logit,
        target_id,
        donor_id,
        target_text,
        donor_text,
    ) in enumerate(zip(*values, strict=True)):
        target_score = float(target_logit)
        donor_score = float(donor_logit)
        _require(
            math.isfinite(target_score) and math.isfinite(donor_score),
            f"Condition logits are non-finite at row {row_index}",
        )
        if target_score > donor_score:
            winning_role = "target"
            winning_token_id: int | None = int(target_id)
            winning_token_text: str | None = str(target_text)
        elif donor_score > target_score:
            winning_role = "donor"
            winning_token_id = int(donor_id)
            winning_token_text = str(donor_text)
        else:
            winning_role = "tie"
            winning_token_id = None
            winning_token_text = None
        reports.append(
            {
                "row_index": row_index,
                "target_token_id": int(target_id),
                "target_token_text": str(target_text),
                "donor_token_id": int(donor_id),
                "donor_token_text": str(donor_text),
                "target_logit": target_score,
                "donor_logit": donor_score,
                "target_minus_donor_margin": target_score - donor_score,
                "winning_role": winning_role,
                "winning_token_id": winning_token_id,
                "winning_token_text": winning_token_text,
            }
        )
    return reports


def _condition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        role: sum(row.get("winning_role") == role for row in rows)
        for role in ("target", "donor", "tie")
    }


def build_checkpoint_gate(
    conditions: Mapping[str, Mapping[str, Any]],
    identical_write_audit: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(conditions) == set(CONDITIONS), "Checkpoint gate conditions differ")
    for name in CONDITIONS:
        rows = conditions[name].get("rows")
        _require(
            isinstance(rows, list) and len(rows) == 4,
            f"Condition {name} does not contain four rows",
        )
    correct_rows = conditions["correct"]["rows"]
    donor_rows = conditions["donor"]["rows"]
    wrong_rows = conditions["wrong_slot"]["rows"]
    no_write_rows = conditions["no_write"]["rows"]
    assert isinstance(correct_rows, list)
    assert isinstance(donor_rows, list)
    assert isinstance(wrong_rows, list)
    assert isinstance(no_write_rows, list)
    margin_dominance = all(
        float(correct["target_minus_donor_margin"])
        > float(control["target_minus_donor_margin"])
        for control_rows in (donor_rows, wrong_rows, no_write_rows)
        for correct, control in zip(correct_rows, control_rows, strict=True)
    )
    causal_content_criteria = {
        "correct_target_wins_all_four": all(
            row["winning_role"] == "target" for row in correct_rows
        ),
        "donor_state_selects_donor_all_four": all(
            row["winning_role"] == "donor" for row in donor_rows
        ),
        "wrong_slot_selects_donor_all_four": all(
            row["winning_role"] == "donor" for row in wrong_rows
        ),
        "correct_margin_exceeds_all_controls_row_wise": margin_dominance,
        "no_write_state_and_routes_absent": (
            conditions["no_write"].get("absence_audit", {}).get("passed")
            is True
        ),
        "identical_write_rows_produce_identical_state": (
            identical_write_audit.get("passed") is True
        ),
    }
    semantic_addressing_criteria = {
        "correct_routes_match_all_42_layers": (
            conditions["correct"].get("route_audit", {}).get("passed") is True
        ),
        "donor_routes_match_all_42_layers": (
            conditions["donor"].get("route_audit", {}).get("passed") is True
        ),
        "wrong_slot_routes_match_all_42_layers": (
            conditions["wrong_slot"].get("route_audit", {}).get("passed") is True
        ),
    }
    causal_content_passed = all(causal_content_criteria.values())
    semantic_addressing_passed = all(semantic_addressing_criteria.values())
    criteria = {**causal_content_criteria, **semantic_addressing_criteria}
    return {
        "passed": causal_content_passed and semantic_addressing_passed,
        "causal_content_passed": causal_content_passed,
        "semantic_addressing_passed": semantic_addressing_passed,
        "criteria": criteria,
        "causal_content_criteria": causal_content_criteria,
        "semantic_addressing_criteria": semantic_addressing_criteria,
        "diagnostics": {
            "no_write_cannot_solve_all_four": not all(
                row["winning_role"] == "target" for row in no_write_rows
            ),
        },
        "identical_write_audit": dict(identical_write_audit),
        "winner_counts": {
            name: _condition_counts(conditions[name]["rows"])
            for name in CONDITIONS
        },
    }


def summarize_trajectory(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(
        [int(result["step"]) for result in checkpoints]
        == list(CHECKPOINT_STEPS),
        "Trajectory checkpoint steps differ",
    )
    causal_passing = [
        result
        for result in checkpoints
        if result.get("gate", {}).get("causal_content_passed") is True
    ]
    semantic_passing = [
        result
        for result in checkpoints
        if result.get("gate", {}).get("semantic_addressing_passed") is True
    ]
    fully_passing = [
        result
        for result in checkpoints
        if result.get("gate", {}).get("passed") is True
    ]
    causal_passing_steps = [int(result["step"]) for result in causal_passing]
    semantic_passing_steps = [int(result["step"]) for result in semantic_passing]
    fully_passing_steps = [int(result["step"]) for result in fully_passing]
    best_causal_step: int | None = None
    best_causal_score: dict[str, float] | None = None
    if causal_passing:
        def selection_key(result: Mapping[str, Any]) -> tuple[float, float, int]:
            margins = [
                float(row["target_minus_donor_margin"])
                for row in result["conditions"]["correct"]["rows"]
            ]
            return min(margins), sum(margins) / len(margins), int(result["step"])

        best = max(causal_passing, key=selection_key)
        minimum_margin, mean_margin, _ = selection_key(best)
        best_causal_step = int(best["step"])
        best_causal_score = {
            "minimum_correct_target_minus_donor_margin": minimum_margin,
            "mean_correct_target_minus_donor_margin": mean_margin,
        }
    if fully_passing_steps:
        decision = "causal_memory_content_and_semantic_router_passed"
    elif causal_passing_steps and not semantic_passing_steps:
        decision = "causal_memory_content_but_semantic_router_failed"
    elif semantic_passing_steps and not causal_passing_steps:
        decision = "semantic_router_but_causal_memory_content_failed"
    elif causal_passing_steps and semantic_passing_steps:
        decision = "causal_and_semantic_gates_passed_at_different_steps"
    else:
        decision = "causal_memory_content_and_semantic_router_failed"
    return {
        "evaluated_steps": list(CHECKPOINT_STEPS),
        "causal_passing_steps": causal_passing_steps,
        "semantic_addressing_passing_steps": semantic_passing_steps,
        "fully_passing_steps": fully_passing_steps,
        "causal_passing_step_count": len(causal_passing_steps),
        "semantic_addressing_passing_step_count": len(semantic_passing_steps),
        "fully_passing_step_count": len(fully_passing_steps),
        "all_steps_fully_passed": fully_passing_steps == list(CHECKPOINT_STEPS),
        "best_causal_step": best_causal_step,
        "best_causal_step_score": best_causal_score,
        "best_causal_step_selection": (
            "causal-content checkpoint with maximum minimum correct margin, then mean "
            "correct margin, then later step"
        ),
        "decision": decision,
    }


def _selected_predictor_logits(
    logits: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor:
    rows = torch.arange(4, device=logits.device)
    predictors = torch.tensor(tuple(positions), device=logits.device) - 1
    return logits[rows, predictors].float().detach().cpu()


def _evaluate_condition(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    modules: Sequence[tuple[str, Any]],
    state: Mapping[str, torch.Tensor] | None,
    positions: Sequence[int],
    target_ids: Sequence[int],
    donor_ids: Sequence[int],
    target_texts: Sequence[str],
    donor_texts: Sequence[str],
    *,
    condition: str,
) -> dict[str, Any]:
    reset_delta_mem_states(model)
    no_write_state_absent_before_read = all(
        all(
            value is None
            for value in (
                module.delta_state,
                module.direct_last_hidden,
                module.projected_last_hidden,
                module.projected_kv_keys,
                module.projected_kv_values,
                module.projected_kv_occupied,
                module.projected_kv_surprise,
                module.rwkv_ms_positions,
                module.rwkv_ms_previous_source,
            )
        )
        for _, module in modules
    )
    if state is not None:
        load_delta_mem_online_state(model, dict(state))
    set_delta_mem_write_message_ids(model, None)
    set_delta_mem_write_sentence_ids(model, None)
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, preflight._read_context_mask(dict(batch)))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    predictor_logits = _selected_predictor_logits(outputs.logits, positions)
    row_indices = torch.arange(4)
    target_scores = predictor_logits[row_indices, torch.tensor(tuple(target_ids))]
    donor_scores = predictor_logits[row_indices, torch.tensor(tuple(donor_ids))]
    rows = build_row_score_reports(
        target_scores.tolist(),
        donor_scores.tolist(),
        target_ids,
        donor_ids,
        target_texts,
        donor_texts,
    )
    result: dict[str, Any] = {
        "condition": condition,
        "state_loaded": state is not None,
        "rows": rows,
        "winner_counts": _condition_counts(rows),
        "predictor_logits_sha256": preflight._tensor_sha256(predictor_logits),
    }
    if condition == "no_write":
        absence = audit_no_write_absence(modules, positions)
        absence["state_absent_before_read"] = no_write_state_absent_before_read
        absence["passed"] = absence["passed"] and no_write_state_absent_before_read
        result["absence_audit"] = absence
        result["route_audit"] = absence["route_audit"]
    else:
        result["route_audit"] = audit_expected_read_routes(modules, positions)
    return result


def _module_contract(modules: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    layers = [int(module.layer_idx) for _, module in modules]
    modes = sorted({str(module.memory_readout_mode) for _, module in modules})
    _require(
        layers == list(TARGET_LAYERS),
        "Runtime target layers differ from 0-41",
    )
    _require(
        modes == ["projected_kv_slots"],
        "Runtime readout mode differs from projected_kv_slots",
    )
    return {
        "validated": True,
        "module_count": len(modules),
        "layers": layers,
        "memory_readout_modes": modes,
    }


def evaluate_checkpoint(
    checkpoint: CheckpointBinding,
    source: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    tokenizer: Any,
    *,
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
) -> dict[str, Any]:
    set_seed(SEED)
    model: torch.nn.Module | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(device)
        _disable_training_cache(model)
        replaced = attach_delta_mem(model, checkpoint.config)
        loaded_config = load_delta_mem_adapter(model, checkpoint.path)
        _require(
            loaded_config.to_dict() == checkpoint.config.to_dict(),
            f"Checkpoint {checkpoint.step} adapter loader config differs",
        )
        model.eval()
        modules = list(iter_delta_mem_modules(model))
        runtime_contract = _module_contract(modules)
        _require(
            len(replaced) == len(TARGET_LAYERS),
            f"Checkpoint {checkpoint.step} did not replace 42 attention modules",
        )

        positions, target_ids, donor_ids = preflight._target_metadata(dict(source))
        target_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in target_ids
        ]
        donor_texts = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in donor_ids
        ]
        with torch.inference_mode():
            preflight._prime_correct_state(model, dict(batch))
            correct_state = get_delta_mem_online_state(model)
            write_audit = audit_projected_kv_write_state(correct_state, modules)
            identical_write_audit = audit_identical_write_rows(correct_state)
            state_hash = online_state_sha256(correct_state)
            donor_state = build_donor_state(correct_state)
            wrong_slot_state = build_wrong_slot_state(correct_state)
            _require(
                online_state_sha256(correct_state) == state_hash,
                "Correct state changed while building controls",
            )
            states: dict[str, Mapping[str, torch.Tensor] | None] = {
                "correct": correct_state,
                "donor": donor_state,
                "wrong_slot": wrong_slot_state,
                "no_write": None,
            }
            conditions = {
                condition: _evaluate_condition(
                    model,
                    batch,
                    modules,
                    states[condition],
                    positions,
                    target_ids,
                    donor_ids,
                    target_texts,
                    donor_texts,
                    condition=condition,
                )
                for condition in CONDITIONS
            }
            _require(
                online_state_sha256(correct_state) == state_hash,
                "Correct state snapshot changed during evaluation",
            )
        gate = build_checkpoint_gate(conditions, identical_write_audit)
        return {
            "step": checkpoint.step,
            "runtime_module_contract": runtime_contract,
            "write_state_audit": write_audit,
            "identical_write_audit": identical_write_audit,
            "state_transforms": {
                "correct_state_sha256": state_hash,
                "donor_state_sha256": online_state_sha256(donor_state),
                "donor_indices": list(DONOR_INDICES),
                "wrong_slot_state_sha256": online_state_sha256(wrong_slot_state),
                "wrong_slot_transform": (
                    "flip projected-KV values across slot dimension only"
                ),
            },
            "target_label_positions": positions,
            "conditions": conditions,
            "gate": gate,
        }
    finally:
        if model is not None:
            reset_delta_mem_states(model)
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_trajectory_eval(
    *,
    source_manifest: Path,
    gate0_receipt: Path,
    preflight_receipt: Path,
    run_dir: Path,
    model_path: Path,
    output: Path,
    device_name: str,
    dtype_name: str,
    attn_implementation: str,
) -> dict[str, Any]:
    _require(
        os.environ.get("HF_ENDPOINT") == HF_MIRROR_ENDPOINT,
        f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}",
    )
    output_path = output.expanduser().resolve()
    _require(
        not output_path.exists() and not output_path.is_symlink(),
        f"Trajectory output must be fresh: {output_path}",
    )
    locked_model = DEFAULT_MODEL_PATH.expanduser().resolve()
    resolved_model = _regular_directory(model_path, description="locked Gemma model")
    _require(
        resolved_model == locked_model,
        f"Model path must be exactly {locked_model}",
    )

    source = canary.load_source_bundle(
        source_manifest,
        model_path=resolved_model,
        verify_model_hashes=True,
    )
    _require(
        source["manifest"]["contract"].get("synthetic_data_only") is True,
        "Source is not synthetic-only",
    )
    _require(
        source["manifest"]["contract"].get("protected_evaluation_included")
        is False,
        "Source includes protected evaluation",
    )
    gate_receipt = gate0.validate_receipt(
        gate0_receipt,
        source_manifest,
        resolved_model,
        verify_model_hashes=False,
    )
    preflight_validation = preflight.validate_receipt(
        preflight_receipt,
        source_manifest,
        gate0_receipt,
        resolved_model,
        verify_model_hashes=False,
    )
    run = load_run_provenance(run_dir, source)
    expected_config = preflight.build_delta_config()
    checkpoints = [
        load_checkpoint_binding(
            run,
            step=step,
            expected_config=expected_config,
            source=source,
        )
        for step in CHECKPOINT_STEPS
    ]

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    device = torch.device(device_name)
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_model),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = preflight._move_batch(
        preflight._prepare_batch(tokenizer, source["rows"]),
        device,
    )

    checkpoint_results: list[dict[str, Any]] = []
    artifacts_after: dict[int, dict[str, dict[str, Any]]]
    try:
        for checkpoint in checkpoints:
            checkpoint_results.append(
                evaluate_checkpoint(
                    checkpoint,
                    source,
                    batch,
                    tokenizer,
                    model_path=resolved_model,
                    device=device,
                    dtype=dtype,
                    attn_implementation=attn_implementation,
                )
            )
    finally:
        artifacts_after = verify_all_checkpoint_artifacts_unchanged(checkpoints)

    for result, checkpoint in zip(checkpoint_results, checkpoints, strict=True):
        result["checkpoint"] = checkpoint_public_binding(
            checkpoint,
            artifacts_after[checkpoint.step],
        )
    trajectory = summarize_trajectory(checkpoint_results)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "evaluation_kind": "synthetic_train_only_projected_kv_associative_trajectory",
        "source": _source_binding(source),
        "gate0_receipt": gate_receipt,
        "preflight_receipt": preflight_validation,
        "run_provenance": run["public"],
        "model": source["model"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "seed": SEED,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "local_files_only": True,
        },
        "data_access": {
            "synthetic_only": True,
            "source_splits": ["train"],
            "protected_evaluation_accessed": False,
            "hard32_accessed": False,
            "protected_paths_resolved": [],
        },
        "contract": {
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "conditions": list(CONDITIONS),
            "donor_indices": list(DONOR_INDICES),
            "expected_query_slot_indices": list(EXPECTED_QUERY_SLOTS),
            "target_layers": list(TARGET_LAYERS),
            "delta_config": expected_config.to_dict(),
            "delta_config_sha256": canonical_sha256(expected_config.to_dict()),
        },
        "checkpoints": checkpoint_results,
        "trajectory": trajectory,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    written = write_json_atomic_fresh(output_path, payload)
    result = dict(payload)
    result["output_path"] = str(written)
    result["output_file_sha256"] = canary.sha256_file(written)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--gate0-receipt", type=Path, default=DEFAULT_GATE0_RECEIPT)
    parser.add_argument("--preflight-receipt", type=Path, default=DEFAULT_PREFLIGHT_RECEIPT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
    args = parse_args()
    result = run_trajectory_eval(
        source_manifest=args.source_manifest,
        gate0_receipt=args.gate0_receipt,
        preflight_receipt=args.preflight_receipt,
        run_dir=args.run_dir,
        model_path=args.model_path,
        output=args.output,
        device_name=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
