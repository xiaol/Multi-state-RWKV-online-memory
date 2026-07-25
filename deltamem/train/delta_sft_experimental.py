from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

import deltamem.chat_templates as project_chat_templates

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_gate_stats,
    collect_delta_mem_output_ratio_stats,
    collect_delta_mem_partition_route_stats,
    collect_delta_mem_state_stats,
    collect_delta_mem_weight_stats,
    freeze_non_delta_mem_params,
    get_delta_mem_state_dict,
    get_delta_mem_online_state,
    get_delta_mem_partition_regularization,
    get_delta_mem_write_regularization,
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    normalize_delta_heads,
    normalize_memory_backend,
    normalize_memory_fusion_placement,
    normalize_memory_readout_mode,
    normalize_state_update_mode,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
from deltamem.chat_templates import (
    apply_chat_template as apply_project_chat_template,
    resolve_effective_chat_template,
)
from deltamem.model_loading import resolve_attn_implementation
from deltamem.core.write_segmentation import split_text_into_sentence_token_chunks


class FrozenMLPActivationCheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        should_checkpoint = (
            self.training
            and torch.is_grad_enabled()
            and hidden_states.requires_grad
        )
        if not should_checkpoint:
            return self.module(hidden_states)

        return checkpoint(self.module, hidden_states, use_reentrant=False)


def checkpoint_frozen_mlp_activations(model: nn.Module) -> list[str]:
    candidates: list[tuple[str, nn.Module]] = []
    missing_mlp_layers: list[str] = []
    for attention_name, _ in iter_delta_mem_modules(model):
        parent_name, attention_attribute = attention_name.rsplit(".", 1)
        if attention_attribute != "self_attn":
            continue
        parent = model.get_submodule(parent_name)
        module = getattr(parent, "mlp", None)
        if module is None:
            missing_mlp_layers.append(parent_name)
            continue
        candidates.append((f"{parent_name}.mlp", module))

    if missing_mlp_layers:
        raise ValueError(
            "Frozen MLP activation checkpointing requires an MLP beside every selected "
            "Delta-Mem attention; missing MLP in: " + ", ".join(missing_mlp_layers)
        )
    if not candidates:
        raise ValueError(
            "Frozen MLP activation checkpointing found no decoder MLPs beside "
            "Delta-Mem attention modules"
        )

    trainable_parameters = [
        f"{name}.{parameter_name}"
        for name, module in candidates
        if not isinstance(module, FrozenMLPActivationCheckpointWrapper)
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if trainable_parameters:
        preview = ", ".join(trainable_parameters[:8])
        if len(trainable_parameters) > 8:
            preview += f", ... ({len(trainable_parameters)} total)"
        raise ValueError(
            "Frozen MLP activation checkpointing requires every selected MLP parameter "
            f"to be frozen; trainable parameters: {preview}"
        )

    checkpointed: list[str] = []
    for name, module in candidates:
        if isinstance(module, FrozenMLPActivationCheckpointWrapper):
            checkpointed.append(name)
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, FrozenMLPActivationCheckpointWrapper(module))
        checkpointed.append(name)
    return checkpointed


@contextmanager
def _temporarily_disable_delta_heads(model):
    active_heads = []
    for _, module in iter_delta_mem_modules(model):
        active_heads.append((module, module.active_delta_heads))
        module.active_delta_heads = frozenset()
    try:
        yield
    finally:
        for module, module_active_heads in active_heads:
            module.active_delta_heads = module_active_heads


@contextmanager
def _preserve_delta_runtime(model):
    runtime_attributes = (
        "delta_state",
        "rwkv_ms_positions",
        "rwkv_ms_previous_source",
        "read_context_mask",
        "last_beta_mean",
        "last_lambda_mean",
        "last_write_routes",
        "last_read_routes",
        "last_base_o_norm",
        "last_delta_o_norm",
        "last_delta_o_ratio",
        "last_delta_o_gate_mean",
        "last_delta_o_gate_min",
        "last_delta_o_gate_max",
        "last_delta_o_gate_lt_001_fraction",
        "last_delta_o_gate_gt_099_fraction",
        "last_fused_delta_o_norm",
        "last_fused_delta_o_ratio",
        "last_delta_o_base_cosine",
        "last_fused_o_ratio",
        "write_enabled",
        "write_message_ids",
        "write_sentence_ids",
    )
    snapshots = []
    for _, module in iter_delta_mem_modules(model):
        snapshots.append(
            (
                module,
                {
                    attribute: getattr(module, attribute)
                    for attribute in runtime_attributes
                    if hasattr(module, attribute)
                },
            )
        )
    try:
        yield
    finally:
        for module, snapshot in snapshots:
            for attribute, value in snapshot.items():
                setattr(module, attribute, value)


class _AccelerateKernelWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Detected kernel version" not in record.getMessage()


def suppress_non_actionable_accelerate_warnings() -> None:
    logger = logging.getLogger("accelerate.utils.other")
    if not any(isinstance(item, _AccelerateKernelWarningFilter) for item in logger.filters):
        logger.addFilter(_AccelerateKernelWarningFilter())


def _disable_training_cache(model) -> None:
    config = model.config
    config.use_cache = False
    get_text_config = getattr(config, "get_text_config", None)
    if not callable(get_text_config):
        return
    try:
        text_config = get_text_config(decoder=True)
    except TypeError:
        text_config = get_text_config()
    if text_config is not None:
        text_config.use_cache = False


def _promote_trainable_parameters_to_fp32(model) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not parameter.is_floating_point():
            raise TypeError(f"Trainable parameter {name} must be floating point")
        parameter.data = parameter.data.to(dtype=torch.float32)


_RESUME_LATEST_VALUES = frozenset({"auto", "latest"})
_RESUME_MODES = ("exact", "extend", "placement_ablation", "objective_ablation")
_ABLATION_RESUME_MODES = frozenset({"placement_ablation", "objective_ablation"})
_CONTINUATION_SCHEDULERS = frozenset({"constant", "constant_with_warmup"})
_REQUIRED_RESUME_CHECKPOINT_FILES = (
    "delta_mem_adapter.pt",
    "delta_mem_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
)
_TRAINING_PROTOCOL_FILENAME = "training_protocol.json"
_TRAINING_PROTOCOL_SCHEMA_VERSION = 2
_MEMORY_OBJECTIVE_VERSION = "canonical_full_context_teacher_v1"
_CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION = 3
_CONTENT_CONTRAST_OBJECTIVE_VERSION = "content_contrast_ce_v1"
_CONTENT_CONTRAST_PAIRING_FILENAME = "content_contrast_pairing_manifest.json"
_CONTENT_CONTRAST_PAIRING_VERSION = "post_split_half_rotation_v1"
_CONTINUATION_MANIFEST_FILENAME = "continuation_manifest.json"
_CONTINUATION_MANIFEST_SCHEMA_VERSION = 1
_ABLATION_LINEAGE_FILENAME = "ablation_lineage_manifest.json"
_ABLATION_LINEAGE_SCHEMA_VERSION = 1
_OBJECTIVE_ABLATION_PROTOCOL_DRIFT = frozenset(
    {
        "schema_version",
        "memory_objective_version",
        "memory_loss_mode",
        "memory_contrast_weight",
        "memory_margin",
        "memory_kl_weight",
        "write_sparsity_weight",
        "memory_partition_alignment_weight",
        "memory_partition_entropy_weight",
        "memory_partition_balance_weight",
        "content_contrast_negative_priming_grad",
        "content_contrast_pairing",
    }
)


def _missing_resume_checkpoint_files(
    checkpoint: Path,
    *,
    require_training_protocol: bool = False,
    require_content_contrast_pairing: bool = False,
) -> tuple[str, ...]:
    required_files = list(_REQUIRED_RESUME_CHECKPOINT_FILES)
    if require_training_protocol or require_content_contrast_pairing:
        required_files.append(_TRAINING_PROTOCOL_FILENAME)
    if require_content_contrast_pairing:
        required_files.append(_CONTENT_CONTRAST_PAIRING_FILENAME)
    return tuple(
        filename
        for filename in required_files
        if not (checkpoint / filename).is_file()
    )


def _validate_resume_checkpoint(
    checkpoint: Path,
    *,
    require_training_protocol: bool = False,
    require_content_contrast_pairing: bool = False,
) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {checkpoint}")
    missing = _missing_resume_checkpoint_files(
        checkpoint,
        require_training_protocol=require_training_protocol,
        require_content_contrast_pairing=require_content_contrast_pairing,
    )
    if missing:
        raise FileNotFoundError(
            f"Resume checkpoint is incomplete: {checkpoint}; missing {', '.join(missing)}"
        )
    return checkpoint.resolve()


def resolve_resume_checkpoint(
    resume_from_checkpoint: str | Path | None,
    trainer_output_dir: str | Path,
    *,
    require_training_protocol: bool = False,
    require_content_contrast_pairing: bool = False,
) -> str | None:
    if resume_from_checkpoint is None:
        return None
    raw_checkpoint = str(resume_from_checkpoint).strip()
    if not raw_checkpoint:
        raise ValueError("--resume-from-checkpoint must not be empty")
    if raw_checkpoint.lower() not in _RESUME_LATEST_VALUES:
        return str(
            _validate_resume_checkpoint(
                Path(raw_checkpoint).expanduser(),
                require_training_protocol=require_training_protocol,
                require_content_contrast_pairing=require_content_contrast_pairing,
            )
        )

    output_path = Path(trainer_output_dir).expanduser()
    if not output_path.is_dir():
        raise FileNotFoundError(
            f"Cannot resolve latest checkpoint because the trainer output directory does not exist: {output_path}"
        )
    candidates: list[tuple[int, Path]] = []
    for candidate in output_path.iterdir():
        prefix = "checkpoint-"
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue
        step = candidate.name[len(prefix) :]
        if step.isdigit():
            candidates.append((int(step), candidate))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in trainer output directory: {output_path}")
    candidates.sort(reverse=True)
    for _, candidate in candidates:
        if not _missing_resume_checkpoint_files(
            candidate,
            require_training_protocol=require_training_protocol,
            require_content_contrast_pairing=require_content_contrast_pairing,
        ):
            return str(candidate.resolve())
    newest = candidates[0][1]
    missing = _missing_resume_checkpoint_files(
        newest,
        require_training_protocol=require_training_protocol,
        require_content_contrast_pairing=require_content_contrast_pairing,
    )
    raise FileNotFoundError(
        f"No complete checkpoints found in trainer output directory: {output_path}; "
        f"newest checkpoint {newest} is missing {', '.join(missing)}"
    )


def _load_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _protocol_sha256(protocol: dict[str, object]) -> str:
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_training_horizon_extension(
    source_protocol: dict[str, object],
    target_protocol: dict[str, object],
) -> None:
    source_scheduler = str(source_protocol.get("lr_scheduler_type", ""))
    target_scheduler = str(target_protocol.get("lr_scheduler_type", ""))
    if source_scheduler != target_scheduler:
        raise ValueError("Training continuation cannot change lr_scheduler_type")
    if source_scheduler not in _CONTINUATION_SCHEDULERS:
        supported = ", ".join(sorted(_CONTINUATION_SCHEDULERS))
        raise ValueError(
            f"Training continuation does not support lr_scheduler_type={source_scheduler!r}; "
            f"supported schedulers: {supported}"
        )

    try:
        source_max_steps = int(source_protocol["max_steps"])
        target_max_steps = int(target_protocol["max_steps"])
        source_epochs = float(source_protocol["num_train_epochs"])
        target_epochs = float(target_protocol["num_train_epochs"])
        source_warmup_steps = int(source_protocol["warmup_steps"])
        target_warmup_steps = int(target_protocol["warmup_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Training continuation requires numeric max_steps, num_train_epochs, and warmup_steps"
        ) from exc

    if source_warmup_steps < 0 or target_warmup_steps != source_warmup_steps:
        raise ValueError("Training continuation must preserve the source warmup_steps")
    if source_max_steps > 0:
        if target_max_steps <= source_max_steps:
            raise ValueError(
                "Training continuation requires target max_steps to be greater than the source"
            )
        if target_epochs < source_epochs:
            raise ValueError(
                "Training continuation cannot reduce num_train_epochs when max_steps is active"
            )
        return
    if target_max_steps != source_max_steps:
        raise ValueError("Training continuation cannot switch from epoch mode to max_steps mode")
    if target_epochs <= source_epochs:
        raise ValueError(
            "Training continuation requires target num_train_epochs to be greater than the source"
        )


def _normalize_frozen_mlp_checkpointing_protocol(
    protocol: dict[str, object],
) -> dict[str, object]:
    normalized = dict(protocol)
    current_key = "frozen_mlp_activation_checkpointing"
    legacy_key = "frozen_mlp_checkpointing"
    current_present = current_key in normalized
    legacy_present = legacy_key in normalized
    if current_present and legacy_present and normalized[current_key] != normalized[legacy_key]:
        raise ValueError(
            "Training protocol has conflicting frozen MLP activation checkpointing values"
        )
    value = normalized.get(
        current_key,
        normalized.get(legacy_key, False),
    )
    if not isinstance(value, bool):
        raise ValueError(
            "Training protocol frozen MLP activation checkpointing value must be Boolean"
        )
    normalized.pop(legacy_key, None)
    normalized[current_key] = value
    return normalized


def _normalize_memory_fusion_placement_protocol(
    protocol: dict[str, object],
) -> dict[str, object]:
    normalized = dict(protocol)
    normalized["memory_fusion_placement"] = normalize_memory_fusion_placement(
        str(normalized.get("memory_fusion_placement", "attention_output"))
    )
    residual_scale = float(normalized.get("memory_fusion_residual_scale", 1.0))
    if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
        raise ValueError(
            "Training protocol memory_fusion_residual_scale must be finite and satisfy "
            "0 <= value <= 1"
        )
    normalized["memory_fusion_residual_scale"] = residual_scale
    return normalized


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_objective_ablation_source_protocol(
    source_protocol: dict[str, object],
) -> None:
    expected = {
        "schema_version": _TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": _MEMORY_OBJECTIVE_VERSION,
        "memory_loss_mode": "context_dropout_ce",
        "memory_base_kl_weight": 0.0,
    }
    mismatches = [
        key for key, value in expected.items() if source_protocol.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "objective_ablation requires a canonical context_dropout_ce source; "
            "invalid fields: " + ", ".join(mismatches)
        )
    source_only_forbidden = sorted(
        (_OBJECTIVE_ABLATION_PROTOCOL_DRIFT - set(expected)).intersection(source_protocol)
    )
    if source_only_forbidden:
        raise ValueError(
            "objective_ablation source unexpectedly contains content-contrast fields: "
            + ", ".join(source_only_forbidden)
        )


def _validate_content_contrast_pairing_summary(
    protocol: dict[str, object],
) -> None:
    pairing = protocol.get("content_contrast_pairing")
    if not isinstance(pairing, dict):
        raise ValueError(
            "objective_ablation target requires content_contrast_pairing metadata"
        )
    if pairing.get("pairing_version") != _CONTENT_CONTRAST_PAIRING_VERSION:
        raise ValueError("objective_ablation target has an unsupported pairing_version")
    if pairing.get("pairing_scope") != "within_post_split_partition":
        raise ValueError("objective_ablation target has an unsupported pairing_scope")
    if pairing.get("data_seed") != protocol.get("data_seed"):
        raise ValueError("objective_ablation target pairing data_seed does not match")
    if pairing.get("tokenized_fingerprint") != protocol.get("tokenized_fingerprint"):
        raise ValueError(
            "objective_ablation target pairing tokenized_fingerprint does not match"
        )
    if not _is_sha256(pairing.get("manifest_sha256")):
        raise ValueError("objective_ablation target pairing manifest hash is invalid")

    splits = pairing.get("splits")
    if not isinstance(splits, dict) or "train" not in splits:
        raise ValueError("objective_ablation target pairing requires a train split")
    expected_split_sizes = {
        "train": protocol.get("train_samples"),
        "eval": protocol.get("eval_samples"),
    }
    for split_name, split in splits.items():
        if split_name not in expected_split_sizes or not isinstance(split, dict):
            raise ValueError(
                f"objective_ablation target pairing has invalid split: {split_name}"
            )
        try:
            sample_count = int(split["sample_count"])
            rotation = int(split["rotation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"objective_ablation target pairing split {split_name} is incomplete"
            ) from exc
        if sample_count < 2 or sample_count % 2 != 0 or rotation != sample_count // 2:
            raise ValueError(
                f"objective_ablation target pairing split {split_name} has invalid rotation"
            )
        expected_size = expected_split_sizes[split_name]
        if expected_size is not None and sample_count != int(expected_size):
            raise ValueError(
                f"objective_ablation target pairing split {split_name} size does not match"
            )
        for hash_name in ("pairs_sha256", "manifest_sha256"):
            if not _is_sha256(split.get(hash_name)):
                raise ValueError(
                    f"objective_ablation target pairing split {split_name} "
                    f"has invalid {hash_name}"
                )


def _validate_objective_ablation_target_protocol(
    target_protocol: dict[str, object],
    *,
    require_pairing: bool,
) -> None:
    expected = {
        "schema_version": _CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "memory_loss_mode": "content_contrast_ce",
        "memory_base_kl_weight": 0.0,
        "memory_kl_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "content_contrast_negative_priming_grad": False,
    }
    mismatches = [
        key for key, value in expected.items() if target_protocol.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "objective_ablation requires the KL-free content_contrast_ce target; "
            "invalid fields: " + ", ".join(mismatches)
        )
    try:
        contrast_weight = float(target_protocol["memory_contrast_weight"])
        margin = float(target_protocol["memory_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "objective_ablation target requires numeric memory_contrast_weight and memory_margin"
        ) from exc
    if not math.isfinite(contrast_weight) or contrast_weight <= 0.0:
        raise ValueError(
            "objective_ablation target requires memory_contrast_weight to be positive"
        )
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("objective_ablation target requires memory_margin to be positive")
    if require_pairing:
        _validate_content_contrast_pairing_summary(target_protocol)


def _validate_objective_ablation_protocol_transition(
    source_protocol: dict[str, object],
    target_protocol: dict[str, object],
    *,
    require_pairing: bool,
) -> None:
    _validate_objective_ablation_source_protocol(source_protocol)
    _validate_objective_ablation_target_protocol(
        target_protocol,
        require_pairing=require_pairing,
    )


def validate_resume_training_protocol(
    source_protocol: dict[str, object],
    target_protocol: dict[str, object],
    *,
    resume_mode: str,
    require_objective_pairing: bool = True,
) -> None:
    if resume_mode not in _RESUME_MODES:
        raise ValueError(f"Unsupported resume mode: {resume_mode}")
    source_protocol = _normalize_frozen_mlp_checkpointing_protocol(source_protocol)
    target_protocol = _normalize_frozen_mlp_checkpointing_protocol(target_protocol)
    source_protocol = _normalize_memory_fusion_placement_protocol(source_protocol)
    target_protocol = _normalize_memory_fusion_placement_protocol(target_protocol)
    mismatches = sorted(
        key
        for key in set(source_protocol) | set(target_protocol)
        if source_protocol.get(key) != target_protocol.get(key)
    )
    if resume_mode in {"extend", "placement_ablation", "objective_ablation"}:
        _validate_training_horizon_extension(source_protocol, target_protocol)
        mismatches = [
            key for key in mismatches if key not in {"max_steps", "num_train_epochs"}
        ]
    if resume_mode == "placement_ablation":
        fusion_keys = {
            "memory_fusion_placement",
            "memory_fusion_residual_scale",
        }
        if all(source_protocol[key] == target_protocol[key] for key in fusion_keys):
            raise ValueError(
                "placement_ablation resume requires memory_fusion_placement or "
                "memory_fusion_residual_scale to change"
            )
        mismatches = [key for key in mismatches if key not in fusion_keys]
    if resume_mode == "objective_ablation":
        _validate_objective_ablation_protocol_transition(
            source_protocol,
            target_protocol,
            require_pairing=require_objective_pairing,
        )
        mismatches = [
            key for key in mismatches if key not in _OBJECTIVE_ABLATION_PROTOCOL_DRIFT
        ]
    if mismatches:
        raise ValueError(
            "Delta-Mem checkpoint training protocol does not match for: "
            + ", ".join(mismatches)
        )


def validate_resume_delta_config(
    source_config: HFDeltaMemConfig,
    target_config: HFDeltaMemConfig,
    *,
    resume_mode: str,
) -> None:
    if resume_mode not in _RESUME_MODES:
        raise ValueError(f"Unsupported resume mode: {resume_mode}")
    source = source_config.to_dict()
    target = target_config.to_dict()
    mismatches = sorted(
        key for key in set(source) | set(target) if source.get(key) != target.get(key)
    )
    if resume_mode == "placement_ablation":
        fusion_keys = {
            "memory_fusion_placement",
            "memory_fusion_residual_scale",
        }
        if all(source[key] == target[key] for key in fusion_keys):
            raise ValueError(
                "placement_ablation resume requires memory_fusion_placement or "
                "memory_fusion_residual_scale to change"
            )
        mismatches = [key for key in mismatches if key not in fusion_keys]
    if mismatches:
        raise ValueError(
            "Delta-Mem checkpoint config does not match the current training config for: "
            + ", ".join(mismatches)
        )


def validate_resume_adapter_topology(
    model: nn.Module,
    checkpoint: Path,
) -> str:
    source_state = torch.load(
        checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source_state, dict):
        raise ValueError("Delta-Mem checkpoint adapter must contain a state dictionary")
    target_state = get_delta_mem_state_dict(model)
    source_keys = list(source_state)
    target_keys = list(target_state)
    if source_keys != target_keys:
        raise ValueError(
            "Delta-Mem ablation requires identical ordered adapter parameter names"
        )
    topology = []
    for name in source_keys:
        source_tensor = source_state[name]
        target_tensor = target_state[name]
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError(f"Delta-Mem adapter entry is not a tensor: {name}")
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                "Delta-Mem ablation adapter shape mismatch for "
                f"{name}: source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                "Delta-Mem ablation adapter dtype mismatch for "
                f"{name}: source={source_tensor.dtype} target={target_tensor.dtype}"
            )
        topology.append(
            {
                "name": name,
                "shape": list(source_tensor.shape),
                "dtype": str(source_tensor.dtype),
            }
        )
    return hashlib.sha256(
        json.dumps(topology, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _lineage_manifest_filename(manifest: dict[str, object]) -> str:
    if manifest.get("mode") in _ABLATION_RESUME_MODES:
        return _ABLATION_LINEAGE_FILENAME
    return _CONTINUATION_MANIFEST_FILENAME


def prepare_training_continuation(
    args: argparse.Namespace,
    resume_from_checkpoint: str | None,
) -> dict[str, object] | None:
    if args.resume_mode == "exact":
        if resume_from_checkpoint is None:
            return None
        checkpoint = Path(resume_from_checkpoint)
        existing = [
            path
            for path in (
                checkpoint / _ABLATION_LINEAGE_FILENAME,
                checkpoint / _CONTINUATION_MANIFEST_FILENAME,
            )
            if path.is_file()
        ]
        if not existing:
            return None
        if len(existing) != 1:
            raise ValueError(f"Checkpoint has ambiguous resume lineage manifests: {checkpoint}")
        manifest_path = existing[0]
        manifest = _load_json_object(manifest_path, description="resume lineage manifest")
        if manifest_path.name == _ABLATION_LINEAGE_FILENAME:
            valid_mode = manifest.get("mode") in _ABLATION_RESUME_MODES
            expected_schema = _ABLATION_LINEAGE_SCHEMA_VERSION
        else:
            valid_mode = manifest.get("mode") == "extend"
            expected_schema = _CONTINUATION_MANIFEST_SCHEMA_VERSION
        if not valid_mode or manifest.get("schema_version") != expected_schema:
            raise ValueError(f"Unsupported resume lineage manifest: {manifest_path}")
        return manifest

    raw_checkpoint = (
        ""
        if args.resume_from_checkpoint is None
        else str(args.resume_from_checkpoint).strip()
    )
    if not raw_checkpoint or raw_checkpoint.lower() in _RESUME_LATEST_VALUES:
        raise ValueError(
            f"--resume-mode {args.resume_mode} requires an explicit "
            "--resume-from-checkpoint path"
        )
    if resume_from_checkpoint is None:
        raise ValueError(f"--resume-mode {args.resume_mode} requires a resolved checkpoint")

    checkpoint = Path(resume_from_checkpoint).resolve()
    source_trainer_dir = checkpoint.parent
    target_output_dir = Path(args.output_dir).expanduser().resolve()
    target_trainer_dir = target_output_dir / "trainer"
    if source_trainer_dir == target_trainer_dir:
        raise ValueError(
            f"--resume-mode {args.resume_mode} requires a distinct --output-dir"
        )
    if (
        args.resume_mode in _ABLATION_RESUME_MODES
        and target_output_dir.exists()
        and any(target_output_dir.iterdir())
    ):
        raise ValueError(
            f"--resume-mode {args.resume_mode} requires a fresh, empty --output-dir"
        )

    rng_state_files = sorted(
        path.name for path in checkpoint.glob("rng_state*.pth") if path.is_file()
    )
    if not rng_state_files:
        raise FileNotFoundError(
            f"Training continuation checkpoint is missing RNG state: {checkpoint}"
        )

    protocol_path = checkpoint / _TRAINING_PROTOCOL_FILENAME
    state_path = checkpoint / "trainer_state.json"
    source_protocol = _load_json_object(protocol_path, description="training protocol")
    trainer_state = _load_json_object(state_path, description="trainer state")
    try:
        global_step = int(trainer_state["global_step"])
        source_effective_max_steps = int(trainer_state["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Training continuation requires global_step and max_steps in trainer_state.json"
        ) from exc
    if global_step <= 0 or global_step != source_effective_max_steps:
        raise ValueError(
            f"--resume-mode {args.resume_mode} requires a completed checkpoint "
            "with global_step equal to max_steps"
        )
    checkpoint_step = checkpoint.name.removeprefix("checkpoint-")
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint_step.isdigit():
        raise ValueError("Training continuation source must be a checkpoint-N directory")
    if int(checkpoint_step) != global_step:
        raise ValueError(
            "Training continuation checkpoint directory step does not match trainer_state.json"
        )
    source_epoch = None
    if args.resume_mode == "objective_ablation":
        try:
            source_epoch = float(trainer_state["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "objective_ablation requires a numeric source epoch in trainer_state.json"
            ) from exc
        if not math.isfinite(source_epoch) or not source_epoch.is_integer():
            raise ValueError(
                "objective_ablation requires a source checkpoint at an epoch boundary"
            )

    target_protocol = dict(source_protocol)
    target_protocol["max_steps"] = args.max_steps
    target_protocol["num_train_epochs"] = args.num_train_epochs
    if args.resume_mode == "placement_ablation":
        target_protocol["memory_fusion_placement"] = args.memory_fusion_placement
        target_protocol["memory_fusion_residual_scale"] = getattr(
            args,
            "memory_fusion_residual_scale",
            1.0,
        )
    elif args.resume_mode == "objective_ablation":
        target_protocol.update(
            {
                "schema_version": _CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION,
                "memory_objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
                "memory_loss_mode": args.memory_loss_mode,
                "memory_contrast_weight": args.memory_contrast_weight,
                "memory_margin": args.memory_margin,
                "memory_kl_weight": args.memory_kl_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": (
                    args.memory_partition_alignment_weight
                ),
                "memory_partition_entropy_weight": args.memory_partition_entropy_weight,
                "memory_partition_balance_weight": args.memory_partition_balance_weight,
                "content_contrast_negative_priming_grad": False,
            }
        )
    validate_resume_training_protocol(
        source_protocol,
        target_protocol,
        resume_mode=args.resume_mode,
        require_objective_pairing=False,
    )
    source_protocol_max_steps = int(source_protocol["max_steps"])
    if source_protocol_max_steps > 0 and source_protocol_max_steps != source_effective_max_steps:
        raise ValueError(
            "Source training protocol max_steps does not match trainer_state.json"
        )

    manifest = {
        "schema_version": (
            _ABLATION_LINEAGE_SCHEMA_VERSION
            if args.resume_mode in _ABLATION_RESUME_MODES
            else _CONTINUATION_MANIFEST_SCHEMA_VERSION
        ),
        "mode": args.resume_mode,
        "source_checkpoint": str(checkpoint),
        "source_global_step": global_step,
        "source_effective_max_steps": source_effective_max_steps,
        "source_max_steps": source_protocol_max_steps,
        "source_num_train_epochs": float(source_protocol["num_train_epochs"]),
        "source_training_protocol_sha256": _protocol_sha256(source_protocol),
        "source_rng_state_files": rng_state_files,
        "target_max_steps": int(args.max_steps),
        "target_num_train_epochs": float(args.num_train_epochs),
        "lr_scheduler_type": str(source_protocol["lr_scheduler_type"]),
        "warmup_steps": int(source_protocol["warmup_steps"]),
    }
    if source_epoch is not None:
        manifest["source_epoch"] = source_epoch
    if args.resume_mode == "placement_ablation":
        source_config = HFDeltaMemConfig.from_pretrained(checkpoint)
        manifest.update(
            {
                "ablation": "memory_fusion_placement",
                "source_memory_fusion_placement": source_config.memory_fusion_placement,
                "target_memory_fusion_placement": normalize_memory_fusion_placement(
                    args.memory_fusion_placement
                ),
                "source_memory_fusion_residual_scale": (
                    source_config.memory_fusion_residual_scale
                ),
                "target_memory_fusion_residual_scale": float(
                    getattr(args, "memory_fusion_residual_scale", 1.0)
                ),
                "source_delta_config_sha256": _protocol_sha256(source_config.to_dict()),
            }
        )
    elif args.resume_mode == "objective_ablation":
        source_config = HFDeltaMemConfig.from_pretrained(checkpoint)
        manifest.update(
            {
                "ablation": "memory_training_objective",
                "source_memory_loss_mode": source_protocol["memory_loss_mode"],
                "target_memory_loss_mode": args.memory_loss_mode,
                "source_memory_objective_version": source_protocol[
                    "memory_objective_version"
                ],
                "target_memory_objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
                "target_memory_contrast_weight": float(args.memory_contrast_weight),
                "target_memory_margin": float(args.memory_margin),
                "source_delta_config_sha256": _protocol_sha256(source_config.to_dict()),
            }
        )
    return manifest


def resolve_resume_warmup_steps(
    computed_warmup_steps: int,
    resume_from_checkpoint: str | None,
) -> int:
    if resume_from_checkpoint is None:
        return computed_warmup_steps
    protocol = _load_json_object(
        Path(resume_from_checkpoint) / _TRAINING_PROTOCOL_FILENAME,
        description="resume training protocol",
    )
    try:
        warmup_steps = int(protocol["warmup_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume training protocol requires numeric warmup_steps") from exc
    if warmup_steps < 0:
        raise ValueError("Resume training protocol warmup_steps must be non-negative")
    return warmup_steps


def compute_warmup_steps(
    *,
    train_samples: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    max_steps: int,
    warmup_ratio: float,
) -> int:
    if warmup_ratio <= 0.0:
        return 0
    if max_steps > 0:
        total_steps = max_steps
    else:
        global_micro_batch = max(1, per_device_train_batch_size) * max(1, world_size)
        num_batches = max(1, math.ceil(train_samples / global_micro_batch))
        steps_per_epoch = max(1, math.ceil(num_batches / max(1, gradient_accumulation_steps)))
        total_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    return max(1, math.ceil(total_steps * warmup_ratio))


def _build_ddp_training_kwargs(
    *,
    distributed: bool,
    ddp_backend: str,
    local_rank: int,
) -> dict[str, object]:
    # Episode loss keeps multiple forward graphs alive; synchronizing buffers
    # between them invalidates autograd versions for Gemma4's layer scalars.
    return {
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "ddp_backend": ddp_backend if distributed else None,
        "local_rank": local_rank if distributed else -1,
    }


@dataclass
class SFTExample:
    messages: list[dict[str, str]]


class DeltaMemTrainer(Trainer):
    def __init__(
        self,
        *args,
        delta_config: HFDeltaMemConfig | None = None,
        write_sparsity_weight: float = 0.0,
        write_sparsity_target: float = 0.0,
        memory_loss_mode: str = "context_dropout_ce",
        memory_contrast_weight: float = 0.1,
        memory_kl_weight: float = 0.1,
        memory_margin: float = 0.1,
        memory_causal_weight: float = 1.0,
        memory_anchor_weight: float = 1.0,
        memory_anchor_margin: float = 0.005,
        memory_full_ce_weight: float = 0.0,
        memory_full_ce_max_length: int = 2048,
        memory_recover_weight: float = 0.25,
        memory_need_floor: float = 0.15,
        memory_probe_weight: float = 0.0,
        memory_probe_alpha: float = 0.4,
        memory_probe_margin: float = 0.01,
        memory_partition_alignment_weight: float = 0.0,
        memory_partition_entropy_weight: float = 0.0,
        memory_partition_balance_weight: float = 0.0,
        memory_dropout_no_memory_prob: float = 0.0,
        memory_dropout_state_only_prob: float = 0.0,
        memory_base_kl_weight: float = 0.0,
        episode_read_write_enabled: bool = False,
        context_ablation_mode: str = "mixed",
        context_ablation_no_state_prob: float = 0.2,
        context_ablation_state_only_prob: float = 0.2,
        training_protocol: dict[str, object] | None = None,
        content_contrast_pairing_manifest: dict[str, object] | None = None,
        resume_mode: str = "exact",
        continuation_manifest: dict[str, object] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.delta_config = delta_config
        self.write_sparsity_weight = write_sparsity_weight
        self.write_sparsity_target = write_sparsity_target
        self.memory_loss_mode = memory_loss_mode
        self.memory_contrast_weight = memory_contrast_weight
        self.memory_kl_weight = memory_kl_weight
        self.memory_margin = memory_margin
        self.memory_causal_weight = memory_causal_weight
        self.memory_anchor_weight = memory_anchor_weight
        self.memory_anchor_margin = memory_anchor_margin
        self.memory_full_ce_weight = memory_full_ce_weight
        self.memory_full_ce_max_length = memory_full_ce_max_length
        self.memory_recover_weight = memory_recover_weight
        self.memory_need_floor = memory_need_floor
        self.memory_probe_weight = memory_probe_weight
        if self.memory_probe_weight > 0.0:
            raise ValueError("memory_probe was removed with archived memory_reader support")
        self.memory_probe_alpha = memory_probe_alpha
        self.memory_probe_margin = memory_probe_margin
        self.memory_partition_alignment_weight = memory_partition_alignment_weight
        self.memory_partition_entropy_weight = memory_partition_entropy_weight
        self.memory_partition_balance_weight = memory_partition_balance_weight
        self.memory_dropout_no_memory_prob = memory_dropout_no_memory_prob
        self.memory_dropout_state_only_prob = memory_dropout_state_only_prob
        if memory_base_kl_weight < 0.0:
            raise ValueError("memory_base_kl_weight must be non-negative")
        self.memory_base_kl_weight = memory_base_kl_weight
        self.episode_read_write_enabled = episode_read_write_enabled
        if memory_loss_mode == "content_contrast_ce":
            if episode_read_write_enabled:
                raise ValueError("content_contrast_ce requires episode read writes to be disabled")
            if memory_kl_weight != 0.0 or memory_base_kl_weight != 0.0:
                raise ValueError("content_contrast_ce requires all KL weights to be zero")
            if memory_contrast_weight < 0.0:
                raise ValueError("content_contrast_ce requires a non-negative contrast weight")
            if memory_margin < 0.0:
                raise ValueError("content_contrast_ce requires a non-negative margin")
            if write_sparsity_weight != 0.0:
                raise ValueError("content_contrast_ce requires write sparsity loss to be disabled")
            if (
                memory_partition_alignment_weight != 0.0
                or memory_partition_entropy_weight != 0.0
                or memory_partition_balance_weight != 0.0
            ):
                raise ValueError(
                    "content_contrast_ce requires memory partition regularization to be disabled"
                )
        self.context_ablation_mode = context_ablation_mode
        self.context_ablation_no_state_prob = context_ablation_no_state_prob
        self.context_ablation_state_only_prob = context_ablation_state_only_prob
        self.training_protocol = None if training_protocol is None else dict(training_protocol)
        self.content_contrast_pairing_manifest = (
            None
            if content_contrast_pairing_manifest is None
            else dict(content_contrast_pairing_manifest)
        )
        if resume_mode not in _RESUME_MODES:
            raise ValueError(f"Unsupported resume mode: {resume_mode}")
        self.resume_mode = resume_mode
        self.continuation_manifest = (
            None if continuation_manifest is None else dict(continuation_manifest)
        )
        self.memory_dropout_counts = {"both": 0, "state_only": 0, "no_memory": 0}
        self._last_write_sparsity_loss = 0.0
        self._last_memory_keep_loss = 0.0
        self._last_memory_reset_loss = 0.0
        self._last_memory_corrupt_loss = 0.0
        self._last_memory_margin_loss = 0.0
        self._last_memory_causal_loss = 0.0
        self._last_memory_anchor_loss = 0.0
        self._last_memory_full_ce_loss = 0.0
        self._last_memory_kl_loss = 0.0
        self._last_memory_reset_kl_loss = 0.0
        self._last_memory_margin_gap = 0.0
        self._last_memory_teacher_loss = 0.0
        self._last_memory_wmem = 0.0
        self._last_memory_probe_keep_loss = 0.0
        self._last_memory_probe_reset_loss = 0.0
        self._last_memory_probe_margin_loss = 0.0
        self._last_memory_probe_gap = 0.0
        self._last_memory_probe_kl_loss = 0.0
        self._last_memory_probe_ce_loss = 0.0
        self._last_memory_partition_alignment_loss = 0.0
        self._last_memory_partition_entropy_loss = 0.0
        self._last_memory_partition_balance_loss = 0.0
        self._last_partition_enabled_modules = 0.0
        self._last_partition_tied_read_write_modules = 0.0
        self._last_partition_active_modules = 0.0
        self._last_partition_write_route_entropy = 0.0
        self._last_partition_read_route_entropy = 0.0
        self._last_partition_route_alignment_mse = 0.0
        self._last_partition_route_overlap = 0.0
        self._last_partition_write_route_max = 0.0
        self._last_partition_read_route_max = 0.0
        self._last_partition_write_route_balance_l2 = 0.0
        self._last_partition_read_route_balance_l2 = 0.0
        self._ddp_static_graph_initialized = False

    def _maybe_enable_static_graph(self, model) -> None:
        if self._ddp_static_graph_initialized:
            return
        # The active trainer only keeps delta readout, so the legacy stacked-read
        # branches no longer need special DDP static-graph handling.
        self._ddp_static_graph_initialized = True

    def _reset_online_state(self, model) -> None:
        reset_delta_mem_states(model)
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)
        set_delta_mem_write_enabled(model, True)

    def _build_read_context_mask(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        labels = model_inputs.get("labels")
        attention_mask = model_inputs.get("attention_mask")
        if labels is None or attention_mask is None:
            return None
        return labels.eq(-100) & attention_mask.ne(0)

    def _unwrap_base_model(self, model):
        while True:
            wrapped_model = getattr(model, "module", None)
            if wrapped_model is not None and wrapped_model is not model:
                model = wrapped_model
                continue
            original_model = getattr(model, "_orig_mod", None)
            if original_model is not None and original_model is not model:
                model = original_model
                continue
            break
        return model

    def _logits_to_keep_kwargs(
        self,
        model,
        logits_to_keep: int | torch.Tensor,
    ) -> dict[str, int | torch.Tensor]:
        base_model = self._unwrap_base_model(model)
        try:
            call_parameters = inspect.signature(model.forward).parameters.values()
            base_parameters = inspect.signature(base_model.forward).parameters
        except (TypeError, ValueError):
            return {}
        call_accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in call_parameters
        )
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name not in base_parameters:
                continue
            if call_accepts_kwargs or any(
                parameter.name == name for parameter in call_parameters
            ):
                return {name: logits_to_keep}
        return {}

    def _scatter_episode_state(
        self,
        model,
        active_rows: torch.Tensor,
        batch_size: int,
    ) -> None:
        for _, module in iter_delta_mem_modules(model):
            if module.delta_state is None:
                continue
            active_state = module.delta_state
            full_state = active_state.new_zeros((batch_size, *active_state.shape[1:]))
            full_state[active_rows.to(device=active_state.device)] = active_state
            module.delta_state = full_state
            if module.rwkv_ms_positions is not None:
                active_positions = module.rwkv_ms_positions
                full_positions = active_positions.new_zeros((batch_size,))
                full_positions[active_rows.to(device=active_positions.device)] = active_positions
                module.rwkv_ms_positions = full_positions
            if module.rwkv_ms_previous_source is not None:
                active_previous_source = module.rwkv_ms_previous_source
                full_previous_source = active_previous_source.new_zeros(
                    (batch_size, *active_previous_source.shape[1:])
                )
                full_previous_source[
                    active_rows.to(device=active_previous_source.device)
                ] = active_previous_source
                module.rwkv_ms_previous_source = full_previous_source

    def _corrupt_online_state(
        self,
        online_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        corrupted: dict[str, torch.Tensor] = {}
        for name, tensor in online_state.items():
            corrupt = tensor.clone()
            if corrupt.ndim == 3:
                size = corrupt.size(-1)
                row_perm = torch.roll(torch.arange(size, device=corrupt.device), shifts=1)
                col_perm = torch.arange(size - 1, -1, -1, device=corrupt.device)
                corrupt = corrupt.index_select(-2, row_perm).index_select(-1, col_perm)
            elif corrupt.ndim == 4:
                num_partitions = corrupt.size(1)
                size = corrupt.size(-1)
                part_perm = torch.roll(torch.arange(num_partitions, device=corrupt.device), shifts=1)
                row_perm = torch.roll(torch.arange(size, device=corrupt.device), shifts=1)
                col_perm = torch.arange(size - 1, -1, -1, device=corrupt.device)
                corrupt = (
                    corrupt.index_select(1, part_perm)
                    .index_select(-2, row_perm)
                    .index_select(-1, col_perm)
                )
            elif corrupt.ndim == 5:
                num_states = corrupt.size(2)
                size = corrupt.size(-1)
                state_perm = torch.roll(torch.arange(num_states, device=corrupt.device), shifts=1)
                row_perm = torch.roll(torch.arange(size, device=corrupt.device), shifts=1)
                col_perm = torch.arange(size - 1, -1, -1, device=corrupt.device)
                corrupt = (
                    corrupt.index_select(2, state_perm)
                    .index_select(-2, row_perm)
                    .index_select(-1, col_perm)
                )
            corrupted[name] = corrupt
        return corrupted

    def _prime_episode_state(
        self,
        model,
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        batch_size: int,
        write_message_ids: torch.Tensor | None = None,
        write_sentence_ids: torch.Tensor | None = None,
    ) -> None:
        if write_input_ids is None:
            set_delta_mem_write_message_ids(model, None)
            set_delta_mem_write_sentence_ids(model, None)
            return
        if write_attention_mask is None:
            raise ValueError("Episode batches require write_attention_mask")
        active_rows = write_attention_mask.any(dim=1)
        if not active_rows.any():
            set_delta_mem_write_message_ids(model, None)
            set_delta_mem_write_sentence_ids(model, None)
            return
        active_message_ids = None
        if write_message_ids is not None:
            active_message_ids = write_message_ids[active_rows]
        active_sentence_ids = None
        if write_sentence_ids is not None:
            active_sentence_ids = write_sentence_ids[active_rows]
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_message_ids(model, active_message_ids)
        set_delta_mem_write_sentence_ids(model, active_sentence_ids)
        set_delta_mem_write_enabled(model, True)
        model(
            input_ids=write_input_ids[active_rows],
            attention_mask=write_attention_mask[active_rows],
            use_cache=False,
            return_dict=True,
            **self._logits_to_keep_kwargs(model, 1),
        )
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)
        self._scatter_episode_state(model, active_rows, batch_size)

    def _gather_teacher_read_logits(
        self,
        teacher_logits: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        read_width: int,
    ) -> torch.Tensor:
        gathered = teacher_logits.new_zeros(
            (teacher_logits.size(0), read_width, teacher_logits.size(-1))
        )
        for row_idx in range(teacher_logits.size(0)):
            write_len = int(write_lengths[row_idx].item())
            read_len = int(read_lengths[row_idx].item())
            gathered[row_idx, :read_len] = teacher_logits[
                row_idx,
                write_len : write_len + read_len,
            ]
        return gathered

    def _masked_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not token_mask.any():
            return student_logits.new_zeros(())
        log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=-1)
        return kl.masked_select(token_mask).mean()

    def _margin_objective(self, gap: torch.Tensor, margin: float) -> torch.Tensor:
        scaled_gap = (margin - gap) / max(margin, 1e-6)
        return F.softplus(scaled_gap)

    def _masked_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_mask = token_mask[:, 1:]
        if not shift_mask.any():
            return logits.new_zeros(())
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        return ce.masked_select(shift_mask).mean()

    def _masked_next_token_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_mask = labels[:, 1:].ne(-100) & attention_mask[:, 1:].ne(0)
        return self._masked_kl_loss(
            student_logits[:, :-1].float(),
            teacher_logits[:, :-1].float(),
            token_mask,
        )

    def _supervised_next_token_metadata(
        self,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError("Supervised next-token labels and attention mask must be matching 2D tensors")
        token_mask = labels[:, 1:].ne(-100) & attention_mask[:, 1:].ne(0)
        counts = token_mask.sum(dim=1)
        targets = labels[:, 1:].masked_select(token_mask)
        return token_mask, counts, targets

    def _validate_supervised_next_token_alignment(
        self,
        student_labels: torch.Tensor,
        student_attention_mask: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_mask, student_counts, student_targets = self._supervised_next_token_metadata(
            student_labels,
            student_attention_mask,
        )
        teacher_mask, teacher_counts, teacher_targets = self._supervised_next_token_metadata(
            teacher_labels,
            teacher_attention_mask,
        )
        if not torch.equal(student_counts, teacher_counts):
            raise ValueError(
                "Canonical teacher supervised next-token target counts do not match the episode read"
            )
        if not torch.equal(student_targets, teacher_targets):
            raise ValueError(
                "Canonical teacher supervised next-token target token IDs do not match the episode read"
            )
        return student_mask, teacher_mask, student_targets

    def _select_supervised_next_token_logits(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[:2] != (
            token_mask.size(0),
            token_mask.size(1) + 1,
        ):
            raise ValueError("Next-token logits do not align with the supervised token mask")
        return logits[:, :-1].masked_select(token_mask.unsqueeze(-1)).view(
            -1,
            logits.size(-1),
        )

    def _teacher_logits_projection_plan(
        self,
        model,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, int | torch.Tensor]]:
        predictor_positions = token_mask.any(dim=0).nonzero(as_tuple=False).flatten()
        projection_kwargs = self._logits_to_keep_kwargs(model, predictor_positions)
        if "logits_to_keep" not in projection_kwargs:
            return None, {}
        return predictor_positions, projection_kwargs

    def _select_projected_supervised_next_token_logits(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor,
        predictor_positions: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[:2] != (
            token_mask.size(0),
            predictor_positions.numel(),
        ):
            raise ValueError("Projected canonical teacher logits do not match requested positions")
        projected_mask = token_mask.index_select(1, predictor_positions)
        return logits.masked_select(projected_mask.unsqueeze(-1)).view(
            -1,
            logits.size(-1),
        )

    def _selected_teacher_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("Student and canonical teacher selected logits must have matching shapes")
        if student_logits.size(0) == 0:
            return student_logits.sum() * 0.0
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    def _forward_without_delta(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
    ):
        self._reset_online_state(model)
        read_context_mask = self._build_read_context_mask(model_inputs)
        set_delta_mem_read_context_mask(model, read_context_mask)
        set_delta_mem_write_enabled(model, False)
        with _temporarily_disable_delta_heads(model):
            return model(**model_inputs, **loss_kwargs)

    def _configure_episode_read(self, model, active_inputs: dict[str, torch.Tensor]) -> None:
        read_context_mask = self._build_read_context_mask(active_inputs)
        set_delta_mem_write_enabled(model, self.episode_read_write_enabled)
        set_delta_mem_read_context_mask(model, read_context_mask)

    def _zero_trainable_anchor(self, model, reference: torch.Tensor) -> torch.Tensor:
        anchor = None
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.numel() == 0:
                continue
            parameter_anchor = parameter.reshape(-1)[0] * 0.0
            anchor = parameter_anchor if anchor is None else anchor + parameter_anchor
        if anchor is None:
            return reference.new_zeros(())
        return anchor

    def _capture_live_online_state(
        self,
        model,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, module in iter_delta_mem_modules(model):
            if module.delta_state is None:
                continue
            state[name] = module.delta_state
            if module.memory_backend == "rwkv_ms" and module.rwkv_ms_positions is not None:
                state[f"{name}.__rwkv_ms_positions"] = module.rwkv_ms_positions
            if module.memory_backend == "rwkv_ms" and module.rwkv_ms_previous_source is not None:
                state[f"{name}.__rwkv_ms_previous_source"] = module.rwkv_ms_previous_source
        return state

    def _stack_batch_tensor(self, tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        return torch.cat([tensor] * repeats, dim=0)

    def _split_stacked_tensor(
        self,
        tensor: torch.Tensor | None,
        batch_size: int,
        num_variants: int,
    ) -> list[torch.Tensor | None]:
        if tensor is None:
            return [None] * num_variants
        return list(torch.split(tensor, batch_size, dim=0))

    def _memory_branch_uses_stacked_variants(self) -> bool:
        return False

    def _compute_memory_branch_loss_stacked(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_labels: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        loss_kwargs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        token_mask = model_inputs["labels"].ne(-100) & model_inputs["attention_mask"].ne(0)
        read_context_mask = self._build_read_context_mask(model_inputs)
        keep_online_state = self._capture_live_online_state(model)
        if not keep_online_state:
            raise RuntimeError("memory_branch stacked read requires primed online state")

        variant_names = ["keep", "reset"]
        if self.memory_loss_mode in {"state_causal_anchor"}:
            variant_names.append("corrupt")
        num_variants = len(variant_names)

        detached_state = {name: tensor.detach().clone() for name, tensor in keep_online_state.items()}
        reset_state = {name: torch.zeros_like(tensor) for name, tensor in detached_state.items()}
        corrupt_state = self._corrupt_online_state(detached_state)
        stacked_state: dict[str, torch.Tensor] = {}
        for name, keep_tensor in keep_online_state.items():
            variant_tensors = [keep_tensor, reset_state[name]]
            if "corrupt" in variant_names:
                variant_tensors.append(corrupt_state[name])
            stacked_state[name] = torch.cat(variant_tensors, dim=0)

        load_delta_mem_online_state(model, stacked_state)
        stacked_model_inputs = {
            key: self._stack_batch_tensor(value, num_variants)
            for key, value in model_inputs.items()
            if key != "labels"
        }
        stacked_read_context_mask = None
        if read_context_mask is not None:
            stacked_read_context_mask = self._stack_batch_tensor(read_context_mask, num_variants)
        set_delta_mem_read_context_mask(model, stacked_read_context_mask)
        set_delta_mem_write_enabled(model, False)
        stacked_outputs = model(**stacked_model_inputs)
        if not isinstance(stacked_outputs, dict):
            stacked_outputs = {
                "logits": stacked_outputs.logits,
            }
        stacked_logits = stacked_outputs["logits"]
        batch_size = int(model_inputs["input_ids"].size(0))
        split_logits = self._split_stacked_tensor(stacked_logits, batch_size, num_variants)
        keep_logits = split_logits[0]
        reset_logits = split_logits[1]
        corrupt_logits = split_logits[2] if len(split_logits) > 2 else None
        assert keep_logits is not None and reset_logits is not None

        keep_loss = self._masked_lm_loss(keep_logits, model_inputs["labels"], token_mask)
        reset_loss = self._masked_lm_loss(reset_logits, model_inputs["labels"], token_mask)
        corrupt_loss = keep_loss.new_zeros(())
        if corrupt_logits is not None:
            corrupt_loss = self._masked_lm_loss(corrupt_logits, model_inputs["labels"], token_mask)

        keep_outputs = {
            "loss": keep_loss,
            "logits": keep_logits,
        }
        reset_outputs = {
            "loss": reset_loss,
            "logits": reset_logits,
        }

        teacher_loss = keep_loss.new_zeros(())
        full_ce_loss = keep_loss.new_zeros(())
        teacher_read_logits = None
        keep_kl_loss = keep_loss.new_zeros(())
        reset_kl_loss = keep_loss.new_zeros(())
        if True:
            with torch.no_grad():
                self._reset_online_state(model)
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                teacher_logits = (
                    teacher_outputs["logits"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs.logits
                )
                teacher_loss = (
                    teacher_outputs["loss"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs[0]
                )
                if teacher_loss.ndim > 0:
                    teacher_loss = teacher_loss.mean()
                teacher_read_logits = self._gather_teacher_read_logits(
                    teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

            keep_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )
            reset_kl_loss = self._masked_kl_loss(
                reset_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )

        if self.memory_full_ce_weight > 0.0:
            aux_input_ids = full_input_ids
            aux_attention_mask = full_attention_mask
            aux_labels = full_labels
            if self.memory_full_ce_max_length > 0 and aux_input_ids.size(1) > self.memory_full_ce_max_length:
                aux_input_ids = aux_input_ids[:, -self.memory_full_ce_max_length :]
                aux_attention_mask = aux_attention_mask[:, -self.memory_full_ce_max_length :]
                aux_labels = aux_labels[:, -self.memory_full_ce_max_length :]
            self._reset_online_state(model)
            set_delta_mem_read_context_mask(model, None)
            set_delta_mem_write_enabled(model, True)
            full_ce_outputs = model(
                input_ids=aux_input_ids,
                attention_mask=aux_attention_mask,
                labels=aux_labels,
                **loss_kwargs,
            )
            full_ce_loss = (
                full_ce_outputs["loss"] if isinstance(full_ce_outputs, dict) else full_ce_outputs[0]
            )
            if full_ce_loss.ndim > 0:
                full_ce_loss = full_ce_loss.mean()

        margin_gap = reset_loss - keep_loss
        wmem = keep_loss.new_zeros(())
        causal_loss = keep_loss.new_zeros(())
        anchor_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode == "teacher_gap_kl":
            margin_gap = reset_kl_loss - keep_kl_loss
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        elif self.memory_loss_mode == "state_causal_anchor":
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            causal_gap = corrupt_loss - keep_loss
            causal_loss = self._margin_objective(causal_gap, self.memory_margin)
            anchor_gap = keep_loss - teacher_loss
            scaled_anchor = (anchor_gap - self.memory_anchor_margin) / max(
                self.memory_anchor_margin,
                1e-6,
            )
            anchor_loss = F.softplus(scaled_anchor)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_causal_weight * causal_loss
                + self.memory_anchor_weight * anchor_loss
            )
        elif self.memory_loss_mode in {"teacher_kl_only", "teacher_kl_wmem1"}:
            margin_loss = keep_loss.new_zeros(())
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = keep_loss.new_tensor(1.0)
            weighted = self.memory_kl_weight * keep_kl_loss
        elif self.memory_loss_mode == "teacher_kl_wmem":
            margin_loss = keep_loss.new_zeros(())
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = margin_gap.clamp_(min=0.0, max=1.0)
            weighted = self.memory_kl_weight * wmem * keep_kl_loss
        elif self.memory_loss_mode in {"state_margin_kl", "latent_prefix_margin"}:
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        else:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")
        weighted = weighted + self.memory_full_ce_weight * full_ce_loss
        probe_stats = {
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

        memory_loss = weighted
        total_loss = keep_loss + memory_loss
        outputs = dict(keep_outputs)
        outputs["memory_loss"] = memory_loss.detach()
        outputs["memory_full_ce_loss"] = total_loss.new_tensor(float(full_ce_loss.detach().float().item())).detach()
        outputs["memory_keep_loss"] = total_loss.detach()
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return total_loss, outputs, {
            "keep_loss": float(keep_loss.detach().float().item()),
            "reset_loss": float(reset_loss.detach().float().item()),
            "corrupt_loss": float(corrupt_loss.detach().float().item()),
            "teacher_loss": float(teacher_loss.detach().float().item()),
            "margin_loss": float(margin_loss.detach().float().item()),
            "causal_loss": float(causal_loss.detach().float().item()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "full_ce_loss": float(full_ce_loss.detach().float().item()),
            "kl_loss": float(keep_kl_loss.detach().float().item()),
            "reset_kl_loss": float(reset_kl_loss.detach().float().item()),
            "margin_gap": float(margin_gap.detach().float().item()),
            "wmem": float(wmem.detach().float().item()),
            **probe_stats,
        }

    def _build_full_sequence_inputs(
        self,
        full_input_ids: torch.Tensor | None,
        full_attention_mask: torch.Tensor | None,
        full_labels: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if full_input_ids is None or full_attention_mask is None or full_labels is None:
            raise ValueError("Full-sequence context ablations require full episode tensors")
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": full_labels,
        }

    def _sample_context_ablation_mode(self) -> str:
        mode = self.context_ablation_mode
        if mode != "mixed":
            return mode
        no_state_prob = self.context_ablation_no_state_prob
        state_only_prob = self.context_ablation_state_only_prob
        if (
            no_state_prob < 0.0
            or state_only_prob < 0.0
            or no_state_prob + state_only_prob > 1.0
        ):
            raise ValueError(
                "context ablation probabilities must satisfy p >= 0, q >= 0, p + q <= 1"
            )
        mode_sample = float(torch.rand(()).item())
        if mode_sample < no_state_prob:
            return "full_context_no_state"
        if mode_sample < no_state_prob + state_only_prob:
            return "state_only"
        return "full_context_plus_state"

    def _compute_context_dropout_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        state_only_input_ids: torch.Tensor | None,
        state_only_attention_mask: torch.Tensor | None,
        state_only_labels: torch.Tensor | None,
        state_only_write_input_ids: torch.Tensor | None,
        state_only_write_attention_mask: torch.Tensor | None,
        state_only_write_message_ids: torch.Tensor | None,
        state_only_write_sentence_ids: torch.Tensor | None,
        teacher_input_ids: torch.Tensor | None,
        teacher_attention_mask: torch.Tensor | None,
        teacher_labels: torch.Tensor | None,
    ):
        no_memory_prob = self.memory_dropout_no_memory_prob
        state_only_prob = self.memory_dropout_state_only_prob
        if no_memory_prob < 0.0 or state_only_prob < 0.0 or no_memory_prob + state_only_prob > 1.0:
            raise ValueError("memory dropout probabilities must satisfy p >= 0, q >= 0, p + q <= 1")

        if not model.training:
            mode = "both"
        else:
            mode_sample = float(torch.rand((), device=model_inputs["input_ids"].device).item())
            if mode_sample < no_memory_prob:
                mode = "no_memory"
            elif mode_sample < no_memory_prob + state_only_prob:
                mode = "state_only"
            else:
                mode = "both"
            counts = getattr(self, "memory_dropout_counts", None)
            if counts is not None:
                counts[mode] += 1

        if mode == "state_only":
            if (
                state_only_input_ids is None
                or state_only_attention_mask is None
                or state_only_labels is None
            ):
                raise ValueError("context_dropout_ce requires state_only episode tensors")
            active_inputs = {
                "input_ids": state_only_input_ids,
                "attention_mask": state_only_attention_mask,
                "labels": state_only_labels,
            }
            batch_size = int(state_only_input_ids.size(0))
            prime_kwargs = {
                "write_input_ids": state_only_write_input_ids,
                "write_attention_mask": state_only_write_attention_mask,
                "batch_size": batch_size,
                "write_message_ids": state_only_write_message_ids,
                "write_sentence_ids": state_only_write_sentence_ids,
            }
            wmem = 1.0
        elif mode == "no_memory":
            active_inputs = model_inputs
            prime_kwargs = None
            wmem = 0.0
        else:
            active_inputs = model_inputs
            batch_size = int(model_inputs["input_ids"].size(0))
            prime_kwargs = {
                "write_input_ids": write_input_ids,
                "write_attention_mask": write_attention_mask,
                "batch_size": batch_size,
                "write_message_ids": write_message_ids,
                "write_sentence_ids": write_sentence_ids,
            }
            wmem = 1.0

        student_supervised_mask = None
        teacher_selected_logits = None
        teacher_loss_value = 0.0
        if mode == "both" and self.memory_base_kl_weight > 0.0:
            if (
                teacher_input_ids is None
                or teacher_attention_mask is None
                or teacher_labels is None
            ):
                raise ValueError(
                    "context_dropout_ce with memory_base_kl_weight requires canonical teacher tensors"
                )
            student_supervised_mask, teacher_supervised_mask, supervised_targets = (
                self._validate_supervised_next_token_alignment(
                    active_inputs["labels"],
                    active_inputs["attention_mask"],
                    teacher_labels,
                    teacher_attention_mask,
                )
            )
            teacher_projection_positions, teacher_projection_kwargs = (
                self._teacher_logits_projection_plan(
                    model,
                    teacher_supervised_mask,
                )
            )
            with _preserve_delta_runtime(model), torch.no_grad():
                teacher_outputs = self._forward_without_delta(
                    model,
                    {
                        "input_ids": teacher_input_ids,
                        "attention_mask": teacher_attention_mask,
                    },
                    loss_kwargs=teacher_projection_kwargs,
                )
                teacher_logits = (
                    teacher_outputs["logits"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs.logits
                )
                if teacher_projection_positions is None:
                    teacher_selected_logits = self._select_supervised_next_token_logits(
                        teacher_logits,
                        teacher_supervised_mask,
                    ).detach()
                else:
                    teacher_selected_logits = (
                        self._select_projected_supervised_next_token_logits(
                            teacher_logits,
                            teacher_supervised_mask,
                            teacher_projection_positions,
                        ).detach()
                    )
                del teacher_logits, teacher_outputs
                if teacher_selected_logits.size(0) > 0:
                    teacher_loss_value = float(
                        F.cross_entropy(
                            teacher_selected_logits.float(),
                            supervised_targets,
                        ).item()
                    )

        if mode == "no_memory":
            outputs = self._forward_without_delta(
                model,
                active_inputs,
                loss_kwargs=loss_kwargs,
            )
        else:
            assert prime_kwargs is not None
            self._reset_online_state(model)
            self._prime_episode_state(model, **prime_kwargs)
            self._configure_episode_read(model, active_inputs)
            outputs = model(**active_inputs, **loss_kwargs)

        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss.ndim > 0:
            loss = loss.mean()
        task_loss = loss
        teacher_loss = loss.new_tensor(teacher_loss_value)
        base_kl_loss = loss.new_zeros(())
        if mode == "no_memory":
            loss = loss + self._zero_trainable_anchor(model, loss)
        elif teacher_selected_logits is not None:
            assert student_supervised_mask is not None
            student_selected_logits = self._select_supervised_next_token_logits(
                outputs["logits"],
                student_supervised_mask,
            )
            base_kl_loss = self._selected_teacher_kl_loss(
                student_selected_logits,
                teacher_selected_logits,
            )
            loss = loss + self.memory_base_kl_weight * base_kl_loss
        outputs = dict(outputs)
        outputs["loss"] = loss
        outputs["memory_loss"] = (loss - task_loss).detach()
        outputs["memory_base_kl_loss"] = base_kl_loss.detach()
        return loss, outputs, {
            "keep_loss": float(task_loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": float(teacher_loss.detach().float().item()),
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": float(base_kl_loss.detach().float().item()),
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": wmem,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _content_contrast_objective(
        self,
        correct_loss: torch.Tensor,
        wrong_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gap = wrong_loss - correct_loss
        contrast_loss = self._margin_objective(gap, self.memory_margin)
        total_loss = correct_loss + self.memory_contrast_weight * contrast_loss
        return total_loss, contrast_loss, gap

    def _compute_content_contrast_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        negative_write_input_ids: torch.Tensor,
        negative_write_attention_mask: torch.Tensor,
        negative_write_message_ids: torch.Tensor | None,
        negative_write_sentence_ids: torch.Tensor | None,
    ):
        if self.episode_read_write_enabled:
            raise ValueError("content_contrast_ce requires episode read writes to be disabled")
        if self.memory_kl_weight != 0.0 or self.memory_base_kl_weight != 0.0:
            raise ValueError("content_contrast_ce requires all KL weights to be zero")
        if getattr(self, "write_sparsity_weight", 0.0) != 0.0:
            raise ValueError("content_contrast_ce requires write sparsity loss to be disabled")
        if (
            getattr(self, "memory_partition_alignment_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_entropy_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_balance_weight", 0.0) != 0.0
        ):
            raise ValueError(
                "content_contrast_ce requires memory partition regularization to be disabled"
            )

        batch_size = int(model_inputs["input_ids"].size(0))
        read_context_mask = self._build_read_context_mask(model_inputs)

        self._reset_online_state(model)
        self._prime_episode_state(
            model,
            write_input_ids=write_input_ids,
            write_attention_mask=write_attention_mask,
            batch_size=batch_size,
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
        )
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_read_context_mask(model, read_context_mask)
        correct_outputs = model(**model_inputs, **loss_kwargs)
        if not isinstance(correct_outputs, dict):
            correct_outputs = {
                "loss": (
                    correct_outputs.loss
                    if hasattr(correct_outputs, "loss")
                    else correct_outputs[0]
                ),
                "logits": correct_outputs.logits,
            }
        correct_loss = correct_outputs["loss"]
        if correct_loss.ndim > 0:
            correct_loss = correct_loss.mean()

        self._reset_online_state(model)
        # The mismatched writer state is a fixed negative target. Detaching donor
        # priming saves VRAM while the mismatched read path still receives gradients.
        with torch.no_grad():
            self._prime_episode_state(
                model,
                write_input_ids=negative_write_input_ids,
                write_attention_mask=negative_write_attention_mask,
                batch_size=batch_size,
                write_message_ids=negative_write_message_ids,
                write_sentence_ids=negative_write_sentence_ids,
            )
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_read_context_mask(model, read_context_mask)
        wrong_outputs = model(**model_inputs, **loss_kwargs)
        wrong_loss = (
            wrong_outputs["loss"] if isinstance(wrong_outputs, dict) else wrong_outputs[0]
        )
        if wrong_loss.ndim > 0:
            wrong_loss = wrong_loss.mean()

        total_loss, contrast_loss, margin_gap = self._content_contrast_objective(
            correct_loss,
            wrong_loss,
        )
        outputs = dict(correct_outputs)
        outputs["loss"] = total_loss
        outputs["memory_loss"] = (total_loss - correct_loss).detach()
        outputs["memory_keep_loss"] = correct_loss.detach()
        return total_loss, outputs, {
            "keep_loss": float(correct_loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": float(wrong_loss.detach().float().item()),
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": float(contrast_loss.detach().float().item()),
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": float(margin_gap.detach().float().item()),
            "wmem": 1.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _compute_context_ablation_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        state_only_input_ids: torch.Tensor | None,
        state_only_attention_mask: torch.Tensor | None,
        state_only_labels: torch.Tensor | None,
        state_only_write_input_ids: torch.Tensor | None,
        state_only_write_attention_mask: torch.Tensor | None,
        state_only_write_message_ids: torch.Tensor | None,
        state_only_write_sentence_ids: torch.Tensor | None,
        full_input_ids: torch.Tensor | None,
        full_attention_mask: torch.Tensor | None,
        full_labels: torch.Tensor | None,
    ):
        mode = self._sample_context_ablation_mode()
        if mode == "full_context_no_state":
            active_inputs = self._build_full_sequence_inputs(
                full_input_ids,
                full_attention_mask,
                full_labels,
            )
            self._reset_online_state(model)
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 0.0
        elif mode == "state_only":
            if (
                state_only_input_ids is None
                or state_only_attention_mask is None
                or state_only_labels is None
            ):
                raise ValueError("context_ablation_ce requires state_only episode tensors")
            active_inputs = {
                "input_ids": state_only_input_ids,
                "attention_mask": state_only_attention_mask,
                "labels": state_only_labels,
            }
            batch_size = int(state_only_input_ids.size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=state_only_write_input_ids,
                write_attention_mask=state_only_write_attention_mask,
                batch_size=batch_size,
                write_message_ids=state_only_write_message_ids,
                write_sentence_ids=state_only_write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0
        elif mode == "full_context_plus_state":
            active_inputs = self._build_full_sequence_inputs(
                full_input_ids,
                full_attention_mask,
                full_labels,
            )
            batch_size = int(model_inputs["input_ids"].size(0))
            self._reset_online_state(model)
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
            read_context_mask = self._build_read_context_mask(active_inputs)
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            outputs = model(**active_inputs, **loss_kwargs)
            wmem = 1.0
        else:
            raise ValueError(f"Unsupported context ablation mode: {mode}")

        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss.ndim > 0:
            loss = loss.mean()
        return loss, outputs, {
            "keep_loss": float(loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": wmem,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _compute_memory_objective(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        keep_outputs,
        keep_loss: torch.Tensor,
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_labels: torch.Tensor,
        write_lengths: torch.Tensor,
        read_lengths: torch.Tensor,
        keep_online_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.memory_loss_mode == "none":
            zero = keep_loss.new_zeros(())
            return zero, {
                "keep_loss": float(keep_loss.detach().float().item()),
                "reset_loss": 0.0,
                "corrupt_loss": 0.0,
                "teacher_loss": 0.0,
                "margin_loss": 0.0,
                "causal_loss": 0.0,
                "anchor_loss": 0.0,
                "full_ce_loss": 0.0,
                "kl_loss": 0.0,
                "reset_kl_loss": 0.0,
                "margin_gap": 0.0,
                "wmem": 0.0,
                "probe_keep_loss": 0.0,
                "probe_reset_loss": 0.0,
                "probe_margin_loss": 0.0,
                "probe_gap": 0.0,
                "probe_kl": 0.0,
                "probe_ce": 0.0,
            }
        if self.memory_loss_mode not in {
            "state_margin_kl",
            "latent_prefix_margin",
            "state_causal_anchor",
            "teacher_gap_kl",
            "teacher_kl_only",
            "teacher_kl_wmem1",
            "teacher_kl_wmem",
            "keep_only",
            "keep_full_kl",
            "keep_fullstate_kl",
            "keep_dual_kl",
        }:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")

        token_mask = model_inputs["labels"].ne(-100) & model_inputs["attention_mask"].ne(0)
        read_context_mask = self._build_read_context_mask(model_inputs)
        self._reset_online_state(model)
        set_delta_mem_read_context_mask(model, read_context_mask)
        set_delta_mem_write_enabled(model, False)
        reset_outputs = model(**model_inputs, **loss_kwargs)
        reset_loss = (
            reset_outputs["loss"] if isinstance(reset_outputs, dict) else reset_outputs[0]
        )
        if reset_loss.ndim > 0:
            reset_loss = reset_loss.mean()
        corrupt_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode in {"state_causal_anchor"}:
            self._reset_online_state(model)
            load_delta_mem_online_state(
                model,
                self._corrupt_online_state(keep_online_state),
            )
            set_delta_mem_read_context_mask(model, read_context_mask)
            set_delta_mem_write_enabled(model, False)
            corrupt_outputs = model(**model_inputs, **loss_kwargs)
            corrupt_loss = (
                corrupt_outputs["loss"] if isinstance(corrupt_outputs, dict) else corrupt_outputs[0]
            )
            if corrupt_loss.ndim > 0:
                corrupt_loss = corrupt_loss.mean()

        teacher_loss = keep_loss.new_zeros(())
        full_ce_loss = keep_loss.new_zeros(())
        teacher_read_logits = None
        fullstate_teacher_read_logits = None
        keep_kl_loss = keep_loss.new_zeros(())
        reset_kl_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode in {
            "state_margin_kl",
            "latent_prefix_margin",
            "state_causal_anchor",
            "teacher_gap_kl",
            "teacher_kl_only",
            "teacher_kl_wmem1",
            "teacher_kl_wmem",
            "keep_full_kl",
            "keep_dual_kl",
        }:
            with torch.no_grad():
                self._reset_online_state(model)
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                teacher_logits = (
                    teacher_outputs["logits"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs.logits
                )
                teacher_loss = (
                    teacher_outputs["loss"]
                    if isinstance(teacher_outputs, dict)
                    else teacher_outputs[0]
                )
                if teacher_loss.ndim > 0:
                    teacher_loss = teacher_loss.mean()
                teacher_read_logits = self._gather_teacher_read_logits(
                    teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

            keep_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )
            reset_kl_loss = self._masked_kl_loss(
                reset_outputs["logits"],
                teacher_read_logits,
                token_mask,
            )

        if self.memory_loss_mode in {"keep_fullstate_kl", "keep_dual_kl"}:
            with torch.no_grad():
                self._reset_online_state(model)
                if keep_online_state:
                    load_delta_mem_online_state(
                        model,
                        {name: tensor.detach().clone() for name, tensor in keep_online_state.items()},
                    )
                set_delta_mem_read_context_mask(model, None)
                set_delta_mem_write_enabled(model, True)
                fullstate_teacher_outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    labels=full_labels,
                    **loss_kwargs,
                )
                fullstate_teacher_logits = (
                    fullstate_teacher_outputs["logits"]
                    if isinstance(fullstate_teacher_outputs, dict)
                    else fullstate_teacher_outputs.logits
                )
                fullstate_teacher_read_logits = self._gather_teacher_read_logits(
                    fullstate_teacher_logits,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    read_width=model_inputs["input_ids"].size(1),
                )

        if self.memory_full_ce_weight > 0.0:
            aux_input_ids = full_input_ids
            aux_attention_mask = full_attention_mask
            aux_labels = full_labels
            if self.memory_full_ce_max_length > 0 and aux_input_ids.size(1) > self.memory_full_ce_max_length:
                aux_input_ids = aux_input_ids[:, -self.memory_full_ce_max_length :]
                aux_attention_mask = aux_attention_mask[:, -self.memory_full_ce_max_length :]
                aux_labels = aux_labels[:, -self.memory_full_ce_max_length :]
            self._reset_online_state(model)
            set_delta_mem_read_context_mask(model, None)
            set_delta_mem_write_enabled(model, True)
            full_ce_outputs = model(
                input_ids=aux_input_ids,
                attention_mask=aux_attention_mask,
                labels=aux_labels,
                **loss_kwargs,
            )
            full_ce_loss = (
                full_ce_outputs["loss"] if isinstance(full_ce_outputs, dict) else full_ce_outputs[0]
            )
            if full_ce_loss.ndim > 0:
                full_ce_loss = full_ce_loss.mean()

        margin_gap = reset_loss - keep_loss
        wmem = keep_loss.new_zeros(())
        causal_loss = keep_loss.new_zeros(())
        anchor_loss = keep_loss.new_zeros(())
        margin_loss = keep_loss.new_zeros(())
        if self.memory_loss_mode == "keep_only":
            weighted = keep_loss.new_zeros(())
        elif self.memory_loss_mode == "keep_full_kl":
            weighted = self.memory_kl_weight * keep_kl_loss
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "keep_fullstate_kl":
            if fullstate_teacher_read_logits is None:
                raise ValueError("keep_fullstate_kl requires fullstate teacher logits")
            reset_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                fullstate_teacher_read_logits,
                token_mask,
            )
            weighted = self.memory_kl_weight * reset_kl_loss
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "keep_dual_kl":
            if fullstate_teacher_read_logits is None:
                raise ValueError("keep_dual_kl requires fullstate teacher logits")
            reset_kl_loss = self._masked_kl_loss(
                keep_outputs["logits"],
                fullstate_teacher_read_logits,
                token_mask,
            )
            weighted = self.memory_kl_weight * (keep_kl_loss + reset_kl_loss)
            wmem = keep_loss.new_tensor(1.0)
        elif self.memory_loss_mode == "teacher_gap_kl":
            margin_gap = reset_kl_loss - keep_kl_loss
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        elif self.memory_loss_mode == "state_causal_anchor":
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            causal_gap = corrupt_loss - keep_loss
            causal_loss = self._margin_objective(causal_gap, self.memory_margin)
            anchor_gap = keep_loss - teacher_loss
            scaled_anchor = (anchor_gap - self.memory_anchor_margin) / max(
                self.memory_anchor_margin,
                1e-6,
            )
            anchor_loss = F.softplus(scaled_anchor)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_causal_weight * causal_loss
                + self.memory_anchor_weight * anchor_loss
            )
        elif self.memory_loss_mode in {"teacher_kl_only", "teacher_kl_wmem1"}:
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = keep_loss.new_tensor(1.0)
            weighted = self.memory_kl_weight * keep_kl_loss
        elif self.memory_loss_mode == "teacher_kl_wmem":
            margin_gap = (reset_loss - teacher_loss).detach()
            wmem = margin_gap.clamp_(min=0.0, max=1.0)
            weighted = self.memory_kl_weight * wmem * keep_kl_loss
        elif self.memory_loss_mode in {"state_margin_kl", "latent_prefix_margin"}:
            margin_loss = self._margin_objective(margin_gap, self.memory_margin)
            weighted = (
                self.memory_contrast_weight * margin_loss
                + self.memory_kl_weight * keep_kl_loss
            )
        else:
            raise ValueError(f"Unsupported memory_loss_mode: {self.memory_loss_mode}")
        weighted = weighted + self.memory_full_ce_weight * full_ce_loss
        probe_stats = {
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }
        return weighted, {
            "keep_loss": float(keep_loss.detach().float().item()),
            "reset_loss": float(reset_loss.detach().float().item()),
            "corrupt_loss": float(corrupt_loss.detach().float().item()),
            "teacher_loss": float(teacher_loss.detach().float().item()),
            "margin_loss": float(margin_loss.detach().float().item()),
            "causal_loss": float(causal_loss.detach().float().item()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "full_ce_loss": float(full_ce_loss.detach().float().item()),
            "kl_loss": float(keep_kl_loss.detach().float().item()),
            "reset_kl_loss": float(reset_kl_loss.detach().float().item()),
            "margin_gap": float(margin_gap.detach().float().item()),
            "wmem": float(wmem.detach().float().item()),
            **probe_stats,
        }

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        loss_kwargs = {}
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch

        model_inputs = dict(inputs)
        write_input_ids = model_inputs.pop("write_input_ids", None)
        write_attention_mask = model_inputs.pop("write_attention_mask", None)
        write_message_ids = model_inputs.pop("write_message_ids", None)
        write_sentence_ids = model_inputs.pop("write_sentence_ids", None)
        negative_write_input_ids = model_inputs.pop("negative_write_input_ids", None)
        negative_write_attention_mask = model_inputs.pop("negative_write_attention_mask", None)
        negative_write_message_ids = model_inputs.pop("negative_write_message_ids", None)
        negative_write_sentence_ids = model_inputs.pop("negative_write_sentence_ids", None)
        state_only_write_input_ids = model_inputs.pop("state_only_write_input_ids", None)
        state_only_write_attention_mask = model_inputs.pop("state_only_write_attention_mask", None)
        state_only_write_message_ids = model_inputs.pop("state_only_write_message_ids", None)
        state_only_write_sentence_ids = model_inputs.pop("state_only_write_sentence_ids", None)
        state_only_input_ids = model_inputs.pop("state_only_input_ids", None)
        state_only_attention_mask = model_inputs.pop("state_only_attention_mask", None)
        state_only_labels = model_inputs.pop("state_only_labels", None)
        teacher_input_ids = model_inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = model_inputs.pop("teacher_attention_mask", None)
        teacher_labels = model_inputs.pop("teacher_labels", None)
        full_input_ids = model_inputs.pop("full_input_ids", None)
        full_attention_mask = model_inputs.pop("full_attention_mask", None)
        full_labels = model_inputs.pop("full_labels", None)
        write_lengths = model_inputs.pop("write_lengths", None)
        read_lengths = model_inputs.pop("read_lengths", None)

        batch_size = int(model_inputs["input_ids"].size(0))

        has_episode_memory_inputs = (
            write_input_ids is not None
            and full_input_ids is not None
            and full_attention_mask is not None
            and full_labels is not None
            and write_lengths is not None
            and read_lengths is not None
        )
        memory_stats = {
            "keep_loss": 0.0,
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": 0.0,
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": 0.0,
            "wmem": 0.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }
        if self.memory_loss_mode == "content_contrast_ce":
            if (
                write_input_ids is None
                or write_attention_mask is None
                or negative_write_input_ids is None
                or negative_write_attention_mask is None
            ):
                raise ValueError(
                    "content_contrast_ce requires materialized positive and negative write tensors"
                )
            loss, outputs, memory_stats = self._compute_content_contrast_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                negative_write_input_ids=negative_write_input_ids,
                negative_write_attention_mask=negative_write_attention_mask,
                negative_write_message_ids=negative_write_message_ids,
                negative_write_sentence_ids=negative_write_sentence_ids,
            )
        elif self.memory_loss_mode == "context_dropout_ce" and has_episode_memory_inputs:
            loss, outputs, memory_stats = self._compute_context_dropout_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                state_only_input_ids=state_only_input_ids,
                state_only_attention_mask=state_only_attention_mask,
                state_only_labels=state_only_labels,
                state_only_write_input_ids=state_only_write_input_ids,
                state_only_write_attention_mask=state_only_write_attention_mask,
                state_only_write_message_ids=state_only_write_message_ids,
                state_only_write_sentence_ids=state_only_write_sentence_ids,
                teacher_input_ids=teacher_input_ids,
                teacher_attention_mask=teacher_attention_mask,
                teacher_labels=teacher_labels,
            )
        elif self.memory_loss_mode == "context_ablation_ce" and has_episode_memory_inputs:
            loss, outputs, memory_stats = self._compute_context_ablation_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                state_only_input_ids=state_only_input_ids,
                state_only_attention_mask=state_only_attention_mask,
                state_only_labels=state_only_labels,
                state_only_write_input_ids=state_only_write_input_ids,
                state_only_write_attention_mask=state_only_write_attention_mask,
                state_only_write_message_ids=state_only_write_message_ids,
                state_only_write_sentence_ids=state_only_write_sentence_ids,
                full_input_ids=full_input_ids,
                full_attention_mask=full_attention_mask,
                full_labels=full_labels,
            )
        else:
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
            if has_episode_memory_inputs and self._memory_branch_uses_stacked_variants():
                loss, outputs, memory_stats = self._compute_memory_branch_loss_stacked(
                    model,
                    model_inputs,
                    full_input_ids=full_input_ids,
                    full_attention_mask=full_attention_mask,
                    full_labels=full_labels,
                    write_lengths=write_lengths,
                    read_lengths=read_lengths,
                    loss_kwargs=loss_kwargs,
                )
            else:
                read_context_mask = self._build_read_context_mask(model_inputs)
                set_delta_mem_read_context_mask(model, read_context_mask)
                set_delta_mem_write_enabled(model, False)
                outputs = model(**model_inputs, **loss_kwargs)
                if not isinstance(outputs, dict):
                    outputs = {
                        "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                        "logits": outputs.logits,
                    }
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                if loss.ndim > 0:
                    loss = loss.mean()
                memory_stats["keep_loss"] = float(loss.detach().float().item())
                if has_episode_memory_inputs:
                    keep_online_state = get_delta_mem_online_state(model)
                    memory_loss, memory_stats = self._compute_memory_objective(
                        model,
                        model_inputs,
                        outputs,
                        loss,
                        loss_kwargs=loss_kwargs,
                        write_input_ids=write_input_ids,
                        write_attention_mask=write_attention_mask,
                        full_input_ids=full_input_ids,
                        full_attention_mask=full_attention_mask,
                        full_labels=full_labels,
                        write_lengths=write_lengths,
                        read_lengths=read_lengths,
                        keep_online_state=keep_online_state,
                    )
                    loss = loss + memory_loss
                    outputs["memory_loss"] = memory_loss.detach()
                    outputs["memory_full_ce_loss"] = loss.new_tensor(memory_stats["full_ce_loss"]).detach()
                    outputs["memory_keep_loss"] = loss.detach()

        partition_regularization = None
        if (
            self.memory_partition_alignment_weight > 0.0
            or self.memory_partition_entropy_weight > 0.0
            or self.memory_partition_balance_weight > 0.0
        ):
            partition_regularization = get_delta_mem_partition_regularization(model)
            self._last_memory_partition_alignment_loss = float(
                partition_regularization["alignment"].detach().float().cpu().item()
            )
            self._last_memory_partition_entropy_loss = float(
                partition_regularization["entropy"].detach().float().cpu().item()
            )
            self._last_memory_partition_balance_loss = float(
                partition_regularization["balance"].detach().float().cpu().item()
            )
        else:
            self._last_memory_partition_alignment_loss = 0.0
            self._last_memory_partition_entropy_loss = 0.0
            self._last_memory_partition_balance_loss = 0.0
        partition_route_stats = collect_delta_mem_partition_route_stats(model)
        self._last_partition_enabled_modules = partition_route_stats["enabled_modules"]
        self._last_partition_tied_read_write_modules = partition_route_stats["tied_read_write_modules"]
        self._last_partition_active_modules = partition_route_stats["active_modules"]
        self._last_partition_write_route_entropy = partition_route_stats["write_route_entropy"]
        self._last_partition_read_route_entropy = partition_route_stats["read_route_entropy"]
        self._last_partition_route_alignment_mse = partition_route_stats["route_alignment_mse"]
        self._last_partition_route_overlap = partition_route_stats["route_overlap"]
        self._last_partition_write_route_max = partition_route_stats["write_route_max"]
        self._last_partition_read_route_max = partition_route_stats["read_route_max"]
        self._last_partition_write_route_balance_l2 = partition_route_stats["write_route_balance_l2"]
        self._last_partition_read_route_balance_l2 = partition_route_stats["read_route_balance_l2"]
        self._last_memory_keep_loss = memory_stats["keep_loss"]
        self._last_memory_reset_loss = memory_stats["reset_loss"]
        self._last_memory_corrupt_loss = memory_stats["corrupt_loss"]
        self._last_memory_teacher_loss = memory_stats["teacher_loss"]
        self._last_memory_margin_loss = memory_stats["margin_loss"]
        self._last_memory_causal_loss = memory_stats["causal_loss"]
        self._last_memory_anchor_loss = memory_stats["anchor_loss"]
        self._last_memory_full_ce_loss = memory_stats["full_ce_loss"]
        self._last_memory_kl_loss = memory_stats["kl_loss"]
        self._last_memory_reset_kl_loss = memory_stats["reset_kl_loss"]
        self._last_memory_margin_gap = memory_stats["margin_gap"]
        self._last_memory_wmem = memory_stats["wmem"]
        self._last_memory_probe_keep_loss = memory_stats["probe_keep_loss"]
        self._last_memory_probe_reset_loss = memory_stats["probe_reset_loss"]
        self._last_memory_probe_margin_loss = memory_stats["probe_margin_loss"]
        self._last_memory_probe_gap = memory_stats["probe_gap"]
        self._last_memory_probe_kl_loss = memory_stats["probe_kl"]
        self._last_memory_probe_ce_loss = memory_stats["probe_ce"]
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        if partition_regularization is not None:
            loss = loss + (
                self.memory_partition_alignment_weight * partition_regularization["alignment"]
                + self.memory_partition_entropy_weight * partition_regularization["entropy"]
                + self.memory_partition_balance_weight * partition_regularization["balance"]
            )
        if self.write_sparsity_weight > 0:
            write_sparsity_loss = get_delta_mem_write_regularization(
                model,
                target=self.write_sparsity_target,
            )
            self._last_write_sparsity_loss = float(write_sparsity_loss.detach().float().cpu().item())
            loss = loss + self.write_sparsity_weight * write_sparsity_loss
            if isinstance(outputs, dict):
                outputs = dict(outputs)
                outputs["write_sparsity_loss"] = write_sparsity_loss.detach()
                if partition_regularization is not None:
                    outputs["partition_alignment_loss"] = partition_regularization["alignment"].detach()
                    outputs["partition_entropy_loss"] = partition_regularization["entropy"].detach()
                    outputs["partition_balance_loss"] = partition_regularization["balance"].detach()
        else:
            self._last_write_sparsity_loss = 0.0
            if isinstance(outputs, dict) and partition_regularization is not None:
                outputs = dict(outputs)
                outputs["partition_alignment_loss"] = partition_regularization["alignment"].detach()
                outputs["partition_entropy_loss"] = partition_regularization["entropy"].detach()
                outputs["partition_balance_loss"] = partition_regularization["balance"].detach()
        if loss.ndim > 0:
            loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        enriched_logs = dict(logs)
        if self.model is not None and getattr(self, "log_delta_debug_stats", False):
            gate_stats = collect_delta_mem_gate_stats(self.model)
            output_ratio_stats = collect_delta_mem_output_ratio_stats(self.model)
            state_stats = collect_delta_mem_state_stats(self.model)
            weight_stats = collect_delta_mem_weight_stats(self.model)
            enriched_logs.update(
                {
                    "delta/beta_mean": gate_stats["beta_mean"],
                    "delta/lambda_mean": gate_stats["lambda_mean"],
                    "delta/rankwise_gate_modules": gate_stats["rankwise_gate_modules"],
                    "delta/partition_enabled_modules": self._last_partition_enabled_modules,
                    "delta/partition_tied_read_write_modules": self._last_partition_tied_read_write_modules,
                    "delta/partition_active_modules": self._last_partition_active_modules,
                    "delta/partition_write_route_entropy": self._last_partition_write_route_entropy,
                    "delta/partition_read_route_entropy": self._last_partition_read_route_entropy,
                    "delta/partition_route_alignment_mse": self._last_partition_route_alignment_mse,
                    "delta/partition_route_overlap": self._last_partition_route_overlap,
                    "delta/partition_write_route_max": self._last_partition_write_route_max,
                    "delta/partition_read_route_max": self._last_partition_read_route_max,
                    "delta/partition_write_route_balance_l2": self._last_partition_write_route_balance_l2,
                    "delta/partition_read_route_balance_l2": self._last_partition_read_route_balance_l2,
                    "delta/nonzero_state_modules": state_stats["nonzero_modules"],
                    "delta/max_state_norm": state_stats["max_state_norm"],
                    "delta/mean_state_norm": state_stats["mean_state_norm"],
                    "delta/max_state_abs": state_stats["max_state_abs"],
                    "delta/delta_o_proj_norm_sum": weight_stats["delta_o_proj_norm_sum"],
                    "delta/content_gated_fusion_modules": output_ratio_stats[
                        "content_gated_modules"
                    ],
                    "delta/attention_output_fusion_modules": output_ratio_stats[
                        "attention_output_fusion_modules"
                    ],
                    "delta/post_attention_norm_fusion_modules": output_ratio_stats[
                        "post_attention_norm_fusion_modules"
                    ],
                    "delta/normalized_residual_correction_fusion_modules": (
                        output_ratio_stats[
                            "normalized_residual_correction_fusion_modules"
                        ]
                    ),
                    "delta/base_o_norm": output_ratio_stats["mean_base_o_norm"],
                    "delta/raw_delta_o_norm": output_ratio_stats["mean_delta_o_norm"],
                    "delta/raw_delta_o_ratio": output_ratio_stats["mean_delta_o_ratio"],
                    "delta/raw_delta_o_ratio_max": output_ratio_stats["max_delta_o_ratio"],
                    "delta/fusion_gate_mean": output_ratio_stats["mean_delta_o_gate"],
                    "delta/fusion_gate_min": output_ratio_stats["min_delta_o_gate"],
                    "delta/fusion_gate_max": output_ratio_stats["max_delta_o_gate"],
                    "delta/fusion_gate_lt_001_fraction": output_ratio_stats[
                        "mean_delta_o_gate_lt_001_fraction"
                    ],
                    "delta/fusion_gate_gt_099_fraction": output_ratio_stats[
                        "mean_delta_o_gate_gt_099_fraction"
                    ],
                    "delta/fused_delta_o_norm": output_ratio_stats[
                        "mean_fused_delta_o_norm"
                    ],
                    "delta/fused_delta_o_ratio": output_ratio_stats[
                        "mean_fused_delta_o_ratio"
                    ],
                    "delta/fused_delta_o_ratio_max": output_ratio_stats[
                        "max_fused_delta_o_ratio"
                    ],
                    "delta/delta_o_base_cosine": output_ratio_stats[
                        "mean_delta_o_base_cosine"
                    ],
                    "delta/fused_o_ratio": output_ratio_stats["mean_fused_o_ratio"],
                    "delta/memory_residual_norm": output_ratio_stats[
                        "mean_memory_residual_norm"
                    ],
                    "delta/memory_residual_ratio": output_ratio_stats[
                        "mean_memory_residual_ratio"
                    ],
                    "delta/memory_residual_ratio_max": output_ratio_stats[
                        "max_memory_residual_ratio"
                    ],
                    "delta/delta_scale_mean": (
                        weight_stats["delta_scale_mean_sum"]
                        / max(weight_stats["trainable_delta_scale_modules"], 1)
                    ),
                    "delta/write_sparsity_loss": self._last_write_sparsity_loss,
                    "delta/memory_keep_loss": self._last_memory_keep_loss,
                    "delta/memory_reset_loss": self._last_memory_reset_loss,
                    "delta/memory_corrupt_loss": self._last_memory_corrupt_loss,
                    "delta/memory_teacher_loss": self._last_memory_teacher_loss,
                    "delta/memory_margin_loss": self._last_memory_margin_loss,
                    "delta/memory_causal_loss": self._last_memory_causal_loss,
                    "delta/memory_anchor_loss": self._last_memory_anchor_loss,
                    "delta/memory_full_ce_loss": self._last_memory_full_ce_loss,
                    "delta/memory_kl_loss": self._last_memory_kl_loss,
                    "delta/memory_reset_kl_loss": self._last_memory_reset_kl_loss,
                    "delta/memory_margin_gap": self._last_memory_margin_gap,
                    "delta/memory_wmem": self._last_memory_wmem,
                    "delta/memory_probe_keep_loss": self._last_memory_probe_keep_loss,
                    "delta/memory_probe_reset_loss": self._last_memory_probe_reset_loss,
                    "delta/memory_probe_margin_loss": self._last_memory_probe_margin_loss,
                    "delta/memory_probe_gap": self._last_memory_probe_gap,
                    "delta/memory_probe_kl_loss": self._last_memory_probe_kl_loss,
                    "delta/memory_probe_ce_loss": self._last_memory_probe_ce_loss,
                    "delta/memory_partition_alignment_loss": self._last_memory_partition_alignment_loss,
                    "delta/memory_partition_entropy_loss": self._last_memory_partition_entropy_loss,
                    "delta/memory_partition_balance_loss": self._last_memory_partition_balance_loss,
                }
            )
            for group_name in (
                "nonshared_local",
                "nonshared_full",
                "shared_local",
                "shared_full",
            ):
                for metric_name in (
                    "modules",
                    "active_modules",
                    "mean_fused_delta_o_ratio",
                    "max_fused_delta_o_ratio",
                    "mean_fused_o_ratio",
                    "mean_delta_o_base_cosine",
                    "mean_delta_o_gate",
                    "mean_delta_o_gate_lt_001_fraction",
                    "mean_delta_o_gate_gt_099_fraction",
                ):
                    enriched_logs[f"delta/group/{group_name}/{metric_name}"] = (
                        output_ratio_stats[f"{group_name}_{metric_name}"]
                    )
        super().log(enriched_logs, start_time=start_time)

    def training_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._maybe_enable_static_graph(model)
        self._reset_online_state(model)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def prediction_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        self._reset_online_state(model)
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self.is_world_process_zero():
            return
        if self.delta_config is None:
            raise ValueError("DeltaMemTrainer.save_model requires delta_config")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        save_delta_mem_adapter(model, output_path, self.delta_config)
        if self.training_protocol is not None:
            (output_path / _TRAINING_PROTOCOL_FILENAME).write_text(
                json.dumps(self.training_protocol, indent=2, sort_keys=True)
            )
        if self.content_contrast_pairing_manifest is not None:
            (output_path / _CONTENT_CONTRAST_PAIRING_FILENAME).write_text(
                json.dumps(
                    self.content_contrast_pairing_manifest,
                    indent=2,
                    sort_keys=True,
                )
            )
        continuation_manifest = getattr(self, "continuation_manifest", None)
        if continuation_manifest is not None:
            (output_path / _lineage_manifest_filename(continuation_manifest)).write_text(
                json.dumps(continuation_manifest, indent=2, sort_keys=True)
            )

    def _validate_checkpoint_training_protocol(
        self,
        checkpoint: Path,
        *,
        resume_mode: str | None = None,
    ) -> None:
        expected = getattr(self, "training_protocol", None)
        if expected is None:
            return
        protocol_path = checkpoint / _TRAINING_PROTOCOL_FILENAME
        if not protocol_path.is_file():
            raise ValueError(
                f"Delta-Mem checkpoint is missing {_TRAINING_PROTOCOL_FILENAME}: {checkpoint}"
            )
        actual = json.loads(protocol_path.read_text())
        active_resume_mode = (
            getattr(self, "resume_mode", "exact")
            if resume_mode is None
            else resume_mode
        )
        validate_resume_training_protocol(
            actual,
            expected,
            resume_mode=active_resume_mode,
        )
        expected_pairing = getattr(self, "content_contrast_pairing_manifest", None)
        if expected_pairing is None or active_resume_mode == "objective_ablation":
            return
        pairing_path = checkpoint / _CONTENT_CONTRAST_PAIRING_FILENAME
        if not pairing_path.is_file():
            raise ValueError(
                f"Delta-Mem checkpoint is missing {_CONTENT_CONTRAST_PAIRING_FILENAME}: "
                f"{checkpoint}"
            )
        actual_pairing = json.loads(pairing_path.read_text())
        if actual_pairing != expected_pairing:
            raise ValueError("Delta-Mem checkpoint content-contrast pairing manifest does not match")

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        active_resume_mode = getattr(self, "resume_mode", "exact")
        checkpoint = _validate_resume_checkpoint(
            Path(resume_from_checkpoint),
            require_training_protocol=(getattr(self, "training_protocol", None) is not None),
            require_content_contrast_pairing=(
                getattr(self, "content_contrast_pairing_manifest", None) is not None
                and active_resume_mode != "objective_ablation"
            ),
        )
        if self.delta_config is None:
            raise ValueError("DeltaMemTrainer checkpoint loading requires delta_config")
        checkpoint_config = HFDeltaMemConfig.from_pretrained(checkpoint)
        validate_resume_delta_config(
            checkpoint_config,
            self.delta_config,
            resume_mode=active_resume_mode,
        )
        load_model = self.model if model is None else model
        if active_resume_mode in _ABLATION_RESUME_MODES:
            topology_sha256 = validate_resume_adapter_topology(load_model, checkpoint)
            lineage = getattr(self, "continuation_manifest", None)
            if lineage is not None:
                lineage["ordered_adapter_parameter_topology_sha256"] = topology_sha256
        self._validate_checkpoint_training_protocol(checkpoint)
        if active_resume_mode == "placement_ablation":
            load_delta_mem_adapter(
                load_model,
                checkpoint,
                allowed_config_mismatches=(
                    "memory_fusion_placement",
                    "memory_fusion_residual_scale",
                ),
            )
        else:
            load_delta_mem_adapter(load_model, checkpoint)

    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint is None:
            raise RuntimeError("Cannot load the best Delta-Mem model without a best checkpoint")
        checkpoint = Path(self.state.best_model_checkpoint).resolve()
        if self.delta_config is None:
            raise ValueError("DeltaMemTrainer best-checkpoint loading requires delta_config")
        checkpoint_config = HFDeltaMemConfig.from_pretrained(checkpoint)
        if checkpoint_config != self.delta_config:
            raise ValueError("Best Delta-Mem checkpoint config does not match the active config")
        self._validate_checkpoint_training_protocol(checkpoint, resume_mode="exact")
        model = self.accelerator.unwrap_model(self.model)
        load_delta_mem_adapter(model, checkpoint)
        self._reset_online_state(model)


def parse_args() -> argparse.Namespace:
    default_optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    parser = argparse.ArgumentParser(description="Train Delta-Mem on a Hugging Face causal LM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--tokenized-dataset-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume, or 'latest'/'auto' for the newest complete checkpoint.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=_RESUME_MODES,
        default="exact",
        help=(
            "Use 'extend' for a larger horizon, or 'placement_ablation' for a paired "
            "fusion-placement fork. Use 'objective_ablation' only for the strict "
            "context-dropout to content-contrast training-objective fork."
        ),
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--tokenized-dataset-root", type=Path, default=None)
    parser.add_argument(
        "--tokenized-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--ddp-backend", default="nccl")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--memory-backend",
        default="delta_rule",
        choices=["delta_rule", "rwkv_ms"],
        help="Online memory state update backend. Both backends still emit q/k/v/o deltas.",
    )
    parser.add_argument("--rwkv-ms-num-states", type=int, default=4)
    parser.add_argument("--rwkv-ms-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--rwkv-ms-boundary-mode",
        default="fixed_chunk",
        choices=["fixed_chunk"],
    )
    parser.add_argument("--rwkv-ms-erase-gate", type=float, default=1.0)
    parser.add_argument("--rwkv-ms-read-top-k", type=int, default=0)
    parser.add_argument("--rwkv-ms-output-init-scale", type=float, default=0.02)
    parser.add_argument("--rwkv-ms-semantics-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--num-state-heads", type=int, default=1)
    parser.add_argument("--beta-bias-init", type=float, default=-1.5)
    parser.add_argument(
        "--couple-lambda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--state-update-mode",
        default="standard",
        choices=["standard", "lambda_outside", "no_lambda"],
    )
    parser.add_argument(
        "--output-init",
        default="base_slice",
        choices=["zero", "base_slice", "base_slice_fixed", "random"],
    )
    parser.add_argument("--base-slice-ref-width", type=int, default=8)
    parser.add_argument(
        "--delta-heads",
        default="q,k,v,o",
        help="Comma-separated subset of Delta attention heads to enable, e.g. q,k,o",
    )
    parser.add_argument(
        "--delta-o-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply RMSNorm to delta_o before adding it to the base attention output.",
    )
    parser.add_argument("--delta-o-rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--memory-fusion-mode",
        default="add",
        choices=["add", "content_gated_add"],
        help="Fuse delta_o additively or through a token-wise hidden/read content gate.",
    )
    parser.add_argument(
        "--memory-fusion-gate-init",
        type=float,
        default=0.1,
        help="Initial token gate probability for content_gated_add fusion.",
    )
    parser.add_argument(
        "--memory-fusion-placement",
        default="attention_output",
        choices=[
            "attention_output",
            "post_attention_norm",
            "normalized_residual_correction",
        ],
        help="Choose where the gated memory output enters Gemma's attention residual branch.",
    )
    parser.add_argument(
        "--memory-fusion-residual-scale",
        type=float,
        default=1.0,
        help="Interpolation scale for normalized_residual_correction fusion.",
    )
    parser.add_argument(
        "--trainable-delta-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn an extra bounded multiplier on top of the fixed alpha/rank scaling.",
    )
    parser.add_argument("--delta-scale-init", type=float, default=1.0)
    parser.add_argument("--delta-scale-max", type=float, default=2.0)
    parser.add_argument(
        "--delta-scale-granularity",
        default="layer",
        choices=["layer", "head"],
        help="Whether the learned delta scale is shared per layer or split per delta head.",
    )
    parser.add_argument(
        "--delta-scale-parameterization",
        default="alpha_over_rank",
        choices=["alpha_over_rank", "rank_over_alpha"],
        help="Base scaling formula used before any learned delta scale multiplier is applied.",
    )
    parser.add_argument("--online-gain", type=float, default=0.05)
    parser.add_argument(
        "--target-layers",
        default="off",
        help="Comma-separated attention layer indices to wrap with Delta-Mem. 'off' means all layers.",
    )
    parser.add_argument(
        "--memory-readout-mode",
        default="delta",
        choices=["delta"],
    )
    parser.add_argument(
        "--memory-write-source",
        default="learned_hidden",
        choices=["learned_hidden", "base_qkv"],
    )
    parser.add_argument(
        "--memory-write-granularity",
        default="token",
        choices=["token", "message_mean", "sentence_mean"],
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--training-mode",
        default="episode",
        choices=["dialogue", "episode"],
    )
    parser.add_argument(
        "--episode-recent-messages",
        type=int,
        default=4,
        help="Number of trailing non-system messages to keep visible during episode training.",
    )
    parser.add_argument(
        "--max-write-length",
        type=int,
        default=1024,
        help="Maximum number of write-history tokens kept in episode training.",
    )
    parser.add_argument(
        "--episode-read-write-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep online-memory writes enabled during the episode read/target forward.",
    )
    parser.add_argument(
        "--memory-loss-mode",
        default="context_dropout_ce",
        choices=[
            "none",
            "context_dropout_ce",
            "context_ablation_ce",
            "content_contrast_ce",
            "state_margin_kl",
            "latent_prefix_margin",
            "state_causal_anchor",
            "teacher_gap_kl",
            "teacher_kl_only",
            "teacher_kl_wmem1",
            "teacher_kl_wmem",
            "keep_only",
            "keep_full_kl",
            "keep_fullstate_kl",
            "keep_dual_kl",
        ],
    )
    parser.add_argument("--memory-contrast-weight", type=float, default=0.1)
    parser.add_argument("--memory-kl-weight", type=float, default=0.1)
    parser.add_argument("--memory-margin", type=float, default=0.1)
    parser.add_argument("--memory-causal-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-margin", type=float, default=0.005)
    parser.add_argument("--memory-recover-weight", type=float, default=0.25)
    parser.add_argument("--memory-need-floor", type=float, default=0.15)
    parser.add_argument("--memory-dropout-no-memory-prob", type=float, default=0.0)
    parser.add_argument("--memory-dropout-state-only-prob", type=float, default=0.0)
    parser.add_argument("--memory-base-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--context-ablation-mode",
        default="mixed",
        choices=["mixed", "full_context_no_state", "state_only", "full_context_plus_state"],
    )
    parser.add_argument("--context-ablation-no-state-prob", type=float, default=0.2)
    parser.add_argument("--context-ablation-state-only-prob", type=float, default=0.2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to --per-device-train-batch-size.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optim", default=default_optim)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--validation-split-ratio", type=float, default=0.0)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--frozen-mlp-activation-checkpointing",
        "--frozen-mlp-checkpointing",
        dest="frozen_mlp_activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checkpoint only frozen decoder MLPs. This reduces activation memory without "
            "recomputing stateful Delta-Mem attention."
        ),
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--deepspeed-config", type=Path, default=None)
    parser.add_argument("--write-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--write-sparsity-target", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="delta-mem-qwen3-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--log-delta-debug-stats", action="store_true")
    parser.add_argument(
        "--assistant-loss-mode",
        default="all_assistant_turns",
        choices=["all_assistant_turns", "final_assistant_only"],
    )
    parser.add_argument("--rankwise-gates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.num_memory_partitions = 1
    args.num_global_memory_partitions = 0
    args.memory_partition_routing = "soft"
    args.memory_partition_basis = "shared"
    args.tie_memory_partition_read_write = False
    args.memory_partition_read_mode = "softmax"
    args.memory_partition_sigmoid_gate_bias_init = -2.0
    args.slot_read_top_k = 0
    args.global_memory_mode = "shared_rw"
    args.global_memory_read_top_k = 0
    args.global_memory_merge_mode = "gated_residual"
    args.global_memory_gate_bias_init = -2.0
    args.global_memory_read_logit_bias = 0.0
    args.memory_write_proposals_per_message = 2
    args.memory_full_ce_weight = 0.0
    args.memory_full_ce_max_length = 2048
    args.memory_probe_weight = 0.0
    args.memory_probe_alpha = 0.4
    args.memory_probe_margin = 0.01
    args.memory_partition_alignment_weight = 0.0
    args.memory_partition_entropy_weight = 0.0
    args.memory_partition_balance_weight = 0.0
    if (
        args.memory_dropout_no_memory_prob < 0.0
        or args.memory_dropout_state_only_prob < 0.0
        or args.memory_dropout_no_memory_prob + args.memory_dropout_state_only_prob > 1.0
    ):
        raise ValueError(
            "memory dropout probabilities must satisfy p >= 0, q >= 0, p + q <= 1"
        )
    if (
        args.context_ablation_no_state_prob < 0.0
        or args.context_ablation_state_only_prob < 0.0
        or args.context_ablation_no_state_prob + args.context_ablation_state_only_prob > 1.0
    ):
        raise ValueError(
            "context ablation probabilities must satisfy p >= 0, q >= 0, p + q <= 1"
        )
    if args.memory_base_kl_weight < 0.0:
        raise ValueError("memory-base-kl-weight must be non-negative")
    if args.rwkv_ms_output_init_scale < 0.0:
        raise ValueError("rwkv-ms-output-init-scale must be non-negative")
    if args.memory_base_kl_weight > 0.0 and args.memory_loss_mode != "context_dropout_ce":
        raise ValueError("memory-base-kl-weight requires memory-loss-mode=context_dropout_ce")
    if args.memory_loss_mode == "content_contrast_ce":
        if args.episode_read_write_enabled:
            raise ValueError("content_contrast_ce requires episode read writes to be disabled")
        if args.memory_kl_weight != 0.0 or args.memory_base_kl_weight != 0.0:
            raise ValueError("content_contrast_ce requires all KL weights to be zero")
        if args.memory_contrast_weight < 0.0:
            raise ValueError("content_contrast_ce requires a non-negative contrast weight")
        if args.memory_margin < 0.0:
            raise ValueError("content_contrast_ce requires a non-negative margin")
        if args.write_sparsity_weight != 0.0:
            raise ValueError("content_contrast_ce requires write sparsity loss to be disabled")
    if args.memory_backend == "rwkv_ms" and args.output_init == "zero":
        raise ValueError(
            "output-init=zero is gradient-dead with memory-backend=rwkv_ms; "
            "use output-init=base_slice_fixed"
        )
    if not 0.0 <= args.validation_split_ratio < 1.0:
        raise ValueError("validation-split-ratio must satisfy 0 <= ratio < 1")
    if args.per_device_eval_batch_size is not None and args.per_device_eval_batch_size <= 0:
        raise ValueError("per-device-eval-batch-size must be positive")
    if args.eval_steps <= 0:
        raise ValueError("eval-steps must be positive")
    if args.save_total_limit is not None and args.save_total_limit <= 0:
        raise ValueError("save-total-limit must be positive")
    if args.load_best_model_at_end and args.validation_split_ratio == 0.0:
        raise ValueError("load-best-model-at-end requires a non-zero validation-split-ratio")
    if args.load_best_model_at_end and args.save_steps % args.eval_steps != 0:
        raise ValueError("save-steps must be a multiple of eval-steps when loading the best model")
    return args


def get_dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return table[name]


def parse_layer_indices(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw or raw.lower() in {"none", "off"}:
        return ()
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def validate_wrapped_target_layers(
    requested_layers: tuple[int, ...],
    wrapped_layers: tuple[int, ...],
) -> None:
    if not requested_layers:
        return
    if len(set(requested_layers)) != len(requested_layers):
        raise ValueError(f"Target layers contain duplicates: {requested_layers}")
    requested = set(requested_layers)
    wrapped = set(wrapped_layers)
    if wrapped != requested:
        missing = tuple(sorted(requested - wrapped))
        unexpected = tuple(sorted(wrapped - requested))
        raise ValueError(
            "Requested target layers do not match wrapped attention layers: "
            f"requested={tuple(sorted(requested))} "
            f"wrapped={tuple(sorted(wrapped))} "
            f"missing={missing} unexpected={unexpected}"
        )


def parse_delta_heads(raw: str) -> tuple[str, ...]:
    return normalize_delta_heads(raw)


def load_examples(args: argparse.Namespace) -> Dataset:
    cache_dir = str(args.hf_cache_dir) if args.hf_cache_dir is not None else None
    if args.train_file is not None:
        suffix = args.train_file.suffix.lower()
        if suffix == ".jsonl":
            dataset = load_dataset(
                "json",
                data_files=str(args.train_file),
                split="train",
                cache_dir=cache_dir,
            )
        elif suffix == ".json":
            loaded = json.loads(args.train_file.read_text())
            if isinstance(loaded, list):
                dataset = Dataset.from_list(loaded)
            elif isinstance(loaded, dict):
                dataset = Dataset.from_list([loaded])
            else:
                raise ValueError("Unsupported JSON format")
        else:
            raise ValueError(f"Unsupported train file: {args.train_file}")
        return dataset
    if args.dataset_name is not None:
        loaded = load_dataset(args.dataset_name, cache_dir=cache_dir)
        if isinstance(loaded, DatasetDict):
            return loaded[args.dataset_split]
        return loaded
    raise ValueError("Provide either --train-file or --dataset-name")


def normalize_example(example: dict) -> SFTExample:
    if "messages" in example:
        messages = example["messages"]
        if not messages:
            raise ValueError("messages examples must not be empty")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("messages examples must contain at least one assistant turn")
        return SFTExample(messages=[dict(message) for message in messages])
    if "prompt" in example and "response" in example:
        return SFTExample(
            messages=[
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ],
        )
    raise ValueError("Each example must have either `messages` or `prompt`/`response`.")


def _tokenize_chat_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    tokenized = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    return tokenized.squeeze(0).tolist()


def _tokenize_text_no_special_tokens(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_subsequence_start(haystack: list[int], needle: list[int]) -> int | None:
    if not needle:
        return 0
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


_MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS = 16


def _longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _chat_template_delta(
    previous_ids: list[int],
    current_ids: list[int],
    *,
    error_message: str,
) -> tuple[int, list[int]]:
    prefix_len = _longest_common_prefix_length(previous_ids, current_ids)
    rollback_tokens = len(previous_ids) - prefix_len
    if rollback_tokens > _MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS:
        raise ValueError(error_message)
    return prefix_len, current_ids[prefix_len:]


def _sentence_ids_for_message_delta(
    tokenizer,
    message_content: str,
    delta_ids: list[int],
    next_sentence_id: int,
) -> tuple[list[int], int]:
    sentence_ids = [-1] * len(delta_ids)
    content_ids = _tokenize_text_no_special_tokens(tokenizer, message_content)
    if not content_ids:
        return sentence_ids, next_sentence_id
    content_start = _find_subsequence_start(delta_ids, content_ids)
    if content_start is None:
        return sentence_ids, next_sentence_id
    sentence_chunks = split_text_into_sentence_token_chunks(message_content)
    sentence_chunk_ids = [
        _tokenize_text_no_special_tokens(tokenizer, sentence_chunk)
        for sentence_chunk in sentence_chunks
    ]
    flat_sentence_ids = [token_id for chunk_ids in sentence_chunk_ids for token_id in chunk_ids]
    if flat_sentence_ids != content_ids:
        sentence_ids[content_start : content_start + len(content_ids)] = [next_sentence_id] * len(content_ids)
        return sentence_ids, next_sentence_id + 1
    position = content_start
    for chunk_ids in sentence_chunk_ids:
        if not chunk_ids:
            continue
        sentence_ids[position : position + len(chunk_ids)] = [next_sentence_id] * len(chunk_ids)
        position += len(chunk_ids)
        next_sentence_id += 1
    return sentence_ids, next_sentence_id


def _tokenize_chat_messages_with_write_span_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    include_sentence_ids: bool,
) -> tuple[list[int], list[int], list[int]]:
    input_ids: list[int] = []
    message_ids: list[int] = []
    sentence_ids: list[int] = []
    previous_ids: list[int] = []
    next_message_id = 0
    next_sentence_id = 0
    for index, message in enumerate(messages):
        current_ids = _tokenize_chat_messages(tokenizer, messages[: index + 1])
        prefix_len, delta_ids = _chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot recover write message spans safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del message_ids[prefix_len:]
            del sentence_ids[prefix_len:]
        input_ids.extend(delta_ids)
        if message["role"] == "system":
            message_ids.extend([-1] * len(delta_ids))
            sentence_ids.extend([-1] * len(delta_ids))
            previous_ids = current_ids
            continue

        message_id = next_message_id
        next_message_id += 1
        message_ids.extend([message_id] * len(delta_ids))
        if include_sentence_ids:
            message_sentence_ids, next_sentence_id = _sentence_ids_for_message_delta(
                tokenizer,
                message["content"],
                delta_ids,
                next_sentence_id,
            )
            sentence_ids.extend(message_sentence_ids)
        else:
            sentence_ids.extend([-1] * len(delta_ids))
        previous_ids = current_ids
    return input_ids, message_ids, sentence_ids


def _tokenize_chat_messages_with_message_ids(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    input_ids, message_ids, _ = _tokenize_chat_messages_with_write_span_ids(
        tokenizer,
        messages,
        include_sentence_ids=False,
    )
    return input_ids, message_ids


def _truncate_sft_sequence(
    input_ids: list[int],
    labels: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if len(input_ids) <= max_length:
        return input_ids, labels
    start = len(input_ids) - max_length
    return input_ids[start:], labels[start:]


def _select_supervised_assistant_indices(
    messages: list[dict[str, str]],
    assistant_loss_mode: str,
) -> list[int]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No assistant turns found for supervision")
    if assistant_loss_mode == "all_assistant_turns":
        return assistant_indices
    if assistant_loss_mode == "final_assistant_only":
        return [assistant_indices[-1]]
    raise ValueError(f"Unsupported assistant_loss_mode: {assistant_loss_mode}")


def _split_system_prefix(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_prefix: list[dict[str, str]] = []
    for message in messages:
        if message["role"] != "system":
            break
        system_prefix.append(dict(message))
    return system_prefix, [dict(message) for message in messages[len(system_prefix) :]]


def tokenize_messages_for_sft(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    *,
    assistant_loss_mode: str,
) -> dict:
    supervised_assistant_indices = set(
        _select_supervised_assistant_indices(messages, assistant_loss_mode)
    )

    input_ids: list[int] = []
    labels: list[int] = []
    previous_ids: list[int] = []
    for index in range(len(messages)):
        current_ids = _tokenize_chat_messages(tokenizer, messages[: index + 1])
        prefix_len, delta_ids = _chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot build assistant-span labels safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del labels[prefix_len:]
        input_ids.extend(delta_ids)
        if index in supervised_assistant_indices:
            labels.extend(delta_ids)
        else:
            labels.extend([-100] * len(delta_ids))
        previous_ids = current_ids

    input_ids, labels = _truncate_sft_sequence(input_ids, labels, max_length)
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _mask_teacher_labels_to_student_targets(
    teacher_labels: list[int],
    student_labels: list[int],
) -> list[int]:
    student_targets = [label for label in student_labels[1:] if label != -100]
    teacher_positions = [
        index
        for index, label in enumerate(teacher_labels[1:], start=1)
        if label != -100
    ]
    teacher_targets = [teacher_labels[index] for index in teacher_positions]
    if len(teacher_targets) < len(student_targets):
        raise ValueError(
            "Canonical teacher has fewer supervised next-token targets than the episode read"
        )
    if student_targets and teacher_targets[-len(student_targets) :] != student_targets:
        raise ValueError(
            "Canonical teacher supervised target token IDs do not match the episode read suffix"
        )
    masked_labels = [-100] * len(teacher_labels)
    if not student_targets:
        return masked_labels
    for position in teacher_positions[-len(student_targets) :]:
        masked_labels[position] = teacher_labels[position]
    return masked_labels


def _build_canonical_teacher_features(
    tokenizer,
    teacher_messages: list[dict[str, str]],
    full_write_input_ids: list[int],
    retained_write_length: int,
    *,
    max_write_length: int,
    max_read_length: int,
    student_labels: list[int],
) -> dict[str, list[int]]:
    canonical_input_ids = _tokenize_chat_messages(tokenizer, teacher_messages)
    teacher_features = tokenize_messages_for_sft(
        tokenizer,
        teacher_messages,
        max(len(canonical_input_ids), 1),
        assistant_loss_mode="final_assistant_only",
    )
    if teacher_features["input_ids"] != canonical_input_ids:
        raise ValueError("Canonical teacher tokenization changed between full-sequence passes")

    teacher_start = 0
    if full_write_input_ids:
        stable_prefix_length, _ = _chat_template_delta(
            full_write_input_ids,
            canonical_input_ids,
            error_message=(
                "Chat template write prefix is not stable enough to align the canonical teacher"
            ),
        )
        teacher_start = len(full_write_input_ids) - retained_write_length
        if teacher_start > stable_prefix_length:
            raise ValueError(
                "Retained write boundary falls inside the chat template's rewritten suffix"
            )

    teacher_max_length = max_write_length + max_read_length
    teacher_start = max(
        teacher_start,
        len(canonical_input_ids) - teacher_max_length,
    )
    teacher_features = {
        key: value[teacher_start:]
        for key, value in teacher_features.items()
    }
    teacher_features["labels"] = _mask_teacher_labels_to_student_targets(
        teacher_features["labels"],
        student_labels,
    )
    return teacher_features


def build_episode_training_examples(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    *,
    assistant_loss_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
    include_sentence_ids: bool,
) -> list[dict]:
    if episode_recent_messages < 0:
        raise ValueError("episode_recent_messages must be >= 0")
    if max_write_length <= 0:
        raise ValueError("max_write_length must be > 0")

    episodes: list[dict] = []
    for target_index in _select_supervised_assistant_indices(messages, assistant_loss_mode):
        prefix_messages = [dict(message) for message in messages[:target_index]]
        system_prefix, non_system_prefix = _split_system_prefix(prefix_messages)
        if episode_recent_messages == 0:
            visible_non_system = []
            write_non_system = non_system_prefix
        else:
            visible_non_system = non_system_prefix[-episode_recent_messages:]
            write_non_system = non_system_prefix[:-episode_recent_messages]

        write_messages = system_prefix + write_non_system if write_non_system else []
        write_input_ids: list[int] = []
        write_message_ids: list[int] = []
        write_sentence_ids: list[int] = []
        full_write_input_ids: list[int] = []
        if write_messages:
            full_write_input_ids, write_message_ids, write_sentence_ids = _tokenize_chat_messages_with_write_span_ids(
                tokenizer,
                write_messages,
                include_sentence_ids=include_sentence_ids,
            )
            write_input_ids, write_message_ids = _truncate_sft_sequence(
                full_write_input_ids,
                write_message_ids,
                max_write_length,
            )
            _, write_sentence_ids = _truncate_sft_sequence(
                full_write_input_ids,
                write_sentence_ids,
                max_write_length,
            )

        read_messages = system_prefix + visible_non_system + [dict(messages[target_index])]
        read_features = tokenize_messages_for_sft(
            tokenizer,
            read_messages,
            max_length,
            assistant_loss_mode="final_assistant_only",
        )
        teacher_features = _build_canonical_teacher_features(
            tokenizer,
            prefix_messages + [dict(messages[target_index])],
            full_write_input_ids,
            len(write_input_ids),
            max_write_length=max_write_length,
            max_read_length=max_length,
            student_labels=read_features["labels"],
        )
        # Keep the immediate query turn visible during state-only dropout so memory focuses on
        # far-history recall instead of reconstructing the local prompt from state.
        state_only_visible_non_system = non_system_prefix[-1:] if non_system_prefix else []
        state_only_write_non_system = non_system_prefix[:-1] if non_system_prefix else []
        state_only_write_messages = (
            system_prefix + state_only_write_non_system if state_only_write_non_system else []
        )
        state_only_write_input_ids: list[int] = []
        state_only_write_message_ids: list[int] = []
        state_only_write_sentence_ids: list[int] = []
        if state_only_write_messages:
            (
                full_state_only_write_input_ids,
                state_only_write_message_ids,
                state_only_write_sentence_ids,
            ) = _tokenize_chat_messages_with_write_span_ids(
                tokenizer,
                state_only_write_messages,
                include_sentence_ids=include_sentence_ids,
            )
            state_only_write_input_ids, state_only_write_message_ids = _truncate_sft_sequence(
                full_state_only_write_input_ids,
                state_only_write_message_ids,
                max_write_length,
            )
            _, state_only_write_sentence_ids = _truncate_sft_sequence(
                full_state_only_write_input_ids,
                state_only_write_sentence_ids,
                max_write_length,
            )
        state_only_read_messages = (
            system_prefix + state_only_visible_non_system + [dict(messages[target_index])]
        )
        state_only_read_features = tokenize_messages_for_sft(
            tokenizer,
            state_only_read_messages,
            max_length,
            assistant_loss_mode="final_assistant_only",
        )

        episodes.append(
            {
                "write_input_ids": write_input_ids,
                "write_attention_mask": [1] * len(write_input_ids),
                "write_message_ids": write_message_ids,
                "write_sentence_ids": write_sentence_ids,
                "input_ids": read_features["input_ids"],
                "attention_mask": read_features["attention_mask"],
                "labels": read_features["labels"],
                "teacher_input_ids": teacher_features["input_ids"],
                "teacher_attention_mask": teacher_features["attention_mask"],
                "teacher_labels": teacher_features["labels"],
                "state_only_write_input_ids": state_only_write_input_ids,
                "state_only_write_attention_mask": [1] * len(state_only_write_input_ids),
                "state_only_write_message_ids": state_only_write_message_ids,
                "state_only_write_sentence_ids": state_only_write_sentence_ids,
                "state_only_input_ids": state_only_read_features["input_ids"],
                "state_only_attention_mask": state_only_read_features["attention_mask"],
                "state_only_labels": state_only_read_features["labels"],
                "episode_target_message_index": target_index,
                "write_message_count": len(write_messages),
                "visible_message_count": len(read_messages) - 1,
            }
        )
    return episodes


def tokenize_example(
    tokenizer,
    example: dict,
    max_length: int,
    *,
    assistant_loss_mode: str,
    training_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
) -> dict:
    normalized = normalize_example(example)
    if training_mode == "episode":
        raise ValueError("tokenize_example does not support episode mode; use tokenize_examples_batch")
    if training_mode != "dialogue":
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    return tokenize_messages_for_sft(
        tokenizer,
        normalized.messages,
        max_length,
        assistant_loss_mode=assistant_loss_mode,
    )


def add_length_column(example: dict) -> dict[str, int]:
    total_length = len(example["input_ids"])
    if "write_input_ids" in example:
        total_length += len(example["write_input_ids"])
    return {"length": total_length}


def tokenize_examples_batch(
    tokenizer,
    batch: dict[str, list],
    max_length: int,
    *,
    assistant_loss_mode: str,
    training_mode: str,
    episode_recent_messages: int,
    max_write_length: int,
    include_sentence_ids: bool,
) -> dict[str, list]:
    tokenized: dict[str, list] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    if training_mode == "episode":
        tokenized["write_input_ids"] = []
        tokenized["write_attention_mask"] = []
        tokenized["write_message_ids"] = []
        tokenized["write_sentence_ids"] = []
        tokenized["teacher_input_ids"] = []
        tokenized["teacher_attention_mask"] = []
        tokenized["teacher_labels"] = []
        tokenized["state_only_write_input_ids"] = []
        tokenized["state_only_write_attention_mask"] = []
        tokenized["state_only_write_message_ids"] = []
        tokenized["state_only_write_sentence_ids"] = []
        tokenized["state_only_input_ids"] = []
        tokenized["state_only_attention_mask"] = []
        tokenized["state_only_labels"] = []
        tokenized["episode_target_message_index"] = []
        tokenized["write_message_count"] = []
        tokenized["visible_message_count"] = []

    batch_size = len(next(iter(batch.values())))
    for row_index in range(batch_size):
        example = {key: value[row_index] for key, value in batch.items()}
        normalized = normalize_example(example)
        if training_mode == "dialogue":
            features = tokenize_messages_for_sft(
                tokenizer,
                normalized.messages,
                max_length,
                assistant_loss_mode=assistant_loss_mode,
            )
            for key, value in features.items():
                tokenized[key].append(value)
            continue
        if training_mode != "episode":
            raise ValueError(f"Unsupported training_mode: {training_mode}")
        for episode in build_episode_training_examples(
            tokenizer,
            normalized.messages,
            max_length,
            assistant_loss_mode=assistant_loss_mode,
            episode_recent_messages=episode_recent_messages,
            max_write_length=max_write_length,
            include_sentence_ids=include_sentence_ids,
        ):
            for key, value in episode.items():
                tokenized[key].append(value)
    return tokenized


def _tokenizer_cache_identity(tokenizer) -> dict[str, object]:
    def normalize(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return str(value)

    name_or_path = str(getattr(tokenizer, "name_or_path", ""))
    effective_chat_template = resolve_effective_chat_template(tokenizer)
    special_token_ids = {
        attribute: normalize(getattr(tokenizer, attribute, None))
        for attribute in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "sep_token_id",
            "cls_token_id",
            "mask_token_id",
            "additional_special_tokens_ids",
        )
    }
    identity: dict[str, object] = {
        "name_or_path": name_or_path,
        "class": tokenizer.__class__.__name__,
        "chat_template": normalize(effective_chat_template),
        "vocab_size": normalize(getattr(tokenizer, "vocab_size", None)),
        "model_max_length": normalize(getattr(tokenizer, "model_max_length", None)),
        "padding_side": normalize(getattr(tokenizer, "padding_side", None)),
        "truncation_side": normalize(getattr(tokenizer, "truncation_side", None)),
        "special_tokens_map": normalize(getattr(tokenizer, "special_tokens_map", None)),
        "special_token_ids": special_token_ids,
    }

    tokenizer_path = Path(name_or_path).expanduser()
    artifact_files: list[Path] = []
    if name_or_path and tokenizer_path.is_file():
        artifact_files = [tokenizer_path]
        artifact_root = tokenizer_path.parent
    elif name_or_path and tokenizer_path.is_dir():
        artifact_root = tokenizer_path
        for candidate in tokenizer_path.iterdir():
            if not candidate.is_file():
                continue
            name = candidate.name.lower()
            if (
                name.startswith(
                    (
                        "tokenizer",
                        "vocab",
                        "merges",
                        "special_tokens",
                        "added_tokens",
                    )
                )
                or name.endswith((".model", ".spm"))
            ):
                artifact_files.append(candidate)
    else:
        artifact_root = None

    if artifact_files and artifact_root is not None:
        artifact_hash = hashlib.sha256()
        artifact_names = []
        for artifact in sorted(artifact_files, key=lambda path: path.name):
            relative_name = artifact.relative_to(artifact_root).as_posix()
            artifact_names.append(relative_name)
            artifact_hash.update(relative_name.encode("utf-8"))
            artifact_hash.update(b"\0")
            artifact_hash.update(str(artifact.stat().st_size).encode("ascii"))
            artifact_hash.update(b"\0")
            with artifact.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    artifact_hash.update(chunk)
        identity["local_artifact_files"] = artifact_names
        identity["local_artifacts_sha256"] = artifact_hash.hexdigest()
    else:
        identity["local_artifact_files"] = []
        identity["local_artifacts_sha256"] = None
    return identity


def _tokenized_dataset_cache_key(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
) -> str:
    code_hash = hashlib.sha256()
    for fn in (
        normalize_example,
        tokenize_messages_for_sft,
        _mask_teacher_labels_to_student_targets,
        _build_canonical_teacher_features,
        _sentence_ids_for_message_delta,
        _tokenize_chat_messages,
        _tokenize_chat_messages_with_write_span_ids,
        build_episode_training_examples,
        tokenize_examples_batch,
        add_length_column,
        _tokenizer_cache_identity,
        split_text_into_sentence_token_chunks,
    ):
        code_hash.update(inspect.getsource(fn).encode("utf-8"))
    code_hash.update(inspect.getsource(project_chat_templates).encode("utf-8"))
    include_sentence_ids = args.memory_write_granularity == "sentence_mean"
    payload = {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "train_file": None if args.train_file is None else str(args.train_file.resolve()),
        "tokenizer_identity": _tokenizer_cache_identity(tokenizer),
        "max_length": args.max_length,
        "training_mode": args.training_mode,
        "assistant_loss_mode": args.assistant_loss_mode,
        "episode_recent_messages": args.episode_recent_messages,
        "max_write_length": args.max_write_length,
        "memory_write_granularity": args.memory_write_granularity,
        "include_sentence_ids": include_sentence_ids,
        "group_by_length": args.group_by_length,
        "code_hash": code_hash.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _build_tokenized_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    if args.training_mode == "dialogue":
        tokenized = dataset.map(
            lambda example: tokenize_example(
                tokenizer,
                example,
                args.max_length,
                assistant_loss_mode=args.assistant_loss_mode,
                training_mode=args.training_mode,
                episode_recent_messages=args.episode_recent_messages,
                max_write_length=args.max_write_length,
            ),
            remove_columns=dataset.column_names,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    elif args.training_mode == "episode":
        tokenized = dataset.map(
            lambda batch: tokenize_examples_batch(
                tokenizer,
                batch,
                args.max_length,
                assistant_loss_mode=args.assistant_loss_mode,
                training_mode=args.training_mode,
                episode_recent_messages=args.episode_recent_messages,
                max_write_length=args.max_write_length,
                include_sentence_ids=args.memory_write_granularity == "sentence_mean",
            ),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    else:
        raise ValueError(f"Unsupported training_mode: {args.training_mode}")
    if args.group_by_length:
        tokenized = tokenized.map(
            add_length_column,
            num_proc=None if args.dataset_num_proc <= 1 else args.dataset_num_proc,
        )
    return tokenized


def prepare_tokenized_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[Dataset, bool, Path | None]:
    if not args.tokenized_cache:
        return _build_tokenized_dataset(args, dataset, tokenizer), False, None
    if args.tokenized_dataset_root is None:
        raise ValueError("--tokenized-dataset-root is required when --tokenized-cache is enabled")

    args.tokenized_dataset_root.mkdir(parents=True, exist_ok=True)
    cache_key = _tokenized_dataset_cache_key(args, dataset, tokenizer)
    cache_dir = args.tokenized_dataset_root / cache_key
    ready_marker = cache_dir / "_READY"
    lock_dir = args.tokenized_dataset_root / f".{cache_key}.lock"
    is_builder = (not distributed) or local_rank in (-1, 0)

    if ready_marker.exists():
        return load_from_disk(str(cache_dir)), True, cache_dir

    if is_builder:
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                if ready_marker.exists():
                    return load_from_disk(str(cache_dir)), True, cache_dir
                time.sleep(2)

        try:
            if cache_dir.exists() and not ready_marker.exists():
                shutil.rmtree(cache_dir)
            temp_dir = args.tokenized_dataset_root / f".{cache_key}.tmp-{os.getpid()}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            tokenized = _build_tokenized_dataset(args, dataset, tokenizer)
            built_fingerprint = getattr(tokenized, "_fingerprint", None)
            tokenized.save_to_disk(str(temp_dir))
            temp_dir.rename(cache_dir)
            # Dataset.save_to_disk can assign a different Arrow fingerprint on
            # reload. Train from the persisted view so checkpoints and cache-hit
            # resumes record the same fingerprint.
            tokenized = load_from_disk(str(cache_dir))
            ready_marker.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "created_at": time.time(),
                        "training_mode": args.training_mode,
                        "group_by_length": args.group_by_length,
                        "assistant_loss_mode": args.assistant_loss_mode,
                        "episode_recent_messages": args.episode_recent_messages,
                        "max_write_length": args.max_write_length,
                        "memory_write_granularity": args.memory_write_granularity,
                        "include_sentence_ids": args.memory_write_granularity == "sentence_mean",
                        "max_length": args.max_length,
                        "built_fingerprint": built_fingerprint,
                        "saved_fingerprint": getattr(tokenized, "_fingerprint", None),
                    },
                    indent=2,
                )
            )
            return tokenized, False, cache_dir
        finally:
            if lock_dir.exists():
                lock_dir.rmdir()

    waited = 0
    while not ready_marker.exists():
        time.sleep(2)
        waited += 2
        if waited > 7200:
            raise TimeoutError(f"Timed out waiting for tokenized dataset cache at {cache_dir}")
    return load_from_disk(str(cache_dir)), True, cache_dir


def detect_training_mode(tokenized: Dataset) -> str:
    if "write_input_ids" in tokenized.column_names:
        return "episode"
    return "dialogue"


def load_or_prepare_tokenized_dataset(
    args: argparse.Namespace,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[Dataset, dict[str, object]]:
    if args.tokenized_dataset_dir is not None:
        tokenized = load_from_disk(str(args.tokenized_dataset_dir))
        return tokenized, {
            "tokenized_cache_hit": True,
            "tokenized_cache_dir": str(args.tokenized_dataset_dir),
            "tokenized_dataset_source": "load_from_disk",
            "train_samples": len(tokenized),
            "training_mode": detect_training_mode(tokenized),
        }

    dataset = load_examples(args)
    tokenized, tokenized_cache_hit, tokenized_cache_dir = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=distributed,
        local_rank=local_rank,
    )
    return tokenized, {
        "tokenized_cache_hit": tokenized_cache_hit,
        "tokenized_cache_dir": None if tokenized_cache_dir is None else str(tokenized_cache_dir),
        "tokenized_dataset_source": "prepared_cache" if args.tokenized_cache else "direct_map",
        "train_samples": len(tokenized),
        "training_mode": detect_training_mode(tokenized),
    }


def split_tokenized_dataset(
    tokenized: Dataset,
    *,
    validation_split_ratio: float,
    data_seed: int,
) -> tuple[Dataset, Dataset | None]:
    if not 0.0 <= validation_split_ratio < 1.0:
        raise ValueError("validation_split_ratio must satisfy 0 <= ratio < 1")
    if validation_split_ratio == 0.0:
        return tokenized, None
    if len(tokenized) < 2:
        raise ValueError("A validation split requires at least two tokenized samples")
    validation_samples = max(1, math.ceil(len(tokenized) * validation_split_ratio))
    if validation_samples >= len(tokenized):
        raise ValueError("validation_split_ratio leaves no training samples")
    split = tokenized.train_test_split(
        test_size=validation_samples,
        seed=data_seed,
        shuffle=True,
    )
    return split["train"], split["test"]


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_contrast_write_payload(
    row: dict,
    *,
    column_prefix: str = "write_",
) -> dict[str, list[int]]:
    field_names = (
        "input_ids",
        "attention_mask",
        "message_ids",
        "sentence_ids",
    )
    required_columns = tuple(f"{column_prefix}{field}" for field in field_names)
    missing = [column for column in required_columns if column not in row]
    if missing:
        raise ValueError(
            "content_contrast_ce requires episode write columns: " + ", ".join(missing)
        )
    payload = {
        field: [int(value) for value in row[f"{column_prefix}{field}"]]
        for field in field_names
    }
    write_length = len(payload["input_ids"])
    if write_length == 0:
        raise ValueError("content_contrast_ce requires every sample to have a non-empty write")
    for column, values in payload.items():
        if len(values) != write_length:
            raise ValueError(
                f"content_contrast_ce write field {column} does not align with input_ids"
            )
    return payload


def materialize_content_contrast_pairs(
    split: Dataset,
    *,
    split_name: str,
) -> tuple[Dataset, dict[str, object]]:
    sample_count = len(split)
    if sample_count < 2 or sample_count % 2 != 0:
        raise ValueError(
            f"content_contrast_ce split {split_name!r} requires an even sample count >= 2; "
            f"got {sample_count}"
        )
    materialized_columns = {
        "negative_write_input_ids",
        "negative_write_attention_mask",
        "negative_write_message_ids",
        "negative_write_sentence_ids",
        "content_contrast_source_index",
        "content_contrast_partner_index",
        "content_contrast_source_id",
        "content_contrast_partner_id",
        "content_contrast_source_write_sha256",
        "content_contrast_partner_write_sha256",
        "content_contrast_negative_write_sha256",
    }
    collisions = sorted(materialized_columns.intersection(split.column_names))
    if collisions:
        raise ValueError(
            "content_contrast_ce pairing must be materialized from the objective-neutral "
            "post-split dataset; columns already exist: " + ", ".join(collisions)
        )

    source_fingerprint = getattr(split, "_fingerprint", None)
    rotation = sample_count // 2
    rows = [split[index] for index in range(sample_count)]
    write_payloads = [_content_contrast_write_payload(row) for row in rows]
    write_hashes = [_canonical_json_sha256(payload) for payload in write_payloads]
    partner_indices = [
        (source_index + rotation) % sample_count
        for source_index in range(sample_count)
    ]
    pair_audit: list[dict[str, object]] = []
    negative_columns: dict[str, list[list[int]]] = {
        "negative_write_input_ids": [],
        "negative_write_attention_mask": [],
        "negative_write_message_ids": [],
        "negative_write_sentence_ids": [],
    }
    source_ids: list[str] = []
    partner_ids: list[str] = []
    source_hashes: list[str] = []
    partner_hashes: list[str] = []
    for source_index, partner_index in enumerate(partner_indices):
        if source_index == partner_index:
            raise ValueError("content_contrast_ce pairing produced a self-pair")
        source_write = write_payloads[source_index]["input_ids"]
        partner_write = write_payloads[partner_index]["input_ids"]
        if source_write == partner_write:
            raise ValueError(
                "content_contrast_ce pairing produced equal writes for "
                f"{split_name}:{source_index} and {split_name}:{partner_index}"
            )
        source_id = f"{split_name}:{source_index}"
        partner_id = f"{split_name}:{partner_index}"
        source_hash = write_hashes[source_index]
        partner_hash = write_hashes[partner_index]
        source_ids.append(source_id)
        partner_ids.append(partner_id)
        source_hashes.append(source_hash)
        partner_hashes.append(partner_hash)
        for source_field, negative_column in (
            ("input_ids", "negative_write_input_ids"),
            ("attention_mask", "negative_write_attention_mask"),
            ("message_ids", "negative_write_message_ids"),
            ("sentence_ids", "negative_write_sentence_ids"),
        ):
            negative_columns[negative_column].append(
                write_payloads[partner_index][source_field]
            )
        pair_audit.append(
            {
                "source_index": source_index,
                "partner_index": partner_index,
                "source_id": source_id,
                "partner_id": partner_id,
                "source_write_sha256": source_hash,
                "partner_write_sha256": partner_hash,
            }
        )

    paired = split
    for column, values in negative_columns.items():
        paired = paired.add_column(column, values)
    materialized_negative_hashes = []
    for source_index in range(sample_count):
        materialized_payload = _content_contrast_write_payload(
            paired[source_index],
            column_prefix="negative_write_",
        )
        materialized_hash = _canonical_json_sha256(materialized_payload)
        if materialized_hash != partner_hashes[source_index]:
            raise RuntimeError(
                "content_contrast_ce materialized negative write does not match its audited donor"
            )
        materialized_negative_hashes.append(materialized_hash)
        pair_audit[source_index]["negative_write_sha256"] = materialized_hash
    paired = paired.add_column("content_contrast_source_index", list(range(sample_count)))
    paired = paired.add_column("content_contrast_partner_index", partner_indices)
    paired = paired.add_column("content_contrast_source_id", source_ids)
    paired = paired.add_column("content_contrast_partner_id", partner_ids)
    paired = paired.add_column("content_contrast_source_write_sha256", source_hashes)
    paired = paired.add_column("content_contrast_partner_write_sha256", partner_hashes)
    paired = paired.add_column(
        "content_contrast_negative_write_sha256",
        materialized_negative_hashes,
    )

    split_manifest: dict[str, object] = {
        "split": split_name,
        "pairing_version": _CONTENT_CONTRAST_PAIRING_VERSION,
        "sample_count": sample_count,
        "rotation": rotation,
        "source_fingerprint": source_fingerprint,
        "paired_fingerprint": getattr(paired, "_fingerprint", None),
        "pairs_sha256": _canonical_json_sha256(pair_audit),
        "pairs": pair_audit,
    }
    split_manifest["manifest_sha256"] = _canonical_json_sha256(split_manifest)
    return paired, split_manifest


def build_content_contrast_pairing_manifest(
    *,
    tokenized_fingerprint: str | None,
    data_seed: int,
    train_manifest: dict[str, object],
    eval_manifest: dict[str, object] | None,
) -> dict[str, object]:
    splits = {"train": train_manifest}
    if eval_manifest is not None:
        splits["eval"] = eval_manifest
    manifest: dict[str, object] = {
        "schema_version": 1,
        "objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "pairing_version": _CONTENT_CONTRAST_PAIRING_VERSION,
        "pairing_scope": "within_post_split_partition",
        "data_seed": data_seed,
        "tokenized_fingerprint": tokenized_fingerprint,
        "splits": splits,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def _content_contrast_protocol_pairing_summary(
    manifest: dict[str, object],
) -> dict[str, object]:
    split_summaries = {}
    for split_name, split_manifest in manifest["splits"].items():
        split_summaries[split_name] = {
            key: split_manifest[key]
            for key in (
                "sample_count",
                "rotation",
                "source_fingerprint",
                "paired_fingerprint",
                "pairs_sha256",
                "manifest_sha256",
            )
        }
    return {
        "pairing_version": manifest["pairing_version"],
        "pairing_scope": manifest["pairing_scope"],
        "data_seed": manifest["data_seed"],
        "tokenized_fingerprint": manifest["tokenized_fingerprint"],
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": split_summaries,
    }


def validate_canonical_teacher_columns(tokenized: Dataset) -> None:
    required_columns = {
        "teacher_input_ids",
        "teacher_attention_mask",
        "teacher_labels",
    }
    missing_columns = sorted(required_columns.difference(tokenized.column_names))
    if missing_columns:
        raise ValueError(
            "Tokenized episode dataset is missing canonical teacher columns; rebuild it with "
            "the current trainer: " + ", ".join(missing_columns)
        )


def build_training_protocol(
    args: argparse.Namespace,
    tokenized: Dataset,
    *,
    effective_training_mode: str,
    train_samples: int,
    eval_samples: int,
    warmup_steps: int,
    content_contrast_pairing_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    is_content_contrast = args.memory_loss_mode == "content_contrast_ce"
    protocol = {
        "schema_version": (
            _CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION
            if is_content_contrast
            else _TRAINING_PROTOCOL_SCHEMA_VERSION
        ),
        "memory_objective_version": (
            _CONTENT_CONTRAST_OBJECTIVE_VERSION
            if is_content_contrast
            else _MEMORY_OBJECTIVE_VERSION
        ),
        "train_file": None if args.train_file is None else str(args.train_file.resolve()),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "tokenized_dataset_dir": (
            None
            if args.tokenized_dataset_dir is None
            else str(args.tokenized_dataset_dir.resolve())
        ),
        "tokenized_fingerprint": getattr(tokenized, "_fingerprint", None),
        "tokenized_samples": len(tokenized),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "training_mode": effective_training_mode,
        "assistant_loss_mode": args.assistant_loss_mode,
        "max_length": args.max_length,
        "max_write_length": args.max_write_length,
        "teacher_max_length": args.max_write_length + args.max_length,
        "episode_recent_messages": args.episode_recent_messages,
        "episode_read_write_enabled": args.episode_read_write_enabled,
        "memory_write_source": args.memory_write_source,
        "memory_write_granularity": args.memory_write_granularity,
        "memory_fusion_mode": getattr(args, "memory_fusion_mode", "add"),
        "memory_fusion_gate_init": getattr(args, "memory_fusion_gate_init", 0.1),
        "memory_fusion_placement": getattr(
            args,
            "memory_fusion_placement",
            "attention_output",
        ),
        "memory_fusion_residual_scale": getattr(
            args,
            "memory_fusion_residual_scale",
            1.0,
        ),
        "rwkv_ms_output_init_scale": getattr(args, "rwkv_ms_output_init_scale", 0.02),
        "rwkv_ms_semantics_version": getattr(args, "rwkv_ms_semantics_version", 2),
        "memory_loss_mode": args.memory_loss_mode,
        "memory_dropout_no_memory_prob": args.memory_dropout_no_memory_prob,
        "memory_dropout_state_only_prob": args.memory_dropout_state_only_prob,
        "memory_base_kl_weight": args.memory_base_kl_weight,
        "context_ablation_mode": args.context_ablation_mode,
        "context_ablation_no_state_prob": args.context_ablation_no_state_prob,
        "context_ablation_state_only_prob": args.context_ablation_state_only_prob,
        "validation_split_ratio": args.validation_split_ratio,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": (
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size is not None
            else args.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "dtype": args.dtype,
        "bf16": args.bf16,
        "tf32": args.tf32,
    }
    protocol["frozen_mlp_activation_checkpointing"] = bool(
        getattr(args, "frozen_mlp_activation_checkpointing", False)
    )
    if is_content_contrast:
        if content_contrast_pairing_manifest is None:
            raise ValueError("content_contrast_ce requires a post-split pairing manifest")
        protocol.update(
            {
                "memory_contrast_weight": args.memory_contrast_weight,
                "memory_margin": args.memory_margin,
                "memory_kl_weight": args.memory_kl_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": args.memory_partition_alignment_weight,
                "memory_partition_entropy_weight": args.memory_partition_entropy_weight,
                "memory_partition_balance_weight": args.memory_partition_balance_weight,
                "content_contrast_negative_priming_grad": False,
                "content_contrast_pairing": _content_contrast_protocol_pairing_summary(
                    content_contrast_pairing_manifest
                ),
            }
        )
    return protocol


class DialogueCausalLMCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EpisodeCausalLMCollator(DialogueCausalLMCollator):
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        pad_token_id = self.tokenizer.pad_token_id

        def _pad_sequences(values: list[list[int]], pad_value: int) -> torch.Tensor | None:
            max_len = max(len(value) for value in values)
            if max_len == 0:
                return None
            padded = [value + [pad_value] * (max_len - len(value)) for value in values]
            return torch.tensor(padded, dtype=torch.long)

        write_lengths = [len(feature["write_input_ids"]) for feature in features]
        read_lengths = [len(feature["input_ids"]) for feature in features]
        write_input_ids = _pad_sequences(
            [feature["write_input_ids"] for feature in features],
            pad_token_id,
        )
        write_attention_mask = _pad_sequences(
            [feature["write_attention_mask"] for feature in features],
            0,
        )
        write_message_ids = _pad_sequences(
            [feature["write_message_ids"] for feature in features],
            -1,
        )
        write_sentence_ids = _pad_sequences(
            [feature["write_sentence_ids"] for feature in features],
            -1,
        )
        if (
            write_input_ids is not None
            and write_attention_mask is not None
            and write_message_ids is not None
            and write_sentence_ids is not None
        ):
            batch["write_input_ids"] = write_input_ids
            batch["write_attention_mask"] = write_attention_mask
            batch["write_message_ids"] = write_message_ids
            batch["write_sentence_ids"] = write_sentence_ids
        batch["write_lengths"] = torch.tensor(write_lengths, dtype=torch.long)
        batch["read_lengths"] = torch.tensor(read_lengths, dtype=torch.long)

        has_negative_writes = [
            "negative_write_input_ids" in feature
            for feature in features
        ]
        if any(has_negative_writes) and not all(has_negative_writes):
            raise ValueError("Episode batch mixes paired and unpaired content-contrast examples")
        if all(has_negative_writes):
            required_negative_columns = (
                "negative_write_input_ids",
                "negative_write_attention_mask",
                "negative_write_message_ids",
                "negative_write_sentence_ids",
            )
            missing_negative_columns = [
                column
                for column in required_negative_columns
                if any(column not in feature for feature in features)
            ]
            if missing_negative_columns:
                raise ValueError(
                    "content_contrast_ce examples have incomplete negative writes: "
                    + ", ".join(missing_negative_columns)
                )
            negative_write_input_ids = _pad_sequences(
                [feature["negative_write_input_ids"] for feature in features],
                pad_token_id,
            )
            negative_write_attention_mask = _pad_sequences(
                [feature["negative_write_attention_mask"] for feature in features],
                0,
            )
            negative_write_message_ids = _pad_sequences(
                [feature["negative_write_message_ids"] for feature in features],
                -1,
            )
            negative_write_sentence_ids = _pad_sequences(
                [feature["negative_write_sentence_ids"] for feature in features],
                -1,
            )
            if (
                negative_write_input_ids is None
                or negative_write_attention_mask is None
                or negative_write_message_ids is None
                or negative_write_sentence_ids is None
            ):
                raise ValueError("content_contrast_ce examples require non-empty negative writes")
            batch["negative_write_input_ids"] = negative_write_input_ids
            batch["negative_write_attention_mask"] = negative_write_attention_mask
            batch["negative_write_message_ids"] = negative_write_message_ids
            batch["negative_write_sentence_ids"] = negative_write_sentence_ids

        teacher_input_ids = _pad_sequences(
            [feature["teacher_input_ids"] for feature in features],
            pad_token_id,
        )
        teacher_attention_mask = _pad_sequences(
            [feature["teacher_attention_mask"] for feature in features],
            0,
        )
        teacher_labels = _pad_sequences(
            [feature["teacher_labels"] for feature in features],
            -100,
        )
        if (
            teacher_input_ids is None
            or teacher_attention_mask is None
            or teacher_labels is None
        ):
            raise ValueError("Episode examples require non-empty canonical teacher tensors")
        batch["teacher_input_ids"] = teacher_input_ids
        batch["teacher_attention_mask"] = teacher_attention_mask
        batch["teacher_labels"] = teacher_labels

        state_only_write_input_ids = _pad_sequences(
            [feature["state_only_write_input_ids"] for feature in features],
            pad_token_id,
        )
        state_only_write_attention_mask = _pad_sequences(
            [feature["state_only_write_attention_mask"] for feature in features],
            0,
        )
        state_only_write_message_ids = _pad_sequences(
            [feature["state_only_write_message_ids"] for feature in features],
            -1,
        )
        state_only_write_sentence_ids = _pad_sequences(
            [feature["state_only_write_sentence_ids"] for feature in features],
            -1,
        )
        state_only_input_ids = _pad_sequences(
            [feature["state_only_input_ids"] for feature in features],
            pad_token_id,
        )
        state_only_attention_mask = _pad_sequences(
            [feature["state_only_attention_mask"] for feature in features],
            0,
        )
        state_only_labels = _pad_sequences(
            [feature["state_only_labels"] for feature in features],
            -100,
        )
        if (
            state_only_write_input_ids is not None
            and state_only_write_attention_mask is not None
            and state_only_write_message_ids is not None
            and state_only_write_sentence_ids is not None
            and state_only_input_ids is not None
            and state_only_attention_mask is not None
            and state_only_labels is not None
        ):
            batch["state_only_write_input_ids"] = state_only_write_input_ids
            batch["state_only_write_attention_mask"] = state_only_write_attention_mask
            batch["state_only_write_message_ids"] = state_only_write_message_ids
            batch["state_only_write_sentence_ids"] = state_only_write_sentence_ids
            batch["state_only_input_ids"] = state_only_input_ids
            batch["state_only_attention_mask"] = state_only_attention_mask
            batch["state_only_labels"] = state_only_labels

        max_full_len = max(
            len(feature["write_input_ids"]) + len(feature["input_ids"]) for feature in features
        )
        full_input_ids = []
        full_attention_mask = []
        full_labels = []
        for feature in features:
            combined_input_ids = feature["write_input_ids"] + feature["input_ids"]
            combined_attention_mask = (
                feature["write_attention_mask"] + feature["attention_mask"]
            )
            combined_labels = ([-100] * len(feature["write_input_ids"])) + feature["labels"]
            pad_len = max_full_len - len(combined_input_ids)
            full_input_ids.append(combined_input_ids + [pad_token_id] * pad_len)
            full_attention_mask.append(combined_attention_mask + [0] * pad_len)
            full_labels.append(combined_labels + [-100] * pad_len)
        batch["full_input_ids"] = torch.tensor(full_input_ids, dtype=torch.long)
        batch["full_attention_mask"] = torch.tensor(full_attention_mask, dtype=torch.long)
        batch["full_labels"] = torch.tensor(full_labels, dtype=torch.long)
        return batch


def build_data_collator(training_mode: str, tokenizer):
    if training_mode == "episode":
        return EpisodeCausalLMCollator(tokenizer)
    if training_mode == "dialogue":
        return DialogueCausalLMCollator(tokenizer)
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def main() -> None:
    args = parse_args()
    # Adapter and RWKV-core parameters are initialized before Trainer exists.
    set_seed(args.seed)
    if args.gradient_checkpointing:
        raise ValueError(
            "Gradient checkpointing is currently incompatible with Delta-Mem's stateful token updates. "
            "Disable --gradient-checkpointing before training."
        )
    if args.resume_mode in {"extend", *_ABLATION_RESUME_MODES}:
        raw_checkpoint = (
            ""
            if args.resume_from_checkpoint is None
            else str(args.resume_from_checkpoint).strip()
        )
        if not raw_checkpoint or raw_checkpoint.lower() in _RESUME_LATEST_VALUES:
            raise ValueError(
                f"--resume-mode {args.resume_mode} requires an explicit "
                "--resume-from-checkpoint path"
            )
    resume_from_checkpoint = resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        args.output_dir / "trainer",
        require_training_protocol=True,
        require_content_contrast_pairing=(
            args.memory_loss_mode == "content_contrast_ce"
            and args.resume_mode != "objective_ablation"
        ),
    )
    continuation_manifest = prepare_training_continuation(
        args,
        resume_from_checkpoint,
    )
    dtype = get_dtype(args.dtype)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized, tokenized_meta = load_or_prepare_tokenized_dataset(
        args,
        tokenizer,
        distributed=distributed,
        local_rank=local_rank,
    )
    effective_training_mode = str(tokenized_meta["training_mode"])
    if (
        effective_training_mode == "episode"
        and args.memory_loss_mode == "context_dropout_ce"
        and args.memory_base_kl_weight > 0.0
    ):
        validate_canonical_teacher_columns(tokenized)
    train_dataset, eval_dataset = split_tokenized_dataset(
        tokenized,
        validation_split_ratio=args.validation_split_ratio,
        data_seed=args.data_seed,
    )
    content_contrast_pairing_manifest = None
    if args.memory_loss_mode == "content_contrast_ce":
        if effective_training_mode != "episode":
            raise ValueError("content_contrast_ce requires episode training mode")
        train_dataset, train_pairing_manifest = materialize_content_contrast_pairs(
            train_dataset,
            split_name="train",
        )
        eval_pairing_manifest = None
        if eval_dataset is not None:
            eval_dataset, eval_pairing_manifest = materialize_content_contrast_pairs(
                eval_dataset,
                split_name="eval",
            )
        content_contrast_pairing_manifest = build_content_contrast_pairing_manifest(
            tokenized_fingerprint=getattr(tokenized, "_fingerprint", None),
            data_seed=args.data_seed,
            train_manifest=train_pairing_manifest,
            eval_manifest=eval_pairing_manifest,
        )
    effective_group_by_length = args.group_by_length and effective_training_mode != "episode"
    if args.group_by_length and not effective_group_by_length and local_rank in (-1, 0):
        print(
            "Disabling group_by_length for episode training because startup sorting is prohibitively slow at episode scale."
        )

    suppress_non_actionable_accelerate_warnings()
    resolved_attn_implementation = resolve_attn_implementation(
        args.model_path,
        args.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=resolved_attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _disable_training_cache(model)
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    requested_target_layers = parse_layer_indices(args.target_layers)
    delta_config = HFDeltaMemConfig(
        rank=args.rank,
        alpha=args.alpha,
        memory_backend=normalize_memory_backend(args.memory_backend),
        rwkv_ms_num_states=args.rwkv_ms_num_states,
        rwkv_ms_chunk_size=args.rwkv_ms_chunk_size,
        rwkv_ms_boundary_mode=args.rwkv_ms_boundary_mode,
        rwkv_ms_erase_gate=args.rwkv_ms_erase_gate,
        rwkv_ms_read_top_k=args.rwkv_ms_read_top_k,
        rwkv_ms_output_init_scale=args.rwkv_ms_output_init_scale,
        rwkv_ms_semantics_version=args.rwkv_ms_semantics_version,
        num_state_heads=args.num_state_heads,
        num_memory_partitions=args.num_memory_partitions,
        num_global_memory_partitions=args.num_global_memory_partitions,
        memory_partition_routing=args.memory_partition_routing,
        memory_partition_basis=args.memory_partition_basis,
        tie_memory_partition_read_write=args.tie_memory_partition_read_write,
        memory_partition_read_mode=args.memory_partition_read_mode,
        memory_partition_sigmoid_gate_bias_init=args.memory_partition_sigmoid_gate_bias_init,
        slot_read_top_k=args.slot_read_top_k,
        global_memory_mode=args.global_memory_mode,
        global_memory_read_top_k=args.global_memory_read_top_k,
        global_memory_merge_mode=args.global_memory_merge_mode,
        global_memory_gate_bias_init=args.global_memory_gate_bias_init,
        global_memory_read_logit_bias=args.global_memory_read_logit_bias,
        beta_bias_init=args.beta_bias_init,
        couple_lambda=args.couple_lambda,
        state_update_mode=normalize_state_update_mode(args.state_update_mode),
        rankwise_gates=args.rankwise_gates,
        output_init=args.output_init,
        base_slice_ref_width=args.base_slice_ref_width,
        delta_heads=parse_delta_heads(args.delta_heads),
        trainable_delta_scale=args.trainable_delta_scale,
        delta_scale_init=args.delta_scale_init,
        delta_scale_max=args.delta_scale_max,
        delta_scale_granularity=args.delta_scale_granularity,
        delta_scale_parameterization=args.delta_scale_parameterization,
        delta_o_rmsnorm=args.delta_o_rmsnorm,
        delta_o_rmsnorm_eps=args.delta_o_rmsnorm_eps,
        memory_fusion_mode=args.memory_fusion_mode,
        memory_fusion_gate_init=args.memory_fusion_gate_init,
        memory_fusion_placement=args.memory_fusion_placement,
        memory_fusion_residual_scale=args.memory_fusion_residual_scale,
        online_gain=args.online_gain,
        target_layers=requested_target_layers,
        memory_readout_mode=normalize_memory_readout_mode(args.memory_readout_mode),
        memory_write_source=args.memory_write_source,
        memory_write_granularity=args.memory_write_granularity,
        memory_write_proposals_per_message=args.memory_write_proposals_per_message,
    )
    replaced = attach_delta_mem(model, delta_config)
    wrapped_target_layers = tuple(
        sorted({int(module.base.layer_idx) for _, module in iter_delta_mem_modules(model)})
    )
    validate_wrapped_target_layers(requested_target_layers, wrapped_target_layers)
    trainable_names = freeze_non_delta_mem_params(model)
    checkpointed_frozen_mlps = (
        checkpoint_frozen_mlp_activations(model)
        if args.frozen_mlp_activation_checkpointing
        else []
    )
    _promote_trainable_parameters_to_fp32(model)

    warmup_steps = compute_warmup_steps(
        train_samples=len(train_dataset),
        per_device_train_batch_size=args.per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
    )
    warmup_steps = resolve_resume_warmup_steps(
        warmup_steps,
        resume_from_checkpoint,
    )
    training_protocol = build_training_protocol(
        args,
        tokenized,
        effective_training_mode=effective_training_mode,
        train_samples=len(train_dataset),
        eval_samples=0 if eval_dataset is None else len(eval_dataset),
        warmup_steps=warmup_steps,
        content_contrast_pairing_manifest=content_contrast_pairing_manifest,
    )
    training_protocol_sha256 = hashlib.sha256(
        json.dumps(training_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.resume_mode == "objective_ablation":
        if resume_from_checkpoint is None:
            raise ValueError("objective_ablation requires a resolved source checkpoint")
        source_protocol = _load_json_object(
            Path(resume_from_checkpoint) / _TRAINING_PROTOCOL_FILENAME,
            description="source training protocol",
        )
        validate_resume_training_protocol(
            source_protocol,
            training_protocol,
            resume_mode=args.resume_mode,
        )
    if continuation_manifest is not None and args.resume_mode in _ABLATION_RESUME_MODES:
        continuation_manifest.update(
            {
                "target_training_protocol_sha256": training_protocol_sha256,
                "target_delta_config_sha256": _protocol_sha256(delta_config.to_dict()),
            }
        )
        if args.resume_mode == "objective_ablation":
            if content_contrast_pairing_manifest is None:
                raise ValueError(
                    "objective_ablation requires a generated content-contrast pairing manifest"
                )
            continuation_manifest["target_content_contrast_pairing_manifest_sha256"] = (
                content_contrast_pairing_manifest["manifest_sha256"]
            )
    training_args_kwargs = dict(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=(
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size is not None
            else args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.data_seed,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        do_eval=eval_dataset is not None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        length_column_name="length",
        gradient_checkpointing=False,
        torch_compile=args.torch_compile,
        tf32=args.tf32,
        deepspeed=None if args.deepspeed_config is None else str(args.deepspeed_config),
        **_build_ddp_training_kwargs(
            distributed=distributed,
            ddp_backend=args.ddp_backend,
            local_rank=local_rank,
        ),
        bf16=args.bf16 or args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        report_to=["wandb"] if args.wandb else ["none"],
        run_name=(args.wandb_run_name or args.wandb_project) if args.wandb else None,
        remove_unused_columns=False,
    )
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["group_by_length"] = effective_group_by_length
    training_args = TrainingArguments(**training_args_kwargs)

    if args.wandb:
        if args.wandb_dir is None:
            raise ValueError("--wandb-dir is required when --wandb is enabled")
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_DIR"] = str(args.wandb_dir)
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
    trainer = DeltaMemTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_data_collator(effective_training_mode, tokenizer),
        delta_config=delta_config,
        write_sparsity_weight=args.write_sparsity_weight,
        write_sparsity_target=args.write_sparsity_target,
        memory_loss_mode=args.memory_loss_mode,
        memory_contrast_weight=args.memory_contrast_weight,
        memory_kl_weight=args.memory_kl_weight,
        memory_margin=args.memory_margin,
        memory_causal_weight=args.memory_causal_weight,
        memory_anchor_weight=args.memory_anchor_weight,
        memory_anchor_margin=args.memory_anchor_margin,
        memory_full_ce_weight=args.memory_full_ce_weight,
        memory_full_ce_max_length=args.memory_full_ce_max_length,
        memory_recover_weight=args.memory_recover_weight,
        memory_need_floor=args.memory_need_floor,
        memory_probe_weight=args.memory_probe_weight,
        memory_probe_alpha=args.memory_probe_alpha,
        memory_probe_margin=args.memory_probe_margin,
        memory_partition_alignment_weight=args.memory_partition_alignment_weight,
        memory_partition_entropy_weight=args.memory_partition_entropy_weight,
        memory_partition_balance_weight=args.memory_partition_balance_weight,
        memory_dropout_no_memory_prob=args.memory_dropout_no_memory_prob,
        memory_dropout_state_only_prob=args.memory_dropout_state_only_prob,
        memory_base_kl_weight=args.memory_base_kl_weight,
        episode_read_write_enabled=args.episode_read_write_enabled,
        context_ablation_mode=args.context_ablation_mode,
        context_ablation_no_state_prob=args.context_ablation_no_state_prob,
        context_ablation_state_only_prob=args.context_ablation_state_only_prob,
        training_protocol=training_protocol,
        content_contrast_pairing_manifest=content_contrast_pairing_manifest,
        resume_mode=args.resume_mode,
        continuation_manifest=continuation_manifest,
    )
    trainer.log_delta_debug_stats = args.log_delta_debug_stats
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.accelerator.wait_for_everyone()

    base_model = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.is_world_process_zero():
        active_lineage = getattr(trainer, "continuation_manifest", None)
        save_delta_mem_adapter(base_model, args.output_dir, delta_config)
        (args.output_dir / _TRAINING_PROTOCOL_FILENAME).write_text(
            json.dumps(training_protocol, indent=2, sort_keys=True)
        )
        if content_contrast_pairing_manifest is not None:
            (args.output_dir / _CONTENT_CONTRAST_PAIRING_FILENAME).write_text(
                json.dumps(content_contrast_pairing_manifest, indent=2, sort_keys=True)
            )
        if active_lineage is not None:
            (
                args.output_dir / _lineage_manifest_filename(active_lineage)
            ).write_text(
                json.dumps(active_lineage, indent=2, sort_keys=True)
            )
        summary = {
            "output_dir": str(args.output_dir),
            "resume_from_checkpoint": resume_from_checkpoint,
            "resume_mode": args.resume_mode,
            "continuation": active_lineage,
            "resume_lineage": active_lineage,
            "num_replaced_modules": len(replaced),
            "num_trainable_tensors": len(trainable_names),
            "num_checkpointed_frozen_mlps": len(checkpointed_frozen_mlps),
            "first_replaced_modules": replaced[:8],
            "first_trainable_tensors": trainable_names[:8],
            "first_checkpointed_frozen_mlps": checkpointed_frozen_mlps[:8],
            "tokenized_samples": len(tokenized),
            "train_samples": len(train_dataset),
            "eval_samples": 0 if eval_dataset is None else len(eval_dataset),
            "training_mode": effective_training_mode,
            "assistant_loss_mode": args.assistant_loss_mode,
            "episode_recent_messages": args.episode_recent_messages,
            "max_write_length": args.max_write_length,
            "episode_read_write_enabled": args.episode_read_write_enabled,
            "memory_loss_mode": args.memory_loss_mode,
            "memory_objective_version": training_protocol["memory_objective_version"],
            "teacher_max_length": args.max_write_length + args.max_length,
            "memory_write_source": args.memory_write_source,
            "memory_write_granularity": args.memory_write_granularity,
            "delta_o_rmsnorm": args.delta_o_rmsnorm,
            "delta_o_rmsnorm_eps": args.delta_o_rmsnorm_eps,
            "memory_fusion_mode": args.memory_fusion_mode,
            "memory_fusion_gate_init": args.memory_fusion_gate_init,
            "memory_fusion_placement": args.memory_fusion_placement,
            "memory_fusion_residual_scale": args.memory_fusion_residual_scale,
            "target_layers": args.target_layers,
            "memory_contrast_weight": args.memory_contrast_weight,
            "memory_kl_weight": args.memory_kl_weight,
            "memory_margin": args.memory_margin,
            "memory_causal_weight": args.memory_causal_weight,
            "memory_anchor_weight": args.memory_anchor_weight,
            "memory_anchor_margin": args.memory_anchor_margin,
            "memory_recover_weight": args.memory_recover_weight,
            "memory_need_floor": args.memory_need_floor,
            "memory_dropout_no_memory_prob": args.memory_dropout_no_memory_prob,
            "memory_dropout_state_only_prob": args.memory_dropout_state_only_prob,
            "memory_base_kl_weight": args.memory_base_kl_weight,
            "context_ablation_mode": args.context_ablation_mode,
            "context_ablation_no_state_prob": args.context_ablation_no_state_prob,
            "context_ablation_state_only_prob": args.context_ablation_state_only_prob,
            "memory_full_ce_weight": args.memory_full_ce_weight,
            "memory_full_ce_max_length": args.memory_full_ce_max_length,
            "output_init": args.output_init,
            "rwkv_ms_output_init_scale": args.rwkv_ms_output_init_scale,
            "rwkv_ms_semantics_version": args.rwkv_ms_semantics_version,
            "base_slice_ref_width": args.base_slice_ref_width,
            "memory_readout_mode": args.memory_readout_mode,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "validation_split_ratio": args.validation_split_ratio,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "training_protocol_sha256": training_protocol_sha256,
            "content_contrast_pairing_manifest_sha256": (
                None
                if content_contrast_pairing_manifest is None
                else content_contrast_pairing_manifest["manifest_sha256"]
            ),
            "memory_dropout_counts_current_process_since_resume": trainer.memory_dropout_counts,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "optim": args.optim,
            "requested_attn_implementation": args.attn_implementation,
            "attn_implementation": resolved_attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "frozen_mlp_activation_checkpointing": (
                args.frozen_mlp_activation_checkpointing
            ),
            "torch_compile": args.torch_compile,
            "tf32": args.tf32,
            "group_by_length": effective_group_by_length,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataset_num_proc": args.dataset_num_proc,
            "hf_cache_dir": str(args.hf_cache_dir),
            "tokenized_dataset_dir": None if args.tokenized_dataset_dir is None else str(args.tokenized_dataset_dir),
            "tokenized_dataset_root": str(args.tokenized_dataset_root),
            "tokenized_cache": args.tokenized_cache,
            "tokenized_cache_hit": tokenized_meta["tokenized_cache_hit"],
            "tokenized_cache_dir": tokenized_meta["tokenized_cache_dir"],
            "tokenized_dataset_source": tokenized_meta["tokenized_dataset_source"],
            "ddp_broadcast_buffers": training_args.ddp_broadcast_buffers,
            "ddp_backend": args.ddp_backend if distributed else None,
            "local_rank": local_rank if distributed else -1,
            "world_size": world_size,
            "deepspeed_config": None if args.deepspeed_config is None else str(args.deepspeed_config),
            "write_sparsity_weight": args.write_sparsity_weight,
            "write_sparsity_target": args.write_sparsity_target,
            "wandb_project": args.wandb_project if args.wandb else None,
            "wandb_entity": args.wandb_entity if args.wandb else None,
            "wandb_run_name": args.wandb_run_name if args.wandb else None,
            "wandb_group": args.wandb_group if args.wandb else None,
            "wandb_tags": args.wandb_tags if args.wandb else None,
            "wandb_mode": args.wandb_mode if args.wandb else None,
            "wandb_dir": str(args.wandb_dir) if args.wandb else None,
            "gate_stats": collect_delta_mem_gate_stats(base_model),
            "output_ratio_stats": collect_delta_mem_output_ratio_stats(base_model),
            "config": asdict(delta_config),
        }
        (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


def _destroy_process_group_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        _destroy_process_group_if_needed()
