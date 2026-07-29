from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import RandomSampler, Sampler
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.utils import is_sagemaker_mp_enabled

import deltamem.chat_templates as project_chat_templates

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_gate_stats,
    collect_delta_mem_output_ratio_stats,
    collect_delta_mem_partition_route_stats,
    collect_delta_mem_read_representations,
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
    set_delta_mem_read_representation_capture_mask,
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
from deltamem.train.scene_state_generation_alignment import (
    clone_detached_online_state,
    generated_unlikelihood_positions,
)
from experiments.rethinking_rwkv_ms_gemma.scene_memory_v8_warm_start import (
    DEFAULT_LOCK_PATH as SCENE_V8_WARM_START_LOCK_PATH,
    REQUIRED_SOURCE_ARTIFACTS as SCENE_V8_REQUIRED_WARM_START_ARTIFACTS,
    RECEIPT_SCHEMA as SCENE_V8_WARM_START_RECEIPT_SCHEMA,
    WARM_START_MODE as SCENE_V8_WARM_START_MODE,
    V8FreshStartContract,
    V8WarmStartContext,
    apply_v8_v7_checkpoint256_adapter_only_warm_start,
    load_v8_warm_start_lock,
    prepare_v8_v7_checkpoint256_warm_start,
)


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
        "read_representation_capture_mask",
        "last_read_representation",
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


def _capture_torch_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return cpu_state, cuda_states


def _restore_torch_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    cpu_state, cuda_states = state
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


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
_WARM_START_MODES = (
    "residual_hybrid_w8_ablation",
    SCENE_V8_WARM_START_MODE,
)
_RESIDUAL_HYBRID_W8_WARM_START_MODE = _WARM_START_MODES[0]
_SCENE_V8_WARM_START_MODE = _WARM_START_MODES[1]
_CONTINUATION_SCHEDULERS = frozenset({"constant", "constant_with_warmup"})
_REPRESENTATION_CAPTURE_FUSION_PLACEMENTS = frozenset(
    {"attention_output", "post_attention_residual_hybrid"}
)
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
_CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION = 8
_CONTENT_CONTRAST_OBJECTIVE_VERSION = "content_contrast_ce_v6"
_CONTENT_CONTRAST_BACKWARD_MODE = "sequential_exact_first_order_v1"
_CONTENT_CONTRAST_READ_MASK_MODE = "valid_context_and_supervised_predictors_v2"
_CONTENT_CONTRAST_TARGET_MODE = "first_differing_supervised_target_span_v1"
_CONTENT_CONTRAST_TARGET_SPAN_TOKENS = 8
_CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD = True
_CONTENT_CONTRAST_REPRESENTATION_MODE = (
    "fused_delta_o_first_selected_target_predictor_relative_l2_v1"
)
_CONTENT_CONTRAST_REPRESENTATION_EPS = 1e-6
_SCENE_BOUNDARY_PAYLOAD_MASK_MODE = "top_level_boundaries_json_array_offset_overlap_v1"
_SCENE_BOUNDARY_PAYLOAD_CE_NORMALIZATION = (
    "selected_sum_over_all_supervised_tokens_v1"
)
_SCENE_STATE_IDENTITY_TRAINING_PROTOCOL_SCHEMA_VERSION = 9
_SCENE_STATE_IDENTITY_OBJECTIVE_VERSION = "scene_state_identity_ce_v2"
_SCENE_STATE_IDENTITY_BACKWARD_MODE = (
    "sequential_replayed_donor_single_zero_diagnostic_exact_first_order_v2"
)
_SCENE_STATE_IDENTITY_READ_PROTOCOL = (
    "state_only_same_read_correct_donor_zero_adapter_active_v1"
)
_SCENE_STATE_IDENTITY_ZERO_PROTOCOL = (
    "adapter_active_reset_state_writes_disabled_v1"
)
_SCENE_STATE_SEMANTIC_MASK_MODE = (
    "top_level_boundaries_nonwhitespace_offset_overlap_v1"
)
_SCENE_STATE_SEMANTIC_LOSS_NORMALIZATION = (
    "selected_tokens_per_row_then_batch_mean_v1"
)
_SCENE_STATE_IDENTITY_TARGET_MODE = (
    "first_pair_distinguishing_semantic_token_v1"
)
_SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE = (
    "exact_input_ids_and_attention_before_pair_target_v1"
)
_SCENE_STATE_IDENTITY_PAIRING_VERSION = (
    "nearest_write_token_length_label_distinct_symmetric_pair_v2"
)
_SCENE_STATE_IDENTITY_PAIRING_REFINEMENT = (
    "maximize_nonempty_same_cardinality_within_nearest_length_budget_v1"
)
_SCENE_STATE_IDENTITY_TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
_SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES = {
    stratum: index
    for index, stratum in enumerate(_SCENE_STATE_IDENTITY_TARGET_STRATA)
}
_SCENE_STATE_IDENTITY_PAIRING_FILENAME = (
    "scene_state_identity_pairing_manifest.json"
)
_SCENE_STATE_FULL_CORRECT_CE_WEIGHT = 1.0
_SCENE_STATE_CORRECT_ALL_SEMANTIC_CE_WEIGHT = 1.0
_SCENE_STATE_DONOR_MARGIN_WEIGHT = 1.0
_SCENE_STATE_GENERATION_TRAINING_PROTOCOL_SCHEMA_VERSION = 10
_SCENE_STATE_GENERATION_OBJECTIVE_VERSION = "scene_state_generation_ce_v1"
_SCENE_STATE_GENERATION_BACKWARD_MODE = (
    "sequential_replayed_donor_zero_detached_generation_first_v1"
)
_SCENE_STATE_GENERATED_UNLIKELIHOOD_TRAINING_PROTOCOL_SCHEMA_VERSION = 11
_SCENE_STATE_GENERATED_UNLIKELIHOOD_BACKWARD_MODE = (
    "v7_sequential_then_edit_aligned_correct_state_reprime_unlikelihood_v2"
)
_SCENE_STATE_GENERATION_SCHEMA_WEIGHT = 2.0
_SCENE_STATE_GENERATION_DECISION_WEIGHT = 4.0
_SCENE_STATE_GENERATION_TERMINATION_WEIGHT = 1.0
_SCENE_STATE_GENERATION_TOP1_MARGIN = 0.2
_SCENE_STATE_GENERATION_ZERO_MARGIN = 0.2
_SCENE_STATE_GENERATED_UNLIKELIHOOD_OBJECTIVE_VERSION = (
    "scene_state_generation_ce_generated_prefix_unlikelihood_v2"
)
_SCENE_STATE_GENERATED_UNLIKELIHOOD_WEIGHT = 0.5
_SCENE_STATE_GENERATED_UNLIKELIHOOD_MAX_WRONG_TOKENS = 4
_SCENE_STATE_GENERATED_ROLLOUT_EXTRA_TOKENS = 4
_SCENE_STATE_GENERATED_ROLLOUT_MAX_TOKENS = 24
_SCENE_STATE_GENERATED_UNLIKELIHOOD_MODE = (
    "greedy_correct_state_edit_aligned_wrong_tokens_v2"
)
_SCENE_STATE_GENERATION_MASK_MODE = (
    "exact_system_only_generation_prefix_content_schema_decision_termination_v1"
)
_SCENE_STATE_GENERATION_DECISION_MASK_MODE = (
    "top_level_boundaries_json_decision_char_overlap_v1"
)
_SCENE_STATE_PAIRED_MEMORY_LOSS_MODES = {
    "scene_state_identity_ce",
    "scene_state_generation_ce",
}
_SEEDED_TRAIN_SAMPLER_MODE = "torch_random_sampler_seed_equals_data_seed_v1"
_DEFAULT_TRAIN_SAMPLER_MODE = "transformers_trainer_default_v1"
_FIXED_TRAIN_SCHEDULE_SAMPLER_MODE = "explicit_ordered_train_row_ordinal_v1"
_SCENE_MEMORY_V8_WARMUP_STEPS = 4
_SCENE_MEMORY_V8_SOURCE_SCHEMA = "rwkv_ms_scene_memory_v8_source.v1"
_SCENE_MEMORY_V8_CURRICULUM_SCHEMA = (
    "rwkv_ms_scene_memory_v8_curriculum_binding.v1"
)
_SCENE_MEMORY_V8_SCHEDULE_ENTRY_SCHEMA = (
    "rwkv_ms_scene_memory_v8_schedule_entry.v1"
)
_SCENE_MEMORY_V8_SCHEDULE_MANIFEST_SCHEMA = (
    "rwkv_ms_scene_memory_v8_schedule_manifest.v1"
)
_CONTENT_CONTRAST_PAIRING_FILENAME = "content_contrast_pairing_manifest.json"
_INITIAL_ADAPTER_DIRNAME = "initial_adapter"
_INITIAL_ADAPTER_MANIFEST_FILENAME = "initial_adapter_manifest.json"
_INITIAL_ADAPTER_MANIFEST_SCHEMA = "deltamem.seeded_initial_adapter.v1"
_LAUNCH_MANIFEST_FILENAME = "launch_manifest.json"
_DATA_CONTRACT_MANIFEST_FILENAME = "data_contract_manifest.json"
_TOKENIZED_CACHE_FORMAT_VERSION = 2
_TOKENIZED_CACHE_READY_SCHEMA = "deltamem.tokenized_dataset_cache.v2"
_TOKENIZED_DATASET_IDENTITY_SCHEMA = "deltamem.tokenized_dataset_identity.v1"
_TOKENIZED_ORDERED_CONTENT_SCHEMA = "deltamem.tokenized_dataset_ordered_content.v1"
_TOKENIZED_CACHE_READY_FILENAME = "_READY"
_CONTENT_CONTRAST_PAIRING_VERSION = "post_split_half_rotation_v1"
_CONTINUATION_MANIFEST_FILENAME = "continuation_manifest.json"
_CONTINUATION_MANIFEST_SCHEMA_VERSION = 1
_ABLATION_LINEAGE_FILENAME = "ablation_lineage_manifest.json"
_ABLATION_LINEAGE_SCHEMA_VERSION = 1
_WARM_START_LINEAGE_FILENAME = "warm_start_lineage_manifest.json"
_WARM_START_LINEAGE_SCHEMA_VERSION = 1
_RESIDUAL_HYBRID_W8_SOURCE_SCHEMA_VERSION = 7
_RESIDUAL_HYBRID_W8_SOURCE_OBJECTIVE_VERSION = "content_contrast_ce_v5"
_RESIDUAL_HYBRID_W8_SOURCE_REPRESENTATION_MODE = (
    "fused_delta_o_first_supervised_predictor_relative_l2_v1"
)
_RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP = 416
_RESIDUAL_HYBRID_W8_SOURCE_EPOCH = 13.0
_RESIDUAL_HYBRID_W8_SOURCE_ADAPTER_TENSORS = 1470
_RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS = 1218
_RESIDUAL_HYBRID_W8_TARGET_LAYERS = tuple(range(42))
_RESIDUAL_HYBRID_W8_TARGET_NEW_GAIN_TENSORS = len(
    _RESIDUAL_HYBRID_W8_TARGET_LAYERS
)
_RESIDUAL_HYBRID_W8_TARGET_ADAPTER_TENSORS = (
    _RESIDUAL_HYBRID_W8_SOURCE_ADAPTER_TENSORS
    + _RESIDUAL_HYBRID_W8_TARGET_NEW_GAIN_TENSORS
)
_RESIDUAL_HYBRID_W8_TARGET_TRAINABLE_TENSORS = (
    _RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS
    + _RESIDUAL_HYBRID_W8_TARGET_NEW_GAIN_TENSORS
)
_RESIDUAL_HYBRID_W8_TARGET_PLACEMENT = "post_attention_residual_hybrid"
_RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE = 0.01
_RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX = 0.02
_RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS = 32
_RESIDUAL_HYBRID_W8_TARGET_EPOCHS = 1.0
_RESIDUAL_HYBRID_W8_TARGET_WARMUP_RATIO = 0.0625
_RESIDUAL_HYBRID_W8_TARGET_WARMUP_STEPS = 2
_RESIDUAL_HYBRID_W8_PROTOCOL_DRIFT = frozenset(
    {
        "schema_version",
        "memory_objective_version",
        "memory_fusion_placement",
        "memory_fusion_residual_scale",
        "memory_fusion_residual_scale_max",
        "content_contrast_target_mode",
        "content_contrast_target_span_tokens",
        "content_contrast_representation_mode",
        "content_contrast_pairing",
        "max_steps",
        "num_train_epochs",
        "warmup_steps",
    }
)
_OBJECTIVE_ABLATION_PROTOCOL_DRIFT = frozenset(
    {
        "schema_version",
        "memory_objective_version",
        "memory_loss_mode",
        "memory_contrast_weight",
        "memory_margin",
        "memory_representation_weight",
        "memory_representation_margin",
        "memory_kl_weight",
        "write_sparsity_weight",
        "memory_partition_alignment_weight",
        "memory_partition_entropy_weight",
        "memory_partition_balance_weight",
        "content_contrast_negative_priming_grad",
        "content_contrast_backward_mode",
        "content_contrast_read_mask_mode",
        "content_contrast_target_mode",
        "content_contrast_target_span_tokens",
        "content_contrast_previous_source_grad",
        "content_contrast_representation_mode",
        "content_contrast_pairing",
    }
)


@dataclass
class AdapterWarmStartContext:
    checkpoint: Path
    mode: str
    source_protocol: dict[str, object]
    source_config: HFDeltaMemConfig
    manifest: dict[str, object]
    scene_v8_context: V8WarmStartContext | None = None
    scene_v8_fresh_start: V8FreshStartContract | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def resolve_initial_adapter_output_dir(
    args,
    *,
    resume_from_checkpoint: str | None,
    warm_start_from_checkpoint: Path | None,
    world_size: int,
) -> Path | None:
    requested = getattr(args, "initial_adapter_output_dir", None)
    if requested is None:
        return None
    if resume_from_checkpoint is not None or warm_start_from_checkpoint is not None:
        raise ValueError("Initial adapter snapshots are supported only for fresh runs")
    if world_size != 1:
        raise ValueError("Initial adapter snapshots currently require a single-process run")

    output_dir = Path(args.output_dir).expanduser().resolve()
    expected = output_dir / _INITIAL_ADAPTER_DIRNAME
    snapshot_dir = Path(requested).expanduser().resolve()
    if snapshot_dir != expected:
        raise ValueError(
            "--initial-adapter-output-dir must be exactly "
            f"OUTPUT_DIR/{_INITIAL_ADAPTER_DIRNAME}: expected={expected} actual={snapshot_dir}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Training output path is not a directory: {output_dir}")
    if output_dir.is_dir():
        allowed_provenance_files = {
            _LAUNCH_MANIFEST_FILENAME,
            _DATA_CONTRACT_MANIFEST_FILENAME,
        }
        unexpected = sorted(
            str(path)
            for path in output_dir.iterdir()
            if path.name not in allowed_provenance_files
        )
        if unexpected:
            raise ValueError(
                "Initial adapter snapshot requires a fresh training output; "
                f"unexpected entries={unexpected}"
            )
        for filename in sorted(allowed_provenance_files):
            provenance_path = output_dir / filename
            if provenance_path.exists() and (
                not provenance_path.is_file() or provenance_path.is_symlink()
            ):
                raise ValueError(
                    f"Initial adapter provenance path is not a regular file: "
                    f"{provenance_path}"
                )
    return snapshot_dir


def _rng_tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(bytes(value.detach().cpu().tolist())).hexdigest()


def save_seeded_initial_adapter_snapshot(
    model: nn.Module,
    snapshot_dir: Path,
    delta_config: HFDeltaMemConfig,
    *,
    args,
    training_protocol: dict[str, object],
    training_protocol_sha256: str,
    train_samples: int,
    replaced_modules: list[str],
    trainable_names: list[str],
) -> dict[str, object]:
    snapshot_dir = snapshot_dir.expanduser().resolve()
    if snapshot_dir.exists():
        raise ValueError(f"Initial adapter output already exists: {snapshot_dir}")
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = snapshot_dir.parent / f".{snapshot_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        raise ValueError(f"Initial adapter temporary output already exists: {temporary_dir}")

    train_file = None if args.train_file is None else Path(args.train_file).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    model_config_path = model_path / "config.json"
    launch_manifest_path = snapshot_dir.parent / _LAUNCH_MANIFEST_FILENAME
    data_contract_manifest_path = (
        snapshot_dir.parent / _DATA_CONTRACT_MANIFEST_FILENAME
    )
    adapter_state = get_delta_mem_state_dict(model)
    cpu_rng_state, cuda_rng_states = _capture_torch_rng_state()
    rng_provenance = {
        "cpu_sha256": _rng_tensor_sha256(cpu_rng_state),
        "cuda_sha256": (
            None
            if cuda_rng_states is None
            else [_rng_tensor_sha256(state) for state in cuda_rng_states]
        ),
    }

    try:
        save_delta_mem_adapter(model, temporary_dir, delta_config)
        training_protocol_path = temporary_dir / _TRAINING_PROTOCOL_FILENAME
        training_protocol_path.write_text(
            json.dumps(training_protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema": _INITIAL_ADAPTER_MANIFEST_SCHEMA,
            "created_at_unix": time.time(),
            "artifact_kind": "seeded_freshly_attached_delta_mem_adapter",
            "global_step": 0,
            "fresh_run": True,
            "training_started": False,
            "optimizer_created": False,
            "optimizer_state_included": False,
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "rng_state_after_attachment": rng_provenance,
            "output_dir": str(snapshot_dir),
            "model": {
                "path": str(model_path),
                "config_sha256": (
                    _sha256_file(model_config_path) if model_config_path.is_file() else None
                ),
            },
            "dataset": {
                "train_file": None if train_file is None else str(train_file),
                "train_file_sha256": (
                    _sha256_file(train_file)
                    if train_file is not None and train_file.is_file()
                    else None
                ),
                "dataset_name": args.dataset_name,
                "dataset_split": args.dataset_split,
                "train_samples": int(train_samples),
                "tokenized_fingerprint": training_protocol.get("tokenized_fingerprint"),
                "tokenized_dataset_sha256": training_protocol.get(
                    "tokenized_dataset_sha256"
                ),
                "tokenized_cache_identity": training_protocol.get(
                    "tokenized_cache_identity"
                ),
            },
            "topology": {
                "replaced_modules": list(replaced_modules),
                "trainable_names": list(trainable_names),
                "adapter_tensor_count": len(adapter_state),
                "adapter_parameter_count": sum(
                    int(tensor.numel()) for tensor in adapter_state.values()
                ),
                "adapter_topology_sha256": _adapter_topology_sha256(adapter_state),
                "delta_config_sha256": _protocol_sha256(delta_config.to_dict()),
            },
            "training_protocol": {
                "canonical_sha256": training_protocol_sha256,
                "file": _TRAINING_PROTOCOL_FILENAME,
                "file_sha256": _sha256_file(training_protocol_path),
            },
            "launch_manifest": (
                None
                if not launch_manifest_path.is_file()
                else {
                    "path": str(launch_manifest_path),
                    "sha256": _sha256_file(launch_manifest_path),
                }
            ),
            "data_contract_manifest": (
                None
                if not data_contract_manifest_path.is_file()
                else {
                    "path": str(data_contract_manifest_path),
                    "sha256": _sha256_file(data_contract_manifest_path),
                }
            ),
            "process_argv": list(sys.argv),
            "trainer_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "files": {
                "adapter": {
                    "path": "delta_mem_adapter.pt",
                    "sha256": _sha256_file(temporary_dir / "delta_mem_adapter.pt"),
                },
                "config": {
                    "path": "delta_mem_config.json",
                    "sha256": _sha256_file(temporary_dir / "delta_mem_config.json"),
                },
            },
        }
        manifest["manifest_sha256"] = _protocol_sha256(manifest)
        (temporary_dir / _INITIAL_ADAPTER_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, snapshot_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        _restore_torch_rng_state((cpu_rng_state, cuda_rng_states))


def validate_prepare_only_snapshot(
    snapshot_dir: Path,
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    if manifest is None:
        raise RuntimeError("Prepare-only mode did not create an initial adapter manifest")
    expected_flags = {
        "global_step": 0,
        "fresh_run": True,
        "training_started": False,
        "optimizer_created": False,
        "optimizer_state_included": False,
    }
    for field, expected in expected_flags.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"Prepare-only initial adapter manifest has invalid {field}: "
                f"expected={expected!r} actual={manifest.get(field)!r}"
            )

    required_files = (
        "delta_mem_adapter.pt",
        "delta_mem_config.json",
        _TRAINING_PROTOCOL_FILENAME,
        _INITIAL_ADAPTER_MANIFEST_FILENAME,
    )
    snapshot_dir = snapshot_dir.expanduser().resolve()
    for filename in required_files:
        artifact = snapshot_dir / filename
        if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size <= 0:
            raise RuntimeError(
                f"Prepare-only initial adapter artifact is missing or invalid: {artifact}"
            )

    saved_manifest = _load_json_object(
        snapshot_dir / _INITIAL_ADAPTER_MANIFEST_FILENAME,
        description="prepare-only initial adapter manifest",
    )
    if saved_manifest != manifest:
        raise RuntimeError("Prepare-only in-memory and saved initial adapter manifests differ")
    return {
        "prepare_only": True,
        "training_started": False,
        "optimizer_created": False,
        "initial_adapter_output_dir": str(snapshot_dir),
        "initial_adapter_manifest_sha256": manifest.get("manifest_sha256"),
    }


def _adapter_topology_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    topology = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Delta-Mem adapter entry is not a tensor: {name}")
        topology.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
    return hashlib.sha256(
        json.dumps(topology, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_optimizer_parameter_count(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Warm-start optimizer checkpoint must contain a dictionary")
    param_groups = payload.get("param_groups")
    optimizer_state = payload.get("state")
    if not isinstance(param_groups, list) or not isinstance(optimizer_state, dict):
        raise ValueError("Warm-start optimizer checkpoint is incomplete")
    parameter_ids: list[object] = []
    for group in param_groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("Warm-start optimizer param_groups are invalid")
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("Warm-start optimizer contains duplicate parameter references")
    if set(optimizer_state) != set(parameter_ids):
        raise ValueError(
            "Warm-start optimizer state keys do not match its parameter references"
        )
    return len(parameter_ids)


def _missing_resume_checkpoint_files(
    checkpoint: Path,
    *,
    require_training_protocol: bool = False,
    require_content_contrast_pairing: bool = False,
    require_scene_state_identity_pairing: bool = False,
) -> tuple[str, ...]:
    required_files = list(_REQUIRED_RESUME_CHECKPOINT_FILES)
    if (
        require_training_protocol
        or require_content_contrast_pairing
        or require_scene_state_identity_pairing
    ):
        required_files.append(_TRAINING_PROTOCOL_FILENAME)
    if require_content_contrast_pairing:
        required_files.append(_CONTENT_CONTRAST_PAIRING_FILENAME)
    if require_scene_state_identity_pairing:
        required_files.append(_SCENE_STATE_IDENTITY_PAIRING_FILENAME)
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
    require_scene_state_identity_pairing: bool = False,
) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory does not exist: {checkpoint}")
    missing = _missing_resume_checkpoint_files(
        checkpoint,
        require_training_protocol=require_training_protocol,
        require_content_contrast_pairing=require_content_contrast_pairing,
        require_scene_state_identity_pairing=require_scene_state_identity_pairing,
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
    require_scene_state_identity_pairing: bool = False,
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
                require_scene_state_identity_pairing=(
                    require_scene_state_identity_pairing
                ),
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
            require_scene_state_identity_pairing=(
                require_scene_state_identity_pairing
            ),
        ):
            return str(candidate.resolve())
    newest = candidates[0][1]
    missing = _missing_resume_checkpoint_files(
        newest,
        require_training_protocol=require_training_protocol,
        require_content_contrast_pairing=require_content_contrast_pairing,
        require_scene_state_identity_pairing=require_scene_state_identity_pairing,
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


def resolve_adapter_warm_start_checkpoint(
    warm_start_from_checkpoint: str | Path | None,
    *,
    warm_start_mode: str | None = None,
) -> str | None:
    if warm_start_from_checkpoint is None:
        return None
    raw_checkpoint = str(warm_start_from_checkpoint).strip()
    if not raw_checkpoint:
        raise ValueError("--warm-start-from-checkpoint must not be empty")
    if raw_checkpoint.lower() in _RESUME_LATEST_VALUES:
        raise ValueError(
            "--warm-start-from-checkpoint requires an explicit checkpoint path"
        )
    checkpoint = Path(raw_checkpoint).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Warm-start checkpoint directory does not exist: {checkpoint}"
        )
    if warm_start_mode == _SCENE_V8_WARM_START_MODE:
        required_files = SCENE_V8_REQUIRED_WARM_START_ARTIFACTS
    else:
        required_files = (
            "delta_mem_adapter.pt",
            "delta_mem_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            _TRAINING_PROTOCOL_FILENAME,
            _CONTENT_CONTRAST_PAIRING_FILENAME,
        )
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    rng_files = sorted(path for path in checkpoint.glob("rng_state*.pth") if path.is_file())
    if not rng_files:
        missing.append("rng_state*.pth")
    if missing:
        raise FileNotFoundError(
            f"Warm-start checkpoint is incomplete: {checkpoint}; missing {', '.join(missing)}"
        )
    return str(checkpoint.resolve())


def _validate_residual_hybrid_w8_warm_start_args(args: argparse.Namespace) -> None:
    if args.warm_start_mode != _RESIDUAL_HYBRID_W8_WARM_START_MODE:
        raise ValueError(
            "Residual-hybrid W8 warm start requires "
            f"--warm-start-mode {_RESIDUAL_HYBRID_W8_WARM_START_MODE}"
        )
    if args.resume_from_checkpoint is not None:
        raise ValueError("Adapter warm start cannot be combined with checkpoint resume")
    if args.resume_mode != "exact":
        raise ValueError("Adapter warm start requires --resume-mode exact")
    expected_values = {
        "memory_loss_mode": "content_contrast_ce",
        "memory_contrast_weight": 0.25,
        "memory_margin": 0.5,
        "memory_representation_weight": 0.1,
        "memory_representation_margin": 0.1,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": _RESIDUAL_HYBRID_W8_TARGET_WARMUP_RATIO,
        "max_steps": _RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS,
        "num_train_epochs": _RESIDUAL_HYBRID_W8_TARGET_EPOCHS,
    }
    mismatches = [
        name for name, expected in expected_values.items() if getattr(args, name) != expected
    ]
    placement = normalize_memory_fusion_placement(args.memory_fusion_placement)
    if placement != _RESIDUAL_HYBRID_W8_TARGET_PLACEMENT:
        mismatches.append("memory_fusion_placement")
    if args.memory_fusion_residual_scale != _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE:
        mismatches.append("memory_fusion_residual_scale")
    if (
        args.memory_fusion_residual_scale_max
        != _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
    ):
        mismatches.append("memory_fusion_residual_scale_max")
    if parse_layer_indices(args.target_layers) != _RESIDUAL_HYBRID_W8_TARGET_LAYERS:
        mismatches.append("target_layers")
    if mismatches:
        raise ValueError(
            "residual_hybrid_w8_ablation target contract mismatch for: "
            + ", ".join(sorted(set(mismatches)))
        )


def _validate_scene_v8_warm_start_args(args: argparse.Namespace) -> None:
    if args.warm_start_mode != _SCENE_V8_WARM_START_MODE:
        raise ValueError(
            "Scene V8 warm start requires "
            f"--warm-start-mode {_SCENE_V8_WARM_START_MODE}"
        )
    if args.resume_from_checkpoint is not None:
        raise ValueError("Scene V8 adapter warm start cannot restore checkpoint state")
    if args.resume_mode != "exact":
        raise ValueError("Scene V8 adapter warm start requires --resume-mode exact")
    mismatches = []
    if args.memory_loss_mode != "scene_state_generation_ce":
        mismatches.append("memory_loss_mode")
    if parse_layer_indices(args.target_layers) != tuple(range(42)):
        mismatches.append("target_layers")
    if parse_delta_heads(args.delta_heads) != ("q", "o"):
        mismatches.append("delta_heads")
    expected_values = {
        "rank": 4,
        "alpha": 8.0,
        "memory_backend": "rwkv_ms",
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_semantics_version": 2,
        "output_init": "base_slice_fixed",
        "base_slice_ref_width": 8,
        "online_gain": 0.2,
        "memory_fusion_mode": "add",
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
        "memory_fusion_residual_scale_max": 1.0,
        "trainable_delta_scale": True,
        "delta_scale_init": 0.1,
        "delta_scale_max": 0.5,
        "delta_scale_granularity": "head",
        "delta_scale_parameterization": "alpha_over_rank",
        "memory_readout_mode": "delta",
        "memory_write_source": "learned_hidden",
        "memory_write_granularity": "token",
    }
    mismatches.extend(
        name
        for name, expected in expected_values.items()
        if getattr(args, name) != expected
    )
    if mismatches:
        raise ValueError(
            "Scene V8 warm-start target topology differs for: "
            + ", ".join(sorted(set(mismatches)))
        )


def _validate_adapter_warm_start_args(args: argparse.Namespace) -> None:
    if args.warm_start_mode == _RESIDUAL_HYBRID_W8_WARM_START_MODE:
        _validate_residual_hybrid_w8_warm_start_args(args)
        return
    if args.warm_start_mode == _SCENE_V8_WARM_START_MODE:
        _validate_scene_v8_warm_start_args(args)
        return
    raise ValueError(f"Unsupported adapter warm-start mode: {args.warm_start_mode}")


def _validate_residual_hybrid_w8_source_protocol(
    source_protocol: dict[str, object],
) -> None:
    expected = {
        "schema_version": _RESIDUAL_HYBRID_W8_SOURCE_SCHEMA_VERSION,
        "memory_objective_version": _RESIDUAL_HYBRID_W8_SOURCE_OBJECTIVE_VERSION,
        "memory_loss_mode": "content_contrast_ce",
        "memory_contrast_weight": 0.25,
        "memory_margin": 0.5,
        "memory_representation_weight": 0.1,
        "memory_representation_margin": 0.1,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "content_contrast_backward_mode": _CONTENT_CONTRAST_BACKWARD_MODE,
        "content_contrast_read_mask_mode": _CONTENT_CONTRAST_READ_MASK_MODE,
        "content_contrast_previous_source_grad": True,
        "content_contrast_representation_mode": (
            _RESIDUAL_HYBRID_W8_SOURCE_REPRESENTATION_MODE
        ),
        "train_samples": 32,
        "eval_samples": 0,
        "learning_rate": 0.001,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": _RESIDUAL_HYBRID_W8_TARGET_WARMUP_RATIO,
        "warmup_steps": 8,
        "max_steps": _RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP,
        "num_train_epochs": _RESIDUAL_HYBRID_W8_SOURCE_EPOCH,
        "memory_fusion_placement": "attention_output",
        "memory_fusion_residual_scale": 1.0,
    }
    mismatches = [
        name for name, value in expected.items() if source_protocol.get(name) != value
    ]
    if "content_contrast_target_mode" in source_protocol:
        mismatches.append("content_contrast_target_mode")
    if "content_contrast_target_span_tokens" in source_protocol:
        mismatches.append("content_contrast_target_span_tokens")
    if mismatches:
        raise ValueError(
            "residual_hybrid_w8_ablation requires the completed V14 schema-7/v5 source; "
            "invalid fields: " + ", ".join(sorted(set(mismatches)))
        )


def prepare_adapter_warm_start(
    args: argparse.Namespace,
    warm_start_from_checkpoint: str | None,
) -> AdapterWarmStartContext | None:
    if warm_start_from_checkpoint is None:
        return None
    _validate_adapter_warm_start_args(args)
    checkpoint = Path(warm_start_from_checkpoint).resolve()
    target_output_dir = Path(args.output_dir).expanduser().resolve()
    if target_output_dir.exists() and any(target_output_dir.iterdir()):
        raise ValueError("Adapter warm start requires a fresh, empty --output-dir")

    if args.warm_start_mode == _SCENE_V8_WARM_START_MODE:
        pinned_context = prepare_v8_v7_checkpoint256_warm_start(checkpoint)
        source_config = HFDeltaMemConfig.from_pretrained(checkpoint)
        return AdapterWarmStartContext(
            checkpoint=checkpoint,
            mode=_SCENE_V8_WARM_START_MODE,
            source_protocol=pinned_context.source_training_protocol,
            source_config=source_config,
            manifest={
                "schema_version": _WARM_START_LINEAGE_SCHEMA_VERSION,
                "mode": _SCENE_V8_WARM_START_MODE,
                "source_checkpoint": str(checkpoint),
            },
            scene_v8_context=pinned_context,
            scene_v8_fresh_start=V8FreshStartContract(
                resume_from_checkpoint=None,
                initial_global_step=0,
                optimizer_created=False,
                scheduler_created=False,
                trainer_state_imported=False,
                rng_state_imported=False,
                optim=args.optim,
            ),
        )

    if checkpoint.name != f"checkpoint-{_RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP}":
        raise ValueError(
            "residual_hybrid_w8_ablation source must be checkpoint-416"
        )

    source_protocol = _load_json_object(
        checkpoint / _TRAINING_PROTOCOL_FILENAME,
        description="warm-start source training protocol",
    )
    _validate_residual_hybrid_w8_source_protocol(source_protocol)
    source_config = HFDeltaMemConfig.from_pretrained(checkpoint)
    if source_config.target_layers != _RESIDUAL_HYBRID_W8_TARGET_LAYERS:
        raise ValueError("Warm-start V14 source must wrap exactly layers 0-41")
    if source_config.memory_fusion_placement != "attention_output":
        raise ValueError("Warm-start V14 source must use attention_output fusion")
    if source_config.memory_fusion_residual_scale != 1.0:
        raise ValueError("Warm-start V14 source residual scale must be 1.0")
    if source_config.memory_fusion_residual_scale_max != 1.0:
        raise ValueError("Warm-start V14 source residual scale max must normalize to 1.0")

    trainer_state = _load_json_object(
        checkpoint / "trainer_state.json",
        description="warm-start source trainer state",
    )
    try:
        global_step = int(trainer_state["global_step"])
        max_steps = int(trainer_state["max_steps"])
        epoch = float(trainer_state["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Warm-start source trainer state requires global_step, max_steps, and epoch"
        ) from exc
    if (
        global_step != _RESIDUAL_HYBRID_W8_SOURCE_GLOBAL_STEP
        or max_steps != global_step
        or epoch != _RESIDUAL_HYBRID_W8_SOURCE_EPOCH
    ):
        raise ValueError(
            "Warm-start source must be the completed checkpoint-416 epoch-13 boundary"
        )

    source_pairing = _load_json_object(
        checkpoint / _CONTENT_CONTRAST_PAIRING_FILENAME,
        description="warm-start source content-contrast pairing manifest",
    )
    protocol_pairing = source_protocol.get("content_contrast_pairing")
    if not isinstance(protocol_pairing, dict) or (
        source_pairing.get("manifest_sha256")
        != protocol_pairing.get("manifest_sha256")
    ):
        raise ValueError(
            "Warm-start source pairing manifest does not match its training protocol"
        )

    source_optimizer_parameter_count = _load_optimizer_parameter_count(
        checkpoint / "optimizer.pt"
    )
    if (
        source_optimizer_parameter_count
        != _RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS
    ):
        raise ValueError(
            "Warm-start V14 optimizer must contain exactly "
            f"{_RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS} parameters"
        )
    rng_files = sorted(path for path in checkpoint.glob("rng_state*.pth") if path.is_file())
    manifest: dict[str, object] = {
        "schema_version": _WARM_START_LINEAGE_SCHEMA_VERSION,
        "mode": _RESIDUAL_HYBRID_W8_WARM_START_MODE,
        "source_checkpoint": str(checkpoint),
        "source_global_step": global_step,
        "source_epoch": epoch,
        "source_training_protocol_sha256": _protocol_sha256(source_protocol),
        "source_training_protocol_file_sha256": _sha256_file(
            checkpoint / _TRAINING_PROTOCOL_FILENAME
        ),
        "source_content_contrast_pairing_file_sha256": _sha256_file(
            checkpoint / _CONTENT_CONTRAST_PAIRING_FILENAME
        ),
        "source_content_contrast_pairing_manifest_sha256": source_pairing[
            "manifest_sha256"
        ],
        "source_delta_config_sha256": _protocol_sha256(source_config.to_dict()),
        "source_delta_config_file_sha256": _sha256_file(
            checkpoint / "delta_mem_config.json"
        ),
        "source_adapter_sha256": _sha256_file(checkpoint / "delta_mem_adapter.pt"),
        "source_optimizer_sha256": _sha256_file(checkpoint / "optimizer.pt"),
        "source_scheduler_sha256": _sha256_file(checkpoint / "scheduler.pt"),
        "source_trainer_state_sha256": _sha256_file(checkpoint / "trainer_state.json"),
        "source_rng_state_sha256": {
            path.name: _sha256_file(path) for path in rng_files
        },
        "source_optimizer_parameter_count": source_optimizer_parameter_count,
        "source_state_imports": {
            "adapter": True,
            "optimizer": False,
            "scheduler": False,
            "trainer_state": False,
            "rng": False,
        },
        "source_optimizer_imported": False,
        "source_scheduler_imported": False,
        "source_trainer_state_imported": False,
        "source_rng_state_imported": False,
        "target_initial_global_step": 0,
        "target_max_steps": _RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS,
        "target_num_train_epochs": _RESIDUAL_HYBRID_W8_TARGET_EPOCHS,
        "target_warmup_steps": _RESIDUAL_HYBRID_W8_TARGET_WARMUP_STEPS,
    }
    return AdapterWarmStartContext(
        checkpoint=checkpoint,
        mode=_RESIDUAL_HYBRID_W8_WARM_START_MODE,
        source_protocol=source_protocol,
        source_config=source_config,
        manifest=manifest,
    )


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
    residual_scale_max = float(
        normalized.get("memory_fusion_residual_scale_max", 1.0)
    )
    if not math.isfinite(residual_scale_max) or not 0.0 < residual_scale_max <= 1.0:
        raise ValueError(
            "Training protocol memory_fusion_residual_scale_max must be finite and satisfy "
            "0 < value <= 1"
        )
    normalized["memory_fusion_residual_scale_max"] = residual_scale_max
    return normalized


def _normalize_scene_boundary_payload_protocol(
    protocol: dict[str, object],
) -> dict[str, object]:
    normalized = dict(protocol)
    try:
        weight = float(normalized.get("scene_boundary_payload_ce_weight", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Training protocol scene_boundary_payload_ce_weight must be numeric"
        ) from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "Training protocol scene_boundary_payload_ce_weight must be finite and "
            "non-negative"
        )
    mask_mode = normalized.get(
        "scene_boundary_payload_mask_mode",
        _SCENE_BOUNDARY_PAYLOAD_MASK_MODE,
    )
    if mask_mode != _SCENE_BOUNDARY_PAYLOAD_MASK_MODE:
        raise ValueError("Training protocol has an unsupported scene-boundary payload mask mode")
    normalization_present = "scene_boundary_payload_ce_normalization" in normalized
    if weight > 0.0 and not normalization_present:
        raise ValueError(
            "Weighted scene-boundary training protocol is missing its payload CE "
            "normalization"
        )
    normalization = normalized.get(
        "scene_boundary_payload_ce_normalization",
        _SCENE_BOUNDARY_PAYLOAD_CE_NORMALIZATION,
    )
    if normalization != _SCENE_BOUNDARY_PAYLOAD_CE_NORMALIZATION:
        raise ValueError(
            "Training protocol has an unsupported scene-boundary payload CE normalization"
        )
    normalized["scene_boundary_payload_ce_weight"] = weight
    normalized["scene_boundary_payload_mask_mode"] = mask_mode
    normalized["scene_boundary_payload_ce_normalization"] = normalization
    return normalized


def _normalize_train_sampler_protocol(
    protocol: dict[str, object],
) -> dict[str, object]:
    normalized = dict(protocol)
    seed = normalized.get("train_sampler_seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Training protocol train_sampler_seed must be an integer or null")
        if not 0 <= seed <= torch.iinfo(torch.int64).max:
            raise ValueError(
                "Training protocol train_sampler_seed must satisfy 0 <= seed <= 2^63 - 1"
            )
    train_schedule = normalized.get("train_schedule")
    if train_schedule is not None and not isinstance(train_schedule, dict):
        raise ValueError("Training protocol train_schedule must be an object or null")
    expected_mode = (
        _FIXED_TRAIN_SCHEDULE_SAMPLER_MODE
        if train_schedule is not None
        else (
            _DEFAULT_TRAIN_SAMPLER_MODE
            if seed is None
            else _SEEDED_TRAIN_SAMPLER_MODE
        )
    )
    mode_present = "train_sampler_mode" in normalized
    if (seed is not None or train_schedule is not None) and not mode_present:
        qualifier = "Fixed-schedule" if train_schedule is not None else "Seeded"
        raise ValueError(
            f"{qualifier} training protocol is missing its train_sampler_mode"
        )
    mode = normalized.get("train_sampler_mode", expected_mode)
    if mode != expected_mode:
        raise ValueError(
            "Training protocol train_sampler_mode does not match its seed/schedule"
        )
    if train_schedule is not None and seed is not None:
        raise ValueError("Fixed training schedule cannot also bind a sampler seed")
    normalized["train_sampler_seed"] = seed
    normalized["train_sampler_mode"] = mode
    normalized["train_schedule"] = train_schedule
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
    if pairing.get("target_mode") != _CONTENT_CONTRAST_TARGET_MODE:
        raise ValueError("objective_ablation target pairing target_mode does not match")
    if pairing.get("target_span_tokens") != _CONTENT_CONTRAST_TARGET_SPAN_TOKENS:
        raise ValueError(
            "objective_ablation target pairing target_span_tokens does not match"
        )
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
        if split.get("target_mode") != _CONTENT_CONTRAST_TARGET_MODE:
            raise ValueError(
                f"objective_ablation target pairing split {split_name} has invalid target_mode"
            )
        if split.get("target_span_tokens") != _CONTENT_CONTRAST_TARGET_SPAN_TOKENS:
            raise ValueError(
                f"objective_ablation target pairing split {split_name} has invalid "
                "target_span_tokens"
            )
        if split.get("target_token_count") != (
            sample_count * _CONTENT_CONTRAST_TARGET_SPAN_TOKENS
        ):
            raise ValueError(
                f"objective_ablation target pairing split {split_name} has invalid "
                "target_token_count"
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
        "content_contrast_negative_priming_grad": True,
        "content_contrast_backward_mode": _CONTENT_CONTRAST_BACKWARD_MODE,
        "content_contrast_read_mask_mode": _CONTENT_CONTRAST_READ_MASK_MODE,
        "content_contrast_target_mode": _CONTENT_CONTRAST_TARGET_MODE,
        "content_contrast_target_span_tokens": _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
        "content_contrast_previous_source_grad": _CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD,
        "content_contrast_representation_mode": (
            _CONTENT_CONTRAST_REPRESENTATION_MODE
        ),
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
        representation_weight = float(target_protocol["memory_representation_weight"])
        representation_margin = float(target_protocol["memory_representation_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "objective_ablation target requires numeric contrast and representation weights "
            "and margins"
        ) from exc
    if not math.isfinite(contrast_weight) or contrast_weight <= 0.0:
        raise ValueError(
            "objective_ablation target requires memory_contrast_weight to be positive"
        )
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("objective_ablation target requires memory_margin to be positive")
    if not math.isfinite(representation_weight) or representation_weight < 0.0:
        raise ValueError(
            "objective_ablation target requires memory_representation_weight to be "
            "finite and non-negative"
        )
    if not math.isfinite(representation_margin) or representation_margin <= 0.0:
        raise ValueError(
            "objective_ablation target requires memory_representation_margin to be positive"
        )
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
    source_protocol = _normalize_scene_boundary_payload_protocol(source_protocol)
    target_protocol = _normalize_scene_boundary_payload_protocol(target_protocol)
    source_protocol = _normalize_train_sampler_protocol(source_protocol)
    target_protocol = _normalize_train_sampler_protocol(target_protocol)
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


def _validate_residual_hybrid_w8_delta_config_transition(
    source_config: HFDeltaMemConfig,
    target_config: HFDeltaMemConfig,
) -> None:
    if source_config.target_layers != _RESIDUAL_HYBRID_W8_TARGET_LAYERS:
        raise ValueError("Warm-start source config must target exactly layers 0-41")
    if target_config.target_layers != _RESIDUAL_HYBRID_W8_TARGET_LAYERS:
        raise ValueError("Warm-start target config must target exactly layers 0-41")
    if target_config.memory_fusion_placement != _RESIDUAL_HYBRID_W8_TARGET_PLACEMENT:
        raise ValueError("Warm-start target config must use residual-hybrid placement")
    if (
        target_config.memory_fusion_residual_scale
        != _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE
    ):
        raise ValueError("Warm-start target config must initialize effective gamma at 0.01")
    if (
        target_config.memory_fusion_residual_scale_max
        != _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
    ):
        raise ValueError("Warm-start target config must cap effective gamma at 0.02")
    source = source_config.to_dict()
    target = target_config.to_dict()
    allowed_mismatches = {
        "memory_fusion_placement",
        "memory_fusion_residual_scale",
        "memory_fusion_residual_scale_max",
    }
    mismatches = {
        key for key in set(source) | set(target) if source.get(key) != target.get(key)
    }
    if mismatches != allowed_mismatches:
        raise ValueError(
            "Residual-hybrid warm start requires only placement and residual-gain config "
            "changes; mismatches: " + ", ".join(sorted(mismatches))
        )


def validate_residual_hybrid_w8_adapter_topology(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    *,
    gain_names: tuple[str, ...],
    source_optimizer_parameter_count: int,
    target_trainable_tensor_count: int,
) -> dict[str, object]:
    if len(source_state) != _RESIDUAL_HYBRID_W8_SOURCE_ADAPTER_TENSORS:
        raise ValueError(
            "Warm-start source adapter must contain exactly "
            f"{_RESIDUAL_HYBRID_W8_SOURCE_ADAPTER_TENSORS} tensors"
        )
    if len(target_state) != _RESIDUAL_HYBRID_W8_TARGET_ADAPTER_TENSORS:
        raise ValueError(
            "Warm-start target adapter must contain exactly "
            f"{_RESIDUAL_HYBRID_W8_TARGET_ADAPTER_TENSORS} tensors"
        )
    if (
        len(gain_names) != _RESIDUAL_HYBRID_W8_TARGET_NEW_GAIN_TENSORS
        or len(set(gain_names)) != len(gain_names)
    ):
        raise ValueError("Warm-start target must expose exactly 42 unique residual gains")
    expected_gain_set = set(gain_names)
    target_gain_names = tuple(
        name
        for name in target_state
        if name.endswith(".memory_fusion_residual_gain_raw")
    )
    if target_gain_names != gain_names:
        raise ValueError(
            "Warm-start target must contain exactly one ordered residual gain per layer"
        )
    source_gain_names = [name for name in source_state if name in expected_gain_set]
    if source_gain_names:
        raise ValueError("Warm-start V14 source must not contain residual-hybrid gains")
    missing = [name for name in target_state if name not in source_state]
    extra = [name for name in source_state if name not in target_state]
    if tuple(missing) != gain_names or extra:
        raise ValueError(
            "Warm-start topology must add only the 42 residual gains; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    target_shared_names = [name for name in target_state if name not in expected_gain_set]
    if list(source_state) != target_shared_names:
        raise ValueError(
            "Warm-start shared adapter parameter names must remain in exact source order"
        )
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(
            target_tensor, torch.Tensor
        ):
            raise ValueError(f"Warm-start adapter entry is not a tensor: {name}")
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Warm-start shared adapter shape mismatch for {name}: "
                f"source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise ValueError(
                f"Warm-start shared adapter dtype mismatch for {name}: "
                f"source={source_tensor.dtype} target={target_tensor.dtype}"
            )
    if (
        source_optimizer_parameter_count
        != _RESIDUAL_HYBRID_W8_SOURCE_OPTIMIZER_PARAMETERS
    ):
        raise ValueError("Warm-start source optimizer parameter count must be 1218")
    if (
        target_trainable_tensor_count
        != source_optimizer_parameter_count + len(gain_names)
        or target_trainable_tensor_count
        != _RESIDUAL_HYBRID_W8_TARGET_TRAINABLE_TENSORS
    ):
        raise ValueError(
            "Warm-start target trainable tensor count must equal source optimizer count + 42"
        )
    shared_target_state = {
        name: target_state[name] for name in target_shared_names
    }
    source_topology_sha256 = _adapter_topology_sha256(source_state)
    shared_target_topology_sha256 = _adapter_topology_sha256(shared_target_state)
    if source_topology_sha256 != shared_target_topology_sha256:
        raise ValueError("Warm-start shared source/target topology hashes do not match")
    return {
        "source_adapter_tensor_count": len(source_state),
        "target_adapter_tensor_count": len(target_state),
        "shared_adapter_tensor_count": len(target_shared_names),
        "new_residual_gain_tensor_count": len(gain_names),
        "target_trainable_tensor_count": target_trainable_tensor_count,
        "source_adapter_topology_sha256": source_topology_sha256,
        "target_shared_adapter_topology_sha256": shared_target_topology_sha256,
        "target_adapter_topology_sha256": _adapter_topology_sha256(target_state),
    }


def apply_adapter_warm_start(
    model: nn.Module,
    context: AdapterWarmStartContext,
    target_config: HFDeltaMemConfig,
    trainable_names: list[str],
) -> dict[str, object]:
    if context.mode == _SCENE_V8_WARM_START_MODE:
        if (
            context.scene_v8_context is None
            or context.scene_v8_fresh_start is None
        ):
            raise ValueError("Scene V8 warm-start context is incomplete")
        source_config = context.source_config.to_dict()
        target_config_payload = target_config.to_dict()
        config_mismatches = sorted(
            key
            for key in set(source_config) | set(target_config_payload)
            if source_config.get(key) != target_config_payload.get(key)
        )
        if config_mismatches:
            raise ValueError(
                "Scene V8 requires topology-exact V7/V8 Delta-Mem config; differs for: "
                + ", ".join(config_mismatches)
            )
        receipt = apply_v8_v7_checkpoint256_adapter_only_warm_start(
            model,
            context.scene_v8_context,
            fresh_start=context.scene_v8_fresh_start,
        )
        receipt.update(
            {
                "target_delta_config_sha256": _protocol_sha256(
                    target_config_payload
                ),
                "target_trainable_tensor_count": len(trainable_names),
                "target_trainable_names_sha256": _protocol_sha256(
                    {"ordered_trainable_names": trainable_names}
                ),
            }
        )
        receipt_without_hash = dict(receipt)
        receipt_without_hash.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = _canonical_json_sha256(receipt_without_hash)
        return receipt
    if context.mode != _RESIDUAL_HYBRID_W8_WARM_START_MODE:
        raise ValueError(f"Unsupported adapter warm-start mode: {context.mode}")
    _validate_residual_hybrid_w8_delta_config_transition(
        context.source_config,
        target_config,
    )
    source_state = torch.load(
        context.checkpoint / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source_state, dict):
        raise ValueError("Warm-start source adapter must contain a state dictionary")
    modules = list(iter_delta_mem_modules(model))
    for _, module in modules:
        module.set_memory_fusion_residual_gain(
            target_config.memory_fusion_residual_scale
        )
    target_state = get_delta_mem_state_dict(model)
    wrapped_layers = tuple(int(module.base.layer_idx) for _, module in modules)
    if wrapped_layers != _RESIDUAL_HYBRID_W8_TARGET_LAYERS:
        raise ValueError("Warm-start target model must wrap ordered layers 0-41")
    gain_names = tuple(
        f"{name}.memory_fusion_residual_gain_raw" for name, _ in modules
    )
    topology_manifest = validate_residual_hybrid_w8_adapter_topology(
        source_state,
        target_state,
        gain_names=gain_names,
        source_optimizer_parameter_count=int(
            context.manifest["source_optimizer_parameter_count"]
        ),
        target_trainable_tensor_count=len(trainable_names),
    )
    if not set(gain_names).issubset(trainable_names):
        raise ValueError("Every residual-hybrid gain must be a trainable target tensor")
    initial_gains = {name: target_state[name].clone() for name in gain_names}
    loaded_config = load_delta_mem_adapter(
        model,
        context.checkpoint,
        initialize_missing_residual_hybrid_gain=True,
    )
    if loaded_config.to_dict() != context.source_config.to_dict():
        raise ValueError("Warm-start adapter loader returned an unexpected source config")
    loaded_state = get_delta_mem_state_dict(model)
    if list(loaded_state) != list(target_state):
        raise ValueError("Warm-start adapter topology changed during loading")
    unequal_shared = [
        name
        for name, source_tensor in source_state.items()
        if not torch.equal(loaded_state[name], source_tensor)
    ]
    if unequal_shared:
        raise ValueError(
            "Warm-start shared adapter tensors are not bit-equal after loading: "
            + ", ".join(unequal_shared[:8])
        )
    changed_gains = [
        name
        for name, initial_tensor in initial_gains.items()
        if not torch.equal(loaded_state[name], initial_tensor)
    ]
    if changed_gains:
        raise ValueError(
            "Warm-start adapter loading changed newly initialized residual gains: "
            + ", ".join(changed_gains[:8])
        )
    effective_gains = []
    for _, module in modules:
        parameter = module.memory_fusion_residual_gain_raw
        resolved = module._resolved_memory_fusion_residual_gain(
            device=parameter.device,
            dtype=torch.float32,
        )
        effective_gains.append(float(resolved.detach().cpu().item()))
    if any(
        not math.isclose(
            value,
            _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for value in effective_gains
    ):
        raise ValueError("Warm-start target effective residual gains must initialize at 0.01")
    return {
        **topology_manifest,
        "target_delta_config_sha256": _protocol_sha256(target_config.to_dict()),
        "shared_adapter_bit_equality_verified": True,
        "new_residual_gains_preserved_during_load": True,
        "target_effective_residual_gain_initial": (
            _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE
        ),
        "target_effective_residual_gain_max": (
            _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
        ),
    }


def validate_residual_hybrid_w8_target_protocol(
    source_protocol: dict[str, object],
    target_protocol: dict[str, object],
) -> None:
    _validate_residual_hybrid_w8_source_protocol(source_protocol)
    _validate_objective_ablation_target_protocol(
        target_protocol,
        require_pairing=True,
    )
    expected = {
        "schema_version": _CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION,
        "memory_objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "memory_loss_mode": "content_contrast_ce",
        "memory_fusion_placement": _RESIDUAL_HYBRID_W8_TARGET_PLACEMENT,
        "memory_fusion_residual_scale": _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE,
        "memory_fusion_residual_scale_max": (
            _RESIDUAL_HYBRID_W8_TARGET_RESIDUAL_SCALE_MAX
        ),
        "content_contrast_target_mode": _CONTENT_CONTRAST_TARGET_MODE,
        "content_contrast_target_span_tokens": _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
        "max_steps": _RESIDUAL_HYBRID_W8_TARGET_MAX_STEPS,
        "num_train_epochs": _RESIDUAL_HYBRID_W8_TARGET_EPOCHS,
        "warmup_steps": _RESIDUAL_HYBRID_W8_TARGET_WARMUP_STEPS,
    }
    mismatches = [
        name for name, expected_value in expected.items()
        if target_protocol.get(name) != expected_value
    ]
    source = _normalize_train_sampler_protocol(
        _normalize_scene_boundary_payload_protocol(
            _normalize_memory_fusion_placement_protocol(source_protocol)
        )
    )
    target = _normalize_train_sampler_protocol(
        _normalize_scene_boundary_payload_protocol(
            _normalize_memory_fusion_placement_protocol(target_protocol)
        )
    )
    unexpected_drift = sorted(
        key
        for key in set(source) | set(target)
        if source.get(key) != target.get(key)
        and key not in _RESIDUAL_HYBRID_W8_PROTOCOL_DRIFT
    )
    if mismatches or unexpected_drift:
        invalid = sorted(set(mismatches) | set(unexpected_drift))
        raise ValueError(
            "Residual-hybrid W8 warm-start target protocol mismatch for: "
            + ", ".join(invalid)
        )


def _lineage_manifest_filename(manifest: dict[str, object]) -> str:
    if manifest.get("mode") in _WARM_START_MODES:
        return _WARM_START_LINEAGE_FILENAME
    if manifest.get("mode") in _ABLATION_RESUME_MODES:
        return _ABLATION_LINEAGE_FILENAME
    return _CONTINUATION_MANIFEST_FILENAME


def _scene_memory_v8_checkpoint_steps(
    checkpoint_steps: object,
) -> tuple[int, ...]:
    if not isinstance(checkpoint_steps, (list, tuple)):
        raise ValueError("Scene-memory V8 checkpoint endpoints must be a list or tuple")
    normalized = tuple(checkpoint_steps)
    if (
        len(normalized) < 2
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in normalized
        )
        or tuple(sorted(set(normalized))) != normalized
    ):
        raise ValueError(
            "Scene-memory V8 checkpoint endpoints must be unique increasing integers"
        )
    return normalized


def _validate_scene_memory_v8_resume_endpoint(
    *,
    source_global_step: object,
    target_max_steps: object,
    checkpoint_steps: object,
) -> None:
    endpoints = _scene_memory_v8_checkpoint_steps(checkpoint_steps)
    if isinstance(source_global_step, bool) or not isinstance(source_global_step, int):
        raise ValueError("Scene-memory V8 resume source step must be an integer")
    if isinstance(target_max_steps, bool) or not isinstance(target_max_steps, int):
        raise ValueError("Scene-memory V8 resume target step must be an integer")
    if source_global_step not in endpoints:
        raise ValueError(
            "Scene-memory V8 resume source is not a locked checkpoint endpoint"
        )
    source_position = endpoints.index(source_global_step)
    if source_position + 1 >= len(endpoints):
        raise ValueError("Scene-memory V8 final checkpoint has no resume endpoint")
    expected_target = endpoints[source_position + 1]
    if target_max_steps != expected_target:
        raise ValueError(
            "Scene-memory V8 resume must advance to the immediate next locked "
            f"endpoint: checkpoint-{source_global_step} -> {expected_target}"
        )


def _scene_memory_v8_protocol_checkpoint_steps(
    protocol: dict[str, object] | None,
) -> tuple[int, ...] | None:
    if protocol is None:
        return None
    schedule = protocol.get("train_schedule")
    if not isinstance(schedule, dict) or schedule.get("schema") != (
        _SCENE_MEMORY_V8_CURRICULUM_SCHEMA
    ):
        return None
    return _scene_memory_v8_checkpoint_steps(schedule.get("checkpoint_steps"))


def _validate_scene_memory_v8_warm_start_lineage(
    manifest: dict[str, object],
    *,
    target_training_protocol_sha256: str,
) -> str:
    if (
        manifest.get("schema") != SCENE_V8_WARM_START_RECEIPT_SCHEMA
        or manifest.get("schema_version") != _WARM_START_LINEAGE_SCHEMA_VERSION
        or manifest.get("mode") != _SCENE_V8_WARM_START_MODE
    ):
        raise ValueError("Scene-memory V8 warm-start lineage schema or mode differs")
    unsigned = dict(manifest)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if not _is_sha256(receipt_sha256) or receipt_sha256 != (
        _canonical_json_sha256(unsigned)
    ):
        raise ValueError("Scene-memory V8 warm-start lineage receipt hash differs")
    if (
        not _is_sha256(target_training_protocol_sha256)
        or manifest.get("target_training_protocol_sha256")
        != target_training_protocol_sha256
    ):
        raise ValueError(
            "Scene-memory V8 warm-start lineage target protocol hash differs"
        )
    lock = load_v8_warm_start_lock(SCENE_V8_WARM_START_LOCK_PATH)
    source_lock = manifest.get("source_lock")
    if not isinstance(source_lock, dict):
        raise ValueError("Scene-memory V8 warm-start lineage source lock is missing")
    expected_source = Path(str(lock.get("source_checkpoint", ""))).expanduser().resolve()
    recorded_source = Path(str(manifest.get("source_checkpoint", ""))).expanduser().resolve()
    if (
        recorded_source != expected_source
        or recorded_source.name != "checkpoint-256"
        or manifest.get("source_global_step") != 256
        or Path(str(source_lock.get("path", ""))).expanduser().resolve()
        != SCENE_V8_WARM_START_LOCK_PATH.resolve()
        or source_lock.get("lock_sha256") != lock.get("lock_sha256")
        or manifest.get("source_state_imports") != lock.get("source_state_imports")
        or manifest.get("source_artifacts") != lock.get("artifacts")
        or manifest.get("post_load_bit_equal") is not True
    ):
        raise ValueError("Scene-memory V8 warm-start lineage is not pinned to V7-256")
    expected_evidence = {
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "pre_train_global_step": 0,
        "fresh_optimizer_created": True,
        "fresh_optimizer_state_entries_before_train": 0,
        "fresh_scheduler_created_before_train": False,
    }
    evidence_drift = [
        name
        for name, expected in expected_evidence.items()
        if manifest.get(name) != expected
    ]
    if evidence_drift:
        raise ValueError(
            "Scene-memory V8 warm-start fresh-state evidence differs for: "
            + ", ".join(evidence_drift)
        )
    expected_fresh_start = {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v8_seed",
    }
    optimizer_class = manifest.get("fresh_optimizer_class")
    if (
        manifest.get("target_fresh_start") != expected_fresh_start
        or not isinstance(optimizer_class, str)
        or not optimizer_class.endswith(".AdamW")
    ):
        raise ValueError("Scene-memory V8 warm-start optimizer receipt differs")
    return receipt_sha256


def _scene_memory_v8_checkpoint_lineage(
    checkpoint: Path,
    *,
    checkpoint_steps: object,
    visited: set[Path] | None = None,
) -> dict[str, object]:
    requested = checkpoint.expanduser()
    if requested.is_symlink():
        raise ValueError("Scene-memory V8 lineage checkpoint must not be a symlink")
    resolved = requested.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Scene-memory V8 lineage checkpoint does not exist: {resolved}"
        )
    active_visited = set() if visited is None else visited
    if resolved in active_visited:
        raise ValueError("Scene-memory V8 continuation lineage contains a cycle")
    active_visited.add(resolved)
    endpoints = _scene_memory_v8_checkpoint_steps(checkpoint_steps)
    checkpoint_suffix = resolved.name.removeprefix("checkpoint-")
    if not resolved.name.startswith("checkpoint-") or not checkpoint_suffix.isdigit():
        raise ValueError("Scene-memory V8 lineage source must be checkpoint-N")
    checkpoint_step = int(checkpoint_suffix)
    if checkpoint_step not in endpoints:
        raise ValueError("Scene-memory V8 lineage source is not a locked endpoint")
    trainer_state = _load_json_object(
        resolved / "trainer_state.json",
        description="Scene-memory V8 lineage trainer state",
    )
    protocol = _load_json_object(
        resolved / _TRAINING_PROTOCOL_FILENAME,
        description="Scene-memory V8 lineage training protocol",
    )
    try:
        state_step = int(trainer_state["global_step"])
        state_max_steps = int(trainer_state["max_steps"])
        protocol_max_steps = int(protocol["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Scene-memory V8 lineage checkpoint horizon is invalid"
        ) from exc
    if checkpoint_step != state_step or state_step != state_max_steps or (
        state_step != protocol_max_steps
    ):
        raise ValueError("Scene-memory V8 lineage checkpoint is not a completed horizon")
    protocol_sha256 = _protocol_sha256(protocol)
    if _scene_memory_v8_protocol_checkpoint_steps(protocol) != endpoints:
        raise ValueError("Scene-memory V8 lineage protocol curriculum differs")
    expected_filename = (
        _WARM_START_LINEAGE_FILENAME
        if checkpoint_step == endpoints[0]
        else _CONTINUATION_MANIFEST_FILENAME
    )
    lineage_candidates = (
        _WARM_START_LINEAGE_FILENAME,
        _CONTINUATION_MANIFEST_FILENAME,
        _ABLATION_LINEAGE_FILENAME,
    )
    present = [
        filename for filename in lineage_candidates if (resolved / filename).is_file()
    ]
    if present != [expected_filename]:
        raise ValueError(
            "Scene-memory V8 checkpoint must contain exactly its expected lineage file"
        )
    lineage_path = resolved / expected_filename
    if lineage_path.is_symlink():
        raise ValueError("Scene-memory V8 lineage file must not be a symlink")
    lineage = _load_json_object(
        lineage_path,
        description="Scene-memory V8 checkpoint lineage",
    )
    lineage_file_sha256 = _sha256_file(lineage_path)
    if checkpoint_step == endpoints[0]:
        root_receipt_sha256 = _validate_scene_memory_v8_warm_start_lineage(
            lineage,
            target_training_protocol_sha256=protocol_sha256,
        )
    else:
        if (
            lineage.get("schema_version") != _CONTINUATION_MANIFEST_SCHEMA_VERSION
            or lineage.get("mode") != "extend"
        ):
            raise ValueError(
                "Scene-memory V8 continuation lineage schema or mode differs"
            )
        unsigned = dict(lineage)
        manifest_sha256 = unsigned.pop("manifest_sha256", None)
        if not _is_sha256(manifest_sha256) or manifest_sha256 != (
            _canonical_json_sha256(unsigned)
        ):
            raise ValueError("Scene-memory V8 continuation lineage self-hash differs")
        source_step = lineage.get("source_global_step")
        _validate_scene_memory_v8_resume_endpoint(
            source_global_step=source_step,
            target_max_steps=checkpoint_step,
            checkpoint_steps=endpoints,
        )
        if (
            lineage.get("target_max_steps") != checkpoint_step
            or lineage.get("target_training_protocol_sha256") != protocol_sha256
        ):
            raise ValueError(
                "Scene-memory V8 continuation target horizon or protocol differs"
            )
        source_checkpoint = Path(
            str(lineage.get("source_checkpoint", ""))
        ).expanduser().resolve()
        if lineage.get("source_checkpoint") != str(source_checkpoint):
            raise ValueError(
                "Scene-memory V8 continuation source checkpoint is not canonical"
            )
        source_lineage = _scene_memory_v8_checkpoint_lineage(
            source_checkpoint,
            checkpoint_steps=endpoints,
            visited=active_visited,
        )
        expected_source_filename = source_lineage["lineage_filename"]
        if (
            source_lineage["checkpoint_step"] != source_step
            or lineage.get("source_lineage_filename") != expected_source_filename
            or lineage.get("source_lineage_file_sha256")
            != source_lineage["lineage_file_sha256"]
            or lineage.get("source_training_protocol_sha256")
            != source_lineage["training_protocol_sha256"]
            or lineage.get("root_warm_start_receipt_sha256")
            != source_lineage["root_warm_start_receipt_sha256"]
        ):
            raise ValueError(
                "Scene-memory V8 continuation immediate-source lineage differs"
            )
        root_receipt_sha256 = str(
            source_lineage["root_warm_start_receipt_sha256"]
        )
    active_visited.remove(resolved)
    return {
        "checkpoint": str(resolved),
        "checkpoint_step": checkpoint_step,
        "lineage_filename": expected_filename,
        "lineage_file_sha256": lineage_file_sha256,
        "training_protocol_sha256": protocol_sha256,
        "root_warm_start_receipt_sha256": root_receipt_sha256,
    }


def prepare_scene_memory_v8_training_continuation(
    continuation_manifest: dict[str, object] | None,
    *,
    resume_from_checkpoint: str | Path,
    checkpoint_steps: object,
) -> dict[str, object]:
    if continuation_manifest is None or continuation_manifest.get("mode") != "extend":
        raise ValueError("Scene-memory V8 resume requires an extend continuation manifest")
    source_checkpoint = Path(resume_from_checkpoint).expanduser().resolve()
    source_lineage = _scene_memory_v8_checkpoint_lineage(
        source_checkpoint,
        checkpoint_steps=checkpoint_steps,
    )
    source_step = source_lineage["checkpoint_step"]
    _validate_scene_memory_v8_resume_endpoint(
        source_global_step=source_step,
        target_max_steps=continuation_manifest.get("target_max_steps"),
        checkpoint_steps=checkpoint_steps,
    )
    recorded_source_checkpoint = continuation_manifest.get("source_checkpoint")
    if (
        recorded_source_checkpoint != str(source_checkpoint)
        or continuation_manifest.get("source_global_step") != source_step
        or continuation_manifest.get("source_training_protocol_sha256")
        != source_lineage["training_protocol_sha256"]
    ):
        raise ValueError("Scene-memory V8 continuation source checkpoint differs")
    prepared = dict(continuation_manifest)
    prepared.update(
        {
            "root_warm_start_receipt_sha256": source_lineage[
                "root_warm_start_receipt_sha256"
            ],
            "source_lineage_filename": source_lineage["lineage_filename"],
            "source_lineage_file_sha256": source_lineage[
                "lineage_file_sha256"
            ],
        }
    )
    prepared.pop("target_training_protocol_sha256", None)
    prepared.pop("manifest_sha256", None)
    return prepared


def finalize_scene_memory_v8_training_continuation(
    continuation_manifest: dict[str, object],
    *,
    target_training_protocol: dict[str, object],
) -> None:
    target_training_protocol_sha256 = _protocol_sha256(target_training_protocol)
    continuation_manifest["target_training_protocol_sha256"] = (
        target_training_protocol_sha256
    )
    unsigned = dict(continuation_manifest)
    unsigned.pop("manifest_sha256", None)
    continuation_manifest["manifest_sha256"] = _canonical_json_sha256(unsigned)


def validate_scene_memory_v8_active_continuation(
    continuation_manifest: dict[str, object] | None,
    *,
    resume_from_checkpoint: str | Path,
    target_training_protocol: dict[str, object],
    checkpoint_steps: object,
) -> None:
    if continuation_manifest is None:
        raise ValueError("Scene-memory V8 Trainer resume lineage is missing")
    unsigned = dict(continuation_manifest)
    manifest_sha256 = unsigned.pop("manifest_sha256", None)
    if not _is_sha256(manifest_sha256) or manifest_sha256 != (
        _canonical_json_sha256(unsigned)
    ):
        raise ValueError("Scene-memory V8 active continuation self-hash differs")
    prepared = prepare_scene_memory_v8_training_continuation(
        continuation_manifest,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_steps=checkpoint_steps,
    )
    expected = dict(prepared)
    expected["target_training_protocol_sha256"] = _protocol_sha256(
        target_training_protocol
    )
    expected_unsigned = dict(expected)
    expected_unsigned.pop("manifest_sha256", None)
    expected["manifest_sha256"] = _canonical_json_sha256(expected_unsigned)
    if continuation_manifest != expected:
        raise ValueError("Scene-memory V8 active continuation lineage differs")
    try:
        target_max_steps = int(target_training_protocol["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Scene-memory V8 target protocol max_steps is invalid") from exc
    if continuation_manifest.get("target_max_steps") != target_max_steps:
        raise ValueError("Scene-memory V8 continuation target protocol horizon differs")


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
                checkpoint / _WARM_START_LINEAGE_FILENAME,
            )
            if path.is_file()
        ]
        if not existing:
            return None
        if len(existing) != 1:
            raise ValueError(f"Checkpoint has ambiguous resume lineage manifests: {checkpoint}")
        manifest_path = existing[0]
        manifest = _load_json_object(manifest_path, description="resume lineage manifest")
        if manifest_path.name == _WARM_START_LINEAGE_FILENAME:
            valid_mode = manifest.get("mode") in _WARM_START_MODES
            expected_schema = _WARM_START_LINEAGE_SCHEMA_VERSION
        elif manifest_path.name == _ABLATION_LINEAGE_FILENAME:
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
                "memory_representation_weight": args.memory_representation_weight,
                "memory_representation_margin": args.memory_representation_margin,
                "memory_kl_weight": args.memory_kl_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": (
                    args.memory_partition_alignment_weight
                ),
                "memory_partition_entropy_weight": args.memory_partition_entropy_weight,
                "memory_partition_balance_weight": args.memory_partition_balance_weight,
                "content_contrast_negative_priming_grad": True,
                "content_contrast_backward_mode": _CONTENT_CONTRAST_BACKWARD_MODE,
                "content_contrast_read_mask_mode": _CONTENT_CONTRAST_READ_MASK_MODE,
                "content_contrast_target_mode": _CONTENT_CONTRAST_TARGET_MODE,
                "content_contrast_target_span_tokens": (
                    _CONTENT_CONTRAST_TARGET_SPAN_TOKENS
                ),
                "content_contrast_previous_source_grad": (
                    _CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD
                ),
                "content_contrast_representation_mode": (
                    _CONTENT_CONTRAST_REPRESENTATION_MODE
                ),
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
                "target_content_contrast_backward_mode": (
                    _CONTENT_CONTRAST_BACKWARD_MODE
                ),
                "target_content_contrast_read_mask_mode": (
                    _CONTENT_CONTRAST_READ_MASK_MODE
                ),
                "target_content_contrast_target_mode": (
                    _CONTENT_CONTRAST_TARGET_MODE
                ),
                "target_content_contrast_target_span_tokens": (
                    _CONTENT_CONTRAST_TARGET_SPAN_TOKENS
                ),
                "target_content_contrast_previous_source_grad": (
                    _CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD
                ),
                "target_content_contrast_representation_mode": (
                    _CONTENT_CONTRAST_REPRESENTATION_MODE
                ),
                "target_memory_contrast_weight": float(args.memory_contrast_weight),
                "target_memory_margin": float(args.memory_margin),
                "target_memory_representation_weight": float(
                    args.memory_representation_weight
                ),
                "target_memory_representation_margin": float(
                    args.memory_representation_margin
                ),
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
    explicit_warmup_steps: int | None = None,
) -> int:
    if explicit_warmup_steps is not None:
        if isinstance(explicit_warmup_steps, bool) or not isinstance(
            explicit_warmup_steps,
            int,
        ):
            raise ValueError("Explicit warmup steps must be an integer or null")
        if explicit_warmup_steps < 0:
            raise ValueError("Explicit warmup steps must be non-negative")
        if warmup_ratio != 0.0:
            raise ValueError(
                "--warmup-steps requires --warmup-ratio 0 to avoid an ambiguous schedule"
            )
        return explicit_warmup_steps
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


def resolve_trainer_resume_checkpoint(
    resume_from_checkpoint: str | None,
    warm_start_context: AdapterWarmStartContext | None,
) -> str | None:
    if warm_start_context is None:
        return resume_from_checkpoint
    if resume_from_checkpoint is not None:
        raise RuntimeError("Adapter warm start must not restore Trainer checkpoint state")
    return None


def finalize_adapter_warm_start_lineage(
    context: AdapterWarmStartContext,
    *,
    target_training_protocol_sha256: str,
    target_pairing_manifest: dict[str, object],
) -> None:
    pairing_sha256 = target_pairing_manifest.get("manifest_sha256")
    if not _is_sha256(target_training_protocol_sha256) or not _is_sha256(
        pairing_sha256
    ):
        raise ValueError("Warm-start target protocol and pairing hashes must be SHA256 values")
    context.manifest.update(
        {
            "target_training_protocol_sha256": target_training_protocol_sha256,
            "target_content_contrast_pairing_manifest_sha256": pairing_sha256,
            "trainer_resume_from_checkpoint": None,
            "fresh_optimizer_created": True,
        }
    )


def finalize_scene_v8_warm_start_lineage(
    context: AdapterWarmStartContext,
    *,
    target_training_protocol_sha256: str,
    target_pairing_manifest: dict[str, object],
) -> None:
    if context.mode != _SCENE_V8_WARM_START_MODE:
        raise ValueError("Scene V8 lineage finalizer received another warm-start mode")
    pairing_sha256 = target_pairing_manifest.get("manifest_sha256")
    if not _is_sha256(target_training_protocol_sha256) or not _is_sha256(
        pairing_sha256
    ):
        raise ValueError(
            "Scene V8 target protocol and pairing hashes must be SHA256 values"
        )
    context.manifest.update(
        {
            "target_training_protocol_sha256": target_training_protocol_sha256,
            "target_scene_state_pairing_manifest_sha256": pairing_sha256,
            "trainer_resume_from_checkpoint": None,
            "target_initial_global_step": 0,
            "fresh_adamw_creation_required_after_adapter_load": True,
        }
    )
    unsigned = dict(context.manifest)
    unsigned.pop("receipt_sha256", None)
    context.manifest["receipt_sha256"] = _canonical_json_sha256(unsigned)


def record_scene_v8_fresh_optimizer_lineage(
    trainer,
    warm_start_context: AdapterWarmStartContext,
) -> None:
    if warm_start_context.mode != _SCENE_V8_WARM_START_MODE:
        raise ValueError("Scene V8 optimizer evidence requires its warm-start mode")
    if trainer.state.global_step != 0:
        raise RuntimeError("Scene V8 Trainer did not initialize at global step 0")
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise RuntimeError(
            "Scene V8 Trainer imported optimizer or scheduler state before creation"
        )
    trainer.create_optimizer()
    if not isinstance(trainer.optimizer, torch.optim.AdamW):
        raise RuntimeError("Scene V8 requires a freshly created torch AdamW")
    if trainer.optimizer.state:
        raise RuntimeError("Scene V8 fresh AdamW unexpectedly contains state")
    warm_start_context.manifest.update(
        {
            "pre_train_global_step": 0,
            "fresh_optimizer_created": True,
            "fresh_optimizer_class": (
                f"{trainer.optimizer.__class__.__module__}."
                f"{trainer.optimizer.__class__.__qualname__}"
            ),
            "fresh_optimizer_state_entries_before_train": 0,
            "fresh_scheduler_created_before_train": False,
        }
    )
    unsigned_warm_start_receipt = dict(warm_start_context.manifest)
    unsigned_warm_start_receipt.pop("receipt_sha256", None)
    warm_start_context.manifest["receipt_sha256"] = _canonical_json_sha256(
        unsigned_warm_start_receipt
    )
    trainer.continuation_manifest = dict(warm_start_context.manifest)


def _training_lineage_summary(trainer) -> dict[str, object]:
    active = getattr(trainer, "continuation_manifest", None)
    snapshot = None if active is None else dict(active)
    return {
        "continuation": snapshot,
        "resume_lineage": snapshot,
    }


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


class FixedIndexSampler(Sampler[int]):
    def __init__(self, dataset, indices: tuple[int, ...]) -> None:
        dataset_length = len(dataset)
        if not indices:
            raise ValueError("Fixed train schedule must contain at least one index")
        invalid = [
            index
            for index in indices
            if isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < dataset_length
        ]
        if invalid:
            raise ValueError(
                "Fixed train schedule contains out-of-range dataset indices: "
                f"{invalid[:8]}"
            )
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


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
        memory_representation_weight: float = 0.0,
        memory_representation_margin: float = 0.1,
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
        scene_boundary_payload_ce_weight: float = 0.0,
        train_sampler_seed: int | None = None,
        train_schedule_indices: tuple[int, ...] | None = None,
        train_schedule_binding: dict[str, object] | None = None,
        scene_state_generated_unlikelihood_weight: float = 0.0,
        scene_state_generated_unlikelihood_max_wrong_tokens: int = (
            _SCENE_STATE_GENERATED_UNLIKELIHOOD_MAX_WRONG_TOKENS
        ),
        scene_state_generated_rollout_extra_tokens: int = (
            _SCENE_STATE_GENERATED_ROLLOUT_EXTRA_TOKENS
        ),
        scene_state_generated_rollout_max_tokens: int = (
            _SCENE_STATE_GENERATED_ROLLOUT_MAX_TOKENS
        ),
        episode_read_write_enabled: bool = False,
        context_ablation_mode: str = "mixed",
        context_ablation_no_state_prob: float = 0.2,
        context_ablation_state_only_prob: float = 0.2,
        training_protocol: dict[str, object] | None = None,
        content_contrast_pairing_manifest: dict[str, object] | None = None,
        scene_state_identity_margin: float = 0.5,
        scene_state_identity_pairing_manifest: dict[str, object] | None = None,
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
        if (
            not math.isfinite(memory_representation_weight)
            or memory_representation_weight < 0.0
        ):
            raise ValueError("memory_representation_weight must be finite and non-negative")
        if (
            not math.isfinite(memory_representation_margin)
            or memory_representation_margin <= 0.0
        ):
            raise ValueError("memory_representation_margin must be finite and positive")
        if memory_representation_weight > 0.0 and memory_loss_mode != "content_contrast_ce":
            raise ValueError(
                "memory_representation_weight requires memory_loss_mode=content_contrast_ce"
            )
        if memory_representation_weight > 0.0 and delta_config is not None:
            if "o" not in delta_config.delta_heads:
                raise ValueError(
                    "memory_representation_weight requires an active delta_o head"
                )
            if (
                delta_config.memory_fusion_placement
                not in _REPRESENTATION_CAPTURE_FUSION_PLACEMENTS
            ):
                raise ValueError(
                    "memory_representation_weight supports only attention_output or "
                    "post_attention_residual_hybrid fusion"
                )
        self.memory_representation_weight = memory_representation_weight
        self.memory_representation_margin = memory_representation_margin
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
        if (
            not math.isfinite(scene_boundary_payload_ce_weight)
            or scene_boundary_payload_ce_weight < 0.0
        ):
            raise ValueError(
                "scene_boundary_payload_ce_weight must be finite and non-negative"
            )
        if scene_boundary_payload_ce_weight > 0.0 and memory_loss_mode != "context_dropout_ce":
            raise ValueError(
                "scene_boundary_payload_ce_weight requires "
                "memory_loss_mode=context_dropout_ce"
            )
        self.scene_boundary_payload_ce_weight = scene_boundary_payload_ce_weight
        if train_sampler_seed is not None:
            if isinstance(train_sampler_seed, bool) or not isinstance(
                train_sampler_seed,
                int,
            ):
                raise ValueError("train_sampler_seed must be an integer or None")
            if not 0 <= train_sampler_seed <= torch.iinfo(torch.int64).max:
                raise ValueError("train_sampler_seed must satisfy 0 <= seed <= 2^63 - 1")
        self.train_sampler_seed = train_sampler_seed
        self.train_schedule_indices = (
            None
            if train_schedule_indices is None
            else tuple(train_schedule_indices)
        )
        self.train_schedule_binding = (
            None
            if train_schedule_binding is None
            else dict(train_schedule_binding)
        )
        if self.train_schedule_indices is not None:
            if train_sampler_seed is not None:
                raise ValueError(
                    "Fixed train schedule is mutually exclusive with train_sampler_seed"
                )
            if self.train_schedule_binding is None:
                raise ValueError("Fixed train schedule requires its audited binding")
            if self.train_dataset is None:
                raise ValueError("Fixed train schedule requires a train dataset")
            FixedIndexSampler(self.train_dataset, self.train_schedule_indices)
            if self.train_schedule_binding.get("total_steps") != len(
                self.train_schedule_indices
            ):
                raise ValueError(
                    "Fixed train schedule length differs from its audited binding"
                )
        elif self.train_schedule_binding is not None:
            raise ValueError("Train schedule binding requires fixed schedule indices")
        if (
            train_sampler_seed is not None
            and getattr(self.args, "data_seed", None) != train_sampler_seed
        ):
            raise ValueError("train_sampler_seed must equal TrainingArguments.data_seed")
        if (
            not math.isfinite(scene_state_generated_unlikelihood_weight)
            or scene_state_generated_unlikelihood_weight < 0.0
        ):
            raise ValueError(
                "scene_state_generated_unlikelihood_weight must be finite and "
                "non-negative"
            )
        if scene_state_generated_unlikelihood_max_wrong_tokens <= 0:
            raise ValueError(
                "scene_state_generated_unlikelihood_max_wrong_tokens must be positive"
            )
        if scene_state_generated_rollout_extra_tokens < 0:
            raise ValueError(
                "scene_state_generated_rollout_extra_tokens must be non-negative"
            )
        if scene_state_generated_rollout_max_tokens <= 0:
            raise ValueError("scene_state_generated_rollout_max_tokens must be positive")
        if (
            scene_state_generated_unlikelihood_weight > 0.0
            and memory_loss_mode != "scene_state_generation_ce"
        ):
            raise ValueError(
                "scene_state_generated_unlikelihood_weight requires "
                "memory_loss_mode=scene_state_generation_ce"
            )
        self.scene_state_generated_unlikelihood_weight = (
            scene_state_generated_unlikelihood_weight
        )
        self.scene_state_generated_unlikelihood_max_wrong_tokens = int(
            scene_state_generated_unlikelihood_max_wrong_tokens
        )
        self.scene_state_generated_rollout_extra_tokens = int(
            scene_state_generated_rollout_extra_tokens
        )
        self.scene_state_generated_rollout_max_tokens = int(
            scene_state_generated_rollout_max_tokens
        )
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
        if memory_loss_mode == "scene_state_identity_ce":
            if episode_read_write_enabled:
                raise ValueError(
                    "scene_state_identity_ce requires episode read writes to be disabled"
                )
            if memory_kl_weight != 0.0 or memory_base_kl_weight != 0.0:
                raise ValueError(
                    "scene_state_identity_ce requires all KL weights to be zero"
                )
            if memory_representation_weight != 0.0:
                raise ValueError(
                    "scene_state_identity_ce requires representation loss to be disabled"
                )
            if not math.isfinite(scene_state_identity_margin) or (
                scene_state_identity_margin <= 0.0
            ):
                raise ValueError(
                    "scene_state_identity_ce requires a finite positive identity margin"
                )
            if write_sparsity_weight != 0.0:
                raise ValueError(
                    "scene_state_identity_ce requires write sparsity loss to be disabled"
                )
            if (
                memory_partition_alignment_weight != 0.0
                or memory_partition_entropy_weight != 0.0
                or memory_partition_balance_weight != 0.0
            ):
                raise ValueError(
                    "scene_state_identity_ce requires memory partition regularization "
                    "to be disabled"
                )
            if scene_boundary_payload_ce_weight != 0.0:
                raise ValueError(
                    "scene_state_identity_ce requires scene-boundary payload CE to be disabled"
                )
        if memory_loss_mode == "scene_state_generation_ce":
            if episode_read_write_enabled:
                raise ValueError(
                    "scene_state_generation_ce requires episode read writes to be disabled"
                )
            if memory_kl_weight != 0.0 or memory_base_kl_weight != 0.0:
                raise ValueError(
                    "scene_state_generation_ce requires all KL weights to be zero"
                )
            if memory_representation_weight != 0.0:
                raise ValueError(
                    "scene_state_generation_ce requires representation loss to be disabled"
                )
            if write_sparsity_weight != 0.0:
                raise ValueError(
                    "scene_state_generation_ce requires write sparsity loss to be disabled"
                )
            if (
                memory_partition_alignment_weight != 0.0
                or memory_partition_entropy_weight != 0.0
                or memory_partition_balance_weight != 0.0
            ):
                raise ValueError(
                    "scene_state_generation_ce requires memory partition regularization "
                    "to be disabled"
                )
            if scene_boundary_payload_ce_weight != 0.0:
                raise ValueError(
                    "scene_state_generation_ce requires scene-boundary payload CE to be disabled"
                )
        self.scene_state_identity_margin = scene_state_identity_margin
        self.context_ablation_mode = context_ablation_mode
        self.context_ablation_no_state_prob = context_ablation_no_state_prob
        self.context_ablation_state_only_prob = context_ablation_state_only_prob
        self.training_protocol = None if training_protocol is None else dict(training_protocol)
        self.content_contrast_pairing_manifest = (
            None
            if content_contrast_pairing_manifest is None
            else dict(content_contrast_pairing_manifest)
        )
        self.scene_state_identity_pairing_manifest = (
            None
            if scene_state_identity_pairing_manifest is None
            else dict(scene_state_identity_pairing_manifest)
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
        self._last_memory_representation_loss = 0.0
        self._last_memory_representation_distance = 0.0
        self._last_content_contrast_full_correct_ce = 0.0
        self._last_content_contrast_full_donor_ce = 0.0
        self._last_content_contrast_targeted_correct_ce = 0.0
        self._last_content_contrast_targeted_donor_ce = 0.0
        self._last_content_contrast_targeted_gap = 0.0
        self._last_content_contrast_targeted_positive_fraction = 0.0
        self._last_content_contrast_targeted_token_count = 0.0
        self._last_scene_state_full_correct_ce = 0.0
        self._last_scene_state_correct_all_semantic_ce = 0.0
        self._last_scene_state_correct_pair_semantic_ce = 0.0
        self._last_scene_state_donor_pair_semantic_ce = 0.0
        self._last_scene_state_zero_all_semantic_ce = 0.0
        self._last_scene_state_donor_pair_gap = 0.0
        self._last_scene_state_zero_all_gap = 0.0
        self._last_scene_state_donor_margin_loss = 0.0
        self._last_scene_state_donor_positive_fraction = 0.0
        self._last_scene_state_zero_positive_fraction = 0.0
        self._last_scene_state_semantic_token_count = 0.0
        self._last_scene_state_semantic_row_count = 0.0
        self._last_scene_state_target_presence_row_count = 0.0
        self._last_scene_state_target_same_cardinality_value_row_count = 0.0
        self._last_scene_state_target_cross_cardinality_value_row_count = 0.0
        self._last_scene_generation_total_loss = 0.0
        self._last_scene_generation_weighted_ce = 0.0
        self._last_scene_generation_schema_ce = 0.0
        self._last_scene_generation_decision_ce = 0.0
        self._last_scene_generation_termination_ce = 0.0
        self._last_scene_generation_first_error_loss = 0.0
        self._last_scene_generation_pair_correct_ce = 0.0
        self._last_scene_generation_pair_donor_ce = 0.0
        self._last_scene_generation_zero_margin_loss = 0.0
        self._last_scene_generation_gold_top1_accuracy = 0.0
        self._last_scene_generation_first_error_ordinal = 0.0
        self._last_scene_generation_solved_fraction = 0.0
        self._last_scene_generation_correct_decision_margin = 0.0
        self._last_scene_generation_donor_decision_margin = 0.0
        self._last_scene_generation_zero_decision_margin = 0.0
        self._last_scene_generation_correct_pair_preference = 0.0
        self._last_scene_generation_donor_pair_preference = 0.0
        self._last_scene_generation_target_token_count = 0.0
        self._last_scene_generation_content_token_count = 0.0
        self._last_scene_generation_schema_token_count = 0.0
        self._last_scene_generation_decision_token_count = 0.0
        self._last_scene_generation_termination_token_count = 0.0
        self._last_scene_generation_generated_unlikelihood_loss = 0.0
        self._last_scene_generation_generated_unlikelihood_weighted_loss = 0.0
        self._last_scene_generation_generated_unlikelihood_applied = 0.0
        self._last_scene_generation_generated_wrong_token_count = 0.0
        self._last_scene_generation_generated_rollout_token_count = 0.0
        self._last_scene_generation_generated_first_divergence = 0.0
        self._last_scene_generation_generated_exact_fraction = 0.0
        self._last_memory_teacher_loss = 0.0
        self._last_scene_boundary_full_ce_loss = 0.0
        self._last_scene_boundary_payload_ce_loss = 0.0
        self._last_scene_boundary_payload_auxiliary_loss = 0.0
        self._last_scene_boundary_payload_token_count = 0.0
        self._last_scene_boundary_supervised_token_count = 0.0
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

    def _get_train_sampler(self, train_dataset=None):
        active_dataset = self.train_dataset if train_dataset is None else train_dataset
        train_schedule_indices = getattr(self, "train_schedule_indices", None)
        if train_schedule_indices is not None:
            return FixedIndexSampler(active_dataset, train_schedule_indices)
        default_sampler = super()._get_train_sampler(train_dataset)
        if self.train_sampler_seed is None or default_sampler is None:
            return default_sampler
        if not isinstance(default_sampler, RandomSampler):
            raise ValueError(
                "train_sampler_seed requires Transformers random train sampling"
            )
        generator = torch.Generator()
        generator.manual_seed(self.train_sampler_seed)
        return RandomSampler(active_dataset, generator=generator)

    def _maybe_enable_static_graph(self, model) -> None:
        if self._ddp_static_graph_initialized:
            return
        # The active trainer only keeps delta readout, so the legacy stacked-read
        # branches no longer need special DDP static-graph handling.
        self._ddp_static_graph_initialized = True

    def _reset_online_state(self, model) -> None:
        reset_delta_mem_states(model)
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_read_representation_capture_mask(model, None)
        set_delta_mem_write_message_ids(model, None)
        set_delta_mem_write_sentence_ids(model, None)
        set_delta_mem_write_enabled(model, True)

    def _build_read_context_mask(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        labels = model_inputs.get("labels")
        attention_mask = model_inputs.get("attention_mask")
        if labels is None or attention_mask is None:
            return None
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError("Read-context labels and attention mask must be matching 2D tensors")

        valid_tokens = attention_mask.ne(0)
        read_context_mask = labels.eq(-100) & valid_tokens
        # A causal LM predicts label[t + 1] from the representation at t. Keep
        # memory active at every supervised predictor, including answer tokens
        # that provide causal context for the following answer token.
        read_context_mask[:, :-1] |= (
            labels[:, 1:].ne(-100)
            & valid_tokens[:, :-1]
            & valid_tokens[:, 1:]
        )
        return read_context_mask

    def _build_read_representation_capture_mask(
        self,
        model_inputs: dict[str, torch.Tensor],
        read_context_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = model_inputs.get("labels")
        attention_mask = model_inputs.get("attention_mask")
        if labels is None or attention_mask is None:
            raise ValueError(
                "Read-representation capture requires labels and attention_mask"
            )
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError(
                "Read-representation labels and attention mask must be matching 2D tensors"
            )
        supervised_labels = labels.ne(-100) & attention_mask.ne(0)
        if target_mask is None:
            supervised_targets = supervised_labels[:, 1:]
            missing_target_description = "supervised target"
        else:
            if target_mask.shape != labels.shape:
                raise ValueError(
                    "Read-representation target mask must match labels"
                )
            target_mask = target_mask.to(device=labels.device, dtype=torch.bool)
            if bool(target_mask[:, 0].any()) or bool(
                (target_mask & ~supervised_labels).any()
            ):
                raise ValueError(
                    "Read-representation target mask must select causally predictable "
                    "supervised labels"
                )
            supervised_targets = target_mask[:, 1:]
            missing_target_description = "selected target"
        has_supervised_target = supervised_targets.any(dim=1)
        if not bool(has_supervised_target.all()):
            missing_rows = (~has_supervised_target).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Read-representation capture requires a causally predictable "
                f"{missing_target_description} in every row; "
                f"missing rows: {missing_rows}"
            )
        predictor_positions = supervised_targets.to(dtype=torch.int64).argmax(dim=1)
        capture_mask = torch.zeros_like(labels, dtype=torch.bool)
        capture_mask.scatter_(1, predictor_positions.unsqueeze(1), True)
        if read_context_mask is None or read_context_mask.shape != capture_mask.shape:
            raise ValueError(
                "Read-representation capture requires a matching read-context mask"
            )
        if not bool(read_context_mask.to(dtype=torch.bool).masked_select(capture_mask).all()):
            raise ValueError(
                "Read-representation capture predictor must be covered by the read-context mask"
            )
        return capture_mask

    @staticmethod
    def _stack_read_representations(
        representations: dict[str, torch.Tensor],
    ) -> tuple[tuple[str, ...], torch.Tensor]:
        if not representations:
            raise RuntimeError("Read-representation capture returned no Delta-Mem modules")
        module_names = tuple(sorted(representations))
        tensors = []
        expected_shape = None
        for module_name in module_names:
            representation = representations[module_name]
            if not isinstance(representation, torch.Tensor) or representation.ndim != 2:
                raise RuntimeError(
                    "Read-representation capture must return [batch, hidden] tensors; "
                    f"invalid module: {module_name}"
                )
            if expected_shape is None:
                expected_shape = representation.shape
            elif representation.shape != expected_shape:
                raise RuntimeError(
                    "Read-representation capture tensors must have identical shapes; "
                    f"module {module_name} has {tuple(representation.shape)}, expected "
                    f"{tuple(expected_shape)}"
                )
            tensors.append(representation)
        return module_names, torch.stack(tensors, dim=0)

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

    def _scene_boundary_payload_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        payload_mask: torch.Tensor,
        *,
        full_token_normalizer: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError(
                "Scene-boundary payload labels and attention mask must be matching 2D tensors"
            )
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError("Scene-boundary payload logits must align with labels")
        if payload_mask.shape != labels.shape:
            raise ValueError("Scene-boundary payload mask must align with labels")
        payload_mask = payload_mask.to(device=labels.device, dtype=torch.bool)
        supervised_mask = labels.ne(-100) & attention_mask.ne(0)
        if bool((payload_mask & ~supervised_mask).any().item()):
            raise ValueError(
                "Scene-boundary payload mask may select only supervised, non-padding labels"
            )
        if bool(payload_mask[:, 0].any().item()):
            raise ValueError(
                "Scene-boundary payload mask selects a token without a causal predictor"
            )
        selected_next_tokens = payload_mask[:, 1:]
        if not bool(selected_next_tokens.any(dim=1).all().item()):
            raise ValueError(
                "Every scene-boundary row must select at least one payload target token"
            )
        if bool(
            (selected_next_tokens & attention_mask[:, :-1].eq(0)).any().item()
        ):
            raise ValueError(
                "Scene-boundary payload mask selects a target with a masked causal predictor"
            )
        full_next_token_mask = supervised_mask[:, 1:] & attention_mask[:, :-1].ne(0)
        supervised_token_count = int(full_next_token_mask.sum().item())
        if supervised_token_count <= 0:
            raise ValueError("Scene-boundary batch has no supervised next-token targets")
        payload_token_count = int(selected_next_tokens.sum().item())
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        selected_sum = token_losses.masked_select(selected_next_tokens).sum()
        payload_mean = selected_sum / payload_token_count
        if full_token_normalizer is None:
            normalizer = selected_sum.new_tensor(float(supervised_token_count))
        elif isinstance(full_token_normalizer, torch.Tensor):
            normalizer = full_token_normalizer.to(
                device=selected_sum.device,
                dtype=selected_sum.dtype,
            )
        else:
            normalizer = selected_sum.new_tensor(float(full_token_normalizer))
        if normalizer.numel() != 1 or not bool(torch.isfinite(normalizer).item()):
            raise ValueError("Scene-boundary full-token normalizer must be one finite scalar")
        if float(normalizer.item()) <= 0.0:
            raise ValueError("Scene-boundary full-token normalizer must be positive")
        payload_auxiliary = selected_sum / normalizer
        return (
            payload_mean,
            payload_auxiliary,
            payload_token_count,
            supervised_token_count,
        )

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
        scene_boundary_payload_mask: torch.Tensor | None,
        state_only_scene_boundary_payload_mask: torch.Tensor | None,
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
            active_payload_mask = state_only_scene_boundary_payload_mask
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
            active_payload_mask = scene_boundary_payload_mask
            prime_kwargs = None
            wmem = 0.0
        else:
            active_inputs = model_inputs
            active_payload_mask = scene_boundary_payload_mask
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
        payload_ce_loss = loss.new_zeros(())
        payload_auxiliary_loss = loss.new_zeros(())
        payload_token_count = 0
        supervised_token_count = 0
        payload_ce_weight = getattr(self, "scene_boundary_payload_ce_weight", 0.0)
        if payload_ce_weight > 0.0:
            if active_payload_mask is None:
                raise ValueError(
                    "scene-boundary payload CE requires tokenizer-derived payload masks"
                )
            (
                payload_ce_loss,
                payload_auxiliary_loss,
                payload_token_count,
                supervised_token_count,
            ) = self._scene_boundary_payload_ce(
                outputs["logits"],
                active_inputs["labels"],
                active_inputs["attention_mask"],
                active_payload_mask,
                full_token_normalizer=loss_kwargs.get("num_items_in_batch"),
            )
            loss = loss + payload_ce_weight * payload_auxiliary_loss
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
        outputs["scene_boundary_full_ce_loss"] = task_loss.detach()
        outputs["scene_boundary_payload_ce_loss"] = payload_ce_loss.detach()
        outputs["scene_boundary_payload_auxiliary_loss"] = (
            payload_auxiliary_loss.detach()
        )
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
            "scene_boundary_full_ce_loss": float(task_loss.detach().float().item()),
            "scene_boundary_payload_ce_loss": float(
                payload_ce_loss.detach().float().item()
            ),
            "scene_boundary_payload_auxiliary_loss": float(
                payload_auxiliary_loss.detach().float().item()
            ),
            "scene_boundary_payload_token_count": float(payload_token_count),
            "scene_boundary_supervised_token_count": float(supervised_token_count),
        }

    def _content_contrast_target_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError(
                "Content-contrast labels and attention mask must be matching 2D tensors"
            )
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError("Content-contrast logits must align with labels")
        if target_mask.shape != labels.shape:
            raise ValueError("Content-contrast target mask must align with labels")
        target_mask = target_mask.to(device=labels.device, dtype=torch.bool)
        supervised_labels = labels.ne(-100) & attention_mask.ne(0)
        if bool(target_mask[:, 0].any()) or bool(
            (target_mask & ~supervised_labels).any()
        ):
            raise ValueError(
                "Content-contrast target mask must select causally predictable "
                "supervised labels"
            )
        shift_mask = target_mask[:, 1:]
        target_counts = shift_mask.sum(dim=1)
        if not bool(
            target_counts.eq(_CONTENT_CONTRAST_TARGET_SPAN_TOKENS).all()
        ):
            raise ValueError(
                "Content-contrast target mask must select exactly "
                f"{_CONTENT_CONTRAST_TARGET_SPAN_TOKENS} targets in every row"
            )
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        row_ce = (token_ce * shift_mask).sum(dim=1) / target_counts
        return row_ce.mean(), row_ce, int(target_counts.sum().item())

    def _content_contrast_objective(
        self,
        correct_full_loss: torch.Tensor,
        correct_target_loss: torch.Tensor,
        wrong_target_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if wrong_target_loss is None:
            wrong_target_loss = correct_target_loss
            correct_target_loss = correct_full_loss
        gap = wrong_target_loss - correct_target_loss
        contrast_loss = self._margin_objective(gap, self.memory_margin)
        total_loss = (
            correct_full_loss + self.memory_contrast_weight * contrast_loss
        )
        return total_loss, contrast_loss, gap

    def _content_contrast_representation_enabled(self) -> bool:
        return getattr(self, "memory_representation_weight", 0.0) > 0.0

    def _content_contrast_representation_objective(
        self,
        correct_representations: torch.Tensor,
        wrong_representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            correct_representations.ndim != 3
            or wrong_representations.shape != correct_representations.shape
        ):
            raise ValueError(
                "Content-contrast representations must be matching [layer, batch, hidden] "
                "tensors"
            )
        correct = correct_representations.float()
        wrong = wrong_representations.float()
        difference_norm = torch.linalg.vector_norm(correct - wrong, dim=-1)
        mean_norm = (
            torch.linalg.vector_norm(correct, dim=-1)
            + torch.linalg.vector_norm(wrong, dim=-1)
        ) / 2.0
        relative_distance = difference_norm / mean_norm.clamp_min(
            _CONTENT_CONTRAST_REPRESENTATION_EPS
        )
        margin = getattr(self, "memory_representation_margin", 0.1)
        representation_loss = F.softplus((margin - relative_distance) / margin).mean()
        return representation_loss, relative_distance

    def _validate_content_contrast_runtime(self) -> None:
        if self.episode_read_write_enabled:
            raise ValueError("content_contrast_ce requires episode read writes to be disabled")
        representation_weight = getattr(self, "memory_representation_weight", 0.0)
        representation_margin = getattr(self, "memory_representation_margin", 0.1)
        if not math.isfinite(representation_weight) or representation_weight < 0.0:
            raise ValueError(
                "content_contrast_ce requires a finite non-negative representation weight"
            )
        if not math.isfinite(representation_margin) or representation_margin <= 0.0:
            raise ValueError(
                "content_contrast_ce requires a finite positive representation margin"
            )
        delta_config = getattr(self, "delta_config", None)
        if representation_weight > 0.0 and delta_config is not None:
            if "o" not in delta_config.delta_heads:
                raise ValueError(
                    "content_contrast representation capture requires an active delta_o head"
                )
            if (
                delta_config.memory_fusion_placement
                not in _REPRESENTATION_CAPTURE_FUSION_PLACEMENTS
            ):
                raise ValueError(
                    "content_contrast representation capture supports only "
                    "attention_output or post_attention_residual_hybrid fusion"
                )
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

    def _validate_content_contrast_sequential_runtime(self) -> None:
        self._validate_content_contrast_runtime()
        accumulation_steps = int(
            getattr(self, "current_gradient_accumulation_steps", 1)
        )
        if accumulation_steps != 1:
            raise ValueError(
                "content_contrast_ce sequential backward requires "
                "gradient_accumulation_steps=1"
            )
        optimizer_name = str(
            getattr(getattr(self, "args", None), "optim", "")
        ).lower()
        if optimizer_name.endswith("lomo") or optimizer_name.endswith("adalomo"):
            raise ValueError(
                "content_contrast_ce sequential backward does not support LOMO or AdaLOMO"
            )
        distributed_type = getattr(
            getattr(self, "accelerator", None),
            "distributed_type",
            None,
        )
        if getattr(distributed_type, "name", None) == "DEEPSPEED":
            raise ValueError(
                "content_contrast_ce sequential backward does not support DeepSpeed"
            )

    def _content_contrast_branch(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        capture_representations: bool,
        content_contrast_target_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, torch.Tensor],
        tuple[str, ...] | None,
        torch.Tensor | None,
    ]:
        batch_size = int(model_inputs["input_ids"].size(0))
        read_context_mask = self._build_read_context_mask(model_inputs)
        capture_mask = (
            self._build_read_representation_capture_mask(
                model_inputs,
                read_context_mask,
                content_contrast_target_mask,
            )
            if capture_representations
            else None
        )
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
        set_delta_mem_read_representation_capture_mask(model, capture_mask)
        try:
            outputs = model(**model_inputs, **loss_kwargs)
            if capture_representations:
                representation_names, representations = self._stack_read_representations(
                    collect_delta_mem_read_representations(model)
                )
                if representations.size(1) != batch_size:
                    raise RuntimeError(
                        "Read-representation capture batch size does not match the read batch"
                    )
            else:
                representation_names = None
                representations = None
        finally:
            set_delta_mem_read_representation_capture_mask(model, None)
        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        full_loss = outputs["loss"]
        if full_loss.ndim > 0:
            full_loss = full_loss.mean()
        if content_contrast_target_mask is None:
            target_loss = full_loss
            target_row_losses = full_loss.detach().expand(batch_size)
            target_token_count = int(
                (
                    model_inputs["labels"][:, 1:].ne(-100)
                    & model_inputs["attention_mask"][:, 1:].ne(0)
                ).sum().item()
            )
        else:
            target_loss, target_row_losses, target_token_count = (
                self._content_contrast_target_ce(
                    outputs["logits"],
                    model_inputs["labels"],
                    model_inputs["attention_mask"],
                    content_contrast_target_mask,
                )
            )
        return (
            full_loss,
            target_loss,
            target_row_losses,
            target_token_count,
            outputs,
            representation_names,
            representations,
        )

    def _content_contrast_loss_and_coefficients(
        self,
        correct_full_loss: torch.Tensor,
        correct_target_loss: torch.Tensor,
        wrong_target_loss: torch.Tensor,
        correct_representations: torch.Tensor | None,
        wrong_representations: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        correct_full_probe = correct_full_loss.detach().float().requires_grad_(True)
        correct_target_probe = (
            correct_target_loss.detach().float().requires_grad_(True)
        )
        wrong_target_probe = wrong_target_loss.detach().float().requires_grad_(True)
        total_loss, contrast_loss, gap = self._content_contrast_objective(
            correct_full_probe,
            correct_target_probe,
            wrong_target_probe,
        )
        representation_loss = total_loss.new_zeros(())
        representation_distance = total_loss.new_zeros(())
        gradient_inputs: list[torch.Tensor] = [
            correct_full_probe,
            correct_target_probe,
            wrong_target_probe,
        ]
        if self._content_contrast_representation_enabled():
            if correct_representations is None or wrong_representations is None:
                raise RuntimeError(
                    "Content-contrast representation objective requires both branch captures"
                )
            correct_representation_probe = (
                correct_representations.detach().float().requires_grad_(True)
            )
            wrong_representation_probe = (
                wrong_representations.detach().float().requires_grad_(True)
            )
            representation_loss, relative_distance = (
                self._content_contrast_representation_objective(
                    correct_representation_probe,
                    wrong_representation_probe,
                )
            )
            representation_distance = relative_distance.mean()
            total_loss = (
                total_loss
                + self.memory_representation_weight * representation_loss
            )
            gradient_inputs.extend(
                (correct_representation_probe, wrong_representation_probe)
            )
        gradients = torch.autograd.grad(total_loss, tuple(gradient_inputs))
        (
            correct_full_coefficient,
            correct_target_coefficient,
            wrong_target_coefficient,
        ) = gradients[:3]
        if len(gradients) == 5:
            correct_representation_gradient, wrong_representation_gradient = gradients[3:]
        else:
            correct_representation_gradient = None
            wrong_representation_gradient = None
        return (
            total_loss.detach(),
            contrast_loss.detach(),
            gap.detach(),
            representation_loss.detach(),
            representation_distance.detach(),
            correct_full_coefficient.detach(),
            correct_target_coefficient.detach(),
            wrong_target_coefficient.detach(),
            None
            if correct_representation_gradient is None
            else correct_representation_gradient.detach(),
            None
            if wrong_representation_gradient is None
            else wrong_representation_gradient.detach(),
        )

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
        content_contrast_target_mask: torch.Tensor | None = None,
    ):
        self._validate_content_contrast_runtime()
        capture_representations = self._content_contrast_representation_enabled()
        (
            correct_full_loss,
            correct_target_loss,
            correct_target_row_losses,
            target_token_count,
            correct_outputs,
            correct_representation_names,
            correct_representations,
        ) = self._content_contrast_branch(
            model,
            model_inputs,
            loss_kwargs=loss_kwargs,
            write_input_ids=write_input_ids,
            write_attention_mask=write_attention_mask,
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
            capture_representations=capture_representations,
            content_contrast_target_mask=content_contrast_target_mask,
        )
        (
            wrong_full_loss,
            wrong_target_loss,
            wrong_target_row_losses,
            wrong_target_token_count,
            _,
            wrong_representation_names,
            wrong_representations,
        ) = self._content_contrast_branch(
            model,
            model_inputs,
            loss_kwargs=loss_kwargs,
            write_input_ids=negative_write_input_ids,
            write_attention_mask=negative_write_attention_mask,
            write_message_ids=negative_write_message_ids,
            write_sentence_ids=negative_write_sentence_ids,
            capture_representations=capture_representations,
            content_contrast_target_mask=content_contrast_target_mask,
        )
        if wrong_target_token_count != target_token_count:
            raise RuntimeError(
                "Content-contrast branches selected different target token counts"
            )

        total_loss, contrast_loss, margin_gap = self._content_contrast_objective(
            correct_full_loss,
            correct_target_loss,
            wrong_target_loss,
        )
        representation_loss = total_loss.new_zeros(())
        representation_distance = total_loss.new_zeros(())
        if capture_representations:
            if correct_representation_names != wrong_representation_names:
                raise RuntimeError(
                    "Content-contrast branches captured different Delta-Mem modules"
                )
            assert correct_representations is not None
            assert wrong_representations is not None
            representation_loss, relative_distance = (
                self._content_contrast_representation_objective(
                    correct_representations,
                    wrong_representations,
                )
            )
            representation_distance = relative_distance.mean()
            total_loss = (
                total_loss
                + self.memory_representation_weight * representation_loss
            )
        outputs = dict(correct_outputs)
        outputs["loss"] = total_loss
        outputs["memory_loss"] = (total_loss - correct_full_loss).detach()
        outputs["memory_keep_loss"] = correct_full_loss.detach()
        outputs["content_contrast_correct_target_ce"] = correct_target_loss.detach()
        outputs["content_contrast_donor_target_ce"] = wrong_target_loss.detach()
        outputs["memory_representation_loss"] = representation_loss.detach()
        outputs["memory_representation_distance"] = representation_distance.detach()
        target_positive_fraction = (
            wrong_target_row_losses.detach().float()
            > correct_target_row_losses.detach().float()
        ).float().mean()
        return total_loss, outputs, {
            "keep_loss": float(correct_full_loss.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": float(wrong_full_loss.detach().float().item()),
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": float(contrast_loss.detach().float().item()),
            "anchor_loss": 0.0,
            "full_ce_loss": float(correct_full_loss.detach().float().item()),
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": float(margin_gap.detach().float().item()),
            "representation_loss": float(
                representation_loss.detach().float().item()
            ),
            "representation_distance": float(
                representation_distance.detach().float().item()
            ),
            "full_correct_ce": float(correct_full_loss.detach().float().item()),
            "full_donor_ce": float(wrong_full_loss.detach().float().item()),
            "targeted_correct_ce": float(correct_target_loss.detach().float().item()),
            "targeted_donor_ce": float(wrong_target_loss.detach().float().item()),
            "targeted_gap": float(margin_gap.detach().float().item()),
            "targeted_positive_fraction": float(target_positive_fraction.item()),
            "targeted_token_count": float(target_token_count),
            "wmem": 1.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _content_contrast_sequential_backward(
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
        gradient_scale: float,
        content_contrast_target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backpropagate the exact first derivative with one live writer graph at a time."""

        self._validate_content_contrast_sequential_runtime()
        capture_representations = self._content_contrast_representation_enabled()
        wrong_probe_rng_state = _capture_torch_rng_state()
        with torch.no_grad(), self.compute_loss_context_manager():
            (
                wrong_full_probe,
                wrong_target_probe,
                wrong_target_row_probe,
                wrong_target_token_count,
                wrong_probe_outputs,
                wrong_probe_representation_names,
                wrong_probe_representations,
            ) = self._content_contrast_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=negative_write_input_ids,
                write_attention_mask=negative_write_attention_mask,
                write_message_ids=negative_write_message_ids,
                write_sentence_ids=negative_write_sentence_ids,
                capture_representations=capture_representations,
                content_contrast_target_mask=content_contrast_target_mask,
            )
        wrong_full_probe = wrong_full_probe.detach()
        wrong_target_probe = wrong_target_probe.detach()
        wrong_target_row_probe = wrong_target_row_probe.detach()
        if wrong_probe_representations is not None:
            wrong_probe_representations = wrong_probe_representations.detach()
        del wrong_probe_outputs

        with self.compute_loss_context_manager():
            (
                correct_full_loss,
                correct_target_loss,
                correct_target_row_losses,
                correct_target_token_count,
                correct_outputs,
                correct_representation_names,
                correct_representations,
            ) = self._content_contrast_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                capture_representations=capture_representations,
                content_contrast_target_mask=content_contrast_target_mask,
            )
        if correct_target_token_count != wrong_target_token_count:
            raise RuntimeError(
                "Content-contrast branches selected different target token counts"
            )
        if correct_representation_names != wrong_probe_representation_names:
            raise RuntimeError(
                "Content-contrast branches captured different Delta-Mem modules"
            )
        (
            total_loss,
            contrast_loss,
            margin_gap,
            representation_loss,
            representation_distance,
            correct_full_coefficient,
            correct_target_coefficient,
            wrong_target_coefficient,
            correct_representation_gradient,
            wrong_representation_gradient,
        ) = self._content_contrast_loss_and_coefficients(
            correct_full_loss,
            correct_target_loss,
            wrong_target_probe,
            correct_representations,
            wrong_probe_representations,
        )
        correct_full_value = correct_full_loss.detach()
        correct_target_value = correct_target_loss.detach()
        correct_target_row_values = correct_target_row_losses.detach()
        correct_backward = (
            correct_full_loss * correct_full_coefficient
            + correct_target_loss * correct_target_coefficient
        )
        if correct_representation_gradient is not None:
            assert correct_representations is not None
            correct_backward = correct_backward + torch.sum(
                correct_representations.float() * correct_representation_gradient
            )
        self.accelerator.backward(
            correct_backward * gradient_scale,
        )
        del (
            correct_backward,
            correct_full_loss,
            correct_target_loss,
            correct_outputs,
            correct_representations,
        )

        post_correct_rng_state = _capture_torch_rng_state()
        _restore_torch_rng_state(wrong_probe_rng_state)
        try:
            with self.compute_loss_context_manager():
                (
                    wrong_full_loss,
                    wrong_target_loss,
                    wrong_target_row_losses,
                    replay_target_token_count,
                    wrong_outputs,
                    wrong_representation_names,
                    wrong_representations,
                ) = self._content_contrast_branch(
                    model,
                    model_inputs,
                    loss_kwargs=loss_kwargs,
                    write_input_ids=negative_write_input_ids,
                    write_attention_mask=negative_write_attention_mask,
                    write_message_ids=negative_write_message_ids,
                    write_sentence_ids=negative_write_sentence_ids,
                    capture_representations=capture_representations,
                    content_contrast_target_mask=content_contrast_target_mask,
                )
        finally:
            _restore_torch_rng_state(post_correct_rng_state)
        if replay_target_token_count != wrong_target_token_count:
            raise RuntimeError(
                "Sequential content-contrast donor replay selected a different target "
                "token count"
            )
        if not torch.allclose(
            wrong_full_loss.detach(),
            wrong_full_probe,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Sequential content-contrast donor full-CE replay is not deterministic: "
                f"probe={float(wrong_full_probe.float().item()):.8f} "
                f"gradient={float(wrong_full_loss.detach().float().item()):.8f}"
            )
        if not torch.allclose(
            wrong_target_loss.detach(),
            wrong_target_probe,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Sequential content-contrast donor targeted-CE replay is not deterministic: "
                f"probe={float(wrong_target_probe.float().item()):.8f} "
                f"gradient={float(wrong_target_loss.detach().float().item()):.8f}"
            )
        if wrong_representation_names != wrong_probe_representation_names:
            raise RuntimeError(
                "Sequential content-contrast donor representation replay captured different "
                "Delta-Mem modules"
            )
        if wrong_probe_representations is not None:
            assert wrong_representations is not None
            if not torch.allclose(
                wrong_representations.detach(),
                wrong_probe_representations,
                rtol=1e-5,
                atol=1e-6,
            ):
                maximum_error = torch.max(
                    torch.abs(
                        wrong_representations.detach().float()
                        - wrong_probe_representations.float()
                    )
                )
                raise RuntimeError(
                    "Sequential content-contrast donor representation replay is not "
                    f"deterministic: max_abs_error={float(maximum_error.item()):.8f}"
                )
        wrong_full_value = wrong_full_loss.detach()
        wrong_target_value = wrong_target_loss.detach()
        wrong_target_row_values = wrong_target_row_losses.detach()
        wrong_backward = wrong_target_loss * wrong_target_coefficient
        if wrong_representation_gradient is not None:
            assert wrong_representations is not None
            wrong_backward = wrong_backward + torch.sum(
                wrong_representations.float() * wrong_representation_gradient
            )
        self.accelerator.backward(
            wrong_backward * gradient_scale,
        )
        del (
            wrong_backward,
            wrong_full_loss,
            wrong_target_loss,
            wrong_outputs,
            wrong_representations,
        )

        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return total_loss * gradient_scale, {
            "keep_loss": float(correct_full_value.float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": float(wrong_full_value.float().item()),
            "teacher_loss": 0.0,
            "margin_loss": 0.0,
            "causal_loss": float(contrast_loss.float().item()),
            "anchor_loss": 0.0,
            "full_ce_loss": float(correct_full_value.float().item()),
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": float(margin_gap.float().item()),
            "representation_loss": float(representation_loss.float().item()),
            "representation_distance": float(representation_distance.float().item()),
            "full_correct_ce": float(correct_full_value.float().item()),
            "full_donor_ce": float(wrong_full_value.float().item()),
            "targeted_correct_ce": float(correct_target_value.float().item()),
            "targeted_donor_ce": float(wrong_target_value.float().item()),
            "targeted_gap": float(margin_gap.float().item()),
            "targeted_positive_fraction": float(
                (
                    wrong_target_row_values.float()
                    > correct_target_row_values.float()
                ).float().mean().item()
            ),
            "targeted_token_count": float(correct_target_token_count),
            "wmem": 1.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
        }

    def _scene_state_semantic_ce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        semantic_target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return a batch mean of per-row semantic CE means."""

        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError(
                "Scene-state labels and attention mask must be matching 2D tensors"
            )
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError("Scene-state logits must align with labels")
        if semantic_target_mask.shape != labels.shape:
            raise ValueError("Scene-state semantic target mask must align with labels")
        semantic_target_mask = semantic_target_mask.to(
            device=labels.device,
            dtype=torch.bool,
        )
        supervised_labels = labels.ne(-100) & attention_mask.ne(0)
        if bool(semantic_target_mask[:, 0].any()) or bool(
            (semantic_target_mask & ~supervised_labels).any()
        ):
            raise ValueError(
                "Scene-state semantic target mask must select causally predictable "
                "supervised labels"
            )
        shift_mask = semantic_target_mask[:, 1:]
        target_counts = shift_mask.sum(dim=1)
        if not bool(target_counts.gt(0).all()):
            missing_rows = (
                target_counts.eq(0).nonzero(as_tuple=False).flatten().tolist()
            )
            raise ValueError(
                "Scene-state semantic target mask must select at least one target in "
                f"every row; missing rows: {missing_rows}"
            )
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        row_ce = (token_ce * shift_mask).sum(dim=1) / target_counts
        return row_ce.mean(), row_ce, int(target_counts.sum().item())

    def _scene_state_identity_objective(
        self,
        correct_full_ce: torch.Tensor,
        correct_all_semantic_row_ce: torch.Tensor,
        correct_pair_semantic_row_ce: torch.Tensor,
        donor_pair_semantic_row_ce: torch.Tensor,
        zero_all_semantic_row_ce: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if correct_full_ce.numel() != 1:
            raise ValueError("Scene-state correct full CE must be scalar")
        if correct_all_semantic_row_ce.ndim != 1:
            raise ValueError("Scene-state semantic CE must contain one value per row")
        if (
            correct_pair_semantic_row_ce.shape != correct_all_semantic_row_ce.shape
            or donor_pair_semantic_row_ce.shape
            != correct_all_semantic_row_ce.shape
            or zero_all_semantic_row_ce.shape != correct_all_semantic_row_ce.shape
        ):
            raise ValueError("Scene-state semantic branch row counts must match")
        if correct_all_semantic_row_ce.numel() == 0:
            raise ValueError("Scene-state identity objective requires at least one row")

        correct_all_semantic_ce = correct_all_semantic_row_ce.mean()
        correct_pair_semantic_ce = correct_pair_semantic_row_ce.mean()
        donor_pair_semantic_ce = donor_pair_semantic_row_ce.mean()
        zero_all_semantic_ce = zero_all_semantic_row_ce.mean()
        donor_gap_rows = (
            donor_pair_semantic_row_ce - correct_pair_semantic_row_ce
        )
        zero_gap_rows = zero_all_semantic_row_ce - correct_all_semantic_row_ce
        donor_gap = donor_gap_rows.mean()
        zero_gap = zero_gap_rows.mean()
        donor_margin_loss = F.relu(
            self.scene_state_identity_margin - donor_gap_rows
        ).mean()
        total_loss = (
            _SCENE_STATE_FULL_CORRECT_CE_WEIGHT * correct_full_ce
            + _SCENE_STATE_CORRECT_ALL_SEMANTIC_CE_WEIGHT
            * correct_all_semantic_ce
            + _SCENE_STATE_DONOR_MARGIN_WEIGHT * donor_margin_loss
        )
        return (
            total_loss,
            correct_all_semantic_ce,
            correct_pair_semantic_ce,
            donor_pair_semantic_ce,
            zero_all_semantic_ce,
            donor_gap,
            zero_gap,
            donor_margin_loss,
        )

    def _validate_scene_state_identity_runtime(self) -> None:
        if self.episode_read_write_enabled:
            raise ValueError(
                "scene_state_identity_ce requires episode read writes to be disabled"
            )
        if self.memory_kl_weight != 0.0 or self.memory_base_kl_weight != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires all KL weights to be zero"
            )
        if getattr(self, "memory_representation_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires representation loss to be disabled"
            )
        if not math.isfinite(self.scene_state_identity_margin) or (
            self.scene_state_identity_margin <= 0.0
        ):
            raise ValueError(
                "scene_state_identity_ce requires a finite positive identity margin"
            )
        if getattr(self, "write_sparsity_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires write sparsity loss to be disabled"
            )
        if (
            getattr(self, "memory_partition_alignment_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_entropy_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_balance_weight", 0.0) != 0.0
        ):
            raise ValueError(
                "scene_state_identity_ce requires memory partition regularization "
                "to be disabled"
            )
        if getattr(self, "scene_boundary_payload_ce_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires scene-boundary payload CE to be disabled"
            )

    def _validate_scene_state_identity_sequential_runtime(self) -> None:
        self._validate_scene_state_identity_runtime()
        accumulation_steps = int(
            getattr(self, "current_gradient_accumulation_steps", 1)
        )
        if accumulation_steps != 1:
            raise ValueError(
                "scene_state_identity_ce sequential backward requires "
                "gradient_accumulation_steps=1"
            )
        optimizer_name = str(
            getattr(getattr(self, "args", None), "optim", "")
        ).lower()
        if optimizer_name.endswith("lomo") or optimizer_name.endswith("adalomo"):
            raise ValueError(
                "scene_state_identity_ce sequential backward does not support "
                "LOMO or AdaLOMO"
            )
        distributed_type = getattr(
            getattr(self, "accelerator", None),
            "distributed_type",
            None,
        )
        if getattr(distributed_type, "name", None) == "DEEPSPEED":
            raise ValueError(
                "scene_state_identity_ce sequential backward does not support DeepSpeed"
            )

    def _scene_state_identity_branch(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        prime_writes: bool,
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        semantic_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        dict[str, torch.Tensor],
    ]:
        batch_size = int(model_inputs["input_ids"].size(0))
        self._reset_online_state(model)
        if prime_writes:
            if write_input_ids is None or write_attention_mask is None:
                raise ValueError(
                    "Scene-state correct and donor branches require materialized writes"
                )
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_read_context_mask(
            model,
            self._build_read_context_mask(model_inputs),
        )
        outputs = model(**model_inputs, **loss_kwargs)
        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        full_ce = outputs["loss"]
        if full_ce.ndim > 0:
            full_ce = full_ce.mean()
        all_semantic_ce, all_semantic_row_ce, all_semantic_token_count = (
            self._scene_state_semantic_ce(
                outputs["logits"],
                model_inputs["labels"],
                model_inputs["attention_mask"],
                semantic_mask,
            )
        )
        pair_semantic_ce, pair_semantic_row_ce, pair_semantic_token_count = (
            self._scene_state_semantic_ce(
                outputs["logits"],
                model_inputs["labels"],
                model_inputs["attention_mask"],
                pair_target_mask,
            )
        )
        return (
            full_ce,
            all_semantic_ce,
            all_semantic_row_ce,
            all_semantic_token_count,
            pair_semantic_ce,
            pair_semantic_row_ce,
            pair_semantic_token_count,
            outputs,
        )

    def _scene_state_identity_stats(
        self,
        *,
        correct_full_ce: torch.Tensor,
        correct_all_semantic_row_ce: torch.Tensor,
        correct_pair_semantic_row_ce: torch.Tensor,
        donor_pair_semantic_row_ce: torch.Tensor,
        zero_all_semantic_row_ce: torch.Tensor,
        semantic_token_count: int,
        target_stratum_codes: torch.Tensor,
        objective_values: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> dict[str, float]:
        (
            _,
            correct_all_semantic_ce,
            correct_pair_semantic_ce,
            donor_pair_semantic_ce,
            zero_all_semantic_ce,
            donor_gap,
            zero_gap,
            donor_margin_loss,
        ) = objective_values
        donor_gap_rows = donor_pair_semantic_row_ce.detach().float() - (
            correct_pair_semantic_row_ce.detach().float()
        )
        zero_gap_rows = zero_all_semantic_row_ce.detach().float() - (
            correct_all_semantic_row_ce.detach().float()
        )
        normalized_stratum_codes = target_stratum_codes.detach().to(
            dtype=torch.long
        )
        if normalized_stratum_codes.ndim != 1 or normalized_stratum_codes.numel() != (
            correct_all_semantic_row_ce.numel()
        ):
            raise ValueError(
                "Scene-state target strata must contain one code per semantic row"
            )
        valid_stratum_codes = set(_SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES.values())
        observed_stratum_codes = set(normalized_stratum_codes.cpu().tolist())
        if not observed_stratum_codes.issubset(valid_stratum_codes):
            raise ValueError(
                "Scene-state target strata contain unsupported codes: "
                f"{sorted(observed_stratum_codes.difference(valid_stratum_codes))}"
            )
        stratum_row_counts = {
            stratum: float(
                normalized_stratum_codes.eq(code).sum().detach().cpu().item()
            )
            for stratum, code in _SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES.items()
        }
        return {
            "keep_loss": float(correct_full_ce.detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": float(donor_margin_loss.detach().float().item()),
            "causal_loss": 0.0,
            "anchor_loss": 0.0,
            "full_ce_loss": float(correct_full_ce.detach().float().item()),
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": float(donor_gap.detach().float().item()),
            "wmem": 1.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
            "scene_state_full_correct_ce": float(
                correct_full_ce.detach().float().item()
            ),
            "scene_state_correct_all_semantic_ce": float(
                correct_all_semantic_ce.detach().float().item()
            ),
            "scene_state_correct_pair_semantic_ce": float(
                correct_pair_semantic_ce.detach().float().item()
            ),
            "scene_state_donor_pair_semantic_ce": float(
                donor_pair_semantic_ce.detach().float().item()
            ),
            "scene_state_zero_all_semantic_ce": float(
                zero_all_semantic_ce.detach().float().item()
            ),
            "scene_state_donor_pair_gap": float(
                donor_gap.detach().float().item()
            ),
            "scene_state_zero_all_gap": float(zero_gap.detach().float().item()),
            "scene_state_donor_margin_loss": float(
                donor_margin_loss.detach().float().item()
            ),
            "scene_state_donor_positive_fraction": float(
                donor_gap_rows.gt(0).float().mean().item()
            ),
            "scene_state_zero_positive_fraction": float(
                zero_gap_rows.gt(0).float().mean().item()
            ),
            "scene_state_semantic_token_count": float(semantic_token_count),
            "scene_state_semantic_row_count": float(
                correct_all_semantic_row_ce.numel()
            ),
            "scene_state_target_presence_row_count": stratum_row_counts[
                "presence"
            ],
            "scene_state_target_same_cardinality_value_row_count": (
                stratum_row_counts["same_cardinality_value"]
            ),
            "scene_state_target_cross_cardinality_value_row_count": (
                stratum_row_counts["cross_cardinality_value"]
            ),
        }

    def _compute_scene_state_identity_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        donor_write_input_ids: torch.Tensor,
        donor_write_attention_mask: torch.Tensor,
        donor_write_message_ids: torch.Tensor | None,
        donor_write_sentence_ids: torch.Tensor | None,
        semantic_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        target_stratum_codes: torch.Tensor,
    ):
        self._validate_scene_state_identity_runtime()
        (
            correct_full_ce,
            _,
            correct_all_semantic_row_ce,
            all_semantic_token_count,
            _,
            correct_pair_semantic_row_ce,
            pair_semantic_token_count,
            correct_outputs,
        ) = self._scene_state_identity_branch(
            model,
            model_inputs,
            loss_kwargs=loss_kwargs,
            prime_writes=True,
            write_input_ids=write_input_ids,
            write_attention_mask=write_attention_mask,
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
            semantic_mask=semantic_mask,
            pair_target_mask=pair_target_mask,
        )
        (
            _,
            _,
            _,
            donor_all_token_count,
            _,
            donor_pair_semantic_row_ce,
            donor_pair_token_count,
            _,
        ) = self._scene_state_identity_branch(
            model,
            model_inputs,
            loss_kwargs=loss_kwargs,
            prime_writes=True,
            write_input_ids=donor_write_input_ids,
            write_attention_mask=donor_write_attention_mask,
            write_message_ids=donor_write_message_ids,
            write_sentence_ids=donor_write_sentence_ids,
            semantic_mask=semantic_mask,
            pair_target_mask=pair_target_mask,
        )
        with torch.no_grad():
            (
                _,
                _,
                zero_all_semantic_row_ce,
                zero_all_token_count,
                _,
                _,
                zero_pair_token_count,
                _,
            ) = self._scene_state_identity_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                prime_writes=False,
                write_input_ids=None,
                write_attention_mask=None,
                write_message_ids=None,
                write_sentence_ids=None,
                semantic_mask=semantic_mask,
                pair_target_mask=pair_target_mask,
            )
        if not (
            all_semantic_token_count
            == donor_all_token_count
            == zero_all_token_count
        ) or not (
            pair_semantic_token_count
            == donor_pair_token_count
            == zero_pair_token_count
        ):
            raise RuntimeError(
                "Scene-state identity branches selected different semantic token counts"
            )
        objective_values = self._scene_state_identity_objective(
            correct_full_ce,
            correct_all_semantic_row_ce,
            correct_pair_semantic_row_ce,
            donor_pair_semantic_row_ce,
            zero_all_semantic_row_ce,
        )
        total_loss = objective_values[0]
        outputs = dict(correct_outputs)
        outputs["loss"] = total_loss
        outputs["memory_loss"] = (total_loss - correct_full_ce).detach()
        outputs["scene_state_correct_all_semantic_ce"] = objective_values[1].detach()
        outputs["scene_state_correct_pair_semantic_ce"] = objective_values[2].detach()
        outputs["scene_state_donor_pair_semantic_ce"] = objective_values[3].detach()
        outputs["scene_state_zero_all_semantic_ce"] = objective_values[4].detach()
        memory_stats = self._scene_state_identity_stats(
            correct_full_ce=correct_full_ce,
            correct_all_semantic_row_ce=correct_all_semantic_row_ce,
            correct_pair_semantic_row_ce=correct_pair_semantic_row_ce,
            donor_pair_semantic_row_ce=donor_pair_semantic_row_ce,
            zero_all_semantic_row_ce=zero_all_semantic_row_ce,
            semantic_token_count=all_semantic_token_count,
            target_stratum_codes=target_stratum_codes,
            objective_values=objective_values,
        )
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return total_loss, outputs, memory_stats

    def _scene_state_identity_loss_and_coefficients(
        self,
        correct_full_ce: torch.Tensor,
        correct_all_semantic_row_ce: torch.Tensor,
        correct_pair_semantic_row_ce: torch.Tensor,
        donor_pair_semantic_row_ce: torch.Tensor,
        zero_all_semantic_row_ce: torch.Tensor,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        correct_full_probe = correct_full_ce.detach().float().requires_grad_(True)
        correct_all_semantic_probe = (
            correct_all_semantic_row_ce.detach().float().requires_grad_(True)
        )
        correct_pair_semantic_probe = (
            correct_pair_semantic_row_ce.detach().float().requires_grad_(True)
        )
        donor_pair_semantic_probe = (
            donor_pair_semantic_row_ce.detach().float().requires_grad_(True)
        )
        zero_all_semantic_probe = zero_all_semantic_row_ce.detach().float()
        objective_values = self._scene_state_identity_objective(
            correct_full_probe,
            correct_all_semantic_probe,
            correct_pair_semantic_probe,
            donor_pair_semantic_probe,
            zero_all_semantic_probe,
        )
        coefficients = torch.autograd.grad(
            objective_values[0],
            (
                correct_full_probe,
                correct_all_semantic_probe,
                correct_pair_semantic_probe,
                donor_pair_semantic_probe,
            ),
        )
        return (
            tuple(value.detach() for value in objective_values),
            coefficients[0].detach(),
            coefficients[1].detach(),
            coefficients[2].detach(),
            coefficients[3].detach(),
        )

    @staticmethod
    def _validate_scene_state_replay(
        *,
        branch_name: str,
        replay_full_ce: torch.Tensor,
        probe_full_ce: torch.Tensor,
        replay_semantic_row_ce: torch.Tensor,
        probe_semantic_row_ce: torch.Tensor,
    ) -> None:
        if not torch.allclose(
            replay_full_ce.detach(),
            probe_full_ce,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Sequential scene-state {branch_name} full-CE replay is not "
                "deterministic"
            )
        if not torch.allclose(
            replay_semantic_row_ce.detach(),
            probe_semantic_row_ce,
            rtol=1e-5,
            atol=1e-6,
        ):
            maximum_error = torch.max(
                torch.abs(
                    replay_semantic_row_ce.detach().float()
                    - probe_semantic_row_ce.float()
                )
            )
            raise RuntimeError(
                f"Sequential scene-state {branch_name} semantic-CE replay is not "
                f"deterministic: max_abs_error={float(maximum_error.item()):.8f}"
            )

    def _scene_state_identity_sequential_backward(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        donor_write_input_ids: torch.Tensor,
        donor_write_attention_mask: torch.Tensor,
        donor_write_message_ids: torch.Tensor | None,
        donor_write_sentence_ids: torch.Tensor | None,
        semantic_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        target_stratum_codes: torch.Tensor,
        gradient_scale: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backpropagate two live branches and probe reset state diagnostically."""

        self._validate_scene_state_identity_sequential_runtime()
        donor_probe_rng_state = _capture_torch_rng_state()
        with torch.no_grad(), self.compute_loss_context_manager():
            (
                donor_full_probe,
                _,
                _,
                donor_all_token_count,
                _,
                donor_pair_semantic_row_probe,
                donor_pair_token_count,
                donor_probe_outputs,
            ) = self._scene_state_identity_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                prime_writes=True,
                write_input_ids=donor_write_input_ids,
                write_attention_mask=donor_write_attention_mask,
                write_message_ids=donor_write_message_ids,
                write_sentence_ids=donor_write_sentence_ids,
                semantic_mask=semantic_mask,
                pair_target_mask=pair_target_mask,
            )
        donor_full_probe = donor_full_probe.detach()
        donor_pair_semantic_row_probe = donor_pair_semantic_row_probe.detach()
        del donor_probe_outputs

        with torch.no_grad(), self.compute_loss_context_manager():
            (
                _,
                _,
                zero_all_semantic_row_probe,
                zero_all_token_count,
                _,
                _,
                zero_pair_token_count,
                zero_probe_outputs,
            ) = self._scene_state_identity_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                prime_writes=False,
                write_input_ids=None,
                write_attention_mask=None,
                write_message_ids=None,
                write_sentence_ids=None,
                semantic_mask=semantic_mask,
                pair_target_mask=pair_target_mask,
            )
        zero_all_semantic_row_probe = zero_all_semantic_row_probe.detach()
        del zero_probe_outputs

        with self.compute_loss_context_manager():
            (
                correct_full_ce,
                _,
                correct_all_semantic_row_ce,
                correct_all_token_count,
                _,
                correct_pair_semantic_row_ce,
                correct_pair_token_count,
                correct_outputs,
            ) = self._scene_state_identity_branch(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                prime_writes=True,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                semantic_mask=semantic_mask,
                pair_target_mask=pair_target_mask,
            )
        if not (
            correct_all_token_count
            == donor_all_token_count
            == zero_all_token_count
        ) or not (
            correct_pair_token_count
            == donor_pair_token_count
            == zero_pair_token_count
        ):
            raise RuntimeError(
                "Scene-state identity branches selected different semantic token counts"
            )
        (
            objective_values,
            correct_full_coefficient,
            correct_all_semantic_coefficient,
            correct_pair_semantic_coefficient,
            donor_pair_semantic_coefficient,
        ) = self._scene_state_identity_loss_and_coefficients(
            correct_full_ce,
            correct_all_semantic_row_ce,
            correct_pair_semantic_row_ce,
            donor_pair_semantic_row_probe,
            zero_all_semantic_row_probe,
        )
        correct_full_value = correct_full_ce.detach()
        correct_all_semantic_row_value = correct_all_semantic_row_ce.detach()
        correct_pair_semantic_row_value = correct_pair_semantic_row_ce.detach()
        correct_backward = (
            correct_full_ce * correct_full_coefficient
            + torch.sum(
                correct_all_semantic_row_ce.float()
                * correct_all_semantic_coefficient
            )
            + torch.sum(
                correct_pair_semantic_row_ce.float()
                * correct_pair_semantic_coefficient
            )
        )
        self.accelerator.backward(correct_backward * gradient_scale)
        del (
            correct_backward,
            correct_full_ce,
            correct_all_semantic_row_ce,
            correct_pair_semantic_row_ce,
            correct_outputs,
        )

        post_correct_rng_state = _capture_torch_rng_state()
        _restore_torch_rng_state(donor_probe_rng_state)
        try:
            with self.compute_loss_context_manager():
                (
                    donor_full_ce,
                    _,
                    _,
                    donor_replay_all_token_count,
                    _,
                    donor_pair_semantic_row_ce,
                    donor_replay_pair_token_count,
                    donor_outputs,
                ) = self._scene_state_identity_branch(
                    model,
                    model_inputs,
                    loss_kwargs=loss_kwargs,
                    prime_writes=True,
                    write_input_ids=donor_write_input_ids,
                    write_attention_mask=donor_write_attention_mask,
                    write_message_ids=donor_write_message_ids,
                    write_sentence_ids=donor_write_sentence_ids,
                    semantic_mask=semantic_mask,
                    pair_target_mask=pair_target_mask,
                )
        finally:
            _restore_torch_rng_state(post_correct_rng_state)
        if (
            donor_replay_all_token_count != donor_all_token_count
            or donor_replay_pair_token_count != donor_pair_token_count
        ):
            raise RuntimeError(
                "Sequential scene-state donor replay selected a different token count"
            )
        self._validate_scene_state_replay(
            branch_name="donor",
            replay_full_ce=donor_full_ce,
            probe_full_ce=donor_full_probe,
            replay_semantic_row_ce=donor_pair_semantic_row_ce,
            probe_semantic_row_ce=donor_pair_semantic_row_probe,
        )
        donor_pair_semantic_row_value = donor_pair_semantic_row_ce.detach()
        donor_backward = torch.sum(
            donor_pair_semantic_row_ce.float() * donor_pair_semantic_coefficient
        )
        self.accelerator.backward(donor_backward * gradient_scale)
        del (
            donor_backward,
            donor_full_ce,
            donor_pair_semantic_row_ce,
            donor_outputs,
        )
        zero_all_semantic_row_value = zero_all_semantic_row_probe

        memory_stats = self._scene_state_identity_stats(
            correct_full_ce=correct_full_value,
            correct_all_semantic_row_ce=correct_all_semantic_row_value,
            correct_pair_semantic_row_ce=correct_pair_semantic_row_value,
            donor_pair_semantic_row_ce=donor_pair_semantic_row_value,
            zero_all_semantic_row_ce=zero_all_semantic_row_value,
            semantic_token_count=correct_all_token_count,
            target_stratum_codes=target_stratum_codes,
            objective_values=objective_values,
        )
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return objective_values[0] * gradient_scale, memory_stats

    def _validate_scene_state_generation_runtime(self) -> None:
        if self.episode_read_write_enabled:
            raise ValueError(
                "scene_state_generation_ce requires episode read writes to be disabled"
            )
        if self.memory_kl_weight != 0.0 or self.memory_base_kl_weight != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires all KL weights to be zero"
            )
        if getattr(self, "memory_representation_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires representation loss to be disabled"
            )
        if getattr(self, "write_sparsity_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires write sparsity loss to be disabled"
            )
        if (
            getattr(self, "memory_partition_alignment_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_entropy_weight", 0.0) != 0.0
            or getattr(self, "memory_partition_balance_weight", 0.0) != 0.0
        ):
            raise ValueError(
                "scene_state_generation_ce requires memory partition regularization "
                "to be disabled"
            )
        if getattr(self, "scene_boundary_payload_ce_weight", 0.0) != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires scene-boundary payload CE to be disabled"
            )

    def _validate_scene_state_generation_sequential_runtime(self) -> None:
        self._validate_scene_state_generation_runtime()
        accumulation_steps = int(
            getattr(self, "current_gradient_accumulation_steps", 1)
        )
        if accumulation_steps != 1:
            raise ValueError(
                "scene_state_generation_ce sequential backward requires "
                "gradient_accumulation_steps=1"
            )
        optimizer_name = str(
            getattr(getattr(self, "args", None), "optim", "")
        ).lower()
        if optimizer_name.endswith("lomo") or optimizer_name.endswith("adalomo"):
            raise ValueError(
                "scene_state_generation_ce sequential backward does not support "
                "LOMO or AdaLOMO"
            )
        distributed_type = getattr(
            getattr(self, "accelerator", None),
            "distributed_type",
            None,
        )
        distributed_name = getattr(distributed_type, "name", None)
        if distributed_name in {
            "DEEPSPEED",
            "MEGATRON_LM",
        }:
            raise ValueError(
                "scene_state_generation_ce sequential backward does not support "
                f"distributed mode {distributed_name}"
            )
        if is_sagemaker_mp_enabled():
            raise ValueError(
                "scene_state_generation_ce sequential backward does not support "
                "SageMaker model parallelism"
            )

    @staticmethod
    def _scene_state_generation_token_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        description: str,
        token_ce: torch.Tensor | None = None,
        ce_target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        target_mask = target_mask.to(device=labels.device, dtype=torch.bool)
        supervised = labels.ne(-100) & attention_mask.ne(0)
        if target_mask.shape != labels.shape or bool(target_mask[:, 0].any()) or bool(
            (target_mask & ~supervised).any()
        ):
            raise ValueError(
                f"Scene-state generation {description} mask is not a causal label subset"
            )
        shift_mask = target_mask[:, 1:]
        counts = shift_mask.sum(dim=1)
        if not bool(counts.gt(0).all()):
            raise ValueError(
                f"Scene-state generation {description} mask misses a batch row"
            )
        shift_labels = labels[:, 1:]
        if token_ce is None:
            if ce_target_mask is None:
                raise ValueError("Scene-state generation CE target mask is required")
            shift_ce_target_mask = ce_target_mask.to(
                device=labels.device,
                dtype=torch.bool,
            )[:, 1:]
            selected_shift_logits = logits[:, :-1][shift_ce_target_mask].float()
            selected_token_ce = F.cross_entropy(
                selected_shift_logits,
                shift_labels[shift_ce_target_mask],
                reduction="none",
                ignore_index=-100,
            )
            token_ce = selected_token_ce.new_zeros(shift_labels.shape).masked_scatter(
                shift_ce_target_mask,
                selected_token_ce,
            )
        return (
            (token_ce * shift_mask).sum(dim=1) / counts,
            int(counts.sum().item()),
            token_ce,
        )

    @staticmethod
    def _scene_state_generation_gold_margins(
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shift_mask = target_mask[:, 1:].to(dtype=torch.bool)
        shift_labels = labels[:, 1:]
        counts = shift_mask.sum(dim=1)
        if not bool(counts.gt(0).all()):
            raise ValueError("Scene-state generation margin mask misses a batch row")

        # Gather in the source dtype before promoting to FP32. At production
        # vocabulary size, promoting every sequence position needlessly keeps a
        # much larger tensor alive than the selected generation targets.
        selected_logits = logits[:, :-1][shift_mask].float()
        selected_labels = shift_labels[shift_mask]
        gold_logits = selected_logits.gather(
            -1,
            selected_labels.unsqueeze(-1),
        ).squeeze(-1)
        top_values, top_indices = selected_logits.topk(k=2, dim=-1)
        max_other = torch.where(
            top_indices[..., 0].eq(selected_labels),
            top_values[..., 1],
            top_values[..., 0],
        )
        margins = gold_logits - max_other
        predictions = top_indices[..., 0]
        correct = predictions.eq(selected_labels)
        row_margins: list[torch.Tensor] = []
        row_accuracies: list[torch.Tensor] = []
        first_error_losses: list[torch.Tensor] = []
        first_error_ordinals: list[float] = []
        solved: list[float] = []
        offset = 0
        for row_index in range(logits.size(0)):
            row_count = int(counts[row_index].item())
            row_slice = slice(offset, offset + row_count)
            row_margin_values = margins[row_slice]
            row_correct = correct[row_slice]
            row_margins.append(row_margin_values.mean())
            row_accuracies.append(row_correct.float().mean())
            row_wrong = ~row_correct
            wrong_ordinals = row_wrong.nonzero(as_tuple=False).flatten()
            if wrong_ordinals.numel() == 0:
                first_error_losses.append(row_margin_values.sum() * 0.0)
                first_error_ordinals.append(float(row_count))
                solved.append(1.0)
            else:
                ordinal = int(wrong_ordinals[0].item())
                selected_index = offset + ordinal
                first_error_losses.append(
                    F.relu(
                        _SCENE_STATE_GENERATION_TOP1_MARGIN
                        + max_other[selected_index]
                        - gold_logits[selected_index]
                    )
                )
                first_error_ordinals.append(float(ordinal))
                solved.append(0.0)
            offset += row_count
        if offset != selected_logits.size(0):
            raise RuntimeError("Scene-state generation margin row accounting differs")
        return (
            torch.stack(row_margins),
            torch.stack(row_accuracies),
            torch.stack(first_error_losses),
            logits.new_tensor(first_error_ordinals),
            logits.new_tensor(solved),
        )

    def _scene_state_generation_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        content_mask: torch.Tensor,
        schema_mask: torch.Tensor,
        decision_mask: torch.Tensor,
        termination_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        donor_target_token_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        masks = {
            "target": target_mask.to(dtype=torch.bool),
            "content": content_mask.to(dtype=torch.bool),
            "schema": schema_mask.to(dtype=torch.bool),
            "decision": decision_mask.to(dtype=torch.bool),
            "termination": termination_mask.to(dtype=torch.bool),
            "pair": pair_target_mask.to(dtype=torch.bool),
        }
        if any(mask.shape != labels.shape for mask in masks.values()):
            raise ValueError("Scene-state generation masks must align with labels")
        if not torch.equal(masks["target"], masks["content"] | masks["termination"]):
            raise ValueError("Scene-state generation target partition differs")
        if not torch.equal(masks["content"], masks["schema"] | masks["decision"]):
            raise ValueError("Scene-state generation content partition differs")
        if bool((masks["schema"] & masks["decision"]).any()) or bool(
            (masks["content"] & masks["termination"]).any()
        ):
            raise ValueError("Scene-state generation masks overlap")
        if not torch.equal(masks["target"], labels.ne(-100)):
            raise ValueError(
                "Scene-state generation labels must equal the generated suffix mask"
            )
        if bool((masks["pair"] & ~masks["decision"]).any()) or not bool(
            masks["pair"].sum(dim=1).eq(1).all()
        ):
            raise ValueError(
                "Scene-state generation pair target must select one decision token per row"
            )
        schema_row_ce, schema_count, token_ce = self._scene_state_generation_token_ce(
            logits,
            labels,
            attention_mask,
            masks["schema"],
            description="schema",
            ce_target_mask=masks["target"],
        )
        decision_row_ce, decision_count, token_ce = (
            self._scene_state_generation_token_ce(
                logits,
                labels,
                attention_mask,
                masks["decision"],
                description="decision",
                token_ce=token_ce,
            )
        )
        termination_row_ce, termination_count, token_ce = (
            self._scene_state_generation_token_ce(
                logits,
                labels,
                attention_mask,
                masks["termination"],
                description="termination",
                token_ce=token_ce,
            )
        )
        token_weights = (
            masks["schema"][:, 1:].float()
            * _SCENE_STATE_GENERATION_SCHEMA_WEIGHT
            + masks["decision"][:, 1:].float()
            * _SCENE_STATE_GENERATION_DECISION_WEIGHT
            + masks["termination"][:, 1:].float()
            * _SCENE_STATE_GENERATION_TERMINATION_WEIGHT
        )
        weighted_generation_row_ce = (token_ce * token_weights).sum(dim=1) / (
            token_weights.sum(dim=1)
        )
        (
            target_margin_row,
            target_accuracy_row,
            first_error_row_loss,
            first_error_ordinal,
            solved_row,
        ) = self._scene_state_generation_gold_margins(
            logits,
            labels,
            masks["target"],
        )
        decision_margin_row, _, _, _, _ = (
            self._scene_state_generation_gold_margins(
                logits,
                labels,
                masks["decision"],
            )
        )
        pair_positions = masks["pair"].long().argmax(dim=1)
        predictor_positions = pair_positions - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        pair_predictor_logits = logits[
            batch_indices,
            predictor_positions,
        ].float()
        source_target_ids = labels[batch_indices, pair_positions]
        donor_target_ids = donor_target_token_ids.to(
            device=logits.device,
            dtype=torch.long,
        )
        if donor_target_ids.ndim != 1 or donor_target_ids.numel() != logits.size(0):
            raise ValueError("Scene-state generation donor targets must contain one ID per row")
        if bool(source_target_ids.eq(donor_target_ids).any()):
            raise ValueError("Scene-state generation source/donor pair targets must differ")
        pair_logits = torch.stack(
            (
                pair_predictor_logits.gather(
                    1,
                    source_target_ids.unsqueeze(1),
                ).squeeze(1),
                pair_predictor_logits.gather(
                    1,
                    donor_target_ids.unsqueeze(1),
                ).squeeze(1),
            ),
            dim=1,
        )
        return {
            "weighted_generation_row_ce": weighted_generation_row_ce,
            "schema_row_ce": schema_row_ce,
            "decision_row_ce": decision_row_ce,
            "termination_row_ce": termination_row_ce,
            "target_margin_row": target_margin_row,
            "decision_margin_row": decision_margin_row,
            "target_accuracy_row": target_accuracy_row,
            "first_error_row_loss": first_error_row_loss,
            "first_error_ordinal": first_error_ordinal,
            "solved_row": solved_row,
            "pair_logits": pair_logits,
            "target_token_count": int(masks["target"][:, 1:].sum().item()),
            "content_token_count": int(masks["content"][:, 1:].sum().item()),
            "schema_token_count": schema_count,
            "decision_token_count": decision_count,
            "termination_token_count": termination_count,
        }

    def _scene_state_generation_branch(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        prime_writes: bool,
        write_input_ids: torch.Tensor | None,
        write_attention_mask: torch.Tensor | None,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        target_mask: torch.Tensor,
        content_mask: torch.Tensor,
        schema_mask: torch.Tensor,
        decision_mask: torch.Tensor,
        termination_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        donor_target_token_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | int]]:
        batch_size = int(model_inputs["input_ids"].size(0))
        self._reset_online_state(model)
        if prime_writes:
            if write_input_ids is None or write_attention_mask is None:
                raise ValueError(
                    "Scene-state generation correct/donor branches require writes"
                )
            self._prime_episode_state(
                model,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                batch_size=batch_size,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
            )
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_read_context_mask(
            model,
            self._build_read_context_mask(model_inputs),
        )
        forward_inputs = dict(model_inputs)
        labels = forward_inputs.pop("labels")
        outputs = model(**forward_inputs, **loss_kwargs)
        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss if hasattr(outputs, "loss") else outputs[0],
                "logits": outputs.logits,
            }
        metrics = self._scene_state_generation_metrics(
            outputs["logits"],
            labels,
            model_inputs["attention_mask"],
            target_mask=target_mask,
            content_mask=content_mask,
            schema_mask=schema_mask,
            decision_mask=decision_mask,
            termination_mask=termination_mask,
            pair_target_mask=pair_target_mask,
            donor_target_token_ids=donor_target_token_ids,
        )
        return outputs, metrics

    @staticmethod
    def _scene_state_generation_objective(
        correct: dict[str, torch.Tensor | int],
        donor: dict[str, torch.Tensor | int],
        zero: dict[str, torch.Tensor | int],
    ) -> dict[str, torch.Tensor]:
        generation_ce = correct["weighted_generation_row_ce"].mean()
        first_error_loss = correct["first_error_row_loss"].mean()
        batch_size = correct["pair_logits"].size(0)
        correct_pair_ce = F.cross_entropy(
            correct["pair_logits"],
            torch.zeros(batch_size, device=correct["pair_logits"].device, dtype=torch.long),
        )
        donor_pair_ce = F.cross_entropy(
            donor["pair_logits"],
            torch.ones(batch_size, device=donor["pair_logits"].device, dtype=torch.long),
        )
        zero_margin_loss = F.relu(
            _SCENE_STATE_GENERATION_ZERO_MARGIN
            - (
                correct["decision_margin_row"]
                - zero["decision_margin_row"].detach()
            )
        ).mean()
        total_loss = (
            generation_ce
            + first_error_loss
            + correct_pair_ce
            + donor_pair_ce
            + zero_margin_loss
        )
        return {
            "total_loss": total_loss,
            "generation_ce": generation_ce,
            "first_error_loss": first_error_loss,
            "correct_pair_ce": correct_pair_ce,
            "donor_pair_ce": donor_pair_ce,
            "zero_margin_loss": zero_margin_loss,
        }

    def _scene_state_generation_stats(
        self,
        *,
        correct: dict[str, torch.Tensor | int],
        donor: dict[str, torch.Tensor | int],
        zero: dict[str, torch.Tensor | int],
        objective: dict[str, torch.Tensor],
        target_stratum_codes: torch.Tensor,
    ) -> dict[str, float]:
        normalized_stratum_codes = target_stratum_codes.detach().to(dtype=torch.long)
        stratum_row_counts = {
            stratum: float(
                normalized_stratum_codes.eq(code).sum().detach().cpu().item()
            )
            for stratum, code in _SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES.items()
        }
        correct_pair_preference = (
            correct["pair_logits"][:, 0] - correct["pair_logits"][:, 1]
        ).mean()
        donor_pair_preference = (
            donor["pair_logits"][:, 1] - donor["pair_logits"][:, 0]
        ).mean()
        total_loss = objective["total_loss"]
        return {
            "keep_loss": float(objective["generation_ce"].detach().float().item()),
            "reset_loss": 0.0,
            "corrupt_loss": 0.0,
            "teacher_loss": 0.0,
            "margin_loss": float(
                objective["zero_margin_loss"].detach().float().item()
            ),
            "causal_loss": float(
                objective["first_error_loss"].detach().float().item()
            ),
            "anchor_loss": float(
                (objective["correct_pair_ce"] + objective["donor_pair_ce"])
                .detach()
                .float()
                .item()
            ),
            "full_ce_loss": float(
                objective["generation_ce"].detach().float().item()
            ),
            "kl_loss": 0.0,
            "reset_kl_loss": 0.0,
            "margin_gap": float(
                (
                    correct["decision_margin_row"].mean()
                    - zero["decision_margin_row"].mean()
                )
                .detach()
                .float()
                .item()
            ),
            "wmem": 1.0,
            "probe_keep_loss": 0.0,
            "probe_reset_loss": 0.0,
            "probe_margin_loss": 0.0,
            "probe_gap": 0.0,
            "probe_kl": 0.0,
            "probe_ce": 0.0,
            "scene_generation_total_loss": float(total_loss.detach().float().item()),
            "scene_generation_weighted_ce": float(
                objective["generation_ce"].detach().float().item()
            ),
            "scene_generation_schema_ce": float(
                correct["schema_row_ce"].mean().detach().float().item()
            ),
            "scene_generation_decision_ce": float(
                correct["decision_row_ce"].mean().detach().float().item()
            ),
            "scene_generation_termination_ce": float(
                correct["termination_row_ce"].mean().detach().float().item()
            ),
            "scene_generation_first_error_loss": float(
                objective["first_error_loss"].detach().float().item()
            ),
            "scene_generation_pair_correct_ce": float(
                objective["correct_pair_ce"].detach().float().item()
            ),
            "scene_generation_pair_donor_ce": float(
                objective["donor_pair_ce"].detach().float().item()
            ),
            "scene_generation_zero_margin_loss": float(
                objective["zero_margin_loss"].detach().float().item()
            ),
            "scene_generation_gold_top1_accuracy": float(
                correct["target_accuracy_row"].mean().detach().float().item()
            ),
            "scene_generation_first_error_ordinal": float(
                correct["first_error_ordinal"].mean().detach().float().item()
            ),
            "scene_generation_solved_fraction": float(
                correct["solved_row"].mean().detach().float().item()
            ),
            "scene_generation_correct_decision_margin": float(
                correct["decision_margin_row"].mean().detach().float().item()
            ),
            "scene_generation_donor_decision_margin": float(
                donor["decision_margin_row"].mean().detach().float().item()
            ),
            "scene_generation_zero_decision_margin": float(
                zero["decision_margin_row"].mean().detach().float().item()
            ),
            "scene_generation_correct_pair_preference": float(
                correct_pair_preference.detach().float().item()
            ),
            "scene_generation_donor_pair_preference": float(
                donor_pair_preference.detach().float().item()
            ),
            "scene_generation_target_token_count": float(
                correct["target_token_count"]
            ),
            "scene_generation_content_token_count": float(
                correct["content_token_count"]
            ),
            "scene_generation_schema_token_count": float(
                correct["schema_token_count"]
            ),
            "scene_generation_decision_token_count": float(
                correct["decision_token_count"]
            ),
            "scene_generation_termination_token_count": float(
                correct["termination_token_count"]
            ),
            "scene_state_target_presence_row_count": stratum_row_counts["presence"],
            "scene_state_target_same_cardinality_value_row_count": (
                stratum_row_counts["same_cardinality_value"]
            ),
            "scene_state_target_cross_cardinality_value_row_count": (
                stratum_row_counts["cross_cardinality_value"]
            ),
        }

    def _compute_scene_state_generation_ce(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        donor_write_input_ids: torch.Tensor,
        donor_write_attention_mask: torch.Tensor,
        donor_write_message_ids: torch.Tensor | None,
        donor_write_sentence_ids: torch.Tensor | None,
        target_mask: torch.Tensor,
        content_mask: torch.Tensor,
        schema_mask: torch.Tensor,
        decision_mask: torch.Tensor,
        termination_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        donor_target_token_ids: torch.Tensor,
        target_stratum_codes: torch.Tensor,
    ):
        self._validate_scene_state_generation_runtime()
        branch_kwargs = {
            "model_inputs": model_inputs,
            "loss_kwargs": loss_kwargs,
            "target_mask": target_mask,
            "content_mask": content_mask,
            "schema_mask": schema_mask,
            "decision_mask": decision_mask,
            "termination_mask": termination_mask,
            "pair_target_mask": pair_target_mask,
            "donor_target_token_ids": donor_target_token_ids,
        }
        correct_outputs, correct = self._scene_state_generation_branch(
            model,
            prime_writes=True,
            write_input_ids=write_input_ids,
            write_attention_mask=write_attention_mask,
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
            **branch_kwargs,
        )
        donor_outputs, donor = self._scene_state_generation_branch(
            model,
            prime_writes=True,
            write_input_ids=donor_write_input_ids,
            write_attention_mask=donor_write_attention_mask,
            write_message_ids=donor_write_message_ids,
            write_sentence_ids=donor_write_sentence_ids,
            **branch_kwargs,
        )
        with torch.no_grad():
            zero_outputs, zero = self._scene_state_generation_branch(
                model,
                prime_writes=False,
                write_input_ids=None,
                write_attention_mask=None,
                write_message_ids=None,
                write_sentence_ids=None,
                **branch_kwargs,
            )
        del donor_outputs, zero_outputs
        objective = self._scene_state_generation_objective(correct, donor, zero)
        outputs = dict(correct_outputs)
        outputs["loss"] = objective["total_loss"]
        outputs["memory_loss"] = objective["total_loss"].detach()
        memory_stats = self._scene_state_generation_stats(
            correct=correct,
            donor=donor,
            zero=zero,
            objective=objective,
            target_stratum_codes=target_stratum_codes,
        )
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return objective["total_loss"], outputs, memory_stats

    @staticmethod
    def _validate_scene_state_generation_replay(
        probe: dict[str, torch.Tensor | int],
        replay: dict[str, torch.Tensor | int],
    ) -> None:
        for field in ("pair_logits", "decision_margin_row"):
            if not torch.allclose(
                replay[field].detach(),
                probe[field],
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"Sequential scene-state generation donor replay differs: {field}"
                )

    @staticmethod
    def _scene_state_generated_wrong_positions(
        generated_token_ids: torch.Tensor,
        gold_token_ids: torch.Tensor,
        *,
        max_wrong_tokens: int,
    ) -> tuple[int, torch.Tensor]:
        return generated_unlikelihood_positions(
            generated_token_ids,
            gold_token_ids,
            max_wrong_tokens=max_wrong_tokens,
        )

    def _scene_state_generated_rollout_inputs(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        target_mask: torch.Tensor,
        termination_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = model_inputs["labels"]
        if input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError(
                "Generated-prefix unlikelihood currently requires batch size 1"
            )
        if not (
            input_ids.shape
            == attention_mask.shape
            == labels.shape
            == target_mask.shape
            == termination_mask.shape
        ):
            raise ValueError("Generated-prefix rollout tensors must have matching shapes")
        normalized_target = target_mask.to(device=input_ids.device, dtype=torch.bool)
        normalized_termination = termination_mask.to(
            device=input_ids.device,
            dtype=torch.bool,
        )
        target_positions = normalized_target[0].nonzero(as_tuple=False).flatten()
        termination_positions = normalized_termination[0].nonzero(
            as_tuple=False
        ).flatten()
        if target_positions.numel() == 0 or termination_positions.numel() == 0:
            raise ValueError(
                "Generated-prefix rollout requires target and termination tokens"
            )
        generation_start = int(target_positions[0].item())
        generation_end = int(target_positions[-1].item())
        if generation_start <= 0 or not torch.equal(
            target_positions,
            torch.arange(
                generation_start,
                generation_end + 1,
                device=target_positions.device,
            ),
        ):
            raise ValueError(
                "Generated-prefix rollout target must be one causal suffix"
            )
        if bool((normalized_termination & ~normalized_target).any()):
            raise ValueError(
                "Generated-prefix termination mask must be a target subset"
            )
        first_termination = int(termination_positions[0].item())
        if first_termination < generation_start:
            raise ValueError("Generated-prefix termination precedes the suffix")
        if not torch.equal(
            labels[0, target_positions],
            input_ids[0, target_positions],
        ):
            raise ValueError(
                "Generated-prefix gold labels must equal the supplied suffix token IDs"
            )
        prompt_attention_mask = attention_mask[:, :generation_start]
        if not bool(prompt_attention_mask.ne(0).all()):
            raise ValueError(
                "Generated-prefix rollout does not support padding inside the prompt"
            )
        full_gold_token_ids = input_ids[0, generation_start : generation_end + 1]
        # Greedy generation stops on the first termination token. Tokens rendered
        # after it by a full chat template are teacher-forcing targets, not part
        # of the benchmark generation that the hard negative must match.
        benchmark_gold_token_ids = input_ids[
            0,
            generation_start : first_termination + 1,
        ]
        rollout_max_new_tokens = min(
            int(full_gold_token_ids.numel())
            + self.scene_state_generated_rollout_extra_tokens,
            self.scene_state_generated_rollout_max_tokens,
        )
        return {
            "prompt_input_ids": input_ids[:, :generation_start],
            "prompt_attention_mask": prompt_attention_mask,
            "gold_token_ids": benchmark_gold_token_ids,
            "full_gold_token_count": int(full_gold_token_ids.numel()),
            "generation_start": generation_start,
            "rollout_max_new_tokens": rollout_max_new_tokens,
        }

    def _scene_state_generated_greedy_rollout(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        online_state_snapshot: dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        termination_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | bool]:
        rollout = self._scene_state_generated_rollout_inputs(
            model_inputs,
            target_mask=target_mask,
            termination_mask=termination_mask,
        )
        prompt_input_ids = rollout["prompt_input_ids"]
        prompt_attention_mask = rollout["prompt_attention_mask"]
        was_training = model.training
        rollout_rng_state = _capture_torch_rng_state()
        model.eval()
        try:
            with torch.no_grad():
                self._reset_online_state(model)
                load_delta_mem_online_state(
                    model,
                    clone_detached_online_state(online_state_snapshot),
                )
                set_delta_mem_write_enabled(model, False)
                set_delta_mem_read_context_mask(
                    model,
                    prompt_attention_mask.to(dtype=torch.bool),
                )
                generated_sequences = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    do_sample=False,
                    max_new_tokens=int(rollout["rollout_max_new_tokens"]),
                    use_cache=True,
                )
        finally:
            _restore_torch_rng_state(rollout_rng_state)
            model.train(was_training)
        if not isinstance(generated_sequences, torch.Tensor) or (
            generated_sequences.ndim != 2 or generated_sequences.size(0) != 1
        ):
            raise RuntimeError("Greedy generated-prefix rollout returned invalid sequences")
        prompt_length = int(prompt_input_ids.size(1))
        if generated_sequences.size(1) <= prompt_length or not torch.equal(
            generated_sequences[:, :prompt_length],
            prompt_input_ids,
        ):
            raise RuntimeError(
                "Greedy generated-prefix rollout did not preserve the exact prompt"
            )
        generated_token_ids = generated_sequences[0, prompt_length:].detach()
        gold_token_ids = rollout["gold_token_ids"]
        first_divergence, wrong_positions = (
            self._scene_state_generated_wrong_positions(
                generated_token_ids,
                gold_token_ids,
                max_wrong_tokens=(
                    self.scene_state_generated_unlikelihood_max_wrong_tokens
                ),
            )
        )
        rollout.update(
            {
                "generated_token_ids": generated_token_ids,
                "wrong_positions": wrong_positions,
                "first_divergence": first_divergence,
                "exact_through_termination": bool(
                    torch.equal(generated_token_ids, gold_token_ids)
                ),
            }
        )
        return rollout

    @staticmethod
    def _scene_state_generated_unlikelihood_from_logits(
        selected_logits: torch.Tensor,
        wrong_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if selected_logits.ndim != 2 or selected_logits.size(0) == 0:
            raise ValueError(
                "Generated-prefix unlikelihood requires selected vocabulary logits"
            )
        if (
            wrong_token_ids.ndim != 1
            or wrong_token_ids.numel() != selected_logits.size(0)
        ):
            raise ValueError(
                "Generated-prefix wrong token IDs must align with selected logits"
            )
        fp32_logits = selected_logits.float()
        normalized_wrong_ids = wrong_token_ids.to(
            device=fp32_logits.device,
            dtype=torch.long,
        )
        wrong_logits = fp32_logits.gather(
            1,
            normalized_wrong_ids.unsqueeze(1),
        ).squeeze(1)
        other_logits = fp32_logits.clone()
        other_logits.scatter_(
            1,
            normalized_wrong_ids.unsqueeze(1),
            -torch.inf,
        )
        other_logsumexp = torch.logsumexp(other_logits, dim=1)
        return F.softplus(wrong_logits - other_logsumexp).mean()

    def _scene_state_generated_unlikelihood_branch(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        online_state_snapshot: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        target_mask: torch.Tensor,
        termination_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        rollout = self._scene_state_generated_greedy_rollout(
            model,
            model_inputs,
            online_state_snapshot=online_state_snapshot,
            target_mask=target_mask,
            termination_mask=termination_mask,
        )
        generated_token_ids = rollout["generated_token_ids"]
        wrong_positions = rollout["wrong_positions"]
        stats = {
            "scene_generation_generated_unlikelihood_loss": 0.0,
            "scene_generation_generated_unlikelihood_applied": float(
                wrong_positions.numel() > 0
            ),
            "scene_generation_generated_wrong_token_count": float(
                wrong_positions.numel()
            ),
            "scene_generation_generated_rollout_token_count": float(
                generated_token_ids.numel()
            ),
            "scene_generation_generated_first_divergence": float(
                rollout["first_divergence"]
            ),
            "scene_generation_generated_exact_fraction": float(
                rollout["exact_through_termination"]
            ),
        }
        if wrong_positions.numel() == 0:
            return None, stats

        prompt_input_ids = rollout["prompt_input_ids"]
        prompt_attention_mask = rollout["prompt_attention_mask"]
        replay_input_ids = torch.cat(
            (prompt_input_ids, generated_token_ids.unsqueeze(0)),
            dim=1,
        )
        replay_attention_mask = torch.cat(
            (
                prompt_attention_mask,
                torch.ones(
                    (1, generated_token_ids.numel()),
                    device=prompt_attention_mask.device,
                    dtype=prompt_attention_mask.dtype,
                ),
            ),
            dim=1,
        )
        self._reset_online_state(model)
        self._prime_episode_state(
            model,
            write_input_ids=write_input_ids,
            write_attention_mask=write_attention_mask,
            batch_size=1,
            write_message_ids=write_message_ids,
            write_sentence_ids=write_sentence_ids,
        )
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_read_context_mask(
            model,
            replay_attention_mask.to(dtype=torch.bool),
        )
        replay_outputs = model(
            input_ids=replay_input_ids,
            attention_mask=replay_attention_mask,
            use_cache=False,
        )
        replay_logits = (
            replay_outputs["logits"]
            if isinstance(replay_outputs, dict)
            else replay_outputs.logits
        )
        predictor_positions = (
            wrong_positions
            + int(rollout["generation_start"])
            - 1
        )
        selected_logits = replay_logits[0].index_select(
            0,
            predictor_positions,
        )
        wrong_token_ids = generated_token_ids.index_select(0, wrong_positions)
        unlikelihood_loss = self._scene_state_generated_unlikelihood_from_logits(
            selected_logits,
            wrong_token_ids,
        )
        stats["scene_generation_generated_unlikelihood_loss"] = float(
            unlikelihood_loss.detach().item()
        )
        return unlikelihood_loss, stats

    def _scene_state_generation_sequential_backward(
        self,
        model,
        model_inputs: dict[str, torch.Tensor],
        *,
        loss_kwargs: dict[str, torch.Tensor],
        write_input_ids: torch.Tensor,
        write_attention_mask: torch.Tensor,
        write_message_ids: torch.Tensor | None,
        write_sentence_ids: torch.Tensor | None,
        donor_write_input_ids: torch.Tensor,
        donor_write_attention_mask: torch.Tensor,
        donor_write_message_ids: torch.Tensor | None,
        donor_write_sentence_ids: torch.Tensor | None,
        target_mask: torch.Tensor,
        content_mask: torch.Tensor,
        schema_mask: torch.Tensor,
        decision_mask: torch.Tensor,
        termination_mask: torch.Tensor,
        pair_target_mask: torch.Tensor,
        donor_target_token_ids: torch.Tensor,
        target_stratum_codes: torch.Tensor,
        gradient_scale: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self._validate_scene_state_generation_sequential_runtime()
        generated_unlikelihood_weight = float(
            getattr(self, "scene_state_generated_unlikelihood_weight", 0.0)
        )
        normalized_strata = target_stratum_codes.detach().to(dtype=torch.long)
        if generated_unlikelihood_weight > 0.0 and normalized_strata.numel() != 1:
            raise ValueError(
                "Generated-prefix unlikelihood requires batch size 1"
            )
        presence_code = _SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES["presence"]
        generated_unlikelihood_eligible = (
            generated_unlikelihood_weight > 0.0
            and int(normalized_strata.item()) != presence_code
        )
        branch_kwargs = {
            "model_inputs": model_inputs,
            "loss_kwargs": loss_kwargs,
            "target_mask": target_mask,
            "content_mask": content_mask,
            "schema_mask": schema_mask,
            "decision_mask": decision_mask,
            "termination_mask": termination_mask,
            "pair_target_mask": pair_target_mask,
            "donor_target_token_ids": donor_target_token_ids,
        }
        donor_probe_rng_state = _capture_torch_rng_state()
        with torch.no_grad(), self.compute_loss_context_manager():
            donor_probe_outputs, donor_probe = self._scene_state_generation_branch(
                model,
                prime_writes=True,
                write_input_ids=donor_write_input_ids,
                write_attention_mask=donor_write_attention_mask,
                write_message_ids=donor_write_message_ids,
                write_sentence_ids=donor_write_sentence_ids,
                **branch_kwargs,
            )
        donor_probe = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in donor_probe.items()
        }
        del donor_probe_outputs
        with torch.no_grad(), self.compute_loss_context_manager():
            zero_outputs, zero = self._scene_state_generation_branch(
                model,
                prime_writes=False,
                write_input_ids=None,
                write_attention_mask=None,
                write_message_ids=None,
                write_sentence_ids=None,
                **branch_kwargs,
            )
        zero = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in zero.items()
        }
        del zero_outputs
        with self.compute_loss_context_manager():
            correct_outputs, correct = self._scene_state_generation_branch(
                model,
                prime_writes=True,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                **branch_kwargs,
            )
        generated_online_state_snapshot = None
        if generated_unlikelihood_eligible:
            generated_online_state_snapshot = clone_detached_online_state(
                self._capture_live_online_state(model)
            )
        objective_probe = self._scene_state_generation_objective(
            correct,
            donor_probe,
            zero,
        )
        correct_values = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in correct.items()
        }
        correct_backward = (
            objective_probe["generation_ce"]
            + objective_probe["first_error_loss"]
            + objective_probe["correct_pair_ce"]
            + objective_probe["zero_margin_loss"]
        )
        correct_backward_root = correct_backward * gradient_scale
        del objective_probe, correct_backward, correct_outputs, correct
        self.accelerator.backward(correct_backward_root)
        del correct_backward_root

        post_correct_rng_state = _capture_torch_rng_state()
        _restore_torch_rng_state(donor_probe_rng_state)
        try:
            with self.compute_loss_context_manager():
                donor_outputs, donor = self._scene_state_generation_branch(
                    model,
                    prime_writes=True,
                    write_input_ids=donor_write_input_ids,
                    write_attention_mask=donor_write_attention_mask,
                    write_message_ids=donor_write_message_ids,
                    write_sentence_ids=donor_write_sentence_ids,
                    **branch_kwargs,
                )
        finally:
            _restore_torch_rng_state(post_correct_rng_state)
        self._validate_scene_state_generation_replay(donor_probe, donor)
        donor_values = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in donor.items()
        }
        donor_pair_ce = F.cross_entropy(
            donor["pair_logits"],
            torch.ones(
                donor["pair_logits"].size(0),
                device=donor["pair_logits"].device,
                dtype=torch.long,
            ),
        )
        donor_backward_root = donor_pair_ce * gradient_scale
        del donor_pair_ce, donor_outputs, donor
        self.accelerator.backward(donor_backward_root)
        del donor_backward_root

        generated_unlikelihood_stats = {
            "scene_generation_generated_unlikelihood_loss": 0.0,
            "scene_generation_generated_unlikelihood_applied": 0.0,
            "scene_generation_generated_wrong_token_count": 0.0,
            "scene_generation_generated_rollout_token_count": 0.0,
            "scene_generation_generated_first_divergence": 0.0,
            "scene_generation_generated_exact_fraction": 0.0,
        }
        generated_unlikelihood_value = zero["decision_margin_row"].new_zeros(())
        if generated_unlikelihood_eligible:
            if generated_online_state_snapshot is None:
                raise RuntimeError(
                    "Generated-prefix unlikelihood lost its correct-state snapshot"
                )
            with self.compute_loss_context_manager():
                generated_unlikelihood_loss, generated_unlikelihood_stats = (
                    self._scene_state_generated_unlikelihood_branch(
                        model,
                        model_inputs,
                        online_state_snapshot=generated_online_state_snapshot,
                        write_input_ids=write_input_ids,
                        write_attention_mask=write_attention_mask,
                        write_message_ids=write_message_ids,
                        write_sentence_ids=write_sentence_ids,
                        target_mask=target_mask,
                        termination_mask=termination_mask,
                    )
                )
            if generated_unlikelihood_loss is not None:
                generated_unlikelihood_value = generated_unlikelihood_loss.detach()
                generated_unlikelihood_root = (
                    generated_unlikelihood_loss
                    * generated_unlikelihood_weight
                    * gradient_scale
                )
                del generated_unlikelihood_loss
                self.accelerator.backward(generated_unlikelihood_root)
                del generated_unlikelihood_root
            del generated_online_state_snapshot
        objective = self._scene_state_generation_objective(
            correct_values,
            donor_values,
            zero,
        )
        memory_stats = self._scene_state_generation_stats(
            correct=correct_values,
            donor=donor_values,
            zero=zero,
            objective=objective,
            target_stratum_codes=target_stratum_codes,
        )
        weighted_generated_unlikelihood = (
            generated_unlikelihood_value
            * generated_unlikelihood_weight
        )
        reported_total_loss = objective["total_loss"] + weighted_generated_unlikelihood
        memory_stats.update(generated_unlikelihood_stats)
        memory_stats["scene_generation_generated_unlikelihood_weighted_loss"] = float(
            weighted_generated_unlikelihood.detach().item()
        )
        memory_stats["scene_generation_total_loss"] = float(
            reported_total_loss.detach().item()
        )
        set_delta_mem_read_context_mask(model, None)
        set_delta_mem_write_enabled(model, True)
        return reported_total_loss * gradient_scale, memory_stats

    def _record_memory_stats(self, model, memory_stats: dict[str, float]) -> None:
        partition_route_stats = collect_delta_mem_partition_route_stats(model)
        self._last_partition_enabled_modules = partition_route_stats["enabled_modules"]
        self._last_partition_tied_read_write_modules = partition_route_stats[
            "tied_read_write_modules"
        ]
        self._last_partition_active_modules = partition_route_stats["active_modules"]
        self._last_partition_write_route_entropy = partition_route_stats[
            "write_route_entropy"
        ]
        self._last_partition_read_route_entropy = partition_route_stats[
            "read_route_entropy"
        ]
        self._last_partition_route_alignment_mse = partition_route_stats[
            "route_alignment_mse"
        ]
        self._last_partition_route_overlap = partition_route_stats["route_overlap"]
        self._last_partition_write_route_max = partition_route_stats["write_route_max"]
        self._last_partition_read_route_max = partition_route_stats["read_route_max"]
        self._last_partition_write_route_balance_l2 = partition_route_stats[
            "write_route_balance_l2"
        ]
        self._last_partition_read_route_balance_l2 = partition_route_stats[
            "read_route_balance_l2"
        ]
        self._last_memory_keep_loss = memory_stats["keep_loss"]
        self._last_memory_reset_loss = memory_stats["reset_loss"]
        self._last_memory_corrupt_loss = memory_stats["corrupt_loss"]
        self._last_memory_teacher_loss = memory_stats["teacher_loss"]
        self._last_scene_boundary_full_ce_loss = memory_stats.get(
            "scene_boundary_full_ce_loss",
            0.0,
        )
        self._last_scene_boundary_payload_ce_loss = memory_stats.get(
            "scene_boundary_payload_ce_loss",
            0.0,
        )
        self._last_scene_boundary_payload_auxiliary_loss = memory_stats.get(
            "scene_boundary_payload_auxiliary_loss",
            0.0,
        )
        self._last_scene_boundary_payload_token_count = memory_stats.get(
            "scene_boundary_payload_token_count",
            0.0,
        )
        self._last_scene_boundary_supervised_token_count = memory_stats.get(
            "scene_boundary_supervised_token_count",
            0.0,
        )
        self._last_memory_margin_loss = memory_stats["margin_loss"]
        self._last_memory_causal_loss = memory_stats["causal_loss"]
        self._last_memory_anchor_loss = memory_stats["anchor_loss"]
        self._last_memory_full_ce_loss = memory_stats["full_ce_loss"]
        self._last_memory_kl_loss = memory_stats["kl_loss"]
        self._last_memory_reset_kl_loss = memory_stats["reset_kl_loss"]
        self._last_memory_margin_gap = memory_stats["margin_gap"]
        self._last_memory_representation_loss = memory_stats.get(
            "representation_loss",
            0.0,
        )
        self._last_memory_representation_distance = memory_stats.get(
            "representation_distance",
            0.0,
        )
        self._last_content_contrast_full_correct_ce = memory_stats.get(
            "full_correct_ce",
            0.0,
        )
        self._last_content_contrast_full_donor_ce = memory_stats.get(
            "full_donor_ce",
            0.0,
        )
        self._last_content_contrast_targeted_correct_ce = memory_stats.get(
            "targeted_correct_ce",
            0.0,
        )
        self._last_content_contrast_targeted_donor_ce = memory_stats.get(
            "targeted_donor_ce",
            0.0,
        )
        self._last_content_contrast_targeted_gap = memory_stats.get(
            "targeted_gap",
            0.0,
        )
        self._last_content_contrast_targeted_positive_fraction = memory_stats.get(
            "targeted_positive_fraction",
            0.0,
        )
        self._last_content_contrast_targeted_token_count = memory_stats.get(
            "targeted_token_count",
            0.0,
        )
        self._last_scene_state_full_correct_ce = memory_stats.get(
            "scene_state_full_correct_ce",
            0.0,
        )
        self._last_scene_state_correct_all_semantic_ce = memory_stats.get(
            "scene_state_correct_all_semantic_ce",
            0.0,
        )
        self._last_scene_state_correct_pair_semantic_ce = memory_stats.get(
            "scene_state_correct_pair_semantic_ce",
            0.0,
        )
        self._last_scene_state_donor_pair_semantic_ce = memory_stats.get(
            "scene_state_donor_pair_semantic_ce",
            0.0,
        )
        self._last_scene_state_zero_all_semantic_ce = memory_stats.get(
            "scene_state_zero_all_semantic_ce",
            0.0,
        )
        self._last_scene_state_donor_pair_gap = memory_stats.get(
            "scene_state_donor_pair_gap",
            0.0,
        )
        self._last_scene_state_zero_all_gap = memory_stats.get(
            "scene_state_zero_all_gap",
            0.0,
        )
        self._last_scene_state_donor_margin_loss = memory_stats.get(
            "scene_state_donor_margin_loss",
            0.0,
        )
        self._last_scene_state_donor_positive_fraction = memory_stats.get(
            "scene_state_donor_positive_fraction",
            0.0,
        )
        self._last_scene_state_zero_positive_fraction = memory_stats.get(
            "scene_state_zero_positive_fraction",
            0.0,
        )
        self._last_scene_state_semantic_token_count = memory_stats.get(
            "scene_state_semantic_token_count",
            0.0,
        )
        self._last_scene_state_semantic_row_count = memory_stats.get(
            "scene_state_semantic_row_count",
            0.0,
        )
        self._last_scene_state_target_presence_row_count = memory_stats.get(
            "scene_state_target_presence_row_count",
            0.0,
        )
        self._last_scene_state_target_same_cardinality_value_row_count = (
            memory_stats.get(
                "scene_state_target_same_cardinality_value_row_count",
                0.0,
            )
        )
        self._last_scene_state_target_cross_cardinality_value_row_count = (
            memory_stats.get(
                "scene_state_target_cross_cardinality_value_row_count",
                0.0,
            )
        )
        self._last_scene_generation_total_loss = memory_stats.get(
            "scene_generation_total_loss",
            0.0,
        )
        self._last_scene_generation_weighted_ce = memory_stats.get(
            "scene_generation_weighted_ce",
            0.0,
        )
        self._last_scene_generation_schema_ce = memory_stats.get(
            "scene_generation_schema_ce",
            0.0,
        )
        self._last_scene_generation_decision_ce = memory_stats.get(
            "scene_generation_decision_ce",
            0.0,
        )
        self._last_scene_generation_termination_ce = memory_stats.get(
            "scene_generation_termination_ce",
            0.0,
        )
        self._last_scene_generation_first_error_loss = memory_stats.get(
            "scene_generation_first_error_loss",
            0.0,
        )
        self._last_scene_generation_pair_correct_ce = memory_stats.get(
            "scene_generation_pair_correct_ce",
            0.0,
        )
        self._last_scene_generation_pair_donor_ce = memory_stats.get(
            "scene_generation_pair_donor_ce",
            0.0,
        )
        self._last_scene_generation_zero_margin_loss = memory_stats.get(
            "scene_generation_zero_margin_loss",
            0.0,
        )
        self._last_scene_generation_generated_unlikelihood_loss = memory_stats.get(
            "scene_generation_generated_unlikelihood_loss",
            0.0,
        )
        self._last_scene_generation_generated_unlikelihood_weighted_loss = (
            memory_stats.get(
                "scene_generation_generated_unlikelihood_weighted_loss",
                0.0,
            )
        )
        self._last_scene_generation_generated_unlikelihood_applied = memory_stats.get(
            "scene_generation_generated_unlikelihood_applied",
            0.0,
        )
        self._last_scene_generation_generated_wrong_token_count = memory_stats.get(
            "scene_generation_generated_wrong_token_count",
            0.0,
        )
        self._last_scene_generation_generated_rollout_token_count = memory_stats.get(
            "scene_generation_generated_rollout_token_count",
            0.0,
        )
        self._last_scene_generation_generated_first_divergence = memory_stats.get(
            "scene_generation_generated_first_divergence",
            0.0,
        )
        self._last_scene_generation_generated_exact_fraction = memory_stats.get(
            "scene_generation_generated_exact_fraction",
            0.0,
        )
        self._last_scene_generation_gold_top1_accuracy = memory_stats.get(
            "scene_generation_gold_top1_accuracy",
            0.0,
        )
        self._last_scene_generation_first_error_ordinal = memory_stats.get(
            "scene_generation_first_error_ordinal",
            0.0,
        )
        self._last_scene_generation_solved_fraction = memory_stats.get(
            "scene_generation_solved_fraction",
            0.0,
        )
        self._last_scene_generation_correct_decision_margin = memory_stats.get(
            "scene_generation_correct_decision_margin",
            0.0,
        )
        self._last_scene_generation_donor_decision_margin = memory_stats.get(
            "scene_generation_donor_decision_margin",
            0.0,
        )
        self._last_scene_generation_zero_decision_margin = memory_stats.get(
            "scene_generation_zero_decision_margin",
            0.0,
        )
        self._last_scene_generation_correct_pair_preference = memory_stats.get(
            "scene_generation_correct_pair_preference",
            0.0,
        )
        self._last_scene_generation_donor_pair_preference = memory_stats.get(
            "scene_generation_donor_pair_preference",
            0.0,
        )
        self._last_scene_generation_target_token_count = memory_stats.get(
            "scene_generation_target_token_count",
            0.0,
        )
        self._last_scene_generation_content_token_count = memory_stats.get(
            "scene_generation_content_token_count",
            0.0,
        )
        self._last_scene_generation_schema_token_count = memory_stats.get(
            "scene_generation_schema_token_count",
            0.0,
        )
        self._last_scene_generation_decision_token_count = memory_stats.get(
            "scene_generation_decision_token_count",
            0.0,
        )
        self._last_scene_generation_termination_token_count = memory_stats.get(
            "scene_generation_termination_token_count",
            0.0,
        )
        self._last_memory_wmem = memory_stats["wmem"]
        self._last_memory_probe_keep_loss = memory_stats["probe_keep_loss"]
        self._last_memory_probe_reset_loss = memory_stats["probe_reset_loss"]
        self._last_memory_probe_margin_loss = memory_stats["probe_margin_loss"]
        self._last_memory_probe_gap = memory_stats["probe_gap"]
        self._last_memory_probe_kl_loss = memory_stats["probe_kl"]
        self._last_memory_probe_ce_loss = memory_stats["probe_ce"]

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
        content_contrast_target_mask = model_inputs.pop(
            "content_contrast_target_mask",
            None,
        )
        scene_state_donor_write_input_ids = model_inputs.pop(
            "scene_state_donor_write_input_ids",
            None,
        )
        scene_state_donor_write_attention_mask = model_inputs.pop(
            "scene_state_donor_write_attention_mask",
            None,
        )
        scene_state_donor_write_message_ids = model_inputs.pop(
            "scene_state_donor_write_message_ids",
            None,
        )
        scene_state_donor_write_sentence_ids = model_inputs.pop(
            "scene_state_donor_write_sentence_ids",
            None,
        )
        scene_state_semantic_mask = model_inputs.pop(
            "scene_state_semantic_mask",
            None,
        )
        scene_state_identity_target_mask = model_inputs.pop(
            "scene_state_identity_target_mask",
            None,
        )
        scene_state_identity_target_stratum = model_inputs.pop(
            "scene_state_identity_target_stratum",
            None,
        )
        scene_state_identity_donor_target_token_id = model_inputs.pop(
            "scene_state_identity_donor_target_token_id",
            None,
        )
        scene_state_generation_target_mask = model_inputs.pop(
            "scene_state_generation_target_mask",
            None,
        )
        scene_state_generation_content_mask = model_inputs.pop(
            "scene_state_generation_content_mask",
            None,
        )
        scene_state_generation_schema_mask = model_inputs.pop(
            "scene_state_generation_schema_mask",
            None,
        )
        scene_state_generation_decision_mask = model_inputs.pop(
            "scene_state_generation_decision_mask",
            None,
        )
        scene_state_generation_termination_mask = model_inputs.pop(
            "scene_state_generation_termination_mask",
            None,
        )
        state_only_write_input_ids = model_inputs.pop("state_only_write_input_ids", None)
        state_only_write_attention_mask = model_inputs.pop("state_only_write_attention_mask", None)
        state_only_write_message_ids = model_inputs.pop("state_only_write_message_ids", None)
        state_only_write_sentence_ids = model_inputs.pop("state_only_write_sentence_ids", None)
        state_only_input_ids = model_inputs.pop("state_only_input_ids", None)
        state_only_attention_mask = model_inputs.pop("state_only_attention_mask", None)
        state_only_labels = model_inputs.pop("state_only_labels", None)
        scene_boundary_payload_mask = model_inputs.pop(
            "scene_boundary_payload_mask",
            None,
        )
        state_only_scene_boundary_payload_mask = model_inputs.pop(
            "state_only_scene_boundary_payload_mask",
            None,
        )
        model_inputs.pop("teacher_scene_boundary_payload_mask", None)
        model_inputs.pop("full_scene_boundary_payload_mask", None)
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
        if getattr(self, "memory_loss_mode", None) == "scene_state_generation_ce":
            required_scene_generation_values = {
                "write_input_ids": write_input_ids,
                "write_attention_mask": write_attention_mask,
                "scene_state_donor_write_input_ids": (
                    scene_state_donor_write_input_ids
                ),
                "scene_state_donor_write_attention_mask": (
                    scene_state_donor_write_attention_mask
                ),
                "scene_state_identity_target_mask": (
                    scene_state_identity_target_mask
                ),
                "scene_state_identity_target_stratum": (
                    scene_state_identity_target_stratum
                ),
                "scene_state_identity_donor_target_token_id": (
                    scene_state_identity_donor_target_token_id
                ),
                "scene_state_generation_target_mask": (
                    scene_state_generation_target_mask
                ),
                "scene_state_generation_content_mask": (
                    scene_state_generation_content_mask
                ),
                "scene_state_generation_schema_mask": (
                    scene_state_generation_schema_mask
                ),
                "scene_state_generation_decision_mask": (
                    scene_state_generation_decision_mask
                ),
                "scene_state_generation_termination_mask": (
                    scene_state_generation_termination_mask
                ),
            }
            missing_scene_generation_values = [
                key
                for key, value in required_scene_generation_values.items()
                if value is None
            ]
            if missing_scene_generation_values:
                raise ValueError(
                    "scene_state_generation_ce requires exact generation masks and "
                    "predeclared donor targets: "
                    + ", ".join(missing_scene_generation_values)
                )
            loss, outputs, memory_stats = self._compute_scene_state_generation_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                donor_write_input_ids=scene_state_donor_write_input_ids,
                donor_write_attention_mask=scene_state_donor_write_attention_mask,
                donor_write_message_ids=scene_state_donor_write_message_ids,
                donor_write_sentence_ids=scene_state_donor_write_sentence_ids,
                target_mask=scene_state_generation_target_mask,
                content_mask=scene_state_generation_content_mask,
                schema_mask=scene_state_generation_schema_mask,
                decision_mask=scene_state_generation_decision_mask,
                termination_mask=scene_state_generation_termination_mask,
                pair_target_mask=scene_state_identity_target_mask,
                donor_target_token_ids=(
                    scene_state_identity_donor_target_token_id
                ),
                target_stratum_codes=scene_state_identity_target_stratum,
            )
        elif getattr(self, "memory_loss_mode", None) == "scene_state_identity_ce":
            required_scene_state_values = {
                "write_input_ids": write_input_ids,
                "write_attention_mask": write_attention_mask,
                "scene_state_donor_write_input_ids": (
                    scene_state_donor_write_input_ids
                ),
                "scene_state_donor_write_attention_mask": (
                    scene_state_donor_write_attention_mask
                ),
                "scene_state_semantic_mask": scene_state_semantic_mask,
                "scene_state_identity_target_mask": (
                    scene_state_identity_target_mask
                ),
                "scene_state_identity_target_stratum": (
                    scene_state_identity_target_stratum
                ),
            }
            missing_scene_state_values = [
                key
                for key, value in required_scene_state_values.items()
                if value is None
            ]
            if missing_scene_state_values:
                raise ValueError(
                    "scene_state_identity_ce requires correct/donor writes and audited "
                    "semantic target masks: " + ", ".join(missing_scene_state_values)
                )
            if scene_state_semantic_mask.shape != scene_state_identity_target_mask.shape:
                raise ValueError(
                    "Scene-state semantic and identity target masks must align"
                )
            if bool(
                (
                    scene_state_identity_target_mask.to(dtype=torch.bool)
                    & ~scene_state_semantic_mask.to(dtype=torch.bool)
                ).any()
            ):
                raise ValueError(
                    "Scene-state identity target mask must be a subset of the semantic mask"
                )
            loss, outputs, memory_stats = self._compute_scene_state_identity_ce(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=write_input_ids,
                write_attention_mask=write_attention_mask,
                write_message_ids=write_message_ids,
                write_sentence_ids=write_sentence_ids,
                donor_write_input_ids=scene_state_donor_write_input_ids,
                donor_write_attention_mask=scene_state_donor_write_attention_mask,
                donor_write_message_ids=scene_state_donor_write_message_ids,
                donor_write_sentence_ids=scene_state_donor_write_sentence_ids,
                semantic_mask=scene_state_semantic_mask,
                pair_target_mask=scene_state_identity_target_mask,
                target_stratum_codes=scene_state_identity_target_stratum,
            )
        elif self.memory_loss_mode == "content_contrast_ce":
            if (
                write_input_ids is None
                or write_attention_mask is None
                or negative_write_input_ids is None
                or negative_write_attention_mask is None
                or content_contrast_target_mask is None
            ):
                raise ValueError(
                    "content_contrast_ce requires materialized positive/negative writes "
                    "and a strict target mask"
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
                content_contrast_target_mask=content_contrast_target_mask,
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
                scene_boundary_payload_mask=scene_boundary_payload_mask,
                state_only_scene_boundary_payload_mask=(
                    state_only_scene_boundary_payload_mask
                ),
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
        self._record_memory_stats(model, memory_stats)
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
        if getattr(self, "scene_boundary_payload_ce_weight", 0.0) > 0.0:
            enriched_logs.update(
                {
                    "delta/scene_boundary_full_ce_loss": (
                        self._last_scene_boundary_full_ce_loss
                    ),
                    "delta/scene_boundary_payload_ce_loss": (
                        self._last_scene_boundary_payload_ce_loss
                    ),
                    "delta/scene_boundary_payload_auxiliary_loss": (
                        self._last_scene_boundary_payload_auxiliary_loss
                    ),
                    "delta/scene_boundary_payload_token_count": (
                        self._last_scene_boundary_payload_token_count
                    ),
                    "delta/scene_boundary_supervised_token_count": (
                        self._last_scene_boundary_supervised_token_count
                    ),
                }
            )
        if getattr(self, "memory_loss_mode", None) == "scene_state_identity_ce":
            enriched_logs.update(
                {
                    "delta/scene_state_full_correct_ce": (
                        self._last_scene_state_full_correct_ce
                    ),
                    "delta/scene_state_correct_all_semantic_ce": (
                        self._last_scene_state_correct_all_semantic_ce
                    ),
                    "delta/scene_state_correct_pair_semantic_ce": (
                        self._last_scene_state_correct_pair_semantic_ce
                    ),
                    "delta/scene_state_donor_pair_semantic_ce": (
                        self._last_scene_state_donor_pair_semantic_ce
                    ),
                    "delta/scene_state_zero_all_semantic_ce": (
                        self._last_scene_state_zero_all_semantic_ce
                    ),
                    "delta/scene_state_donor_pair_gap": (
                        self._last_scene_state_donor_pair_gap
                    ),
                    "delta/scene_state_zero_all_gap": (
                        self._last_scene_state_zero_all_gap
                    ),
                    "delta/scene_state_donor_margin_loss": (
                        self._last_scene_state_donor_margin_loss
                    ),
                    "delta/scene_state_donor_positive_fraction": (
                        self._last_scene_state_donor_positive_fraction
                    ),
                    "delta/scene_state_zero_positive_fraction": (
                        self._last_scene_state_zero_positive_fraction
                    ),
                    "delta/scene_state_semantic_token_count": (
                        self._last_scene_state_semantic_token_count
                    ),
                    "delta/scene_state_semantic_row_count": (
                        self._last_scene_state_semantic_row_count
                    ),
                    "delta/scene_state_target_presence_row_count": (
                        self._last_scene_state_target_presence_row_count
                    ),
                    "delta/scene_state_target_same_cardinality_value_row_count": (
                        self._last_scene_state_target_same_cardinality_value_row_count
                    ),
                    "delta/scene_state_target_cross_cardinality_value_row_count": (
                        self._last_scene_state_target_cross_cardinality_value_row_count
                    ),
                }
            )
        if getattr(self, "memory_loss_mode", None) == "scene_state_generation_ce":
            enriched_logs.update(
                {
                    "delta/scene_generation_total_loss": (
                        self._last_scene_generation_total_loss
                    ),
                    "delta/scene_generation_weighted_ce": (
                        self._last_scene_generation_weighted_ce
                    ),
                    "delta/scene_generation_schema_ce": (
                        self._last_scene_generation_schema_ce
                    ),
                    "delta/scene_generation_decision_ce": (
                        self._last_scene_generation_decision_ce
                    ),
                    "delta/scene_generation_termination_ce": (
                        self._last_scene_generation_termination_ce
                    ),
                    "delta/scene_generation_first_error_loss": (
                        self._last_scene_generation_first_error_loss
                    ),
                    "delta/scene_generation_pair_correct_ce": (
                        self._last_scene_generation_pair_correct_ce
                    ),
                    "delta/scene_generation_pair_donor_ce": (
                        self._last_scene_generation_pair_donor_ce
                    ),
                    "delta/scene_generation_zero_margin_loss": (
                        self._last_scene_generation_zero_margin_loss
                    ),
                    "delta/scene_generation_generated_unlikelihood_loss": (
                        self._last_scene_generation_generated_unlikelihood_loss
                    ),
                    "delta/scene_generation_generated_unlikelihood_weighted_loss": (
                        self._last_scene_generation_generated_unlikelihood_weighted_loss
                    ),
                    "delta/scene_generation_generated_unlikelihood_applied": (
                        self._last_scene_generation_generated_unlikelihood_applied
                    ),
                    "delta/scene_generation_generated_wrong_token_count": (
                        self._last_scene_generation_generated_wrong_token_count
                    ),
                    "delta/scene_generation_generated_rollout_token_count": (
                        self._last_scene_generation_generated_rollout_token_count
                    ),
                    "delta/scene_generation_generated_first_divergence": (
                        self._last_scene_generation_generated_first_divergence
                    ),
                    "delta/scene_generation_generated_exact_fraction": (
                        self._last_scene_generation_generated_exact_fraction
                    ),
                    "delta/scene_generation_gold_top1_accuracy": (
                        self._last_scene_generation_gold_top1_accuracy
                    ),
                    "delta/scene_generation_first_error_ordinal": (
                        self._last_scene_generation_first_error_ordinal
                    ),
                    "delta/scene_generation_solved_fraction": (
                        self._last_scene_generation_solved_fraction
                    ),
                    "delta/scene_generation_correct_decision_margin": (
                        self._last_scene_generation_correct_decision_margin
                    ),
                    "delta/scene_generation_donor_decision_margin": (
                        self._last_scene_generation_donor_decision_margin
                    ),
                    "delta/scene_generation_zero_decision_margin": (
                        self._last_scene_generation_zero_decision_margin
                    ),
                    "delta/scene_generation_correct_pair_preference": (
                        self._last_scene_generation_correct_pair_preference
                    ),
                    "delta/scene_generation_donor_pair_preference": (
                        self._last_scene_generation_donor_pair_preference
                    ),
                    "delta/scene_generation_target_token_count": (
                        self._last_scene_generation_target_token_count
                    ),
                    "delta/scene_generation_content_token_count": (
                        self._last_scene_generation_content_token_count
                    ),
                    "delta/scene_generation_schema_token_count": (
                        self._last_scene_generation_schema_token_count
                    ),
                    "delta/scene_generation_decision_token_count": (
                        self._last_scene_generation_decision_token_count
                    ),
                    "delta/scene_generation_termination_token_count": (
                        self._last_scene_generation_termination_token_count
                    ),
                    "delta/scene_generation_target_presence_row_count": (
                        self._last_scene_state_target_presence_row_count
                    ),
                    "delta/scene_generation_target_same_cardinality_value_row_count": (
                        self._last_scene_state_target_same_cardinality_value_row_count
                    ),
                    "delta/scene_generation_target_cross_cardinality_value_row_count": (
                        self._last_scene_state_target_cross_cardinality_value_row_count
                    ),
                }
            )
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
                    "delta/memory_representation_loss": (
                        self._last_memory_representation_loss
                    ),
                    "delta/memory_representation_distance": (
                        self._last_memory_representation_distance
                    ),
                    "delta/content_contrast_full_correct_ce": (
                        self._last_content_contrast_full_correct_ce
                    ),
                    "delta/content_contrast_full_donor_ce": (
                        self._last_content_contrast_full_donor_ce
                    ),
                    "delta/content_contrast_targeted_correct_ce": (
                        self._last_content_contrast_targeted_correct_ce
                    ),
                    "delta/content_contrast_targeted_donor_ce": (
                        self._last_content_contrast_targeted_donor_ce
                    ),
                    "delta/content_contrast_targeted_gap": (
                        self._last_content_contrast_targeted_gap
                    ),
                    "delta/content_contrast_targeted_positive_fraction": (
                        self._last_content_contrast_targeted_positive_fraction
                    ),
                    "delta/content_contrast_targeted_token_count": (
                        self._last_content_contrast_targeted_token_count
                    ),
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
        if self.memory_loss_mode == "scene_state_generation_ce":
            return self._scene_state_generation_sequential_training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        if self.memory_loss_mode == "scene_state_identity_ce":
            return self._scene_state_identity_sequential_training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        if self.memory_loss_mode == "content_contrast_ce":
            return self._content_contrast_sequential_training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def _scene_state_generation_sequential_training_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        *,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            prepared_inputs = self._prepare_inputs(inputs)
            model_inputs = dict(prepared_inputs)
            payload = {
                key: model_inputs.pop(key, None)
                for key in (
                    "write_input_ids",
                    "write_attention_mask",
                    "write_message_ids",
                    "write_sentence_ids",
                    "scene_state_donor_write_input_ids",
                    "scene_state_donor_write_attention_mask",
                    "scene_state_donor_write_message_ids",
                    "scene_state_donor_write_sentence_ids",
                    "scene_state_identity_target_mask",
                    "scene_state_identity_target_stratum",
                    "scene_state_identity_donor_target_token_id",
                    "scene_state_generation_target_mask",
                    "scene_state_generation_content_mask",
                    "scene_state_generation_schema_mask",
                    "scene_state_generation_decision_mask",
                    "scene_state_generation_termination_mask",
                )
            }
            for key in (
                "scene_state_semantic_mask",
                "state_only_write_input_ids",
                "state_only_write_attention_mask",
                "state_only_write_message_ids",
                "state_only_write_sentence_ids",
                "state_only_input_ids",
                "state_only_attention_mask",
                "state_only_labels",
                "scene_boundary_payload_mask",
                "state_only_scene_boundary_payload_mask",
                "teacher_scene_boundary_payload_mask",
                "full_scene_boundary_payload_mask",
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_labels",
                "full_input_ids",
                "full_attention_mask",
                "full_labels",
                "write_lengths",
                "read_lengths",
            ):
                model_inputs.pop(key, None)
            required = (
                "write_input_ids",
                "write_attention_mask",
                "scene_state_donor_write_input_ids",
                "scene_state_donor_write_attention_mask",
                "scene_state_identity_target_mask",
                "scene_state_identity_target_stratum",
                "scene_state_identity_donor_target_token_id",
                "scene_state_generation_target_mask",
                "scene_state_generation_content_mask",
                "scene_state_generation_schema_mask",
                "scene_state_generation_decision_mask",
                "scene_state_generation_termination_mask",
            )
            missing = [key for key in required if payload[key] is None]
            if missing:
                raise ValueError(
                    "scene_state_generation_ce requires exact generation and pairing "
                    "metadata: " + ", ".join(missing)
                )
            loss_kwargs = {}
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            gradient_scale = 1.0
            if (
                (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
                and self.compute_loss_func is None
            ):
                gradient_scale /= self.current_gradient_accumulation_steps
            loss, memory_stats = self._scene_state_generation_sequential_backward(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=payload["write_input_ids"],
                write_attention_mask=payload["write_attention_mask"],
                write_message_ids=payload["write_message_ids"],
                write_sentence_ids=payload["write_sentence_ids"],
                donor_write_input_ids=payload[
                    "scene_state_donor_write_input_ids"
                ],
                donor_write_attention_mask=payload[
                    "scene_state_donor_write_attention_mask"
                ],
                donor_write_message_ids=payload[
                    "scene_state_donor_write_message_ids"
                ],
                donor_write_sentence_ids=payload[
                    "scene_state_donor_write_sentence_ids"
                ],
                target_mask=payload["scene_state_generation_target_mask"],
                content_mask=payload["scene_state_generation_content_mask"],
                schema_mask=payload["scene_state_generation_schema_mask"],
                decision_mask=payload["scene_state_generation_decision_mask"],
                termination_mask=payload[
                    "scene_state_generation_termination_mask"
                ],
                pair_target_mask=payload["scene_state_identity_target_mask"],
                donor_target_token_ids=payload[
                    "scene_state_identity_donor_target_token_id"
                ],
                target_stratum_codes=payload[
                    "scene_state_identity_target_stratum"
                ],
                gradient_scale=gradient_scale,
            )
            self._last_memory_partition_alignment_loss = 0.0
            self._last_memory_partition_entropy_loss = 0.0
            self._last_memory_partition_balance_loss = 0.0
            self._last_write_sparsity_loss = 0.0
            self._record_memory_stats(model, memory_stats)
            del prepared_inputs, model_inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            return loss.detach()

    def _scene_state_identity_sequential_training_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        *,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            prepared_inputs = self._prepare_inputs(inputs)
            model_inputs = dict(prepared_inputs)
            payload = {
                key: model_inputs.pop(key, None)
                for key in (
                    "write_input_ids",
                    "write_attention_mask",
                    "write_message_ids",
                    "write_sentence_ids",
                    "scene_state_donor_write_input_ids",
                    "scene_state_donor_write_attention_mask",
                    "scene_state_donor_write_message_ids",
                    "scene_state_donor_write_sentence_ids",
                    "scene_state_semantic_mask",
                    "scene_state_identity_target_mask",
                    "scene_state_identity_target_stratum",
                )
            }
            for key in (
                "state_only_write_input_ids",
                "state_only_write_attention_mask",
                "state_only_write_message_ids",
                "state_only_write_sentence_ids",
                "state_only_input_ids",
                "state_only_attention_mask",
                "state_only_labels",
                "scene_boundary_payload_mask",
                "state_only_scene_boundary_payload_mask",
                "teacher_scene_boundary_payload_mask",
                "full_scene_boundary_payload_mask",
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_labels",
                "full_input_ids",
                "full_attention_mask",
                "full_labels",
                "write_lengths",
                "read_lengths",
            ):
                model_inputs.pop(key, None)

            required = (
                "write_input_ids",
                "write_attention_mask",
                "scene_state_donor_write_input_ids",
                "scene_state_donor_write_attention_mask",
                "scene_state_semantic_mask",
                "scene_state_identity_target_mask",
                "scene_state_identity_target_stratum",
            )
            missing = [key for key in required if payload[key] is None]
            if missing:
                raise ValueError(
                    "scene_state_identity_ce requires correct/donor writes and audited "
                    "semantic target masks: " + ", ".join(missing)
                )
            semantic_mask = payload["scene_state_semantic_mask"].to(
                dtype=torch.bool
            )
            target_mask = payload["scene_state_identity_target_mask"].to(
                dtype=torch.bool
            )
            if semantic_mask.shape != target_mask.shape:
                raise ValueError(
                    "Scene-state semantic and identity target masks must align"
                )
            if bool((target_mask & ~semantic_mask).any()):
                raise ValueError(
                    "Scene-state identity target mask must be a subset of the semantic mask"
                )

            loss_kwargs = {}
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            gradient_scale = 1.0
            if (
                (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
                and self.compute_loss_func is None
            ):
                gradient_scale /= self.current_gradient_accumulation_steps
            loss, memory_stats = self._scene_state_identity_sequential_backward(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=payload["write_input_ids"],
                write_attention_mask=payload["write_attention_mask"],
                write_message_ids=payload["write_message_ids"],
                write_sentence_ids=payload["write_sentence_ids"],
                donor_write_input_ids=payload[
                    "scene_state_donor_write_input_ids"
                ],
                donor_write_attention_mask=payload[
                    "scene_state_donor_write_attention_mask"
                ],
                donor_write_message_ids=payload[
                    "scene_state_donor_write_message_ids"
                ],
                donor_write_sentence_ids=payload[
                    "scene_state_donor_write_sentence_ids"
                ],
                semantic_mask=semantic_mask,
                pair_target_mask=target_mask,
                target_stratum_codes=payload[
                    "scene_state_identity_target_stratum"
                ],
                gradient_scale=gradient_scale,
            )
            self._last_memory_partition_alignment_loss = 0.0
            self._last_memory_partition_entropy_loss = 0.0
            self._last_memory_partition_balance_loss = 0.0
            self._last_write_sparsity_loss = 0.0
            self._record_memory_stats(model, memory_stats)
            del prepared_inputs, model_inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            return loss.detach()

    def _content_contrast_sequential_training_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        *,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            prepared_inputs = self._prepare_inputs(inputs)
            model_inputs = dict(prepared_inputs)
            payload = {
                key: model_inputs.pop(key, None)
                for key in (
                    "write_input_ids",
                    "write_attention_mask",
                    "write_message_ids",
                    "write_sentence_ids",
                    "negative_write_input_ids",
                    "negative_write_attention_mask",
                    "negative_write_message_ids",
                    "negative_write_sentence_ids",
                    "content_contrast_target_mask",
                )
            }
            for key in (
                "state_only_write_input_ids",
                "state_only_write_attention_mask",
                "state_only_write_message_ids",
                "state_only_write_sentence_ids",
                "state_only_input_ids",
                "state_only_attention_mask",
                "state_only_labels",
                "scene_boundary_payload_mask",
                "state_only_scene_boundary_payload_mask",
                "teacher_scene_boundary_payload_mask",
                "full_scene_boundary_payload_mask",
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_labels",
                "full_input_ids",
                "full_attention_mask",
                "full_labels",
                "write_lengths",
                "read_lengths",
            ):
                model_inputs.pop(key, None)

            required = (
                "write_input_ids",
                "write_attention_mask",
                "negative_write_input_ids",
                "negative_write_attention_mask",
                "content_contrast_target_mask",
            )
            missing = [key for key in required if payload[key] is None]
            if missing:
                raise ValueError(
                    "content_contrast_ce requires materialized positive/negative writes "
                    "and a strict target mask: " + ", ".join(missing)
                )

            loss_kwargs = {}
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            gradient_scale = 1.0
            if (
                (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
                and self.compute_loss_func is None
            ):
                gradient_scale /= self.current_gradient_accumulation_steps

            loss, memory_stats = self._content_contrast_sequential_backward(
                model,
                model_inputs,
                loss_kwargs=loss_kwargs,
                write_input_ids=payload["write_input_ids"],
                write_attention_mask=payload["write_attention_mask"],
                write_message_ids=payload["write_message_ids"],
                write_sentence_ids=payload["write_sentence_ids"],
                negative_write_input_ids=payload["negative_write_input_ids"],
                negative_write_attention_mask=payload["negative_write_attention_mask"],
                negative_write_message_ids=payload["negative_write_message_ids"],
                negative_write_sentence_ids=payload["negative_write_sentence_ids"],
                gradient_scale=gradient_scale,
                content_contrast_target_mask=payload[
                    "content_contrast_target_mask"
                ],
            )
            self._last_memory_partition_alignment_loss = 0.0
            self._last_memory_partition_entropy_loss = 0.0
            self._last_memory_partition_balance_loss = 0.0
            self._last_write_sparsity_loss = 0.0
            self._record_memory_stats(model, memory_stats)
            del prepared_inputs, model_inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
            return loss.detach()

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
        scene_state_identity_pairing_manifest = getattr(
            self,
            "scene_state_identity_pairing_manifest",
            None,
        )
        if scene_state_identity_pairing_manifest is not None:
            (output_path / _SCENE_STATE_IDENTITY_PAIRING_FILENAME).write_text(
                json.dumps(
                    scene_state_identity_pairing_manifest,
                    indent=2,
                    sort_keys=True,
                )
            )
        continuation_manifest = _training_lineage_summary(self)["continuation"]
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
        if expected_pairing is not None and active_resume_mode != "objective_ablation":
            pairing_path = checkpoint / _CONTENT_CONTRAST_PAIRING_FILENAME
            if not pairing_path.is_file():
                raise ValueError(
                    f"Delta-Mem checkpoint is missing {_CONTENT_CONTRAST_PAIRING_FILENAME}: "
                    f"{checkpoint}"
                )
            actual_pairing = json.loads(pairing_path.read_text())
            if actual_pairing != expected_pairing:
                raise ValueError(
                    "Delta-Mem checkpoint content-contrast pairing manifest does not match"
                )
        expected_scene_pairing = getattr(
            self,
            "scene_state_identity_pairing_manifest",
            None,
        )
        if expected_scene_pairing is not None:
            scene_pairing_path = checkpoint / _SCENE_STATE_IDENTITY_PAIRING_FILENAME
            if not scene_pairing_path.is_file():
                raise ValueError(
                    "Delta-Mem checkpoint is missing "
                    f"{_SCENE_STATE_IDENTITY_PAIRING_FILENAME}: {checkpoint}"
                )
            actual_scene_pairing = json.loads(scene_pairing_path.read_text())
            if actual_scene_pairing != expected_scene_pairing:
                raise ValueError(
                    "Delta-Mem checkpoint scene-state pairing manifest does not match"
                )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        active_resume_mode = getattr(self, "resume_mode", "exact")
        checkpoint = _validate_resume_checkpoint(
            Path(resume_from_checkpoint),
            require_training_protocol=(getattr(self, "training_protocol", None) is not None),
            require_content_contrast_pairing=(
                getattr(self, "content_contrast_pairing_manifest", None) is not None
                and active_resume_mode != "objective_ablation"
            ),
            require_scene_state_identity_pairing=(
                getattr(self, "scene_state_identity_pairing_manifest", None)
                is not None
            ),
        )
        checkpoint_steps = _scene_memory_v8_protocol_checkpoint_steps(
            getattr(self, "training_protocol", None)
        )
        if checkpoint_steps is not None:
            validate_scene_memory_v8_active_continuation(
                getattr(self, "continuation_manifest", None),
                resume_from_checkpoint=checkpoint,
                target_training_protocol=self.training_protocol,
                checkpoint_steps=checkpoint_steps,
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


def _scene_state_source_manifest_identity(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    manifest_path = getattr(args, "scene_state_source_manifest", None)
    expected_sha256 = getattr(
        args,
        "expected_scene_state_source_manifest_sha256",
        None,
    )
    if manifest_path is None and expected_sha256 is None:
        return None
    if manifest_path is None or expected_sha256 is None:
        raise ValueError(
            "--scene-state-source-manifest and "
            "--expected-scene-state-source-manifest-sha256 must be provided together"
        )
    manifest_path = Path(manifest_path).expanduser().resolve()
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(
            "--expected-scene-state-source-manifest-sha256 must be exactly 64 "
            "hexadecimal characters"
        )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Scene-state source manifest is invalid: {manifest_path}")
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Scene-state source manifest SHA-256 differs from the launch lock: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    manifest = _load_json_object(
        manifest_path,
        description="scene-state source manifest",
    )
    partitions = manifest.get("partitions")
    train_partition = (
        None if not isinstance(partitions, dict) else partitions.get("train")
    )
    train_data = (
        None if not isinstance(train_partition, dict) else train_partition.get("data")
    )
    if not isinstance(train_data, dict):
        raise ValueError("Scene-state source manifest omits partitions.train.data")
    contract = manifest.get("contract")
    episode_contract = (
        None if not isinstance(contract, dict) else contract.get("episode_contract")
    )
    if not isinstance(episode_contract, dict):
        raise ValueError("Scene-state source manifest omits contract.episode_contract")
    required_episode_contract = {
        "episode_recent_messages": 0,
        "write_phase": "system + user",
        "read_supervision": "system + assistant",
    }
    episode_contract_mismatches = {
        key: episode_contract.get(key)
        for key, expected in required_episode_contract.items()
        if episode_contract.get(key) != expected
    }
    if episode_contract_mismatches:
        raise ValueError(
            "Scene-state source manifest has an incompatible episode contract: "
            f"{episode_contract_mismatches}"
        )
    train_path_raw = train_data.get("path")
    train_sha256 = train_data.get("sha256")
    train_file = getattr(args, "train_file", None)
    if train_file is None:
        raise ValueError("Scene-state source manifest requires --train-file")
    train_path = Path(str(train_path_raw)).expanduser().resolve()
    resolved_train_file = Path(train_file).expanduser().resolve()
    if train_path != resolved_train_file:
        raise ValueError(
            "Scene-state source manifest train path differs from --train-file"
        )
    actual_train_sha256 = _sha256_file(resolved_train_file)
    if train_sha256 != actual_train_sha256:
        raise ValueError(
            "Scene-state source manifest train SHA-256 differs from --train-file"
        )
    return {
        "path": str(manifest_path),
        "file_sha256": actual_sha256,
        "schema": manifest.get("schema"),
        "train_file": str(resolved_train_file),
        "train_file_sha256": actual_train_sha256,
        "train_rows": train_partition.get("rows", train_data.get("rows")),
        "train_source_split": train_partition.get("source_split"),
        "episode_contract": {
            key: episode_contract[key]
            for key in (
                "episode_recent_messages",
                "write_phase",
                "read_supervision",
            )
        },
    }


def _scene_state_v8_curriculum_binding(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    source_identity = _scene_state_source_manifest_identity(args)
    if source_identity is None or source_identity.get("schema") != (
        _SCENE_MEMORY_V8_SOURCE_SCHEMA
    ):
        return None
    source_manifest_path = Path(str(source_identity["path"]))
    source_manifest = _load_json_object(
        source_manifest_path,
        description="scene-state V8 source manifest",
    )
    unsigned_source = dict(source_manifest)
    declared_source_hash = unsigned_source.pop("manifest_sha256", None)
    if declared_source_hash != _canonical_json_sha256(unsigned_source):
        raise ValueError("Scene-state V8 source-manifest canonical SHA-256 differs")
    curriculum = source_manifest.get("v8_curriculum")
    if not isinstance(curriculum, dict) or curriculum.get("schema") != (
        _SCENE_MEMORY_V8_CURRICULUM_SCHEMA
    ):
        raise ValueError("Scene-state V8 source manifest has no valid curriculum")
    if curriculum.get("parent_train32_sha256") != source_identity.get(
        "train_file_sha256"
    ):
        raise ValueError("Scene-state V8 curriculum binds a different Train32")
    train_rows = source_identity.get("train_rows")
    if isinstance(train_rows, bool) or not isinstance(train_rows, int) or train_rows <= 0:
        raise ValueError("Scene-state V8 source train-row count is invalid")

    def resolve_artifact(
        record: object,
        *,
        description: str,
    ) -> tuple[Path, str]:
        if not isinstance(record, dict):
            raise ValueError(f"Scene-state V8 curriculum omits {description}")
        raw_path = Path(str(record.get("path", ""))).expanduser()
        path = (
            raw_path
            if raw_path.is_absolute()
            else source_manifest_path.parent / raw_path
        ).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Scene-state V8 {description} is invalid: {path}")
        actual_sha256 = _sha256_file(path)
        if record.get("sha256") != actual_sha256:
            raise ValueError(f"Scene-state V8 {description} file SHA-256 differs")
        return path, actual_sha256

    schedule_record = curriculum.get("schedule")
    schedule_path, schedule_file_sha256 = resolve_artifact(
        schedule_record,
        description="schedule",
    )
    if not isinstance(schedule_record, dict):
        raise AssertionError("validated schedule record changed type")
    schedule_lines = schedule_path.read_text(encoding="utf-8").splitlines()
    if not schedule_lines or any(not line.strip() for line in schedule_lines):
        raise ValueError("Scene-state V8 schedule contains blank rows")
    schedule_entries: list[dict[str, object]] = []
    schedule_indices: list[int] = []
    for schedule_index, line in enumerate(schedule_lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Scene-state V8 schedule row {schedule_index} is invalid JSON"
            ) from error
        if not isinstance(entry, dict) or entry.get("schema") != (
            _SCENE_MEMORY_V8_SCHEDULE_ENTRY_SCHEMA
        ):
            raise ValueError(
                f"Scene-state V8 schedule row {schedule_index} schema differs"
            )
        unsigned_entry = dict(entry)
        declared_entry_hash = unsigned_entry.pop("entry_sha256", None)
        if declared_entry_hash != _canonical_json_sha256(unsigned_entry):
            raise ValueError(
                f"Scene-state V8 schedule row {schedule_index} SHA-256 differs"
            )
        ordinal = entry.get("train_row_ordinal")
        if (
            entry.get("schedule_index") != schedule_index
            or entry.get("step") != schedule_index + 1
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < train_rows
        ):
            raise ValueError(
                f"Scene-state V8 schedule row {schedule_index} indexing differs"
            )
        if entry.get("phase") not in {"value14", "balanced"} or entry.get(
            "target_stratum"
        ) not in _SCENE_STATE_IDENTITY_TARGET_STRATA:
            raise ValueError(
                f"Scene-state V8 schedule row {schedule_index} category differs"
            )
        schedule_entries.append(entry)
        schedule_indices.append(ordinal)
    entries_sha256 = _canonical_json_sha256(schedule_entries)
    if (
        schedule_record.get("rows") != len(schedule_entries)
        or schedule_record.get("entries_sha256") != entries_sha256
        or curriculum.get("total_steps") != len(schedule_entries)
    ):
        raise ValueError("Scene-state V8 schedule length or entries SHA-256 differs")

    schedule_manifest_record = curriculum.get("schedule_manifest")
    schedule_manifest_path, schedule_manifest_file_sha256 = resolve_artifact(
        schedule_manifest_record,
        description="schedule manifest",
    )
    if not isinstance(schedule_manifest_record, dict):
        raise AssertionError("validated schedule-manifest record changed type")
    schedule_manifest = _load_json_object(
        schedule_manifest_path,
        description="scene-state V8 schedule manifest",
    )
    if schedule_manifest.get("schema") != _SCENE_MEMORY_V8_SCHEDULE_MANIFEST_SCHEMA:
        raise ValueError("Scene-state V8 schedule-manifest schema differs")
    unsigned_manifest = dict(schedule_manifest)
    declared_manifest_hash = unsigned_manifest.pop("manifest_sha256", None)
    actual_manifest_hash = _canonical_json_sha256(unsigned_manifest)
    if (
        declared_manifest_hash != actual_manifest_hash
        or schedule_manifest_record.get("manifest_sha256") != actual_manifest_hash
    ):
        raise ValueError("Scene-state V8 schedule-manifest canonical SHA-256 differs")
    manifest_schedule = schedule_manifest.get("schedule")
    manifest_curriculum = schedule_manifest.get("curriculum")
    if (
        not isinstance(manifest_schedule, dict)
        or manifest_schedule.get("sha256") != schedule_file_sha256
        or manifest_schedule.get("entries_sha256") != entries_sha256
        or manifest_schedule.get("rows") != len(schedule_entries)
        or not isinstance(manifest_curriculum, dict)
        or manifest_curriculum.get("entries_sha256") != entries_sha256
        or manifest_curriculum.get("total_steps") != len(schedule_entries)
        or manifest_curriculum.get("checkpoint_steps")
        != curriculum.get("checkpoint_steps")
    ):
        raise ValueError("Scene-state V8 schedule manifest differs from its binding")
    return {
        "schema": _SCENE_MEMORY_V8_CURRICULUM_SCHEMA,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_file_sha256": source_identity["file_sha256"],
        "schedule_path": str(schedule_path),
        "schedule_file_sha256": schedule_file_sha256,
        "schedule_entries_sha256": entries_sha256,
        "schedule_manifest_path": str(schedule_manifest_path),
        "schedule_manifest_file_sha256": schedule_manifest_file_sha256,
        "schedule_manifest_sha256": actual_manifest_hash,
        "ordered_train_row_ordinals_sha256": manifest_schedule.get(
            "ordered_train_row_ordinals_sha256"
        ),
        "total_steps": len(schedule_entries),
        "checkpoint_steps": list(curriculum.get("checkpoint_steps", [])),
        "value14_ordinals": list(curriculum.get("value14_ordinals", [])),
        "indices": tuple(schedule_indices),
    }


def _scene_state_v8_curriculum_protocol_summary(
    binding: dict[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in binding.items()
        if key != "indices"
    }


def _validate_scene_state_v8_locked_training_args(
    args: argparse.Namespace,
    curriculum_binding: dict[str, object],
) -> None:
    expected_values = {
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "max_length": 256,
        "max_write_length": 2048,
        "episode_read_write_enabled": False,
        "memory_loss_mode": "scene_state_generation_ce",
        "scene_state_generated_unlikelihood_weight": (
            _SCENE_STATE_GENERATED_UNLIKELIHOOD_WEIGHT
        ),
        "scene_state_generated_unlikelihood_max_wrong_tokens": (
            _SCENE_STATE_GENERATED_UNLIKELIHOOD_MAX_WRONG_TOKENS
        ),
        "scene_state_generated_rollout_extra_tokens": (
            _SCENE_STATE_GENERATED_ROLLOUT_EXTRA_TOKENS
        ),
        "scene_state_generated_rollout_max_tokens": (
            _SCENE_STATE_GENERATED_ROLLOUT_MAX_TOKENS
        ),
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_contrast_weight": 0.0,
        "memory_representation_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_causal_weight": 0.0,
        "memory_anchor_weight": 0.0,
        "memory_recover_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.0,
        "warmup_steps": _SCENE_MEMORY_V8_WARMUP_STEPS,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "num_train_epochs": 1.0,
        "logging_steps": 1,
        "save_total_limit": 1,
        "validation_split_ratio": 0.0,
        "load_best_model_at_end": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "frozen_mlp_activation_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "train_sampler_seed": None,
        "group_by_length": False,
        "initial_adapter_output_dir": None,
        "prepare_only": False,
    }
    mismatches = [
        name
        for name, expected in expected_values.items()
        if getattr(args, name) != expected
    ]
    checkpoint_steps = curriculum_binding.get("checkpoint_steps")
    if not isinstance(checkpoint_steps, list) or not checkpoint_steps:
        raise ValueError("Scene-memory V8 curriculum has no checkpoint endpoints")
    allowed_horizons = {1, *(int(step) for step in checkpoint_steps)}
    if args.max_steps not in allowed_horizons:
        mismatches.append("max_steps")
    expected_save_steps = 1 if args.max_steps == 1 else 14
    if args.save_steps != expected_save_steps:
        mismatches.append("save_steps")

    is_resume = args.resume_from_checkpoint is not None
    if is_resume:
        if args.warm_start_from_checkpoint is not None:
            mismatches.append("warm_start_from_checkpoint")
        if args.warm_start_mode is not None:
            mismatches.append("warm_start_mode")
        if args.resume_mode != "extend" or args.max_steps == 1:
            mismatches.append("resume_mode")
    else:
        if args.warm_start_from_checkpoint is None:
            mismatches.append("warm_start_from_checkpoint")
        if args.warm_start_mode != _SCENE_V8_WARM_START_MODE:
            mismatches.append("warm_start_mode")
        if args.resume_mode != "exact":
            mismatches.append("resume_mode")
        if args.max_steps not in {1, int(checkpoint_steps[0])}:
            mismatches.append("max_steps")
    if mismatches:
        raise ValueError(
            "Scene-memory V8 locked training contract differs for: "
            + ", ".join(sorted(set(mismatches)))
        )


def _scene_state_generation_pairing_binding(
    args: argparse.Namespace,
) -> dict[str, object]:
    source_identity = _scene_state_source_manifest_identity(args)
    if source_identity is None:
        raise ValueError(
            "scene_state_generation_ce requires a bound V7 source manifest"
        )
    source_manifest_path = Path(str(source_identity["path"]))
    source_manifest = _load_json_object(
        source_manifest_path,
        description="scene-state V7 source manifest",
    )
    if source_manifest.get("schema") not in {
        "rwkv_ms_scene_memory_v7_source.v1",
        _SCENE_MEMORY_V8_SOURCE_SCHEMA,
    }:
        raise ValueError(
            "scene_state_generation_ce requires a V7 or V8 scene-memory source schema"
        )
    binding = source_manifest.get("v7_pairing")
    if not isinstance(binding, dict) or binding.get("schema") != (
        "rwkv_ms_scene_memory_v7_pairing_binding.v1"
    ):
        raise ValueError("Scene-state V7 source manifest has no valid pairing binding")
    pair_artifact = binding.get("pair_manifest")
    if not isinstance(pair_artifact, dict):
        raise ValueError("Scene-state V7 pairing binding omits pair_manifest")
    pair_path = Path(str(pair_artifact.get("path", ""))).expanduser()
    if not pair_path.is_absolute():
        pair_path = source_manifest_path.parent / pair_path
    pair_path = pair_path.resolve()
    if not pair_path.is_file() or pair_path.is_symlink():
        raise ValueError(f"Scene-state V7 pair manifest is invalid: {pair_path}")
    pair_file_sha256 = _sha256_file(pair_path)
    if pair_artifact.get("sha256") != pair_file_sha256:
        raise ValueError("Scene-state V7 pair-manifest file SHA-256 differs")
    pair_manifest = _load_json_object(
        pair_path,
        description="scene-state V7 pair manifest",
    )
    if pair_manifest.get("schema") != "rwkv_ms_scene_memory_v7_pairing.v1":
        raise ValueError("Unexpected scene-state V7 pair-manifest schema")
    declared_manifest_sha256 = pair_manifest.get("manifest_sha256")
    manifest_without_hash = dict(pair_manifest)
    manifest_without_hash.pop("manifest_sha256", None)
    actual_manifest_sha256 = _canonical_json_sha256(manifest_without_hash)
    if declared_manifest_sha256 != actual_manifest_sha256 or pair_artifact.get(
        "manifest_sha256"
    ) != actual_manifest_sha256:
        raise ValueError("Scene-state V7 pair-manifest canonical SHA-256 differs")
    dataset = pair_manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Scene-state V7 pair manifest omits dataset binding")
    train_file = Path(str(source_identity["train_file"])).resolve()
    if Path(str(dataset.get("path", ""))).expanduser().resolve() != train_file:
        raise ValueError("Scene-state V7 pair manifest binds a different train file")
    if dataset.get("sha256") != source_identity["train_file_sha256"] or binding.get(
        "dataset_sha256"
    ) != source_identity["train_file_sha256"]:
        raise ValueError("Scene-state V7 pairing dataset SHA-256 differs")
    raw_lines = [
        line.rstrip("\n")
        for line in train_file.read_text().splitlines(keepends=True)
        if line.strip()
    ]
    ordered_row_hashes = [
        hashlib.sha256(line.encode("utf-8")).hexdigest() for line in raw_lines
    ]
    raw_boundary_counts: list[int] = []
    for row_index, line in enumerate(raw_lines):
        try:
            raw_row = json.loads(line)
            messages = raw_row["messages"]
            assistant = messages[-1]
            payload = json.loads(assistant["content"])
            boundaries = payload["boundaries"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Scene-state V7 raw train row {row_index} has invalid scene labels"
            ) from error
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(assistant, dict)
            or assistant.get("role") != "assistant"
            or not isinstance(payload, dict)
            or not isinstance(boundaries, list)
            or any(
                isinstance(boundary, bool) or not isinstance(boundary, int)
                for boundary in boundaries
            )
        ):
            raise ValueError(
                f"Scene-state V7 raw train row {row_index} has invalid scene labels"
            )
        raw_boundary_counts.append(len(boundaries))
    if dataset.get("rows") != len(raw_lines) or dataset.get(
        "ordered_row_sha256"
    ) != _canonical_json_sha256(ordered_row_hashes):
        raise ValueError("Scene-state V7 ordered train-row binding differs")
    directed_pairs = pair_manifest.get("directed_pairs")
    if not isinstance(directed_pairs, list) or len(directed_pairs) != len(raw_lines):
        raise ValueError(
            "Scene-state V7 pairing must contain one directed entry per train row"
        )
    directed_entries_sha256 = _canonical_json_sha256(directed_pairs)
    if (
        pair_manifest.get("entries_sha256") != directed_entries_sha256
        or binding.get("directed_entry_count") != len(directed_pairs)
        or binding.get("entries_sha256") != directed_entries_sha256
    ):
        raise ValueError("Scene-state V7 directed-entry binding differs")
    quotas = pair_manifest.get("quotas")
    if not isinstance(quotas, dict) or binding.get("quotas") != quotas:
        raise ValueError("Scene-state V7 pairing quotas differ from the source binding")
    required_entry_fields = {
        "train_row_ordinal",
        "donor_train_row_ordinal",
        "official_source_index",
        "donor_official_source_index",
        "source_row_sha256",
        "donor_row_sha256",
        "source_label_sha256",
        "donor_label_sha256",
        "source_base_record_sha256",
        "donor_base_record_sha256",
        "source_strict_failure_stratum",
        "donor_strict_failure_stratum",
        "source_strict_score_sha256",
        "donor_strict_score_sha256",
        "source_boundary_count",
        "donor_boundary_count",
        "target_stratum",
        "source_generation_prefix_sha256",
        "donor_generation_prefix_sha256",
        "source_write_sha256",
        "donor_write_sha256",
        "source_write_token_count",
        "donor_write_token_count",
        "write_token_count_delta",
        "first_differing_semantic_ordinal",
        "selected_target_positions",
        "selected_target_predictor_positions",
        "selected_target_token_ids",
        "donor_target_token_ids",
        "causal_prefix_sha256",
        "entry_sha256",
    }
    entries_by_ordinal: list[dict[str, object] | None] = [None] * len(raw_lines)
    for entry_index, entry in enumerate(directed_pairs):
        if not isinstance(entry, dict) or set(entry) != required_entry_fields:
            raise ValueError(
                f"Scene-state V7 directed entry {entry_index} has unexpected fields"
            )
        entry_without_hash = dict(entry)
        entry_without_hash.pop("entry_sha256")
        if entry["entry_sha256"] != _canonical_json_sha256(entry_without_hash):
            raise ValueError(
                f"Scene-state V7 directed entry {entry_index} SHA-256 differs"
            )
        source_ordinal = entry["train_row_ordinal"]
        donor_ordinal = entry["donor_train_row_ordinal"]
        if (
            isinstance(source_ordinal, bool)
            or not isinstance(source_ordinal, int)
            or isinstance(donor_ordinal, bool)
            or not isinstance(donor_ordinal, int)
            or not 0 <= source_ordinal < len(raw_lines)
            or not 0 <= donor_ordinal < len(raw_lines)
            or source_ordinal == donor_ordinal
            or entries_by_ordinal[source_ordinal] is not None
        ):
            raise ValueError(
                f"Scene-state V7 directed entry {entry_index} has invalid ordinals"
            )
        if entry["source_row_sha256"] != ordered_row_hashes[source_ordinal] or entry[
            "donor_row_sha256"
        ] != ordered_row_hashes[donor_ordinal]:
            raise ValueError(
                f"Scene-state V7 directed entry {entry_index} row hash differs"
            )
        if (
            entry["source_boundary_count"] != raw_boundary_counts[source_ordinal]
            or entry["donor_boundary_count"] != raw_boundary_counts[donor_ordinal]
        ):
            raise ValueError(
                f"Scene-state V7 directed entry {entry_index} boundary binding differs"
            )
        entries_by_ordinal[source_ordinal] = entry
    entries = [entry for entry in entries_by_ordinal if entry is not None]
    donor_ordinals = [int(entry["donor_train_row_ordinal"]) for entry in entries]
    for source_ordinal, donor_ordinal in enumerate(donor_ordinals):
        reverse = entries[donor_ordinal]
        if int(reverse["donor_train_row_ordinal"]) != source_ordinal:
            raise ValueError("Scene-state V7 directed pairing is not symmetric")
        for source_field, donor_field in (
            ("official_source_index", "donor_official_source_index"),
            ("source_row_sha256", "donor_row_sha256"),
            ("source_label_sha256", "donor_label_sha256"),
            ("source_base_record_sha256", "donor_base_record_sha256"),
            ("source_strict_failure_stratum", "donor_strict_failure_stratum"),
            ("source_strict_score_sha256", "donor_strict_score_sha256"),
            (
                "source_generation_prefix_sha256",
                "donor_generation_prefix_sha256",
            ),
            ("source_write_sha256", "donor_write_sha256"),
            ("source_write_token_count", "donor_write_token_count"),
            ("source_boundary_count", "donor_boundary_count"),
            ("selected_target_token_ids", "donor_target_token_ids"),
        ):
            if entries[source_ordinal][source_field] != reverse[donor_field]:
                raise ValueError(
                    "Scene-state V7 symmetric directed entries disagree: "
                    f"{source_field}/{donor_field}"
                )
    observed_quotas = {
        stratum: sum(entry["target_stratum"] == stratum for entry in entries)
        for stratum in _SCENE_STATE_IDENTITY_TARGET_STRATA
    }
    if quotas != observed_quotas or sum(observed_quotas.values()) != len(entries):
        raise ValueError("Scene-state V7 directed pairing quotas do not match entries")
    return {
        "source_identity": source_identity,
        "pair_path": str(pair_path),
        "pair_file_sha256": pair_file_sha256,
        "pair_manifest_sha256": actual_manifest_sha256,
        "quotas": dict(quotas),
        "entries_sha256": directed_entries_sha256,
        "entries": entries,
    }


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
        "--initial-adapter-output-dir",
        type=Path,
        default=None,
        help=(
            "Fresh single-process runs only: save the seeded step-0 adapter, config, "
            "training protocol, and provenance before Trainer construction. The path "
            "must be exactly OUTPUT_DIR/initial_adapter."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Validate data and topology, save the seeded step-0 adapter, then exit "
            "before TrainingArguments, Trainer, or optimizer construction."
        ),
    )
    checkpoint_source = parser.add_mutually_exclusive_group()
    checkpoint_source.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path to resume, or 'latest'/'auto' for the newest complete checkpoint.",
    )
    checkpoint_source.add_argument(
        "--warm-start-from-checkpoint",
        default=None,
        help="Completed adapter checkpoint to import without Trainer state.",
    )
    parser.add_argument(
        "--warm-start-mode",
        choices=_WARM_START_MODES,
        default=None,
        help="Strict adapter-only warm-start contract.",
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
    parser.add_argument(
        "--expected-tokenized-dataset-sha256",
        default=None,
        help=(
            "Optional canonical ordered-row SHA-256 lock for the tokenized dataset. "
            "Managed cache hits, fresh cache builds, direct maps, and explicit "
            "--tokenized-dataset-dir loads must match it."
        ),
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
            "post_attention_residual_hybrid",
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
        "--memory-fusion-residual-scale-max",
        type=float,
        default=1.0,
        help="Maximum effective direct residual gain for residual-hybrid fusion.",
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
            "scene_state_identity_ce",
            "scene_state_generation_ce",
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
    parser.add_argument("--memory-representation-weight", type=float, default=0.0)
    parser.add_argument("--memory-representation-margin", type=float, default=0.1)
    parser.add_argument("--memory-causal-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-weight", type=float, default=1.0)
    parser.add_argument("--memory-anchor-margin", type=float, default=0.005)
    parser.add_argument("--memory-recover-weight", type=float, default=0.25)
    parser.add_argument("--memory-need-floor", type=float, default=0.15)
    parser.add_argument("--memory-dropout-no-memory-prob", type=float, default=0.0)
    parser.add_argument("--memory-dropout-state-only-prob", type=float, default=0.0)
    parser.add_argument("--memory-base-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--scene-state-identity-margin",
        type=float,
        default=0.5,
        help=(
            "Required donor-minus-correct and zero-minus-correct per-row semantic "
            "CE margin for scene_state_identity_ce."
        ),
    )
    parser.add_argument(
        "--scene-state-source-manifest",
        type=Path,
        default=None,
        help="Dataset-bound scene failure-pair source manifest.",
    )
    parser.add_argument(
        "--expected-scene-state-source-manifest-sha256",
        default=None,
        help="Exact SHA-256 lock for --scene-state-source-manifest.",
    )
    parser.add_argument(
        "--scene-state-generated-unlikelihood-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for correct-state greedy generated-prefix unlikelihood on "
            "same/cross-cardinality scene rows."
        ),
    )
    parser.add_argument(
        "--scene-state-generated-unlikelihood-max-wrong-tokens",
        type=int,
        default=_SCENE_STATE_GENERATED_UNLIKELIHOOD_MAX_WRONG_TOKENS,
    )
    parser.add_argument(
        "--scene-state-generated-rollout-extra-tokens",
        type=int,
        default=_SCENE_STATE_GENERATED_ROLLOUT_EXTRA_TOKENS,
    )
    parser.add_argument(
        "--scene-state-generated-rollout-max-tokens",
        type=int,
        default=_SCENE_STATE_GENERATED_ROLLOUT_MAX_TOKENS,
    )
    parser.add_argument(
        "--scene-boundary-payload-ce-weight",
        type=float,
        default=0.0,
        help=(
            "For context_dropout_ce scene training, add this weight times the CE sum "
            "over only the tokenizer-aligned top-level JSON boundaries list, normalized "
            "by the same full supervised-token denominator as the base CE."
        ),
    )
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
    parser.add_argument(
        "--train-sampler-seed",
        type=int,
        default=None,
        help=(
            "Opt in to a torch RandomSampler locked to this seed. It must equal "
            "--data-seed so Accelerate's seedable sampler preserves the same order. "
            "Unset preserves the Transformers Trainer sampler exactly."
        ),
    )
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help=(
            "Use an exact number of optimizer warmup steps. Set --warmup-ratio 0 "
            "when this override is supplied."
        ),
    )
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
    if (args.warm_start_from_checkpoint is None) != (args.warm_start_mode is None):
        raise ValueError(
            "--warm-start-from-checkpoint and --warm-start-mode must be provided together"
        )
    if args.initial_adapter_output_dir is not None and (
        args.resume_from_checkpoint is not None
        or args.warm_start_from_checkpoint is not None
    ):
        raise ValueError("--initial-adapter-output-dir is valid only for fresh runs")
    if args.prepare_only and args.initial_adapter_output_dir is None:
        raise ValueError("--prepare-only requires --initial-adapter-output-dir")
    if args.train_sampler_seed is not None and not (
        0 <= args.train_sampler_seed <= torch.iinfo(torch.int64).max
    ):
        raise ValueError("train-sampler-seed must satisfy 0 <= seed <= 2^63 - 1")
    if args.train_sampler_seed is not None and args.group_by_length:
        raise ValueError("train-sampler-seed is incompatible with group-by-length")
    if (
        args.train_sampler_seed is not None
        and args.train_sampler_seed != args.data_seed
    ):
        raise ValueError("train-sampler-seed must equal data-seed")
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
    if (
        not math.isfinite(args.scene_boundary_payload_ce_weight)
        or args.scene_boundary_payload_ce_weight < 0.0
    ):
        raise ValueError("scene-boundary-payload-ce-weight must be finite and non-negative")
    if (
        args.scene_boundary_payload_ce_weight > 0.0
        and args.memory_loss_mode != "context_dropout_ce"
    ):
        raise ValueError(
            "scene-boundary-payload-ce-weight requires "
            "memory-loss-mode=context_dropout_ce"
        )
    if (
        not math.isfinite(args.memory_representation_weight)
        or args.memory_representation_weight < 0.0
    ):
        raise ValueError("memory-representation-weight must be finite and non-negative")
    if (
        not math.isfinite(args.memory_representation_margin)
        or args.memory_representation_margin <= 0.0
    ):
        raise ValueError("memory-representation-margin must be finite and positive")
    if (
        not math.isfinite(args.scene_state_generated_unlikelihood_weight)
        or args.scene_state_generated_unlikelihood_weight < 0.0
    ):
        raise ValueError(
            "scene-state-generated-unlikelihood-weight must be finite and non-negative"
        )
    if args.scene_state_generated_unlikelihood_max_wrong_tokens <= 0:
        raise ValueError(
            "scene-state-generated-unlikelihood-max-wrong-tokens must be positive"
        )
    if args.scene_state_generated_rollout_extra_tokens < 0:
        raise ValueError(
            "scene-state-generated-rollout-extra-tokens must be non-negative"
        )
    if args.scene_state_generated_rollout_max_tokens <= 0:
        raise ValueError(
            "scene-state-generated-rollout-max-tokens must be positive"
        )
    if (
        args.scene_state_generated_unlikelihood_weight > 0.0
        and args.memory_loss_mode != "scene_state_generation_ce"
    ):
        raise ValueError(
            "scene-state-generated-unlikelihood-weight requires "
            "memory-loss-mode=scene_state_generation_ce"
        )
    if (
        args.scene_state_generated_unlikelihood_weight > 0.0
        and args.per_device_train_batch_size != 1
    ):
        raise ValueError(
            "scene-state generated-prefix unlikelihood requires "
            "per-device-train-batch-size=1"
        )
    if args.rwkv_ms_output_init_scale < 0.0:
        raise ValueError("rwkv-ms-output-init-scale must be non-negative")
    if args.memory_base_kl_weight > 0.0 and args.memory_loss_mode != "context_dropout_ce":
        raise ValueError("memory-base-kl-weight requires memory-loss-mode=context_dropout_ce")
    if (
        args.memory_representation_weight > 0.0
        and args.memory_loss_mode != "content_contrast_ce"
    ):
        raise ValueError(
            "memory-representation-weight requires memory-loss-mode=content_contrast_ce"
        )
    if args.memory_representation_weight > 0.0:
        if "o" not in parse_delta_heads(args.delta_heads):
            raise ValueError(
                "memory-representation-weight requires an active delta_o head"
            )
        if (
            normalize_memory_fusion_placement(args.memory_fusion_placement)
            not in _REPRESENTATION_CAPTURE_FUSION_PLACEMENTS
        ):
            raise ValueError(
                "memory-representation-weight supports only attention_output or "
                "post_attention_residual_hybrid fusion"
            )
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
    scene_state_source_identity = _scene_state_source_manifest_identity(args)
    if args.memory_loss_mode == "scene_state_identity_ce":
        if scene_state_source_identity is None:
            raise ValueError(
                "scene_state_identity_ce requires a source manifest and exact SHA-256 lock"
            )
        if args.training_mode != "episode":
            raise ValueError("scene_state_identity_ce requires training-mode=episode")
        if args.episode_recent_messages != 0:
            raise ValueError(
                "scene_state_identity_ce requires episode-recent-messages=0"
            )
        if args.assistant_loss_mode != "final_assistant_only":
            raise ValueError(
                "scene_state_identity_ce requires assistant-loss-mode=final_assistant_only"
            )
        if args.episode_read_write_enabled:
            raise ValueError(
                "scene_state_identity_ce requires episode read writes to be disabled"
            )
        if args.memory_kl_weight != 0.0 or args.memory_base_kl_weight != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires all KL weights to be zero"
            )
        if args.memory_representation_weight != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires representation loss to be disabled"
            )
        if (
            not math.isfinite(args.scene_state_identity_margin)
            or args.scene_state_identity_margin <= 0.0
        ):
            raise ValueError(
                "scene-state-identity-margin must be finite and positive"
            )
        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                "scene_state_identity_ce requires gradient-accumulation-steps=1"
            )
        if args.write_sparsity_weight != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires write sparsity loss to be disabled"
            )
        if args.scene_boundary_payload_ce_weight != 0.0:
            raise ValueError(
                "scene_state_identity_ce requires scene-boundary-payload-ce-weight=0"
            )
        if args.deepspeed_config is not None:
            raise ValueError("scene_state_identity_ce does not support DeepSpeed")
    if args.memory_loss_mode == "scene_state_generation_ce":
        if scene_state_source_identity is None:
            raise ValueError(
                "scene_state_generation_ce requires a source manifest and exact SHA-256 lock"
            )
        if args.training_mode != "episode":
            raise ValueError("scene_state_generation_ce requires training-mode=episode")
        if args.episode_recent_messages != 0:
            raise ValueError(
                "scene_state_generation_ce requires episode-recent-messages=0"
            )
        if args.assistant_loss_mode != "final_assistant_only":
            raise ValueError(
                "scene_state_generation_ce requires assistant-loss-mode=final_assistant_only"
            )
        if args.episode_read_write_enabled:
            raise ValueError(
                "scene_state_generation_ce requires episode read writes to be disabled"
            )
        if args.memory_kl_weight != 0.0 or args.memory_base_kl_weight != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires all KL weights to be zero"
            )
        if args.memory_representation_weight != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires representation loss to be disabled"
            )
        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                "scene_state_generation_ce requires gradient-accumulation-steps=1"
            )
        if args.validation_split_ratio != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires validation-split-ratio=0 because "
                "the directed pairing is bound to the complete train order"
            )
        if args.write_sparsity_weight != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires write sparsity loss to be disabled"
            )
        if args.scene_boundary_payload_ce_weight != 0.0:
            raise ValueError(
                "scene_state_generation_ce requires scene-boundary-payload-ce-weight=0"
            )
        if args.deepspeed_config is not None:
            raise ValueError("scene_state_generation_ce does not support DeepSpeed")
        _scene_state_generation_pairing_binding(args)
    elif (
        scene_state_source_identity is not None
        and args.memory_loss_mode not in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
    ):
        raise ValueError(
            "Scene-state source manifest flags require "
            "a scene-state paired memory loss mode"
        )
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


def _tokenize_chat_generation_prompt(
    tokenizer,
    messages: list[dict[str, str]],
) -> list[int]:
    tokenized = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
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


def _scene_boundary_payload_metadata(
    content: str,
) -> tuple[tuple[int, int], int] | None:
    decoder = json.JSONDecoder()

    def skip_whitespace(position: int) -> int:
        while position < len(content) and content[position].isspace():
            position += 1
        return position

    position = skip_whitespace(0)
    if position >= len(content) or content[position] != "{":
        return None
    position += 1
    boundary_metadata: tuple[tuple[int, int], int] | None = None
    try:
        while True:
            position = skip_whitespace(position)
            if position >= len(content):
                return None
            if content[position] == "}":
                position = skip_whitespace(position + 1)
                return boundary_metadata if position == len(content) else None
            key, position = decoder.raw_decode(content, position)
            if not isinstance(key, str):
                return None
            position = skip_whitespace(position)
            if position >= len(content) or content[position] != ":":
                return None
            value_start = skip_whitespace(position + 1)
            value, value_end = decoder.raw_decode(content, value_start)
            if key == "boundaries":
                if boundary_metadata is not None:
                    return None
                if not isinstance(value, list) or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    for item in value
                ):
                    return None
                boundary_metadata = (
                    (value_start, value_end),
                    len(value),
                )
            position = skip_whitespace(value_end)
            if position >= len(content):
                return None
            if content[position] == ",":
                position += 1
                continue
            if content[position] == "}":
                position = skip_whitespace(position + 1)
                return boundary_metadata if position == len(content) else None
            return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _scene_boundary_payload_char_span(content: str) -> tuple[int, int] | None:
    metadata = _scene_boundary_payload_metadata(content)
    return None if metadata is None else metadata[0]


def _rendered_message_content_span(
    tokenizer,
    messages: list[dict[str, str]],
    message_index: int,
) -> tuple[str, int, int]:
    content = messages[message_index]["content"]
    sentinel = "__DELTAMEM_SCENE_BOUNDARY_CONTENT_SENTINEL__"
    all_content = "\n".join(str(message["content"]) for message in messages)
    while sentinel in all_content:
        sentinel += "_"
    probe_messages = [dict(message) for message in messages]
    probe_messages[message_index]["content"] = sentinel
    rendered = apply_project_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    probe_rendered = apply_project_chat_template(
        tokenizer,
        probe_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not isinstance(probe_rendered, str):
        raise ValueError("Chat template must render text for scene-boundary span alignment")
    if probe_rendered.count(sentinel) != 1:
        raise ValueError(
            "Chat template did not preserve the scene-boundary assistant content sentinel"
        )
    prefix, suffix = probe_rendered.split(sentinel)
    if rendered != prefix + content + suffix:
        raise ValueError(
            "Chat template transformed scene-boundary assistant content; cannot align payload"
        )
    content_start = len(prefix)
    return rendered, content_start, content_start + len(content)


def _tokenizer_ids_and_offsets(tokenizer, rendered: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "Scene-boundary payload CE requires a tokenizer with character offset mappings"
        ) from exc
    try:
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Scene-boundary payload CE tokenizer did not return IDs and offset mappings"
        ) from exc
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Scene-boundary tokenizer returned an unexpected ID batch")
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        if len(offsets) != 1:
            raise ValueError("Scene-boundary tokenizer returned an unexpected offset batch")
        offsets = offsets[0]
    normalized_ids = [int(token_id) for token_id in input_ids]
    normalized_offsets = [(int(start), int(end)) for start, end in offsets]
    if len(normalized_ids) != len(normalized_offsets):
        raise ValueError("Scene-boundary tokenizer IDs and offsets have different lengths")
    return normalized_ids, normalized_offsets


def _scene_boundary_payload_token_mask(
    tokenizer,
    messages: list[dict[str, str]],
    message_index: int,
    expected_input_ids: list[int],
) -> list[bool]:
    content = messages[message_index]["content"]
    payload_span = _scene_boundary_payload_char_span(content)
    if payload_span is None:
        raise ValueError(
            "Scene-boundary payload CE requires a top-level integer `boundaries` JSON list"
        )
    rendered, content_start, _ = _rendered_message_content_span(
        tokenizer,
        messages,
        message_index,
    )
    token_ids, offsets = _tokenizer_ids_and_offsets(tokenizer, rendered)
    if token_ids != expected_input_ids:
        raise ValueError(
            "Chat-template token IDs differ from tokenizer offset-alignment token IDs"
        )
    payload_start = content_start + payload_span[0]
    payload_end = content_start + payload_span[1]
    mask = [
        start < payload_end and end > payload_start and end > start
        for start, end in offsets
    ]
    if not any(mask):
        raise ValueError("Scene-boundary JSON list did not align to any tokenizer token")
    return mask


def _scene_boundary_semantic_token_mask(
    tokenizer,
    messages: list[dict[str, str]],
    message_index: int,
    expected_input_ids: list[int],
) -> list[bool]:
    """Select non-whitespace tokens participating in the boundaries-array decision."""

    content = messages[message_index]["content"]
    payload_span = _scene_boundary_payload_char_span(content)
    if payload_span is None:
        raise ValueError(
            "Scene-state identity CE requires a top-level integer `boundaries` JSON list"
        )
    rendered, content_start, _ = _rendered_message_content_span(
        tokenizer,
        messages,
        message_index,
    )
    token_ids, offsets = _tokenizer_ids_and_offsets(tokenizer, rendered)
    if token_ids != expected_input_ids:
        raise ValueError(
            "Chat-template token IDs differ from tokenizer offset-alignment token IDs"
        )
    payload_start = content_start + payload_span[0]
    payload_end = content_start + payload_span[1]
    mask: list[bool] = []
    for start, end in offsets:
        overlap_start = max(start, payload_start)
        overlap_end = min(end, payload_end)
        mask.append(
            overlap_start < overlap_end
            and any(
                not character.isspace()
                for character in rendered[overlap_start:overlap_end]
            )
        )
    if not any(mask):
        raise ValueError(
            "Scene-boundary semantic decision did not align to any non-whitespace token"
        )
    return mask


def _scene_state_generation_token_masks(
    tokenizer,
    messages: list[dict[str, str]],
    message_index: int,
    expected_input_ids: list[int],
) -> dict[str, list[bool]]:
    if message_index <= 0 or messages[message_index]["role"] != "assistant":
        raise ValueError(
            "Scene-state generation supervision requires an assistant target with a prompt"
        )
    rendered, content_start, content_end = _rendered_message_content_span(
        tokenizer,
        messages,
        message_index,
    )
    token_ids, offsets = _tokenizer_ids_and_offsets(tokenizer, rendered)
    if token_ids != expected_input_ids:
        raise ValueError(
            "Scene-state generation token IDs differ from offset-aligned chat-template IDs"
        )
    generation_prompt_ids = _tokenize_chat_generation_prompt(
        tokenizer,
        messages[:message_index],
    )
    if expected_input_ids[: len(generation_prompt_ids)] != generation_prompt_ids:
        raise ValueError(
            "Scene-state assistant content does not follow the exact system-only "
            "generation prompt"
        )
    target_mask = [
        index >= len(generation_prompt_ids)
        for index in range(len(expected_input_ids))
    ]
    content_mask = [
        start < content_end and end > content_start and end > start
        for start, end in offsets
    ]
    payload_start, payload_end = _scene_boundary_payload_char_span(
        messages[message_index]["content"]
    ) or (None, None)
    if payload_start is None or payload_end is None:
        raise ValueError(
            "Scene-state generation supervision requires a top-level boundaries array"
        )
    decision_positions = {
        content_start + position
        for position in range(payload_start, payload_end)
        if messages[message_index]["content"][position] in "[],0123456789"
    }
    decision_mask = [
        any(start <= position < end for position in decision_positions)
        if end > start
        else False
        for start, end in offsets
    ]
    schema_mask = [
        content and not decision
        for content, decision in zip(content_mask, decision_mask)
    ]
    termination_mask = [
        target and not content
        for target, content in zip(target_mask, content_mask)
    ]
    if any(content and not target for content, target in zip(content_mask, target_mask)):
        raise ValueError(
            "Scene-state assistant content begins before the exact generation prefix"
        )
    if any(
        decision and not content
        for decision, content in zip(decision_mask, content_mask)
    ):
        raise ValueError("Scene-state decision mask escapes assistant content")
    if not any(content_mask) or not any(schema_mask) or not any(decision_mask):
        raise ValueError(
            "Scene-state generation supervision requires content, schema, and decision tokens"
        )
    if not any(termination_mask):
        raise ValueError(
            "Scene-state generation supervision requires a chat-template termination token"
        )
    if [
        content or termination
        for content, termination in zip(content_mask, termination_mask)
    ] != target_mask:
        raise ValueError(
            "Scene-state content and termination masks do not cover the generation suffix"
        )
    return {
        "scene_state_generation_target_mask": target_mask,
        "scene_state_generation_content_mask": content_mask,
        "scene_state_generation_schema_mask": schema_mask,
        "scene_state_generation_decision_mask": decision_mask,
        "scene_state_generation_termination_mask": termination_mask,
    }


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
    require_scene_boundary_payload_mask: bool = False,
    require_scene_state_semantic_mask: bool = False,
    require_scene_state_generation_masks: bool = False,
) -> dict:
    supervised_assistant_indices = set(
        _select_supervised_assistant_indices(messages, assistant_loss_mode)
    )

    input_ids: list[int] = []
    labels: list[int] = []
    scene_boundary_payload_mask: list[bool] = []
    scene_state_semantic_mask: list[bool] = []
    scene_state_generation_masks = {
        column: []
        for column in (
            "scene_state_generation_target_mask",
            "scene_state_generation_content_mask",
            "scene_state_generation_schema_mask",
            "scene_state_generation_decision_mask",
            "scene_state_generation_termination_mask",
        )
    }
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
            del scene_boundary_payload_mask[prefix_len:]
            del scene_state_semantic_mask[prefix_len:]
            for mask in scene_state_generation_masks.values():
                del mask[prefix_len:]
        input_ids.extend(delta_ids)
        if index in supervised_assistant_indices:
            labels.extend(delta_ids)
            if require_scene_boundary_payload_mask:
                current_payload_mask = _scene_boundary_payload_token_mask(
                    tokenizer,
                    messages[: index + 1],
                    index,
                    current_ids,
                )
                scene_boundary_payload_mask.extend(current_payload_mask[prefix_len:])
            else:
                scene_boundary_payload_mask.extend([False] * len(delta_ids))
            if require_scene_state_semantic_mask:
                current_semantic_mask = _scene_boundary_semantic_token_mask(
                    tokenizer,
                    messages[: index + 1],
                    index,
                    current_ids,
                )
                scene_state_semantic_mask.extend(
                    current_semantic_mask[prefix_len:]
                )
            else:
                scene_state_semantic_mask.extend([False] * len(delta_ids))
            if require_scene_state_generation_masks:
                current_generation_masks = _scene_state_generation_token_masks(
                    tokenizer,
                    messages[: index + 1],
                    index,
                    current_ids,
                )
                for column, mask in scene_state_generation_masks.items():
                    mask.extend(current_generation_masks[column][prefix_len:])
            else:
                for mask in scene_state_generation_masks.values():
                    mask.extend([False] * len(delta_ids))
        else:
            labels.extend([-100] * len(delta_ids))
            scene_boundary_payload_mask.extend([False] * len(delta_ids))
            scene_state_semantic_mask.extend([False] * len(delta_ids))
            for mask in scene_state_generation_masks.values():
                mask.extend([False] * len(delta_ids))
        previous_ids = current_ids

    untruncated_input_ids = input_ids
    input_ids, labels = _truncate_sft_sequence(untruncated_input_ids, labels, max_length)
    _, scene_boundary_payload_mask = _truncate_sft_sequence(
        untruncated_input_ids,
        scene_boundary_payload_mask,
        max_length,
    )
    _, scene_state_semantic_mask = _truncate_sft_sequence(
        untruncated_input_ids,
        scene_state_semantic_mask,
        max_length,
    )
    for column, mask in tuple(scene_state_generation_masks.items()):
        _, scene_state_generation_masks[column] = _truncate_sft_sequence(
            untruncated_input_ids,
            mask,
            max_length,
        )
    if require_scene_state_generation_masks:
        generation_target_mask = scene_state_generation_masks[
            "scene_state_generation_target_mask"
        ]
        labels = [
            label if selected else -100
            for label, selected in zip(labels, generation_target_mask)
        ]
    attention_mask = [1] * len(input_ids)
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if require_scene_boundary_payload_mask:
        if not any(scene_boundary_payload_mask[1:]):
            raise ValueError(
                "Scene-boundary payload was truncated or has no causal predictor"
            )
        if any(
            selected and label == -100
            for selected, label in zip(scene_boundary_payload_mask, labels)
        ):
            raise ValueError("Scene-boundary payload mask escaped assistant supervision")
        features["scene_boundary_payload_mask"] = scene_boundary_payload_mask
    if require_scene_state_semantic_mask:
        if not any(scene_state_semantic_mask[1:]):
            raise ValueError(
                "Scene-state semantic decision was truncated or has no causal predictor"
            )
        if any(
            selected and label == -100
            for selected, label in zip(scene_state_semantic_mask, labels)
        ):
            raise ValueError("Scene-state semantic mask escaped assistant supervision")
        features["scene_state_semantic_mask"] = scene_state_semantic_mask
        supervised_indices = sorted(supervised_assistant_indices)
        if len(supervised_indices) != 1:
            raise ValueError(
                "Scene-state identity CE requires exactly one supervised assistant turn"
            )
        payload_metadata = _scene_boundary_payload_metadata(
            messages[supervised_indices[0]]["content"]
        )
        if payload_metadata is None:
            raise ValueError(
                "Scene-state identity CE requires an audited boundaries cardinality"
            )
        features["scene_state_boundary_count"] = payload_metadata[1]
    if require_scene_state_generation_masks:
        generation_target_mask = scene_state_generation_masks[
            "scene_state_generation_target_mask"
        ]
        if not any(generation_target_mask[1:]):
            raise ValueError(
                "Scene-state generation suffix was truncated or has no causal predictor"
            )
        if any(
            selected != (label != -100)
            for selected, label in zip(generation_target_mask, labels)
        ):
            raise ValueError(
                "Scene-state generation labels must select exactly the generated suffix"
            )
        for column, mask in scene_state_generation_masks.items():
            if len(mask) != len(labels):
                raise ValueError(f"Scene-state generation mask is misaligned: {column}")
            features[column] = mask
    return features


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
    require_scene_boundary_payload_mask: bool = False,
) -> dict[str, list[int]]:
    canonical_input_ids = _tokenize_chat_messages(tokenizer, teacher_messages)
    teacher_features = tokenize_messages_for_sft(
        tokenizer,
        teacher_messages,
        max(len(canonical_input_ids), 1),
        assistant_loss_mode="final_assistant_only",
        require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
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
    require_scene_boundary_payload_mask: bool = False,
    require_scene_state_semantic_mask: bool = False,
    require_scene_state_generation_masks: bool = False,
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
            require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
            require_scene_state_semantic_mask=require_scene_state_semantic_mask,
            require_scene_state_generation_masks=(
                require_scene_state_generation_masks
            ),
        )
        teacher_features = _build_canonical_teacher_features(
            tokenizer,
            prefix_messages + [dict(messages[target_index])],
            full_write_input_ids,
            len(write_input_ids),
            max_write_length=max_write_length,
            max_read_length=max_length,
            student_labels=read_features["labels"],
            require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
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
            require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
        )

        episode = {
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
        if require_scene_boundary_payload_mask:
            episode["scene_boundary_payload_mask"] = read_features[
                "scene_boundary_payload_mask"
            ]
            episode["teacher_scene_boundary_payload_mask"] = teacher_features[
                "scene_boundary_payload_mask"
            ]
            episode["state_only_scene_boundary_payload_mask"] = (
                state_only_read_features["scene_boundary_payload_mask"]
            )
        if require_scene_state_semantic_mask:
            episode["scene_state_semantic_mask"] = read_features[
                "scene_state_semantic_mask"
            ]
            episode["scene_state_boundary_count"] = read_features[
                "scene_state_boundary_count"
            ]
        if require_scene_state_generation_masks:
            for column in (
                "scene_state_generation_target_mask",
                "scene_state_generation_content_mask",
                "scene_state_generation_schema_mask",
                "scene_state_generation_decision_mask",
                "scene_state_generation_termination_mask",
            ):
                episode[column] = read_features[column]
        episodes.append(episode)
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
    require_scene_boundary_payload_mask: bool = False,
    require_scene_state_semantic_mask: bool = False,
    require_scene_state_generation_masks: bool = False,
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
        require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
        require_scene_state_semantic_mask=require_scene_state_semantic_mask,
        require_scene_state_generation_masks=require_scene_state_generation_masks,
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
    require_scene_boundary_payload_mask: bool = False,
    require_scene_state_semantic_mask: bool = False,
    require_scene_state_generation_masks: bool = False,
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
        if require_scene_boundary_payload_mask:
            tokenized["scene_boundary_payload_mask"] = []
            tokenized["teacher_scene_boundary_payload_mask"] = []
            tokenized["state_only_scene_boundary_payload_mask"] = []
        if require_scene_state_semantic_mask:
            tokenized["scene_state_semantic_mask"] = []
            tokenized["scene_state_boundary_count"] = []
        if require_scene_state_generation_masks:
            for column in (
                "scene_state_generation_target_mask",
                "scene_state_generation_content_mask",
                "scene_state_generation_schema_mask",
                "scene_state_generation_decision_mask",
                "scene_state_generation_termination_mask",
            ):
                tokenized[column] = []

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
                require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
                require_scene_state_semantic_mask=require_scene_state_semantic_mask,
                require_scene_state_generation_masks=(
                    require_scene_state_generation_masks
                ),
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
            require_scene_boundary_payload_mask=require_scene_boundary_payload_mask,
            require_scene_state_semantic_mask=require_scene_state_semantic_mask,
            require_scene_state_generation_masks=require_scene_state_generation_masks,
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


def _normalized_expected_tokenized_dataset_sha256(
    args: argparse.Namespace,
) -> str | None:
    expected = getattr(args, "expected_tokenized_dataset_sha256", None)
    if expected is None:
        return None
    if not isinstance(expected, str):
        raise ValueError("Expected tokenized dataset SHA-256 must be a string")
    expected = expected.lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "--expected-tokenized-dataset-sha256 must be exactly 64 hexadecimal characters"
        )
    return expected


def _tokenized_dataset_ordered_sha256(tokenized: Dataset) -> str:
    logical = tokenized.with_format(None)
    header = {
        "schema": _TOKENIZED_ORDERED_CONTENT_SCHEMA,
        "column_names": list(logical.column_names),
        "features": logical.features.to_dict(),
        "rows": len(logical),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\n")
    try:
        for row in logical:
            digest.update(_canonical_json_bytes(row))
            digest.update(b"\n")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Tokenized dataset rows must have canonical JSON-compatible values"
        ) from exc
    return digest.hexdigest()


def _tokenized_cache_persisted_files(cache_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(cache_dir.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative_path = path.relative_to(cache_dir).as_posix()
        if relative_path == _TOKENIZED_CACHE_READY_FILENAME:
            continue
        if path.is_symlink():
            raise ValueError(
                f"Tokenized dataset cache contains a symbolic link: {relative_path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"Tokenized dataset cache contains a non-regular path: {relative_path}"
            )
        records.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"Tokenized dataset cache has no persisted files: {cache_dir}")
    return records


def _build_tokenized_dataset_identity(
    tokenized: Dataset,
    *,
    persisted_files: list[dict[str, object]],
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema": _TOKENIZED_DATASET_IDENTITY_SCHEMA,
        "cache_format_version": _TOKENIZED_CACHE_FORMAT_VERSION,
        "ordered_content_schema": _TOKENIZED_ORDERED_CONTENT_SCHEMA,
        "ordered_content_sha256": _tokenized_dataset_ordered_sha256(tokenized),
        "rows": len(tokenized),
        "column_names": list(tokenized.column_names),
        "saved_fingerprint": getattr(tokenized, "_fingerprint", None),
        "persisted_files": persisted_files,
        "persisted_files_sha256": hashlib.sha256(
            _canonical_json_bytes(persisted_files)
        ).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    return identity


def _validate_expected_tokenized_dataset_sha256(
    args: argparse.Namespace,
    identity: dict[str, object],
) -> None:
    expected = _normalized_expected_tokenized_dataset_sha256(args)
    if expected is None:
        return
    actual = identity.get("ordered_content_sha256")
    if actual != expected:
        raise ValueError(
            "Tokenized dataset ordered-content SHA-256 differs from the launch lock: "
            f"expected={expected} actual={actual}"
        )


def _tokenized_cache_ready_payload(
    *,
    args: argparse.Namespace,
    cache_key: str,
    built_fingerprint: str | None,
    identity: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": _TOKENIZED_CACHE_READY_SCHEMA,
        "cache_format_version": _TOKENIZED_CACHE_FORMAT_VERSION,
        "cache_key": cache_key,
        "preparation": {
            "training_mode": args.training_mode,
            "group_by_length": args.group_by_length,
            "assistant_loss_mode": args.assistant_loss_mode,
            "episode_recent_messages": args.episode_recent_messages,
            "max_write_length": args.max_write_length,
            "memory_write_granularity": args.memory_write_granularity,
            "include_sentence_ids": args.memory_write_granularity == "sentence_mean",
            "require_scene_boundary_payload_mask": (
                getattr(args, "scene_boundary_payload_ce_weight", 0.0) > 0.0
            ),
            "require_scene_state_semantic_mask": (
                getattr(args, "memory_loss_mode", None)
                in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
            ),
            "require_scene_state_generation_masks": (
                getattr(args, "memory_loss_mode", None)
                == "scene_state_generation_ce"
            ),
            "max_length": args.max_length,
        },
        "built_fingerprint": built_fingerprint,
        "identity": identity,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"Atomic JSON temporary path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_tokenized_cache_ready(
    ready_marker: Path,
    *,
    cache_key: str,
) -> dict[str, object]:
    if not ready_marker.is_file() or ready_marker.is_symlink():
        raise ValueError(f"Tokenized dataset cache ready marker is invalid: {ready_marker}")
    try:
        payload = json.loads(ready_marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Tokenized dataset cache ready marker is invalid JSON: {ready_marker}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Tokenized dataset cache ready marker must be a JSON object")
    if payload.get("schema") != _TOKENIZED_CACHE_READY_SCHEMA:
        raise ValueError("Tokenized dataset cache ready-marker schema differs")
    if payload.get("cache_format_version") != _TOKENIZED_CACHE_FORMAT_VERSION:
        raise ValueError("Tokenized dataset cache format version differs")
    if payload.get("cache_key") != cache_key:
        raise ValueError("Tokenized dataset cache ready-marker key differs")
    recorded_manifest_sha256 = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual_manifest_sha256 = hashlib.sha256(
        _canonical_json_bytes(unsigned)
    ).hexdigest()
    if recorded_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("Tokenized dataset cache ready-marker checksum differs")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Tokenized dataset cache ready marker omits its identity")
    return payload


def _validate_tokenized_dataset_identity_shape(identity: dict[str, object]) -> None:
    if identity.get("schema") != _TOKENIZED_DATASET_IDENTITY_SCHEMA:
        raise ValueError("Tokenized dataset cache identity schema differs")
    if identity.get("cache_format_version") != _TOKENIZED_CACHE_FORMAT_VERSION:
        raise ValueError("Tokenized dataset cache identity format version differs")
    if identity.get("ordered_content_schema") != _TOKENIZED_ORDERED_CONTENT_SCHEMA:
        raise ValueError("Tokenized dataset ordered-content schema differs")
    ordered_sha256 = identity.get("ordered_content_sha256")
    if not isinstance(ordered_sha256, str) or len(ordered_sha256) != 64:
        raise ValueError("Tokenized dataset ordered-content SHA-256 is invalid")
    persisted_files = identity.get("persisted_files")
    if not isinstance(persisted_files, list) or not persisted_files:
        raise ValueError("Tokenized dataset cache identity omits persisted files")
    unsigned = dict(identity)
    recorded_identity_sha256 = unsigned.pop("identity_sha256", None)
    actual_identity_sha256 = hashlib.sha256(
        _canonical_json_bytes(unsigned)
    ).hexdigest()
    if recorded_identity_sha256 != actual_identity_sha256:
        raise ValueError("Tokenized dataset cache identity checksum differs")


def _load_verified_tokenized_cache(
    args: argparse.Namespace,
    *,
    cache_dir: Path,
    cache_key: str,
) -> tuple[Dataset, dict[str, object], str]:
    ready_payload = _read_tokenized_cache_ready(
        cache_dir / _TOKENIZED_CACHE_READY_FILENAME,
        cache_key=cache_key,
    )
    recorded_identity = ready_payload["identity"]
    assert isinstance(recorded_identity, dict)
    _validate_tokenized_dataset_identity_shape(recorded_identity)
    recorded_files = recorded_identity.get("persisted_files")
    actual_files_before = _tokenized_cache_persisted_files(cache_dir)
    if actual_files_before != recorded_files:
        raise ValueError("Tokenized dataset cache persisted files differ from its manifest")
    tokenized = load_from_disk(
        str(cache_dir),
        keep_in_memory=_normalized_expected_tokenized_dataset_sha256(args) is not None,
    )
    if not isinstance(tokenized, Dataset):
        raise ValueError("Tokenized dataset cache must contain one Dataset")
    actual_identity = _build_tokenized_dataset_identity(
        tokenized,
        persisted_files=actual_files_before,
    )
    actual_files_after = _tokenized_cache_persisted_files(cache_dir)
    if actual_files_after != actual_files_before:
        raise ValueError("Tokenized dataset cache changed while it was being validated")
    if actual_identity != recorded_identity:
        raise ValueError("Tokenized dataset cache content identity differs from its manifest")
    _validate_expected_tokenized_dataset_sha256(args, actual_identity)
    manifest_sha256 = ready_payload["manifest_sha256"]
    assert isinstance(manifest_sha256, str)
    return tokenized, actual_identity, manifest_sha256


def validate_tokenized_cache_directory(
    cache_dir: Path,
    expected_cache_key: str,
    expected_ordered_sha256: str,
) -> dict[str, object]:
    cache_dir = cache_dir.expanduser().resolve()
    if cache_dir.name != expected_cache_key:
        raise ValueError(
            "Tokenized dataset cache directory name differs from the expected key"
        )
    ready_marker = cache_dir / _TOKENIZED_CACHE_READY_FILENAME
    if not ready_marker.is_file() or ready_marker.is_symlink():
        raise ValueError(
            f"Tokenized dataset cache ready marker is invalid: {ready_marker}"
        )
    ready_file_sha256_before = _sha256_file(ready_marker)
    validation_args = argparse.Namespace(
        expected_tokenized_dataset_sha256=expected_ordered_sha256,
    )
    tokenized, identity, manifest_sha256 = _load_verified_tokenized_cache(
        validation_args,
        cache_dir=cache_dir,
        cache_key=expected_cache_key,
    )
    ready_file_sha256_after = _sha256_file(ready_marker)
    if ready_file_sha256_after != ready_file_sha256_before:
        raise ValueError(
            "Tokenized dataset cache ready marker changed while it was being validated"
        )
    del tokenized
    return {
        "identity": identity,
        "manifest_sha256": manifest_sha256,
        "ready_file_sha256": ready_file_sha256_after,
    }


def _tokenized_dataset_cache_key(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
) -> str:
    code_hash = hashlib.sha256()
    for fn in (
        normalize_example,
        _scene_boundary_payload_char_span,
        _rendered_message_content_span,
        _tokenizer_ids_and_offsets,
        _scene_boundary_payload_token_mask,
        _scene_boundary_semantic_token_mask,
        _scene_state_generation_token_masks,
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
        "cache_format_version": _TOKENIZED_CACHE_FORMAT_VERSION,
        "ordered_content_schema": _TOKENIZED_ORDERED_CONTENT_SCHEMA,
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
        "require_scene_boundary_payload_mask": (
            getattr(args, "scene_boundary_payload_ce_weight", 0.0) > 0.0
        ),
        "require_scene_state_semantic_mask": (
            getattr(args, "memory_loss_mode", None)
            in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
        ),
        "require_scene_state_generation_masks": (
            getattr(args, "memory_loss_mode", None)
            == "scene_state_generation_ce"
        ),
        "group_by_length": args.group_by_length,
        "dataset_num_proc": getattr(args, "dataset_num_proc", 1),
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
                require_scene_boundary_payload_mask=(
                    getattr(args, "scene_boundary_payload_ce_weight", 0.0) > 0.0
                ),
                require_scene_state_semantic_mask=(
                    getattr(args, "memory_loss_mode", None)
                    in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
                ),
                require_scene_state_generation_masks=(
                    getattr(args, "memory_loss_mode", None)
                    == "scene_state_generation_ce"
                ),
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
                require_scene_boundary_payload_mask=(
                    getattr(args, "scene_boundary_payload_ce_weight", 0.0) > 0.0
                ),
                require_scene_state_semantic_mask=(
                    getattr(args, "memory_loss_mode", None)
                    in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
                ),
                require_scene_state_generation_masks=(
                    getattr(args, "memory_loss_mode", None)
                    == "scene_state_generation_ce"
                ),
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


def _prepare_tokenized_dataset_with_identity(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[
    Dataset,
    bool,
    Path | None,
    dict[str, object],
    str | None,
]:
    if not args.tokenized_cache:
        tokenized = _build_tokenized_dataset(args, dataset, tokenizer)
        identity = _build_tokenized_dataset_identity(tokenized, persisted_files=[])
        _validate_expected_tokenized_dataset_sha256(args, identity)
        return tokenized, False, None, identity, None
    if args.tokenized_dataset_root is None:
        raise ValueError("--tokenized-dataset-root is required when --tokenized-cache is enabled")

    args.tokenized_dataset_root.mkdir(parents=True, exist_ok=True)
    cache_key = _tokenized_dataset_cache_key(args, dataset, tokenizer)
    cache_dir = args.tokenized_dataset_root / cache_key
    ready_marker = cache_dir / _TOKENIZED_CACHE_READY_FILENAME
    lock_dir = args.tokenized_dataset_root / f".{cache_key}.lock"
    is_builder = (not distributed) or local_rank in (-1, 0)

    if ready_marker.exists():
        tokenized, identity, manifest_sha256 = _load_verified_tokenized_cache(
            args,
            cache_dir=cache_dir,
            cache_key=cache_key,
        )
        return tokenized, True, cache_dir, identity, manifest_sha256

    if is_builder:
        while True:
            try:
                lock_dir.mkdir()
                break
            except FileExistsError:
                if ready_marker.exists():
                    tokenized, identity, manifest_sha256 = _load_verified_tokenized_cache(
                        args,
                        cache_dir=cache_dir,
                        cache_key=cache_key,
                    )
                    return tokenized, True, cache_dir, identity, manifest_sha256
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
            persisted_files_before = _tokenized_cache_persisted_files(cache_dir)
            tokenized = load_from_disk(
                str(cache_dir),
                keep_in_memory=(
                    _normalized_expected_tokenized_dataset_sha256(args) is not None
                ),
            )
            if not isinstance(tokenized, Dataset):
                raise ValueError("Tokenized dataset cache must contain one Dataset")
            identity = _build_tokenized_dataset_identity(
                tokenized,
                persisted_files=persisted_files_before,
            )
            persisted_files_after = _tokenized_cache_persisted_files(cache_dir)
            if persisted_files_after != persisted_files_before:
                raise ValueError(
                    "Tokenized dataset cache changed while it was being prepared"
                )
            _validate_expected_tokenized_dataset_sha256(args, identity)
            ready_payload = _tokenized_cache_ready_payload(
                args=args,
                cache_key=cache_key,
                built_fingerprint=built_fingerprint,
                identity=identity,
            )
            _write_json_atomic(ready_marker, ready_payload)
            manifest_sha256 = ready_payload["manifest_sha256"]
            assert isinstance(manifest_sha256, str)
            return tokenized, False, cache_dir, identity, manifest_sha256
        finally:
            if lock_dir.exists():
                lock_dir.rmdir()

    waited = 0
    while not ready_marker.exists():
        time.sleep(2)
        waited += 2
        if waited > 7200:
            raise TimeoutError(f"Timed out waiting for tokenized dataset cache at {cache_dir}")
    tokenized, identity, manifest_sha256 = _load_verified_tokenized_cache(
        args,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )
    return tokenized, True, cache_dir, identity, manifest_sha256


def prepare_tokenized_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer,
    *,
    distributed: bool,
    local_rank: int,
) -> tuple[Dataset, bool, Path | None]:
    tokenized, cache_hit, cache_dir, _, _ = _prepare_tokenized_dataset_with_identity(
        args,
        dataset,
        tokenizer,
        distributed=distributed,
        local_rank=local_rank,
    )
    return tokenized, cache_hit, cache_dir


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
        tokenized_path = args.tokenized_dataset_dir.expanduser().resolve()
        ready_marker = tokenized_path / _TOKENIZED_CACHE_READY_FILENAME
        if ready_marker.exists():
            tokenized, identity, manifest_sha256 = _load_verified_tokenized_cache(
                args,
                cache_dir=tokenized_path,
                cache_key=tokenized_path.name,
            )
        else:
            persisted_files_before = _tokenized_cache_persisted_files(tokenized_path)
            tokenized = load_from_disk(
                str(tokenized_path),
                keep_in_memory=(
                    _normalized_expected_tokenized_dataset_sha256(args) is not None
                ),
            )
            if not isinstance(tokenized, Dataset):
                raise ValueError("--tokenized-dataset-dir must contain one Dataset")
            identity = _build_tokenized_dataset_identity(
                tokenized,
                persisted_files=persisted_files_before,
            )
            persisted_files_after = _tokenized_cache_persisted_files(tokenized_path)
            if persisted_files_after != persisted_files_before:
                raise ValueError(
                    "Tokenized dataset directory changed while it was being validated"
                )
            _validate_expected_tokenized_dataset_sha256(args, identity)
            manifest_sha256 = None
        return tokenized, {
            "tokenized_cache_hit": True,
            "tokenized_cache_dir": str(tokenized_path),
            "tokenized_dataset_source": "load_from_disk",
            "train_samples": len(tokenized),
            "training_mode": detect_training_mode(tokenized),
            "tokenized_dataset_sha256": identity["ordered_content_sha256"],
            "tokenized_cache_identity": identity,
            "tokenized_cache_manifest_sha256": manifest_sha256,
        }

    dataset = load_examples(args)
    (
        tokenized,
        tokenized_cache_hit,
        tokenized_cache_dir,
        identity,
        manifest_sha256,
    ) = _prepare_tokenized_dataset_with_identity(
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
        "tokenized_dataset_sha256": identity["ordered_content_sha256"],
        "tokenized_cache_identity": identity,
        "tokenized_cache_manifest_sha256": manifest_sha256,
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
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _content_contrast_supervised_trace(
    row: dict,
    *,
    row_name: str,
) -> list[dict[str, int]]:
    required = ("input_ids", "attention_mask", "labels")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            f"content_contrast_ce {row_name} row is missing: " + ", ".join(missing)
        )
    input_ids = [int(value) for value in row["input_ids"]]
    attention_mask = [int(value) for value in row["attention_mask"]]
    labels = [int(value) for value in row["labels"]]
    if not len(input_ids) == len(attention_mask) == len(labels):
        raise ValueError(
            f"content_contrast_ce {row_name} input IDs, attention mask, and labels "
            "must have equal lengths"
        )
    if labels and labels[0] != -100 and attention_mask[0] != 0:
        raise ValueError(
            f"content_contrast_ce {row_name} has a supervised target without a "
            "causal predictor"
        )

    trace: list[dict[str, int]] = []
    for label_position in range(1, len(labels)):
        label = labels[label_position]
        if label == -100 or attention_mask[label_position] == 0:
            continue
        predictor_position = label_position - 1
        if attention_mask[predictor_position] == 0:
            raise ValueError(
                f"content_contrast_ce {row_name} supervised target at "
                f"{label_position} has a masked causal predictor"
            )
        if input_ids[label_position] != label:
            raise ValueError(
                f"content_contrast_ce {row_name} supervised label does not match "
                f"its input token at position {label_position}"
            )
        trace.append(
            {
                "ordinal": len(trace),
                "token_id": label,
                "label_position": label_position,
                "predictor_position": predictor_position,
            }
        )
    if not trace:
        raise ValueError(
            f"content_contrast_ce {row_name} row has no supervised next-token targets"
        )
    return trace


def _select_content_contrast_target_span_with_metadata(
    source_row: dict,
    donor_row: dict,
    *,
    span_tokens: int,
) -> tuple[list[bool], dict[str, object]]:
    if span_tokens <= 0:
        raise ValueError("content_contrast_ce target span_tokens must be positive")
    source_trace = _content_contrast_supervised_trace(source_row, row_name="source")
    donor_trace = _content_contrast_supervised_trace(donor_row, row_name="donor")
    if len(source_trace) != len(donor_trace):
        raise ValueError(
            "content_contrast_ce source and donor supervised target counts differ: "
            f"source={len(source_trace)} donor={len(donor_trace)}"
        )
    source_positions = [item["label_position"] for item in source_trace]
    donor_positions = [item["label_position"] for item in donor_trace]
    if source_positions != donor_positions:
        raise ValueError(
            "content_contrast_ce source and donor supervised target positions differ"
        )

    first_differing_ordinal = next(
        (
            ordinal
            for ordinal, (source_item, donor_item) in enumerate(
                zip(source_trace, donor_trace)
            )
            if source_item["token_id"] != donor_item["token_id"]
        ),
        None,
    )
    if first_differing_ordinal is None:
        raise ValueError(
            "content_contrast_ce source and donor have identical supervised answers"
        )
    if len(source_trace) - first_differing_ordinal < span_tokens:
        raise ValueError(
            f"content_contrast_ce fewer than {span_tokens} supervised targets remain "
            "after the first differing target"
        )

    first_label_position = source_trace[first_differing_ordinal]["label_position"]
    source_prefix = [
        int(value) for value in source_row["input_ids"][:first_label_position]
    ]
    donor_prefix = [
        int(value) for value in donor_row["input_ids"][:first_label_position]
    ]
    if source_prefix != donor_prefix:
        raise ValueError(
            "content_contrast_ce source and donor causal prefix differs before the "
            "first selected target"
        )
    source_attention_prefix = [
        int(value)
        for value in source_row["attention_mask"][:first_label_position]
    ]
    donor_attention_prefix = [
        int(value)
        for value in donor_row["attention_mask"][:first_label_position]
    ]
    if source_attention_prefix != donor_attention_prefix:
        raise ValueError(
            "content_contrast_ce source and donor causal attention-mask prefix differs "
            "before the first selected target"
        )

    selected_trace = source_trace[
        first_differing_ordinal : first_differing_ordinal + span_tokens
    ]
    donor_selected_trace = donor_trace[
        first_differing_ordinal : first_differing_ordinal + span_tokens
    ]
    target_mask = [False] * len(source_row["labels"])
    for item in selected_trace:
        target_mask[item["label_position"]] = True
    metadata: dict[str, object] = {
        "target_mode": _CONTENT_CONTRAST_TARGET_MODE,
        "target_span_tokens": span_tokens,
        "first_differing_supervised_ordinal": first_differing_ordinal,
        "first_target_label_position": first_label_position,
        "first_target_predictor_position": first_label_position - 1,
        "target_label_positions": [item["label_position"] for item in selected_trace],
        "target_token_ids": [item["token_id"] for item in selected_trace],
        "donor_token_ids": [item["token_id"] for item in donor_selected_trace],
    }
    return target_mask, metadata


def select_content_contrast_target_span(
    source_row: dict,
    donor_row: dict,
    *,
    span_tokens: int = _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
) -> list[bool]:
    target_mask, _ = _select_content_contrast_target_span_with_metadata(
        source_row,
        donor_row,
        span_tokens=span_tokens,
    )
    return target_mask


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
        "content_contrast_target_mask",
        "content_contrast_target_mask_sha256",
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
    target_masks: list[list[bool]] = []
    target_mask_hashes: list[str] = []
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
        target_mask, target_metadata = (
            _select_content_contrast_target_span_with_metadata(
                rows[source_index],
                rows[partner_index],
                span_tokens=_CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
            )
        )
        target_mask_hash = _canonical_json_sha256(target_mask)
        target_masks.append(target_mask)
        target_mask_hashes.append(target_mask_hash)
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
                **target_metadata,
                "target_mask_sha256": target_mask_hash,
            }
        )

    paired = split
    for column, values in negative_columns.items():
        paired = paired.add_column(column, values)
    paired = paired.add_column("content_contrast_target_mask", target_masks)
    paired = paired.add_column(
        "content_contrast_target_mask_sha256",
        target_mask_hashes,
    )
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
        "target_mode": _CONTENT_CONTRAST_TARGET_MODE,
        "target_span_tokens": _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
        "target_token_count": sample_count * _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
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
    target_token_count = sum(
        int(split_manifest["target_token_count"])
        for split_manifest in splits.values()
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "objective_version": _CONTENT_CONTRAST_OBJECTIVE_VERSION,
        "pairing_version": _CONTENT_CONTRAST_PAIRING_VERSION,
        "pairing_scope": "within_post_split_partition",
        "target_mode": _CONTENT_CONTRAST_TARGET_MODE,
        "target_span_tokens": _CONTENT_CONTRAST_TARGET_SPAN_TOKENS,
        "target_token_count": target_token_count,
        "data_seed": data_seed,
        "tokenized_fingerprint": tokenized_fingerprint,
        "splits": splits,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def _scene_state_semantic_trace(
    row: dict,
    *,
    row_name: str,
) -> list[dict[str, int]]:
    required = (
        "input_ids",
        "attention_mask",
        "labels",
        "scene_state_semantic_mask",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(
            f"scene_state_identity_ce {row_name} row is missing: "
            + ", ".join(missing)
        )
    input_ids = [int(value) for value in row["input_ids"]]
    attention_mask = [int(value) for value in row["attention_mask"]]
    labels = [int(value) for value in row["labels"]]
    semantic_mask = [bool(value) for value in row["scene_state_semantic_mask"]]
    if not (
        len(input_ids)
        == len(attention_mask)
        == len(labels)
        == len(semantic_mask)
    ):
        raise ValueError(
            f"scene_state_identity_ce {row_name} semantic tensors must align"
        )
    trace: list[dict[str, int]] = []
    for label_position in range(1, len(labels)):
        if not semantic_mask[label_position]:
            continue
        if labels[label_position] == -100 or attention_mask[label_position] == 0:
            raise ValueError(
                f"scene_state_identity_ce {row_name} semantic target is not supervised"
            )
        predictor_position = label_position - 1
        if attention_mask[predictor_position] == 0:
            raise ValueError(
                f"scene_state_identity_ce {row_name} semantic target has no predictor"
            )
        if input_ids[label_position] != labels[label_position]:
            raise ValueError(
                f"scene_state_identity_ce {row_name} label differs from its input token"
            )
        trace.append(
            {
                "ordinal": len(trace),
                "token_id": labels[label_position],
                "label_position": label_position,
                "predictor_position": predictor_position,
            }
        )
    if not trace:
        raise ValueError(
            f"scene_state_identity_ce {row_name} row has no semantic targets"
        )
    return trace


def _scene_state_label_identity(row: dict, *, row_name: str) -> dict[str, object]:
    trace = _scene_state_semantic_trace(row, row_name=row_name)
    token_ids = [item["token_id"] for item in trace]
    return {
        "semantic_token_ids": token_ids,
        "semantic_token_count": len(token_ids),
        "label_sha256": _canonical_json_sha256(token_ids),
    }


def _select_scene_state_identity_target_with_metadata(
    source_row: dict,
    donor_row: dict,
) -> tuple[list[bool], dict[str, object]]:
    source_trace = _scene_state_semantic_trace(source_row, row_name="source")
    donor_trace = _scene_state_semantic_trace(donor_row, row_name="donor")
    first_differing_ordinal = next(
        (
            ordinal
            for ordinal, (source_item, donor_item) in enumerate(
                zip(source_trace, donor_trace)
            )
            if source_item["token_id"] != donor_item["token_id"]
        ),
        min(len(source_trace), len(donor_trace)),
    )
    if (
        first_differing_ordinal == len(source_trace)
        and first_differing_ordinal == len(donor_trace)
    ):
        raise ValueError(
            "scene_state_identity_ce donor must have an exact-distinct semantic label"
        )
    if first_differing_ordinal >= len(source_trace):
        raise ValueError(
            "scene_state_identity_ce source semantic sequence ends before its first "
            "donor distinction and cannot supply a supervised target"
        )
    source_item = source_trace[first_differing_ordinal]
    if first_differing_ordinal >= len(donor_trace):
        raise ValueError(
            "scene_state_identity_ce donor semantic sequence ends before the "
            "distinguishing source target"
        )
    donor_item = donor_trace[first_differing_ordinal]
    source_prefix = [
        int(token_id)
        for token_id in source_row["input_ids"][: source_item["label_position"]]
    ]
    donor_prefix = [
        int(token_id)
        for token_id in donor_row["input_ids"][: donor_item["label_position"]]
    ]
    source_prefix_attention = [
        int(value)
        for value in source_row["attention_mask"][: source_item["label_position"]]
    ]
    donor_prefix_attention = [
        int(value)
        for value in donor_row["attention_mask"][: donor_item["label_position"]]
    ]
    if (
        source_prefix != donor_prefix
        or source_prefix_attention != donor_prefix_attention
    ):
        raise ValueError(
            "scene_state_identity_ce source and donor causal prefixes differ before "
            "their first semantic distinction"
        )
    target_mask = [False] * len(source_row["labels"])
    target_mask[source_item["label_position"]] = True
    metadata: dict[str, object] = {
        "target_mode": _SCENE_STATE_IDENTITY_TARGET_MODE,
        "causal_prefix_mode": _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "target_span_tokens": 1,
        "first_differing_semantic_ordinal": first_differing_ordinal,
        "target_label_positions": [source_item["label_position"]],
        "target_predictor_positions": [source_item["predictor_position"]],
        "target_token_ids": [source_item["token_id"]],
        "donor_target_label_positions": [donor_item["label_position"]],
        "donor_target_predictor_positions": [donor_item["predictor_position"]],
        "donor_token_ids": [donor_item["token_id"]],
        "causal_prefix_token_count": len(source_prefix),
        "causal_prefix_sha256": _canonical_json_sha256(source_prefix),
    }
    return target_mask, metadata


def select_scene_state_identity_target(
    source_row: dict,
    donor_row: dict,
) -> list[bool]:
    target_mask, _ = _select_scene_state_identity_target_with_metadata(
        source_row,
        donor_row,
    )
    return target_mask


def _scene_state_identity_target_stratum(
    source_boundary_count: int,
    donor_boundary_count: int,
) -> str:
    for name, value in (
        ("source", source_boundary_count),
        ("donor", donor_boundary_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"Scene-state {name} boundaries cardinality must be a non-negative integer"
            )
    if (source_boundary_count == 0) != (donor_boundary_count == 0):
        return "presence"
    if source_boundary_count == donor_boundary_count:
        return "same_cardinality_value"
    return "cross_cardinality_value"


def materialize_scene_state_identity_pairs(
    split: Dataset,
    *,
    split_name: str,
) -> tuple[Dataset, dict[str, object]]:
    sample_count = len(split)
    if sample_count < 2 or sample_count % 2 != 0:
        raise ValueError(
            f"scene_state_identity_ce split {split_name!r} requires a positive even "
            f"sample count; got {sample_count}"
        )
    materialized_columns = {
        "scene_state_donor_write_input_ids",
        "scene_state_donor_write_attention_mask",
        "scene_state_donor_write_message_ids",
        "scene_state_donor_write_sentence_ids",
        "scene_state_donor_boundary_count",
        "scene_state_identity_target_mask",
        "scene_state_identity_target_mask_sha256",
        "scene_state_identity_target_stratum",
        "scene_state_source_index",
        "scene_state_donor_index",
        "scene_state_source_row_sha256",
        "scene_state_donor_row_sha256",
        "scene_state_source_label_sha256",
        "scene_state_donor_label_sha256",
        "scene_state_source_write_sha256",
        "scene_state_donor_write_sha256",
    }
    collisions = sorted(materialized_columns.intersection(split.column_names))
    if collisions:
        raise ValueError(
            "scene_state_identity_ce pairing requires an objective-neutral post-split "
            "dataset; columns already exist: " + ", ".join(collisions)
        )

    source_fingerprint = getattr(split, "_fingerprint", None)
    rows = [split[index] for index in range(sample_count)]
    boundary_counts: list[int] = []
    for index, row in enumerate(rows):
        if "scene_state_boundary_count" not in row:
            raise ValueError(
                "scene_state_identity_ce pairing requires scene_state_boundary_count; "
                f"missing at {split_name}:{index}"
            )
        boundary_count = row["scene_state_boundary_count"]
        _scene_state_identity_target_stratum(boundary_count, boundary_count)
        boundary_counts.append(boundary_count)
    writes = [_content_contrast_write_payload(row) for row in rows]
    write_hashes = [_canonical_json_sha256(payload) for payload in writes]
    row_hashes = [_canonical_json_sha256(row) for row in rows]
    labels = [
        _scene_state_label_identity(row, row_name=f"{split_name}:{index}")
        for index, row in enumerate(rows)
    ]
    ordered_indices = sorted(
        range(sample_count),
        key=lambda index: (
            len(writes[index]["input_ids"]),
            row_hashes[index],
        ),
    )

    def pair_remaining(
        remaining: tuple[int, ...],
    ) -> list[tuple[int, int]] | None:
        if not remaining:
            return []
        source_index = remaining[0]
        candidates = sorted(
            (
                candidate_ordinal
                for candidate_ordinal in range(1, len(remaining))
                if labels[remaining[candidate_ordinal]]["label_sha256"]
                != labels[source_index]["label_sha256"]
            ),
            key=lambda candidate_ordinal: (
                abs(
                    len(writes[remaining[candidate_ordinal]]["input_ids"])
                    - len(writes[source_index]["input_ids"])
                ),
                len(writes[remaining[candidate_ordinal]]["input_ids"]),
                row_hashes[remaining[candidate_ordinal]],
            ),
        )
        for candidate_ordinal in candidates:
            donor_index = remaining[candidate_ordinal]
            tail = remaining[1:candidate_ordinal] + remaining[candidate_ordinal + 1 :]
            paired_tail = pair_remaining(tail)
            if paired_tail is not None:
                return [(source_index, donor_index), *paired_tail]
        return None

    pairs = pair_remaining(tuple(ordered_indices))
    if pairs is None:
        raise ValueError(
            "No complete nearest-feasible write-length exact-label-distinct donor "
            "pairing exists"
        )

    def pairing_length_stats(
        candidate_pairs: list[tuple[int, int]],
    ) -> tuple[int, int]:
        deltas = [
            abs(
                len(writes[left]["input_ids"])
                - len(writes[right]["input_ids"])
            )
            for left, right in candidate_pairs
        ]
        return sum(deltas), max(deltas)

    def nonempty_same_cardinality_pair_count(
        candidate_pairs: list[tuple[int, int]],
    ) -> int:
        return sum(
            boundary_counts[left] > 0
            and boundary_counts[left] == boundary_counts[right]
            for left, right in candidate_pairs
        )

    baseline_pairs = [tuple(sorted(pair)) for pair in pairs]
    baseline_pairs.sort()
    baseline_total_delta, baseline_max_delta = pairing_length_stats(
        baseline_pairs
    )
    valid_edges = [
        (left, right)
        for left in range(sample_count)
        for right in range(left + 1, sample_count)
        if labels[left]["label_sha256"] != labels[right]["label_sha256"]
        and abs(
            len(writes[left]["input_ids"])
            - len(writes[right]["input_ids"])
        )
        <= baseline_max_delta
    ]
    ordered_edges = sorted(
        valid_edges,
        key=lambda edge: (
            min(row_hashes[edge[0]], row_hashes[edge[1]]),
            max(row_hashes[edge[0]], row_hashes[edge[1]]),
            edge,
        ),
    )
    edge_count = len(ordered_edges)
    pair_count = sample_count // 2
    tie_scale = pair_count * max(edge_count, 1) + 1
    primary_scale = (pair_count * baseline_max_delta + 1) * tie_scale
    graph = nx.Graph()
    graph.add_nodes_from(range(sample_count))
    for edge_rank, (left, right) in enumerate(ordered_edges):
        length_delta = abs(
            len(writes[left]["input_ids"])
            - len(writes[right]["input_ids"])
        )
        is_nonempty_same_cardinality = (
            boundary_counts[left] > 0
            and boundary_counts[left] == boundary_counts[right]
        )
        graph.add_edge(
            left,
            right,
            weight=(
                (0 if is_nonempty_same_cardinality else primary_scale)
                + length_delta * tie_scale
                + edge_rank
            ),
        )
    refined_matching = nx.min_weight_matching(graph, weight="weight")
    refined_pairs = sorted(
        tuple(sorted((int(left), int(right))))
        for left, right in refined_matching
    )
    if len(refined_pairs) == pair_count:
        refined_total_delta, refined_max_delta = pairing_length_stats(
            refined_pairs
        )
        baseline_objective = (
            -nonempty_same_cardinality_pair_count(baseline_pairs),
            baseline_total_delta,
            baseline_max_delta,
            baseline_pairs,
        )
        refined_objective = (
            -nonempty_same_cardinality_pair_count(refined_pairs),
            refined_total_delta,
            refined_max_delta,
            refined_pairs,
        )
        if (
            refined_total_delta <= baseline_total_delta
            and refined_max_delta <= baseline_max_delta
            and refined_objective < baseline_objective
        ):
            pairs = refined_pairs
        else:
            pairs = baseline_pairs
    else:
        pairs = baseline_pairs
    pairing_refinement_applied = pairs != baseline_pairs

    donor_indices = [-1] * sample_count
    for left, right in pairs:
        donor_indices[left] = right
        donor_indices[right] = left
    if any(index < 0 for index in donor_indices):
        raise RuntimeError("Scene-state donor pairing did not cover every row")

    donor_columns: dict[str, list[list[int]]] = {
        "scene_state_donor_write_input_ids": [],
        "scene_state_donor_write_attention_mask": [],
        "scene_state_donor_write_message_ids": [],
        "scene_state_donor_write_sentence_ids": [],
    }
    target_masks: list[list[bool]] = []
    target_mask_hashes: list[str] = []
    target_strata: list[str] = []
    pair_audit: list[dict[str, object]] = []
    for source_index, donor_index in enumerate(donor_indices):
        if donor_indices[donor_index] != source_index:
            raise RuntimeError("Scene-state donor map is not symmetric")
        for field, column in (
            ("input_ids", "scene_state_donor_write_input_ids"),
            ("attention_mask", "scene_state_donor_write_attention_mask"),
            ("message_ids", "scene_state_donor_write_message_ids"),
            ("sentence_ids", "scene_state_donor_write_sentence_ids"),
        ):
            donor_columns[column].append(writes[donor_index][field])
        target_mask, target_metadata = (
            _select_scene_state_identity_target_with_metadata(
                rows[source_index],
                rows[donor_index],
            )
        )
        target_mask_hash = _canonical_json_sha256(target_mask)
        target_stratum = _scene_state_identity_target_stratum(
            boundary_counts[source_index],
            boundary_counts[donor_index],
        )
        target_masks.append(target_mask)
        target_mask_hashes.append(target_mask_hash)
        target_strata.append(target_stratum)
        pair_audit.append(
            {
                "source_index": source_index,
                "donor_index": donor_index,
                "source_row_sha256": row_hashes[source_index],
                "donor_row_sha256": row_hashes[donor_index],
                "source_label_sha256": labels[source_index]["label_sha256"],
                "donor_label_sha256": labels[donor_index]["label_sha256"],
                "source_write_sha256": write_hashes[source_index],
                "donor_write_sha256": write_hashes[donor_index],
                "source_write_token_count": len(writes[source_index]["input_ids"]),
                "donor_write_token_count": len(writes[donor_index]["input_ids"]),
                "source_boundary_count": boundary_counts[source_index],
                "donor_boundary_count": boundary_counts[donor_index],
                "target_stratum": target_stratum,
                "write_token_count_delta": abs(
                    len(writes[source_index]["input_ids"])
                    - len(writes[donor_index]["input_ids"])
                ),
                **target_metadata,
                "target_mask_sha256": target_mask_hash,
            }
        )

    paired = split
    for column, values in donor_columns.items():
        paired = paired.add_column(column, values)
    paired = paired.add_column(
        "scene_state_donor_boundary_count",
        [boundary_counts[index] for index in donor_indices],
    )
    paired = paired.add_column("scene_state_identity_target_mask", target_masks)
    paired = paired.add_column(
        "scene_state_identity_target_mask_sha256",
        target_mask_hashes,
    )
    paired = paired.add_column(
        "scene_state_identity_target_stratum",
        target_strata,
    )
    paired = paired.add_column("scene_state_source_index", list(range(sample_count)))
    paired = paired.add_column("scene_state_donor_index", donor_indices)
    paired = paired.add_column("scene_state_source_row_sha256", row_hashes)
    paired = paired.add_column(
        "scene_state_donor_row_sha256",
        [row_hashes[index] for index in donor_indices],
    )
    paired = paired.add_column(
        "scene_state_source_label_sha256",
        [str(identity["label_sha256"]) for identity in labels],
    )
    paired = paired.add_column(
        "scene_state_donor_label_sha256",
        [str(labels[index]["label_sha256"]) for index in donor_indices],
    )
    paired = paired.add_column("scene_state_source_write_sha256", write_hashes)
    paired = paired.add_column(
        "scene_state_donor_write_sha256",
        [write_hashes[index] for index in donor_indices],
    )

    target_stratum_row_counts = {
        stratum: target_strata.count(stratum)
        for stratum in _SCENE_STATE_IDENTITY_TARGET_STRATA
    }
    boundary_count_histogram = {
        str(boundary_count): boundary_counts.count(boundary_count)
        for boundary_count in sorted(set(boundary_counts))
    }
    write_token_count_deltas = [
        int(item["write_token_count_delta"])
        for item in pair_audit
    ]
    split_manifest: dict[str, object] = {
        "split": split_name,
        "pairing_version": _SCENE_STATE_IDENTITY_PAIRING_VERSION,
        "pairing_refinement": _SCENE_STATE_IDENTITY_PAIRING_REFINEMENT,
        "pairing_refinement_applied": pairing_refinement_applied,
        "target_mode": _SCENE_STATE_IDENTITY_TARGET_MODE,
        "causal_prefix_mode": _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "sample_count": sample_count,
        "pair_count": len(pairs),
        "target_token_count": sample_count,
        "target_stratum_row_counts": target_stratum_row_counts,
        "source_boundary_count_histogram": boundary_count_histogram,
        "write_token_count_delta_max": max(write_token_count_deltas),
        "write_token_count_delta_mean": (
            sum(write_token_count_deltas) / len(write_token_count_deltas)
        ),
        "write_token_count_delta_total": pairing_length_stats(pairs)[0],
        "nearest_baseline_write_token_count_delta_max": baseline_max_delta,
        "nearest_baseline_write_token_count_delta_total": baseline_total_delta,
        "source_fingerprint": source_fingerprint,
        "paired_fingerprint": getattr(paired, "_fingerprint", None),
        "pairs_sha256": _canonical_json_sha256(pair_audit),
        "pairs": pair_audit,
    }
    split_manifest["manifest_sha256"] = _canonical_json_sha256(split_manifest)
    return paired, split_manifest


def materialize_scene_state_generation_pairs(
    split: Dataset,
    *,
    split_name: str,
    pairing_binding: dict[str, object],
) -> tuple[Dataset, dict[str, object]]:
    if split_name != "train":
        raise ValueError(
            "scene_state_generation_ce accepts only the predeclared train pairing"
        )
    entries = pairing_binding.get("entries")
    if not isinstance(entries, list) or len(entries) != len(split):
        raise ValueError(
            "Scene-state V7 directed entries do not cover the tokenized train rows"
        )
    materialized_columns = {
        "scene_state_donor_write_input_ids",
        "scene_state_donor_write_attention_mask",
        "scene_state_donor_write_message_ids",
        "scene_state_donor_write_sentence_ids",
        "scene_state_donor_boundary_count",
        "scene_state_identity_target_mask",
        "scene_state_identity_target_mask_sha256",
        "scene_state_identity_target_stratum",
        "scene_state_identity_donor_target_token_id",
        "scene_state_source_index",
        "scene_state_donor_index",
        "scene_state_source_official_index",
        "scene_state_donor_official_index",
        "scene_state_source_row_sha256",
        "scene_state_donor_row_sha256",
        "scene_state_source_label_sha256",
        "scene_state_donor_label_sha256",
        "scene_state_source_write_sha256",
        "scene_state_donor_write_sha256",
    }
    collisions = sorted(materialized_columns.intersection(split.column_names))
    if collisions:
        raise ValueError(
            "scene_state_generation_ce pairing requires an objective-neutral post-split "
            "dataset; columns already exist: " + ", ".join(collisions)
        )

    rows = [split[index] for index in range(len(split))]
    writes = [_content_contrast_write_payload(row) for row in rows]
    write_hashes = [
        _canonical_json_sha256(payload["input_ids"])
        for payload in writes
    ]
    boundary_counts = [int(row["scene_state_boundary_count"]) for row in rows]
    donor_indices: list[int] = []
    donor_target_token_ids: list[int] = []
    target_masks: list[list[bool]] = []
    target_mask_hashes: list[str] = []
    target_strata: list[str] = []
    pair_audit: list[dict[str, object]] = []
    donor_columns: dict[str, list[list[int]]] = {
        "scene_state_donor_write_input_ids": [],
        "scene_state_donor_write_attention_mask": [],
        "scene_state_donor_write_message_ids": [],
        "scene_state_donor_write_sentence_ids": [],
    }
    for source_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("train_row_ordinal") != source_index:
            raise ValueError(
                f"Scene-state V7 entry order differs at train row {source_index}"
            )
        donor_index = entry.get("donor_train_row_ordinal")
        if isinstance(donor_index, bool) or not isinstance(donor_index, int):
            raise ValueError(
                f"Scene-state V7 donor ordinal is invalid at train row {source_index}"
            )
        donor_indices.append(donor_index)
        for field, column in (
            ("input_ids", "scene_state_donor_write_input_ids"),
            ("attention_mask", "scene_state_donor_write_attention_mask"),
            ("message_ids", "scene_state_donor_write_message_ids"),
            ("sentence_ids", "scene_state_donor_write_sentence_ids"),
        ):
            donor_columns[column].append(writes[donor_index][field])
        source_generation_mask = [
            bool(value)
            for value in rows[source_index]["scene_state_generation_target_mask"]
        ]
        donor_generation_mask = [
            bool(value)
            for value in rows[donor_index]["scene_state_generation_target_mask"]
        ]
        if not any(source_generation_mask) or not any(donor_generation_mask):
            raise ValueError(
                f"Scene-state V7 generation target is empty at row {source_index}"
            )
        source_generation_start = source_generation_mask.index(True)
        donor_generation_start = donor_generation_mask.index(True)
        expected_values = {
            "source_boundary_count": boundary_counts[source_index],
            "donor_boundary_count": boundary_counts[donor_index],
            "source_generation_prefix_sha256": _canonical_json_sha256(
                [
                    int(value)
                    for value in rows[source_index]["input_ids"][
                        :source_generation_start
                    ]
                ]
            ),
            "donor_generation_prefix_sha256": _canonical_json_sha256(
                [
                    int(value)
                    for value in rows[donor_index]["input_ids"][
                        :donor_generation_start
                    ]
                ]
            ),
            "source_write_sha256": write_hashes[source_index],
            "donor_write_sha256": write_hashes[donor_index],
            "source_write_token_count": len(writes[source_index]["input_ids"]),
            "donor_write_token_count": len(writes[donor_index]["input_ids"]),
            "write_token_count_delta": abs(
                len(writes[source_index]["input_ids"])
                - len(writes[donor_index]["input_ids"])
            ),
        }
        mismatches = {
            field: {"expected": expected, "actual": entry.get(field)}
            for field, expected in expected_values.items()
            if entry.get(field) != expected
        }
        if mismatches:
            raise ValueError(
                f"Scene-state V7 tokenized pairing differs at row {source_index}: "
                f"{mismatches}"
            )
        target_mask, target_metadata = (
            _select_scene_state_identity_target_with_metadata(
                rows[source_index],
                rows[donor_index],
            )
        )
        metadata_expected = {
            "first_differing_semantic_ordinal": target_metadata[
                "first_differing_semantic_ordinal"
            ],
            "selected_target_positions": target_metadata["target_label_positions"],
            "selected_target_predictor_positions": target_metadata[
                "target_predictor_positions"
            ],
            "selected_target_token_ids": target_metadata["target_token_ids"],
            "donor_target_token_ids": target_metadata["donor_token_ids"],
            "causal_prefix_sha256": target_metadata["causal_prefix_sha256"],
        }
        metadata_mismatches = {
            field: {"expected": expected, "actual": entry.get(field)}
            for field, expected in metadata_expected.items()
            if entry.get(field) != expected
        }
        if metadata_mismatches:
            raise ValueError(
                f"Scene-state V7 causal target differs at row {source_index}: "
                f"{metadata_mismatches}"
            )
        target_stratum = _scene_state_identity_target_stratum(
            boundary_counts[source_index],
            boundary_counts[donor_index],
        )
        if entry.get("target_stratum") != target_stratum:
            raise ValueError(
                f"Scene-state V7 target stratum differs at row {source_index}"
            )
        decision_mask = [
            bool(value)
            for value in rows[source_index]["scene_state_generation_decision_mask"]
        ]
        if len(decision_mask) != len(target_mask) or any(
            selected and not decision
            for selected, decision in zip(target_mask, decision_mask)
        ):
            raise ValueError(
                f"Scene-state V7 pair target escapes decision tokens at row {source_index}"
            )
        donor_tokens = target_metadata["donor_token_ids"]
        if not isinstance(donor_tokens, list) or len(donor_tokens) != 1:
            raise ValueError(
                f"Scene-state V7 donor target is not scalar at row {source_index}"
            )
        donor_target_token_ids.append(int(donor_tokens[0]))
        target_mask_hash = _canonical_json_sha256(target_mask)
        target_masks.append(target_mask)
        target_mask_hashes.append(target_mask_hash)
        target_strata.append(target_stratum)
        pair_audit.append(
            {
                **entry,
                "target_mask_sha256": target_mask_hash,
            }
        )
    if any(donor_indices[donor] != source for source, donor in enumerate(donor_indices)):
        raise ValueError("Scene-state V7 tokenized donor map is not symmetric")

    paired = split
    for column, values in donor_columns.items():
        paired = paired.add_column(column, values)
    paired = paired.add_column(
        "scene_state_donor_boundary_count",
        [boundary_counts[index] for index in donor_indices],
    )
    paired = paired.add_column("scene_state_identity_target_mask", target_masks)
    paired = paired.add_column(
        "scene_state_identity_target_mask_sha256",
        target_mask_hashes,
    )
    paired = paired.add_column(
        "scene_state_identity_target_stratum",
        target_strata,
    )
    paired = paired.add_column(
        "scene_state_identity_donor_target_token_id",
        donor_target_token_ids,
    )
    paired = paired.add_column("scene_state_source_index", list(range(len(rows))))
    paired = paired.add_column("scene_state_donor_index", donor_indices)
    paired = paired.add_column(
        "scene_state_source_official_index",
        [int(entry["official_source_index"]) for entry in entries],
    )
    paired = paired.add_column(
        "scene_state_donor_official_index",
        [int(entry["donor_official_source_index"]) for entry in entries],
    )
    for column, field in (
        ("scene_state_source_row_sha256", "source_row_sha256"),
        ("scene_state_donor_row_sha256", "donor_row_sha256"),
        ("scene_state_source_label_sha256", "source_label_sha256"),
        ("scene_state_donor_label_sha256", "donor_label_sha256"),
        ("scene_state_source_write_sha256", "source_write_sha256"),
        ("scene_state_donor_write_sha256", "donor_write_sha256"),
    ):
        paired = paired.add_column(column, [str(entry[field]) for entry in entries])

    target_stratum_row_counts = {
        stratum: target_strata.count(stratum)
        for stratum in _SCENE_STATE_IDENTITY_TARGET_STRATA
    }
    if target_stratum_row_counts != pairing_binding.get("quotas"):
        raise ValueError("Scene-state V7 materialized target quotas differ")
    write_token_count_deltas = [
        int(entry["write_token_count_delta"]) for entry in entries
    ]
    source_boundary_count_histogram = {
        str(value): boundary_counts.count(value) for value in sorted(set(boundary_counts))
    }
    split_manifest: dict[str, object] = {
        "split": split_name,
        "pairing_version": "predeclared_directed_scene_v7_v1",
        "pairing_refinement": "predeclared_quota_locked_v1",
        "pairing_refinement_applied": False,
        "pairing_scope": "predeclared_bound_train_order",
        "target_mode": _SCENE_STATE_IDENTITY_TARGET_MODE,
        "causal_prefix_mode": _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "sample_count": len(rows),
        "pair_count": len(rows) // 2,
        "target_token_count": len(rows),
        "target_stratum_row_counts": target_stratum_row_counts,
        "source_boundary_count_histogram": source_boundary_count_histogram,
        "write_token_count_delta_max": max(write_token_count_deltas),
        "write_token_count_delta_mean": (
            sum(write_token_count_deltas) / len(write_token_count_deltas)
        ),
        "write_token_count_delta_total": sum(write_token_count_deltas) // 2,
        "nearest_baseline_write_token_count_delta_max": max(
            write_token_count_deltas
        ),
        "nearest_baseline_write_token_count_delta_total": (
            sum(write_token_count_deltas) // 2
        ),
        "source_fingerprint": getattr(split, "_fingerprint", None),
        "paired_fingerprint": getattr(paired, "_fingerprint", None),
        "source_pair_manifest_path": pairing_binding["pair_path"],
        "source_pair_manifest_file_sha256": pairing_binding["pair_file_sha256"],
        "source_pair_manifest_sha256": pairing_binding["pair_manifest_sha256"],
        "source_entries_sha256": pairing_binding["entries_sha256"],
        "pairs_sha256": _canonical_json_sha256(pair_audit),
        "pairs": pair_audit,
    }
    split_manifest["manifest_sha256"] = _canonical_json_sha256(split_manifest)
    return paired, split_manifest


def build_scene_state_identity_pairing_manifest(
    *,
    tokenized_fingerprint: str | None,
    tokenized_dataset_sha256: str | None,
    data_seed: int,
    train_manifest: dict[str, object],
    eval_manifest: dict[str, object] | None,
    objective_version: str = _SCENE_STATE_IDENTITY_OBJECTIVE_VERSION,
) -> dict[str, object]:
    splits = {"train": train_manifest}
    if eval_manifest is not None:
        splits["eval"] = eval_manifest
    target_stratum_row_counts = {
        stratum: sum(
            int(split_manifest["target_stratum_row_counts"][stratum])
            for split_manifest in splits.values()
        )
        for stratum in _SCENE_STATE_IDENTITY_TARGET_STRATA
    }
    source_boundary_count_histogram: dict[str, int] = {}
    for split_manifest in splits.values():
        for boundary_count, count in split_manifest[
            "source_boundary_count_histogram"
        ].items():
            source_boundary_count_histogram[str(boundary_count)] = (
                source_boundary_count_histogram.get(str(boundary_count), 0)
                + int(count)
            )
    total_rows = sum(int(item["sample_count"]) for item in splits.values())
    manifest: dict[str, object] = {
        "schema_version": 2,
        "objective_version": objective_version,
        "pairing_version": train_manifest["pairing_version"],
        "pairing_refinement": train_manifest["pairing_refinement"],
        "pairing_refinement_applied": any(
            bool(item["pairing_refinement_applied"])
            for item in splits.values()
        ),
        "pairing_scope": train_manifest.get(
            "pairing_scope",
            "within_post_split_partition",
        ),
        "target_mode": _SCENE_STATE_IDENTITY_TARGET_MODE,
        "causal_prefix_mode": _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE,
        "semantic_mask_mode": _SCENE_STATE_SEMANTIC_MASK_MODE,
        "semantic_loss_normalization": _SCENE_STATE_SEMANTIC_LOSS_NORMALIZATION,
        "target_token_count": sum(
            int(item["target_token_count"]) for item in splits.values()
        ),
        "target_stratum_row_counts": target_stratum_row_counts,
        "source_boundary_count_histogram": dict(
            sorted(
                source_boundary_count_histogram.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "write_token_count_delta_max": max(
            int(item["write_token_count_delta_max"])
            for item in splits.values()
        ),
        "write_token_count_delta_mean": sum(
            float(item["write_token_count_delta_mean"])
            * int(item["sample_count"])
            for item in splits.values()
        )
        / total_rows,
        "write_token_count_delta_total": sum(
            int(item["write_token_count_delta_total"])
            for item in splits.values()
        ),
        "nearest_baseline_write_token_count_delta_max": max(
            int(item["nearest_baseline_write_token_count_delta_max"])
            for item in splits.values()
        ),
        "nearest_baseline_write_token_count_delta_total": sum(
            int(item["nearest_baseline_write_token_count_delta_total"])
            for item in splits.values()
        ),
        "data_seed": data_seed,
        "tokenized_fingerprint": tokenized_fingerprint,
        "tokenized_dataset_sha256": tokenized_dataset_sha256,
        "splits": splits,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def persist_scene_state_identity_pairing_manifest(
    output_dir: Path,
    manifest: dict[str, object],
) -> Path:
    unsigned = dict(manifest)
    recorded_sha256 = unsigned.pop("manifest_sha256", None)
    actual_sha256 = _canonical_json_sha256(unsigned)
    if recorded_sha256 != actual_sha256:
        raise ValueError(
            "Scene-state identity pairing manifest checksum differs: "
            f"expected={recorded_sha256} actual={actual_sha256}"
        )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _SCENE_STATE_IDENTITY_PAIRING_FILENAME
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "Scene-state identity pairing manifest path is not a regular file: "
                f"{path}"
            )
        persisted = _load_json_object(
            path,
            description="scene-state identity pairing manifest",
        )
        if persisted != manifest:
            raise ValueError(
                "Existing scene-state identity pairing manifest differs from the "
                "materialized training pairing"
            )
        return path

    _write_json_atomic(path, manifest)
    persisted = _load_json_object(
        path,
        description="scene-state identity pairing manifest",
    )
    if persisted != manifest:
        raise RuntimeError(
            "Persisted scene-state identity pairing manifest differs from the "
            "materialized training pairing"
        )
    return path


def _scene_state_identity_protocol_pairing_summary(
    manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "pairing_version": manifest["pairing_version"],
        "pairing_refinement": manifest["pairing_refinement"],
        "pairing_refinement_applied": manifest[
            "pairing_refinement_applied"
        ],
        "pairing_scope": manifest["pairing_scope"],
        "target_mode": manifest["target_mode"],
        "causal_prefix_mode": manifest["causal_prefix_mode"],
        "semantic_mask_mode": manifest["semantic_mask_mode"],
        "semantic_loss_normalization": manifest["semantic_loss_normalization"],
        "target_token_count": manifest["target_token_count"],
        "target_stratum_row_counts": manifest["target_stratum_row_counts"],
        "source_boundary_count_histogram": manifest[
            "source_boundary_count_histogram"
        ],
        "write_token_count_delta_max": manifest[
            "write_token_count_delta_max"
        ],
        "write_token_count_delta_mean": manifest[
            "write_token_count_delta_mean"
        ],
        "write_token_count_delta_total": manifest[
            "write_token_count_delta_total"
        ],
        "nearest_baseline_write_token_count_delta_max": manifest[
            "nearest_baseline_write_token_count_delta_max"
        ],
        "nearest_baseline_write_token_count_delta_total": manifest[
            "nearest_baseline_write_token_count_delta_total"
        ],
        "data_seed": manifest["data_seed"],
        "tokenized_fingerprint": manifest["tokenized_fingerprint"],
        "tokenized_dataset_sha256": manifest["tokenized_dataset_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": {
            split_name: {
                key: split_manifest[key]
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
            for split_name, split_manifest in manifest["splits"].items()
        },
    }


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
                "target_mode",
                "target_span_tokens",
                "target_token_count",
                "source_fingerprint",
                "paired_fingerprint",
                "pairs_sha256",
                "manifest_sha256",
            )
        }
    return {
        "pairing_version": manifest["pairing_version"],
        "pairing_scope": manifest["pairing_scope"],
        "target_mode": manifest["target_mode"],
        "target_span_tokens": manifest["target_span_tokens"],
        "target_token_count": manifest["target_token_count"],
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


def validate_scene_boundary_payload_columns(
    tokenized: Dataset,
    *,
    training_mode: str,
) -> None:
    if training_mode != "episode":
        raise ValueError("Scene-boundary payload CE requires episode training mode")
    tensor_groups = (
        (
            "scene_boundary_payload_mask",
            "labels",
            "attention_mask",
        ),
        (
            "state_only_scene_boundary_payload_mask",
            "state_only_labels",
            "state_only_attention_mask",
        ),
        (
            "teacher_scene_boundary_payload_mask",
            "teacher_labels",
            "teacher_attention_mask",
        ),
    )
    required_columns = {
        column
        for tensor_group in tensor_groups
        for column in tensor_group
    }
    missing_columns = sorted(required_columns.difference(tokenized.column_names))
    if missing_columns:
        raise ValueError(
            "Tokenized scene-boundary dataset is missing payload metadata; rebuild it "
            "with the current trainer: " + ", ".join(missing_columns)
        )
    for row_index, row in enumerate(tokenized):
        for mask_column, labels_column, attention_column in tensor_groups:
            mask = [bool(value) for value in row[mask_column]]
            labels = [int(value) for value in row[labels_column]]
            attention_mask = [int(value) for value in row[attention_column]]
            if not len(mask) == len(labels) == len(attention_mask):
                raise ValueError(
                    f"Scene-boundary payload metadata is misaligned at row {row_index}: "
                    f"{mask_column}"
                )
            if not any(mask[1:]):
                raise ValueError(
                    f"Scene-boundary payload metadata has no causal target at row "
                    f"{row_index}: {mask_column}"
                )
            if any(
                selected and (label == -100 or attention == 0)
                for selected, label, attention in zip(mask, labels, attention_mask)
            ):
                raise ValueError(
                    f"Scene-boundary payload metadata escapes supervised tokens at row "
                    f"{row_index}: {mask_column}"
                )


def validate_scene_state_semantic_columns(
    tokenized: Dataset,
    *,
    training_mode: str,
) -> None:
    if training_mode != "episode":
        raise ValueError("scene_state_identity_ce requires episode training mode")
    required_columns = {
        "scene_state_semantic_mask",
        "scene_state_boundary_count",
        "labels",
        "attention_mask",
        "write_input_ids",
        "write_attention_mask",
    }
    missing_columns = sorted(required_columns.difference(tokenized.column_names))
    if missing_columns:
        raise ValueError(
            "Tokenized scene-state dataset is missing semantic metadata; rebuild it "
            "with the current trainer: " + ", ".join(missing_columns)
        )
    for row_index, row in enumerate(tokenized):
        boundary_count = row["scene_state_boundary_count"]
        if (
            not isinstance(boundary_count, int)
            or isinstance(boundary_count, bool)
            or boundary_count < 0
        ):
            raise ValueError(
                f"Scene-state row {row_index} has an invalid boundaries cardinality"
            )
        mask = [bool(value) for value in row["scene_state_semantic_mask"]]
        labels = [int(value) for value in row["labels"]]
        attention_mask = [int(value) for value in row["attention_mask"]]
        if not len(mask) == len(labels) == len(attention_mask):
            raise ValueError(
                f"Scene-state semantic metadata is misaligned at row {row_index}"
            )
        if not any(mask[1:]):
            raise ValueError(
                f"Scene-state semantic metadata has no causal target at row {row_index}"
            )
        if any(
            selected and (label == -100 or attention == 0)
            for selected, label, attention in zip(mask, labels, attention_mask)
        ):
            raise ValueError(
                f"Scene-state semantic metadata escapes supervised tokens at row {row_index}"
            )
        write_ids = [int(value) for value in row["write_input_ids"]]
        write_attention = [int(value) for value in row["write_attention_mask"]]
        if not write_ids or len(write_ids) != len(write_attention):
            raise ValueError(
                f"Scene-state row {row_index} must have one aligned non-empty write"
            )


def validate_scene_state_generation_columns(
    tokenized: Dataset,
    *,
    training_mode: str,
) -> None:
    validate_scene_state_semantic_columns(
        tokenized,
        training_mode=training_mode,
    )
    mask_columns = (
        "scene_state_generation_target_mask",
        "scene_state_generation_content_mask",
        "scene_state_generation_schema_mask",
        "scene_state_generation_decision_mask",
        "scene_state_generation_termination_mask",
    )
    missing_columns = sorted(set(mask_columns).difference(tokenized.column_names))
    if missing_columns:
        raise ValueError(
            "Tokenized scene-state generation dataset is missing exact-generation masks: "
            + ", ".join(missing_columns)
        )
    for row_index, row in enumerate(tokenized):
        labels = [int(value) for value in row["labels"]]
        masks = {
            column: [bool(value) for value in row[column]]
            for column in mask_columns
        }
        if any(len(mask) != len(labels) for mask in masks.values()):
            raise ValueError(
                f"Scene-state generation masks are misaligned at row {row_index}"
            )
        target = masks["scene_state_generation_target_mask"]
        content = masks["scene_state_generation_content_mask"]
        schema = masks["scene_state_generation_schema_mask"]
        decision = masks["scene_state_generation_decision_mask"]
        termination = masks["scene_state_generation_termination_mask"]
        if target != [
            content_value or termination_value
            for content_value, termination_value in zip(content, termination)
        ]:
            raise ValueError(
                f"Scene-state generation suffix partition differs at row {row_index}"
            )
        if content != [
            schema_value or decision_value
            for schema_value, decision_value in zip(schema, decision)
        ]:
            raise ValueError(
                f"Scene-state generation content partition differs at row {row_index}"
            )
        if any(
            schema_value and decision_value
            for schema_value, decision_value in zip(schema, decision)
        ) or any(
            content_value and termination_value
            for content_value, termination_value in zip(content, termination)
        ):
            raise ValueError(
                f"Scene-state generation masks overlap at row {row_index}"
            )
        if any(selected != (label != -100) for selected, label in zip(target, labels)):
            raise ValueError(
                f"Scene-state generation labels include non-generated tokens at row {row_index}"
            )
        if not all(any(mask[1:]) for mask in (content, schema, decision, termination)):
            raise ValueError(
                f"Scene-state generation row {row_index} lacks a causal mask partition"
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
    scene_state_identity_pairing_manifest: dict[str, object] | None = None,
    train_schedule_binding: dict[str, object] | None = None,
    tokenized_cache_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    is_content_contrast = args.memory_loss_mode == "content_contrast_ce"
    is_scene_state_identity = args.memory_loss_mode == "scene_state_identity_ce"
    is_scene_state_generation = args.memory_loss_mode == "scene_state_generation_ce"
    generated_unlikelihood_weight = float(
        getattr(args, "scene_state_generated_unlikelihood_weight", 0.0)
    )
    generated_unlikelihood_max_wrong_tokens = int(
        getattr(
            args,
            "scene_state_generated_unlikelihood_max_wrong_tokens",
            _SCENE_STATE_GENERATED_UNLIKELIHOOD_MAX_WRONG_TOKENS,
        )
    )
    generated_rollout_extra_tokens = int(
        getattr(
            args,
            "scene_state_generated_rollout_extra_tokens",
            _SCENE_STATE_GENERATED_ROLLOUT_EXTRA_TOKENS,
        )
    )
    generated_rollout_max_tokens = int(
        getattr(
            args,
            "scene_state_generated_rollout_max_tokens",
            _SCENE_STATE_GENERATED_ROLLOUT_MAX_TOKENS,
        )
    )
    uses_generated_unlikelihood = (
        is_scene_state_generation
        and generated_unlikelihood_weight > 0.0
    )
    scene_generation_schema_version = (
        _SCENE_STATE_GENERATED_UNLIKELIHOOD_TRAINING_PROTOCOL_SCHEMA_VERSION
        if uses_generated_unlikelihood
        else _SCENE_STATE_GENERATION_TRAINING_PROTOCOL_SCHEMA_VERSION
    )
    scene_generation_objective_version = (
        _SCENE_STATE_GENERATED_UNLIKELIHOOD_OBJECTIVE_VERSION
        if uses_generated_unlikelihood
        else _SCENE_STATE_GENERATION_OBJECTIVE_VERSION
    )
    if tokenized_cache_identity is not None:
        if tokenized_cache_identity.get("rows") != len(tokenized):
            raise ValueError("Tokenized cache identity row count differs from the dataset")
        if tokenized_cache_identity.get("column_names") != list(tokenized.column_names):
            raise ValueError("Tokenized cache identity columns differ from the dataset")
        if tokenized_cache_identity.get("saved_fingerprint") != getattr(
            tokenized,
            "_fingerprint",
            None,
        ):
            raise ValueError("Tokenized cache identity fingerprint differs from the dataset")
    protocol = {
        "schema_version": (
            _CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION
            if is_content_contrast
            else (
                _SCENE_STATE_IDENTITY_TRAINING_PROTOCOL_SCHEMA_VERSION
                if is_scene_state_identity
                else (
                    scene_generation_schema_version
                    if is_scene_state_generation
                    else _TRAINING_PROTOCOL_SCHEMA_VERSION
                )
            )
        ),
        "memory_objective_version": (
            _CONTENT_CONTRAST_OBJECTIVE_VERSION
            if is_content_contrast
            else (
                _SCENE_STATE_IDENTITY_OBJECTIVE_VERSION
                if is_scene_state_identity
                else (
                    scene_generation_objective_version
                    if is_scene_state_generation
                    else _MEMORY_OBJECTIVE_VERSION
                )
            )
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
        "tokenized_dataset_sha256": (
            None
            if tokenized_cache_identity is None
            else tokenized_cache_identity.get("ordered_content_sha256")
        ),
        "expected_tokenized_dataset_sha256": (
            _normalized_expected_tokenized_dataset_sha256(args)
        ),
        "tokenized_cache_identity": tokenized_cache_identity,
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
        "memory_fusion_residual_scale_max": getattr(
            args,
            "memory_fusion_residual_scale_max",
            1.0,
        ),
        "rwkv_ms_output_init_scale": getattr(args, "rwkv_ms_output_init_scale", 0.02),
        "rwkv_ms_semantics_version": getattr(args, "rwkv_ms_semantics_version", 2),
        "memory_loss_mode": args.memory_loss_mode,
        "memory_dropout_no_memory_prob": args.memory_dropout_no_memory_prob,
        "memory_dropout_state_only_prob": args.memory_dropout_state_only_prob,
        "memory_base_kl_weight": args.memory_base_kl_weight,
        "scene_boundary_payload_ce_weight": getattr(
            args,
            "scene_boundary_payload_ce_weight",
            0.0,
        ),
        "scene_boundary_payload_mask_mode": _SCENE_BOUNDARY_PAYLOAD_MASK_MODE,
        "scene_boundary_payload_ce_normalization": (
            _SCENE_BOUNDARY_PAYLOAD_CE_NORMALIZATION
        ),
        "context_ablation_mode": args.context_ablation_mode,
        "context_ablation_no_state_prob": args.context_ablation_no_state_prob,
        "context_ablation_state_only_prob": args.context_ablation_state_only_prob,
        "validation_split_ratio": args.validation_split_ratio,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_sampler_seed": getattr(args, "train_sampler_seed", None),
        "train_sampler_mode": (
            _FIXED_TRAIN_SCHEDULE_SAMPLER_MODE
            if train_schedule_binding is not None
            else (
                _DEFAULT_TRAIN_SAMPLER_MODE
                if getattr(args, "train_sampler_seed", None) is None
                else _SEEDED_TRAIN_SAMPLER_MODE
            )
        ),
        "train_schedule": (
            None
            if train_schedule_binding is None
            else _scene_state_v8_curriculum_protocol_summary(
                train_schedule_binding
            )
        ),
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
                "memory_representation_weight": args.memory_representation_weight,
                "memory_representation_margin": args.memory_representation_margin,
                "memory_kl_weight": args.memory_kl_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": args.memory_partition_alignment_weight,
                "memory_partition_entropy_weight": args.memory_partition_entropy_weight,
                "memory_partition_balance_weight": args.memory_partition_balance_weight,
                "content_contrast_negative_priming_grad": True,
                "content_contrast_backward_mode": _CONTENT_CONTRAST_BACKWARD_MODE,
                "content_contrast_read_mask_mode": _CONTENT_CONTRAST_READ_MASK_MODE,
                "content_contrast_target_mode": _CONTENT_CONTRAST_TARGET_MODE,
                "content_contrast_target_span_tokens": (
                    _CONTENT_CONTRAST_TARGET_SPAN_TOKENS
                ),
                "content_contrast_previous_source_grad": (
                    _CONTENT_CONTRAST_PREVIOUS_SOURCE_GRAD
                ),
                "content_contrast_representation_mode": (
                    _CONTENT_CONTRAST_REPRESENTATION_MODE
                ),
                "content_contrast_pairing": _content_contrast_protocol_pairing_summary(
                    content_contrast_pairing_manifest
                ),
            }
        )
    if is_scene_state_identity:
        if scene_state_identity_pairing_manifest is None:
            raise ValueError(
                "scene_state_identity_ce requires a post-split pairing manifest"
            )
        source_manifest_identity = _scene_state_source_manifest_identity(args)
        if source_manifest_identity is None:
            raise ValueError(
                "scene_state_identity_ce requires a bound source manifest identity"
            )
        protocol.update(
            {
                "scene_state_identity_margin": args.scene_state_identity_margin,
                "scene_state_margin_mode": "per_row_hinge_relu_v1",
                "scene_state_objective_formula": (
                    "full_correct_ce + correct_all_semantic_ce + "
                    "mean(relu(margin - (donor_pair_semantic_ce - "
                    "correct_pair_semantic_ce)))"
                ),
                "scene_state_correct_all_semantic_scope": (
                    "all_semantic_tokens_v1"
                ),
                "scene_state_pair_semantic_scope": (
                    "first_pair_distinguishing_semantic_token_v1"
                ),
                "scene_state_donor_margin_scope": (
                    "first_pair_distinguishing_semantic_token_v1"
                ),
                "scene_state_zero_diagnostic_scope": "all_semantic_tokens_v1",
                "scene_state_zero_diagnostic_gradient": False,
                "scene_state_read_time_positions_observable": False,
                "scene_state_pairing_length_control": (
                    "nearest_feasible_symmetric_absolute_write_token_delta_v1"
                ),
                "scene_state_pairing_refinement": (
                    _SCENE_STATE_IDENTITY_PAIRING_REFINEMENT
                ),
                "scene_state_identity_target_strata": list(
                    _SCENE_STATE_IDENTITY_TARGET_STRATA
                ),
                "scene_state_full_correct_ce_weight": (
                    _SCENE_STATE_FULL_CORRECT_CE_WEIGHT
                ),
                "scene_state_correct_all_semantic_ce_weight": (
                    _SCENE_STATE_CORRECT_ALL_SEMANTIC_CE_WEIGHT
                ),
                "scene_state_donor_margin_weight": (
                    _SCENE_STATE_DONOR_MARGIN_WEIGHT
                ),
                "scene_state_identity_backward_mode": (
                    _SCENE_STATE_IDENTITY_BACKWARD_MODE
                ),
                "scene_state_identity_read_protocol": (
                    _SCENE_STATE_IDENTITY_READ_PROTOCOL
                ),
                "scene_state_identity_zero_protocol": (
                    _SCENE_STATE_IDENTITY_ZERO_PROTOCOL
                ),
                "scene_state_semantic_mask_mode": (
                    _SCENE_STATE_SEMANTIC_MASK_MODE
                ),
                "scene_state_semantic_loss_normalization": (
                    _SCENE_STATE_SEMANTIC_LOSS_NORMALIZATION
                ),
                "scene_state_identity_target_mode": (
                    _SCENE_STATE_IDENTITY_TARGET_MODE
                ),
                "scene_state_identity_causal_prefix_mode": (
                    _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE
                ),
                "scene_state_source_manifest": source_manifest_identity,
                "scene_state_identity_pairing": (
                    _scene_state_identity_protocol_pairing_summary(
                        scene_state_identity_pairing_manifest
                    )
                ),
                "memory_kl_weight": args.memory_kl_weight,
                "memory_base_kl_weight": args.memory_base_kl_weight,
                "memory_representation_weight": args.memory_representation_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": (
                    args.memory_partition_alignment_weight
                ),
                "memory_partition_entropy_weight": (
                    args.memory_partition_entropy_weight
                ),
                "memory_partition_balance_weight": (
                    args.memory_partition_balance_weight
                ),
            }
        )
    if is_scene_state_generation:
        if scene_state_identity_pairing_manifest is None:
            raise ValueError(
                "scene_state_generation_ce requires a predeclared pairing manifest"
            )
        source_manifest_identity = _scene_state_source_manifest_identity(args)
        if source_manifest_identity is None:
            raise ValueError(
                "scene_state_generation_ce requires a bound source manifest identity"
            )
        protocol.update(
            {
                "scene_generation_objective_formula": (
                    "weighted_generation_ce(schema=2,decision=4,termination=1) + "
                    "first_wrong_gold_prefix_top1_hinge(0.2) + "
                    "correct_source_vs_donor_two_token_ce + "
                    "donor_donor_vs_source_two_token_ce + "
                    "correct_vs_detached_zero_decision_margin_hinge(0.2)"
                    + (
                        " + "
                        f"{generated_unlikelihood_weight} * "
                        "correct_state_generated_prefix_unlikelihood"
                        if uses_generated_unlikelihood
                        else ""
                    )
                ),
                "scene_generation_backward_mode": (
                    _SCENE_STATE_GENERATED_UNLIKELIHOOD_BACKWARD_MODE
                    if uses_generated_unlikelihood
                    else _SCENE_STATE_GENERATION_BACKWARD_MODE
                ),
                "scene_generation_read_protocol": (
                    "exact_system_only_generation_prefix_same_read_correct_donor_zero_v1"
                ),
                "scene_generation_zero_protocol": (
                    "adapter_active_reset_state_writes_disabled_detached_reference_v1"
                ),
                "scene_generation_mask_mode": _SCENE_STATE_GENERATION_MASK_MODE,
                "scene_generation_decision_mask_mode": (
                    _SCENE_STATE_GENERATION_DECISION_MASK_MODE
                ),
                "scene_generation_schema_weight": (
                    _SCENE_STATE_GENERATION_SCHEMA_WEIGHT
                ),
                "scene_generation_decision_weight": (
                    _SCENE_STATE_GENERATION_DECISION_WEIGHT
                ),
                "scene_generation_termination_weight": (
                    _SCENE_STATE_GENERATION_TERMINATION_WEIGHT
                ),
                "scene_generation_top1_margin": _SCENE_STATE_GENERATION_TOP1_MARGIN,
                "scene_generation_zero_margin": _SCENE_STATE_GENERATION_ZERO_MARGIN,
                "scene_generation_generated_unlikelihood_weight": (
                    generated_unlikelihood_weight
                ),
                "scene_generation_generated_unlikelihood_mode": (
                    _SCENE_STATE_GENERATED_UNLIKELIHOOD_MODE
                    if uses_generated_unlikelihood
                    else None
                ),
                "scene_generation_generated_unlikelihood_scope": (
                    "same_and_cross_cardinality_value_rows_v1"
                    if uses_generated_unlikelihood
                    else None
                ),
                "scene_generation_generated_unlikelihood_max_wrong_tokens": (
                    generated_unlikelihood_max_wrong_tokens
                ),
                "scene_generation_generated_rollout_extra_tokens": (
                    generated_rollout_extra_tokens
                ),
                "scene_generation_generated_rollout_max_tokens": (
                    generated_rollout_max_tokens
                ),
                "scene_generation_generated_rollout_decoding": (
                    "greedy_use_cache_true_exact_system_only_prompt_v1"
                    if uses_generated_unlikelihood
                    else None
                ),
                "scene_generation_generated_replay_state_gradient": (
                    True if uses_generated_unlikelihood else None
                ),
                "scene_generation_generated_replay_read_path_gradient": (
                    True if uses_generated_unlikelihood else None
                ),
                "scene_generation_assistant_role_labels": False,
                "scene_generation_zero_gradient": False,
                "scene_generation_zero_affects_correct_gradient": True,
                "scene_state_semantic_mask_mode": _SCENE_STATE_SEMANTIC_MASK_MODE,
                "scene_state_identity_target_mode": (
                    _SCENE_STATE_IDENTITY_TARGET_MODE
                ),
                "scene_state_identity_causal_prefix_mode": (
                    _SCENE_STATE_IDENTITY_CAUSAL_PREFIX_MODE
                ),
                "scene_state_source_manifest": source_manifest_identity,
                "scene_state_identity_pairing": (
                    _scene_state_identity_protocol_pairing_summary(
                        scene_state_identity_pairing_manifest
                    )
                ),
                "memory_kl_weight": args.memory_kl_weight,
                "memory_base_kl_weight": args.memory_base_kl_weight,
                "memory_representation_weight": args.memory_representation_weight,
                "write_sparsity_weight": args.write_sparsity_weight,
                "memory_partition_alignment_weight": (
                    args.memory_partition_alignment_weight
                ),
                "memory_partition_entropy_weight": (
                    args.memory_partition_entropy_weight
                ),
                "memory_partition_balance_weight": (
                    args.memory_partition_balance_weight
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
        has_payload_masks = [
            "scene_boundary_payload_mask" in feature for feature in features
        ]
        if any(has_payload_masks) and not all(has_payload_masks):
            raise ValueError("Batch mixes scene-boundary payload metadata presence")
        payload_masks = []
        has_scene_state_semantic_masks = [
            "scene_state_semantic_mask" in feature for feature in features
        ]
        if any(has_scene_state_semantic_masks) and not all(
            has_scene_state_semantic_masks
        ):
            raise ValueError("Batch mixes scene-state semantic metadata presence")
        scene_state_semantic_masks = []
        generation_mask_columns = (
            "scene_state_generation_target_mask",
            "scene_state_generation_content_mask",
            "scene_state_generation_schema_mask",
            "scene_state_generation_decision_mask",
            "scene_state_generation_termination_mask",
        )
        generation_mask_presence = {
            column: [column in feature for feature in features]
            for column in generation_mask_columns
        }
        for column, presence in generation_mask_presence.items():
            if any(presence) and not all(presence):
                raise ValueError(
                    f"Batch mixes scene-state generation metadata presence: {column}"
                )
        has_generation_masks = all(
            all(generation_mask_presence[column])
            for column in generation_mask_columns
        )
        generation_masks = {column: [] for column in generation_mask_columns}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            if all(has_payload_masks):
                payload_mask = feature["scene_boundary_payload_mask"]
                if len(payload_mask) != len(feature["labels"]):
                    raise ValueError(
                        "Scene-boundary payload mask must align with labels"
                    )
                payload_masks.append(payload_mask + [False] * pad_len)
            if all(has_scene_state_semantic_masks):
                semantic_mask = feature["scene_state_semantic_mask"]
                if len(semantic_mask) != len(feature["labels"]):
                    raise ValueError(
                        "Scene-state semantic mask must align with labels"
                    )
                if not any(bool(value) for value in semantic_mask[1:]):
                    raise ValueError(
                        "Scene-state semantic mask must select a causal target"
                    )
                scene_state_semantic_masks.append(
                    semantic_mask + [False] * pad_len
                )
            if has_generation_masks:
                for column in generation_mask_columns:
                    mask = [bool(value) for value in feature[column]]
                    if len(mask) != len(feature["labels"]):
                        raise ValueError(
                            f"Scene-state generation mask must align with labels: {column}"
                        )
                    generation_masks[column].append(mask + [False] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if all(has_payload_masks):
            batch["scene_boundary_payload_mask"] = torch.tensor(
                payload_masks,
                dtype=torch.bool,
            )
        if all(has_scene_state_semantic_masks):
            batch["scene_state_semantic_mask"] = torch.tensor(
                scene_state_semantic_masks,
                dtype=torch.bool,
            )
        if has_generation_masks:
            for column, masks in generation_masks.items():
                batch[column] = torch.tensor(masks, dtype=torch.bool)
        return batch


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
                "content_contrast_target_mask",
                "content_contrast_target_mask_sha256",
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
            for feature in features:
                target_mask = feature["content_contrast_target_mask"]
                if len(target_mask) != len(feature["labels"]):
                    raise ValueError(
                        "content_contrast_ce target mask must align with labels"
                    )
                if sum(bool(value) for value in target_mask) != (
                    _CONTENT_CONTRAST_TARGET_SPAN_TOKENS
                ):
                    raise ValueError(
                        "content_contrast_ce target mask must select exactly "
                        f"{_CONTENT_CONTRAST_TARGET_SPAN_TOKENS} labels"
                    )
                expected_target_mask_hash = str(
                    feature["content_contrast_target_mask_sha256"]
                )
                actual_target_mask_hash = _canonical_json_sha256(
                    [bool(value) for value in target_mask]
                )
                if actual_target_mask_hash != expected_target_mask_hash:
                    raise ValueError(
                        "content_contrast_ce target mask does not match its audited hash"
                    )
            content_contrast_target_mask = _pad_sequences(
                [feature["content_contrast_target_mask"] for feature in features],
                False,
            )
            if content_contrast_target_mask is None:
                raise ValueError("content_contrast_ce target masks must be non-empty")
            batch["content_contrast_target_mask"] = (
                content_contrast_target_mask.to(dtype=torch.bool)
            )

        has_scene_state_donor_writes = [
            "scene_state_donor_write_input_ids" in feature
            for feature in features
        ]
        if any(has_scene_state_donor_writes) and not all(
            has_scene_state_donor_writes
        ):
            raise ValueError(
                "Episode batch mixes paired and unpaired scene-state examples"
            )
        if all(has_scene_state_donor_writes):
            required_scene_state_columns = (
                "scene_state_donor_write_input_ids",
                "scene_state_donor_write_attention_mask",
                "scene_state_donor_write_message_ids",
                "scene_state_donor_write_sentence_ids",
                "scene_state_semantic_mask",
                "scene_state_identity_target_mask",
                "scene_state_identity_target_mask_sha256",
                "scene_state_boundary_count",
                "scene_state_donor_boundary_count",
                "scene_state_identity_target_stratum",
                "scene_state_source_index",
                "scene_state_donor_index",
                "scene_state_source_row_sha256",
                "scene_state_donor_row_sha256",
                "scene_state_source_label_sha256",
                "scene_state_donor_label_sha256",
                "scene_state_source_write_sha256",
                "scene_state_donor_write_sha256",
            )
            missing_scene_state_columns = [
                column
                for column in required_scene_state_columns
                if any(column not in feature for feature in features)
            ]
            if missing_scene_state_columns:
                raise ValueError(
                    "scene_state_identity_ce examples have incomplete pairing metadata: "
                    + ", ".join(missing_scene_state_columns)
                )
            donor_write_input_ids = _pad_sequences(
                [
                    feature["scene_state_donor_write_input_ids"]
                    for feature in features
                ],
                pad_token_id,
            )
            donor_write_attention_mask = _pad_sequences(
                [
                    feature["scene_state_donor_write_attention_mask"]
                    for feature in features
                ],
                0,
            )
            donor_write_message_ids = _pad_sequences(
                [
                    feature["scene_state_donor_write_message_ids"]
                    for feature in features
                ],
                -1,
            )
            donor_write_sentence_ids = _pad_sequences(
                [
                    feature["scene_state_donor_write_sentence_ids"]
                    for feature in features
                ],
                -1,
            )
            if (
                donor_write_input_ids is None
                or donor_write_attention_mask is None
                or donor_write_message_ids is None
                or donor_write_sentence_ids is None
            ):
                raise ValueError(
                    "scene_state_identity_ce examples require non-empty donor writes"
                )
            batch["scene_state_donor_write_input_ids"] = donor_write_input_ids
            batch["scene_state_donor_write_attention_mask"] = (
                donor_write_attention_mask
            )
            batch["scene_state_donor_write_message_ids"] = donor_write_message_ids
            batch["scene_state_donor_write_sentence_ids"] = (
                donor_write_sentence_ids
            )

            for feature in features:
                target_mask = [
                    bool(value)
                    for value in feature["scene_state_identity_target_mask"]
                ]
                semantic_mask = [
                    bool(value) for value in feature["scene_state_semantic_mask"]
                ]
                if not len(target_mask) == len(semantic_mask) == len(feature["labels"]):
                    raise ValueError(
                        "Scene-state identity and semantic masks must align with labels"
                    )
                if sum(target_mask) != 1:
                    raise ValueError(
                        "Scene-state identity target mask must select exactly one label"
                    )
                if any(
                    selected and not semantic
                    for selected, semantic in zip(target_mask, semantic_mask)
                ):
                    raise ValueError(
                        "Scene-state identity target must be a subset of the semantic mask"
                    )
                actual_target_hash = _canonical_json_sha256(target_mask)
                if actual_target_hash != str(
                    feature["scene_state_identity_target_mask_sha256"]
                ):
                    raise ValueError(
                        "Scene-state identity target mask does not match its audited hash"
                    )

                source_write = {
                    "input_ids": [int(value) for value in feature["write_input_ids"]],
                    "attention_mask": [
                        int(value) for value in feature["write_attention_mask"]
                    ],
                    "message_ids": [
                        int(value) for value in feature["write_message_ids"]
                    ],
                    "sentence_ids": [
                        int(value) for value in feature["write_sentence_ids"]
                    ],
                }
                donor_write = {
                    "input_ids": [
                        int(value)
                        for value in feature[
                            "scene_state_donor_write_input_ids"
                        ]
                    ],
                    "attention_mask": [
                        int(value)
                        for value in feature[
                            "scene_state_donor_write_attention_mask"
                        ]
                    ],
                    "message_ids": [
                        int(value)
                        for value in feature[
                            "scene_state_donor_write_message_ids"
                        ]
                    ],
                    "sentence_ids": [
                        int(value)
                        for value in feature[
                            "scene_state_donor_write_sentence_ids"
                        ]
                    ],
                }
                generation_pair = (
                    "scene_state_identity_donor_target_token_id" in feature
                )
                source_write_hash_value = (
                    source_write["input_ids"] if generation_pair else source_write
                )
                donor_write_hash_value = (
                    donor_write["input_ids"] if generation_pair else donor_write
                )
                if _canonical_json_sha256(source_write_hash_value) != str(
                    feature["scene_state_source_write_sha256"]
                ):
                    raise ValueError(
                        "Scene-state source write does not match its audited hash"
                    )
                if _canonical_json_sha256(donor_write_hash_value) != str(
                    feature["scene_state_donor_write_sha256"]
                ):
                    raise ValueError(
                        "Scene-state donor write does not match its audited hash"
                    )
                for field in (
                    "scene_state_source_row_sha256",
                    "scene_state_donor_row_sha256",
                    "scene_state_source_label_sha256",
                    "scene_state_donor_label_sha256",
                ):
                    value = str(feature[field])
                    if len(value) != 64 or any(
                        character not in "0123456789abcdef" for character in value
                    ):
                        raise ValueError(
                            f"Scene-state pairing has an invalid SHA-256: {field}"
                        )
                if feature["scene_state_source_label_sha256"] == feature[
                    "scene_state_donor_label_sha256"
                ]:
                    raise ValueError(
                        "Scene-state donor label must be exact-distinct"
                    )
                if int(feature["scene_state_source_index"]) == int(
                    feature["scene_state_donor_index"]
                ):
                    raise ValueError("Scene-state row cannot donate to itself")
                expected_stratum = _scene_state_identity_target_stratum(
                    feature["scene_state_boundary_count"],
                    feature["scene_state_donor_boundary_count"],
                )
                if feature["scene_state_identity_target_stratum"] != expected_stratum:
                    raise ValueError(
                        "Scene-state target stratum does not match source/donor "
                        "boundaries cardinalities"
                    )

            scene_state_target_mask = _pad_sequences(
                [
                    feature["scene_state_identity_target_mask"]
                    for feature in features
                ],
                False,
            )
            if scene_state_target_mask is None:
                raise ValueError("Scene-state identity target masks must be non-empty")
            batch["scene_state_identity_target_mask"] = (
                scene_state_target_mask.to(dtype=torch.bool)
            )
            batch["scene_state_identity_target_stratum"] = torch.tensor(
                [
                    _SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES[
                        str(feature["scene_state_identity_target_stratum"])
                    ]
                    for feature in features
                ],
                dtype=torch.long,
            )
            has_generation_pairing = [
                "scene_state_identity_donor_target_token_id" in feature
                for feature in features
            ]
            if any(has_generation_pairing) and not all(has_generation_pairing):
                raise ValueError(
                    "Episode batch mixes V7 generation pairing metadata presence"
                )
            if all(has_generation_pairing):
                generation_mask_columns = (
                    "scene_state_generation_target_mask",
                    "scene_state_generation_content_mask",
                    "scene_state_generation_schema_mask",
                    "scene_state_generation_decision_mask",
                    "scene_state_generation_termination_mask",
                )
                missing_generation_columns = [
                    column
                    for column in generation_mask_columns
                    if column not in batch
                ]
                if missing_generation_columns:
                    raise ValueError(
                        "scene_state_generation_ce examples omit generation masks: "
                        + ", ".join(missing_generation_columns)
                    )
                target = batch["scene_state_generation_target_mask"]
                content = batch["scene_state_generation_content_mask"]
                schema = batch["scene_state_generation_schema_mask"]
                decision = batch["scene_state_generation_decision_mask"]
                termination = batch["scene_state_generation_termination_mask"]
                if not torch.equal(target, content | termination):
                    raise ValueError(
                        "Scene-state generation target mask differs from content/termination"
                    )
                if not torch.equal(content, schema | decision):
                    raise ValueError(
                        "Scene-state generation content mask differs from schema/decision"
                    )
                if bool((schema & decision).any()) or bool(
                    (content & termination).any()
                ):
                    raise ValueError("Scene-state generation mask partitions overlap")
                if not torch.equal(target, batch["labels"].ne(-100)):
                    raise ValueError(
                        "Scene-state generation labels differ from the generated suffix"
                    )
                if bool(
                    (
                        batch["scene_state_identity_target_mask"]
                        & ~decision
                    ).any()
                ):
                    raise ValueError(
                        "Scene-state generation identity target escapes decision tokens"
                    )
                donor_target_ids = torch.tensor(
                    [
                        int(feature["scene_state_identity_donor_target_token_id"])
                        for feature in features
                    ],
                    dtype=torch.long,
                )
                source_target_ids = batch["labels"].masked_select(
                    batch["scene_state_identity_target_mask"]
                )
                if source_target_ids.numel() != len(features) or bool(
                    source_target_ids.eq(donor_target_ids).any()
                ):
                    raise ValueError(
                        "Scene-state generation source/donor targets are not scalar-distinct"
                    )
                batch["scene_state_identity_donor_target_token_id"] = (
                    donor_target_ids
                )

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
        has_scene_boundary_payload = "scene_boundary_payload_mask" in batch
        if has_scene_boundary_payload:
            required_payload_columns = (
                "teacher_scene_boundary_payload_mask",
                "state_only_scene_boundary_payload_mask",
            )
            missing_payload_columns = [
                column
                for column in required_payload_columns
                if any(column not in feature for feature in features)
            ]
            if missing_payload_columns:
                raise ValueError(
                    "Scene-boundary episode examples have incomplete payload metadata: "
                    + ", ".join(missing_payload_columns)
                )
            teacher_payload_mask = _pad_sequences(
                [feature["teacher_scene_boundary_payload_mask"] for feature in features],
                False,
            )
            if teacher_payload_mask is None or teacher_payload_mask.shape != teacher_labels.shape:
                raise ValueError(
                    "Teacher scene-boundary payload mask must align with teacher labels"
                )
            batch["teacher_scene_boundary_payload_mask"] = teacher_payload_mask.to(
                dtype=torch.bool
            )

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
            if has_scene_boundary_payload:
                state_only_payload_mask = _pad_sequences(
                    [
                        feature["state_only_scene_boundary_payload_mask"]
                        for feature in features
                    ],
                    False,
                )
                if (
                    state_only_payload_mask is None
                    or state_only_payload_mask.shape != state_only_labels.shape
                ):
                    raise ValueError(
                        "State-only scene-boundary payload mask must align with labels"
                    )
                batch["state_only_scene_boundary_payload_mask"] = (
                    state_only_payload_mask.to(dtype=torch.bool)
                )

        max_full_len = max(
            len(feature["write_input_ids"]) + len(feature["input_ids"]) for feature in features
        )
        full_input_ids = []
        full_attention_mask = []
        full_labels = []
        full_payload_masks = []
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
            if has_scene_boundary_payload:
                combined_payload_mask = (
                    [False] * len(feature["write_input_ids"])
                    + feature["scene_boundary_payload_mask"]
                )
                full_payload_masks.append(
                    combined_payload_mask + [False] * pad_len
                )
        batch["full_input_ids"] = torch.tensor(full_input_ids, dtype=torch.long)
        batch["full_attention_mask"] = torch.tensor(full_attention_mask, dtype=torch.long)
        batch["full_labels"] = torch.tensor(full_labels, dtype=torch.long)
        if has_scene_boundary_payload:
            batch["full_scene_boundary_payload_mask"] = torch.tensor(
                full_payload_masks,
                dtype=torch.bool,
            )
        return batch


def build_data_collator(training_mode: str, tokenizer):
    if training_mode == "episode":
        return EpisodeCausalLMCollator(tokenizer)
    if training_mode == "dialogue":
        return DialogueCausalLMCollator(tokenizer)
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def main() -> None:
    args = parse_args()
    train_schedule_binding = _scene_state_v8_curriculum_binding(args)
    if train_schedule_binding is not None:
        _validate_scene_state_v8_locked_training_args(
            args,
            train_schedule_binding,
        )
        if args.train_sampler_seed is not None:
            raise ValueError(
                "Scene-memory V8 fixed curriculum forbids train-sampler-seed"
            )
        if args.group_by_length:
            raise ValueError(
                "Scene-memory V8 fixed curriculum forbids group-by-length"
            )
        if not 0 < args.max_steps <= int(train_schedule_binding["total_steps"]):
            raise ValueError(
                "Scene-memory V8 max-steps must stay within the locked curriculum"
            )
    # Adapter and RWKV-core parameters are initialized before Trainer exists.
    set_seed(args.seed)
    if args.warm_start_from_checkpoint is not None:
        _validate_adapter_warm_start_args(args)
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
        require_scene_state_identity_pairing=(
            args.memory_loss_mode in _SCENE_STATE_PAIRED_MEMORY_LOSS_MODES
        ),
    )
    warm_start_from_checkpoint = resolve_adapter_warm_start_checkpoint(
        args.warm_start_from_checkpoint,
        warm_start_mode=args.warm_start_mode,
    )
    warm_start_context = prepare_adapter_warm_start(
        args,
        warm_start_from_checkpoint,
    )
    continuation_manifest = (
        warm_start_context.manifest
        if warm_start_context is not None
        else prepare_training_continuation(args, resume_from_checkpoint)
    )
    if train_schedule_binding is not None and resume_from_checkpoint is not None:
        continuation_manifest = prepare_scene_memory_v8_training_continuation(
            continuation_manifest,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_steps=train_schedule_binding["checkpoint_steps"],
        )
    dtype = get_dtype(args.dtype)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if args.train_sampler_seed is not None and distributed:
        raise ValueError("train-sampler-seed currently requires a single-process run")
    if train_schedule_binding is not None and distributed:
        raise ValueError(
            "Scene-memory V8 fixed curriculum currently requires a single-process run"
        )
    initial_adapter_output_dir = resolve_initial_adapter_output_dir(
        args,
        resume_from_checkpoint=resume_from_checkpoint,
        warm_start_from_checkpoint=warm_start_from_checkpoint,
        world_size=world_size,
    )
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
    if args.scene_boundary_payload_ce_weight > 0.0:
        validate_scene_boundary_payload_columns(
            tokenized,
            training_mode=effective_training_mode,
        )
    if args.memory_loss_mode == "scene_state_identity_ce":
        validate_scene_state_semantic_columns(
            tokenized,
            training_mode=effective_training_mode,
        )
    if args.memory_loss_mode == "scene_state_generation_ce":
        validate_scene_state_generation_columns(
            tokenized,
            training_mode=effective_training_mode,
        )
    train_dataset, eval_dataset = split_tokenized_dataset(
        tokenized,
        validation_split_ratio=args.validation_split_ratio,
        data_seed=args.data_seed,
    )
    content_contrast_pairing_manifest = None
    scene_state_identity_pairing_manifest = None
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
    if args.memory_loss_mode == "scene_state_identity_ce":
        train_dataset, train_scene_pairing_manifest = (
            materialize_scene_state_identity_pairs(
                train_dataset,
                split_name="train",
            )
        )
        eval_scene_pairing_manifest = None
        if eval_dataset is not None:
            eval_dataset, eval_scene_pairing_manifest = (
                materialize_scene_state_identity_pairs(
                    eval_dataset,
                    split_name="eval",
                )
            )
        scene_state_identity_pairing_manifest = (
            build_scene_state_identity_pairing_manifest(
                tokenized_fingerprint=getattr(tokenized, "_fingerprint", None),
                tokenized_dataset_sha256=tokenized_meta.get(
                    "tokenized_dataset_sha256"
                ),
                data_seed=args.data_seed,
                train_manifest=train_scene_pairing_manifest,
                eval_manifest=eval_scene_pairing_manifest,
            )
        )
        persist_scene_state_identity_pairing_manifest(
            args.output_dir,
            scene_state_identity_pairing_manifest,
        )
    if args.memory_loss_mode == "scene_state_generation_ce":
        generation_pairing_binding = _scene_state_generation_pairing_binding(args)
        train_dataset, train_scene_pairing_manifest = (
            materialize_scene_state_generation_pairs(
                train_dataset,
                split_name="train",
                pairing_binding=generation_pairing_binding,
            )
        )
        if eval_dataset is not None:
            raise ValueError(
                "scene_state_generation_ce cannot materialize an unbound eval pairing"
            )
        scene_state_identity_pairing_manifest = (
            build_scene_state_identity_pairing_manifest(
                tokenized_fingerprint=getattr(tokenized, "_fingerprint", None),
                tokenized_dataset_sha256=tokenized_meta.get(
                    "tokenized_dataset_sha256"
                ),
                data_seed=args.data_seed,
                train_manifest=train_scene_pairing_manifest,
                eval_manifest=None,
                objective_version=_SCENE_STATE_GENERATION_OBJECTIVE_VERSION,
            )
        )
        persist_scene_state_identity_pairing_manifest(
            args.output_dir,
            scene_state_identity_pairing_manifest,
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
        memory_fusion_residual_scale_max=args.memory_fusion_residual_scale_max,
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
    if warm_start_context is not None:
        warm_start_context.manifest.update(
            apply_adapter_warm_start(
                model,
                warm_start_context,
                delta_config,
                trainable_names,
            )
        )

    warmup_steps = compute_warmup_steps(
        train_samples=len(train_dataset),
        per_device_train_batch_size=args.per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        explicit_warmup_steps=args.warmup_steps,
    )
    warmup_steps = resolve_resume_warmup_steps(
        warmup_steps,
        resume_from_checkpoint,
    )
    if (
        train_schedule_binding is not None
        and warmup_steps != _SCENE_MEMORY_V8_WARMUP_STEPS
    ):
        raise ValueError(
            "Scene-memory V8 fixed curriculum requires exactly "
            f"{_SCENE_MEMORY_V8_WARMUP_STEPS} warmup steps"
        )
    if (
        warm_start_context is not None
        and warm_start_context.mode == _RESIDUAL_HYBRID_W8_WARM_START_MODE
        and warmup_steps != _RESIDUAL_HYBRID_W8_TARGET_WARMUP_STEPS
    ):
        raise ValueError("Residual-hybrid W8 warm start requires exactly 2 warmup steps")
    training_protocol = build_training_protocol(
        args,
        tokenized,
        effective_training_mode=effective_training_mode,
        train_samples=len(train_dataset),
        eval_samples=0 if eval_dataset is None else len(eval_dataset),
        warmup_steps=warmup_steps,
        content_contrast_pairing_manifest=content_contrast_pairing_manifest,
        scene_state_identity_pairing_manifest=(
            scene_state_identity_pairing_manifest
        ),
        train_schedule_binding=train_schedule_binding,
        tokenized_cache_identity=tokenized_meta.get("tokenized_cache_identity"),
    )
    training_protocol_sha256 = hashlib.sha256(
        json.dumps(training_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if warm_start_context is not None:
        if warm_start_context.mode == _RESIDUAL_HYBRID_W8_WARM_START_MODE:
            validate_residual_hybrid_w8_target_protocol(
                warm_start_context.source_protocol,
                training_protocol,
            )
            if content_contrast_pairing_manifest is None:
                raise RuntimeError(
                    "Residual-hybrid W8 warm start requires target pairing metadata"
                )
            finalize_adapter_warm_start_lineage(
                warm_start_context,
                target_training_protocol_sha256=training_protocol_sha256,
                target_pairing_manifest=content_contrast_pairing_manifest,
            )
        elif warm_start_context.mode == _SCENE_V8_WARM_START_MODE:
            if scene_state_identity_pairing_manifest is None:
                raise RuntimeError(
                    "Scene V8 warm start requires scene-state pairing metadata"
                )
            finalize_scene_v8_warm_start_lineage(
                warm_start_context,
                target_training_protocol_sha256=training_protocol_sha256,
                target_pairing_manifest=scene_state_identity_pairing_manifest,
            )
        else:
            raise RuntimeError(
                f"Unsupported adapter warm-start mode: {warm_start_context.mode}"
            )
    if train_schedule_binding is not None and resume_from_checkpoint is not None:
        if continuation_manifest is None:
            raise RuntimeError("Scene V8 resume continuation lineage is missing")
        finalize_scene_memory_v8_training_continuation(
            continuation_manifest,
            target_training_protocol=training_protocol,
        )
    trainer_resume_from_checkpoint = resolve_trainer_resume_checkpoint(
        resume_from_checkpoint,
        warm_start_context,
    )
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
    initial_adapter_manifest = None
    if initial_adapter_output_dir is not None:
        initial_adapter_manifest = save_seeded_initial_adapter_snapshot(
            model,
            initial_adapter_output_dir,
            delta_config,
            args=args,
            training_protocol=training_protocol,
            training_protocol_sha256=training_protocol_sha256,
            train_samples=len(train_dataset),
            replaced_modules=replaced,
            trainable_names=trainable_names,
        )
    if args.prepare_only:
        if initial_adapter_output_dir is None:
            raise RuntimeError("Prepare-only mode requires a resolved initial adapter path")
        prepare_receipt = validate_prepare_only_snapshot(
            initial_adapter_output_dir,
            initial_adapter_manifest,
        )
        print(json.dumps(prepare_receipt, sort_keys=True), flush=True)
        return
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
        memory_representation_weight=args.memory_representation_weight,
        memory_representation_margin=args.memory_representation_margin,
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
        scene_boundary_payload_ce_weight=args.scene_boundary_payload_ce_weight,
        train_sampler_seed=args.train_sampler_seed,
        train_schedule_indices=(
            None
            if train_schedule_binding is None
            else train_schedule_binding["indices"]
        ),
        train_schedule_binding=train_schedule_binding,
        scene_state_generated_unlikelihood_weight=(
            args.scene_state_generated_unlikelihood_weight
        ),
        scene_state_generated_unlikelihood_max_wrong_tokens=(
            args.scene_state_generated_unlikelihood_max_wrong_tokens
        ),
        scene_state_generated_rollout_extra_tokens=(
            args.scene_state_generated_rollout_extra_tokens
        ),
        scene_state_generated_rollout_max_tokens=(
            args.scene_state_generated_rollout_max_tokens
        ),
        episode_read_write_enabled=args.episode_read_write_enabled,
        context_ablation_mode=args.context_ablation_mode,
        context_ablation_no_state_prob=args.context_ablation_no_state_prob,
        context_ablation_state_only_prob=args.context_ablation_state_only_prob,
        training_protocol=training_protocol,
        content_contrast_pairing_manifest=content_contrast_pairing_manifest,
        scene_state_identity_margin=args.scene_state_identity_margin,
        scene_state_identity_pairing_manifest=(
            scene_state_identity_pairing_manifest
        ),
        resume_mode=args.resume_mode,
        continuation_manifest=continuation_manifest,
    )
    trainer.log_delta_debug_stats = args.log_delta_debug_stats
    if (
        warm_start_context is not None
        and warm_start_context.mode == _SCENE_V8_WARM_START_MODE
    ):
        record_scene_v8_fresh_optimizer_lineage(
            trainer,
            warm_start_context,
        )
    cuda_memory_device = None
    cuda_memory_baseline = None
    if torch.cuda.is_available():
        cuda_memory_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize(cuda_memory_device)
        torch.cuda.reset_peak_memory_stats(cuda_memory_device)
        cuda_memory_baseline = {
            "allocated_bytes": int(torch.cuda.memory_allocated(cuda_memory_device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(cuda_memory_device)),
        }
    trainer.train(resume_from_checkpoint=trainer_resume_from_checkpoint)
    trainer.accelerator.wait_for_everyone()
    cuda_memory = None
    if cuda_memory_device is not None:
        torch.cuda.synchronize(cuda_memory_device)
        cuda_memory = {
            "device": str(cuda_memory_device),
            "baseline_allocated_bytes": cuda_memory_baseline["allocated_bytes"],
            "baseline_reserved_bytes": cuda_memory_baseline["reserved_bytes"],
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(cuda_memory_device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(cuda_memory_device)
            ),
            "post_train_allocated_bytes": int(
                torch.cuda.memory_allocated(cuda_memory_device)
            ),
            "post_train_reserved_bytes": int(
                torch.cuda.memory_reserved(cuda_memory_device)
            ),
        }

    base_model = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.is_world_process_zero():
        lineage_summary = _training_lineage_summary(trainer)
        active_lineage = lineage_summary["continuation"]
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
            "warm_start_from_checkpoint": warm_start_from_checkpoint,
            "warm_start_mode": args.warm_start_mode,
            "initial_adapter_output_dir": (
                None
                if initial_adapter_output_dir is None
                else str(initial_adapter_output_dir)
            ),
            "initial_adapter_manifest_sha256": (
                None
                if initial_adapter_manifest is None
                else initial_adapter_manifest["manifest_sha256"]
            ),
            **lineage_summary,
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
            "train_sampler_seed": args.train_sampler_seed,
            "train_sampler_mode": training_protocol["train_sampler_mode"],
            "train_schedule": training_protocol.get("train_schedule"),
            "episode_recent_messages": args.episode_recent_messages,
            "max_write_length": args.max_write_length,
            "episode_read_write_enabled": args.episode_read_write_enabled,
            "memory_loss_mode": args.memory_loss_mode,
            "memory_objective_version": training_protocol["memory_objective_version"],
            "scene_boundary_payload_ce_weight": (
                args.scene_boundary_payload_ce_weight
            ),
            "scene_boundary_payload_mask_mode": (
                training_protocol["scene_boundary_payload_mask_mode"]
            ),
            "scene_boundary_payload_ce_normalization": (
                training_protocol["scene_boundary_payload_ce_normalization"]
            ),
            "content_contrast_backward_mode": training_protocol.get(
                "content_contrast_backward_mode"
            ),
            "content_contrast_read_mask_mode": training_protocol.get(
                "content_contrast_read_mask_mode"
            ),
            "content_contrast_target_mode": training_protocol.get(
                "content_contrast_target_mode"
            ),
            "content_contrast_target_span_tokens": training_protocol.get(
                "content_contrast_target_span_tokens"
            ),
            "content_contrast_previous_source_grad": training_protocol.get(
                "content_contrast_previous_source_grad"
            ),
            "content_contrast_representation_mode": training_protocol.get(
                "content_contrast_representation_mode"
            ),
            "teacher_max_length": args.max_write_length + args.max_length,
            "memory_write_source": args.memory_write_source,
            "memory_write_granularity": args.memory_write_granularity,
            "delta_o_rmsnorm": args.delta_o_rmsnorm,
            "delta_o_rmsnorm_eps": args.delta_o_rmsnorm_eps,
            "memory_fusion_mode": args.memory_fusion_mode,
            "memory_fusion_gate_init": args.memory_fusion_gate_init,
            "memory_fusion_placement": args.memory_fusion_placement,
            "memory_fusion_residual_scale": args.memory_fusion_residual_scale,
            "memory_fusion_residual_scale_max": args.memory_fusion_residual_scale_max,
            "target_layers": args.target_layers,
            "memory_contrast_weight": args.memory_contrast_weight,
            "memory_kl_weight": args.memory_kl_weight,
            "memory_margin": args.memory_margin,
            "memory_representation_weight": args.memory_representation_weight,
            "memory_representation_margin": args.memory_representation_margin,
            "memory_causal_weight": args.memory_causal_weight,
            "memory_anchor_weight": args.memory_anchor_weight,
            "memory_anchor_margin": args.memory_anchor_margin,
            "memory_recover_weight": args.memory_recover_weight,
            "memory_need_floor": args.memory_need_floor,
            "memory_dropout_no_memory_prob": args.memory_dropout_no_memory_prob,
            "memory_dropout_state_only_prob": args.memory_dropout_state_only_prob,
            "memory_base_kl_weight": args.memory_base_kl_weight,
            "scene_state_generated_unlikelihood_weight": (
                args.scene_state_generated_unlikelihood_weight
            ),
            "scene_state_generated_unlikelihood_max_wrong_tokens": (
                args.scene_state_generated_unlikelihood_max_wrong_tokens
            ),
            "scene_state_generated_rollout_extra_tokens": (
                args.scene_state_generated_rollout_extra_tokens
            ),
            "scene_state_generated_rollout_max_tokens": (
                args.scene_state_generated_rollout_max_tokens
            ),
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
            "scene_state_identity_margin": training_protocol.get(
                "scene_state_identity_margin"
            ),
            "scene_state_margin_mode": training_protocol.get(
                "scene_state_margin_mode"
            ),
            "scene_state_identity_backward_mode": training_protocol.get(
                "scene_state_identity_backward_mode"
            ),
            "scene_state_identity_read_protocol": training_protocol.get(
                "scene_state_identity_read_protocol"
            ),
            "scene_state_identity_zero_protocol": training_protocol.get(
                "scene_state_identity_zero_protocol"
            ),
            "scene_state_semantic_mask_mode": training_protocol.get(
                "scene_state_semantic_mask_mode"
            ),
            "scene_state_semantic_loss_normalization": training_protocol.get(
                "scene_state_semantic_loss_normalization"
            ),
            "scene_state_identity_target_mode": training_protocol.get(
                "scene_state_identity_target_mode"
            ),
            "scene_state_source_manifest": training_protocol.get(
                "scene_state_source_manifest"
            ),
            "scene_state_full_correct_ce_weight": training_protocol.get(
                "scene_state_full_correct_ce_weight"
            ),
            "scene_state_correct_all_semantic_ce_weight": training_protocol.get(
                "scene_state_correct_all_semantic_ce_weight"
            ),
            "scene_state_donor_margin_weight": training_protocol.get(
                "scene_state_donor_margin_weight"
            ),
            "scene_state_identity_pairing_manifest_sha256": (
                None
                if scene_state_identity_pairing_manifest is None
                else scene_state_identity_pairing_manifest["manifest_sha256"]
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
            "cuda_memory": cuda_memory,
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
            "tokenized_dataset_sha256": tokenized_meta["tokenized_dataset_sha256"],
            "tokenized_cache_identity": tokenized_meta["tokenized_cache_identity"],
            "tokenized_cache_manifest_sha256": tokenized_meta[
                "tokenized_cache_manifest_sha256"
            ],
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
