from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    eager_attention_forward as qwen3_eager_attention_forward,
)

from deltamem.core.backbone_compat import (
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    HAS_SMOLLM3,
    Qwen3_5Attention,
    SmolLM3Attention,
    ensure_attention_compat_views,
    gemma4_apply_rotary_pos_emb,
    gemma4_eager_attention_forward,
    qwen3_5_apply_rotary_pos_emb,
    qwen3_5_eager_attention_forward,
    smollm3_apply_rotary_pos_emb,
    smollm3_eager_attention_forward,
)
from deltamem.core.hrm_rwkv7 import HRMRWKV7LowRankCore
from deltamem.kernels.affine_scan import triton_affine_scan, triton_scan_support
from deltamem.kernels.rwkv_ms_write_scan import (
    rwkv_ms_write_scan,
    write_slot_indices as rwkv_ms_write_slot_indices,
)

SUPPORTED_BASE_ATTENTION_TYPES = (Qwen3Attention,)
if HAS_SMOLLM3:
    SUPPORTED_BASE_ATTENTION_TYPES += (SmolLM3Attention,)
if Qwen3_5Attention is not None:
    SUPPORTED_BASE_ATTENTION_TYPES += (Qwen3_5Attention,)
if Gemma4TextAttention is not None:
    SUPPORTED_BASE_ATTENTION_TYPES += (Gemma4TextAttention,)


VALID_DELTA_HEADS = ("q", "k", "v", "o")
VALID_MEMORY_BACKENDS = ("delta_rule", "rwkv_ms")
VALID_STATE_UPDATE_MODES = ("standard", "lambda_outside", "no_lambda")
VALID_RWKV_MS_BOUNDARY_MODES = ("fixed_chunk",)
VALID_RWKV_MS_WRITE_MODES = ("recurrent", "last_token_overwrite")
VALID_MEMORY_PARTITION_ROUTING = ("soft",)
VALID_MEMORY_PARTITION_BASIS = ("shared",)
VALID_MEMORY_READOUT_MODES = (
    "delta",
    "direct_last_hidden",
    "projected_last_hidden",
    "projected_kv_slots",
    "projected_kv_rwkv_hybrid",
)
PROJECTED_KV_MEMORY_READOUT_MODES = frozenset(
    {"projected_kv_slots", "projected_kv_rwkv_hybrid"}
)
VALID_RWKV_MS_HYBRID_MODES = (
    "residual",
    "vector_gate",
    "scalar_gate",
    "addressed_value",
    "chunk_addressed_value",
    "recurrent_value",
)
RWKV_MS_ADDRESSED_VALUE_MODES = frozenset(
    {"addressed_value", "chunk_addressed_value"}
)
RWKV_MS_VALUE_BOTTLENECK_MODES = (
    RWKV_MS_ADDRESSED_VALUE_MODES | {"recurrent_value"}
)
VALID_MEMORY_WRITE_SOURCES = ("learned_hidden",)
VALID_MEMORY_WRITE_GRANULARITIES = (
    "token",
    "message_mean",
    "sentence_mean",
)
VALID_MEMORY_PARTITION_READ_MODES = ("softmax",)
VALID_GLOBAL_MEMORY_MODES = ("shared_rw",)
VALID_GLOBAL_MEMORY_MERGE_MODES = ("gated_residual",)
VALID_MEMORY_FUSION_MODES = (
    "add",
    "content_gated_add",
    "content_gated_qo_add",
)
CONTENT_GATED_MEMORY_FUSION_MODES = frozenset(
    {"content_gated_add", "content_gated_qo_add"}
)
VALID_MEMORY_FUSION_PLACEMENTS = (
    "attention_output",
    "post_attention_norm",
    "normalized_residual_correction",
    "post_attention_residual_hybrid",
)
MEMORY_FUSION_NORM_HOOK_PLACEMENTS = frozenset(
    {
        "post_attention_norm",
        "normalized_residual_correction",
        "post_attention_residual_hybrid",
    }
)
MEMORY_FUSION_NORMALIZED_RESIDUAL_PLACEMENTS = frozenset(
    {"normalized_residual_correction", "post_attention_residual_hybrid"}
)
VALID_DELTA_SCALE_GRANULARITIES = ("layer", "head")
VALID_DELTA_SCALE_PARAMETERIZATIONS = ("alpha_over_rank")


def normalize_delta_heads(heads: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        items = tuple(part.strip().lower() for part in heads.split(",") if part.strip())
    else:
        items = tuple(str(part).strip().lower() for part in heads if str(part).strip())
    if not items or items == ("none",):
        return ()
    invalid = [head for head in items if head not in VALID_DELTA_HEADS]
    if invalid:
        raise ValueError(f"Unsupported delta heads: {invalid}; expected subset of {VALID_DELTA_HEADS}")
    deduped: list[str] = []
    for head in items:
        if head not in deduped:
            deduped.append(head)
    return tuple(deduped)


def normalize_state_update_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_STATE_UPDATE_MODES:
        raise ValueError(
            f"Unsupported state update mode: {mode}; expected one of {VALID_STATE_UPDATE_MODES}"
        )
    return normalized


def normalize_memory_backend(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "delta":
        normalized = "delta_rule"
    if normalized not in VALID_MEMORY_BACKENDS:
        raise ValueError(
            f"Unsupported memory backend: {mode}; expected one of {VALID_MEMORY_BACKENDS}"
        )
    return normalized


def normalize_rwkv_ms_boundary_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "fixed":
        normalized = "fixed_chunk"
    if normalized not in VALID_RWKV_MS_BOUNDARY_MODES:
        raise ValueError(
            "Unsupported RWKV-MS boundary mode: "
            f"{mode}; expected one of {VALID_RWKV_MS_BOUNDARY_MODES}"
        )
    return normalized


def normalize_rwkv_ms_write_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in VALID_RWKV_MS_WRITE_MODES:
        raise ValueError(
            "Unsupported RWKV-MS write mode: "
            f"{mode}; expected one of {VALID_RWKV_MS_WRITE_MODES}"
        )
    return normalized


def normalize_memory_partition_routing(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_ROUTING:
        raise ValueError(
            "Unsupported memory partition routing mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_ROUTING}"
        )
    return normalized


def normalize_memory_partition_basis(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_BASIS:
        raise ValueError(
            "Unsupported memory partition basis mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_BASIS}"
        )
    return normalized


def normalize_memory_readout_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_READOUT_MODES:
        raise ValueError(
            "Only memory_readout_mode='delta', 'direct_last_hidden', "
            "'projected_last_hidden', 'projected_kv_slots', or "
            "'projected_kv_rwkv_hybrid' is supported. "
            f"Got {mode!r}."
        )
    return normalized


def normalize_rwkv_ms_hybrid_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_RWKV_MS_HYBRID_MODES:
        raise ValueError(
            "Unsupported RWKV-MS hybrid mode: "
            f"{mode}; expected one of {VALID_RWKV_MS_HYBRID_MODES}"
        )
    return normalized


def normalize_memory_write_source(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_WRITE_SOURCES:
        raise ValueError(
            "Unsupported memory write source: "
            f"{mode}; expected one of {VALID_MEMORY_WRITE_SOURCES}"
        )
    return normalized


def normalize_memory_write_granularity(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_WRITE_GRANULARITIES:
        raise ValueError(
            "Unsupported memory write granularity: "
            f"{mode}; expected one of {VALID_MEMORY_WRITE_GRANULARITIES}"
        )
    return normalized


def normalize_memory_partition_read_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MEMORY_PARTITION_READ_MODES:
        raise ValueError(
            "Unsupported memory partition read mode: "
            f"{mode}; expected one of {VALID_MEMORY_PARTITION_READ_MODES}"
        )
    return normalized


def normalize_global_memory_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_GLOBAL_MEMORY_MODES:
        raise ValueError(
            "Unsupported global memory mode: "
            f"{mode}; expected one of {VALID_GLOBAL_MEMORY_MODES}"
        )
    return normalized


def normalize_global_memory_merge_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_GLOBAL_MEMORY_MERGE_MODES:
        raise ValueError(
            "Unsupported global memory merge mode: "
            f"{mode}; expected one of {VALID_GLOBAL_MEMORY_MERGE_MODES}"
        )
    return normalized


def normalize_memory_fusion_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in VALID_MEMORY_FUSION_MODES:
        raise ValueError(
            "Unsupported memory fusion mode: "
            f"{mode}; expected one of {VALID_MEMORY_FUSION_MODES}"
        )
    return normalized


def normalize_memory_fusion_placement(placement: str) -> str:
    normalized = str(placement).strip().lower().replace("-", "_")
    if normalized not in VALID_MEMORY_FUSION_PLACEMENTS:
        raise ValueError(
            "Unsupported memory fusion placement: "
            f"{placement}; expected one of {VALID_MEMORY_FUSION_PLACEMENTS}"
        )
    return normalized


def normalize_delta_scale_granularity(granularity: str) -> str:
    normalized = str(granularity).strip().lower()
    if normalized not in VALID_DELTA_SCALE_GRANULARITIES:
        raise ValueError(
            "Unsupported delta scale granularity: "
            f"{granularity}; expected one of {VALID_DELTA_SCALE_GRANULARITIES}"
        )
    return normalized


def inverse_bounded_sigmoid(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        raise ValueError("max_value must be > 0")
    clipped = min(max(value / max_value, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


def normalize_delta_scale_parameterization(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DELTA_SCALE_PARAMETERIZATIONS:
        raise ValueError(
            "Unsupported delta scale parameterization: "
            f"{mode}; expected one of {VALID_DELTA_SCALE_PARAMETERIZATIONS}"
        )
    return normalized


@dataclass(frozen=True)
class HFDeltaMemConfig:
    rank: int = 8
    alpha: float = 16.0
    memory_backend: str = "delta_rule"
    beta_bias_init: float = -1.5
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"
    rankwise_gates: bool = True
    output_init: str = "zero"
    base_slice_ref_width: int = 8
    online_gain: float = 0.05
    num_state_heads: int = 1
    num_memory_partitions: int = 1
    num_global_memory_partitions: int = 0
    memory_partition_routing: str = "soft"
    memory_partition_basis: str = "shared"
    tie_memory_partition_read_write: bool = False
    memory_partition_read_mode: str = "softmax"
    memory_partition_sigmoid_gate_bias_init: float = -2.0
    slot_read_top_k: int = 0
    global_memory_mode: str = "shared_rw"
    global_memory_read_top_k: int = 0
    global_memory_merge_mode: str = "gated_residual"
    global_memory_gate_bias_init: float = -2.0
    global_memory_read_logit_bias: float = 0.0
    memory_reader_layers: tuple[int, ...] = ()
    memory_reader_hidden_size: int = 1024
    memory_reader_residual_scale: float = 0.1
    memory_reader_read_only: bool = True
    memory_readout_mode: str = "delta"
    projected_kv_key_dim: int = 32
    projected_kv_temperature: float = 16.0
    projected_kv_update_cosine_threshold: float = 0.95
    memory_write_source: str = "learned_hidden"
    memory_write_granularity: str = "token"
    memory_write_proposals_per_message: int = 2
    synthetic_memory_slots: int = 1
    latent_memory_layers: tuple[int, ...] = ()
    latent_memory_hidden_size: int = 1024
    latent_memory_residual_scale: float = 0.1
    latent_memory_slots: int = 4
    latent_memory_init_std: float = 0.002
    latent_gate_init: float = 0.01
    target_modules: tuple[str, ...] = ("self_attn",)
    target_layers: tuple[int, ...] = ()
    delta_heads: tuple[str, ...] = VALID_DELTA_HEADS
    delta_o_rmsnorm: bool = False
    delta_o_rmsnorm_eps: float = 1e-6
    memory_fusion_mode: str = "add"
    memory_fusion_gate_init: float = 0.1
    memory_fusion_placement: str = "attention_output"
    memory_fusion_residual_scale: float = 1.0
    memory_fusion_residual_scale_max: float = 1.0
    trainable_delta_scale: bool = False
    delta_scale_init: float = 1.0
    delta_scale_max: float = 2.0
    delta_scale_granularity: str = "layer"
    delta_scale_parameterization: str = "alpha_over_rank"
    rwkv_ms_num_states: int = 4
    rwkv_ms_chunk_size: int = 1024
    rwkv_ms_boundary_mode: str = "fixed_chunk"
    rwkv_ms_write_mode: str = "recurrent"
    rwkv_ms_erase_gate: float = 1.0
    rwkv_ms_read_top_k: int = 0
    rwkv_ms_mask_empty_slots: bool = False
    rwkv_ms_output_init_scale: float = 0.02
    rwkv_ms_semantics_version: int = 2
    rwkv_ms_hybrid_mode: str = "residual"
    rwkv_ms_hybrid_gain: float = 0.125

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_heads", normalize_delta_heads(self.delta_heads))
        object.__setattr__(self, "memory_backend", normalize_memory_backend(self.memory_backend))
        object.__setattr__(
            self,
            "state_update_mode",
            normalize_state_update_mode(self.state_update_mode),
        )
        if int(self.rwkv_ms_num_states) < 1:
            raise ValueError("rwkv_ms_num_states must be >= 1")
        if int(self.rwkv_ms_chunk_size) < 1:
            raise ValueError("rwkv_ms_chunk_size must be >= 1")
        if float(self.rwkv_ms_erase_gate) < 0.0:
            raise ValueError("rwkv_ms_erase_gate must be >= 0")
        if int(self.rwkv_ms_read_top_k) < 0:
            raise ValueError("rwkv_ms_read_top_k must be >= 0")
        if float(self.rwkv_ms_output_init_scale) < 0.0:
            raise ValueError("rwkv_ms_output_init_scale must be >= 0")
        if int(self.rwkv_ms_semantics_version) not in {1, 2}:
            raise ValueError("rwkv_ms_semantics_version must be 1 or 2")
        hybrid_gain = float(self.rwkv_ms_hybrid_gain)
        if not math.isfinite(hybrid_gain) or not (0.0 <= hybrid_gain <= 1.0):
            raise ValueError("rwkv_ms_hybrid_gain must be finite and satisfy 0 <= gain <= 1")
        object.__setattr__(
            self,
            "rwkv_ms_hybrid_mode",
            normalize_rwkv_ms_hybrid_mode(self.rwkv_ms_hybrid_mode),
        )
        object.__setattr__(self, "rwkv_ms_hybrid_gain", hybrid_gain)
        object.__setattr__(self, "rwkv_ms_num_states", int(self.rwkv_ms_num_states))
        object.__setattr__(self, "rwkv_ms_chunk_size", int(self.rwkv_ms_chunk_size))
        object.__setattr__(
            self,
            "rwkv_ms_boundary_mode",
            normalize_rwkv_ms_boundary_mode(self.rwkv_ms_boundary_mode),
        )
        object.__setattr__(
            self,
            "rwkv_ms_write_mode",
            normalize_rwkv_ms_write_mode(self.rwkv_ms_write_mode),
        )
        if self.rwkv_ms_write_mode != "recurrent" and self.memory_backend != "rwkv_ms":
            raise ValueError(
                "Non-recurrent RWKV-MS write modes require memory_backend='rwkv_ms'"
            )
        object.__setattr__(self, "rwkv_ms_erase_gate", float(self.rwkv_ms_erase_gate))
        object.__setattr__(self, "rwkv_ms_read_top_k", int(self.rwkv_ms_read_top_k))
        object.__setattr__(self, "rwkv_ms_mask_empty_slots", bool(self.rwkv_ms_mask_empty_slots))
        object.__setattr__(
            self,
            "rwkv_ms_output_init_scale",
            float(self.rwkv_ms_output_init_scale),
        )
        object.__setattr__(
            self,
            "rwkv_ms_semantics_version",
            int(self.rwkv_ms_semantics_version),
        )
        if int(self.num_state_heads) < 1:
            raise ValueError("num_state_heads must be >= 1")
        if int(self.num_memory_partitions) < 1:
            raise ValueError("num_memory_partitions must be >= 1")
        if int(self.num_global_memory_partitions) < 0:
            raise ValueError("num_global_memory_partitions must be >= 0")
        if int(self.num_global_memory_partitions) >= int(self.num_memory_partitions):
            raise ValueError(
                "num_global_memory_partitions must be smaller than num_memory_partitions"
            )
        if int(self.base_slice_ref_width) < 1:
            raise ValueError("base_slice_ref_width must be >= 1")
        if float(self.delta_o_rmsnorm_eps) <= 0.0:
            raise ValueError("delta_o_rmsnorm_eps must be > 0")
        if not 0.0 < float(self.memory_fusion_gate_init) < 1.0:
            raise ValueError("memory_fusion_gate_init must satisfy 0 < value < 1")
        memory_fusion_placement = normalize_memory_fusion_placement(
            self.memory_fusion_placement
        )
        memory_fusion_residual_scale = float(self.memory_fusion_residual_scale)
        if not math.isfinite(memory_fusion_residual_scale) or not (
            0.0 <= memory_fusion_residual_scale <= 1.0
        ):
            raise ValueError(
                "memory_fusion_residual_scale must be finite and satisfy 0 <= value <= 1"
            )
        memory_fusion_residual_scale_max = float(
            self.memory_fusion_residual_scale_max
        )
        if not math.isfinite(memory_fusion_residual_scale_max) or not (
            0.0 < memory_fusion_residual_scale_max <= 1.0
        ):
            raise ValueError(
                "memory_fusion_residual_scale_max must be finite and satisfy "
                "0 < value <= 1"
            )
        if (
            memory_fusion_placement == "post_attention_residual_hybrid"
            and memory_fusion_residual_scale > memory_fusion_residual_scale_max
        ):
            raise ValueError(
                "post_attention_residual_hybrid requires "
                "memory_fusion_residual_scale <= memory_fusion_residual_scale_max"
            )
        if float(self.delta_scale_init) <= 0.0:
            raise ValueError("delta_scale_init must be > 0")
        if float(self.delta_scale_max) <= 0.0:
            raise ValueError("delta_scale_max must be > 0")
        if float(self.delta_scale_init) >= float(self.delta_scale_max):
            raise ValueError("delta_scale_init must be smaller than delta_scale_max")
        object.__setattr__(self, "num_state_heads", int(self.num_state_heads))
        object.__setattr__(self, "num_memory_partitions", int(self.num_memory_partitions))
        object.__setattr__(
            self,
            "num_global_memory_partitions",
            int(self.num_global_memory_partitions),
        )
        object.__setattr__(self, "base_slice_ref_width", int(self.base_slice_ref_width))
        object.__setattr__(self, "delta_o_rmsnorm", bool(self.delta_o_rmsnorm))
        object.__setattr__(self, "delta_o_rmsnorm_eps", float(self.delta_o_rmsnorm_eps))
        object.__setattr__(
            self,
            "memory_fusion_mode",
            normalize_memory_fusion_mode(self.memory_fusion_mode),
        )
        object.__setattr__(
            self,
            "memory_fusion_gate_init",
            float(self.memory_fusion_gate_init),
        )
        object.__setattr__(
            self,
            "memory_fusion_placement",
            memory_fusion_placement,
        )
        object.__setattr__(
            self,
            "memory_fusion_residual_scale",
            memory_fusion_residual_scale,
        )
        object.__setattr__(
            self,
            "memory_fusion_residual_scale_max",
            memory_fusion_residual_scale_max,
        )
        object.__setattr__(self, "trainable_delta_scale", bool(self.trainable_delta_scale))
        object.__setattr__(self, "delta_scale_init", float(self.delta_scale_init))
        object.__setattr__(self, "delta_scale_max", float(self.delta_scale_max))
        object.__setattr__(
            self,
            "delta_scale_granularity",
            normalize_delta_scale_granularity(self.delta_scale_granularity),
        )
        object.__setattr__(
            self,
            "delta_scale_parameterization",
            normalize_delta_scale_parameterization(self.delta_scale_parameterization),
        )
        object.__setattr__(
            self,
            "memory_partition_routing",
            normalize_memory_partition_routing(self.memory_partition_routing),
        )
        object.__setattr__(
            self,
            "memory_partition_basis",
            normalize_memory_partition_basis(self.memory_partition_basis),
        )
        object.__setattr__(
            self,
            "memory_partition_read_mode",
            normalize_memory_partition_read_mode(self.memory_partition_read_mode),
        )
        object.__setattr__(
            self,
            "global_memory_mode",
            normalize_global_memory_mode(self.global_memory_mode),
        )
        object.__setattr__(
            self,
            "global_memory_merge_mode",
            normalize_global_memory_merge_mode(self.global_memory_merge_mode),
        )
        if int(self.slot_read_top_k) < 0:
            raise ValueError("slot_read_top_k must be >= 0")
        if int(self.global_memory_read_top_k) < 0:
            raise ValueError("global_memory_read_top_k must be >= 0")
        if int(self.synthetic_memory_slots) < 1:
            raise ValueError("synthetic_memory_slots must be >= 1")
        if int(self.memory_write_proposals_per_message) < 1:
            raise ValueError("memory_write_proposals_per_message must be >= 1")
        if int(self.latent_memory_hidden_size) < 1:
            raise ValueError("latent_memory_hidden_size must be >= 1")
        if int(self.latent_memory_slots) < 1:
            raise ValueError("latent_memory_slots must be >= 1")
        if float(self.latent_memory_init_std) <= 0.0:
            raise ValueError("latent_memory_init_std must be > 0")
        if float(self.latent_gate_init) <= 0.0:
            raise ValueError("latent_gate_init must be > 0")
        object.__setattr__(self, "slot_read_top_k", int(self.slot_read_top_k))
        object.__setattr__(
            self,
            "global_memory_read_top_k",
            int(self.global_memory_read_top_k),
        )
        object.__setattr__(
            self,
            "global_memory_gate_bias_init",
            float(self.global_memory_gate_bias_init),
        )
        object.__setattr__(
            self,
            "memory_partition_sigmoid_gate_bias_init",
            float(self.memory_partition_sigmoid_gate_bias_init),
        )
        object.__setattr__(
            self,
            "global_memory_read_logit_bias",
            float(self.global_memory_read_logit_bias),
        )
        object.__setattr__(
            self,
            "synthetic_memory_slots",
            int(self.synthetic_memory_slots),
        )
        object.__setattr__(
            self,
            "memory_write_proposals_per_message",
            int(self.memory_write_proposals_per_message),
        )
        object.__setattr__(
            self,
            "latent_memory_hidden_size",
            int(self.latent_memory_hidden_size),
        )
        object.__setattr__(
            self,
            "latent_memory_slots",
            int(self.latent_memory_slots),
        )
        object.__setattr__(
            self,
            "latent_memory_init_std",
            float(self.latent_memory_init_std),
        )
        object.__setattr__(
            self,
            "latent_gate_init",
            float(self.latent_gate_init),
        )
        object.__setattr__(
            self,
            "memory_readout_mode",
            normalize_memory_readout_mode(self.memory_readout_mode),
        )
        projected_kv_key_dim = int(self.projected_kv_key_dim)
        projected_kv_temperature = float(self.projected_kv_temperature)
        projected_kv_update_cosine_threshold = float(
            self.projected_kv_update_cosine_threshold
        )
        if projected_kv_key_dim < 1:
            raise ValueError("projected_kv_key_dim must be >= 1")
        if not math.isfinite(projected_kv_temperature) or projected_kv_temperature <= 0.0:
            raise ValueError("projected_kv_temperature must be finite and > 0")
        if not math.isfinite(projected_kv_update_cosine_threshold) or not (
            -1.0 <= projected_kv_update_cosine_threshold <= 1.0
        ):
            raise ValueError(
                "projected_kv_update_cosine_threshold must be finite and satisfy "
                "-1 <= value <= 1"
            )
        object.__setattr__(self, "projected_kv_key_dim", projected_kv_key_dim)
        object.__setattr__(self, "projected_kv_temperature", projected_kv_temperature)
        object.__setattr__(
            self,
            "projected_kv_update_cosine_threshold",
            projected_kv_update_cosine_threshold,
        )
        if self.memory_readout_mode in PROJECTED_KV_MEMORY_READOUT_MODES:
            if self.memory_backend != "rwkv_ms":
                raise ValueError(
                    "Projected-KV memory readout requires "
                    "memory_backend='rwkv_ms'"
                )
        elif (
            projected_kv_key_dim != 32
            or projected_kv_temperature != 16.0
            or projected_kv_update_cosine_threshold != 0.95
        ):
            raise ValueError(
                "projected_kv_* options require "
                "a projected-KV memory readout mode"
            )
        object.__setattr__(
            self,
            "memory_write_source",
            normalize_memory_write_source(self.memory_write_source),
        )
        object.__setattr__(
            self,
            "memory_write_granularity",
            normalize_memory_write_granularity(self.memory_write_granularity),
        )
        if (
            self.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and self.memory_write_granularity != "token"
        ):
            raise ValueError(
                "projected_kv_rwkv_hybrid requires memory_write_granularity='token'"
            )
        if self.memory_reader_layers:
            raise ValueError(
                "memory_reader_layers is archived; active Delta-Mem only keeps TSW / MSW / SSW paths."
            )
        if self.num_memory_partitions != 1:
            raise ValueError(
                "num_memory_partitions is archived; active Delta-Mem only supports dense single-state memory (num_memory_partitions=1)."
            )
        if self.num_global_memory_partitions != 0:
            raise ValueError(
                "num_global_memory_partitions is archived; active Delta-Mem does not support global partitions."
            )
        if self.memory_partition_routing != "soft":
            raise ValueError(
                "memory_partition_routing is archived; active Delta-Mem only supports memory_partition_routing='soft'."
            )
        if self.memory_partition_basis != "shared":
            raise ValueError(
                "memory_partition_basis is archived; active Delta-Mem only supports memory_partition_basis='shared'."
            )
        if self.tie_memory_partition_read_write:
            raise ValueError(
                "tie_memory_partition_read_write is archived; active Delta-Mem only supports the dense single-state path."
            )
        if self.memory_partition_read_mode != "softmax":
            raise ValueError(
                "memory_partition_read_mode is archived; active Delta-Mem only supports memory_partition_read_mode='softmax'."
            )
        if self.slot_read_top_k != 0:
            raise ValueError(
                "slot_read_top_k is archived; active Delta-Mem only supports slot_read_top_k=0."
            )
        if self.global_memory_mode != "shared_rw":
            raise ValueError(
                "global_memory_mode is archived; active Delta-Mem only supports global_memory_mode='shared_rw'."
            )
        if self.global_memory_read_top_k != 0:
            raise ValueError(
                "global_memory_read_top_k is archived; active Delta-Mem only supports global_memory_read_top_k=0."
            )
        if self.global_memory_merge_mode != "gated_residual":
            raise ValueError(
                "global_memory_merge_mode is archived; active Delta-Mem only supports global_memory_merge_mode='gated_residual'."
            )
        if self.memory_write_source != "learned_hidden":
            raise ValueError(
                "memory_write_source is archived; active Delta-Mem only supports memory_write_source='learned_hidden'."
            )
        if self.memory_write_granularity == "message_proposals":
            raise ValueError(
                "message_proposals is archived; active Delta-Mem only supports token / message_mean / sentence_mean writes."
            )
        if self.memory_write_proposals_per_message != 2:
            raise ValueError(
                "memory_write_proposals_per_message is archived together with message_proposals writes."
            )
        if self.synthetic_memory_slots != 1:
            raise ValueError(
                "synthetic_memory_slots is archived together with synthetic_kv readout."
            )
        if self.latent_memory_layers:
            raise ValueError(
                "latent memory readouts are archived; active Delta-Mem only supports "
                "memory_readout_mode='delta', 'direct_last_hidden', or "
                "'projected_last_hidden', 'projected_kv_slots', or "
                "'projected_kv_rwkv_hybrid'."
            )
        if self.num_state_heads > 1 and self.num_memory_partitions > 1:
            raise ValueError(
                "num_state_heads > 1 is currently only supported with num_memory_partitions == 1"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HFDeltaMemConfig":
        data = dict(data)
        if (
            normalize_memory_backend(data.get("memory_backend", "delta_rule")) == "rwkv_ms"
            and "rwkv_ms_semantics_version" not in data
        ):
            # Checkpoints written before v2 used raw sources, lambda-scaled
            # carry, and magnitude-sensitive routing.
            data["rwkv_ms_semantics_version"] = 1
        if "target_modules" in data and isinstance(data["target_modules"], list):
            data["target_modules"] = tuple(data["target_modules"])
        if "memory_reader_layers" in data and isinstance(data["memory_reader_layers"], list):
            data = dict(data)
            data["memory_reader_layers"] = tuple(data["memory_reader_layers"])
        if "target_layers" in data and isinstance(data["target_layers"], list):
            data = dict(data)
            data["target_layers"] = tuple(data["target_layers"])
        if "latent_memory_layers" in data and isinstance(data["latent_memory_layers"], list):
            data = dict(data)
            data["latent_memory_layers"] = tuple(data["latent_memory_layers"])
        if "delta_heads" in data and isinstance(data["delta_heads"], list):
            data = dict(data)
            data["delta_heads"] = tuple(data["delta_heads"])
        return cls(**data)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "delta_mem_config.json").write_text(
            json.dumps(self.to_dict(), indent=2)
        )

    @classmethod
    def from_pretrained(cls, input_dir: str | Path) -> "HFDeltaMemConfig":
        input_path = Path(input_dir)
        return cls.from_dict(
            json.loads((input_path / "delta_mem_config.json").read_text())
        )


class DeltaMemAttention(nn.Module):
    def __init__(self, base: Qwen3Attention | SmolLM3Attention, config: HFDeltaMemConfig) -> None:
        super().__init__()
        self.base = ensure_attention_compat_views(base)
        base = self.base
        self.config = base.config
        self.delta_config = config
        self.layer_idx = base.layer_idx
        self.head_dim = base.head_dim
        self.num_key_value_groups = base.num_key_value_groups
        self.scaling = base.scaling
        self.attention_dropout = base.attention_dropout
        self.is_causal = base.is_causal
        self.sliding_window = getattr(base, "sliding_window", getattr(base.config, "sliding_window", None))
        self.is_smollm3_attention = isinstance(base, SmolLM3Attention)
        self.is_qwen3_5_attention = Qwen3_5Attention is not None and isinstance(base, Qwen3_5Attention)
        self.is_gemma4_attention = Gemma4TextAttention is not None and isinstance(base, Gemma4TextAttention)
        if self.is_smollm3_attention:
            self.eager_attention_forward = smollm3_eager_attention_forward
        elif self.is_qwen3_5_attention:
            self.eager_attention_forward = qwen3_5_eager_attention_forward
        elif self.is_gemma4_attention:
            self.eager_attention_forward = gemma4_eager_attention_forward
        else:
            self.eager_attention_forward = qwen3_eager_attention_forward
        self.layer_type = getattr(base, "layer_type", None)
        self.is_sliding = getattr(base, "is_sliding", False)
        self.is_kv_shared_layer = getattr(base, "is_kv_shared_layer", False)
        self.kv_shared_layer_index = getattr(base, "kv_shared_layer_index", None)
        self.store_full_length_kv = getattr(base, "store_full_length_kv", False)
        self.has_packed_qkv_proj = hasattr(base, "qkv_proj") and getattr(base, "qkv_proj", None) is not None

        self.rank = config.rank
        self.num_state_heads = config.num_state_heads
        self.state_read_dim = self.rank * self.num_state_heads
        self.multi_head_state = self.num_state_heads > 1
        self.memory_backend = config.memory_backend
        self.memory_readout_mode = config.memory_readout_mode
        self.projected_kv_key_dim = config.projected_kv_key_dim
        self.projected_kv_temperature = config.projected_kv_temperature
        self.projected_kv_update_cosine_threshold = (
            config.projected_kv_update_cosine_threshold
        )
        self.rwkv_ms_num_states = config.rwkv_ms_num_states
        self.rwkv_ms_chunk_size = config.rwkv_ms_chunk_size
        self.rwkv_ms_boundary_mode = config.rwkv_ms_boundary_mode
        self.rwkv_ms_write_mode = config.rwkv_ms_write_mode
        self.rwkv_ms_erase_gate = config.rwkv_ms_erase_gate
        self.rwkv_ms_read_top_k = config.rwkv_ms_read_top_k
        self.rwkv_ms_mask_empty_slots = config.rwkv_ms_mask_empty_slots
        self.rwkv_ms_output_init_scale = config.rwkv_ms_output_init_scale
        self.rwkv_ms_semantics_version = config.rwkv_ms_semantics_version
        self.rwkv_ms_hybrid_mode = config.rwkv_ms_hybrid_mode
        self.rwkv_ms_hybrid_gain = config.rwkv_ms_hybrid_gain
        self.delta_scaling = config.alpha / config.rank
        self.trainable_delta_scale = config.trainable_delta_scale
        self.delta_scale_max = config.delta_scale_max
        self.delta_scale_granularity = config.delta_scale_granularity
        self.memory_fusion_mode = config.memory_fusion_mode
        self.memory_fusion_gate_init = config.memory_fusion_gate_init
        self.memory_fusion_placement = config.memory_fusion_placement
        self.memory_fusion_residual_scale = config.memory_fusion_residual_scale
        self.memory_fusion_residual_scale_max = config.memory_fusion_residual_scale_max
        self.normalize_qk = config.normalize_qk
        self.couple_lambda = config.couple_lambda
        self.state_update_mode = config.state_update_mode
        self.rankwise_gates = config.rankwise_gates
        self.output_init = config.output_init
        self.base_slice_ref_width = config.base_slice_ref_width
        self.online_gain = config.online_gain
        self.num_memory_partitions = config.num_memory_partitions
        self.num_global_memory_partitions = config.num_global_memory_partitions
        self.memory_partition_routing = config.memory_partition_routing
        self.memory_partition_basis = config.memory_partition_basis
        self.tie_memory_partition_read_write = config.tie_memory_partition_read_write
        self.memory_partition_read_mode = config.memory_partition_read_mode
        self.memory_partition_sigmoid_gate_bias_init = config.memory_partition_sigmoid_gate_bias_init
        self.slot_read_top_k = config.slot_read_top_k
        self.global_memory_mode = config.global_memory_mode
        self.global_memory_read_top_k = config.global_memory_read_top_k
        self.global_memory_merge_mode = config.global_memory_merge_mode
        self.global_memory_gate_bias_init = config.global_memory_gate_bias_init
        self.global_memory_read_logit_bias = config.global_memory_read_logit_bias
        self.gate_dim_per_head = config.rank if config.rankwise_gates else 1
        self.gate_dim = self.gate_dim_per_head * self.num_state_heads
        self.active_delta_heads = frozenset(config.delta_heads)
        if self.trainable_delta_scale:
            scale_shape = (len(VALID_DELTA_HEADS),) if self.delta_scale_granularity == "head" else (1,)
            init_raw = inverse_bounded_sigmoid(config.delta_scale_init, self.delta_scale_max)
            self.delta_scale_raw = nn.Parameter(torch.full(scale_shape, init_raw))
        self.delta_o_rmsnorm = config.delta_o_rmsnorm
        self.delta_o_rmsnorm_eps = config.delta_o_rmsnorm_eps

        if self.is_gemma4_attention and self.is_kv_shared_layer:
            unsupported_heads = sorted(self.active_delta_heads - {"q", "o"})
            if unsupported_heads:
                raise ValueError(
                    "Gemma4 KV-shared attention layers support only Q/O Delta-Mem; "
                    f"unsupported delta heads: {unsupported_heads}"
                )

        hidden_size = base.q_proj.in_features
        self.hidden_size = hidden_size
        if self.is_qwen3_5_attention:
            self.query_out_features = int(base.config.num_attention_heads) * self.head_dim
            expected_q_proj_width = self.query_out_features * 2
            if base.q_proj.out_features != expected_q_proj_width:
                raise ValueError(
                    "Qwen3.5 gated q_proj width mismatch: "
                    f"expected {expected_q_proj_width}, got {base.q_proj.out_features}"
                )
        else:
            self.query_out_features = base.q_proj.out_features
        if self.is_gemma4_attention and self.is_kv_shared_layer:
            self.num_key_value_heads = int(
                base.config.num_global_key_value_heads
                if base.use_alternative_attention
                else base.config.num_key_value_heads
            )
            self.key_out_features = self.num_key_value_heads * self.head_dim
            self.base_v_out_features = self.key_out_features
        else:
            self.key_out_features = base.k_proj.out_features
            self.base_v_out_features = (
                base.v_proj.out_features if base.v_proj is not None else base.k_proj.out_features
            )
            self.num_key_value_heads = base.k_proj.out_features // self.head_dim
        self.partition_state_dim = config.rank * config.rank
        self.memory_write_source = config.memory_write_source
        self.memory_write_granularity = config.memory_write_granularity
        self.memory_write_proposals_per_message = config.memory_write_proposals_per_message
        self.hrm_rwkv7_core = HRMRWKV7LowRankCore(
            dim=self.state_read_dim,
            head_size=self.rank,
            layer_id=self.layer_idx,
            n_layer=base.config.num_hidden_layers,
            output_init_scale=(
                0.0 if self.output_init == "zero" else self.rwkv_ms_output_init_scale
            ),
        ) if self.memory_backend == "rwkv_ms" else None
        memory_qk_trainable = self.memory_backend != "rwkv_ms"
        self.memory_q_proj = nn.Parameter(
            torch.empty(self.state_read_dim, hidden_size),
            requires_grad=memory_qk_trainable,
        )
        self.memory_k_proj = nn.Parameter(
            torch.empty(self.state_read_dim, hidden_size),
            requires_grad=memory_qk_trainable,
        )
        self.memory_v_proj = nn.Parameter(torch.empty(self.state_read_dim, hidden_size))
        if self.memory_readout_mode in PROJECTED_KV_MEMORY_READOUT_MODES:
            self.projected_kv_key_proj = nn.Parameter(
                torch.empty(self.projected_kv_key_dim, hidden_size)
            )

        self.delta_q_proj = nn.Parameter(torch.empty(self.query_out_features, self.state_read_dim))
        self.delta_k_proj = nn.Parameter(torch.empty(self.key_out_features, self.state_read_dim))
        self.delta_v_proj = nn.Parameter(torch.empty(self.base_v_out_features, self.state_read_dim))
        self.delta_o_proj = nn.Parameter(torch.empty(base.o_proj.out_features, self.state_read_dim))
        if self.delta_o_rmsnorm:
            self.delta_o_rmsnorm_weight = nn.Parameter(torch.ones(base.o_proj.out_features))
        if self.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES:
            self.memory_fusion_hidden_weight = nn.Parameter(torch.empty(1, hidden_size))
            self.memory_fusion_read_weight = nn.Parameter(torch.empty(1, self.state_read_dim))
            self.memory_fusion_bias = nn.Parameter(torch.empty(1))
        if self.memory_fusion_placement == "post_attention_residual_hybrid":
            self.memory_fusion_residual_gain_raw = nn.Parameter(torch.empty(1))

        self.beta_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), config.beta_bias_init))
        if not config.couple_lambda:
            self.lambda_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
            self.lambda_bias = nn.Parameter(
                torch.full((self.gate_dim,), -config.beta_bias_init)
            )

        self.reset_parameters()
        self.delta_state: torch.Tensor | None = None
        self.direct_last_hidden: torch.Tensor | None = None
        self.projected_last_hidden: torch.Tensor | None = None
        self.projected_kv_keys: torch.Tensor | None = None
        self.projected_kv_values: torch.Tensor | None = None
        self.projected_kv_occupied: torch.Tensor | None = None
        self.projected_kv_surprise: torch.Tensor | None = None
        self.rwkv_ms_positions: torch.Tensor | None = None
        self.rwkv_ms_previous_source: torch.Tensor | None = None
        self.read_context_mask: torch.Tensor | None = None
        self.read_representation_capture_mask: torch.Tensor | None = None
        self.last_read_representation: torch.Tensor | None = None
        self.last_beta_mean: torch.Tensor | None = None
        self.last_lambda_mean: torch.Tensor | None = None
        self.write_enabled = True
        self.last_write_routes: torch.Tensor | None = None
        self.last_read_routes: torch.Tensor | None = None
        self.last_read_route_logits: torch.Tensor | None = None
        self.last_base_o_norm: torch.Tensor | None = None
        self.last_delta_o_norm: torch.Tensor | None = None
        self.last_delta_o_ratio: torch.Tensor | None = None
        self.last_delta_o_gate_mean: torch.Tensor | None = None
        self.last_delta_o_gate_min: torch.Tensor | None = None
        self.last_delta_o_gate_max: torch.Tensor | None = None
        self.last_delta_o_gate_lt_001_fraction: torch.Tensor | None = None
        self.last_delta_o_gate_gt_099_fraction: torch.Tensor | None = None
        self.last_fused_delta_o_norm: torch.Tensor | None = None
        self.last_fused_delta_o_ratio: torch.Tensor | None = None
        self.last_delta_o_base_cosine: torch.Tensor | None = None
        self.last_fused_o_ratio: torch.Tensor | None = None
        self.last_applied_memory_correction_norm: torch.Tensor | None = None
        self.last_applied_memory_correction_ratio: torch.Tensor | None = None
        self.last_memory_residual_norm: torch.Tensor | None = None
        self.last_memory_residual_ratio: torch.Tensor | None = None
        self.last_memory_residual_gain: torch.Tensor | None = None
        self._eval_memory_delta_controller: Callable[..., torch.Tensor] | None = None
        self._pending_post_attention_delta: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ] | None = None
        self._post_attention_norm_hook_handle = None
        self.write_message_ids: torch.Tensor | None = None
        self.write_sentence_ids: torch.Tensor | None = None
        self.projected_kv_write_key_mask: torch.Tensor | None = None
        self.projected_kv_write_value_mask: torch.Tensor | None = None
        self.projected_kv_write_slot_indices: torch.Tensor | None = None
        self.projected_kv_read_query_mask: torch.Tensor | None = None
        self.scan_impl = os.environ.get("DELTA_MEM_SCAN_IMPL", "auto")

    def _normalize_query_states(self, states: torch.Tensor) -> torch.Tensor:
        q_norm = getattr(self.base, "q_norm", None)
        if q_norm is None:
            return states
        return q_norm(states)

    def _normalize_key_states(self, states: torch.Tensor) -> torch.Tensor:
        k_norm = getattr(self.base, "k_norm", None)
        if k_norm is None:
            return states
        return k_norm(states)

    def _normalize_value_states(self, states: torch.Tensor) -> torch.Tensor:
        v_norm = getattr(self.base, "v_norm", None)
        if v_norm is None:
            return states
        return v_norm(states)

    def _apply_standard_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_smollm3_attention:
            if not bool(getattr(self.base, "use_rope", True)):
                return query_states, key_states
            return smollm3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.is_gemma4_attention:
            if gemma4_apply_rotary_pos_emb is None:  # pragma: no cover
                raise RuntimeError("Gemma4 rotary function is unavailable")
            query_states = gemma4_apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=1)
            key_states = gemma4_apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=1)
            return query_states, key_states
        if self.is_qwen3_5_attention:
            if qwen3_5_apply_rotary_pos_emb is None:  # pragma: no cover
                raise RuntimeError("Qwen3.5 rotary function is unavailable")
            return qwen3_5_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def _query_projection_weight(self) -> torch.Tensor:
        if not self.is_qwen3_5_attention:
            return self.base.q_proj.weight
        return (
            self.base.q_proj.weight.view(-1, self.head_dim * 2, self.hidden_size)[:, : self.head_dim, :]
            .reshape(self.query_out_features, self.hidden_size)
        )

    def _init_delta_head(self, head: nn.Parameter, base_weight: torch.Tensor) -> None:
        if self.output_init == "zero":
            nn.init.zeros_(head)
            return
        if self.output_init == "random":
            nn.init.kaiming_uniform_(head, a=math.sqrt(5))
            with torch.no_grad():
                head.mul_(self.online_gain)
            return
        if self.output_init not in {"base_slice", "base_slice_fixed"}:
            raise ValueError(f"Unsupported output_init: {self.output_init}")
        with torch.no_grad():
            if self.output_init == "base_slice":
                slice_width = min(self.rank, base_weight.shape[1])
            else:
                slice_width = min(self.base_slice_ref_width, self.rank, base_weight.shape[1])
            head.zero_()
            if slice_width == 0:
                return
            base_slice = base_weight[:, :slice_width].detach().clone().float()
            base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
            head[:, :slice_width].copy_((base_slice * self.online_gain).to(head.dtype))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.memory_q_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_k_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_v_proj, a=math.sqrt(5))
        if self.memory_readout_mode in PROJECTED_KV_MEMORY_READOUT_MODES:
            nn.init.kaiming_uniform_(self.projected_kv_key_proj, a=math.sqrt(5))
        self._init_delta_head(self.delta_q_proj, self._query_projection_weight())
        if self.is_gemma4_attention and self.is_kv_shared_layer:
            nn.init.zeros_(self.delta_k_proj)
            nn.init.zeros_(self.delta_v_proj)
        else:
            self._init_delta_head(self.delta_k_proj, self.base.k_proj.weight)
            self._init_delta_head(
                self.delta_v_proj,
                self.base.v_proj.weight if self.base.v_proj is not None else self.base.k_proj.weight,
            )
        self._init_delta_head(self.delta_o_proj, self.base.o_proj.weight)
        for head_name, param in (
            ("q", self.delta_q_proj),
            ("k", self.delta_k_proj),
            ("v", self.delta_v_proj),
            ("o", self.delta_o_proj),
        ):
            if head_name not in self.active_delta_heads:
                nn.init.zeros_(param)
        if self.delta_o_rmsnorm:
            nn.init.ones_(self.delta_o_rmsnorm_weight)
        if self.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES:
            nn.init.zeros_(self.memory_fusion_hidden_weight)
            nn.init.zeros_(self.memory_fusion_read_weight)
            gate_logit = math.log(
                self.memory_fusion_gate_init / (1.0 - self.memory_fusion_gate_init)
            )
            nn.init.constant_(self.memory_fusion_bias, gate_logit)
        if self.memory_fusion_placement == "post_attention_residual_hybrid":
            self.set_memory_fusion_residual_gain(
                self.memory_fusion_residual_scale
            )
        nn.init.zeros_(self.beta_proj)
        if not self.couple_lambda:
            nn.init.zeros_(self.lambda_proj)

    def reset_state(self) -> None:
        self.delta_state = None
        self.direct_last_hidden = None
        self.projected_last_hidden = None
        self.projected_kv_keys = None
        self.projected_kv_values = None
        self.projected_kv_occupied = None
        self.projected_kv_surprise = None
        self.rwkv_ms_positions = None
        self.rwkv_ms_previous_source = None
        self.read_context_mask = None
        self.read_representation_capture_mask = None
        self.last_read_representation = None
        self.last_beta_mean = None
        self.last_lambda_mean = None
        self.last_write_routes = None
        self.last_read_routes = None
        self.last_read_route_logits = None
        self.last_base_o_norm = None
        self.last_delta_o_norm = None
        self.last_delta_o_ratio = None
        self.last_delta_o_gate_mean = None
        self.last_delta_o_gate_min = None
        self.last_delta_o_gate_max = None
        self.last_delta_o_gate_lt_001_fraction = None
        self.last_delta_o_gate_gt_099_fraction = None
        self.last_fused_delta_o_norm = None
        self.last_fused_delta_o_ratio = None
        self.last_delta_o_base_cosine = None
        self.last_fused_o_ratio = None
        self.last_applied_memory_correction_norm = None
        self.last_applied_memory_correction_ratio = None
        self.last_memory_residual_norm = None
        self.last_memory_residual_ratio = None
        self.last_memory_residual_gain = None
        self._pending_post_attention_delta = None
        self.write_message_ids = None
        self.write_sentence_ids = None
        self.projected_kv_write_key_mask = None
        self.projected_kv_write_value_mask = None
        self.projected_kv_write_slot_indices = None
        self.projected_kv_read_query_mask = None

    def set_write_enabled(self, enabled: bool) -> None:
        self.last_read_route_logits = None
        if enabled:
            self.read_context_mask = None
            self.projected_kv_read_query_mask = None
        else:
            self.write_message_ids = None
            self.write_sentence_ids = None
            self.projected_kv_write_key_mask = None
            self.projected_kv_write_value_mask = None
            self.projected_kv_write_slot_indices = None
        self.write_enabled = enabled

    def set_write_message_ids(self, message_ids: torch.Tensor | None) -> None:
        self.write_message_ids = message_ids

    def set_write_sentence_ids(self, sentence_ids: torch.Tensor | None) -> None:
        self.write_sentence_ids = sentence_ids

    def set_projected_kv_write_spans(
        self,
        key_mask: torch.Tensor | None,
        value_mask: torch.Tensor | None,
        slot_indices: torch.Tensor | None = None,
    ) -> None:
        if (key_mask is None) != (value_mask is None):
            raise ValueError(
                "Projected-KV key and value write masks must both be set or both be absent"
            )
        if key_mask is None and slot_indices is not None:
            raise ValueError(
                "Projected-KV forced write slots require key and value write masks"
            )
        self.projected_kv_write_key_mask = key_mask
        self.projected_kv_write_value_mask = value_mask
        self.projected_kv_write_slot_indices = slot_indices

    def set_projected_kv_read_query_mask(
        self,
        query_mask: torch.Tensor | None,
    ) -> None:
        if query_mask is not None and query_mask.ndim != 2:
            raise ValueError(
                "Projected-KV read query mask must have shape [batch, sequence]"
            )
        self.projected_kv_read_query_mask = query_mask

    def is_trainable_parameter(self, sub_name: str) -> bool:
        if sub_name == "projected_kv_key_proj":
            return self.memory_readout_mode in PROJECTED_KV_MEMORY_READOUT_MODES
        if sub_name in {"memory_q_proj", "memory_k_proj"}:
            return self.memory_backend != "rwkv_ms"
        if sub_name == "memory_v_proj":
            return True
        if sub_name == "hrm_rwkv7_core.ln_x.bias":
            return False
        if sub_name == "delta_q_proj":
            return "q" in self.active_delta_heads
        if sub_name == "delta_k_proj":
            return "k" in self.active_delta_heads
        if sub_name == "delta_v_proj":
            return "v" in self.active_delta_heads
        if sub_name == "delta_o_proj":
            return "o" in self.active_delta_heads
        if sub_name == "delta_o_rmsnorm_weight":
            return self.delta_o_rmsnorm and "o" in self.active_delta_heads
        if sub_name == "memory_fusion_residual_gain_raw":
            return (
                self.memory_fusion_placement == "post_attention_residual_hybrid"
                and "o" in self.active_delta_heads
            )
        if sub_name.startswith("memory_fusion_"):
            return (
                self.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES
                and "o" in self.active_delta_heads
            )
        if sub_name == "delta_scale_raw":
            return self.trainable_delta_scale
        return True

    def _ensure_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self.delta_state is None
            or self.delta_state.size(0) != batch_size
            or self.delta_state.device != device
        ):
            if self.memory_backend == "rwkv_ms":
                self.delta_state = torch.zeros(
                    batch_size,
                    self.num_state_heads,
                    self.rwkv_ms_num_states,
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=torch.float32,
                )
                self.rwkv_ms_positions = torch.zeros(
                    batch_size,
                    device=device,
                    dtype=torch.long,
                )
                self.rwkv_ms_previous_source = torch.zeros(
                    batch_size,
                    self.state_read_dim,
                    device=device,
                    dtype=dtype,
                )
            elif self.multi_head_state:
                self.delta_state = torch.zeros(
                    batch_size,
                    self.num_state_heads,
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=dtype,
                )
            else:
                self.delta_state = torch.zeros(
                    batch_size,
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=dtype,
                )
        elif self.memory_backend == "rwkv_ms":
            if self.delta_state.dtype != torch.float32:
                self.delta_state = self.delta_state.float()
        elif self.delta_state.dtype != dtype:
            self.delta_state = self.delta_state.to(dtype=dtype)
        return self.delta_state

    def _reshape_state_heads(self, projected: torch.Tensor) -> torch.Tensor:
        if not self.multi_head_state:
            return projected
        return projected.view(*projected.shape[:-1], self.num_state_heads, self.rank)

    def _flatten_state_heads(self, projected: torch.Tensor) -> torch.Tensor:
        if not self.multi_head_state:
            return projected
        return projected.reshape(*projected.shape[:-2], self.state_read_dim)

    def _normalize_memory_projection(
        self,
        projected: torch.Tensor,
        *,
        force: bool = False,
    ) -> torch.Tensor:
        if self.normalize_qk or force:
            if self.multi_head_state and projected.size(-1) == self.state_read_dim:
                projected = self._reshape_state_heads(projected)
                projected = torch.tanh(projected)
                projected = F.normalize(projected, dim=-1, eps=1e-6)
                projected = self._flatten_state_heads(projected)
            else:
                projected = torch.tanh(projected)
                projected = F.normalize(projected, dim=-1, eps=1e-6)
        return projected

    def _split_packed_qkv(self, packed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_end = self.query_out_features
        key_end = query_end + self.key_out_features
        query_states = packed_qkv[..., :query_end]
        key_states = packed_qkv[..., query_end:key_end]
        value_states = packed_qkv[..., key_end:key_end + self.base_v_out_features]
        return query_states, key_states, value_states

    def _base_query_projection(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_packed_qkv_proj:
            packed_qkv = self.base.qkv_proj(hidden_states)
            query_states, _, _ = self._split_packed_qkv(packed_qkv)
            return query_states
        if self.is_qwen3_5_attention:
            query_states, _ = self._split_qwen3_5_query_gate(self.base.q_proj(hidden_states))
            return query_states
        return self.base.q_proj(hidden_states)

    def _base_qkv_projections(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_packed_qkv_proj:
            return self._split_packed_qkv(self.base.qkv_proj(hidden_states))
        query_states = self._base_query_projection(hidden_states)
        key_states = self.base.k_proj(hidden_states)
        value_states = self.base.v_proj(hidden_states) if self.base.v_proj is not None else key_states
        return query_states, key_states, value_states

    def _split_qwen3_5_query_gate(
        self,
        projected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_qwen3_5_attention:  # pragma: no cover - internal contract
            raise RuntimeError("Qwen3.5 query/gate splitting requested for another backbone")
        grouped = projected.view(*projected.shape[:-1], -1, self.head_dim * 2)
        query_states, output_gate = torch.chunk(grouped, 2, dim=-1)
        output_shape = (*projected.shape[:-1], self.query_out_features)
        return query_states.reshape(output_shape), output_gate.reshape(output_shape)

    def _compute_delta_qkv_from_reads(
        self,
        reads: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        delta_q = self._project_delta_head(reads, self.delta_q_proj, "q")
        delta_k = self._project_delta_head(reads, self.delta_k_proj, "k")
        delta_v = self._project_delta_head(reads, self.delta_v_proj, "v")
        return delta_q, delta_k, delta_v

    def _add_memory_delta(
        self,
        reference: torch.Tensor,
        raw_delta: torch.Tensor | None,
        *,
        head_name: str,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if raw_delta is None:
            return reference
        controller = self._eval_memory_delta_controller
        if controller is None:
            return reference + raw_delta.to(dtype=reference.dtype)
        if self.training:
            raise RuntimeError("Memory-delta interventions are restricted to eval mode")
        output = controller(self, head_name, reference, raw_delta, token_mask)
        if not torch.is_tensor(output):
            raise TypeError("Memory-delta intervention must return a tensor")
        if output.shape != reference.shape:
            raise ValueError(
                "Memory-delta intervention changed the projection shape: "
                f"head={head_name} reference={tuple(reference.shape)} "
                f"output={tuple(output.shape)}"
            )
        if output.device != reference.device or output.dtype != reference.dtype:
            raise ValueError(
                "Memory-delta intervention changed projection device or dtype: "
                f"head={head_name} reference={reference.device}/{reference.dtype} "
                f"output={output.device}/{output.dtype}"
            )
        return output

    def _apply_delta_qkv(
        self,
        hidden_states: torch.Tensor,
        delta_q: torch.Tensor | None,
        delta_k: torch.Tensor | None,
        delta_v: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.has_packed_qkv_proj:
            packed_qkv = self.base.qkv_proj(hidden_states)
            if self._eval_memory_delta_controller is not None:
                query_states, key_states, value_states = self._split_packed_qkv(packed_qkv)
                query_states = self._add_memory_delta(
                    query_states,
                    delta_q,
                    head_name="q",
                    token_mask=token_mask,
                )
                key_states = self._add_memory_delta(
                    key_states,
                    delta_k,
                    head_name="k",
                    token_mask=token_mask,
                )
                value_states = self._add_memory_delta(
                    value_states,
                    delta_v,
                    head_name="v",
                    token_mask=token_mask,
                )
                return query_states, key_states, value_states, None
            packed_delta_parts = []
            for delta_part, width in (
                (delta_q, self.query_out_features),
                (delta_k, self.key_out_features),
                (delta_v, self.base_v_out_features),
            ):
                if delta_part is None:
                    packed_delta_parts.append(packed_qkv.new_zeros(*packed_qkv.shape[:-1], width))
                else:
                    packed_delta_parts.append(delta_part.to(hidden_states.dtype))
            packed_qkv = packed_qkv + torch.cat(packed_delta_parts, dim=-1)
            query_states, key_states, value_states = self._split_packed_qkv(packed_qkv)
            return query_states, key_states, value_states, None
        projected_query = self.base.q_proj(hidden_states)
        output_gate = None
        if self.is_qwen3_5_attention:
            query_states, output_gate = self._split_qwen3_5_query_gate(projected_query)
        else:
            query_states = projected_query
        query_states = self._add_memory_delta(
            query_states,
            delta_q,
            head_name="q",
            token_mask=token_mask,
        )
        key_states = self.base.k_proj(hidden_states)
        key_states = self._add_memory_delta(
            key_states,
            delta_k,
            head_name="k",
            token_mask=token_mask,
        )
        value_states = self.base.v_proj(hidden_states) if self.base.v_proj is not None else key_states
        value_states = self._add_memory_delta(
            value_states,
            delta_v,
            head_name="v",
            token_mask=token_mask,
        )
        return query_states, key_states, value_states, output_gate

    def _forward_gemma4_shared_kv_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
        delta_q: torch.Tensor | None,
        delta_o: torch.Tensor | None,
        reads: torch.Tensor,
        read_mask: torch.Tensor | None,
        fusion_gate: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if shared_kv_states is None or self.layer_type not in shared_kv_states:
            raise ValueError(
                "Gemma4 KV-shared attention requires shared K/V states for "
                f"layer type {self.layer_type!r}"
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.base.q_proj(hidden_states)
        query_states = self._add_memory_delta(
            query_states,
            delta_q,
            head_name="q",
            token_mask=read_mask,
        )
        query_states = self._normalize_query_states(query_states.view(hidden_shape)).transpose(
            1, 2
        )
        cos, sin = position_embeddings
        if gemma4_apply_rotary_pos_emb is None:  # pragma: no cover
            raise RuntimeError("Gemma4 rotary function is unavailable")
        query_states = gemma4_apply_rotary_pos_emb(
            query_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

        key_states, value_states = shared_kv_states[self.layer_type]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)

        attention_interface = self.eager_attention_forward
        if self.base.config._attn_implementation != "eager":
            if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.base.config._attn_implementation,
                    self.eager_attention_forward,
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.base.config._attn_implementation
                ]

        attn_kwargs = dict(kwargs)
        if self.sliding_window is not None:
            attn_kwargs["sliding_window"] = self.sliding_window
        attn_output, attn_weights = attention_interface(
            self.base,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base.attention_dropout,
            scaling=self.base.scaling,
            **attn_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        base_o_output = self.base.o_proj(attn_output)
        return (
            self._fuse_delta_o_output(
                base_o_output,
                delta_o,
                hidden_states,
                reads,
                read_mask,
                fusion_gate,
            ),
            attn_weights,
        )

    def _memory_sequence_projections(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_weights = [self.beta_proj]
        split_sizes = [self.gate_dim]
        if not self.couple_lambda:
            gate_weights.append(self.lambda_proj)
            split_sizes.append(self.gate_dim)

        packed_gate_weight = torch.cat(gate_weights, dim=0)
        packed_gates = F.linear(hidden_states, packed_gate_weight)
        gate_splits = torch.split(packed_gates, split_sizes, dim=-1)

        if self.memory_backend == "rwkv_ms":
            memory_v = F.linear(hidden_states, self.memory_v_proj)
            if self.rwkv_ms_semantics_version >= 2:
                memory_v = self._normalize_memory_projection(memory_v, force=True)
            # The RWKV core derives its own r/k/v features from this source.
            # Preserve the shared projection API without evaluating dead q/k matrices.
            memory_q = memory_v
            memory_k = memory_v
        else:
            packed_memory_weight = torch.cat(
                [self.memory_q_proj, self.memory_k_proj, self.memory_v_proj],
                dim=0,
            )
            packed_memory = F.linear(hidden_states, packed_memory_weight)
            memory_q, memory_k, memory_v = torch.split(
                packed_memory,
                [self.state_read_dim, self.state_read_dim, self.state_read_dim],
                dim=-1,
            )
            memory_q = self._normalize_memory_projection(memory_q)
            memory_k = self._normalize_memory_projection(memory_k)

        beta = torch.sigmoid(
            gate_splits[0]
            + self.beta_bias.view(*([1] * (hidden_states.dim() - 1)), self.gate_dim)
        ).unsqueeze(-1)
        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            lam = torch.sigmoid(
                gate_splits[1]
                + self.lambda_bias.view(
                    *([1] * (hidden_states.dim() - 1)),
                    self.gate_dim,
                )
            ).unsqueeze(-1)
        return memory_q, memory_k, memory_v, beta, lam

    def _partition_memory_projections(
        self,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return memory_q_seq, memory_k_seq, memory_v_seq

    def _memory_partition_routes(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        message_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ones = hidden_states.new_ones(*hidden_states.shape[:-1], 1)
        return ones, ones

    def _memory_update_coefficients(
        self,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        *,
        write_route_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta_rows = beta_seq.squeeze(-1) if beta_seq.ndim == 4 else beta_seq
        lambda_rows = lambda_seq.squeeze(-1) if lambda_seq.ndim == 4 else lambda_seq
        if self.multi_head_state:
            beta_rows = beta_rows.view(
                beta_rows.size(0),
                beta_rows.size(1),
                self.num_state_heads,
                self.gate_dim_per_head,
            )
            lambda_rows = lambda_rows.view(
                lambda_rows.size(0),
                lambda_rows.size(1),
                self.num_state_heads,
                self.gate_dim_per_head,
            )
            if self.gate_dim_per_head == 1:
                beta_rows = beta_rows.expand(-1, -1, -1, self.rank)
                lambda_rows = lambda_rows.expand(-1, -1, -1, self.rank)
        else:
            if beta_rows.size(-1) == 1:
                beta_rows = beta_rows.expand(beta_rows.size(0), beta_rows.size(1), self.rank)
            if lambda_rows.size(-1) == 1:
                lambda_rows = lambda_rows.expand(lambda_rows.size(0), lambda_rows.size(1), self.rank)

        if self.state_update_mode == "standard":
            keep_seq = lambda_rows
            erase_seq = beta_rows
            write_seq = beta_rows
        elif self.state_update_mode == "lambda_outside":
            keep_seq = lambda_rows
            erase_seq = lambda_rows * beta_rows
            write_seq = beta_rows
        elif self.state_update_mode == "no_lambda":
            keep_seq = torch.ones_like(beta_rows)
            erase_seq = beta_rows
            write_seq = beta_rows
        else:  # pragma: no cover
            raise ValueError(f"Unsupported state update mode: {self.state_update_mode}")

        if write_route_seq is None:
            return keep_seq, erase_seq, write_seq

        route = write_route_seq.permute(0, 2, 1).unsqueeze(-1)
        keep_seq = 1.0 - route + route * keep_seq.unsqueeze(1)
        erase_seq = route * erase_seq.unsqueeze(1)
        write_seq = route * write_seq.unsqueeze(1)
        return keep_seq, erase_seq, write_seq

    def _token_validity_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask[:, -seq_len:].to(device=device).ne(0)
        if attention_mask.dim() == 4:
            if attention_mask.size(0) != batch_size:
                raise ValueError(
                    "attention_mask batch dimension does not match hidden_states batch size"
                )
            if attention_mask.size(-2) < seq_len or attention_mask.size(-1) < seq_len:
                raise ValueError("attention_mask is shorter than the current sequence length")
            query_mask = attention_mask[:, 0, -seq_len:, -seq_len:]
            diagonal = query_mask.diagonal(dim1=-2, dim2=-1)
            if diagonal.dtype == torch.bool:
                return diagonal
            return diagonal.eq(0)
        raise ValueError(
            f"Unsupported attention_mask shape for Delta-Mem state updates: {tuple(attention_mask.shape)}"
        )

    def _masked_gate_mean(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_mask is None:
            return values.mean()
        expanded_mask = token_mask.unsqueeze(-1).unsqueeze(-1)
        masked_values = values * expanded_mask.to(dtype=values.dtype)
        denom = expanded_mask.sum().clamp_min(1).to(dtype=values.dtype)
        return masked_values.sum() / denom

    def _masked_hidden_norm(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        norms = torch.linalg.vector_norm(
            values.detach(),
            dim=-1,
            dtype=torch.float32,
        )
        if token_mask is None:
            return norms.mean()
        if not token_mask.any():
            return norms.new_zeros(())
        return norms.masked_select(token_mask).mean()

    def _masked_hidden_mean(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if token_mask is None:
            return values.mean(dim=1)
        weights = token_mask.unsqueeze(-1).to(dtype=values.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denom

    def _masked_ratio_mean(
        self,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        ratios = numerator.detach().float() / denominator.detach().float().clamp_min(eps)
        if token_mask is None:
            return ratios.mean()
        if not token_mask.any():
            return ratios.new_zeros(())
        return ratios.masked_select(token_mask).mean()

    def _masked_token_mean(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_values = values.detach().float().squeeze(-1)
        if token_mask is None:
            return token_values.mean()
        if not token_mask.any():
            return token_values.new_zeros(())
        return token_values.masked_select(token_mask).mean()

    def _masked_token_min_max(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_values = values.detach().float().squeeze(-1)
        if token_mask is not None:
            if not token_mask.any():
                zero = token_values.new_zeros(())
                return zero, zero
            token_values = token_values.masked_select(token_mask)
        return token_values.min(), token_values.max()

    def _masked_token_fraction(
        self,
        values: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        threshold: float,
        greater: bool,
    ) -> torch.Tensor:
        token_values = values.detach().float().squeeze(-1)
        if token_mask is not None:
            if not token_mask.any():
                return token_values.new_zeros(())
            token_values = token_values.masked_select(token_mask)
        selected = token_values > threshold if greater else token_values < threshold
        return selected.float().mean()

    def _masked_cosine_mean(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        left = left.detach()
        right = right.detach()
        dot = torch.sum(left * right, dim=-1, dtype=torch.float32)
        left_norm = torch.linalg.vector_norm(left, dim=-1, dtype=torch.float32)
        right_norm = torch.linalg.vector_norm(right, dim=-1, dtype=torch.float32)
        cosine = dot / (left_norm.clamp_min(1e-6) * right_norm.clamp_min(1e-6))
        if token_mask is None:
            return cosine.mean()
        if not token_mask.any():
            return cosine.new_zeros(())
        return cosine.masked_select(token_mask).mean()

    def _content_memory_fusion_gate(
        self,
        hidden_states: torch.Tensor,
        reads: torch.Tensor,
    ) -> torch.Tensor:
        normalized_hidden = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            eps=1e-6,
        )
        normalized_reads = F.rms_norm(
            reads.float(),
            (reads.shape[-1],),
            eps=1e-6,
        )
        logits = F.linear(normalized_hidden, self.memory_fusion_hidden_weight.float())
        logits = logits + F.linear(normalized_reads, self.memory_fusion_read_weight.float())
        logits = logits + self.memory_fusion_bias.float()
        return torch.sigmoid(logits)

    def _memory_fusion_gate(
        self,
        hidden_states: torch.Tensor,
        reads: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory_fusion_mode == "add":
            return reads.new_ones(*reads.shape[:-1], 1)
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                self._content_memory_fusion_gate,
                hidden_states,
                reads,
                use_reentrant=False,
            )
        return self._content_memory_fusion_gate(hidden_states, reads)

    def set_memory_fusion_residual_gain(self, gain: float) -> None:
        if self.memory_fusion_placement != "post_attention_residual_hybrid":
            raise RuntimeError(
                "A trainable residual gain is only available for "
                "memory_fusion_placement='post_attention_residual_hybrid'"
            )
        resolved_gain = float(gain)
        if not math.isfinite(resolved_gain) or not (
            0.0 <= resolved_gain <= self.memory_fusion_residual_scale_max
        ):
            raise ValueError(
                "Memory fusion residual gain must be finite and satisfy "
                "0 <= value <= memory_fusion_residual_scale_max"
            )
        with torch.no_grad():
            self.memory_fusion_residual_gain_raw.fill_(resolved_gain)

    def _resolved_memory_fusion_residual_gain(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.memory_fusion_placement != "post_attention_residual_hybrid":
            raise RuntimeError(
                "A trainable residual gain is only available for "
                "memory_fusion_placement='post_attention_residual_hybrid'"
            )
        raw_gain = self.memory_fusion_residual_gain_raw.float()
        bounded_gain = raw_gain.clamp(
            min=0.0,
            max=self.memory_fusion_residual_scale_max,
        )
        gain = bounded_gain.detach() + (raw_gain - raw_gain.detach())
        return gain.to(device=device, dtype=dtype)[0]

    def _apply_delta_o_rmsnorm(self, delta_o: torch.Tensor) -> torch.Tensor:
        if not self.delta_o_rmsnorm:
            return delta_o
        normalized = F.rms_norm(
            delta_o.float(),
            (delta_o.shape[-1],),
            weight=self.delta_o_rmsnorm_weight.float(),
            eps=self.delta_o_rmsnorm_eps,
        )
        return normalized.to(dtype=delta_o.dtype)

    def _delta_scale_multiplier(self, head_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.trainable_delta_scale:
            return torch.ones((), dtype=dtype, device=device)
        if self.delta_scale_granularity == "head":
            head_index = VALID_DELTA_HEADS.index(head_name)
            raw = self.delta_scale_raw[head_index]
        else:
            raw = self.delta_scale_raw[0]
        return (torch.sigmoid(raw) * self.delta_scale_max).to(device=device, dtype=dtype)

    def _project_delta_head(
        self,
        reads: torch.Tensor,
        weight: torch.Tensor,
        head_name: str,
    ) -> torch.Tensor | None:
        if head_name not in self.active_delta_heads:
            return None
        projected = F.linear(reads, weight)
        scale = self._delta_scale_multiplier(head_name, projected.dtype, projected.device)
        return projected * self.delta_scaling * scale

    def _resolve_read_context_mask(
        self,
        token_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.read_context_mask is None:
            return token_mask
        if self.read_context_mask.size(0) != batch_size or self.read_context_mask.size(1) != seq_len:
            return token_mask
        read_context_mask = self.read_context_mask.to(device=device, dtype=torch.bool)
        if token_mask is None:
            return read_context_mask
        return read_context_mask & token_mask.to(device=device, dtype=torch.bool)

    def _capture_read_representation(
        self,
        applied_correction: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> None:
        self.last_read_representation = None
        capture_mask = self.read_representation_capture_mask
        if capture_mask is None:
            return
        if applied_correction is None:
            raise RuntimeError(
                "Read representation capture requires an active delta_o head"
            )
        expected_shape = applied_correction.shape[:2]
        if capture_mask.ndim != 2 or tuple(capture_mask.shape) != expected_shape:
            raise ValueError(
                "Read representation capture mask must match the model token shape: "
                f"expected={expected_shape} actual={tuple(capture_mask.shape)}"
            )
        resolved_mask = capture_mask.to(device=applied_correction.device, dtype=torch.bool)
        selected_per_row = resolved_mask.sum(dim=1)
        if not torch.equal(selected_per_row, torch.ones_like(selected_per_row)):
            raise ValueError(
                "Read representation capture mask must select exactly one token per batch row"
            )
        if token_mask is not None:
            valid_mask = token_mask.to(device=applied_correction.device, dtype=torch.bool)
            if tuple(valid_mask.shape) != expected_shape:
                raise ValueError(
                    "Read representation validity mask must match the model token shape: "
                    f"expected={expected_shape} actual={tuple(valid_mask.shape)}"
                )
            if (resolved_mask & ~valid_mask).any():
                raise ValueError(
                    "Read representation capture mask selected a token outside the valid read mask"
                )
        self.last_read_representation = torch.einsum(
            "bt,bth->bh",
            resolved_mask.to(dtype=applied_correction.dtype),
            applied_correction,
        )

    def _record_applied_memory_correction(
        self,
        reference_output: torch.Tensor,
        applied_correction: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> None:
        if applied_correction is None:
            self._capture_read_representation(None, token_mask)
            return
        self.last_applied_memory_correction_norm = self._masked_hidden_norm(
            applied_correction,
            token_mask,
        )
        self.last_applied_memory_correction_ratio = self._masked_ratio_mean(
            applied_correction.norm(dim=-1),
            reference_output.norm(dim=-1),
            token_mask,
        )
        self._capture_read_representation(applied_correction, token_mask)

    def _global_partition_logit_bias(
        self,
        partition_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if (
            self.num_global_memory_partitions <= 0
            or self.global_memory_read_logit_bias == 0.0
            or partition_count < self.num_global_memory_partitions
        ):
            return None
        bias = torch.zeros(partition_count, device=device, dtype=dtype)
        bias[: self.num_global_memory_partitions] = self.global_memory_read_logit_bias
        return bias

    def _partition_query_scores(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        *,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_q_seq.ndim == 4:
            partition_queries = memory_q_seq.permute(0, 2, 1, 3)
        else:
            partition_queries = memory_q_seq.unsqueeze(2).expand(
                -1,
                -1,
                partition_reads.size(2),
                -1,
            )
        scores = (partition_reads * partition_queries).sum(dim=-1) / math.sqrt(float(self.rank))
        if partition_logit_bias is not None:
            scores = scores + partition_logit_bias.view(1, 1, -1)
        return scores

    def _mask_partition_top_k(
        self,
        scores: torch.Tensor,
        *,
        top_k: int,
    ) -> torch.Tensor:
        if 0 < top_k < scores.size(-1):
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
            masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
            masked_scores.scatter_(-1, top_indices, top_scores)
            return masked_scores
        return scores

    def _partition_query_softmax_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        top_k: int,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._partition_query_scores(
            partition_reads,
            memory_q_seq,
            partition_logit_bias=partition_logit_bias,
        )
        scores = self._mask_partition_top_k(scores, top_k=top_k)
        weights = F.softmax(scores, dim=-1)
        if token_mask is not None:
            weights = weights * token_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights

    def _partition_sigmoid_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        *,
        top_k: int,
        partition_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._partition_query_scores(
            partition_reads,
            memory_q_seq,
            partition_logit_bias=partition_logit_bias,
        )
        scores = self._mask_partition_top_k(scores, top_k=top_k)
        if hasattr(self, "partition_sigmoid_gate_bias"):
            scores = scores + self.partition_sigmoid_gate_bias.view(1, 1, -1)
        weights = torch.sigmoid(scores)
        if token_mask is not None:
            weights = weights * token_mask.unsqueeze(-1).to(dtype=weights.dtype)
        return weights

    def _slot_query_softmax_weights(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._partition_query_softmax_weights(
            partition_reads,
            memory_q_seq,
            token_mask,
            top_k=self.slot_read_top_k,
            partition_logit_bias=self._global_partition_logit_bias(
                partition_reads.size(2),
                device=partition_reads.device,
                dtype=partition_reads.dtype,
            ),
        )

    def _split_global_partition_queries(
        self,
        memory_q_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if memory_q_seq.ndim == 4:
            return (
                memory_q_seq[:, : self.num_global_memory_partitions],
                memory_q_seq[:, self.num_global_memory_partitions :],
            )
        return memory_q_seq, memory_q_seq

    def _merge_split_partition_reads(
        self,
        local_reads: torch.Tensor,
        global_reads: torch.Tensor,
        local_routes: torch.Tensor,
        global_routes: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = local_reads.new_ones(*local_reads.shape[:-1], 1)
        if self.global_memory_merge_mode == "gated_residual":
            gate_input = torch.cat([local_reads, global_reads], dim=-1)
            gate = torch.sigmoid(self.global_memory_gate_proj(gate_input))
            if token_mask is not None:
                gate = gate * token_mask.unsqueeze(-1).to(dtype=gate.dtype)
            reads = local_reads + gate * global_reads
        else:
            reads = local_reads + global_reads
        effective_routes = torch.cat([global_routes * gate, local_routes], dim=-1)
        denom = effective_routes.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        effective_routes = effective_routes / denom
        return reads, effective_routes

    def _aggregate_split_partition_reads(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.global_memory_merge_mode == "joint_softmax":
            read_routes = self._partition_query_softmax_weights(
                partition_reads,
                memory_q_seq,
                token_mask,
                top_k=self.slot_read_top_k,
                partition_logit_bias=self._global_partition_logit_bias(
                    partition_reads.size(2),
                    device=partition_reads.device,
                    dtype=partition_reads.dtype,
                ),
            )
            self.last_read_routes = read_routes
            return torch.einsum("btp,btpi->bti", read_routes, partition_reads)

        global_partition_reads = partition_reads[:, :, : self.num_global_memory_partitions, :]
        local_partition_reads = partition_reads[:, :, self.num_global_memory_partitions :, :]
        global_memory_q_seq, local_memory_q_seq = self._split_global_partition_queries(memory_q_seq)
        global_routes = self._partition_query_softmax_weights(
            global_partition_reads,
            global_memory_q_seq,
            token_mask,
            top_k=self.global_memory_read_top_k,
        )
        local_routes = self._partition_query_softmax_weights(
            local_partition_reads,
            local_memory_q_seq,
            token_mask,
            top_k=self.slot_read_top_k,
        )
        global_reads = torch.einsum("btp,btpi->bti", global_routes, global_partition_reads)
        local_reads = torch.einsum("btp,btpi->bti", local_routes, local_partition_reads)
        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).to(dtype=global_reads.dtype)
            global_reads = global_reads * mask
            local_reads = local_reads * mask
        reads, effective_routes = self._merge_split_partition_reads(
            local_reads,
            global_reads,
            local_routes,
            global_routes,
            token_mask,
        )
        self.last_read_routes = effective_routes
        return reads

    def _aggregate_partition_reads(
        self,
        partition_reads: torch.Tensor,
        memory_q_seq: torch.Tensor,
        read_route_seq: torch.Tensor | None,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if read_route_seq is None:
            read_route_seq = self._slot_query_softmax_weights(
                partition_reads,
                memory_q_seq,
                token_mask,
            )
        self.last_read_routes = read_route_seq
        return torch.einsum("btp,btpi->bti", read_route_seq, partition_reads)

    def _token_state_reads(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        read_route_seq: torch.Tensor | None,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.multi_head_state:
            head_q = memory_q_seq.view(
                memory_q_seq.size(0),
                memory_q_seq.size(1),
                self.num_state_heads,
                self.rank,
            )
            reads = torch.einsum("bhij,bthj->bthi", state, head_q)
            reads = reads.reshape(memory_q_seq.size(0), memory_q_seq.size(1), self.state_read_dim)
        else:
            reads = torch.einsum("bij,btj->bti", state, memory_q_seq)
        if token_mask is not None:
            reads = reads * token_mask.unsqueeze(-1).to(dtype=reads.dtype)
        return reads

    def _message_write_inputs(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.write_enabled or self.write_message_ids is None:
            return None
        message_ids = self.write_message_ids
        if message_ids.dim() != 2:
            return None
        if message_ids.size(0) != hidden_states.size(0) or message_ids.size(1) != hidden_states.size(1):
            return None
        message_ids = message_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None
        return message_ids, active_mask

    def _build_message_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        message_inputs = self._message_write_inputs(hidden_states, token_mask)
        if not self.write_enabled or self.memory_write_granularity != "message_mean" or message_inputs is None:
            return None, None, None
        message_ids, active_mask = message_inputs
        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages_max = max_message_id + 1
        message_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_messages_max,
            hidden_states.size(-1),
        )
        message_mask = torch.zeros(
            hidden_states.size(0),
            num_messages_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        summary_message_ids = torch.full(
            (hidden_states.size(0), num_messages_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            for message_id in sample_message_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_message_id = int(message_id)
                token_selector = sample_active_mask & sample_message_ids.eq(current_message_id)
                message_hidden[batch_idx, current_message_id] = hidden_states[batch_idx, token_selector].mean(dim=0)
                message_mask[batch_idx, current_message_id] = True
                summary_message_ids[batch_idx, current_message_id] = current_message_id
        return message_hidden, message_mask, summary_message_ids

    def _sentence_write_inputs(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.write_enabled or self.write_message_ids is None or self.write_sentence_ids is None:
            return None
        message_ids = self.write_message_ids
        sentence_ids = self.write_sentence_ids
        if message_ids.dim() != 2 or sentence_ids.dim() != 2:
            return None
        if (
            message_ids.size(0) != hidden_states.size(0)
            or message_ids.size(1) != hidden_states.size(1)
            or sentence_ids.size(0) != hidden_states.size(0)
            or sentence_ids.size(1) != hidden_states.size(1)
        ):
            return None
        message_ids = message_ids.to(device=hidden_states.device)
        sentence_ids = sentence_ids.to(device=hidden_states.device)
        active_mask = message_ids.ge(0) & sentence_ids.ge(0)
        if token_mask is not None:
            active_mask = active_mask & token_mask
        if not active_mask.any():
            return None
        return message_ids, sentence_ids, active_mask

    def _build_sentence_write_means(
        self,
        hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        sentence_inputs = self._sentence_write_inputs(hidden_states, token_mask)
        if not self.write_enabled or self.memory_write_granularity != "sentence_mean" or sentence_inputs is None:
            return None, None, None
        message_ids, sentence_ids, active_mask = sentence_inputs
        max_sentence_id = int(sentence_ids.masked_select(active_mask).max().item())
        num_sentences_max = max_sentence_id + 1
        sentence_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_sentences_max,
            hidden_states.size(-1),
        )
        sentence_mask = torch.zeros(
            hidden_states.size(0),
            num_sentences_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        sentence_message_ids = torch.full(
            (hidden_states.size(0), num_sentences_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for batch_idx in range(hidden_states.size(0)):
            sample_sentence_ids = sentence_ids[batch_idx]
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            for sentence_id in sample_sentence_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_sentence_id = int(sentence_id)
                token_selector = sample_active_mask & sample_sentence_ids.eq(current_sentence_id)
                sentence_hidden[batch_idx, current_sentence_id] = hidden_states[batch_idx, token_selector].mean(dim=0)
                sentence_mask[batch_idx, current_sentence_id] = True
                sentence_message_ids[batch_idx, current_sentence_id] = int(
                    sample_message_ids.masked_select(token_selector)[0].item()
                )
        return sentence_hidden, sentence_mask, sentence_message_ids

    def _build_message_write_proposals(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        token_k_seq: torch.Tensor,
        token_v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        message_inputs = self._message_write_inputs(hidden_states, token_mask)
        if (
            not self.write_enabled
            or self.memory_write_granularity != "message_proposals"
            or message_inputs is None
        ):
            return None, None, None

        message_ids, active_mask = message_inputs
        state_summary = self.message_proposal_state_proj(
            state.reshape(hidden_states.size(0), -1).float().to(hidden_states.dtype)
        )
        token_features = torch.tanh(self.message_proposal_token_proj(hidden_states))
        novelty = hidden_states.new_zeros(hidden_states.size(0), hidden_states.size(1))
        if state.ndim == 3 and token_k_seq.ndim == 3 and token_v_seq.ndim == 3:
            predicted_values = torch.einsum("bij,btj->bti", state, token_k_seq)
            novelty = (token_v_seq - predicted_values).float().norm(dim=-1).to(hidden_states.dtype)
            novelty = novelty * active_mask.to(dtype=novelty.dtype)

        max_message_id = int(message_ids.masked_select(active_mask).max().item())
        num_messages_max = max_message_id + 1
        num_proposals_max = num_messages_max * self.memory_write_proposals_per_message
        proposal_hidden = hidden_states.new_zeros(
            hidden_states.size(0),
            num_proposals_max,
            hidden_states.size(-1),
        )
        proposal_mask = torch.zeros(
            hidden_states.size(0),
            num_proposals_max,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        proposal_message_ids = torch.full(
            (hidden_states.size(0), num_proposals_max),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )

        for batch_idx in range(hidden_states.size(0)):
            sample_message_ids = message_ids[batch_idx]
            sample_active_mask = active_mask[batch_idx]
            if not sample_active_mask.any():
                continue
            sample_state_summary = state_summary[batch_idx]
            proposal_index = 0
            for message_id in sample_message_ids.masked_select(sample_active_mask).unique(sorted=True).tolist():
                current_message_id = int(message_id)
                message_mask = sample_active_mask & sample_message_ids.eq(current_message_id)
                message_hidden_slice = hidden_states[batch_idx, message_mask]
                message_features = token_features[batch_idx, message_mask]
                message_novelty = novelty[batch_idx, message_mask]
                message_summary = message_hidden_slice.mean(dim=0)
                query_base = sample_state_summary + self.message_proposal_message_proj(message_summary)
                coverage = message_novelty.new_zeros(message_hidden_slice.size(0))
                for slot_idx in range(self.memory_write_proposals_per_message):
                    slot_query = query_base + self.message_proposal_slot_queries[slot_idx].to(
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    logits = torch.matmul(message_features, slot_query)
                    logits = logits + self.message_proposal_novelty_scale.to(hidden_states.dtype) * message_novelty
                    if slot_idx > 0:
                        logits = logits - self.message_proposal_coverage_scale.to(hidden_states.dtype) * coverage
                    attention = torch.softmax(logits, dim=0)
                    proposal_hidden[batch_idx, proposal_index] = torch.matmul(attention, message_hidden_slice)
                    proposal_mask[batch_idx, proposal_index] = True
                    proposal_message_ids[batch_idx, proposal_index] = current_message_id
                    coverage = coverage + attention
                    proposal_index += 1
        return proposal_hidden, proposal_mask, proposal_message_ids

    def _memory_affine_scan_torch(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        keep_seq: torch.Tensor,
        erase_seq: torch.Tensor,
        write_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = memory_q_seq.shape
        current_state = state
        read_steps: list[torch.Tensor] = []

        for token_idx in range(seq_len):
            q_t = memory_q_seq[:, token_idx, :]
            k_t = memory_k_seq[:, token_idx, :]
            v_t = memory_v_seq[:, token_idx, :]
            keep_t = keep_seq[:, token_idx, :].unsqueeze(-1)
            erase_t = erase_seq[:, token_idx, :].unsqueeze(-1)
            write_t = write_seq[:, token_idx, :].unsqueeze(-1)

            read_t = torch.einsum("bij,bj->bi", current_state, q_t)

            if token_mask is not None:
                valid = token_mask[:, token_idx].view(batch_size, 1)
                read_t = read_t * valid.to(dtype=read_t.dtype)

            pred_t = torch.einsum("bij,bj->bi", current_state, k_t)
            write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
            pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
            next_state = keep_t * current_state - erase_t * pred_outer + write_t * write_outer

            if token_mask is not None:
                valid_state = token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=next_state.dtype)
                current_state = next_state * valid_state + current_state * (1.0 - valid_state)
            else:
                current_state = next_state

            read_steps.append(read_t)

        reads = torch.stack(read_steps, dim=1)
        return current_state, reads

    def _memory_affine_scan(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        write_route_seq: torch.Tensor | None = None,
        read_route_seq: torch.Tensor | None = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep_seq, erase_seq, write_seq = self._memory_update_coefficients(
            beta_seq,
            lambda_seq,
            write_route_seq=write_route_seq,
        )
        if self.multi_head_state:
            batch_size, seq_len, _ = memory_q_seq.shape
            q_for_scan = memory_q_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            k_for_scan = memory_k_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            v_for_scan = memory_v_seq.view(batch_size, seq_len, self.num_state_heads, self.rank)
            state_for_scan = state.reshape(batch_size * self.num_state_heads, self.rank, self.rank)
            q_for_scan = q_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            k_for_scan = k_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            v_for_scan = v_for_scan.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            keep_for_scan = keep_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            erase_for_scan = erase_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            write_for_scan = write_seq.permute(0, 2, 1, 3).reshape(batch_size * self.num_state_heads, seq_len, self.rank)
            token_mask_for_scan = None
            if token_mask is not None:
                token_mask_for_scan = (
                    token_mask.unsqueeze(1)
                    .expand(batch_size, self.num_state_heads, seq_len)
                    .reshape(batch_size * self.num_state_heads, seq_len)
                )
        else:
            single_partition = state.ndim == 3
            if single_partition:
                state_for_scan = state
                q_for_scan = memory_q_seq
                k_for_scan = memory_k_seq
                v_for_scan = memory_v_seq
                keep_for_scan = keep_seq
                erase_for_scan = erase_seq
                write_for_scan = write_seq
                token_mask_for_scan = token_mask
            else:
                batch_size, num_partitions, rank, _ = state.shape
                seq_len = memory_q_seq.size(-2)
                state_for_scan = state.reshape(batch_size * num_partitions, rank, rank)
                if memory_q_seq.ndim == 4:
                    q_for_scan = memory_q_seq.reshape(batch_size * num_partitions, seq_len, rank)
                    k_for_scan = memory_k_seq.reshape(batch_size * num_partitions, seq_len, rank)
                    v_for_scan = memory_v_seq.reshape(batch_size * num_partitions, seq_len, rank)
                else:
                    q_for_scan = (
                        memory_q_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                    k_for_scan = (
                        memory_k_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                    v_for_scan = (
                        memory_v_seq.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len, rank)
                        .reshape(batch_size * num_partitions, seq_len, rank)
                    )
                keep_for_scan = keep_seq.reshape(batch_size * num_partitions, seq_len, rank)
                erase_for_scan = erase_seq.reshape(batch_size * num_partitions, seq_len, rank)
                write_for_scan = write_seq.reshape(batch_size * num_partitions, seq_len, rank)
                token_mask_for_scan = None
                if token_mask is not None:
                    token_mask_for_scan = (
                        token_mask.unsqueeze(1)
                        .expand(batch_size, num_partitions, seq_len)
                        .reshape(batch_size * num_partitions, seq_len)
                    )

        use_triton = self.scan_impl != "torch"
        if use_triton:
            support = triton_scan_support(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
            )
            use_triton = support.supported and self.scan_impl in {"auto", "triton"}
            if self.scan_impl == "triton" and not support.supported:
                raise RuntimeError(f"Triton scan requested but unavailable: {support.reason}")
        if use_triton:
            final_state, reads = triton_affine_scan(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
                token_mask=token_mask_for_scan,
            )
        else:
            final_state, reads = self._memory_affine_scan_torch(
                state_for_scan,
                q_for_scan,
                k_for_scan,
                v_for_scan,
                keep_for_scan,
                erase_for_scan,
                write_for_scan,
                token_mask=token_mask_for_scan,
            )

        if self.multi_head_state:
            batch_size, seq_len, _ = memory_q_seq.shape
            final_state = final_state.reshape(batch_size, self.num_state_heads, self.rank, self.rank)
            reads = reads.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
            reads = reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim)
            return final_state, reads

        if state.ndim == 3:
            return final_state, reads

        batch_size, num_partitions, rank, _ = state.shape
        seq_len = memory_q_seq.size(-2)
        final_state = final_state.reshape(batch_size, num_partitions, rank, rank)
        partition_reads = reads.reshape(batch_size, num_partitions, seq_len, rank).permute(0, 2, 1, 3)
        aggregated_reads = self._aggregate_partition_reads(
            partition_reads,
            memory_q_seq,
            read_route_seq,
            token_mask,
        )
        if token_mask is not None:
            aggregated_reads = aggregated_reads * token_mask.unsqueeze(-1).to(dtype=aggregated_reads.dtype)
        return final_state, aggregated_reads

    def _ensure_rwkv_ms_positions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            self.rwkv_ms_positions is None
            or self.rwkv_ms_positions.size(0) != batch_size
            or self.rwkv_ms_positions.device != device
        ):
            self.rwkv_ms_positions = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )
        return self.rwkv_ms_positions

    def _ensure_rwkv_ms_previous_source(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_shape = (batch_size, self.state_read_dim)
        if (
            self.rwkv_ms_previous_source is None
            or self.rwkv_ms_previous_source.shape != expected_shape
            or self.rwkv_ms_previous_source.device != device
        ):
            self.rwkv_ms_previous_source = torch.zeros(
                expected_shape,
                device=device,
                dtype=dtype,
            )
        elif self.rwkv_ms_previous_source.dtype != dtype:
            self.rwkv_ms_previous_source = self.rwkv_ms_previous_source.to(dtype=dtype)
        return self.rwkv_ms_previous_source

    def _rwkv_ms_project_heads(self, projected: torch.Tensor) -> torch.Tensor:
        return projected.view(
            projected.size(0),
            projected.size(1),
            self.num_state_heads,
            self.rank,
        )

    def _rwkv_ms_update_coefficients(
        self,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_seq, erase_seq, write_seq = self._memory_update_coefficients(beta_seq, lambda_seq)
        if self.rwkv_ms_semantics_version >= 2:
            # RWKV's learned decay already controls carry. Beta gates the
            # complete write/correction pair without shortening that memory.
            keep_seq = torch.ones_like(keep_seq)
            erase_seq = write_seq
        if self.num_state_heads == 1:
            keep_seq = keep_seq.unsqueeze(2)
            erase_seq = erase_seq.unsqueeze(2)
            write_seq = write_seq.unsqueeze(2)
        return keep_seq, erase_seq, write_seq

    def _rwkv_ms_slot_indices(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.rwkv_ms_boundary_mode != "fixed_chunk":  # pragma: no cover
            raise ValueError(f"Unsupported RWKV-MS boundary mode: {self.rwkv_ms_boundary_mode}")
        return torch.div(
            positions,
            self.rwkv_ms_chunk_size,
            rounding_mode="floor",
        ).remainder(self.rwkv_ms_num_states)

    def _rwkv_ms_read_routes(
        self,
        slot_reads: torch.Tensor,
        q_t: torch.Tensor,
        valid_t: torch.Tensor | None,
        occupied_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rwkv_ms_semantics_version >= 2:
            scores = F.cosine_similarity(
                slot_reads.float(),
                q_t.float().unsqueeze(2),
                dim=-1,
                eps=1e-6,
            )
        else:
            scores = (slot_reads * q_t.unsqueeze(2)).sum(dim=-1) / math.sqrt(float(self.rank))
        if self.rwkv_ms_mask_empty_slots:
            if occupied_slots is None:
                occupied_slots = slot_reads.ne(0).any(dim=-1)
            all_slots_empty = ~occupied_slots.any(dim=-1, keepdim=True)
            routable_slots = occupied_slots | all_slots_empty
            scores = scores.masked_fill(~routable_slots, torch.finfo(scores.dtype).min)
        if 0 < self.rwkv_ms_read_top_k < scores.size(-1):
            top_scores, top_indices = torch.topk(scores, k=self.rwkv_ms_read_top_k, dim=-1)
            masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
            scores = masked_scores.scatter_(-1, top_indices, top_scores)
        routes = F.softmax(scores, dim=-1)
        if valid_t is not None:
            routes = routes * valid_t.view(valid_t.size(0), 1, 1).to(dtype=routes.dtype)
        return routes

    def _rwkv_ms_last_token_overwrite(
        self,
        state: torch.Tensor,
        r_seq: torch.Tensor,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        g_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        next_previous_source: torch.Tensor,
        *,
        update_positions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _, _ = r_seq.shape
        current_state = state.float()
        positions = self._ensure_rwkv_ms_positions(
            batch_size,
            r_seq.device,
        ).clone()
        if token_mask is None:
            valid_tokens = torch.ones(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=r_seq.device,
            )
        else:
            valid_tokens = token_mask.to(device=r_seq.device, dtype=torch.bool)

        token_indices = torch.arange(seq_len, device=r_seq.device).unsqueeze(0)
        last_valid_indices = torch.where(valid_tokens, token_indices, -1).amax(dim=1)
        has_valid_token = last_valid_indices.ge(0)
        gather_indices = last_valid_indices.clamp_min(0).view(batch_size, 1, 1, 1)
        gather_indices = gather_indices.expand(-1, 1, self.num_state_heads, self.rank)
        last_k = k_seq.gather(1, gather_indices).squeeze(1)
        last_v = v_seq.gather(1, gather_indices).squeeze(1)

        valid_counts = valid_tokens.sum(dim=1, dtype=torch.long)
        last_positions = positions + valid_counts.clamp_min(1) - 1
        slot_indices = self._rwkv_ms_slot_indices(last_positions)
        selected_slots = F.one_hot(
            slot_indices,
            num_classes=self.rwkv_ms_num_states,
        ).to(dtype=current_state.dtype)
        selected_slots = selected_slots * has_valid_token.unsqueeze(-1).to(
            dtype=current_state.dtype
        )
        write_outer = last_v.unsqueeze(2).unsqueeze(-1) * last_k.unsqueeze(2).unsqueeze(-2)
        state_mask = selected_slots.view(
            batch_size,
            1,
            self.rwkv_ms_num_states,
            1,
            1,
        )
        next_state = write_outer * state_mask + current_state * (1.0 - state_mask)

        occupied_slots = current_state.ne(0).any(dim=(-1, -2))
        read_steps: list[torch.Tensor] = []
        read_routes: list[torch.Tensor] = []
        write_routes: list[torch.Tensor] = []
        for token_idx in range(seq_len):
            r_t = r_seq[:, token_idx]
            valid_t = None if token_mask is None else valid_tokens[:, token_idx]
            slot_reads = torch.einsum("bhsij,bhj->bhsi", current_state, r_t)
            routes = self._rwkv_ms_read_routes(
                slot_reads,
                r_t,
                valid_t,
                occupied_slots=occupied_slots,
            )
            read_t = torch.einsum("bhs,bhsi->bhi", routes, slot_reads)
            read_steps.append(read_t.reshape(batch_size, self.state_read_dim))
            read_routes.append(routes.mean(dim=1))
            is_selected_token = has_valid_token & last_valid_indices.eq(token_idx)
            write_routes.append(
                selected_slots
                * is_selected_token.unsqueeze(-1).to(dtype=selected_slots.dtype)
            )

        reads = torch.stack(read_steps, dim=1).to(dtype=g_seq.dtype)
        reads = self.hrm_rwkv7_core.readout(reads, g_seq)
        self.last_read_routes = torch.stack(read_routes, dim=1)
        self.last_write_routes = torch.stack(write_routes, dim=1)
        if update_positions:
            self.rwkv_ms_positions = (positions + valid_counts).detach()
            self.rwkv_ms_previous_source = next_previous_source
        return next_state, reads

    def _rwkv_ms_scan(
        self,
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        *,
        update_positions: bool = True,
        write_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = memory_source_seq.shape
        if seq_len == 0:
            self.last_read_routes = memory_source_seq.new_zeros(
                batch_size,
                0,
                self.rwkv_ms_num_states,
            )
            self.last_write_routes = memory_source_seq.new_zeros(
                batch_size,
                0,
                self.rwkv_ms_num_states,
            )
            return state.float(), memory_source_seq.new_zeros(batch_size, 0, self.state_read_dim)
        if self.hrm_rwkv7_core is None:  # pragma: no cover
            raise RuntimeError("RWKV-MS backend requires HRM RWKV-7 core")
        previous_source = self._ensure_rwkv_ms_previous_source(
            batch_size,
            memory_source_seq.device,
            memory_source_seq.dtype,
        )
        features, next_previous_source = self.hrm_rwkv7_core.project(
            memory_source_seq,
            previous_x=previous_source,
            token_mask=token_mask,
            return_previous=True,
        )
        r_seq = self._rwkv_ms_project_heads(features.r).float()
        k_seq = self._rwkv_ms_project_heads(features.k).float()
        v_seq = self._rwkv_ms_project_heads(features.v).float()
        if self.rwkv_ms_write_mode == "last_token_overwrite":
            return self._rwkv_ms_last_token_overwrite(
                state,
                r_seq,
                k_seq,
                v_seq,
                features.g,
                token_mask,
                next_previous_source,
                update_positions=update_positions,
            )
        w_seq = self._rwkv_ms_project_heads(features.w).float()
        a_seq = self._rwkv_ms_project_heads(features.a).float()
        b_seq = self._rwkv_ms_project_heads(features.b).float()
        keep_seq, erase_seq, write_seq = self._rwkv_ms_update_coefficients(beta_seq, lambda_seq)
        keep_seq = keep_seq.float()
        erase_seq = erase_seq.float()
        write_seq = write_seq.float()

        current_state = state.float()
        positions = self._ensure_rwkv_ms_positions(batch_size, memory_source_seq.device).clone()
        if write_only:
            slots = rwkv_ms_write_slot_indices(
                token_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                positions=positions,
                chunk_size=self.rwkv_ms_chunk_size,
                num_slots=self.rwkv_ms_num_states,
            )
            current_state = rwkv_ms_write_scan(
                current_state,
                torch.exp(-torch.exp(w_seq)),
                k_seq,
                v_seq,
                a_seq,
                b_seq,
                keep_seq,
                erase_seq,
                write_seq,
                slots,
                self.rwkv_ms_erase_gate,
            )
            valid = slots.ge(0)
            self.last_write_routes = F.one_hot(
                slots.clamp_min(0),
                num_classes=self.rwkv_ms_num_states,
            ).to(dtype=current_state.dtype)
            self.last_write_routes = self.last_write_routes * valid.unsqueeze(-1).to(
                dtype=current_state.dtype
            )
            self.last_read_routes = current_state.new_zeros(
                batch_size,
                seq_len,
                self.rwkv_ms_num_states,
            )
            if update_positions:
                self.rwkv_ms_positions = (
                    positions + valid.sum(dim=1, dtype=torch.long)
                ).detach()
                self.rwkv_ms_previous_source = next_previous_source
            return current_state, memory_source_seq.new_zeros(
                batch_size,
                seq_len,
                self.state_read_dim,
            )
        read_steps: list[torch.Tensor] = []
        read_routes: list[torch.Tensor] = []
        write_routes: list[torch.Tensor] = []

        for token_idx in range(seq_len):
            r_t = r_seq[:, token_idx]
            w_t = torch.exp(-torch.exp(w_seq[:, token_idx]))
            k_t = k_seq[:, token_idx]
            v_t = v_seq[:, token_idx]
            a_t = a_seq[:, token_idx]
            b_t = b_seq[:, token_idx]
            keep_t = keep_seq[:, token_idx]
            erase_t = erase_seq[:, token_idx]
            write_t = write_seq[:, token_idx]
            valid_t = None if token_mask is None else token_mask[:, token_idx]

            slot_reads = torch.einsum("bhsij,bhj->bhsi", current_state, r_t)
            occupied_slots = current_state.ne(0).any(dim=(-1, -2))
            routes = self._rwkv_ms_read_routes(
                slot_reads,
                r_t,
                valid_t,
                occupied_slots=occupied_slots,
            )
            read_t = torch.einsum("bhs,bhsi->bhi", routes, slot_reads)
            read_steps.append(read_t.reshape(batch_size, self.state_read_dim))
            read_routes.append(routes.mean(dim=1))

            slot_idx = self._rwkv_ms_slot_indices(positions)
            slot_mask = F.one_hot(slot_idx, num_classes=self.rwkv_ms_num_states).to(
                dtype=current_state.dtype,
            )
            if valid_t is not None:
                slot_mask = slot_mask * valid_t.to(dtype=current_state.dtype).unsqueeze(-1)
            write_routes.append(slot_mask)

            correction_read = torch.einsum("bhsij,bhj->bhsi", current_state, a_t)
            write_outer = v_t.unsqueeze(2).unsqueeze(-1) * k_t.unsqueeze(2).unsqueeze(-2)
            correction_outer = correction_read.unsqueeze(-1) * b_t.unsqueeze(2).unsqueeze(-2)
            candidate_state = (
                keep_t.unsqueeze(2).unsqueeze(-1)
                * w_t.unsqueeze(2).unsqueeze(-2)
                * current_state
                + write_t.unsqueeze(2).unsqueeze(-1) * write_outer
                + self.rwkv_ms_erase_gate
                * erase_t.unsqueeze(2).unsqueeze(-1)
                * correction_outer
            )
            state_mask = slot_mask.view(batch_size, 1, self.rwkv_ms_num_states, 1, 1)
            current_state = candidate_state * state_mask + current_state * (1.0 - state_mask)
            if valid_t is None:
                positions = positions + 1
            else:
                positions = positions + valid_t.to(dtype=torch.long)

        reads = torch.stack(read_steps, dim=1).to(dtype=features.g.dtype)
        reads = self.hrm_rwkv7_core.readout(reads, features.g)
        self.last_read_routes = torch.stack(read_routes, dim=1)
        self.last_write_routes = torch.stack(write_routes, dim=1)
        if update_positions:
            self.rwkv_ms_positions = positions.detach()
            # Keep the write-to-read time-mix path live inside an episode. Online
            # state export remains the persistence boundary that detaches tensors.
            self.rwkv_ms_previous_source = next_previous_source
        return current_state, reads

    def _rwkv_ms_token_state_reads(
        self,
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = memory_source_seq.shape
        if seq_len == 0:
            self.last_read_routes = memory_source_seq.new_zeros(
                batch_size,
                0,
                self.rwkv_ms_num_states,
            )
            return memory_source_seq.new_zeros(batch_size, 0, self.state_read_dim)
        if self.hrm_rwkv7_core is None:  # pragma: no cover
            raise RuntimeError("RWKV-MS backend requires HRM RWKV-7 core")
        previous_source = self.rwkv_ms_previous_source
        if previous_source is not None:
            expected_shape = (batch_size, self.state_read_dim)
            if previous_source.shape != expected_shape:
                previous_source = None
            else:
                previous_source = previous_source.to(
                    device=memory_source_seq.device,
                    dtype=memory_source_seq.dtype,
                )
        features = self.hrm_rwkv7_core.project(
            memory_source_seq,
            previous_x=previous_source,
            token_mask=token_mask,
            advance_within_sequence=False,
        )
        current_state = state.float()
        r_seq = self._rwkv_ms_project_heads(features.r).float()
        slot_reads = torch.einsum(
            "bhsij,bthj->bthsi",
            current_state,
            r_seq,
        )
        if self.rwkv_ms_semantics_version >= 2:
            scores = F.cosine_similarity(
                slot_reads.float(),
                r_seq.float().unsqueeze(3),
                dim=-1,
                eps=1e-6,
            )
        else:
            scores = (
                slot_reads * r_seq.unsqueeze(3)
            ).sum(dim=-1) / math.sqrt(float(self.rank))
        if self.rwkv_ms_mask_empty_slots:
            occupied_slots = current_state.ne(0).any(dim=(-1, -2))
            all_slots_empty = ~occupied_slots.any(dim=-1, keepdim=True)
            routable_slots = occupied_slots | all_slots_empty
            scores = scores.masked_fill(
                ~routable_slots.unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
        if 0 < self.rwkv_ms_read_top_k < scores.size(-1):
            top_scores, top_indices = torch.topk(
                scores,
                k=self.rwkv_ms_read_top_k,
                dim=-1,
            )
            masked_scores = torch.full_like(
                scores,
                torch.finfo(scores.dtype).min,
            )
            scores = masked_scores.scatter_(-1, top_indices, top_scores)
        routes = F.softmax(scores, dim=-1)
        if token_mask is not None:
            routes = routes * token_mask.to(
                device=routes.device,
                dtype=routes.dtype,
            ).view(batch_size, seq_len, 1, 1)
        read_inputs = torch.einsum(
            "bths,bthsi->bthi",
            routes,
            slot_reads,
        ).reshape(batch_size, seq_len, self.state_read_dim)
        self.last_read_routes = routes.mean(dim=2)
        read_inputs = read_inputs.to(dtype=features.g.dtype)
        return self.hrm_rwkv7_core.readout(read_inputs, features.g)

    def _rwkv_ms_addressed_token_state_reads(
        self,
        state: torch.Tensor,
        memory_source_seq: torch.Tensor,
        projected_routes: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = memory_source_seq.shape
        expected_routes_shape = (batch_size, seq_len, self.rwkv_ms_num_states)
        if tuple(projected_routes.shape) != expected_routes_shape:
            raise ValueError(
                "Projected routes must match RWKV-MS addressed read shape: "
                f"expected={expected_routes_shape} "
                f"actual={tuple(projected_routes.shape)}"
            )
        if self.hrm_rwkv7_core is None:  # pragma: no cover
            raise RuntimeError("RWKV-MS backend requires HRM RWKV-7 core")
        previous_source = self.rwkv_ms_previous_source
        if previous_source is not None:
            expected_previous_shape = (batch_size, self.state_read_dim)
            if previous_source.shape != expected_previous_shape:
                previous_source = None
            else:
                previous_source = previous_source.to(
                    device=memory_source_seq.device,
                    dtype=memory_source_seq.dtype,
                )
        features = self.hrm_rwkv7_core.project(
            memory_source_seq,
            previous_x=previous_source,
            token_mask=token_mask,
            advance_within_sequence=False,
        )
        r_seq = self._rwkv_ms_project_heads(features.r).float()
        slot_reads = torch.einsum(
            "bhsij,bthj->bthsi",
            state.float(),
            r_seq,
        )
        routes = projected_routes.to(
            device=slot_reads.device,
            dtype=slot_reads.dtype,
        )
        if token_mask is not None:
            routes = routes * token_mask.to(
                device=routes.device,
                dtype=routes.dtype,
            ).unsqueeze(-1)
        read_inputs = torch.einsum(
            "bts,bthsi->bthi",
            routes,
            slot_reads,
        ).reshape(batch_size, seq_len, self.state_read_dim)
        self.last_read_routes = routes
        return self.hrm_rwkv7_core.readout(
            read_inputs.to(dtype=features.g.dtype),
            features.g,
        )

    def _memory_backend_scan(
        self,
        state: torch.Tensor,
        memory_q_seq: torch.Tensor,
        memory_k_seq: torch.Tensor,
        memory_v_seq: torch.Tensor,
        beta_seq: torch.Tensor,
        lambda_seq: torch.Tensor,
        write_route_seq: torch.Tensor | None = None,
        read_route_seq: torch.Tensor | None = None,
        token_mask: Optional[torch.Tensor] = None,
        write_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_backend == "rwkv_ms":
            return self._rwkv_ms_scan(
                state,
                memory_v_seq,
                beta_seq,
                lambda_seq,
                token_mask,
                write_only=write_only,
            )
        if write_only:  # pragma: no cover - hybrid validation requires RWKV-MS
            raise ValueError("Write-only backend scan requires RWKV-MS")
        return self._memory_affine_scan(
            state,
            memory_q_seq,
            memory_k_seq,
            memory_v_seq,
            beta_seq,
            lambda_seq,
            write_route_seq=write_route_seq,
            read_route_seq=read_route_seq,
            token_mask=token_mask,
        )

    def _memory_backend_token_reads(
        self,
        state: torch.Tensor,
        memory_read_seq: torch.Tensor,
        read_route_seq: torch.Tensor | None,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.memory_backend == "rwkv_ms":
            return self._rwkv_ms_token_state_reads(state, memory_read_seq, token_mask)
        return self._token_state_reads(
            state,
            memory_read_seq,
            read_route_seq,
            token_mask,
        )

    def _last_valid_hidden(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Last-hidden capture width does not match the attention hidden size: "
                f"expected={self.hidden_size} actual={hidden_size}"
            )
        if token_mask is None:
            return hidden_states[:, -1, :]
        if tuple(token_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                "Last-hidden capture mask must match the model token shape: "
                f"expected={(batch_size, seq_len)} actual={tuple(token_mask.shape)}"
            )
        valid_tokens = token_mask.to(device=hidden_states.device, dtype=torch.bool)
        token_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        last_valid_indices = torch.where(valid_tokens, token_indices, -1).amax(dim=1)
        has_valid_token = last_valid_indices.ge(0)
        gather_indices = last_valid_indices.clamp_min(0).view(batch_size, 1, 1)
        gather_indices = gather_indices.expand(-1, 1, hidden_size)
        captured = hidden_states.gather(dim=1, index=gather_indices).squeeze(1)
        return captured * has_valid_token.unsqueeze(-1).to(captured.dtype)

    def _capture_direct_last_hidden(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> None:
        self.direct_last_hidden = self._last_valid_hidden(hidden_states, token_mask)

    def _capture_projected_last_hidden(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> None:
        captured = self._last_valid_hidden(hidden_states, token_mask)
        normalized_hidden = F.rms_norm(
            captured.float(),
            (self.hidden_size,),
            eps=1e-6,
        )
        self.projected_last_hidden = F.linear(
            normalized_hidden,
            self.memory_v_proj.float(),
        ).to(dtype=self.memory_v_proj.dtype)

    def _direct_last_hidden_token_reads(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        saved_hidden = self.direct_last_hidden
        if saved_hidden is None:
            return torch.zeros(
                batch_size,
                seq_len,
                self.state_read_dim,
                device=hidden_states.device,
                dtype=self.memory_v_proj.dtype,
            )
        if tuple(saved_hidden.shape) != (batch_size, self.hidden_size):
            raise ValueError(
                "Saved direct-hidden state does not match the current batch: "
                f"expected={(batch_size, self.hidden_size)} actual={tuple(saved_hidden.shape)}"
            )
        normalized_hidden = F.rms_norm(
            saved_hidden.float(),
            (self.hidden_size,),
            eps=1e-6,
        )
        row_reads = F.linear(normalized_hidden, self.memory_v_proj.float())
        row_reads = row_reads.to(dtype=self.memory_v_proj.dtype)
        return row_reads.unsqueeze(1).expand(-1, seq_len, -1)

    def _projected_last_hidden_token_reads(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        projected_hidden = self.projected_last_hidden
        if projected_hidden is None:
            return torch.zeros(
                batch_size,
                seq_len,
                self.state_read_dim,
                device=hidden_states.device,
                dtype=self.memory_v_proj.dtype,
            )
        if tuple(projected_hidden.shape) != (batch_size, self.state_read_dim):
            raise ValueError(
                "Saved projected-hidden state does not match the current batch: "
                f"expected={(batch_size, self.state_read_dim)} "
                f"actual={tuple(projected_hidden.shape)}"
            )
        return projected_hidden.unsqueeze(1).expand(-1, seq_len, -1)

    def _projected_kv_write_proposals(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        key_span_mask = self.projected_kv_write_key_mask
        value_span_mask = self.projected_kv_write_value_mask
        if (key_span_mask is None) != (value_span_mask is None):
            raise RuntimeError(
                "Projected-KV key and value write masks must both be set or both be absent"
            )
        if key_span_mask is not None and value_span_mask is not None:
            batch_size, seq_len, _ = hidden_states.shape
            expected_shape = (batch_size, seq_len)
            if tuple(key_span_mask.shape) != expected_shape:
                raise ValueError(
                    "Projected-KV key write mask must match the model token shape: "
                    f"expected={expected_shape} actual={tuple(key_span_mask.shape)}"
                )
            if tuple(value_span_mask.shape) != expected_shape:
                raise ValueError(
                    "Projected-KV value write mask must match the model token shape: "
                    f"expected={expected_shape} actual={tuple(value_span_mask.shape)}"
                )
            if token_mask is None:
                valid_tokens = torch.ones(
                    expected_shape,
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            else:
                if tuple(token_mask.shape) != expected_shape:
                    raise ValueError(
                        "Projected-KV token mask must match the model token shape: "
                        f"expected={expected_shape} actual={tuple(token_mask.shape)}"
                    )
                valid_tokens = token_mask.to(
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            key_span_mask = key_span_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            value_span_mask = value_span_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            if bool(((key_span_mask | value_span_mask) & ~valid_tokens).any().item()):
                raise ValueError(
                    "Projected-KV key and value write spans may select only valid tokens"
                )
            if bool((key_span_mask & value_span_mask).any().item()):
                raise ValueError("Projected-KV key and value write spans must not overlap")

            message_ids = self.write_message_ids
            if message_ids is not None:
                if tuple(message_ids.shape) != expected_shape:
                    raise ValueError(
                        "Projected-KV write message IDs must match the model token shape: "
                        f"expected={expected_shape} actual={tuple(message_ids.shape)}"
                    )
                message_ids = message_ids.to(device=hidden_states.device)
                selected_tokens = key_span_mask | value_span_mask
                if bool((selected_tokens & message_ids.lt(0)).any().item()):
                    raise ValueError(
                        "Projected-KV write spans may select only tokens with a message ID"
                    )
                if not bool(selected_tokens.any().item()):
                    return None, None, None
                num_proposals = int(
                    message_ids.masked_select(selected_tokens).max().item()
                ) + 1
                key_hidden = hidden_states.new_zeros(
                    batch_size,
                    num_proposals,
                    self.hidden_size,
                )
                value_hidden = torch.zeros_like(key_hidden)
                proposal_mask = torch.zeros(
                    batch_size,
                    num_proposals,
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
                for batch_idx in range(batch_size):
                    for proposal_idx in range(num_proposals):
                        in_proposal = message_ids[batch_idx].eq(proposal_idx)
                        key_tokens = key_span_mask[batch_idx] & in_proposal
                        value_tokens = value_span_mask[batch_idx] & in_proposal
                        key_present = bool(key_tokens.any().item())
                        value_present = bool(value_tokens.any().item())
                        if key_present != value_present:
                            raise ValueError(
                                "Every Projected-KV record must contain both a key span "
                                "and a value span"
                            )
                        if not key_present:
                            continue
                        key_hidden[batch_idx, proposal_idx] = hidden_states[
                            batch_idx, key_tokens
                        ].mean(dim=0)
                        value_hidden[batch_idx, proposal_idx] = hidden_states[
                            batch_idx, value_tokens
                        ].mean(dim=0)
                        proposal_mask[batch_idx, proposal_idx] = True
                return key_hidden, value_hidden, proposal_mask

            key_present = key_span_mask.any(dim=1)
            value_present = value_span_mask.any(dim=1)
            if not torch.equal(key_present, value_present):
                raise ValueError(
                    "Every Projected-KV record must contain both a key span and a value span"
                )
            if not bool(key_present.any().item()):
                return None, None, None
            key_counts = key_span_mask.sum(dim=1, keepdim=True).clamp_min(1)
            value_counts = value_span_mask.sum(dim=1, keepdim=True).clamp_min(1)
            key_hidden = torch.einsum(
                "bt,bth->bh",
                key_span_mask.to(dtype=hidden_states.dtype),
                hidden_states,
            ) / key_counts.to(dtype=hidden_states.dtype)
            value_hidden = torch.einsum(
                "bt,bth->bh",
                value_span_mask.to(dtype=hidden_states.dtype),
                hidden_states,
            ) / value_counts.to(dtype=hidden_states.dtype)
            return (
                key_hidden.unsqueeze(1),
                value_hidden.unsqueeze(1),
                key_present.unsqueeze(1),
            )

        if self.memory_write_granularity == "message_mean":
            proposal_hidden, proposal_mask, _ = self._build_message_write_means(
                hidden_states,
                token_mask,
            )
            return proposal_hidden, proposal_hidden, proposal_mask
        if self.memory_write_granularity == "sentence_mean":
            proposal_hidden, proposal_mask, _ = self._build_sentence_write_means(
                hidden_states,
                token_mask,
            )
            return proposal_hidden, proposal_hidden, proposal_mask

        batch_size, seq_len, _ = hidden_states.shape
        if token_mask is None:
            valid_tokens = torch.ones(
                batch_size,
                seq_len,
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            if tuple(token_mask.shape) != (batch_size, seq_len):
                raise ValueError(
                    "Projected-KV token mask must match the model token shape: "
                    f"expected={(batch_size, seq_len)} actual={tuple(token_mask.shape)}"
                )
            valid_tokens = token_mask.to(device=hidden_states.device, dtype=torch.bool)
        token_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        last_valid_indices = torch.where(valid_tokens, token_indices, -1).amax(dim=1)
        proposal_mask = last_valid_indices.ge(0).unsqueeze(1)
        gather_indices = last_valid_indices.clamp_min(0).view(batch_size, 1, 1)
        proposal_hidden = hidden_states.gather(
            dim=1,
            index=gather_indices.expand(-1, 1, self.hidden_size),
        )
        return proposal_hidden, proposal_hidden, proposal_mask

    def _projected_kv_project_hidden(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden = F.rms_norm(
            hidden_states.float(),
            (self.hidden_size,),
            eps=1e-6,
        )
        keys = F.linear(normalized_hidden, self.projected_kv_key_proj.float())
        keys = F.normalize(keys, dim=-1, eps=1e-6).to(
            dtype=self.projected_kv_key_proj.dtype
        )
        values = F.linear(normalized_hidden, self.memory_v_proj.float()).to(
            dtype=self.memory_v_proj.dtype
        )
        return keys, values

    def _ensure_projected_kv_slot_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sidecars = (
            self.projected_kv_keys,
            self.projected_kv_values,
            self.projected_kv_occupied,
            self.projected_kv_surprise,
        )
        if any(sidecar is None for sidecar in sidecars) and not all(
            sidecar is None for sidecar in sidecars
        ):
            raise RuntimeError("Projected-KV slot state is incomplete")
        needs_init = all(sidecar is None for sidecar in sidecars)
        if not needs_init:
            assert self.projected_kv_keys is not None
            needs_init = (
                self.projected_kv_keys.size(0) != batch_size
                or self.projected_kv_keys.device != device
            )
        if needs_init:
            self.projected_kv_keys = torch.zeros(
                batch_size,
                self.rwkv_ms_num_states,
                self.projected_kv_key_dim,
                device=device,
                dtype=self.projected_kv_key_proj.dtype,
            )
            self.projected_kv_values = torch.zeros(
                batch_size,
                self.rwkv_ms_num_states,
                self.state_read_dim,
                device=device,
                dtype=self.memory_v_proj.dtype,
            )
            self.projected_kv_occupied = torch.zeros(
                batch_size,
                self.rwkv_ms_num_states,
                device=device,
                dtype=torch.bool,
            )
            self.projected_kv_surprise = torch.zeros(
                batch_size,
                self.rwkv_ms_num_states,
                device=device,
                dtype=torch.float32,
            )
        assert self.projected_kv_keys is not None
        assert self.projected_kv_values is not None
        assert self.projected_kv_occupied is not None
        assert self.projected_kv_surprise is not None
        expected_shapes = (
            (batch_size, self.rwkv_ms_num_states, self.projected_kv_key_dim),
            (batch_size, self.rwkv_ms_num_states, self.state_read_dim),
            (batch_size, self.rwkv_ms_num_states),
            (batch_size, self.rwkv_ms_num_states),
        )
        for label, sidecar, expected_shape in zip(
            ("keys", "values", "occupied", "surprise"),
            (
                self.projected_kv_keys,
                self.projected_kv_values,
                self.projected_kv_occupied,
                self.projected_kv_surprise,
            ),
            expected_shapes,
        ):
            if tuple(sidecar.shape) != expected_shape:
                raise ValueError(
                    f"Projected-KV {label} state has shape {tuple(sidecar.shape)}; "
                    f"expected {expected_shape}"
                )
        return (
            self.projected_kv_keys,
            self.projected_kv_values,
            self.projected_kv_occupied,
            self.projected_kv_surprise,
        )

    def _write_projected_kv_slots(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> None:
        key_hidden, value_hidden, proposal_mask = (
            self._projected_kv_write_proposals(
                hidden_states,
                token_mask,
            )
        )
        if key_hidden is None or value_hidden is None or proposal_mask is None:
            self.last_write_routes = None
            return
        proposal_keys, _ = self._projected_kv_project_hidden(key_hidden)
        _, proposal_values = self._projected_kv_project_hidden(value_hidden)
        batch_size, num_proposals, _ = proposal_keys.shape
        forced_slots = self.projected_kv_write_slot_indices
        if forced_slots is not None:
            if forced_slots.ndim == 1 and num_proposals == 1:
                forced_slots = forced_slots.unsqueeze(1)
            expected_shape = (batch_size, num_proposals)
            if tuple(forced_slots.shape) != expected_shape:
                raise ValueError(
                    "Projected-KV forced write slots must match the proposal shape: "
                    f"expected={expected_shape} actual={tuple(forced_slots.shape)}"
                )
            forced_slots = forced_slots.to(
                device=hidden_states.device,
                dtype=torch.long,
            )
            invalid_for_valid = proposal_mask & (
                forced_slots.lt(0) | forced_slots.ge(self.rwkv_ms_num_states)
            )
            if bool(invalid_for_valid.any().item()):
                raise ValueError(
                    "Projected-KV forced write slot is outside the configured capacity"
                )
        keys, values, occupied, surprise = self._ensure_projected_kv_slot_state(
            batch_size,
            hidden_states.device,
        )
        write_routes: list[torch.Tensor] = []
        for proposal_idx in range(num_proposals):
            candidate_key = proposal_keys[:, proposal_idx]
            candidate_value = proposal_values[:, proposal_idx]
            candidate_valid = proposal_mask[:, proposal_idx].to(
                device=hidden_states.device,
                dtype=torch.bool,
            )

            if forced_slots is not None:
                target_index = forced_slots[:, proposal_idx].clamp(
                    min=0,
                    max=self.rwkv_ms_num_states - 1,
                )
                selected = F.one_hot(
                    target_index,
                    num_classes=self.rwkv_ms_num_states,
                ).to(dtype=torch.bool)
                selected = selected & candidate_valid.unsqueeze(-1)
                keys = torch.where(
                    selected.unsqueeze(-1),
                    candidate_key.unsqueeze(1),
                    keys,
                )
                values = torch.where(
                    selected.unsqueeze(-1),
                    candidate_value.unsqueeze(1),
                    values,
                )
                occupied = occupied | selected
                surprise = torch.where(
                    selected,
                    torch.ones_like(surprise),
                    surprise,
                )
                write_routes.append(selected.to(dtype=proposal_values.dtype))
                continue

            cosine = torch.einsum(
                "bck,bk->bc",
                keys.float(),
                candidate_key.float(),
            ).clamp(min=-1.0, max=1.0)
            masked_cosine = cosine.masked_fill(~occupied, -torch.inf)
            closest_cosine, closest_index = masked_cosine.max(dim=-1)
            has_occupied = occupied.any(dim=-1)
            candidate_surprise = torch.where(
                has_occupied,
                1.0 - closest_cosine,
                torch.ones_like(closest_cosine),
            )
            matching = (
                candidate_valid
                & has_occupied
                & closest_cosine.ge(self.projected_kv_update_cosine_threshold)
            )

            has_empty = (~occupied).any(dim=-1)
            empty_index = (~occupied).to(dtype=torch.int64).argmax(dim=-1)
            minimum_surprise, minimum_index = surprise.masked_fill(
                ~occupied,
                torch.inf,
            ).min(dim=-1)
            evict = (
                candidate_valid
                & ~matching
                & ~has_empty
                & candidate_surprise.ge(minimum_surprise)
            )
            insert = candidate_valid & ~matching & has_empty
            should_update = matching | insert | evict
            target_index = torch.where(
                matching,
                closest_index,
                torch.where(insert, empty_index, minimum_index),
            )
            selected = F.one_hot(
                target_index,
                num_classes=self.rwkv_ms_num_states,
            ).to(dtype=torch.bool)
            selected = selected & should_update.unsqueeze(-1)
            keys = torch.where(
                selected.unsqueeze(-1),
                candidate_key.unsqueeze(1),
                keys,
            )
            values = torch.where(
                selected.unsqueeze(-1),
                candidate_value.unsqueeze(1),
                values,
            )
            occupied = occupied | selected
            surprise = torch.where(
                selected,
                candidate_surprise.detach().unsqueeze(-1).to(dtype=torch.float32),
                surprise,
            )
            write_routes.append(selected.to(dtype=proposal_values.dtype))

        self.projected_kv_keys = keys
        self.projected_kv_values = values
        self.projected_kv_occupied = occupied
        self.projected_kv_surprise = surprise
        self.last_write_routes = torch.stack(write_routes, dim=1)

    def _write_chunk_addressed_kv_slots(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> None:
        if any(
            metadata is not None
            for metadata in (
                self.projected_kv_write_key_mask,
                self.projected_kv_write_value_mask,
                self.projected_kv_write_slot_indices,
            )
        ):
            raise ValueError(
                "chunk_addressed_value derives projected writes from RWKV chunks "
                "and does not accept explicit projected-KV write spans"
            )
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len == 0:
            self.last_write_routes = None
            return
        positions = self._ensure_rwkv_ms_positions(
            batch_size,
            hidden_states.device,
        )
        token_slots = rwkv_ms_write_slot_indices(
            token_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            positions=positions,
            chunk_size=self.rwkv_ms_chunk_size,
            num_slots=self.rwkv_ms_num_states,
        )
        slot_ids = torch.arange(
            self.rwkv_ms_num_states,
            device=hidden_states.device,
        ).view(1, 1, self.rwkv_ms_num_states)
        token_indices = torch.arange(
            seq_len,
            device=hidden_states.device,
        ).view(1, seq_len, 1)
        matching_indices = torch.where(
            token_slots.unsqueeze(-1).eq(slot_ids),
            token_indices,
            token_indices.new_full((), -1),
        )
        last_token_indices = matching_indices.amax(dim=1)
        written_slots = last_token_indices.ge(0)
        gather_indices = last_token_indices.clamp_min(0).unsqueeze(-1).expand(
            -1,
            -1,
            self.hidden_size,
        )
        chunk_hidden = hidden_states.gather(dim=1, index=gather_indices)
        chunk_keys, _ = self._projected_kv_project_hidden(chunk_hidden)
        keys, values, occupied, surprise = self._ensure_projected_kv_slot_state(
            batch_size,
            hidden_states.device,
        )
        keys = torch.where(
            written_slots.unsqueeze(-1),
            chunk_keys,
            keys,
        )
        values = torch.where(
            written_slots.unsqueeze(-1),
            torch.zeros_like(values),
            values,
        )
        self.projected_kv_keys = keys
        self.projected_kv_values = values
        self.projected_kv_occupied = occupied | written_slots
        self.projected_kv_surprise = torch.where(
            written_slots,
            torch.ones_like(surprise),
            surprise,
        )
        self.last_write_routes = F.one_hot(
            token_slots.clamp_min(0),
            num_classes=self.rwkv_ms_num_states,
        ).to(dtype=self.memory_v_proj.dtype)
        self.last_write_routes = self.last_write_routes * token_slots.ge(0).unsqueeze(
            -1
        ).to(dtype=self.last_write_routes.dtype)

    def _projected_kv_slot_token_reads(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        sidecars = (
            self.projected_kv_keys,
            self.projected_kv_values,
            self.projected_kv_occupied,
            self.projected_kv_surprise,
        )
        if all(sidecar is None for sidecar in sidecars):
            self.last_read_routes = None
            self.last_read_route_logits = None
            return torch.zeros(
                batch_size,
                seq_len,
                self.state_read_dim,
                device=hidden_states.device,
                dtype=self.memory_v_proj.dtype,
            )
        keys, values, occupied, _ = self._ensure_projected_kv_slot_state(
            batch_size,
            hidden_states.device,
        )
        query_hidden = hidden_states
        query_mask = self.projected_kv_read_query_mask
        pooled_query = query_mask is not None
        if query_mask is not None:
            expected_shape = (batch_size, seq_len)
            if tuple(query_mask.shape) != expected_shape:
                raise ValueError(
                    "Projected-KV read query mask must match the model token shape: "
                    f"expected={expected_shape} actual={tuple(query_mask.shape)}"
                )
            query_mask = query_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            query_counts = query_mask.sum(dim=1, keepdim=True)
            if bool(query_counts.eq(0).any().item()):
                raise ValueError(
                    "Projected-KV read query mask must select at least one token per row"
                )
            query_hidden = (
                torch.einsum(
                    "bt,bth->bh",
                    query_mask.to(dtype=hidden_states.dtype),
                    hidden_states,
                )
                / query_counts.to(dtype=hidden_states.dtype)
            ).unsqueeze(1)
        query_keys, _ = self._projected_kv_project_hidden(query_hidden)
        cosine = torch.einsum(
            "btk,bck->btc",
            query_keys.float(),
            keys.float(),
        ).clamp(min=-1.0, max=1.0)
        has_memory = occupied.any(dim=-1, keepdim=True)
        routable = occupied.clone()
        routable[:, 0] |= ~has_memory.squeeze(-1)
        logits = (cosine * self.projected_kv_temperature).masked_fill(
            ~routable.unsqueeze(1),
            -torch.inf,
        )
        soft_routes = torch.softmax(logits, dim=-1)
        hard_routes = F.one_hot(
            logits.argmax(dim=-1),
            num_classes=self.rwkv_ms_num_states,
        ).to(dtype=soft_routes.dtype)
        routes = hard_routes + soft_routes - soft_routes.detach()
        routes = routes * has_memory.unsqueeze(1).to(dtype=routes.dtype)
        if pooled_query:
            logits = logits.expand(-1, seq_len, -1)
            routes = routes.expand(-1, seq_len, -1)
        self.last_read_routes = routes
        self.last_read_route_logits = logits
        return torch.einsum(
            "btc,bcd->btd",
            routes,
            values.float(),
        ).to(dtype=self.memory_v_proj.dtype)

    def _fuse_projected_rwkv_reads(
        self,
        projected_reads: torch.Tensor,
        recurrent_reads: torch.Tensor,
    ) -> torch.Tensor:
        if projected_reads.shape != recurrent_reads.shape:
            raise ValueError(
                "Projected and recurrent hybrid reads must have identical shapes: "
                f"projected={tuple(projected_reads.shape)} "
                f"recurrent={tuple(recurrent_reads.shape)}"
            )
        projected = projected_reads.float()
        recurrent = recurrent_reads.float()
        recurrent_rms = recurrent.square().mean(dim=-1, keepdim=True).sqrt()
        recurrent_direction = torch.tanh(
            recurrent / recurrent_rms.clamp_min(1e-6)
        )
        carrier_rms = projected.square().mean(dim=-1, keepdim=True).sqrt()
        gain = float(self.rwkv_ms_hybrid_gain)
        if self.rwkv_ms_hybrid_mode == "residual":
            fused = projected + gain * carrier_rms * recurrent_direction
        elif self.rwkv_ms_hybrid_mode == "vector_gate":
            fused = projected * (1.0 + gain * recurrent_direction)
        elif self.rwkv_ms_hybrid_mode == "scalar_gate":
            alignment = (
                F.normalize(projected, dim=-1, eps=1e-6)
                * F.normalize(recurrent, dim=-1, eps=1e-6)
            ).sum(dim=-1, keepdim=True)
            fused = projected * (1.0 + gain * alignment.clamp(-1.0, 1.0))
        elif self.rwkv_ms_hybrid_mode in RWKV_MS_VALUE_BOTTLENECK_MODES:
            fused = gain * recurrent_direction
        else:  # pragma: no cover - configuration validation is authoritative
            raise RuntimeError(
                f"Unsupported RWKV-MS hybrid mode: {self.rwkv_ms_hybrid_mode}"
            )
        return fused.to(dtype=projected_reads.dtype)

    def _projected_rwkv_hybrid_step(
        self,
        state: torch.Tensor,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            token_memory_q_seq,
            token_memory_k_seq,
            token_memory_v_seq,
            beta_seq,
            lambda_seq,
        ) = self._memory_sequence_projections(hidden_states)
        if self.write_enabled:
            if self.rwkv_ms_hybrid_mode == "chunk_addressed_value":
                self._write_chunk_addressed_kv_slots(hidden_states, token_mask)
            else:
                self._write_projected_kv_slots(hidden_states, token_mask)
            state, _ = self._memory_backend_scan(
                state,
                token_memory_q_seq,
                token_memory_k_seq,
                token_memory_v_seq,
                beta_seq,
                lambda_seq,
                token_mask=token_mask,
                write_only=True,
            )
            reads = torch.zeros(
                hidden_states.size(0),
                hidden_states.size(1),
                self.state_read_dim,
                device=hidden_states.device,
                dtype=self.memory_v_proj.dtype,
            )
            self.last_read_routes = None
            self.last_read_route_logits = None
        else:
            if self.rwkv_ms_hybrid_mode == "recurrent_value":
                projected_reads = torch.zeros(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.state_read_dim,
                    device=hidden_states.device,
                    dtype=self.memory_v_proj.dtype,
                )
                projected_routes = None
            else:
                projected_reads = self._projected_kv_slot_token_reads(hidden_states)
                projected_routes = self.last_read_routes
            if self.rwkv_ms_hybrid_mode in RWKV_MS_ADDRESSED_VALUE_MODES:
                recurrent_reads = (
                    torch.zeros_like(projected_reads)
                    if projected_routes is None
                    else self._rwkv_ms_addressed_token_state_reads(
                        state,
                        token_memory_v_seq,
                        projected_routes,
                        token_mask,
                    )
                )
            else:
                recurrent_reads = self._memory_backend_token_reads(
                    state,
                    token_memory_v_seq,
                    None,
                    token_mask,
                )
            recurrent_routes = self.last_read_routes
            reads = self._fuse_projected_rwkv_reads(
                projected_reads,
                recurrent_reads,
            )
            self.last_read_routes = (
                recurrent_routes
                if self.rwkv_ms_hybrid_mode == "recurrent_value"
                else projected_routes
            )
            self.last_write_routes = None
        return state, reads, beta_seq, lambda_seq

    def _fuse_delta_o_output(
        self,
        base_o_output: torch.Tensor,
        delta_o: torch.Tensor | None,
        hidden_states: torch.Tensor,
        reads: torch.Tensor,
        token_mask: torch.Tensor | None,
        fusion_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.last_base_o_norm = None
        self.last_delta_o_norm = None
        self.last_delta_o_ratio = None
        self.last_delta_o_gate_mean = None
        self.last_delta_o_gate_min = None
        self.last_delta_o_gate_max = None
        self.last_delta_o_gate_lt_001_fraction = None
        self.last_delta_o_gate_gt_099_fraction = None
        self.last_fused_delta_o_norm = None
        self.last_fused_delta_o_ratio = None
        self.last_delta_o_base_cosine = None
        self.last_fused_o_ratio = None
        self.last_applied_memory_correction_norm = None
        self.last_applied_memory_correction_ratio = None
        self.last_memory_residual_norm = None
        self.last_memory_residual_ratio = None
        self.last_memory_residual_gain = None
        delta_o_typed = None
        fused_delta_o = None
        if delta_o is not None:
            delta_o_typed = self._apply_delta_o_rmsnorm(delta_o.to(hidden_states.dtype))
            fusion_gate = (
                self._memory_fusion_gate(hidden_states, reads)
                if fusion_gate is None
                else fusion_gate
            )
            self.last_delta_o_gate_mean = self._masked_token_mean(fusion_gate, token_mask)
            self.last_delta_o_gate_min, self.last_delta_o_gate_max = (
                self._masked_token_min_max(fusion_gate, token_mask)
            )
            self.last_delta_o_gate_lt_001_fraction = self._masked_token_fraction(
                fusion_gate,
                token_mask,
                threshold=0.01,
                greater=False,
            )
            self.last_delta_o_gate_gt_099_fraction = self._masked_token_fraction(
                fusion_gate,
                token_mask,
                threshold=0.99,
                greater=True,
            )
            fused_delta_o = delta_o_typed * fusion_gate.to(dtype=delta_o_typed.dtype)

        if self.memory_fusion_placement in MEMORY_FUSION_NORM_HOOK_PLACEMENTS:
            if self._post_attention_norm_hook_handle is None:
                raise RuntimeError(
                    f"{self.memory_fusion_placement} fusion requires attach_delta_mem to bind "
                    "Gemma's post_attention_layernorm"
                )
            if self._pending_post_attention_delta is not None:
                raise RuntimeError(
                    "Previous post-attention memory delta was not consumed before the next "
                    f"attention forward in layer {self.layer_idx}"
                )
            self._pending_post_attention_delta = (
                delta_o_typed,
                fused_delta_o,
                token_mask,
            )
            return base_o_output

        return self._add_delta_o_to_reference(
            base_o_output,
            delta_o_typed,
            fused_delta_o,
            token_mask,
        )

    def _add_delta_o_to_reference(
        self,
        reference_output: torch.Tensor,
        delta_o: torch.Tensor | None,
        fused_delta_o: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._record_delta_o_reference_stats(
            reference_output,
            delta_o,
            fused_delta_o,
            token_mask,
        )
        if delta_o is None or fused_delta_o is None:
            self._record_applied_memory_correction(
                reference_output,
                None,
                token_mask,
            )
            return reference_output
        if self._eval_memory_delta_controller is None:
            applied_correction = fused_delta_o.to(reference_output.dtype)
            fused_output = reference_output + applied_correction
        else:
            fused_output = self._add_memory_delta(
                reference_output,
                fused_delta_o,
                head_name="o",
                token_mask=token_mask,
            )
            applied_correction = fused_output - reference_output
        self._record_applied_memory_correction(
            reference_output,
            applied_correction,
            token_mask,
        )
        self.last_fused_o_ratio = self._masked_ratio_mean(
            fused_output.norm(dim=-1),
            reference_output.norm(dim=-1),
            token_mask,
        )
        return fused_output

    def _add_scaled_delta_o_to_reference(
        self,
        reference_output: torch.Tensor,
        delta_o: torch.Tensor | None,
        fused_delta_o: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        scale: float | torch.Tensor,
    ) -> torch.Tensor:
        resolved_scale = (
            reference_output.new_tensor(scale)
            if isinstance(scale, float)
            else scale.to(device=reference_output.device, dtype=reference_output.dtype)
        )
        self.last_memory_residual_gain = resolved_scale.detach()
        if isinstance(scale, float) and scale == 1.0:
            fused_output = self._add_delta_o_to_reference(
                reference_output,
                delta_o,
                fused_delta_o,
                token_mask,
            )
            self.last_memory_residual_norm = self.last_applied_memory_correction_norm
            self.last_memory_residual_ratio = self.last_applied_memory_correction_ratio
            return fused_output

        self._record_delta_o_reference_stats(
            reference_output,
            delta_o,
            fused_delta_o,
            token_mask,
        )
        if delta_o is None or fused_delta_o is None:
            self._record_applied_memory_correction(
                reference_output,
                None,
                token_mask,
            )
            self.last_memory_residual_norm = reference_output.new_zeros(())
            self.last_memory_residual_ratio = reference_output.new_zeros(())
            return reference_output
        if isinstance(scale, float) and scale == 0.0:
            applied_correction = torch.zeros_like(reference_output)
            self._record_applied_memory_correction(
                reference_output,
                applied_correction,
                token_mask,
            )
            self.last_memory_residual_norm = reference_output.new_zeros(())
            self.last_memory_residual_ratio = reference_output.new_zeros(())
            self.last_fused_o_ratio = self._masked_ratio_mean(
                reference_output.norm(dim=-1),
                reference_output.norm(dim=-1),
                token_mask,
            )
            return reference_output

        applied_correction = fused_delta_o.to(reference_output.dtype) * resolved_scale
        self._record_applied_memory_correction(
            reference_output,
            applied_correction,
            token_mask,
        )
        self.last_memory_residual_norm = self.last_applied_memory_correction_norm
        self.last_memory_residual_ratio = self.last_applied_memory_correction_ratio
        fused_output = reference_output + applied_correction
        self.last_fused_o_ratio = self._masked_ratio_mean(
            fused_output.norm(dim=-1),
            reference_output.norm(dim=-1),
            token_mask,
        )
        return fused_output

    def _record_delta_o_reference_stats(
        self,
        reference_output: torch.Tensor,
        delta_o: torch.Tensor | None,
        fused_delta_o: torch.Tensor | None,
        token_mask: torch.Tensor | None,
    ) -> None:
        self.last_base_o_norm = self._masked_hidden_norm(reference_output, token_mask)
        if delta_o is None or fused_delta_o is None:
            return
        self.last_delta_o_norm = self._masked_hidden_norm(delta_o, token_mask)
        self.last_delta_o_ratio = self._masked_ratio_mean(
            delta_o.norm(dim=-1),
            reference_output.norm(dim=-1),
            token_mask,
        )
        self.last_fused_delta_o_norm = self._masked_hidden_norm(fused_delta_o, token_mask)
        self.last_fused_delta_o_ratio = self._masked_ratio_mean(
            fused_delta_o.norm(dim=-1),
            reference_output.norm(dim=-1),
            token_mask,
        )
        self.last_delta_o_base_cosine = self._masked_cosine_mean(
            fused_delta_o,
            reference_output,
            token_mask,
        )

    def _post_attention_norm_fusion_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        payload = self._pending_post_attention_delta
        if payload is None:
            raise RuntimeError(
                "Gemma post_attention_layernorm ran without a pending memory delta for "
                f"layer {self.layer_idx}"
            )
        self._pending_post_attention_delta = None
        delta_o, fused_delta_o, token_mask = payload
        if self.memory_fusion_placement == "post_attention_norm":
            return self._add_scaled_delta_o_to_reference(
                output,
                delta_o,
                fused_delta_o,
                token_mask,
                self.memory_fusion_residual_scale,
            )
        if self.memory_fusion_placement not in MEMORY_FUSION_NORMALIZED_RESIDUAL_PLACEMENTS:
            raise RuntimeError(
                "A post-attention RMSNorm hook received an unsupported memory fusion "
                f"placement: {self.memory_fusion_placement}"
            )
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError(
                f"{self.memory_fusion_placement} requires Gemma's post-attention "
                "RMSNorm to receive exactly one tensor input"
            )
        raw_attention = inputs[0]
        if raw_attention.shape != output.shape:
            raise RuntimeError(
                f"{self.memory_fusion_placement} RMSNorm input/output shape mismatch: "
                f"input={tuple(raw_attention.shape)} output={tuple(output.shape)}"
            )
        self._record_delta_o_reference_stats(
            output,
            delta_o,
            fused_delta_o,
            token_mask,
        )
        if delta_o is None or fused_delta_o is None:
            self._record_applied_memory_correction(output, None, token_mask)
            self.last_memory_residual_norm = output.new_zeros(())
            self.last_memory_residual_ratio = output.new_zeros(())
            self.last_fused_o_ratio = self._masked_ratio_mean(
                output.norm(dim=-1),
                output.norm(dim=-1),
                token_mask,
            )
            return output

        scale = self.memory_fusion_residual_scale
        if self.memory_fusion_placement == "normalized_residual_correction":
            self.last_memory_residual_gain = output.new_tensor(scale)
            if scale == 0.0:
                self._record_applied_memory_correction(
                    output,
                    torch.zeros_like(output),
                    token_mask,
                )
                self.last_memory_residual_norm = output.new_zeros(())
                self.last_memory_residual_ratio = output.new_zeros(())
                self.last_fused_o_ratio = self._masked_ratio_mean(
                    output.norm(dim=-1),
                    output.norm(dim=-1),
                    token_mask,
                )
                return output

        memory_norm = module.forward(
            raw_attention + fused_delta_o.to(dtype=raw_attention.dtype)
        )
        correction = memory_norm - output
        if self.memory_fusion_placement == "post_attention_residual_hybrid":
            gain = self._resolved_memory_fusion_residual_gain(
                device=output.device,
                dtype=output.dtype,
            )
            self.last_memory_residual_gain = gain.detach()
            direct_residual = fused_delta_o.to(dtype=output.dtype) * gain
            applied_correction = correction + direct_residual
            self._record_applied_memory_correction(
                output,
                applied_correction,
                token_mask,
            )
            self.last_memory_residual_norm = self.last_applied_memory_correction_norm
            self.last_memory_residual_ratio = self.last_applied_memory_correction_ratio
            fused_output = memory_norm + direct_residual
            self.last_fused_o_ratio = self._masked_ratio_mean(
                fused_output.norm(dim=-1),
                output.norm(dim=-1),
                token_mask,
            )
            return fused_output

        applied_correction = correction * scale
        self._record_applied_memory_correction(
            output,
            applied_correction,
            token_mask,
        )
        self.last_memory_residual_norm = self.last_applied_memory_correction_norm
        self.last_memory_residual_ratio = self.last_applied_memory_correction_ratio
        if scale == 1.0:
            fused_output = memory_norm
        else:
            fused_output = output + applied_correction
        self.last_fused_o_ratio = self._masked_ratio_mean(
            fused_output.norm(dim=-1),
            output.norm(dim=-1),
            token_mask,
        )
        return fused_output

    def bind_post_attention_layernorm(self, layernorm: nn.Module) -> None:
        if self.memory_fusion_placement not in MEMORY_FUSION_NORM_HOOK_PLACEMENTS:
            raise ValueError(
                "A post-attention RMSNorm hook can only be bound for a norm-hook fusion "
                f"placement, got {self.memory_fusion_placement}"
            )
        if self._post_attention_norm_hook_handle is not None:
            raise RuntimeError(
                f"Layer {self.layer_idx} already has a post-attention RMSNorm fusion hook"
            )
        self._post_attention_norm_hook_handle = layernorm.register_forward_hook(
            self._post_attention_norm_fusion_hook
        )

    def remove_post_attention_layernorm_hook(self) -> None:
        handle = self._post_attention_norm_hook_handle
        self._post_attention_norm_hook_handle = None
        self._pending_post_attention_delta = None
        if handle is not None:
            handle.remove()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"DeltaMemAttention expects [batch, seq, hidden], got {tuple(hidden_states.shape)}"
            )
        if self.memory_fusion_placement in MEMORY_FUSION_NORM_HOOK_PLACEMENTS:
            if not self.is_gemma4_attention or self._post_attention_norm_hook_handle is None:
                raise RuntimeError(
                    f"{self.memory_fusion_placement} fusion is only valid for a Gemma "
                    "attention wrapper bound to its decoder layer's "
                    "post_attention_layernorm"
                )
            if self._pending_post_attention_delta is not None:
                raise RuntimeError(
                    "Previous post-attention memory delta was not consumed before the next "
                    f"attention forward in layer {self.layer_idx}"
                )

        batch_size, seq_len, _ = hidden_states.shape
        state = self._ensure_state(batch_size, hidden_states.device, hidden_states.dtype)
        token_mask = self._token_validity_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )
        read_mask = self._resolve_read_context_mask(
            token_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )
        if self.memory_readout_mode == "projected_kv_rwkv_hybrid":
            state, reads, stats_beta, stats_lambda = (
                self._projected_rwkv_hybrid_step(
                    state,
                    hidden_states,
                    token_mask,
                )
            )
            stats_mask = token_mask
        elif self.memory_readout_mode in {
            "direct_last_hidden",
            "projected_last_hidden",
            "projected_kv_slots",
        }:
            if self.write_enabled:
                if self.memory_readout_mode == "direct_last_hidden":
                    self._capture_direct_last_hidden(hidden_states, token_mask)
                elif self.memory_readout_mode == "projected_last_hidden":
                    self._capture_projected_last_hidden(hidden_states, token_mask)
                else:
                    self._write_projected_kv_slots(hidden_states, token_mask)
                reads = torch.zeros(
                    batch_size,
                    seq_len,
                    self.state_read_dim,
                    device=hidden_states.device,
                    dtype=self.memory_v_proj.dtype,
                )
            else:
                if self.memory_readout_mode == "direct_last_hidden":
                    reads = self._direct_last_hidden_token_reads(hidden_states)
                elif self.memory_readout_mode == "projected_last_hidden":
                    reads = self._projected_last_hidden_token_reads(hidden_states)
                else:
                    reads = self._projected_kv_slot_token_reads(hidden_states)
            stats_beta = hidden_states.new_zeros((batch_size, seq_len, 1, 1))
            stats_lambda = torch.zeros_like(stats_beta)
            stats_mask = token_mask
            if self.memory_readout_mode != "projected_kv_slots":
                self.last_write_routes = None
                self.last_read_routes = None
                self.last_read_route_logits = None
            elif self.write_enabled:
                self.last_read_routes = None
                self.last_read_route_logits = None
            else:
                self.last_write_routes = None
        else:
            token_memory_q_seq, token_memory_k_seq, token_memory_v_seq, beta_seq, lambda_seq = (
                self._memory_sequence_projections(hidden_states)
            )
            token_memory_q_seq, token_memory_k_seq, token_memory_v_seq = self._partition_memory_projections(
                token_memory_q_seq,
                token_memory_k_seq,
                token_memory_v_seq,
            )
            write_route_seq, read_route_seq = self._memory_partition_routes(hidden_states, token_mask)
            stats_beta = beta_seq
            stats_lambda = lambda_seq
            stats_mask = token_mask
            if self.num_memory_partitions > 1:
                self.last_write_routes = write_route_seq
                self.last_read_routes = read_route_seq
            else:
                self.last_write_routes = None
                self.last_read_routes = None
            self.last_read_route_logits = None
            if self.write_enabled:
                state_before_write = state
                write_hidden = None
                write_mask = None
                write_message_ids = None
                if self.memory_write_granularity == "message_mean":
                    write_hidden, write_mask, write_message_ids = self._build_message_write_means(
                        hidden_states,
                        token_mask,
                    )
                elif self.memory_write_granularity == "sentence_mean":
                    write_hidden, write_mask, write_message_ids = self._build_sentence_write_means(
                        hidden_states,
                        token_mask,
                    )
                if write_hidden is not None and write_mask is not None:
                    write_memory_q_seq, write_memory_k_seq, write_memory_v_seq, stats_beta, stats_lambda = (
                        self._memory_sequence_projections(write_hidden)
                    )
                    write_memory_q_seq, write_memory_k_seq, write_memory_v_seq = self._partition_memory_projections(
                        write_memory_q_seq,
                        write_memory_k_seq,
                        write_memory_v_seq,
                    )
                    proposal_write_route_seq, proposal_read_route_seq = self._memory_partition_routes(
                        write_hidden,
                        write_mask,
                        message_ids=write_message_ids,
                    )
                    if self.num_memory_partitions > 1:
                        self.last_write_routes = proposal_write_route_seq
                        self.last_read_routes = proposal_read_route_seq
                    state, _ = self._memory_backend_scan(
                        state,
                        write_memory_q_seq,
                        write_memory_k_seq,
                        write_memory_v_seq,
                        stats_beta,
                        stats_lambda,
                        write_route_seq=proposal_write_route_seq if self.num_memory_partitions > 1 else None,
                        read_route_seq=proposal_read_route_seq if self.num_memory_partitions > 1 else None,
                        token_mask=write_mask,
                    )
                    reads = self._memory_backend_token_reads(
                        state_before_write,
                        token_memory_v_seq if self.memory_backend == "rwkv_ms" else token_memory_q_seq,
                        read_route_seq,
                        token_mask,
                    )
                    stats_mask = write_mask
                else:
                    state, reads = self._memory_backend_scan(
                        state,
                        token_memory_q_seq,
                        token_memory_k_seq,
                        token_memory_v_seq,
                        beta_seq,
                        lambda_seq,
                        write_route_seq=write_route_seq if self.num_memory_partitions > 1 else None,
                        read_route_seq=read_route_seq if self.num_memory_partitions > 1 else None,
                        token_mask=token_mask,
                    )
            else:
                reads = self._memory_backend_token_reads(
                    state,
                    token_memory_v_seq if self.memory_backend == "rwkv_ms" else token_memory_q_seq,
                    read_route_seq,
                    token_mask,
                )
        if read_mask is not None:
            reads = reads * read_mask.unsqueeze(-1).to(dtype=reads.dtype)
            if (
                self.last_read_routes is not None
                and self.last_read_routes.shape[:2] == read_mask.shape
            ):
                self.last_read_routes = self.last_read_routes * read_mask.unsqueeze(-1).to(
                    dtype=self.last_read_routes.dtype
                )
        self.delta_state = state
        self.last_beta_mean = self._masked_gate_mean(stats_beta, stats_mask)
        self.last_lambda_mean = self._masked_gate_mean(stats_lambda, stats_mask)
        delta_q, delta_k, delta_v = self._compute_delta_qkv_from_reads(reads)
        delta_o = self._project_delta_head(reads, self.delta_o_proj, "o")
        shared_qo_fusion_gate = None
        if self.memory_fusion_mode == "content_gated_qo_add":
            shared_qo_fusion_gate = self._memory_fusion_gate(hidden_states, reads)
            if delta_q is not None:
                delta_q = delta_q * shared_qo_fusion_gate.to(dtype=delta_q.dtype)

        if self.memory_fusion_placement in MEMORY_FUSION_NORM_HOOK_PLACEMENTS:
            base_kwargs = dict(kwargs)
            if cache_position is not None:
                base_kwargs["cache_position"] = cache_position
            base_o_output, attn_weights = self.base(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                shared_kv_states=shared_kv_states,
                past_key_values=past_key_values,
                **base_kwargs,
            )
            return (
                self._fuse_delta_o_output(
                    base_o_output,
                    delta_o,
                    hidden_states,
                    reads,
                    read_mask,
                    shared_qo_fusion_gate,
                ),
                attn_weights,
            )

        if self.is_gemma4_attention and self.is_kv_shared_layer:
            return self._forward_gemma4_shared_kv_attention(
                hidden_states,
                position_embeddings,
                attention_mask,
                shared_kv_states,
                delta_q,
                delta_o,
                reads,
                read_mask,
                fusion_gate=shared_qo_fusion_gate,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base.head_dim)

        query_states, key_states, value_states, output_gate = self._apply_delta_qkv(
            hidden_states,
            delta_q,
            delta_k,
            delta_v,
            read_mask,
        )

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        cos, sin = position_embeddings
        query_states = self._normalize_query_states(query_states).transpose(1, 2)
        key_states = self._normalize_key_states(key_states).transpose(1, 2)
        value_states = self._normalize_value_states(value_states).transpose(1, 2)
        query_states, key_states = self._apply_standard_rotary(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_values is not None:
            if self.is_gemma4_attention or self.is_qwen3_5_attention:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.base.layer_idx,
                )
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.base.layer_idx,
                    cache_kwargs,
                )
        if self.is_gemma4_attention and self.store_full_length_kv:
            if shared_kv_states is None:
                shared_kv_states = {}
            shared_kv_states[self.layer_type] = key_states, value_states

        attention_interface = self.eager_attention_forward
        if self.base.config._attn_implementation != "eager":
            if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.base.config._attn_implementation,
                    self.eager_attention_forward,
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.base.config._attn_implementation
                ]

        attn_kwargs = dict(kwargs)
        if self.sliding_window is not None:
            attn_kwargs["sliding_window"] = self.sliding_window
        attn_output, attn_weights = attention_interface(
            self.base,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base.attention_dropout,
            scaling=self.base.scaling,
            **attn_kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if output_gate is not None:
            attn_output = attn_output * torch.sigmoid(output_gate)
        base_o_output = self.base.o_proj(attn_output)
        return (
            self._fuse_delta_o_output(
                base_o_output,
                delta_o,
                hidden_states,
                reads,
                read_mask,
                shared_qo_fusion_gate,
            ),
            attn_weights,
        )


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def validate_gemma4_shared_delta_heads(
    module: nn.Module,
    config: HFDeltaMemConfig,
) -> None:
    if not (
        Gemma4TextAttention is not None
        and isinstance(module, Gemma4TextAttention)
        and getattr(module, "is_kv_shared_layer", False)
    ):
        return
    unsupported_heads = sorted(set(config.delta_heads) - {"q", "o"})
    if unsupported_heads:
        raise ValueError(
            "Gemma4 KV-shared attention layers support only Q/O Delta-Mem; "
            f"layer {module.layer_idx} requested unsupported delta heads: {unsupported_heads}"
        )


def validate_memory_fusion_placement_target(
    module: nn.Module,
    parent: nn.Module,
    attribute: str,
    config: HFDeltaMemConfig,
    *,
    module_name: str,
) -> nn.Module | None:
    if config.memory_fusion_placement == "attention_output":
        return None
    if Gemma4TextAttention is None or not isinstance(module, Gemma4TextAttention):
        raise ValueError(
            f"memory_fusion_placement={config.memory_fusion_placement!r} is currently "
            "Gemma4-only; "
            f"unsupported target: {module_name} ({type(module).__name__})"
        )
    if attribute != "self_attn":
        raise ValueError(
            f"{config.memory_fusion_placement} fusion requires a Gemma decoder self_attn "
            "target; "
            f"got {module_name}"
        )
    if Gemma4TextDecoderLayer is None or not isinstance(parent, Gemma4TextDecoderLayer):
        raise ValueError(
            f"{config.memory_fusion_placement} fusion requires the attention's parent to be a "
            f"Gemma4TextDecoderLayer; got {type(parent).__name__} for {module_name}"
        )
    if set(config.delta_heads) != {"o"}:
        raise ValueError(
            f"{config.memory_fusion_placement} fusion currently requires O-only Delta-Mem "
            "heads; "
            f"got {config.delta_heads} for {module_name}"
        )
    layernorm = getattr(parent, "post_attention_layernorm", None)
    if not isinstance(layernorm, nn.Module):
        raise ValueError(
            f"{config.memory_fusion_placement} fusion requires a "
            "post_attention_layernorm module; "
            f"missing from the parent of {module_name}"
        )
    return layernorm


def attach_delta_mem(model: nn.Module, config: HFDeltaMemConfig) -> list[str]:
    candidates: list[tuple[str, nn.Module, nn.Module, str, nn.Module | None]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, SUPPORTED_BASE_ATTENTION_TYPES):
            continue
        if name.split(".")[-1] not in config.target_modules:
            continue
        if config.target_layers and module.layer_idx not in config.target_layers:
            continue
        if (
            Gemma4TextAttention is not None
            and isinstance(module, Gemma4TextAttention)
            and getattr(module, "is_kv_shared_layer", False)
            and not config.target_layers
        ):
            continue
        validate_gemma4_shared_delta_heads(module, config)
        parent, attr = _get_parent_module(model, name)
        layernorm = validate_memory_fusion_placement_target(
            module,
            parent,
            attr,
            config,
            module_name=name,
        )
        candidates.append((name, module, parent, attr, layernorm))

    if not candidates:
        raise RuntimeError("No target modules were replaced")

    installed: list[tuple[nn.Module, str, nn.Module, DeltaMemAttention]] = []
    try:
        for name, module, parent, attr, layernorm in candidates:
            module = ensure_attention_compat_views(module)
            wrapped = DeltaMemAttention(module, config).to(
                device=module.q_proj.weight.device,
                dtype=module.q_proj.weight.dtype,
            )
            setattr(parent, attr, wrapped)
            installed.append((parent, attr, module, wrapped))
            if layernorm is not None:
                wrapped.bind_post_attention_layernorm(layernorm)
    except Exception:
        for parent, attr, original, wrapped in reversed(installed):
            wrapped.remove_post_attention_layernorm_hook()
            setattr(parent, attr, original)
        raise
    return [name for name, *_ in candidates]


def reset_delta_mem_states(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, DeltaMemAttention):
            module.reset_state()


def iter_delta_mem_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, DeltaMemAttention):
            yield name, module


def set_delta_mem_write_enabled(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_enabled(enabled)


def set_delta_mem_write_message_ids(
    model: nn.Module,
    message_ids: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_message_ids(message_ids)


def set_delta_mem_write_sentence_ids(
    model: nn.Module,
    sentence_ids: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_sentence_ids(sentence_ids)


def set_delta_mem_projected_kv_write_spans(
    model: nn.Module,
    key_mask: torch.Tensor | None,
    value_mask: torch.Tensor | None,
    slot_indices: torch.Tensor | None = None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_projected_kv_write_spans(
            key_mask,
            value_mask,
            slot_indices,
        )


def set_delta_mem_projected_kv_read_query_mask(
    model: nn.Module,
    query_mask: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_projected_kv_read_query_mask(query_mask)


def set_delta_mem_read_context_mask(
    model: nn.Module,
    token_mask: torch.Tensor | None,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.read_context_mask = token_mask


def set_delta_mem_read_representation_capture_mask(
    model: nn.Module,
    token_mask: torch.Tensor | None,
) -> None:
    if token_mask is not None:
        if token_mask.ndim != 2:
            raise ValueError(
                "Read representation capture mask must have shape [batch, sequence]"
            )
        selected_per_row = token_mask.to(dtype=torch.bool).sum(dim=1)
        if not torch.equal(selected_per_row, torch.ones_like(selected_per_row)):
            raise ValueError(
                "Read representation capture mask must select exactly one token per batch row"
            )
    for _, module in iter_delta_mem_modules(model):
        module.read_representation_capture_mask = token_mask
        module.last_read_representation = None


def collect_delta_mem_read_representations(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    representations: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        if module.read_representation_capture_mask is None:
            continue
        representation = module.last_read_representation
        if representation is None:
            raise RuntimeError(
                f"Delta-Mem module {name!r} has a capture mask but no read representation"
            )
        if representation.ndim != 2:
            raise RuntimeError(
                f"Delta-Mem module {name!r} produced an invalid read representation shape: "
                f"{tuple(representation.shape)}"
            )
        representations[name] = representation
    return representations


def collect_delta_mem_projected_kv_read_logits(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    logits_by_module: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        if module.memory_readout_mode not in PROJECTED_KV_MEMORY_READOUT_MODES:
            continue
        logits = module.last_read_route_logits
        if logits is None:
            continue
        expected_slots = module.rwkv_ms_num_states
        if logits.ndim != 3 or logits.size(-1) != expected_slots:
            raise RuntimeError(
                f"Delta-Mem module {name!r} produced invalid projected-KV read logits: "
                f"shape={tuple(logits.shape)} expected=[batch, sequence, {expected_slots}]"
            )
        logits_by_module[name] = logits
    return logits_by_module


def get_delta_mem_write_regularization(
    model: nn.Module,
    *,
    target: float = 0.0,
) -> torch.Tensor:
    penalties = []
    for _, module in iter_delta_mem_modules(model):
        if module.last_beta_mean is None:
            continue
        penalties.append((module.last_beta_mean - target).pow(2))
    if penalties:
        return torch.stack(penalties).mean()
    try:
        reference = next(model.parameters())
        return reference.new_zeros(())
    except StopIteration:
        return torch.zeros(())


def collect_delta_mem_gate_stats(model: nn.Module) -> dict[str, float]:
    stats = {
        "num_modules": 0,
        "beta_mean": 0.0,
        "lambda_mean": 0.0,
        "rankwise_gate_modules": 0,
    }
    for _, module in iter_delta_mem_modules(model):
        stats["num_modules"] += 1
        if module.rankwise_gates:
            stats["rankwise_gate_modules"] += 1
        if module.last_beta_mean is not None:
            stats["beta_mean"] += float(module.last_beta_mean.detach().float().item())
        if module.last_lambda_mean is not None:
            stats["lambda_mean"] += float(module.last_lambda_mean.detach().float().item())
    if stats["num_modules"] > 0:
        stats["beta_mean"] /= stats["num_modules"]
        stats["lambda_mean"] /= stats["num_modules"]
    return stats


def collect_delta_mem_weight_stats(model: nn.Module) -> dict[str, float]:
    stats: dict[str, float] = {
        "num_modules": 0,
        "memory_q_proj_norm_sum": 0.0,
        "memory_k_proj_norm_sum": 0.0,
        "memory_v_proj_norm_sum": 0.0,
        "delta_q_proj_norm_sum": 0.0,
        "delta_k_proj_norm_sum": 0.0,
        "delta_v_proj_norm_sum": 0.0,
        "delta_o_proj_norm_sum": 0.0,
        "memory_fusion_weight_norm_sum": 0.0,
        "content_gated_fusion_modules": 0,
        "post_attention_residual_hybrid_modules": 0,
        "memory_fusion_residual_gain_sum": 0.0,
        "delta_scale_mean_sum": 0.0,
        "trainable_delta_scale_modules": 0,
        "beta_proj_norm_sum": 0.0,
        "beta_bias_mean_sum": 0.0,
        "hrm_rwkv7_core_norm_sum": 0.0,
    }
    for _, module in iter_delta_mem_modules(model):
        stats["num_modules"] += 1
        stats["memory_q_proj_norm_sum"] += module.memory_q_proj.float().norm().item()
        stats["memory_k_proj_norm_sum"] += module.memory_k_proj.float().norm().item()
        stats["memory_v_proj_norm_sum"] += module.memory_v_proj.float().norm().item()
        stats["delta_q_proj_norm_sum"] += module.delta_q_proj.float().norm().item()
        stats["delta_k_proj_norm_sum"] += module.delta_k_proj.float().norm().item()
        stats["delta_v_proj_norm_sum"] += module.delta_v_proj.float().norm().item()
        stats["delta_o_proj_norm_sum"] += module.delta_o_proj.float().norm().item()
        if module.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES:
            stats["content_gated_fusion_modules"] += 1
            stats["memory_fusion_weight_norm_sum"] += (
                module.memory_fusion_hidden_weight.float().norm().item()
                + module.memory_fusion_read_weight.float().norm().item()
            )
        if module.memory_fusion_placement == "post_attention_residual_hybrid":
            stats["post_attention_residual_hybrid_modules"] += 1
            stats["memory_fusion_residual_gain_sum"] += float(
                module.memory_fusion_residual_gain_raw.detach()
                .float()
                .clamp(min=0.0, max=module.memory_fusion_residual_scale_max)
                .item()
            )
        if module.trainable_delta_scale:
            stats["trainable_delta_scale_modules"] += 1
            stats["delta_scale_mean_sum"] += (
                torch.sigmoid(module.delta_scale_raw.float()).mean().item() * module.delta_scale_max
            )
        stats["beta_proj_norm_sum"] += module.beta_proj.float().norm().item()
        stats["beta_bias_mean_sum"] += module.beta_bias.float().mean().item()
        if module.hrm_rwkv7_core is not None:
            stats["hrm_rwkv7_core_norm_sum"] += sum(
                param.detach().float().norm().item()
                for param in module.hrm_rwkv7_core.parameters()
            )
    return stats


def snapshot_delta_mem_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        for sub_name, param in module.named_parameters():
            if sub_name.startswith("base."):
                continue
            snapshot[f"{name}.{sub_name}"] = param.detach().float().cpu().clone()
    return snapshot


def diff_delta_mem_snapshots(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, float]:
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    for key, before_tensor in before.items():
        diff = (after[key] - before_tensor).abs()
        max_abs_diff = max(max_abs_diff, diff.max().item())
        total_abs_diff += diff.sum().item()
    return {
        "max_abs_diff": max_abs_diff,
        "total_abs_diff": total_abs_diff,
    }


def collect_delta_mem_state_stats(model: nn.Module) -> dict[str, float]:
    num_modules = 0
    nonzero_modules = 0
    max_state_norm = 0.0
    mean_state_norm = 0.0
    max_state_abs = 0.0
    for _, module in iter_delta_mem_modules(model):
        num_modules += 1
        if module.delta_state is None:
            continue
        state = module.delta_state.float()
        state_norm = state.norm().item()
        mean_state_norm += state_norm
        max_state_norm = max(max_state_norm, state_norm)
        max_state_abs = max(max_state_abs, state.abs().max().item())
        if state.abs().max().item() > 0:
            nonzero_modules += 1
    if num_modules > 0:
        mean_state_norm /= num_modules
    return {
        "num_modules": num_modules,
        "nonzero_modules": nonzero_modules,
        "max_state_norm": max_state_norm,
        "mean_state_norm": mean_state_norm,
        "max_state_abs": max_state_abs,
    }


def collect_delta_mem_output_ratio_stats(model: nn.Module) -> dict[str, float]:
    num_modules = 0
    modules_with_delta_o = 0
    content_gated_modules = 0
    attention_output_fusion_modules = 0
    post_attention_norm_fusion_modules = 0
    normalized_residual_correction_fusion_modules = 0
    post_attention_residual_hybrid_fusion_modules = 0
    mean_base_o_norm = 0.0
    mean_delta_o_norm = 0.0
    mean_delta_o_ratio = 0.0
    max_delta_o_ratio = 0.0
    mean_delta_o_gate = 0.0
    min_delta_o_gate = math.inf
    max_delta_o_gate = -math.inf
    mean_delta_o_gate_lt_001_fraction = 0.0
    mean_delta_o_gate_gt_099_fraction = 0.0
    mean_fused_delta_o_norm = 0.0
    mean_fused_delta_o_ratio = 0.0
    max_fused_delta_o_ratio = 0.0
    mean_delta_o_base_cosine = 0.0
    mean_fused_o_ratio = 0.0
    mean_applied_memory_correction_norm = 0.0
    mean_applied_memory_correction_ratio = 0.0
    max_applied_memory_correction_ratio = 0.0
    mean_memory_residual_norm = 0.0
    mean_memory_residual_ratio = 0.0
    max_memory_residual_ratio = 0.0
    modules_with_memory_residual_gain = 0
    mean_memory_residual_gain = 0.0
    min_memory_residual_gain = math.inf
    max_memory_residual_gain = -math.inf
    for _, module in iter_delta_mem_modules(model):
        num_modules += 1
        if module.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES:
            content_gated_modules += 1
        if module.memory_fusion_placement == "attention_output":
            attention_output_fusion_modules += 1
        elif module.memory_fusion_placement == "post_attention_norm":
            post_attention_norm_fusion_modules += 1
        elif module.memory_fusion_placement == "normalized_residual_correction":
            normalized_residual_correction_fusion_modules += 1
        elif module.memory_fusion_placement == "post_attention_residual_hybrid":
            post_attention_residual_hybrid_fusion_modules += 1
        if module.last_base_o_norm is not None:
            mean_base_o_norm += float(module.last_base_o_norm.detach().float().item())
        if module.last_delta_o_norm is not None:
            modules_with_delta_o += 1
            mean_delta_o_norm += float(module.last_delta_o_norm.detach().float().item())
        if module.last_delta_o_ratio is not None:
            ratio = float(module.last_delta_o_ratio.detach().float().item())
            mean_delta_o_ratio += ratio
            max_delta_o_ratio = max(max_delta_o_ratio, ratio)
        if module.last_delta_o_gate_mean is not None:
            mean_delta_o_gate += float(module.last_delta_o_gate_mean.detach().float().item())
            min_delta_o_gate = min(
                min_delta_o_gate,
                float(module.last_delta_o_gate_min.detach().float().item()),
            )
            max_delta_o_gate = max(
                max_delta_o_gate,
                float(module.last_delta_o_gate_max.detach().float().item()),
            )
            mean_delta_o_gate_lt_001_fraction += float(
                module.last_delta_o_gate_lt_001_fraction.detach().float().item()
            )
            mean_delta_o_gate_gt_099_fraction += float(
                module.last_delta_o_gate_gt_099_fraction.detach().float().item()
            )
        if module.last_fused_delta_o_norm is not None:
            mean_fused_delta_o_norm += float(
                module.last_fused_delta_o_norm.detach().float().item()
            )
        if module.last_fused_delta_o_ratio is not None:
            fused_ratio = float(module.last_fused_delta_o_ratio.detach().float().item())
            mean_fused_delta_o_ratio += fused_ratio
            max_fused_delta_o_ratio = max(max_fused_delta_o_ratio, fused_ratio)
        if module.last_delta_o_base_cosine is not None:
            mean_delta_o_base_cosine += float(
                module.last_delta_o_base_cosine.detach().float().item()
            )
        if module.last_fused_o_ratio is not None:
            mean_fused_o_ratio += float(module.last_fused_o_ratio.detach().float().item())
        if module.last_applied_memory_correction_norm is not None:
            mean_applied_memory_correction_norm += float(
                module.last_applied_memory_correction_norm.detach().float().item()
            )
        if module.last_applied_memory_correction_ratio is not None:
            applied_correction_ratio = float(
                module.last_applied_memory_correction_ratio.detach().float().item()
            )
            mean_applied_memory_correction_ratio += applied_correction_ratio
            max_applied_memory_correction_ratio = max(
                max_applied_memory_correction_ratio,
                applied_correction_ratio,
            )
        if module.last_memory_residual_norm is not None:
            mean_memory_residual_norm += float(
                module.last_memory_residual_norm.detach().float().item()
            )
        if module.last_memory_residual_ratio is not None:
            memory_residual_ratio = float(
                module.last_memory_residual_ratio.detach().float().item()
            )
            mean_memory_residual_ratio += memory_residual_ratio
            max_memory_residual_ratio = max(
                max_memory_residual_ratio,
                memory_residual_ratio,
            )
        if module.last_memory_residual_gain is not None:
            modules_with_memory_residual_gain += 1
            residual_gain = float(
                module.last_memory_residual_gain.detach().float().item()
            )
            mean_memory_residual_gain += residual_gain
            min_memory_residual_gain = min(min_memory_residual_gain, residual_gain)
            max_memory_residual_gain = max(max_memory_residual_gain, residual_gain)
    if num_modules > 0:
        mean_base_o_norm /= num_modules
    if modules_with_delta_o > 0:
        mean_delta_o_norm /= modules_with_delta_o
        mean_delta_o_ratio /= modules_with_delta_o
        mean_delta_o_gate /= modules_with_delta_o
        mean_delta_o_gate_lt_001_fraction /= modules_with_delta_o
        mean_delta_o_gate_gt_099_fraction /= modules_with_delta_o
        mean_fused_delta_o_norm /= modules_with_delta_o
        mean_fused_delta_o_ratio /= modules_with_delta_o
        mean_delta_o_base_cosine /= modules_with_delta_o
        mean_fused_o_ratio /= modules_with_delta_o
        mean_applied_memory_correction_norm /= modules_with_delta_o
        mean_applied_memory_correction_ratio /= modules_with_delta_o
        mean_memory_residual_norm /= modules_with_delta_o
        mean_memory_residual_ratio /= modules_with_delta_o
    if modules_with_memory_residual_gain > 0:
        mean_memory_residual_gain /= modules_with_memory_residual_gain
    else:
        min_memory_residual_gain = 0.0
        max_memory_residual_gain = 0.0
    if modules_with_delta_o == 0:
        min_delta_o_gate = 0.0
        max_delta_o_gate = 0.0
    result = {
        "num_modules": num_modules,
        "modules_with_delta_o": modules_with_delta_o,
        "content_gated_modules": content_gated_modules,
        "attention_output_fusion_modules": attention_output_fusion_modules,
        "post_attention_norm_fusion_modules": post_attention_norm_fusion_modules,
        "normalized_residual_correction_fusion_modules": (
            normalized_residual_correction_fusion_modules
        ),
        "post_attention_residual_hybrid_fusion_modules": (
            post_attention_residual_hybrid_fusion_modules
        ),
        "mean_base_o_norm": mean_base_o_norm,
        "mean_delta_o_norm": mean_delta_o_norm,
        "mean_delta_o_ratio": mean_delta_o_ratio,
        "max_delta_o_ratio": max_delta_o_ratio,
        "mean_delta_o_gate": mean_delta_o_gate,
        "min_delta_o_gate": min_delta_o_gate,
        "max_delta_o_gate": max_delta_o_gate,
        "mean_delta_o_gate_lt_001_fraction": mean_delta_o_gate_lt_001_fraction,
        "mean_delta_o_gate_gt_099_fraction": mean_delta_o_gate_gt_099_fraction,
        "mean_fused_delta_o_norm": mean_fused_delta_o_norm,
        "mean_fused_delta_o_ratio": mean_fused_delta_o_ratio,
        "max_fused_delta_o_ratio": max_fused_delta_o_ratio,
        "mean_delta_o_base_cosine": mean_delta_o_base_cosine,
        "mean_fused_o_ratio": mean_fused_o_ratio,
        "mean_applied_memory_correction_norm": mean_applied_memory_correction_norm,
        "mean_applied_memory_correction_ratio": mean_applied_memory_correction_ratio,
        "max_applied_memory_correction_ratio": max_applied_memory_correction_ratio,
        "mean_memory_residual_norm": mean_memory_residual_norm,
        "mean_memory_residual_ratio": mean_memory_residual_ratio,
        "max_memory_residual_ratio": max_memory_residual_ratio,
        "modules_with_memory_residual_gain": modules_with_memory_residual_gain,
        "mean_memory_residual_gain": mean_memory_residual_gain,
        "min_memory_residual_gain": min_memory_residual_gain,
        "max_memory_residual_gain": max_memory_residual_gain,
    }
    modules = [module for _, module in iter_delta_mem_modules(model)]
    for sharing in ("nonshared", "shared"):
        for attention_kind in ("local", "full"):
            group_name = f"{sharing}_{attention_kind}"
            group_modules = [
                module
                for module in modules
                if ("shared" if module.is_kv_shared_layer else "nonshared") == sharing
                and (
                    "local"
                    if module.is_sliding
                    or module.layer_type in {"sliding_attention", "local_attention"}
                    else "full"
                )
                == attention_kind
            ]
            active_modules = [
                module
                for module in group_modules
                if module.last_fused_delta_o_ratio is not None
            ]
            result[f"{group_name}_modules"] = len(group_modules)
            result[f"{group_name}_active_modules"] = len(active_modules)
            for metric_name, attribute in (
                ("mean_fused_delta_o_ratio", "last_fused_delta_o_ratio"),
                ("mean_fused_o_ratio", "last_fused_o_ratio"),
                (
                    "mean_applied_memory_correction_ratio",
                    "last_applied_memory_correction_ratio",
                ),
                ("mean_delta_o_base_cosine", "last_delta_o_base_cosine"),
                ("mean_delta_o_gate", "last_delta_o_gate_mean"),
                (
                    "mean_delta_o_gate_lt_001_fraction",
                    "last_delta_o_gate_lt_001_fraction",
                ),
                (
                    "mean_delta_o_gate_gt_099_fraction",
                    "last_delta_o_gate_gt_099_fraction",
                ),
            ):
                values = [
                    float(getattr(module, attribute).detach().float().item())
                    for module in active_modules
                    if getattr(module, attribute) is not None
                ]
                result[f"{group_name}_{metric_name}"] = (
                    sum(values) / len(values) if values else 0.0
                )
            fused_ratios = [
                float(module.last_fused_delta_o_ratio.detach().float().item())
                for module in active_modules
            ]
            result[f"{group_name}_max_fused_delta_o_ratio"] = (
                max(fused_ratios) if fused_ratios else 0.0
            )
    return result


def collect_delta_mem_partition_route_stats(model: nn.Module) -> dict[str, float]:
    stats = {
        "enabled_modules": 0,
        "tied_read_write_modules": 0,
        "active_modules": 0,
        "write_route_entropy": 0.0,
        "read_route_entropy": 0.0,
        "route_alignment_mse": 0.0,
        "route_overlap": 0.0,
        "write_route_max": 0.0,
        "read_route_max": 0.0,
        "write_route_balance_l2": 0.0,
        "read_route_balance_l2": 0.0,
    }
    for _, module in iter_delta_mem_modules(model):
        if module.num_memory_partitions <= 1:
            continue
        stats["enabled_modules"] += 1
        if module.tie_memory_partition_read_write:
            stats["tied_read_write_modules"] += 1
        if module.last_write_routes is None or module.last_read_routes is None:
            continue
        stats["active_modules"] += 1
        write_routes = module.last_write_routes.detach().float()
        read_routes = module.last_read_routes.detach().float()
        write_entropy = -(write_routes * write_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        read_entropy = -(read_routes * read_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        uniform = write_routes.new_full(
            (module.num_memory_partitions,),
            1.0 / module.num_memory_partitions,
        )
        write_usage = write_routes.mean(dim=(0, 1))
        read_usage = read_routes.mean(dim=(0, 1))
        stats["write_route_entropy"] += float(write_entropy.item())
        stats["read_route_entropy"] += float(read_entropy.item())
        stats["route_alignment_mse"] += float((write_routes - read_routes).pow(2).mean().item())
        stats["route_overlap"] += float((write_routes * read_routes).sum(dim=-1).mean().item())
        stats["write_route_max"] += float(write_routes.max(dim=-1).values.mean().item())
        stats["read_route_max"] += float(read_routes.max(dim=-1).values.mean().item())
        stats["write_route_balance_l2"] += float(((write_usage - uniform).pow(2)).mean().item())
        stats["read_route_balance_l2"] += float(((read_usage - uniform).pow(2)).mean().item())
    if stats["active_modules"] > 0:
        for key in (
            "write_route_entropy",
            "read_route_entropy",
            "route_alignment_mse",
            "route_overlap",
            "write_route_max",
            "read_route_max",
            "write_route_balance_l2",
            "read_route_balance_l2",
        ):
            stats[key] /= stats["active_modules"]
    return stats


def get_delta_mem_partition_regularization(model: nn.Module) -> dict[str, torch.Tensor]:
    alignment_losses: list[torch.Tensor] = []
    entropy_losses: list[torch.Tensor] = []
    balance_losses: list[torch.Tensor] = []
    reference = None
    for _, module in iter_delta_mem_modules(model):
        if module.num_memory_partitions <= 1:
            continue
        if module.last_write_routes is None or module.last_read_routes is None:
            continue
        if reference is None:
            reference = module.memory_q_proj
        write_routes = module.last_write_routes
        read_routes = module.last_read_routes
        alignment_losses.append((write_routes - read_routes).pow(2).mean())
        write_entropy = -(write_routes * write_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        read_entropy = -(read_routes * read_routes.clamp_min(1e-6).log()).sum(dim=-1).mean()
        entropy_losses.append((write_entropy + read_entropy) * 0.5)
        uniform = write_routes.new_full(
            (module.num_memory_partitions,),
            1.0 / module.num_memory_partitions,
        )
        write_usage = write_routes.mean(dim=(0, 1))
        read_usage = read_routes.mean(dim=(0, 1))
        balance_losses.append(
            (((write_usage - uniform).pow(2)).mean() + ((read_usage - uniform).pow(2)).mean()) * 0.5
        )
    if reference is None:
        try:
            reference = next(model.parameters())
        except StopIteration:
            zero = torch.zeros(())
            return {"alignment": zero, "entropy": zero, "balance": zero}
    zero = reference.new_zeros(())
    return {
        "alignment": torch.stack(alignment_losses).mean() if alignment_losses else zero,
        "entropy": torch.stack(entropy_losses).mean() if entropy_losses else zero,
        "balance": torch.stack(balance_losses).mean() if balance_losses else zero,
    }


def get_delta_mem_online_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        if module.delta_state is not None:
            state[name] = module.delta_state.detach().cpu().clone()
            if module.memory_backend == "rwkv_ms" and module.rwkv_ms_positions is not None:
                state[f"{name}.__rwkv_ms_positions"] = module.rwkv_ms_positions.detach().cpu().clone()
            if module.memory_backend == "rwkv_ms" and module.rwkv_ms_previous_source is not None:
                state[f"{name}.__rwkv_ms_previous_source"] = (
                    module.rwkv_ms_previous_source.detach().cpu().clone()
                )
        if module.direct_last_hidden is not None:
            state[f"{name}.__direct_last_hidden"] = (
                module.direct_last_hidden.detach().cpu().clone()
            )
        if module.projected_last_hidden is not None:
            state[f"{name}.__projected_last_hidden"] = (
                module.projected_last_hidden.detach().cpu().clone()
            )
        for suffix, tensor in (
            ("__projected_kv_keys", module.projected_kv_keys),
            ("__projected_kv_values", module.projected_kv_values),
            ("__projected_kv_occupied", module.projected_kv_occupied),
            ("__projected_kv_surprise", module.projected_kv_surprise),
        ):
            if tensor is not None:
                state[f"{name}.{suffix}"] = tensor.detach().cpu().clone()
    return state


def load_delta_mem_online_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    module_map = dict(model.named_modules())
    for name, tensor in state.items():
        projected_kv_suffixes = {
            ".__projected_kv_keys": ("projected_kv_keys", None),
            ".__projected_kv_values": ("projected_kv_values", None),
            ".__projected_kv_occupied": ("projected_kv_occupied", torch.bool),
            ".__projected_kv_surprise": ("projected_kv_surprise", torch.float32),
        }
        matched_projected_kv_suffix = next(
            (suffix for suffix in projected_kv_suffixes if name.endswith(suffix)),
            None,
        )
        if matched_projected_kv_suffix is not None:
            module_name = name[: -len(matched_projected_kv_suffix)]
            module = module_map[module_name]
            if not isinstance(module, DeltaMemAttention):
                raise TypeError(f"{module_name} is not a DeltaMemAttention")
            if module.memory_readout_mode not in PROJECTED_KV_MEMORY_READOUT_MODES:
                raise ValueError(
                    f"{module_name} does not use a projected-KV readout"
                )
            attribute, fixed_dtype = projected_kv_suffixes[
                matched_projected_kv_suffix
            ]
            if fixed_dtype is None:
                fixed_dtype = (
                    module.projected_kv_key_proj.dtype
                    if attribute == "projected_kv_keys"
                    else module.memory_v_proj.dtype
                )
            setattr(
                module,
                attribute,
                tensor.to(
                    device=module.memory_v_proj.device,
                    dtype=fixed_dtype,
                ),
            )
            continue
        if name.endswith(".__projected_last_hidden"):
            module_name = name[: -len(".__projected_last_hidden")]
            module = module_map[module_name]
            if not isinstance(module, DeltaMemAttention):
                raise TypeError(f"{module_name} is not a DeltaMemAttention")
            module.projected_last_hidden = tensor.to(
                device=module.memory_v_proj.device,
                dtype=module.memory_v_proj.dtype,
            )
            continue
        if name.endswith(".__direct_last_hidden"):
            module_name = name[: -len(".__direct_last_hidden")]
            module = module_map[module_name]
            if not isinstance(module, DeltaMemAttention):
                raise TypeError(f"{module_name} is not a DeltaMemAttention")
            module.direct_last_hidden = tensor.to(
                device=module.base.q_proj.weight.device,
                dtype=module.base.q_proj.weight.dtype,
            )
            continue
        if name.endswith(".__rwkv_ms_positions"):
            module_name = name[: -len(".__rwkv_ms_positions")]
            module = module_map[module_name]
            if not isinstance(module, DeltaMemAttention):
                raise TypeError(f"{module_name} is not a DeltaMemAttention")
            module.rwkv_ms_positions = tensor.to(
                device=module.base.q_proj.weight.device,
                dtype=torch.long,
            )
            continue
        if name.endswith(".__rwkv_ms_previous_source"):
            module_name = name[: -len(".__rwkv_ms_previous_source")]
            module = module_map[module_name]
            if not isinstance(module, DeltaMemAttention):
                raise TypeError(f"{module_name} is not a DeltaMemAttention")
            # The predecessor comes from the projected RWKV source, whose live
            # dtype can differ from the frozen base attention weights.
            module.rwkv_ms_previous_source = tensor.to(
                device=module.base.q_proj.weight.device,
            )
            continue
        module = module_map[name]
        if not isinstance(module, DeltaMemAttention):
            raise TypeError(f"{name} is not a DeltaMemAttention")
        state_dtype = (
            torch.float32
            if module.memory_backend == "rwkv_ms"
            else module.base.q_proj.weight.dtype
        )
        module.delta_state = tensor.to(
            device=module.base.q_proj.weight.device,
            dtype=state_dtype,
        )


def freeze_non_delta_mem_params(model: nn.Module) -> list[str]:
    trainable = []
    for param in model.parameters():
        param.requires_grad = False
    for name, module in iter_delta_mem_modules(model):
        for sub_name, param in module.named_parameters():
            if sub_name.startswith("base."):
                param.requires_grad = False
                continue
            trainable_flag = module.is_trainable_parameter(sub_name)
            param.requires_grad = trainable_flag
            if trainable_flag:
                trainable.append(f"{name}.{sub_name}")
    return trainable


def get_delta_mem_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for name, module in iter_delta_mem_modules(model):
        for sub_name, param in module.named_parameters():
            if sub_name.startswith("base."):
                continue
            state_dict[f"{name}.{sub_name}"] = param.detach().cpu().clone()
    return state_dict


def load_delta_mem_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    initialize_missing_residual_hybrid_gain: bool = False,
    initialize_missing_content_gate: bool = False,
) -> None:
    expected_state = get_delta_mem_state_dict(model)
    expected_keys = list(expected_state)
    actual_keys = list(state_dict)
    missing = [key for key in expected_keys if key not in state_dict]
    extra = [key for key in actual_keys if key not in expected_state]
    module_map = dict(model.named_modules())
    residual_hybrid_gain_missing = []
    content_gate_missing = []
    for key in missing:
        module_name, param_name = key.rsplit(".", 1)
        module = module_map.get(module_name)
        if (
            param_name == "memory_fusion_residual_gain_raw"
            and isinstance(module, DeltaMemAttention)
            and module.memory_fusion_placement == "post_attention_residual_hybrid"
        ):
            residual_hybrid_gain_missing.append(key)
        if (
            param_name
            in {
                "memory_fusion_hidden_weight",
                "memory_fusion_read_weight",
                "memory_fusion_bias",
            }
            and isinstance(module, DeltaMemAttention)
            and module.memory_fusion_mode in CONTENT_GATED_MEMORY_FUSION_MODES
        ):
            content_gate_missing.append(key)
    allowed_missing: set[str] = set()
    if initialize_missing_residual_hybrid_gain:
        allowed_missing.update(residual_hybrid_gain_missing)
    if initialize_missing_content_gate:
        allowed_missing.update(content_gate_missing)
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if disallowed_missing or extra:
        warm_start_hint = ""
        if (
            not initialize_missing_residual_hybrid_gain
            and missing
            and len(residual_hybrid_gain_missing) == len(missing)
            and not extra
        ):
            warm_start_hint = (
                " To warm-start these weights into post_attention_residual_hybrid, "
                "pass initialize_missing_residual_hybrid_gain=True."
            )
        elif (
            not initialize_missing_content_gate
            and missing
            and len(content_gate_missing) == len(missing)
            and not extra
        ):
            warm_start_hint = (
                " To warm-start these weights into a content-gated fusion mode, "
                "pass initialize_missing_content_gate=True."
            )
        raise ValueError(
            "Delta-Mem adapter parameter topology does not match the attached model; "
            f"missing={missing[:8]} extra={extra[:8]}.{warm_start_hint}"
        )
    for name in expected_keys:
        if name in allowed_missing:
            continue
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Delta-Mem adapter entry is not a tensor: {name}")
        if tensor.shape != expected_state[name].shape:
            raise ValueError(
                f"Delta-Mem adapter shape mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)} model={tuple(expected_state[name].shape)}"
            )
    for full_name, tensor in state_dict.items():
        module_name, param_name = full_name.rsplit(".", 1)
        module = module_map[module_name]
        param = getattr(module, param_name)
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def validate_attached_delta_config(
    model: nn.Module,
    config: HFDeltaMemConfig,
    *,
    allowed_config_mismatches: tuple[str, ...] = (),
) -> None:
    modules = list(iter_delta_mem_modules(model))
    if not modules:
        raise ValueError("Model has no attached Delta-Mem modules")
    allowed = set(allowed_config_mismatches)
    expected = config.to_dict()
    for name, module in modules:
        actual = module.delta_config.to_dict()
        mismatches = sorted(
            key
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key) and key not in allowed
        )
        if mismatches:
            raise ValueError(
                f"Attached Delta-Mem config does not match adapter config at {name}: "
                + ", ".join(mismatches)
            )
        hook_bound = module._post_attention_norm_hook_handle is not None
        expects_hook = module.memory_fusion_placement in MEMORY_FUSION_NORM_HOOK_PLACEMENTS
        if hook_bound != expects_hook:
            raise ValueError(
                f"Attached Delta-Mem fusion topology is invalid at {name}: "
                f"placement={module.memory_fusion_placement!r} hook_bound={hook_bound}"
            )


def save_delta_mem_adapter(
    model: nn.Module,
    output_dir: str | Path,
    config: HFDeltaMemConfig,
) -> None:
    output_path = Path(output_dir)
    validate_attached_delta_config(model, config)
    output_path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output_path)
    torch.save(get_delta_mem_state_dict(model), output_path / "delta_mem_adapter.pt")


def load_delta_mem_adapter(
    model: nn.Module,
    input_dir: str | Path,
    *,
    allowed_config_mismatches: tuple[str, ...] = (),
    initialize_missing_residual_hybrid_gain: bool = False,
    initialize_missing_content_gate: bool = False,
) -> HFDeltaMemConfig:
    input_path = Path(input_dir)
    config = HFDeltaMemConfig.from_pretrained(input_path)
    if initialize_missing_residual_hybrid_gain:
        modules = list(iter_delta_mem_modules(model))
        non_hybrid = [
            name
            for name, module in modules
            if module.memory_fusion_placement != "post_attention_residual_hybrid"
        ]
        if non_hybrid:
            raise ValueError(
                "initialize_missing_residual_hybrid_gain=True requires every attached "
                "Delta-Mem module to use post_attention_residual_hybrid; "
                f"non_hybrid={non_hybrid[:8]}"
            )
        allowed_config_mismatches = tuple(
            dict.fromkeys(
                (
                    *allowed_config_mismatches,
                    "memory_fusion_placement",
                    "memory_fusion_residual_scale",
                    "memory_fusion_residual_scale_max",
                )
            )
        )
    if initialize_missing_content_gate:
        modules = list(iter_delta_mem_modules(model))
        non_gated = [
            name
            for name, module in modules
            if module.memory_fusion_mode not in CONTENT_GATED_MEMORY_FUSION_MODES
        ]
        if non_gated:
            raise ValueError(
                "initialize_missing_content_gate=True requires every attached "
                "Delta-Mem module to use a content-gated fusion mode; "
                f"non_gated={non_gated[:8]}"
            )
        allowed_config_mismatches = tuple(
            dict.fromkeys(
                (
                    *allowed_config_mismatches,
                    "memory_fusion_mode",
                    "memory_fusion_gate_init",
                )
            )
        )
    validate_attached_delta_config(
        model,
        config,
        allowed_config_mismatches=allowed_config_mismatches,
    )
    adapter_state = torch.load(
        input_path / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_delta_mem_state_dict(
        model,
        adapter_state,
        initialize_missing_residual_hybrid_gain=(
            initialize_missing_residual_hybrid_gain
        ),
        initialize_missing_content_gate=initialize_missing_content_gate,
    )
    return config
