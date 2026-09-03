#!/usr/bin/env python3
"""Train RWKV online memory and Gemma's native PLE path jointly."""

from __future__ import annotations

from contextlib import contextmanager
import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    HFDeltaMemConfig,
    attach_delta_mem,
    get_delta_mem_state_dict,
    iter_delta_mem_modules,
    reset_delta_mem_states,
    save_delta_mem_adapter,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain as trainer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_direct_ple_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "5fa2e1dbfc4513ff36722cbccfe5c3dfee94117fc696bd981c0e775e6054e381"
SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_causal_train_input.v1"
SEED = 20260902
UPDATES = 32
PRELIGHT_UPDATES = 1
HYBRID_MODE = "address_keyed_moe_ple"
HYBRID_GAIN = 0.015625
WRITE_ADDRESS_GAIN = 0.25
PLE_RANK = 4
PLE_GAIN = 0.125
PLE_INPUT_GAIN = 0.015625
RWKV_LOW_RANK_BOOTSTRAP_SCALE = 0.001
TRAINING_READ_MAX_TOKENS = 1536
BACKWARD_CONTROL_NAMES = (
    "zero_recurrent_state",
    "matched_donor_recurrent_state",
)
BUNDLED_CONDITION_GROUP_SIZE = 2
READ_TEMPERATURE = 16.0
READ_TOP_K = 2
DETACH_READ_SCORES = False
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CONTRAST_WEIGHT = 0.125
MARGIN = 0.05
WORLD_SIZE = 4
PASS_STATUS = "rwkv_direct_ple_heldout_passed_generation_authorized"
FAIL_STATUS = "rwkv_direct_ple_heldout_failed_generation_blocked"
WARMSTART = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"


def _training_read_window(
    example: evolution.NativeFullRowExample,
) -> evolution.NativeFullRowExample:
    if len(example.read_input_ids) <= TRAINING_READ_MAX_TOKENS:
        return example
    supervised = [
        index for index, label in enumerate(example.labels) if label != -100
    ]
    if not supervised:
        raise RuntimeError("Training read window has no supervised target")
    start = max(0, len(example.read_input_ids) - TRAINING_READ_MAX_TOKENS)
    if start > supervised[0]:
        start = supervised[0]
    end = start + TRAINING_READ_MAX_TOKENS
    return replace(
        example,
        read_input_ids=example.read_input_ids[start:end],
        read_attention_mask=example.read_attention_mask[start:end],
        labels=example.labels[start:end],
        assistant_target_tokens=sum(
            label != -100 for label in example.labels[start:end]
        ),
    )
_BASE_NATIVE_WRITE = evolution._native_write


def _canonical(value: Any) -> str:
    return distributed.canonical_sha256(value)


def _config() -> HFDeltaMemConfig:
    source = json.loads((WARMSTART / "delta_mem_config.json").read_text(encoding="utf-8"))
    return HFDeltaMemConfig.from_dict(
        {
            **source,
            "memory_readout_mode": "projected_kv_rwkv_hybrid",
            "memory_fusion_mode": "add",
            "delta_heads": [],
            "rwkv_ms_hybrid_mode": HYBRID_MODE,
            "rwkv_ms_hybrid_gain": HYBRID_GAIN,
            "rwkv_ms_write_address_gain": WRITE_ADDRESS_GAIN,
            "rwkv_ms_read_temperature": READ_TEMPERATURE,
            "rwkv_ms_read_top_k": READ_TOP_K,
            "rwkv_ms_detach_read_scores": DETACH_READ_SCORES,
            "rwkv_ms_outer_ffn_gain": 0.0,
            "rwkv_ms_outer_ffn_layers": [],
            "rwkv_ms_ple_rank": PLE_RANK,
            "rwkv_ms_ple_gain": PLE_GAIN,
            "rwkv_ms_ple_input_gain": PLE_INPUT_GAIN,
            "rwkv_ms_ple_fusion": "additive",
            "rwkv_ms_write_address_value_adapter": False,
        }
    )


def _load_warmstart(model: torch.nn.Module) -> Mapping[str, Any]:
    if (
        common.sha256_file(WARMSTART / "delta_mem_adapter.pt")
        != common.WARMSTART_WEIGHTS_SHA256
    ):
        raise ValueError("Direct PLE warm-start weights differ")
    if (
        common.sha256_file(WARMSTART / "delta_mem_config.json")
        != common.WARMSTART_CONFIG_SHA256
    ):
        raise ValueError("Direct PLE warm-start config differs")
    source = torch.load(
        WARMSTART / "delta_mem_adapter.pt", map_location="cpu", weights_only=True
    )
    expected = get_delta_mem_state_dict(model)
    fresh_suffixes = (
        ".rwkv_moe_hidden_weight",
        ".rwkv_moe_addressed_weight",
        ".rwkv_moe_global_weight",
        ".rwkv_moe_bias",
        ".rwkv_ple_down_weight",
        ".rwkv_ple_up_weight",
    )
    copied = 0
    missing: list[str] = []
    for key, parameter in ((key, value) for key, value in expected.items()):
        if key in source:
            value = source[key]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Warm-start shape differs for {key}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            copied += 1
        elif key.endswith(fresh_suffixes):
            missing.append(key)
        else:
            raise ValueError(f"Warm-start is missing required common tensor {key}")
    unexpected = sorted(
        key
        for key in set(source) - set(expected)
        if not key.endswith(
            (
                ".memory_fusion_hidden_weight",
                ".memory_fusion_read_weight",
                ".memory_fusion_bias",
            )
        )
    )
    if unexpected:
        raise ValueError(f"Warm-start contains unexpected tensors: {unexpected[:8]}")
    if len(missing) != 6 * common.EXPECTED_LAYERS:
        raise ValueError(f"Direct PLE fresh tensor count differs: {len(missing)}")
    return {
        "source": str(WARMSTART),
        "source_parameter_tensors": len(source),
        "copied_common_parameter_tensors": copied,
        "fresh_parameter_tensors": len(missing),
        "fresh_parameter_suffixes": list(fresh_suffixes),
    }


def _bootstrap_rwkv_low_rank_factors(model: torch.nn.Module) -> Mapping[str, Any]:
    factors = ("w1", "a1", "g1")
    changed: list[str] = []
    with torch.no_grad():
        for module_name, module in iter_delta_mem_modules(model):
            core = module.hrm_rwkv7_core
            if core is None:
                raise RuntimeError("Direct PLE bootstrap requires an RWKV core")
            for factor_name in factors:
                factor = getattr(core, factor_name)
                source_name = factor_name[:-1] + "2"
                source = getattr(core, source_name)
                if tuple(factor.shape) != tuple(source.shape[::-1]):
                    raise RuntimeError(
                        f"Direct PLE bootstrap shape differs for {module_name}.{factor_name}"
                    )
                pattern = torch.sign(source.transpose(0, 1)).to(
                    device=factor.device,
                    dtype=factor.dtype,
                )
                pattern = torch.where(
                    pattern.eq(0),
                    torch.ones_like(pattern),
                    pattern,
                )
                factor.copy_(RWKV_LOW_RANK_BOOTSTRAP_SCALE * pattern)
                changed.append(f"{module_name}.hrm_rwkv7_core.{factor_name}")
    expected = common.EXPECTED_LAYERS * len(factors)
    if len(changed) != expected:
        raise RuntimeError(
            f"Direct PLE bootstrap changed {len(changed)} tensors, expected {expected}"
        )
    return {
        "scale": RWKV_LOW_RANK_BOOTSTRAP_SCALE,
        "rule": "sign_transpose_of_second_factor",
        "families": list(factors),
        "parameter_tensors": len(changed),
        "parameter_names_sha256": _canonical(changed),
    }


def _configure_runtime(model: torch.nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.memory_fusion_mode = "add"
        module.rwkv_ms_hybrid_mode = HYBRID_MODE
        module.rwkv_ms_hybrid_gain = HYBRID_GAIN
        module.rwkv_ms_write_address_gain = WRITE_ADDRESS_GAIN
        module.rwkv_ms_read_temperature = READ_TEMPERATURE
        module.rwkv_ms_read_top_k = READ_TOP_K
        module.rwkv_ms_detach_read_scores = DETACH_READ_SCORES
        module.rwkv_ms_ple_rank = PLE_RANK
        module.rwkv_ms_ple_gain = PLE_GAIN
        module.rwkv_ms_ple_input_gain = PLE_INPUT_GAIN


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    trainable: bool,
    configure_trainables: Any = None,
) -> tuple[torch.nn.Module, Any, HFDeltaMemConfig, Mapping[str, Any]]:
    base_model = base_model.expanduser().resolve(strict=True)
    if (
        common.sha256_file(base_model / "model.safetensors")
        != common.BASE_MODEL_WEIGHTS_SHA256
    ):
        raise ValueError("Direct PLE base-model weights differ")
    if common.sha256_file(base_model / "config.json") != common.BASE_CONFIG_SHA256:
        raise ValueError("Direct PLE base-model config differs")
    if common.sha256_file(base_model / "tokenizer.json") != common.TOKENIZER_SHA256:
        raise ValueError("Direct PLE tokenizer differs")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    runtime._disable_training_cache(model)
    delta_config = _config()
    replaced = attach_delta_mem(model, delta_config)
    _configure_runtime(model)
    warmstart = _load_warmstart(model)
    rwkv_low_rank_bootstrap = _bootstrap_rwkv_low_rank_factors(model)
    named_trainable: tuple[tuple[str, torch.nn.Parameter], ...] = ()
    trainable_audit: Mapping[str, Any] = {"passed": True, "parameter_tensors": 0}
    if trainable:
        configure = configure_trainables or configure_trainable_parameters
        named_trainable, trainable_audit = configure(model)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    configured = (
        len(replaced) == common.EXPECTED_LAYERS
        and len(modules) == common.EXPECTED_LAYERS
        and all(
            module.memory_backend == "rwkv_ms"
            and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.memory_fusion_mode == "add"
            and module.rwkv_ms_hybrid_mode == HYBRID_MODE
            and module._ple_input_projection_hook_handle is not None
            for _, module in modules
        )
    )
    audit = {
        "replaced_layers": len(replaced),
        "wrapped_layers": len(modules),
        "configured": configured,
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "gradient_checkpointing": bool(model.is_gradient_checkpointing),
        "warmstart": warmstart,
        "rwkv_low_rank_bootstrap": rwkv_low_rank_bootstrap,
        "trainables": trainable_audit,
        "named_trainable": named_trainable,
        "native_gemma_ple": True,
        "native_ple_width": int(getattr(model.config.get_text_config(), "hidden_size_per_layer_input", 0)),
    }
    if not configured:
        raise RuntimeError(f"Direct PLE attachment failed: {audit!r}")
    return model, tokenizer, delta_config, audit


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suffixes = (
        ".memory_v_proj",
        ".projected_kv_key_proj",
        ".beta_proj",
        ".beta_bias",
        ".rwkv_moe_hidden_weight",
        ".rwkv_moe_addressed_weight",
        ".rwkv_moe_global_weight",
        ".rwkv_moe_bias",
        ".rwkv_ple_down_weight",
        ".rwkv_ple_up_weight",
    )
    selected_names: list[str] = []
    for name, parameter in model.named_parameters():
        selected = (
            name.endswith(suffixes)
            or (
                ".hrm_rwkv7_core." in name
                and not name.endswith(".hrm_rwkv7_core.ln_x.bias")
            )
        )
        parameter.requires_grad_(selected)
        if selected:
            selected_names.append(name)
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    )
    names = [name for name, _ in selected]
    family_counts = {
        suffix: sum(name.endswith(suffix) for name in names) for suffix in suffixes
    }
    core_names = [name for name in names if ".hrm_rwkv7_core." in name]
    required_families = {
        suffix: common.EXPECTED_LAYERS
        for suffix in suffixes
    }
    passed = (
        bool(selected)
        and names == sorted(selected_names)
        and all(family_counts[suffix] == expected for suffix, expected in required_families.items())
        and len(core_names) > common.EXPECTED_LAYERS
        and all(parameter.dtype == torch.float32 for _, parameter in selected)
    )
    audit = {
        "architecture": "direct_ungated_rwkv_online_memory_to_native_gemma_ple",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": _canonical(names),
        "trainable_suffixes": list(suffixes),
        "family_counts": family_counts,
        "rwkv_core_parameter_tensors": len(core_names),
        "projected_key_router_trainable_tensors": family_counts[".projected_kv_key_proj"],
        "native_ple_parameter_tensors": family_counts[".rwkv_ple_down_weight"] + family_counts[".rwkv_ple_up_weight"],
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Direct PLE trainable isolation failed: {audit!r}")
    return selected, audit


def route_subset_sha256(state: Mapping[str, torch.Tensor]) -> str:
    selected = {
        name: tensor
        for name, tensor in state.items()
        if name.endswith(".projected_kv_key_proj")
    }
    if len(selected) != common.EXPECTED_LAYERS:
        raise ValueError("Direct PLE projected-key route subset differs")
    return runtime._state_dict_sha256(selected)


def _gradient_audit(named_trainable: Sequence[tuple[str, torch.nn.Parameter]]) -> Mapping[str, Any]:
    finite: list[int] = []
    active: list[int] = []
    families: dict[str, dict[str, int]] = {}
    for name, parameter in named_trainable:
        gradient = parameter.grad
        is_finite = bool(
            gradient is not None
            and gradient.dtype == torch.float32
            and torch.isfinite(gradient).all().item()
        )
        is_active = bool(is_finite and gradient.abs().gt(0).any().item())
        finite.append(int(is_finite))
        active.append(int(is_active))
        family = "rwkv_core" if ".hrm_rwkv7_core." in name else name.rsplit(".", 1)[-1]
        entry = families.setdefault(family, {"tensors": 0, "active": 0})
        entry["tensors"] += 1
        entry["active"] += int(is_active)
    device = named_trainable[0][1].device
    finite_tensor = torch.tensor(finite, device=device, dtype=torch.int32)
    active_tensor = torch.tensor(active, device=device, dtype=torch.int32)
    torch.distributed.all_reduce(finite_tensor)
    torch.distributed.all_reduce(active_tensor)
    return {
        "trainable_tensors": len(named_trainable),
        "global_finite_fp32_tensors": int(finite_tensor.gt(0).sum().item()),
        "global_finite_nonzero_tensors": int(active_tensor.gt(0).sum().item()),
        "families": families,
        "passed": bool(finite_tensor.gt(0).all().item() and active_tensor.gt(0).all().item()),
    }


def _evaluate_conditions(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    donor: evolution.NativeFullRowBatch,
    device: torch.device,
    capture_top_k: int = 0,
) -> Mapping[str, Any]:
    del capture_top_k
    conditions = common.CONDITIONS
    modules = common.ordered_modules(model)
    logits: torch.Tensor | None = None
    try:
        with torch.no_grad():
            native_write(model, target, dtype=torch.bfloat16)
            correct = {
                name: {
                    attribute: value.detach().clone()
                    for attribute, value in attributes.items()
                }
                for name, attributes in common.capture_online_state_references(modules).items()
            }
            native_write(model, donor, dtype=torch.bfloat16)
            donor_state = {
                name: {
                    attribute: value.detach().clone()
                    for attribute, value in attributes.items()
                }
                for name, attributes in common.capture_online_state_references(modules).items()
            }
            metrics: dict[str, Any] = {}
            for condition in conditions:
                references_fixed = common.install_condition_state(
                    modules,
                    correct=correct,
                    donor=(
                        donor_state
                        if condition == "matched_donor_recurrent_state"
                        else None
                    ),
                    condition=condition,
                )
                read_batch = evolution.NativeFullRowBatch(
                    examples=target.examples,
                    write_input_ids=target.write_input_ids,
                    write_attention_mask=target.write_attention_mask,
                    read_input_ids=target.read_input_ids,
                    read_attention_mask=target.read_attention_mask,
                    labels=target.labels,
                )
                logits = evolution._native_read(model, read_batch, dtype=torch.bfloat16)
                bytes_fixed = all(
                    torch.equal(getattr(module, attribute), correct[name][attribute])
                    for name, module in modules
                    for attribute in common.PROJECTED_ATTRIBUTES
                )
                ce, tokens = contrast.detached_answer_ce(logits, target.labels)
                metrics[condition] = (
                    ce,
                    tokens,
                    {
                        "projected_carrier_references_fixed": references_fixed,
                        "projected_carrier_bytes_fixed": bytes_fixed,
                    },
                )
                logits = None
            return metrics
    finally:
        del logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def _bundle_backward_conditions(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    donor: evolution.NativeFullRowBatch,
    active: Mapping[str, bool],
    correct_coefficient: float,
    control_weights: Mapping[str, float],
    device: torch.device,
) -> tuple[int, int, Mapping[str, bool]]:
    if donor is None:
        raise RuntimeError("Direct PLE bundled backward requires a donor batch")
    conditions = (common.CONDITIONS[0],) + tuple(
        condition for condition in common.CONDITIONS[1:] if active.get(condition) is True
    )
    modules = common.ordered_modules(model)
    module_names = tuple(name for name, _ in modules)
    total_token_count: int | None = None
    total_chunk_count = 0
    condition_groups = tuple(
        conditions[start : start + BUNDLED_CONDITION_GROUP_SIZE]
        for start in range(0, len(conditions), BUNDLED_CONDITION_GROUP_SIZE)
    )
    for group_index, group_conditions in enumerate(condition_groups):
        labels = target.labels.expand(len(group_conditions), -1)
        include_donor = "matched_donor_recurrent_state" in group_conditions

        def stack_state(
            correct: Mapping[str, Mapping[str, torch.Tensor]],
            donor_state: Mapping[str, Mapping[str, torch.Tensor]],
        ) -> None:
            for index, (name, module) in enumerate(modules):
                for attribute in common.PROJECTED_ATTRIBUTES:
                    value = correct[name][attribute]
                    setattr(
                        module,
                        attribute,
                        value.expand(len(group_conditions), *value.shape[1:]),
                    )
                for attribute in common.RECURRENT_ATTRIBUTES:
                    values: list[torch.Tensor] = []
                    for condition in group_conditions:
                        if condition == "correct_recurrent_state":
                            value = correct[name][attribute]
                        elif condition == "zero_recurrent_state":
                            value = torch.zeros_like(correct[name][attribute])
                        elif condition == "matched_donor_recurrent_state":
                            value = donor_state[name][attribute]
                        elif condition == "slot_shuffled_recurrent_state":
                            value = correct[name][attribute]
                            if attribute == "delta_state":
                                value = value.roll(shifts=1, dims=2)
                        elif condition == "layer_permuted_recurrent_state":
                            source_name = module_names[(index + 1) % len(module_names)]
                            value = correct[source_name][attribute]
                        else:  # pragma: no cover - conditions are locked above
                            raise RuntimeError(f"Unknown bundled condition: {condition}")
                        values.append(value)
                    setattr(module, attribute, torch.cat(values, dim=0))

        logits: torch.Tensor | None = None

        def write_read(*tensors: torch.Tensor) -> torch.Tensor:
            target_batch = evolution.NativeFullRowBatch(
                examples=target.examples,
                write_input_ids=tensors[0],
                write_attention_mask=tensors[1],
                read_input_ids=tensors[2],
                read_attention_mask=tensors[3],
                labels=labels,
            )
            native_write(model, target_batch, dtype=torch.bfloat16)
            correct = common.capture_online_state_references(modules)
            if include_donor:
                donor_batch = evolution.NativeFullRowBatch(
                    examples=donor.examples,
                    write_input_ids=tensors[4],
                    write_attention_mask=tensors[5],
                    read_input_ids=tensors[2],
                    read_attention_mask=tensors[3],
                    labels=labels,
                )
                native_write(model, donor_batch, dtype=torch.bfloat16)
                donor_state = common.capture_online_state_references(modules)
            else:
                donor_state = correct
            stack_state(correct, donor_state)
            read_batch = evolution.NativeFullRowBatch(
                examples=target.examples,
                write_input_ids=tensors[0],
                write_attention_mask=tensors[1],
                read_input_ids=tensors[2].expand(len(group_conditions), -1),
                read_attention_mask=tensors[3].expand(len(group_conditions), -1),
                labels=labels,
            )
            return evolution._native_read(model, read_batch, dtype=torch.bfloat16)

        try:
            inputs = [
                target.write_input_ids,
                target.write_attention_mask,
                target.read_input_ids,
                target.read_attention_mask,
                donor.write_input_ids,
                donor.write_attention_mask,
            ]
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                logits = checkpoint(write_read, *inputs, use_reentrant=False)
            losses: list[torch.Tensor] = []
            token_count: int | None = None
            chunk_count = 0
            for index, condition in enumerate(group_conditions):
                loss_sum, tokens, chunks = evolution.checkpointed_native_answer_loss_sum_and_count(
                    logits[index : index + 1],
                    labels[index : index + 1],
                    chunk_tokens=contrast.CE_CHUNK_TOKENS,
                )
                if token_count is None:
                    token_count = tokens
                elif token_count != tokens:
                    raise RuntimeError("Direct PLE bundled token counts differ")
                coefficient = (
                    correct_coefficient
                    if condition == "correct_recurrent_state"
                    else -float(control_weights[condition])
                )
                losses.append(
                    (loss_sum / tokens)
                    * (coefficient / trainer.GLOBAL_BATCH_SIZE)
                )
                chunk_count += chunks
            if token_count is None or not losses:
                raise RuntimeError("Direct PLE bundled backward emitted no losses")
            total_loss = torch.stack(losses).sum()
            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError("Direct PLE bundled loss is non-finite")
            total_loss.backward()
            if total_token_count is None:
                total_token_count = token_count
            elif total_token_count != token_count:
                raise RuntimeError("Direct PLE bundled group token counts differ")
            total_chunk_count += chunk_count
        finally:
            logits = None
            reset_delta_mem_states(model)
            evolution.release_native_row_allocator_cache(device)
    if total_token_count is None:
        raise RuntimeError("Direct PLE bundled backward emitted no groups")
    return total_token_count, total_chunk_count, {
        "projected_carrier_references_fixed": True,
        "projected_carrier_bytes_fixed": True,
    }


def native_write(model: torch.nn.Module, batch: Any, *, dtype: torch.dtype) -> Mapping[str, Any]:
    _configure_runtime(model)
    result = _BASE_NATIVE_WRITE(model, batch, dtype=dtype)
    if not all(module.rwkv_ms_hybrid_mode == HYBRID_MODE for _, module in iter_delta_mem_modules(model)):
        raise RuntimeError("Direct PLE write escaped the locked architecture")
    return result


def validate_training_protocol(updates: int) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    if updates not in {PRELIGHT_UPDATES, UPDATES}:
        raise ValueError("Direct PLE updates must be 1 or 32")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    protocol_digest = common.canonical_sha256(unsigned)
    if (
        not isinstance(receipt, Mapping)
        or protocol_digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != protocol_digest
    ):
        raise ValueError("Direct PLE protocol receipt differs")
    if protocol.get("architecture", {}).get("benchmark_time_selector") is not False:
        raise ValueError("Direct PLE protocol permits no benchmark-time selector")
    if protocol.get("architecture", {}).get("identity_binder") is not False:
        raise ValueError("Direct PLE protocol permits no identity binder")
    bootstrap = protocol.get("architecture", {}).get("rwkv_low_rank_bootstrap")
    if bootstrap != {
        "families": ["w1", "a1", "g1"],
        "rule": "sign_transpose_of_second_factor",
        "scale": RWKV_LOW_RANK_BOOTSTRAP_SCALE,
    }:
        raise ValueError("Direct PLE RWKV bootstrap differs from protocol")
    if protocol.get("architecture", {}).get("training_read_max_tokens") != TRAINING_READ_MAX_TOKENS:
        raise ValueError("Direct PLE training read window differs from protocol")
    if protocol.get("training", {}).get("backward_control_names") != list(BACKWARD_CONTROL_NAMES):
        raise ValueError("Direct PLE backward control set differs from protocol")
    return protocol


@contextmanager
def bindings() -> Iterator[None]:
    previous = {
        "HYBRID_MODE": common.HYBRID_MODE,
        "HYBRID_GAIN": common.HYBRID_GAIN,
        "READ_TEMPERATURE": common.READ_TEMPERATURE,
        "READ_TOP_K": common.READ_TOP_K,
        "COMMON_PROTOCOL": common.PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": common.PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": trainer.SCHEMA,
        "STEP_SCHEMA": trainer.STEP_SCHEMA,
        "INPUT_SCHEMA": trainer.INPUT_SCHEMA,
        "SEED": trainer.SEED,
        "TRAIN_UPDATES": trainer.TRAIN_UPDATES,
        "PREFLIGHT_UPDATES": trainer.PREFLIGHT_UPDATES,
        "LEARNING_RATE": trainer.LEARNING_RATE,
        "MAX_GRAD_NORM": trainer.MAX_GRAD_NORM,
        "CONTRAST_WEIGHT": trainer.CONTRAST_WEIGHT,
        "MARGIN": trainer.MARGIN,
        "TRAINING_READ_MAX_TOKENS": trainer.TRAINING_READ_MAX_TOKENS,
    }
    previous_load = common.load_model
    previous_protocol = trainer.validate_training_protocol
    previous_write = evolution._native_write
    previous_audit = trainer.common.audit_joint_routing_gradients
    previous_route_subset = trainer.route_subset_sha256
    try:
        common.HYBRID_MODE = HYBRID_MODE
        common.HYBRID_GAIN = HYBRID_GAIN
        common.READ_TEMPERATURE = READ_TEMPERATURE
        common.READ_TOP_K = READ_TOP_K
        common.PROTOCOL = PROTOCOL
        common.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
        common.load_model = load_model
        trainer.SCHEMA = SCHEMA
        trainer.STEP_SCHEMA = STEP_SCHEMA
        trainer.INPUT_SCHEMA = INPUT_SCHEMA
        trainer.SEED = SEED
        trainer.TRAIN_UPDATES = UPDATES
        trainer.PREFLIGHT_UPDATES = PRELIGHT_UPDATES
        trainer.LEARNING_RATE = LEARNING_RATE
        trainer.MAX_GRAD_NORM = MAX_GRAD_NORM
        trainer.CONTRAST_WEIGHT = CONTRAST_WEIGHT
        trainer.MARGIN = MARGIN
        trainer.TRAINING_READ_MAX_TOKENS = TRAINING_READ_MAX_TOKENS
        trainer.validate_training_protocol = validate_training_protocol
        evolution._native_write = native_write
        trainer.common.audit_joint_routing_gradients = _gradient_audit
        trainer.route_subset_sha256 = route_subset_sha256
        previous_train = trainer.train

        def train_with_audit(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            kwargs["gradient_audit_fn"] = _gradient_audit
            kwargs["always_active_controls"] = common.CONDITIONS[1:]
            kwargs["evaluate_conditions_fn"] = _evaluate_conditions
            kwargs["backward_control_names"] = BACKWARD_CONTROL_NAMES
            kwargs["example_transform_fn"] = _training_read_window
            return previous_train(*args, **kwargs)

        trainer.train = train_with_audit
        yield
    finally:
        evolution._native_write = previous_write
        trainer.common.audit_joint_routing_gradients = previous_audit
        trainer.route_subset_sha256 = previous_route_subset
        trainer.train = previous_train
        trainer.validate_training_protocol = previous_protocol
        common.load_model = previous_load
        for name, value in previous.items():
            target_name = {
                "COMMON_PROTOCOL": (common, "PROTOCOL"),
            }.get(name)
            if target_name is not None:
                setattr(*target_name, value)
            elif hasattr(common, name):
                setattr(common, name, value)
            if hasattr(trainer, name):
                setattr(trainer, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return trainer.run(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, 32))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with bindings():
        context = distributed.initialize_distributed_training(
            args.device,
            timeout_seconds=7200,
        )
        if context is None or context.world_size != WORLD_SIZE:
            raise ValueError("Direct PLE training requires exactly four ranks")
        try:
            result = trainer.run(
                context=context,
                output_dir=args.output_dir,
                updates=args.updates,
                base_model=args.base_model,
            )
        finally:
            distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": None if not context.is_primary else result["receipt"]["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
