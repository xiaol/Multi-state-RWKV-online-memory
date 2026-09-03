#!/usr/bin/env python3
"""Post-train recurrent routing against fixed-carrier causal controls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    reset_delta_mem_states,
    save_delta_mem_adapter,
    snapshot_delta_mem_weights,
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
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_input.v1"
WORLD_SIZE = 4
SEED = 20260828
GLOBAL_BATCH_SIZE = 8
LOCAL_ROWS = 2
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 32
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CONTRAST_WEIGHT = 0.125
MARGIN = 0.05
DISTILL_TEACHER_ADAPTER: Path | None = None
DISTILL_BASE_MODEL: Path | None = None
DISTILL_CACHE_ROOT: Path | None = None
DISTILL_WEIGHT = 0.0
DISTILL_TEMPERATURE = 1.0
DISTILL_TOP_K = 64
TRAINING_READ_MAX_TOKENS: int | None = None


def write_fresh_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Recurrent-routed output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def validate_training_protocol(updates: int) -> Mapping[str, Any]:
    protocol = common.validate_protocol()
    training = protocol.get("training", {})
    expected = {
        "hf_endpoint": common.HF_MIRROR_ENDPOINT,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "preflight_optimizer_updates": PREFLIGHT_UPDATES,
        "optimizer_updates": TRAIN_UPDATES,
        "global_batch_rows": GLOBAL_BATCH_SIZE,
        "local_rows_per_rank": LOCAL_ROWS,
        "contrast_weight_per_active_control": CONTRAST_WEIGHT,
        "contrast_margin": MARGIN,
        "final_rows_opened_during_training": False,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("Recurrent-routed training protocol differs")
    if updates not in {PREFLIGHT_UPDATES, TRAIN_UPDATES}:
        raise ValueError("Recurrent-routed updates must be 1 or 32")
    return protocol


def four_distinct_a100s(rank_devices: Sequence[Mapping[str, Any]]) -> bool:
    return (
        len(rank_devices) == WORLD_SIZE
        and len({str(device.get("device_uuid")) for device in rank_devices})
        == WORLD_SIZE
        and all("A100" in str(device.get("device_name")) for device in rank_devices)
    )


def backward_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float,
) -> tuple[int, int]:
    loss_sum, tokens, chunks = evolution.checkpointed_native_answer_loss_sum_and_count(
        logits,
        labels,
        chunk_tokens=contrast.CE_CHUNK_TOKENS,
    )
    mean_ce = loss_sum / tokens
    scaled = mean_ce * (coefficient / GLOBAL_BATCH_SIZE)
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Recurrent-routed signed loss is non-finite")
    scaled.backward()
    del loss_sum, mean_ce, scaled
    return tokens, chunks


def evaluate_condition(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    condition: str,
    donor: evolution.NativeFullRowBatch | None,
    device: torch.device,
    capture_top_k: int = 0,
) -> tuple[float, int, Mapping[str, bool]] | tuple[
    float,
    int,
    Mapping[str, bool],
    Mapping[str, torch.Tensor],
]:
    logits: torch.Tensor | None = None
    try:
        with torch.no_grad():
            logits, audit = common.direct_condition_logits(
                model,
                target,
                condition=condition,
                donor=donor,
                dtype=torch.bfloat16,
            )
            ce, tokens = contrast.detached_answer_ce(logits, target.labels)
            if capture_top_k > 0:
                selected, labels = selected_answer_logits(logits, target.labels)
                values, indices = torch.topk(
                    selected.float(),
                    k=min(int(capture_top_k), int(selected.size(-1))),
                    dim=-1,
                )
                return ce, tokens, audit, {
                    "teacher_values": values.detach(),
                    "teacher_indices": indices.detach(),
                    "teacher_labels": labels.detach(),
                }
        return ce, tokens, audit
    finally:
        del logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def backward_condition(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    *,
    condition: str,
    donor: evolution.NativeFullRowBatch | None,
    coefficient: float,
    device: torch.device,
) -> tuple[int, int, Mapping[str, bool]]:
    logits: torch.Tensor | None = None
    try:
        logits, audit = common.checkpointed_condition_logits(
            model,
            target,
            condition=condition,
            donor=donor,
            dtype=torch.bfloat16,
        )
        tokens, chunks = backward_logits(
            logits,
            target.labels,
            coefficient=coefficient,
        )
        return tokens, chunks, audit
    finally:
        del logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def selected_answer_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = labels[:, 1:].ne(-100)
    predictor_indices = supervised.any(dim=0).nonzero(as_tuple=False).flatten()
    if logits.size(1) == labels.size(1):
        selected_logits = logits.index_select(1, predictor_indices)
    elif logits.size(1) == predictor_indices.numel():
        selected_logits = logits
    else:
        raise ValueError("Distillation logits do not cover supervised predictors")
    selected_labels = labels.index_select(1, predictor_indices + 1)
    mask = selected_labels.ne(-100)
    return selected_logits[mask], selected_labels[mask]


def backward_distillation(
    model: torch.nn.Module,
    target: evolution.NativeFullRowBatch,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_labels: torch.Tensor,
    *,
    coefficient: float,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[int, Mapping[str, bool]]:
    logits: torch.Tensor | None = None
    try:
        logits, audit = common.checkpointed_condition_logits(
            model,
            target,
            condition="correct_recurrent_state",
            donor=None,
            dtype=torch.bfloat16,
        )
        student, labels = selected_answer_logits(logits, target.labels)
        if labels.shape != teacher_labels.shape or not torch.equal(labels, teacher_labels):
            raise RuntimeError("Distillation teacher and student labels differ")
        if not math.isfinite(float(temperature)) or temperature <= 0.0:
            raise ValueError("Distillation temperature must be positive")
        if top_k <= 0:
            raise ValueError("Distillation top_k must be positive")
        if teacher_values.ndim != 2 or teacher_indices.shape != teacher_values.shape:
            raise RuntimeError("Distillation cache top-k tensors differ")
        if teacher_values.size(0) != student.size(0):
            raise RuntimeError("Distillation cache token count differs")
        k = min(int(top_k), int(teacher_values.size(-1)))
        teacher_values = teacher_values[:, :k].float() / temperature
        teacher_indices = teacher_indices[:, :k].long()
        teacher_probs = torch.softmax(teacher_values, dim=-1)
        student_log_probs = torch.log_softmax(student.float() / temperature, dim=-1)
        loss = -(
            teacher_probs
            * student_log_probs.gather(-1, teacher_indices)
        ).sum(dim=-1).mean() * (temperature * temperature)
        scaled = loss * (float(coefficient) / GLOBAL_BATCH_SIZE)
        if not bool(torch.isfinite(scaled).item()):
            raise RuntimeError("Distillation loss is non-finite")
        scaled.backward()
        return int(labels.numel()), audit
    finally:
        del logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def trainable_subset_sha256(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    return runtime._state_dict_sha256(
        {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in named_trainable
        }
    )


def route_subset_sha256(state: Mapping[str, torch.Tensor]) -> str:
    selected = {
        name: tensor
        for name, tensor in state.items()
        if name.endswith((".rwkv_route_query_proj", ".rwkv_route_state_proj"))
    }
    if len(selected) != common.EXPECTED_LAYERS * 2:
        raise ValueError("Recurrent-routed adapter route subset differs")
    return runtime._state_dict_sha256(selected)


def accumulate_gradients_on_cpu(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    accumulated: dict[str, torch.Tensor],
) -> tuple[int, int]:
    tensors = 0
    transferred_bytes = 0
    for name, parameter in named_trainable:
        gradient = parameter.grad
        if gradient is None:
            continue
        cpu_gradient = gradient.detach().to(device="cpu", copy=True)
        if name in accumulated:
            accumulated[name].add_(cpu_gradient)
        else:
            accumulated[name] = cpu_gradient
        tensors += 1
        transferred_bytes += cpu_gradient.numel() * cpu_gradient.element_size()
        parameter.grad = None
    return tensors, transferred_bytes


def materialize_accumulated_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    accumulated: Mapping[str, torch.Tensor],
) -> None:
    for name, parameter in named_trainable:
        gradient = accumulated.get(name)
        parameter.grad = (
            None
            if gradient is None
            else gradient.to(device=parameter.device, non_blocking=False)
        )


def train(
    model: torch.nn.Module,
    tokenizer: Any,
    schedule: Sequence[common.ScheduledRow],
    schedule_payload: Sequence[Mapping[str, Any]],
    *,
    updates: int,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    learning_rate: float = LEARNING_RATE,
    max_grad_norm: float = MAX_GRAD_NORM,
    margin: float = MARGIN,
    control_weights: Mapping[str, float] | None = None,
    always_active_controls: Sequence[str] = (),
    baseline_anchor_weight: float = 0.0,
    baseline_anchor_temperature: float = 1.0,
    baseline_anchor_top_k: int = 64,
    learning_rate_multipliers: Mapping[str, float] | None = None,
    protocol_payload_sha256: str = common.PROTOCOL_PAYLOAD_SHA256,
    gradient_audit_fn: Callable[
        [Sequence[tuple[str, torch.nn.Parameter]]], Mapping[str, Any]
    ] = common.audit_joint_routing_gradients,
    backward_bundle_fn: Callable[..., tuple[int, int, Mapping[str, bool]]] | None = None,
    evaluate_conditions_fn: Callable[..., Mapping[str, Any]] | None = None,
    backward_control_names: Sequence[str] | None = None,
    example_transform_fn: Callable[[Any], Any] | None = None,
) -> Mapping[str, Any]:
    if control_weights is None:
        control_weights = {
            condition: CONTRAST_WEIGHT for condition in common.CONDITIONS[1:]
        }
    if set(control_weights) != set(common.CONDITIONS[1:]) or any(
        not math.isfinite(float(weight)) or float(weight) <= 0.0
        for weight in control_weights.values()
    ):
        raise ValueError("Recurrent-routed control weights differ")
    control_names = common.CONDITIONS[1:]
    if backward_control_names is None:
        backward_control_names = control_names
    backward_control_names = tuple(backward_control_names)
    if not backward_control_names or any(
        condition not in control_names for condition in backward_control_names
    ) or len(set(backward_control_names)) != len(backward_control_names):
        raise ValueError("Recurrent-routed backward control names differ")
    always_active_controls = tuple(always_active_controls)
    if any(condition not in control_names for condition in always_active_controls):
        raise ValueError("Recurrent-routed always-active control differs")
    if len(set(always_active_controls)) != len(always_active_controls):
        raise ValueError("Recurrent-routed always-active controls contain duplicates")
    if not math.isfinite(float(baseline_anchor_weight)) or baseline_anchor_weight < 0.0:
        raise ValueError("Recurrent-routed baseline anchor weight is invalid")
    if (
        not math.isfinite(float(baseline_anchor_temperature))
        or baseline_anchor_temperature <= 0.0
    ):
        raise ValueError("Recurrent-routed baseline anchor temperature is invalid")
    if baseline_anchor_top_k <= 0:
        raise ValueError("Recurrent-routed baseline anchor top-k is invalid")
    if learning_rate_multipliers is None:
        learning_rate_multipliers = {}
    if any(
        not suffix
        or not math.isfinite(float(multiplier))
        or float(multiplier) <= 0.0
        for suffix, multiplier in learning_rate_multipliers.items()
    ):
        raise ValueError("Recurrent-routed learning-rate multipliers differ")
    parameter_groups: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in named_trainable:
        matching = [
            float(multiplier)
            for suffix, multiplier in learning_rate_multipliers.items()
            if name.endswith(suffix)
        ]
        if len(matching) > 1:
            raise ValueError(f"Multiple learning-rate multipliers match {name}")
        parameter_groups.setdefault(matching[0] if matching else 1.0, []).append(
            parameter
        )
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": learning_rate * multiplier,
            }
            for multiplier, parameters in sorted(parameter_groups.items())
        ],
        lr=learning_rate,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    initial_state = snapshot_delta_mem_weights(model)
    initial_adapter_sha256 = runtime._state_dict_sha256(initial_state)
    initial_route_sha256 = route_subset_sha256(initial_state)
    initial_trainable_sha256 = trainable_subset_sha256(named_trainable)
    for description, digest in (
        ("initial recurrent-routed adapter", initial_adapter_sha256),
        ("initial recurrent-routed route subset", initial_route_sha256),
        ("initial recurrent-routed trainable subset", initial_trainable_sha256),
    ):
        distributed.require_consensus(context, digest, description=description)

    progress_path = output_dir / "training_progress.jsonl"
    total_correct_ce = 0.0
    total_margins = {condition: 0.0 for condition in control_names}
    total_active = {condition: 0.0 for condition in control_names}
    total_task_rows = {task: 0 for task in common.TASKS}
    total_prompt_variants = {str(index): 0 for index in range(4)}
    total_baseline_anchor_rows = 0
    total_baseline_anchor_tokens = 0
    minimum_gradient_norm = math.inf
    maximum_global_inactive = 0
    projected_carrier_fixed_every_row = True
    first_gradient_audit: Mapping[str, Any] | None = None
    optimizer_state_cpu_offload_steps = 0
    optimizer_state_cpu_offload_tensors = 0
    optimizer_state_cpu_offload_bytes = 0
    started = time.time()
    for step in range(1, updates + 1):
        step_rows = [row for row in schedule if row.step == step]
        if len(step_rows) != GLOBAL_BATCH_SIZE:
            raise RuntimeError("Recurrent-routed global schedule size differs")
        local_rows = step_rows[
            context.process_rank * LOCAL_ROWS : (context.process_rank + 1) * LOCAL_ROWS
        ]
        if len(local_rows) != LOCAL_ROWS:
            raise RuntimeError("Recurrent-routed local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        optimizer_state_offload = None
        optimizer_state_restore = None
        if optimizer.state:
            optimizer_state_offload = evolution.move_optimizer_state(
                optimizer,
                device=torch.device("cpu"),
            )
            if optimizer_state_offload.tensors <= 0:
                raise RuntimeError("Recurrent-routed optimizer offload moved no tensors")
            optimizer_state_cpu_offload_steps += 1
            optimizer_state_cpu_offload_tensors = max(
                optimizer_state_cpu_offload_tensors,
                optimizer_state_offload.tensors,
            )
            optimizer_state_cpu_offload_bytes = max(
                optimizer_state_cpu_offload_bytes,
                optimizer_state_offload.bytes,
            )
            evolution.release_native_row_allocator_cache(context.device)
        local_correct_ce = 0.0
        local_margins = {condition: 0.0 for condition in control_names}
        local_active = {condition: 0.0 for condition in control_names}
        local_tokens = 0
        local_chunks = 0
        local_carrier_fixed = 0
        local_tasks = {task: 0 for task in common.TASKS}
        local_variants = {str(index): 0 for index in range(4)}
        local_baseline_anchor_rows = 0
        local_baseline_anchor_tokens = 0
        accumulated_gradients: dict[str, torch.Tensor] = {}
        local_gradient_offload_tensors = 0
        local_gradient_offload_bytes = 0
        for scheduled in local_rows:
            target_example = common.encode_row(
                tokenizer,
                scheduled.target,
                prompt_variant=scheduled.prompt_variant,
            )
            if example_transform_fn is not None:
                target_example = example_transform_fn(target_example)
            donor_example = common.encode_row(
                tokenizer,
                scheduled.donor,
                prompt_variant=scheduled.prompt_variant,
            )
            if example_transform_fn is not None:
                donor_example = example_transform_fn(donor_example)
            target = evolution.collate_native_examples(
                [target_example],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            donor_write = evolution.collate_native_examples(
                [donor_example],
                pad_token_id=int(tokenizer.pad_token_id),
                device=context.device,
            )
            donor = evolution.NativeFullRowBatch(
                examples=target.examples,
                write_input_ids=donor_write.write_input_ids,
                write_attention_mask=donor_write.write_attention_mask,
                read_input_ids=target.read_input_ids,
                read_attention_mask=target.read_attention_mask,
                labels=target.labels,
            )
            teacher_cache = None
            if DISTILL_CACHE_ROOT is not None:
                cache_path = DISTILL_CACHE_ROOT / (
                    f"{scheduled.target.task}-{scheduled.target.source_ordinal}-"
                    f"{scheduled.prompt_variant}.pt"
                )
                teacher_cache = torch.load(
                    cache_path.expanduser().resolve(strict=True),
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    teacher_cache.get("row_sha256") != scheduled.target.row_sha256
                    or int(teacher_cache.get("prompt_variant", -1))
                    != scheduled.prompt_variant
                ):
                    raise RuntimeError("Distillation cache binding differs")
            condition_metrics: dict[str, tuple[float, int, Mapping[str, bool]]] = {}
            baseline_teacher_cache = None
            if evaluate_conditions_fn is not None:
                if baseline_anchor_weight > 0.0:
                    raise RuntimeError(
                        "Bulk recurrent evaluation does not support baseline teachers"
                    )
                condition_metrics = dict(
                    evaluate_conditions_fn(
                        model,
                        target,
                        donor=donor,
                        device=context.device,
                        capture_top_k=baseline_anchor_top_k,
                    )
                )
            else:
                for condition in common.CONDITIONS:
                    evaluated = evaluate_condition(
                        model,
                        target,
                        condition=condition,
                        donor=(
                            donor
                            if condition == "matched_donor_recurrent_state"
                            else None
                        ),
                        device=context.device,
                        capture_top_k=(
                            baseline_anchor_top_k
                            if baseline_anchor_weight > 0.0
                            and condition == "zero_recurrent_state"
                            else 0
                        ),
                    )
                    condition_metrics[condition] = evaluated[:3]
                    if len(evaluated) == 4:
                        baseline_teacher_cache = evaluated[3]
            token_counts = {value[1] for value in condition_metrics.values()}
            if len(token_counts) != 1:
                raise RuntimeError("Recurrent-routed condition token counts differ")
            correct_ce = condition_metrics["correct_recurrent_state"][0]
            margins = {
                condition: condition_metrics[condition][0] - correct_ce
                for condition in control_names
            }
            active = {
                condition: condition in always_active_controls or value < margin
                for condition, value in margins.items()
            }
            backward_active = {
                condition: active[condition] and condition in backward_control_names
                for condition in control_names
            }
            correct_coefficient = 1.0 + sum(
                float(control_weights[condition])
                for condition, is_active in backward_active.items()
                if is_active
            )
            if backward_bundle_fn is not None:
                if teacher_cache is not None or baseline_anchor_weight > 0.0:
                    raise RuntimeError(
                        "Bundled recurrent backward does not support auxiliary teachers"
                    )
                correct_tokens, row_chunks, bundle_audit = backward_bundle_fn(
                    model,
                    target,
                    donor=donor,
                    active=active,
                    correct_coefficient=correct_coefficient,
                    control_weights=control_weights,
                    device=context.device,
                )
                row_fixed = bool(
                    bundle_audit["projected_carrier_references_fixed"]
                    and bundle_audit["projected_carrier_bytes_fixed"]
                )
            else:
                correct_tokens, correct_chunks, correct_audit = backward_condition(
                    model,
                    target,
                    condition="correct_recurrent_state",
                    donor=None,
                    coefficient=correct_coefficient,
                    device=context.device,
                )
                row_chunks = correct_chunks
                row_fixed = bool(
                    correct_audit["projected_carrier_references_fixed"]
                    and correct_audit["projected_carrier_bytes_fixed"]
                )
                for condition in control_names:
                    metric_audit = condition_metrics[condition][2]
                    row_fixed = bool(
                        row_fixed
                        and metric_audit["projected_carrier_references_fixed"]
                        and metric_audit["projected_carrier_bytes_fixed"]
                    )
                    if not backward_active[condition]:
                        continue
                    control_tokens, control_chunks, backward_audit = backward_condition(
                        model,
                        target,
                        condition=condition,
                        donor=(
                            donor
                            if condition == "matched_donor_recurrent_state"
                            else None
                        ),
                        coefficient=-float(control_weights[condition]),
                        device=context.device,
                    )
                    if control_tokens != correct_tokens:
                        raise RuntimeError("Recurrent-routed backward token counts differ")
                    row_chunks += control_chunks
                    row_fixed = bool(
                        row_fixed
                        and backward_audit["projected_carrier_references_fixed"]
                        and backward_audit["projected_carrier_bytes_fixed"]
                    )
            for condition in control_names:
                metric_audit = condition_metrics[condition][2]
                row_fixed = bool(
                    row_fixed
                    and metric_audit["projected_carrier_references_fixed"]
                    and metric_audit["projected_carrier_bytes_fixed"]
                )
            if teacher_cache is not None and DISTILL_WEIGHT > 0.0:
                distill_tokens, distill_audit = backward_distillation(
                    model,
                    target,
                    teacher_cache["teacher_values"].to(context.device),
                    teacher_cache["teacher_indices"].to(context.device),
                    teacher_cache["teacher_labels"].to(context.device),
                    coefficient=DISTILL_WEIGHT,
                    temperature=DISTILL_TEMPERATURE,
                    top_k=DISTILL_TOP_K,
                    device=context.device,
                )
                if distill_tokens != correct_tokens:
                    raise RuntimeError("Distillation token counts differ")
                row_fixed = bool(
                    row_fixed
                    and distill_audit["projected_carrier_references_fixed"]
                    and distill_audit["projected_carrier_bytes_fixed"]
                )
            if baseline_anchor_weight > 0.0:
                if baseline_teacher_cache is None:
                    raise RuntimeError("Baseline anchor teacher logits are missing")
                anchor_tokens, anchor_audit = backward_distillation(
                    model,
                    target,
                    baseline_teacher_cache["teacher_values"],
                    baseline_teacher_cache["teacher_indices"],
                    baseline_teacher_cache["teacher_labels"],
                    coefficient=baseline_anchor_weight,
                    temperature=baseline_anchor_temperature,
                    top_k=baseline_anchor_top_k,
                    device=context.device,
                )
                if anchor_tokens != correct_tokens:
                    raise RuntimeError("Baseline anchor token count differs")
                row_fixed = bool(
                    row_fixed
                    and anchor_audit["projected_carrier_references_fixed"]
                    and anchor_audit["projected_carrier_bytes_fixed"]
                )
                local_baseline_anchor_rows += 1
                local_baseline_anchor_tokens += anchor_tokens
            projected_carrier_fixed_every_row = (
                projected_carrier_fixed_every_row and row_fixed
            )
            local_correct_ce += correct_ce
            for condition in control_names:
                local_margins[condition] += margins[condition]
                local_active[condition] += float(active[condition])
            local_tokens += correct_tokens
            local_chunks += row_chunks
            local_carrier_fixed += int(row_fixed)
            local_tasks[scheduled.target.task] += 1
            local_variants[str(scheduled.prompt_variant)] += 1
            offload_tensors, offload_bytes = accumulate_gradients_on_cpu(
                named_trainable,
                accumulated_gradients,
            )
            if offload_tensors <= 0 or offload_bytes <= 0:
                raise RuntimeError("Recurrent-routed row produced no gradients")
            local_gradient_offload_tensors += offload_tensors
            local_gradient_offload_bytes += offload_bytes
            del (
                teacher_cache,
                baseline_teacher_cache,
                target,
                donor,
                donor_write,
                target_example,
                donor_example,
            )
            reset_delta_mem_states(model)
            evolution.release_native_row_allocator_cache(context.device)

        materialize_accumulated_gradients(named_trainable, accumulated_gradients)
        del accumulated_gradients

        scalar_values = [local_correct_ce]
        scalar_values.extend(local_margins[condition] for condition in control_names)
        scalar_values.extend(local_active[condition] for condition in control_names)
        scalar_values.extend((local_tokens, local_chunks, local_carrier_fixed))
        scalar_values.extend(local_tasks[task] for task in common.TASKS)
        scalar_values.extend(local_variants[str(index)] for index in range(4))
        scalar_values.extend(
            (local_baseline_anchor_rows, local_baseline_anchor_tokens)
        )
        metrics = contrast.gate._distributed_scalar_sums(
            context,
            contrast.gate._prepare_distributed_scalar_sums(
                context,
                tuple(float(value) for value in scalar_values),
            ),
        )
        local_gradient_validation = distributed.validate_local_gradients(
            named_trainable
        )
        if local_gradient_validation["passed"] is not True:
            raise RuntimeError(
                f"Recurrent-routed local gradients are invalid: "
                f"{local_gradient_validation!r}"
            )
        collective = distributed.sum_gradients(context, named_trainable)
        inactive = len(collective["global_inactive_parameter_indices"])
        maximum_global_inactive = max(maximum_global_inactive, inactive)
        if inactive:
            raise RuntimeError("Recurrent-routed optimizer has inactive parameters")
        gradient_audit = None
        if step == 1:
            gradient_audit = gradient_audit_fn(named_trainable)
            if gradient_audit["passed"] is not True:
                failed_families = {
                    name: family
                    for name, family in gradient_audit.get(
                        "families",
                        {},
                    ).items()
                    if family.get("passed") is not True
                }
                raise RuntimeError(
                    "Recurrent-routed first update did not activate all required "
                    f"families: {failed_families!r}"
                )
            first_gradient_audit = gradient_audit
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        gradient_norm_value = float(gradient_norm.detach().float().item())
        if not bool(torch.isfinite(gradient_norm).item()) or gradient_norm_value <= 0.0:
            raise RuntimeError("Recurrent-routed gradient norm is invalid")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm_value)
        if optimizer_state_offload is not None:
            optimizer_state_restore = evolution.move_optimizer_state(
                optimizer,
                device=context.device,
            )
            if optimizer_state_restore != optimizer_state_offload:
                raise RuntimeError("Recurrent-routed optimizer state restore differs")
        optimizer.step()

        offset = 1
        step_margins = {
            condition: metrics[offset + index] / GLOBAL_BATCH_SIZE
            for index, condition in enumerate(control_names)
        }
        offset += len(control_names)
        step_active = {
            condition: int(metrics[offset + index])
            for index, condition in enumerate(control_names)
        }
        offset += len(control_names)
        answer_tokens = int(metrics[offset])
        checkpointed_chunks = int(metrics[offset + 1])
        fixed_rows = int(metrics[offset + 2])
        offset += 3
        task_rows = {
            task: int(metrics[offset + index])
            for index, task in enumerate(common.TASKS)
        }
        offset += len(common.TASKS)
        prompt_variants = {
            str(index): int(metrics[offset + index])
            for index in range(4)
        }
        offset += 4
        baseline_anchor_rows = int(metrics[offset])
        baseline_anchor_tokens = int(metrics[offset + 1])
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": protocol_payload_sha256,
            "step": step,
            "schedule_step_sha256": schedule_payload[step - 1]["payload_sha256"],
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "mean_correct_ce": metrics[0] / GLOBAL_BATCH_SIZE,
            "mean_control_minus_correct_ce": step_margins,
            "active_control_rows": step_active,
            "answer_target_tokens": answer_tokens,
            "checkpointed_ce_chunks": checkpointed_chunks,
            "projected_carrier_fixed_rows": fixed_rows,
            "task_rows": task_rows,
            "prompt_variant_rows": prompt_variants,
            "baseline_anchor_rows": baseline_anchor_rows,
            "baseline_anchor_tokens": baseline_anchor_tokens,
            "gradient_norm_before_clip": gradient_norm_value,
            "gradient_collective_sha256": common.canonical_sha256(collective),
            "local_gradient_validation": local_gradient_validation,
            "joint_routing_gradient_audit": gradient_audit,
            "optimizer_state_cpu_offload": {
                "enabled": True,
                "offload": (
                    None
                    if optimizer_state_offload is None
                    else optimizer_state_offload.__dict__
                ),
                "restore": (
                    None
                    if optimizer_state_restore is None
                    else optimizer_state_restore.__dict__
                ),
            },
            "gradient_cpu_accumulation": {
                "enabled": True,
                "local_rows": LOCAL_ROWS,
                "local_transfers": local_gradient_offload_tensors,
                "local_bytes": local_gradient_offload_bytes,
            },
            "control_graphs": (
                "bundled_condition_groups"
                if backward_bundle_fn is not None
                else "serialized_one_at_a_time"
            ),
            "learning_rate": learning_rate,
            "learning_rate_multipliers": dict(learning_rate_multipliers),
            "max_gradient_norm": max_grad_norm,
            "contrast_margin": margin,
            "control_weights": dict(control_weights),
            "always_active_controls": list(always_active_controls),
            "backward_control_names": list(backward_control_names),
            "baseline_anchor": {
                "weight": baseline_anchor_weight,
                "temperature": baseline_anchor_temperature,
                "top_k": baseline_anchor_top_k,
                "teacher_condition": "zero_recurrent_state",
            },
        }
        if context.is_primary:
            append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": step,
                        "correct_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "margins": {
                            key: round(value, 6)
                            for key, value in step_margins.items()
                        },
                        "active": step_active,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )
        total_correct_ce += metrics[0]
        for condition in control_names:
            total_margins[condition] += (
                step_margins[condition] * GLOBAL_BATCH_SIZE
            )
            total_active[condition] += step_active[condition]
        for task, value in task_rows.items():
            total_task_rows[task] += value
        for variant, value in prompt_variants.items():
            total_prompt_variants[variant] += value
        total_baseline_anchor_rows += baseline_anchor_rows
        total_baseline_anchor_tokens += baseline_anchor_tokens

    final_state = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = runtime._state_dict_sha256(final_state)
    final_route_sha256 = route_subset_sha256(final_state)
    final_trainable_sha256 = trainable_subset_sha256(named_trainable)
    for description, digest in (
        ("final recurrent-routed adapter", final_adapter_sha256),
        ("final recurrent-routed route subset", final_route_sha256),
        ("final recurrent-routed trainable subset", final_trainable_sha256),
    ):
        distributed.require_consensus(context, digest, description=description)
    denominator = updates * GLOBAL_BATCH_SIZE
    return {
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": denominator,
        "mean_correct_ce": total_correct_ce / denominator,
        "mean_control_minus_correct_ce": {
            condition: total_margins[condition] / denominator
            for condition in control_names
        },
        "active_control_fraction": {
            condition: total_active[condition] / denominator
            for condition in control_names
        },
        "task_rows": total_task_rows,
        "prompt_variant_rows": total_prompt_variants,
        "baseline_anchor": {
            "weight": baseline_anchor_weight,
            "temperature": baseline_anchor_temperature,
            "top_k": baseline_anchor_top_k,
            "teacher_condition": "zero_recurrent_state",
            "rows": total_baseline_anchor_rows,
            "tokens": total_baseline_anchor_tokens,
        },
        "minimum_gradient_norm_before_clip": minimum_gradient_norm,
        "maximum_global_inactive_parameter_tensors": maximum_global_inactive,
        "projected_carrier_fixed_every_row": projected_carrier_fixed_every_row,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "initial_route_subset_sha256": initial_route_sha256,
        "final_route_subset_sha256": final_route_sha256,
        "route_subset_changed": initial_route_sha256 != final_route_sha256,
        "initial_trainable_subset_sha256": initial_trainable_sha256,
        "final_trainable_subset_sha256": final_trainable_sha256,
        "trainable_subset_changed": (
            initial_trainable_sha256 != final_trainable_sha256
        ),
        "first_update_joint_routing_gradient_audit": first_gradient_audit,
        "optimizer_state_cpu_offload": {
            "enabled": True,
            "steps": optimizer_state_cpu_offload_steps,
            "maximum_tensors_per_rank": optimizer_state_cpu_offload_tensors,
            "maximum_bytes_per_rank": optimizer_state_cpu_offload_bytes,
        },
        "always_active_controls": list(always_active_controls),
        "backward_control_names": list(backward_control_names),
        "learning_rate_multipliers": dict(learning_rate_multipliers),
        "progress_sha256": (
            common.sha256_file(progress_path) if context.is_primary else None
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Recurrent-routed post-training requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    protocol = validate_training_protocol(updates)
    manifest, open_receipt = common.validate_split_artifacts()
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Recurrent-routed output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-output-freshness",
        error=freshness_error,
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, delta_config, model_audit = common.load_model(
        base_model,
        device=context.device,
        trainable=True,
    )
    named_trainable = model_audit.pop("named_trainable")
    rows_by_task = common.load_open_rows("train", manifest=manifest)
    schedule, schedule_payload = common.build_training_schedule(
        rows_by_task,
        updates=TRAIN_UPDATES,
    )
    schedule_sha256 = common.canonical_sha256(schedule_payload)
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": common.PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "seed": SEED,
        "updates": updates,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "base_model": str(base_model.expanduser().resolve()),
        "base_model_revision": common.BASE_MODEL_REVISION,
        "base_model_weights_sha256": common.BASE_MODEL_WEIGHTS_SHA256,
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "open_split_files": open_receipt["files"],
        "training_schedule_sha256": schedule_sha256,
        "schedule_prefix_sha256": common.canonical_sha256(
            schedule_payload[:updates]
        ),
        "prompt_variants_sha256": common.canonical_sha256(
            common.PROMPT_VARIANTS
        ),
        "training_read_max_tokens": TRAINING_READ_MAX_TOKENS,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "runner_sha256": common.sha256_file(Path(__file__)),
        "common_helper_sha256": common.sha256_file(Path(common.__file__)),
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    distributed.require_consensus(
        context,
        common.canonical_sha256(input_binding),
        description="recurrent-routed input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_fresh_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-input-binding-save",
        error=binding_error,
    )

    training = train(
        model,
        tokenizer,
        schedule,
        schedule_payload,
        updates=updates,
        context=context,
        output_dir=resolved_output,
        named_trainable=named_trainable,
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": training["peak_cuda_memory_bytes"],
        },
    )
    passed = (
        four_distinct_a100s(context.rank_devices)
        and training["initial_adapter_sha256"]
        != training["final_adapter_sha256"]
        and training["route_subset_changed"] is True
        and training["trainable_subset_changed"] is True
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["first_update_joint_routing_gradient_audit"]["passed"]
        is True
    )
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, delta_config)
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    "preflight_passed"
                    if updates == PREFLIGHT_UPDATES and passed
                    else "training_complete_development_evaluation_authorized"
                    if passed
                    else "training_failed_development_evaluation_blocked"
                ),
                "passed": passed,
                "protocol_payload_sha256": common.PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": updates,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_development_evaluation_authorized": (
                    passed and updates == TRAIN_UPDATES
                ),
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
                "code_bindings": {
                    "runner_sha256": common.sha256_file(Path(__file__)),
                    "common_helper_sha256": common.sha256_file(Path(common.__file__)),
                    "protocol_file_sha256": common.sha256_file(common.PROTOCOL),
                    "split_manifest_sha256": common.sha256_file(
                        common.SPLIT_ROOT / "manifest.json"
                    ),
                    "open_split_receipt_sha256": common.sha256_file(
                        common.SPLIT_ROOT / "open_split_receipt.json"
                    ),
                    "delta_impl_sha256": common.sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": common.sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": common.canonical_sha256(result),
            }
            write_fresh_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-result-save",
        error=save_error,
    )
    del model, tokenizer, rows_by_task
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": passed,
        "seed": SEED,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, 32))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Recurrent-routed post-training requires four-rank torchrun")
    try:
        result = run(
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
                "result_receipt": (
                    None
                    if not context.is_primary
                    else result["receipt"]["payload_sha256"]
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
