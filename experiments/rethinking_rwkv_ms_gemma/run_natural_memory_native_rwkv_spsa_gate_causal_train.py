#!/usr/bin/env python3
"""Optimize RWKV memory acceptance with forward-only SPSA causal contrasts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_train as benchmark_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as addressed_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_gate_bias_causal_train as gate_bias,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as screen_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_spsa_gate_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_spsa_gate_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_spsa_gate_causal_train_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_spsa_gate_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "c099c9a870a9732f8fb83530089e0fc4fc88331b96453e38522580c7d3c6a87c"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = shared.BASE_MODEL
DATASET_ROOT = shared.DATASET_ROOT
WORLD_SIZE = 4
SEED = 72
UPDATES = 8
PERTURBATION_SCALE = 0.1
ADAM_LEARNING_RATE = 0.05
MAX_ESTIMATED_GRAD_NORM = 1.0
MIN_GATE_BIAS = -3.0
MAX_GATE_BIAS = 1.0
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
FAILED_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_gate_bias_causal_train_failed_step2_v1/"
    "input_binding.json"
)
FAILED_INPUT_BINDING_SHA256 = (
    "ccba69721770a1857845e95695c49fed97eb59fe8238a4cbaa1eabbb7175b57f"
)
FAILED_PROGRESS = FAILED_INPUT_BINDING.parent / "training_progress.jsonl"
FAILED_PROGRESS_SHA256 = (
    "69e8d0240e3403d7e4625ea3ae876cb9abff827d0f4e6b51d6e37434657ced19"
)
SELECTED_CANDIDATE = gate_bias.SELECTED_CANDIDATE


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def spsa_direction(step: int, size: int) -> torch.Tensor:
    if step < 1 or size < 1:
        raise ValueError("SPSA direction requires a positive step and size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED * 1_000_003 + step)
    return (
        torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
        .mul(2)
        .sub(1)
        .to(torch.float32)
    )


def causal_hinge_objective(
    correct_ce: float,
    zero_ce: float,
    donor_ce: float,
    permuted_ce: float,
) -> float:
    margins = (
        zero_ce - correct_ce,
        donor_ce - correct_ce,
        permuted_ce - correct_ce,
    )
    return correct_ce + CONTRAST_WEIGHT * sum(
        max(0.0, MARGIN - value) for value in margins
    )


def estimated_spsa_gradient(
    plus_objective: float,
    minus_objective: float,
    direction: torch.Tensor,
) -> torch.Tensor:
    if not math.isfinite(plus_objective) or not math.isfinite(minus_objective):
        raise ValueError("SPSA objectives must be finite")
    if direction.ndim != 1 or not bool(direction.abs().eq(1).all().item()):
        raise ValueError("SPSA direction must be a one-dimensional Rademacher vector")
    coefficient = (plus_objective - minus_objective) / (
        2.0 * PERTURBATION_SCALE
    )
    return direction * coefficient


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("SPSA gate protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("SPSA gate protocol payload differs")
    if sha256_file(FAILED_INPUT_BINDING) != FAILED_INPUT_BINDING_SHA256:
        raise ValueError("Failed gate-bias input binding differs")
    if sha256_file(FAILED_PROGRESS) != FAILED_PROGRESS_SHA256:
        raise ValueError("Failed gate-bias progress binding differs")
    training = protocol.get("training", {})
    required = {
        "seed": SEED,
        "optimizer_updates": UPDATES,
        "spsa_perturbation_scale": PERTURBATION_SCALE,
        "adam_learning_rate": ADAM_LEARNING_RATE,
        "max_estimated_gradient_norm": MAX_ESTIMATED_GRAD_NORM,
        "gate_bias_bounds": [MIN_GATE_BIAS, MAX_GATE_BIAS],
    }
    if any(training.get(name) != value for name, value in required.items()):
        raise ValueError("SPSA gate training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("SPSA gate training may not open protected data")
    return protocol


def set_perturbed_parameters(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    centers: Sequence[torch.Tensor],
    direction: torch.Tensor,
    *,
    scale: float,
) -> None:
    if len(named_trainable) != len(centers) or len(named_trainable) != direction.numel():
        raise ValueError("SPSA parameter, center, and direction counts differ")
    with torch.no_grad():
        for index, ((_, parameter), center) in enumerate(zip(named_trainable, centers)):
            parameter.copy_(center + scale * float(direction[index].item()))


def evaluate_local_causal_objective(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule_step: contrast.ContrastScheduleStep,
    *,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
) -> Mapping[str, Any]:
    model.eval()
    modules = causal_train.ordered_modules(model)
    local_start = context.process_rank * causal_train.LOCAL_ROWS
    local_sources = schedule_step.source_ordinals[
        local_start : local_start + causal_train.LOCAL_ROWS
    ]
    if len(local_sources) != causal_train.LOCAL_ROWS:
        raise RuntimeError("SPSA local schedule size differs")
    local_metrics = [0.0] * 13
    with torch.no_grad():
        for source_ordinal in local_sources:
            source_offset = schedule_step.source_ordinals.index(source_ordinal)
            donor_ordinal = schedule_step.donor_ordinals[source_offset]
            target_batch = evolution.collate_native_examples(
                [rows[source_ordinal].example],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            donor_batch = contrast.build_donor_batch(
                target_batch,
                rows[donor_ordinal].example,
                device=context.device,
            )
            evolution._native_write(model, target_batch, dtype=torch.bfloat16)
            correct_state = causal_train.capture_online_state_references(modules)
            correct_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            evolution._native_write(model, donor_batch, dtype=torch.bfloat16)
            donor_state = causal_train.capture_online_state_references(modules)
            donor_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=donor_state,
                rotate_recurrent_layers=False,
            )
            donor_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            permuted_carrier_fixed = causal_train.install_intervened_state(
                modules,
                projected=correct_state,
                recurrent=correct_state,
                rotate_recurrent_layers=True,
            )
            permuted_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            reset_delta_mem_states(model)
            zero_logits = evolution._native_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            condition_logits = (
                correct_logits,
                zero_logits,
                donor_logits,
                permuted_logits,
            )
            condition_values = [
                contrast.detached_answer_ce(logits, target_batch.labels)
                for logits in condition_logits
            ]
            token_counts = {tokens for _, tokens in condition_values}
            if len(token_counts) != 1:
                raise RuntimeError("SPSA causal condition token counts differ")
            correct_ce, zero_ce, donor_ce, permuted_ce = (
                value for value, _ in condition_values
            )
            objective = causal_hinge_objective(
                correct_ce,
                zero_ce,
                donor_ce,
                permuted_ce,
            )
            zero_margin = zero_ce - correct_ce
            donor_margin = donor_ce - correct_ce
            permuted_margin = permuted_ce - correct_ce
            carrier_fixed = bool(donor_carrier_fixed and permuted_carrier_fixed)
            finite = math.isfinite(objective) and all(
                bool(torch.isfinite(logits).all().item())
                for logits in condition_logits
            )
            values = (
                objective,
                correct_ce,
                zero_ce,
                donor_ce,
                permuted_ce,
                zero_margin,
                donor_margin,
                permuted_margin,
                float(zero_margin < MARGIN),
                float(donor_margin < MARGIN),
                float(permuted_margin < MARGIN),
                float(carrier_fixed),
                float(finite),
            )
            local_metrics = [
                total + value for total, value in zip(local_metrics, values)
            ]
            del target_batch, donor_batch, correct_state, donor_state
            del correct_logits, zero_logits, donor_logits, permuted_logits
            reset_delta_mem_states(model)
            evolution.release_native_row_allocator_cache(context.device)
    metric_tensor = contrast.gate._prepare_distributed_scalar_sums(
        context,
        local_metrics,
    )
    metrics = contrast.gate._distributed_scalar_sums(context, metric_tensor)
    count = causal_train.GLOBAL_BATCH_SIZE
    if metrics[11] != count or metrics[12] != count:
        raise RuntimeError("SPSA forward causal invariants failed")
    return {
        "objective": metrics[0] / count,
        "mean_correct_ce": metrics[1] / count,
        "mean_zero_ce": metrics[2] / count,
        "mean_donor_ce": metrics[3] / count,
        "mean_layer_permuted_ce": metrics[4] / count,
        "mean_zero_minus_correct_ce": metrics[5] / count,
        "mean_donor_minus_correct_ce": metrics[6] / count,
        "mean_layer_permuted_minus_correct_ce": metrics[7] / count,
        "active_zero_rows": int(metrics[8]),
        "active_donor_rows": int(metrics[9]),
        "active_layer_permuted_rows": int(metrics[10]),
        "projected_carrier_fixed_rows": int(metrics[11]),
        "finite_rows": int(metrics[12]),
    }


def train_forward_only(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[contrast.ContrastScheduleStep],
    *,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    parameters = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=ADAM_LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    initial_state = snapshot_delta_mem_weights(model)
    initial_adapter_sha256 = runtime._state_dict_sha256(initial_state)
    initial_recurrent_sha256 = benchmark_train.state_subset_sha256(
        initial_state,
        recurrent_only=True,
    )
    initial_trainable_sha256 = causal_train.trainable_subset_sha256(named_trainable)
    for description, value in (
        ("initial SPSA adapter", initial_adapter_sha256),
        ("initial SPSA recurrent subset", initial_recurrent_sha256),
        ("initial SPSA trainable subset", initial_trainable_sha256),
    ):
        distributed.require_consensus(context, value, description=description)

    progress_path = output_dir / "training_progress.jsonl"
    perturbation_differences: list[float] = []
    gradient_norms: list[float] = []
    all_forward_rows_finite = True
    projected_carrier_fixed_every_row = True
    started = time.time()
    for schedule_step in schedule[:UPDATES]:
        centers = [parameter.detach().clone() for parameter in parameters]
        direction = spsa_direction(schedule_step.step, len(parameters))
        set_perturbed_parameters(
            named_trainable,
            centers,
            direction,
            scale=PERTURBATION_SCALE,
        )
        plus = evaluate_local_causal_objective(
            model,
            rows,
            schedule_step,
            context=context,
            pad_token_id=pad_token_id,
        )
        set_perturbed_parameters(
            named_trainable,
            centers,
            direction,
            scale=-PERTURBATION_SCALE,
        )
        minus = evaluate_local_causal_objective(
            model,
            rows,
            schedule_step,
            context=context,
            pad_token_id=pad_token_id,
        )
        set_perturbed_parameters(
            named_trainable,
            centers,
            direction,
            scale=0.0,
        )
        estimate = estimated_spsa_gradient(
            plus["objective"],
            minus["objective"],
            direction,
        )
        optimizer.zero_grad(set_to_none=True)
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, float(estimate[index].item()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            MAX_ESTIMATED_GRAD_NORM,
        )
        gradient_norm_value = float(gradient_norm.detach().float().item())
        if not math.isfinite(gradient_norm_value):
            raise RuntimeError("SPSA estimated gradient norm is non-finite")
        optimizer.step()
        with torch.no_grad():
            for parameter in parameters:
                parameter.clamp_(min=MIN_GATE_BIAS, max=MAX_GATE_BIAS)
        objective_difference = plus["objective"] - minus["objective"]
        perturbation_differences.append(objective_difference)
        gradient_norms.append(gradient_norm_value)
        all_forward_rows_finite = all_forward_rows_finite and all(
            branch["finite_rows"] == causal_train.GLOBAL_BATCH_SIZE
            for branch in (plus, minus)
        )
        projected_carrier_fixed_every_row = (
            projected_carrier_fixed_every_row
            and all(
                branch["projected_carrier_fixed_rows"]
                == causal_train.GLOBAL_BATCH_SIZE
                for branch in (plus, minus)
            )
        )
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "direction_sha256": canonical_sha256(direction.tolist()),
            "plus": plus,
            "minus": minus,
            "objective_difference": objective_difference,
            "estimated_gradient_norm_before_clip": gradient_norm_value,
            "gate_bias_min": min(float(parameter.item()) for parameter in parameters),
            "gate_bias_max": max(float(parameter.item()) for parameter in parameters),
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
        }
        if context.is_primary:
            causal_train.append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "plus_objective": round(plus["objective"], 6),
                        "minus_objective": round(minus["objective"], 6),
                        "difference": round(objective_difference, 8),
                        "gradient_norm": round(gradient_norm_value, 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_state = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = runtime._state_dict_sha256(final_state)
    final_recurrent_sha256 = benchmark_train.state_subset_sha256(
        final_state,
        recurrent_only=True,
    )
    final_trainable_sha256 = causal_train.trainable_subset_sha256(named_trainable)
    for description, value in (
        ("final SPSA adapter", final_adapter_sha256),
        ("final SPSA recurrent subset", final_recurrent_sha256),
        ("final SPSA trainable subset", final_trainable_sha256),
    ):
        distributed.require_consensus(context, value, description=description)
    return {
        "optimization": "forward_only_spsa_adam",
        "updates": UPDATES,
        "rows_per_perturbation_branch": UPDATES * causal_train.GLOBAL_BATCH_SIZE,
        "forward_causal_row_evaluations": (
            2 * UPDATES * causal_train.GLOBAL_BATCH_SIZE
        ),
        "elapsed_seconds": time.time() - started,
        "perturbation_scale": PERTURBATION_SCALE,
        "adam_learning_rate": ADAM_LEARNING_RATE,
        "minimum_abs_objective_difference": min(
            abs(value) for value in perturbation_differences
        ),
        "maximum_abs_objective_difference": max(
            abs(value) for value in perturbation_differences
        ),
        "nonzero_objective_differences": sum(
            value != 0.0 for value in perturbation_differences
        ),
        "minimum_estimated_gradient_norm": min(gradient_norms),
        "maximum_estimated_gradient_norm": max(gradient_norms),
        "all_forward_rows_finite": all_forward_rows_finite,
        "projected_carrier_fixed_every_row": projected_carrier_fixed_every_row,
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "initial_recurrent_subset_sha256": initial_recurrent_sha256,
        "final_recurrent_subset_sha256": final_recurrent_sha256,
        "recurrent_subset_changed": initial_recurrent_sha256 != final_recurrent_sha256,
        "initial_trainable_subset_sha256": initial_trainable_sha256,
        "final_trainable_subset_sha256": final_trainable_sha256,
        "trainable_subset_changed": initial_trainable_sha256 != final_trainable_sha256,
        "progress_sha256": sha256_file(progress_path) if context.is_primary else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("SPSA gate training requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    calibration_result = shared.validate_calibration_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("SPSA gate pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(f"SPSA gate output must be fresh: {resolved_output}")
    distributed.phase_consensus(
        context,
        phase="spsa-gate-output-freshness",
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
        phase="spsa-gate-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = screen.load_model(
        base_model,
        device=context.device,
        candidate=SELECTED_CANDIDATE,
    )
    screen_helper.configure_candidate(model, SELECTED_CANDIDATE)
    named_trainable, trainable_audit = gate_bias.configure_gate_bias_parameters(model)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    schedule, schedule_payload = contrast.build_schedule(rows, donor_mapping, donor_deltas)
    endpoint_payload = shared.heldout_payload(rows, donor_mapping)
    training_used = {
        ordinal
        for step in schedule[:UPDATES]
        for ordinal in (*step.source_ordinals, *step.donor_ordinals)
    }
    endpoint_disjoint = all(
        row["source_ordinal"] not in training_used
        and row["donor_ordinal"] not in training_used
        for row in endpoint_payload
    )
    if (
        canonical_sha256(schedule_payload) != contrast.FULL_SCHEDULE_SHA256
        or canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256
        or canonical_sha256(schedule_payload[:UPDATES]) != shared.TRAINING_PREFIX_SHA256
        or canonical_sha256(endpoint_payload) != shared.HELDOUT_PAYLOAD_SHA256
        or not endpoint_disjoint
    ):
        raise RuntimeError("SPSA gate training or endpoint binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "calibration_result_file_sha256": shared.CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": shared.CALIBRATION_RESULT_RECEIPT,
        "calibration_status": calibration_result["status"],
        "selected_candidate": SELECTED_CANDIDATE,
        "optimization": "forward_only_spsa_adam",
        "seed": SEED,
        "updates": UPDATES,
        "spsa_perturbation_scale": PERTURBATION_SCALE,
        "adam_learning_rate": ADAM_LEARNING_RATE,
        "max_estimated_gradient_norm": MAX_ESTIMATED_GRAD_NORM,
        "gate_bias_bounds": [MIN_GATE_BIAS, MAX_GATE_BIAS],
        "world_size": WORLD_SIZE,
        "global_batch_size": causal_train.GLOBAL_BATCH_SIZE,
        "local_rows": causal_train.LOCAL_ROWS,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "training_schedule_sha256": canonical_sha256(schedule_payload),
        "schedule_prefix_sha256": canonical_sha256(schedule_payload[:UPDATES]),
        "heldout_payload_sha256": canonical_sha256(endpoint_payload),
        "heldout_rows": len(endpoint_payload),
        "heldout_disjoint_from_training": endpoint_disjoint,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "trainable_audit": trainable_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="SPSA gate input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            shared.write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="spsa-gate-input-binding-save",
        error=binding_error,
    )

    training = train_forward_only(
        model,
        rows,
        schedule,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        output_dir=resolved_output,
        named_trainable=named_trainable,
    )
    endpoint = shared.evaluate_heldout_causal_endpoint(
        model,
        rows,
        donor_mapping,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
        },
    )
    training_passed = (
        addressed_screen.four_distinct_a100s(context.rank_devices)
        and trainable_audit["passed"] is True
        and training["trainable_subset_changed"] is True
        and training["recurrent_subset_changed"] is False
        and training["nonzero_objective_differences"] == UPDATES
        and training["all_forward_rows_finite"] is True
        and training["projected_carrier_fixed_every_row"] is True
    )
    passed = training_passed and endpoint["passed"] is True
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(
                model,
                adapter_dir,
                screen.build_config(SELECTED_CANDIDATE),
            )
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    "spsa_gate_heldout_passed_generation_authorized"
                    if passed
                    else "spsa_gate_heldout_failed_generation_blocked"
                ),
                "passed": passed,
                "training_passed": training_passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": UPDATES,
                "input_binding": input_binding,
                "training": training,
                "heldout_causal_endpoint": endpoint,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_native_generation_authorized": passed,
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "failed_gate_input_binding_sha256": sha256_file(
                        FAILED_INPUT_BINDING
                    ),
                    "failed_gate_progress_sha256": sha256_file(FAILED_PROGRESS),
                    "shared_endpoint_runner_sha256": sha256_file(Path(shared.__file__)),
                    "delta_impl_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            shared.write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="spsa-gate-result-save",
        error=save_error,
    )
    del model, rows
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
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("SPSA gate training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            base_model=args.base_model,
            dataset_root=args.dataset_root,
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
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
