#!/usr/bin/env python3
"""Train and audit Delta-Mem on the natural four-slot causal memory gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import torch
import torch.distributed as torch_dist
from torch.distributed.elastic.multiprocessing.errors import record

from deltamem.core import delta as delta_core
from deltamem.core import delta_impl
from deltamem.core.delta import (
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source
from experiments.rethinking_rwkv_ms_gemma import (
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


RUN_SCHEMA = "rwkv_ms_natural_memory_gate_run.v3"
EVALUATION_SCHEMA = "rwkv_ms_natural_memory_gate_evaluation.v3"
PROTOCOL_SCHEMA = "rwkv_ms_natural_memory_gate_protocol.v3"
TRAINING_CONFIGURATION_SCHEMA = (
    "rwkv_ms_natural_memory_gate_training_configuration.v3"
)
TRAIN_STEP_SCHEMA = "rwkv_ms_natural_memory_gate_train_step.v4"
DISTRIBUTED_PREFLIGHT_SCHEMA = "rwkv_ms_natural_memory_distributed_preflight.v3"
DISTRIBUTED_PREFLIGHT_GATE_SCHEMA = (
    "rwkv_ms_natural_memory_distributed_preflight_gate.v3"
)
REPLICATION_PROTOCOL_SCHEMA = "rwkv_ms_natural_memory_replication_protocol.v1"
REPLICATION_AMENDMENT_SCHEMA = (
    "rwkv_ms_natural_memory_replication_protocol_amendment.v1"
)
REPLICATION_AUTHORIZATION_SCHEMA = (
    "rwkv_ms_natural_memory_replication_authorization.v1"
)
ACCEPTANCE_SCHEMA = "rwkv_ms_natural_memory_gate_acceptance.v1"
TRAINING_DATASET_AUDIT_SCHEMA = "rwkv_ms_natural_memory_training_dataset_audit.v1"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
RECORDS_PER_EPISODE = 4
CONDITIONS = (
    "correct_state",
    "donor_state",
    "value_swap",
    "target_slot_rewrite",
    "shuffled_slots",
    "no_state",
    "pristine_frozen_base",
)
POSITIVE_CONDITIONS = CONDITIONS[:5]
SHARED_WRITE_CONDITIONS = (
    "correct_state",
    "donor_state",
    "value_swap",
    "shuffled_slots",
)
COUNTERFACTUAL_STATE_CONDITIONS = (
    "donor_state",
    "value_swap",
    "target_slot_rewrite",
    "shuffled_slots",
)
SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS = POSITIVE_CONDITIONS
BASELINE_TRAINING_CONDITIONS = ("correct_state",)
DEFAULT_TRAINING_CONDITIONS = SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS
CONTROL_CONDITIONS = CONDITIONS[5:]
PROFILES = ("train", "development", "sealed_validation")
FORMAL_PROFILES = ("development", "sealed_validation")
DEFAULT_TARGET_LAYERS = tuple(range(42))
PRODUCTION_SEED = 42
PREREGISTERED_REPLICATION_PROTOCOL_COMMIT = (
    "15566239208b57bd2520ef8333e5ac1a0c7c7287"
)
PREREGISTERED_REPLICATION_PROTOCOL_FILE_SHA256 = (
    "298ba1d921dffae456344d1bdc8b2d6e4244be0efc080b8c56c1498977bcc423"
)
PREREGISTERED_REPLICATION_PROTOCOL_PAYLOAD_SHA256 = (
    "b7dfd59294f97ccd8034cb604840b842857ca4a4b64cb6fa74a83d1ac2c87389"
)
PREREGISTERED_REPLICATION_RUNNER_SHA256 = (
    "6372f6197dd1152a22c9352fe1cdd62d472877ed965db5c6a841428a98095687"
)
PRODUCTION_EPOCHS = 8
PRODUCTION_TASKS = ("attribution", "narrative", "scene")
PRODUCTION_ROWS_PER_CONDITION_TASK = 128
PRODUCTION_TRAINING_ROWS = (
    len(SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS)
    * len(PRODUCTION_TASKS)
    * PRODUCTION_ROWS_PER_CONDITION_TASK
)
PRODUCTION_UPDATES = (
    PRODUCTION_TRAINING_ROWS
    * PRODUCTION_EPOCHS
    // distributed.REQUIRED_GLOBAL_BATCH_SIZE
)
PRODUCTION_ADAPTER_RANK = 32
# The first sealed run exposed two genuine held-out content-address collisions
# at layer 19.  Widen only the projected address key; rank, value path, losses,
# and all acceptance thresholds remain frozen.
PRODUCTION_KEY_DIM = 64
PRODUCTION_TEMPERATURE = 16.0
PRODUCTION_EVAL_BATCH_SIZE = 8
PRODUCTION_LEARNING_RATE = 2e-4
PRODUCTION_ANSWER_WEIGHT = 1.0
PRODUCTION_ROUTE_WEIGHT = 1.0
# The hard-negative screen improved correct-state answers but broadened development
# route errors and regressed several counterfactual answer metrics.
# Return to the stronger CE-only baseline while the larger four-GPU batch reduces
# noisy late optimizer updates from 3,840 to 960 over the same eight epochs.
PRODUCTION_HARD_NEGATIVE_MARGIN = 0.0
PRODUCTION_HARD_NEGATIVE_WEIGHT = 0.0
PRODUCTION_MAX_GRAD_NORM = 1.0
PRODUCTION_DTYPE = "bfloat16"
PRODUCTION_ATTN_IMPLEMENTATION = "sdpa"
PRODUCTION_ANSWER_EXACT_MIN = 0.80
PRODUCTION_ROUTE_ACCURACY_MIN = 0.95
PRODUCTION_REWRITE_OUTPUT_CHANGE_MIN = 0.80
DISTRIBUTED_PREFLIGHT_STEPS = 3
MINIMUM_DISTRIBUTED_HEADROOM_BYTES = 5 * 1024**3
CURRENT_PRODUCTION_COMPLETE_SCHEDULE = (
    PRODUCTION_EPOCHS,
    distributed.REQUIRED_GLOBAL_BATCH_SIZE,
    PRODUCTION_UPDATES,
)
LEGACY_LOCAL_BATCH_ONE_COMPLETE_SCHEDULE = (8, 4, 3840)
RETAINED_PRODUCTION_COMPLETE_SCHEDULES = (
    CURRENT_PRODUCTION_COMPLETE_SCHEDULE,
    LEGACY_LOCAL_BATCH_ONE_COMPLETE_SCHEDULE,
)
DISTRIBUTED_STEP_PHASE_ORDER = (
    "microbatch_preparation",
    "objective_denominator_preparation",
    "objective_denominator_global_sum",
    "microbatch_1_forward",
    "microbatch_1_backward",
    "microbatch_1_online_state_reset",
    "microbatch_2_forward",
    "microbatch_2_backward",
    "microbatch_2_online_state_reset",
    "objective_preparation",
    "objective_global_sum",
    "local_gradient_validation",
    "gradient_sum",
    "global_gradient_clip",
    "adamw_step",
    "metric_global_sum",
)
_T = TypeVar("_T")
SHARED_STATE_BATCHING_POLICY = (
    "complete four-query shared-write families are kept in one evaluation batch"
)
ANSWER_LOGIT_POLICY = (
    "full-sequence hidden states with vocabulary logits projected only at the union "
    "of supervised causal answer-predictor positions; ignored labels and token-mean "
    "cross-entropy are unchanged"
)
TRAINING_ROW_ID_POLICY = (
    "source query ID plus an explicit positive-condition suffix; evaluation IDs remain "
    "source query IDs for cross-condition causal pairing"
)
TRAINING_SAMPLING_POLICY = (
    "exactly balanced condition-task strata, shuffled over complete epochs; every "
    "source query is supervised once under every selected positive condition per epoch"
)
TRAINING_PAYLOAD_DIGEST_POLICY = (
    "canonical SHA-256 over every ordered encoded training-example field used by the "
    "write, read, routing, and answer objectives"
)
TRAINING_FAMILY_INVARIANT_POLICY = (
    "condition variants of one source query retain source split, episode, task, "
    "semantic target, mapping offset, and encoded query prefix"
)


# These functions operate on a small duck-typed example/batch interface and are
# shared with the synthetic causal proof runner.
build_delta_config = runtime.build_delta_config
collate_examples = runtime.collate_examples
causal_answer_loss = runtime.causal_answer_loss
route_loss_and_predictions = runtime.route_loss_and_predictions
selected_route_logits = runtime.selected_route_logits
_write_episode_batch = runtime._write_episode_batch
_read_episode_batch = runtime._read_episode_batch
_answer_prediction_token_ids = runtime._answer_prediction_token_ids
_answer_exact_predictions = runtime._answer_exact_predictions
_greedy_answer_predictions = runtime._greedy_answer_predictions
_state_digests = runtime._state_digests
_dtype = runtime._dtype
_signed_payload = runtime._signed_payload
_state_dict_sha256 = runtime._state_dict_sha256
_load_model_and_tokenizer = runtime._load_model_and_tokenizer


def train_model(
    model: torch.nn.Module,
    examples: Sequence[Any],
    *,
    seed: int,
    epochs: int,
    max_steps: int | None,
    batch_size: int,
    learning_rate: float,
    answer_weight: float,
    route_weight: float,
    max_grad_norm: float,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    progress_path: Path,
    training_conditions: str | Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
    hard_negative_margin: float = 0.0,
    hard_negative_weight: float = 0.0,
) -> dict[str, Any]:
    """Reuse the proven optimizer loop while emitting natural-run evidence."""

    runtime_progress = progress_path.with_name(
        f".{progress_path.name}.synthetic-runtime.tmp"
    )
    if progress_path.exists() or runtime_progress.exists():
        raise ValueError("Natural training progress paths must be fresh")
    try:
        result = dict(
            runtime.train_model(
                model,
                examples,
                seed=seed,
                epochs=epochs,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                answer_weight=answer_weight,
                route_weight=route_weight,
                max_grad_norm=max_grad_norm,
                pad_token_id=pad_token_id,
                device=device,
                dtype=dtype,
                progress_path=runtime_progress,
                hard_negative_margin=hard_negative_margin,
                hard_negative_weight=hard_negative_weight,
            )
        )
        records: list[dict[str, Any]] = []
        with runtime_progress.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = _require_mapping(
                    json.loads(line), f"runtime training step {line_number}"
                )
                if record.get("schema") != "rwkv_ms_synthetic_compositional_train_step.v3":
                    raise ValueError("Shared optimizer emitted an unexpected progress schema")
                natural_record = dict(record)
                natural_record["schema"] = TRAIN_STEP_SCHEMA
                natural_record["training_conditions"] = list(
                    _parse_training_conditions(training_conditions)
                )
                records.append(natural_record)
        if not records or len(records) != result.get("steps"):
            raise ValueError("Natural training progress does not bind every optimizer step")
        progress_path.write_text(
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        result["progress_schema"] = TRAIN_STEP_SCHEMA
        result["progress_sha256"] = source.sha256_file(progress_path)
        return result
    finally:
        runtime_progress.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _run_consensused_local_phase(
    context: distributed.DistributedTrainingContext,
    *,
    phase: str,
    operation: Callable[[], _T],
) -> _T:
    """Run rank-local work and publish failure before any later collective."""

    result: _T | None = None
    local_error: BaseException | None = None
    try:
        result = operation()
    except BaseException as error:
        local_error = error
    distributed.phase_consensus(context, phase=phase, error=local_error)
    if local_error is not None:
        raise local_error
    return result  # type: ignore[return-value]


def _named_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )


def _named_adapter_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    parameters: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in iter_delta_mem_modules(model):
        for sub_name, parameter in module.named_parameters():
            if not sub_name.startswith("base."):
                parameters.append((f"{module_name}.{sub_name}", parameter))
    return distributed.stable_named_parameters(parameters)


def _prepare_distributed_scalar_sums(
    context: distributed.DistributedTrainingContext,
    values: Sequence[int | float],
) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("Distributed scalar statistics must be a nonempty vector")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("Distributed scalar statistics must be finite")
    return tensor


def _distributed_scalar_sums(
    context: distributed.DistributedTrainingContext,
    tensor: torch.Tensor,
) -> tuple[float, ...]:
    if tensor.device != context.device or tensor.dtype != torch.float64:
        raise ValueError("Prepared scalar statistics have the wrong device or dtype")
    torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.SUM)
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError("Reduced scalar statistics must be finite")
    return tuple(float(value) for value in tensor.tolist())


def _global_router_gradient_audit(
    context: distributed.DistributedTrainingContext,
    local_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    rank_audits = distributed.gather_objects(context, dict(local_audit))
    expected_modules = local_audit.get("modules")
    passed = (
        type(expected_modules) is int
        and expected_modules > 0
        and all(
            audit.get("modules") == expected_modules
            and audit.get("all_modules_finite_nonzero") is True
            for audit in rank_audits
        )
    )
    return {
        "world_size": context.world_size,
        "modules_per_rank": expected_modules,
        "all_ranks_all_modules_finite_nonzero": passed,
        "all_modules_finite_nonzero": passed,
        "rank_audits": list(rank_audits),
    }


def train_model_distributed(
    model: torch.nn.Module,
    examples: Sequence[Any],
    *,
    context: distributed.DistributedTrainingContext,
    seed: int,
    epochs: int,
    max_steps: int | None,
    global_batch_size: int,
    learning_rate: float,
    answer_weight: float,
    route_weight: float,
    max_grad_norm: float,
    pad_token_id: int,
    dtype: torch.dtype,
    progress_path: Path,
    training_conditions: str | Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
    capture_step_evidence: bool = False,
    hard_negative_margin: float = 0.0,
    hard_negative_weight: float = 0.0,
) -> dict[str, Any]:
    """Train raw replicas with global normalization and explicit gradient SUM."""

    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("Training epochs and learning rate must be positive")
    if answer_weight <= 0.0 or route_weight <= 0.0:
        raise ValueError("Both answer and route loss weights must be positive")
    if hard_negative_margin < 0.0 or hard_negative_weight < 0.0:
        raise ValueError("Hard-negative margin and weight must be nonnegative")
    if global_batch_size <= 0 or global_batch_size % context.world_size:
        raise ValueError("Global batch size must divide evenly across ranks")
    local_batch_size = global_batch_size // context.world_size
    local_microbatch_size = min(
        distributed.REQUIRED_LOCAL_MICROBATCH_SIZE, local_batch_size
    )
    if local_batch_size % local_microbatch_size:
        raise ValueError("Local batch size must divide evenly into microbatches")
    gradient_accumulation_steps = local_batch_size // local_microbatch_size

    def prepare_schedule() -> tuple[
        list[str], tuple[distributed.GlobalTrainingStep, ...], str
    ]:
        prepared_row_ids = [str(example.row_id) for example in examples]
        prepared_schedule, prepared_hash = distributed.build_global_training_schedule(
            prepared_row_ids,
            seed=seed,
            epochs=epochs,
            max_steps=max_steps,
            world_size=context.world_size,
            local_batch_size=local_batch_size,
        )
        return prepared_row_ids, prepared_schedule, prepared_hash

    row_ids, schedule, schedule_sha256 = _run_consensused_local_phase(
        context,
        phase="training-schedule-preparation",
        operation=prepare_schedule,
    )
    distributed.require_consensus(
        context,
        {
            "ordered_row_ids_sha256": distributed.canonical_sha256(row_ids),
            "schedule_sha256": schedule_sha256,
            "steps": len(schedule),
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
            "local_microbatch_size": local_microbatch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        description="training schedule",
    )

    def prepare_training_runtime() -> tuple[
        tuple[tuple[str, torch.nn.Parameter], ...],
        list[torch.nn.Parameter],
        tuple[dict[str, Any], ...],
        str,
        torch.optim.AdamW,
        Mapping[str, Any],
        tuple[str, ...],
    ]:
        prepared_named_trainable = _named_trainable_parameters(model)
        prepared_trainable = [parameter for _, parameter in prepared_named_trainable]
        prepared_metadata = distributed.named_tensor_metadata(
            prepared_named_trainable
        )
        prepared_metadata_hash = distributed.canonical_sha256(prepared_metadata)
        prepared_optimizer = torch.optim.AdamW(
            prepared_trainable,
            lr=learning_rate,
            weight_decay=0.0,
            fused=context.device.type == "cuda",
        )
        model.train()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        prepared_memory = distributed.cuda_memory_snapshot(context)
        prepared_conditions = _parse_training_conditions(training_conditions)
        return (
            prepared_named_trainable,
            prepared_trainable,
            prepared_metadata,
            prepared_metadata_hash,
            prepared_optimizer,
            prepared_memory,
            prepared_conditions,
        )

    (
        named_trainable,
        trainable,
        trainable_metadata,
        trainable_metadata_sha256,
        optimizer,
        memory_before,
        selected_training_conditions,
    ) = _run_consensused_local_phase(
        context,
        phase="training-runtime-preparation",
        operation=prepare_training_runtime,
    )
    distributed.require_consensus(
        context,
        trainable_metadata_sha256,
        description="trainable parameter metadata",
    )
    trainable_names_sha256 = distributed.canonical_sha256(
        [value["name"] for value in trainable_metadata]
    )
    totals = {
        "answer_loss": 0.0,
        "route_loss": 0.0,
        "total_loss": 0.0,
        "answer_exact_correct": 0.0,
        "answer_rows": 0.0,
        "route_correct": 0.0,
        "route_total": 0.0,
        "full_occupancy_count": 0.0,
        "full_occupancy_total": 0.0,
        "forced_write_route_correct": 0.0,
        "forced_write_route_total": 0.0,
    }
    router_gradient: Mapping[str, Any] | None = None
    collective_evidence: Mapping[str, Any] | None = None
    collective_evidence_by_step: list[Mapping[str, Any]] = []
    step_evidence: list[Mapping[str, Any]] = []
    started = time.time()
    for schedule_step in schedule:
        observed_phase_order: list[str] = []

        def prepare_microbatches() -> tuple[
            tuple[int, ...], list[Any], list[Any], int, int
        ]:
            prepared_local_indices = distributed.local_step_indices(
                schedule_step,
                process_rank=context.process_rank,
                world_size=context.world_size,
                local_batch_size=local_batch_size,
            )
            prepared_selected = [examples[index] for index in prepared_local_indices]
            prepared_batches = [
                collate_examples(
                    prepared_selected[start : start + local_microbatch_size],
                    pad_token_id=pad_token_id,
                    device=context.device,
                )
                for start in range(0, local_batch_size, local_microbatch_size)
            ]
            if len(prepared_batches) != gradient_accumulation_steps:
                raise RuntimeError("Prepared the wrong number of local microbatches")
            prepared_answer_tokens = sum(
                int(batch.labels[:, 1:].ne(-100).sum().item())
                for batch in prepared_batches
            )
            prepared_route_rows = sum(
                int(batch.target_slots.numel()) for batch in prepared_batches
            )
            if prepared_answer_tokens <= 0 or prepared_route_rows != local_batch_size:
                raise RuntimeError("Prepared microbatch denominators are invalid")
            optimizer.zero_grad(set_to_none=True)
            return (
                prepared_local_indices,
                prepared_selected,
                prepared_batches,
                prepared_answer_tokens,
                prepared_route_rows,
            )

        (
            local_indices,
            selected,
            microbatch_batches,
            local_answer_tokens,
            local_route_rows,
        ) = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-microbatch-preparation",
            operation=prepare_microbatches,
        )
        observed_phase_order.append("microbatch_preparation")

        local_denominator_statistics = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-objective-denominator-preparation",
            operation=lambda: _prepare_distributed_scalar_sums(
                context, (local_answer_tokens, local_route_rows)
            ),
        )
        observed_phase_order.append("objective_denominator_preparation")

        global_denominators = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-objective-denominator-global-sum",
            operation=lambda: _distributed_scalar_sums(
                context, local_denominator_statistics
            ),
        )
        observed_phase_order.append("objective_denominator_global_sum")
        global_answer_tokens, global_route_rows = (
            int(value) for value in global_denominators
        )
        if (
            tuple(float(value) for value in (global_answer_tokens, global_route_rows))
            != global_denominators
            or global_answer_tokens <= 0
            or global_route_rows != global_batch_size
        ):
            raise distributed.DistributedTrainingError(
                "Global microbatch denominators are invalid"
            )

        local_answer_loss_sum = 0.0
        local_route_loss_sum = 0.0
        local_online_state_digests: list[str] = []
        local_metric_values = [0.0] * 8

        for microbatch_offset, batch in enumerate(microbatch_batches):
            microbatch_number = microbatch_offset + 1
            microbatch_selected = selected[
                microbatch_offset
                * local_microbatch_size : (microbatch_offset + 1)
                * local_microbatch_size
            ]

            def forward_microbatch() -> tuple[
                Mapping[str, Any],
                torch.Tensor,
                Mapping[str, torch.Tensor],
                list[str],
                torch.Tensor,
                int,
                torch.Tensor,
                int,
                Mapping[str, torch.Tensor],
            ]:
                prepared_write_audit = _write_episode_batch(
                    model, batch, dtype=dtype
                )
                prepared_logits, prepared_route_logits = _read_episode_batch(
                    model, batch, dtype=dtype
                )
                prepared_state_digests = (
                    _state_digests(model, len(microbatch_selected))
                    if capture_step_evidence
                    else []
                )
                prepared_answer_sum, prepared_answer_tokens = (
                    distributed.answer_loss_sum_and_count(
                        prepared_logits, batch.labels
                    )
                )
                (
                    prepared_route_sum,
                    prepared_route_rows,
                    prepared_route_predictions,
                ) = distributed.route_loss_sum_and_predictions(
                    prepared_route_logits,
                    batch.query_mask,
                    batch.target_slots,
                    hard_negative_margin=hard_negative_margin,
                    hard_negative_weight=hard_negative_weight,
                )
                return (
                    prepared_write_audit,
                    prepared_logits,
                    prepared_route_logits,
                    prepared_state_digests,
                    prepared_answer_sum,
                    prepared_answer_tokens,
                    prepared_route_sum,
                    prepared_route_rows,
                    prepared_route_predictions,
                )

            (
                write_audit,
                logits,
                route_logits,
                state_digests,
                answer_sum,
                answer_tokens,
                route_sum,
                route_rows,
                route_predictions,
            ) = _run_consensused_local_phase(
                context,
                phase=(
                    f"step-{schedule_step.step}-microbatch-{microbatch_number}-forward"
                ),
                operation=forward_microbatch,
            )
            observed_phase_order.append(f"microbatch_{microbatch_number}_forward")
            if answer_tokens <= 0 or route_rows != len(microbatch_selected):
                raise distributed.DistributedTrainingError(
                    f"Microbatch {microbatch_number} objective counts are invalid"
                )

            def prepare_microbatch_objective() -> tuple[
                torch.Tensor, Mapping[str, Any] | None
            ]:
                prepared_total_loss = (
                    answer_weight * answer_sum / global_answer_tokens
                    + route_weight * route_sum / global_route_rows
                )
                if not bool(torch.isfinite(prepared_total_loss).item()):
                    raise RuntimeError("Distributed objective is non-finite")
                prepared_router_gradient = (
                    runtime._router_gradient_audit(
                        model, route_sum / global_route_rows
                    )
                    if router_gradient is None and microbatch_number == 1
                    else None
                )
                return prepared_total_loss, prepared_router_gradient

            total_loss, local_router_gradient = _run_consensused_local_phase(
                context,
                phase=(
                    f"step-{schedule_step.step}-microbatch-{microbatch_number}-objective"
                ),
                operation=prepare_microbatch_objective,
            )
            if router_gradient is None and microbatch_number == 1:
                router_gradient = _run_consensused_local_phase(
                    context,
                    phase=f"step-{schedule_step.step}-router-gradient-audit",
                    operation=lambda: _global_router_gradient_audit(
                        context, local_router_gradient or {}
                    ),
                )

            _run_consensused_local_phase(
                context,
                phase=(
                    f"step-{schedule_step.step}-microbatch-{microbatch_number}-backward"
                ),
                operation=total_loss.backward,
            )
            observed_phase_order.append(f"microbatch_{microbatch_number}_backward")

            def prepare_microbatch_metrics() -> tuple[float, ...]:
                exact, _, _ = _answer_exact_predictions(
                    logits.detach(), batch.labels
                )
                route_matches = sum(
                    int(prediction.eq(batch.target_slots).sum().item())
                    for prediction in route_predictions.values()
                )
                route_count = len(route_predictions) * int(
                    batch.target_slots.numel()
                )
                return (
                    float(sum(exact)),
                    float(len(exact)),
                    float(route_matches),
                    float(route_count),
                    float(write_audit["full_occupancy_count"]),
                    float(write_audit["full_occupancy_total"]),
                    float(write_audit["forced_write_route_match_count"]),
                    float(write_audit["forced_write_route_total"]),
                )

            microbatch_metrics = _run_consensused_local_phase(
                context,
                phase=(
                    f"step-{schedule_step.step}-microbatch-{microbatch_number}-metrics"
                ),
                operation=prepare_microbatch_metrics,
            )
            local_answer_loss_sum += float(answer_sum.detach().float().item())
            local_route_loss_sum += float(route_sum.detach().float().item())
            local_online_state_digests.extend(state_digests)
            for index, value in enumerate(microbatch_metrics):
                local_metric_values[index] += value

            _run_consensused_local_phase(
                context,
                phase=(
                    f"step-{schedule_step.step}-microbatch-{microbatch_number}-"
                    "online-state-reset"
                ),
                operation=lambda: reset_delta_mem_states(model),
            )
            observed_phase_order.append(
                f"microbatch_{microbatch_number}_online_state_reset"
            )
            microbatch_batches[microbatch_offset] = None
            answer_sum = None
            batch = None
            logits = None
            local_router_gradient = None
            route_logits = None
            route_predictions = None
            route_sum = None
            total_loss = None
            write_audit = None

        local_objective_statistics = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-objective-preparation",
            operation=lambda: distributed.prepare_objective_statistics(
                answer_loss_sum=torch.tensor(
                    local_answer_loss_sum,
                    dtype=torch.float64,
                    device=context.device,
                ),
                answer_token_count=local_answer_tokens,
                route_loss_sum=torch.tensor(
                    local_route_loss_sum,
                    dtype=torch.float64,
                    device=context.device,
                ),
                route_row_count=local_route_rows,
            ),
        )
        observed_phase_order.append("objective_preparation")

        objective = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-objective-global-sum",
            operation=lambda: distributed.reduce_objective_statistics(
                context, local_objective_statistics
            ),
        )
        observed_phase_order.append("objective_global_sum")
        if (
            objective["answer_token_count"] != global_answer_tokens
            or objective["route_row_count"] != global_route_rows
        ):
            raise distributed.DistributedTrainingError(
                "Accumulated objective counts differ from prereduced denominators"
            )

        def validate_gradients() -> Mapping[str, Any]:
            prepared_validation = distributed.validate_local_gradients(named_trainable)
            if prepared_validation["passed"] is not True:
                raise RuntimeError(f"Invalid local gradients: {prepared_validation!r}")
            return prepared_validation

        gradient_validation = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-gradient-validation",
            operation=validate_gradients,
        )
        observed_phase_order.append("local_gradient_validation")

        collective_evidence = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-gradient-sum",
            operation=lambda: distributed.sum_gradients(context, named_trainable),
        )
        collective_evidence_by_step.append(dict(collective_evidence))
        observed_phase_order.append("gradient_sum")

        def clip_gradients() -> torch.Tensor:
            prepared_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            if not bool(torch.isfinite(prepared_norm).item()):
                raise RuntimeError("Distributed gradient norm is non-finite")
            return prepared_norm

        grad_norm = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-gradient-clip",
            operation=clip_gradients,
        )
        observed_phase_order.append("global_gradient_clip")

        _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-adamw-step",
            operation=optimizer.step,
        )
        observed_phase_order.append("adamw_step")

        local_metric_statistics = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-metric-preparation",
            operation=lambda: _prepare_distributed_scalar_sums(
                context, local_metric_values
            ),
        )
        metrics = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-metric-global-sum",
            operation=lambda: _distributed_scalar_sums(
                context, local_metric_statistics
            ),
        )
        observed_phase_order.append("metric_global_sum")

        def prepare_step_record() -> dict[str, Any]:
            if len(metrics) != 8 or any(
                metrics[index] <= 0.0 for index in (1, 3, 5, 7)
            ):
                raise RuntimeError("Distributed metric denominators must be positive")
            global_answer_loss = float(objective["answer_loss_sum"]) / int(
                objective["answer_token_count"]
            )
            global_route_loss = float(objective["route_loss_sum"]) / int(
                objective["route_row_count"]
            )
            global_total_loss = (
                answer_weight * global_answer_loss
                + route_weight * global_route_loss
            )
            return {
                "schema": TRAIN_STEP_SCHEMA,
                "step": schedule_step.step,
                "epoch": schedule_step.epoch,
                "rows": global_batch_size,
                "local_rows_per_rank": local_batch_size,
                "local_microbatch_size": local_microbatch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "world_size": context.world_size,
                "global_batch_size": global_batch_size,
                "global_answer_tokens": int(objective["answer_token_count"]),
                "answer_loss": global_answer_loss,
                "route_loss": global_route_loss,
                "total_loss": global_total_loss,
                "gradient_norm_before_clip": float(
                    grad_norm.detach().float().item()
                ),
                "gradient_reduction": "sum_before_global_clip",
                "teacher_forced_answer_exact_accuracy": metrics[0] / metrics[1],
                "semantic_route_accuracy": metrics[2] / metrics[3],
                "full_occupancy_fraction": metrics[4] / metrics[5],
                "forced_write_route_accuracy": metrics[6] / metrics[7],
                "global_row_ids": list(schedule_step.global_row_ids),
                "schedule_step_sha256": schedule_step.step_sha256,
                "training_conditions": list(selected_training_conditions),
            }

        step_record = _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-record-preparation",
            operation=prepare_step_record,
        )

        primary_step_evidence: Mapping[str, Any] | None = None
        if capture_step_evidence:
            def prepare_local_evidence() -> dict[str, Any]:
                return {
                    "rank": context.process_rank,
                    "local_row_ids": [row_ids[index] for index in local_indices],
                    "local_microbatch_row_ids": [
                        [
                            row_ids[index]
                            for index in local_indices[
                                start : start + local_microbatch_size
                            ]
                        ]
                        for start in range(
                            0, local_batch_size, local_microbatch_size
                        )
                    ],
                    "local_online_state_sha256": list(local_online_state_digests),
                    "local_online_state_sha256_by_microbatch": [
                        local_online_state_digests[
                            start : start + local_microbatch_size
                        ]
                        for start in range(
                            0, local_batch_size, local_microbatch_size
                        )
                    ],
                    "online_state_reset_count": gradient_accumulation_steps,
                    "local_answer_tokens": local_answer_tokens,
                    "local_route_rows": local_route_rows,
                    "trainable_metadata_sha256": trainable_metadata_sha256,
                    "trainable_names_sha256": trainable_names_sha256,
                    "gradient_validation": dict(gradient_validation),
                    "gradient_collective": dict(collective_evidence),
                    "adapter_state_sha256": _state_dict_sha256(
                        snapshot_delta_mem_weights(model)
                    ),
                    "optimizer_state_sha256": distributed.tensor_mapping_sha256(
                        optimizer.state_dict()
                    ),
                    "cuda_memory": distributed.cuda_memory_snapshot(context),
                }

            local_step_evidence = _run_consensused_local_phase(
                context,
                phase=f"step-{schedule_step.step}-evidence-preparation",
                operation=prepare_local_evidence,
            )
            gathered_step_evidence = distributed.gather_objects(
                context, local_step_evidence
            )

            def validate_step_evidence() -> Mapping[str, Any]:
                if len(gathered_step_evidence) != context.world_size or any(
                    not isinstance(value, Mapping)
                    for value in gathered_step_evidence
                ):
                    raise distributed.DistributedTrainingError(
                        f"Malformed rank evidence after step {schedule_step.step}"
                    )
                ordered_evidence = sorted(
                    gathered_step_evidence, key=lambda value: value.get("rank", -1)
                )
                if [value.get("rank") for value in ordered_evidence] != list(
                    range(context.world_size)
                ):
                    raise distributed.DistributedTrainingError(
                        f"Rank evidence identity failed after step {schedule_step.step}"
                    )
                adapter_hashes = {
                    value.get("adapter_state_sha256")
                    for value in ordered_evidence
                }
                optimizer_hashes = {
                    value.get("optimizer_state_sha256")
                    for value in ordered_evidence
                }
                metadata_hashes = {
                    value.get("trainable_metadata_sha256")
                    for value in ordered_evidence
                }
                if (
                    len(adapter_hashes) != 1
                    or len(optimizer_hashes) != 1
                    or metadata_hashes != {trainable_metadata_sha256}
                ):
                    raise distributed.DistributedTrainingError(
                        f"Replica consensus failed after step {schedule_step.step}"
                    )
                return {
                    "step": schedule_step.step,
                    "global_row_ids": list(schedule_step.global_row_ids),
                    "global_answer_tokens": int(objective["answer_token_count"]),
                    "global_route_rows": int(objective["route_row_count"]),
                    "phase_order": list(observed_phase_order),
                    "ranks": list(ordered_evidence),
                    "adapter_state_sha256": next(iter(adapter_hashes)),
                    "optimizer_state_sha256": next(iter(optimizer_hashes)),
                    "trainable_metadata_sha256": trainable_metadata_sha256,
                    "trainable_names_sha256": trainable_names_sha256,
                }

            primary_step_evidence = _run_consensused_local_phase(
                context,
                phase=f"step-{schedule_step.step}-evidence-validation",
                operation=validate_step_evidence,
            )

        def commit_progress() -> None:
            if not context.is_primary:
                return
            _append_jsonl(progress_path, step_record)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "answer_loss": round(step_record["answer_loss"], 6),
                        "route_loss": round(step_record["route_loss"], 6),
                        "answer_exact": round(
                            step_record["teacher_forced_answer_exact_accuracy"], 4
                        ),
                        "route_accuracy": round(
                            step_record["semantic_route_accuracy"], 4
                        ),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )

        _run_consensused_local_phase(
            context,
            phase=f"step-{schedule_step.step}-rank-zero-progress-commit",
            operation=commit_progress,
        )
        if context.is_primary and primary_step_evidence is not None:
            step_evidence.append(primary_step_evidence)

        totals["answer_loss"] += step_record["answer_loss"]
        totals["route_loss"] += step_record["route_loss"]
        totals["total_loss"] += step_record["total_loss"]
        totals["answer_exact_correct"] += metrics[0]
        totals["answer_rows"] += metrics[1]
        totals["route_correct"] += metrics[2]
        totals["route_total"] += metrics[3]
        totals["full_occupancy_count"] += metrics[4]
        totals["full_occupancy_total"] += metrics[5]
        totals["forced_write_route_correct"] += metrics[6]
        totals["forced_write_route_total"] += metrics[7]

    local_elapsed = _run_consensused_local_phase(
        context,
        phase="elapsed-time-preparation",
        operation=lambda: float(time.time() - started),
    )
    elapsed_by_rank = distributed.gather_objects(context, local_elapsed)
    elapsed_values = _run_consensused_local_phase(
        context,
        phase="elapsed-time-validation",
        operation=lambda: tuple(float(value) for value in elapsed_by_rank),
    )

    final_adapter_hash, final_optimizer_hash = _run_consensused_local_phase(
        context,
        phase="final-replica-hash-preparation",
        operation=lambda: (
            _state_dict_sha256(snapshot_delta_mem_weights(model)),
            distributed.tensor_mapping_sha256(optimizer.state_dict()),
        ),
    )
    adapter_hashes = distributed.require_consensus(
        context, final_adapter_hash, description="final adapter state"
    )
    optimizer_hashes = distributed.require_consensus(
        context, final_optimizer_hash, description="final optimizer state"
    )
    memory_after = _run_consensused_local_phase(
        context,
        phase="final-cuda-memory-preparation",
        operation=lambda: distributed.cuda_memory_snapshot(context),
    )
    rank_memory = distributed.gather_objects(
        context,
        {"before_training": memory_before, "after_training": memory_after},
    )
    validated_rank_memory = _run_consensused_local_phase(
        context,
        phase="final-cuda-memory-validation",
        operation=lambda: tuple(
            dict(value)
            for value in rank_memory
            if isinstance(value, Mapping)
        ),
    )
    if len(validated_rank_memory) != context.world_size:
        raise distributed.DistributedTrainingError(
            "Final CUDA memory evidence is incomplete"
        )

    def prepare_progress_hash() -> str | None:
        return source.sha256_file(progress_path) if context.is_primary else None

    progress_sha256 = _run_consensused_local_phase(
        context,
        phase="rank-zero-progress-hash",
        operation=prepare_progress_hash,
    )

    def prepare_result() -> dict[str, Any]:
        if router_gradient is None or collective_evidence is None:
            raise RuntimeError("Distributed training executed no optimization steps")
        if len(collective_evidence_by_step) != len(schedule):
            raise RuntimeError("Distributed collective evidence is incomplete")
        denominators = (
            totals["answer_rows"],
            totals["route_total"],
            totals["full_occupancy_total"],
            totals["forced_write_route_total"],
        )
        if any(value <= 0.0 for value in denominators):
            raise RuntimeError("Distributed aggregate metric denominator is zero")
        return {
            "steps": len(schedule),
            "epochs_requested": epochs,
            "max_steps": max_steps,
            "elapsed_seconds": max(elapsed_values),
            "mean_answer_loss": totals["answer_loss"] / len(schedule),
            "mean_route_loss": totals["route_loss"] / len(schedule),
            "mean_total_loss": totals["total_loss"] / len(schedule),
            "teacher_forced_answer_exact_accuracy": (
                totals["answer_exact_correct"] / totals["answer_rows"]
            ),
            "semantic_route_accuracy": (
                totals["route_correct"] / totals["route_total"]
            ),
            "full_occupancy_fraction": (
                totals["full_occupancy_count"] / totals["full_occupancy_total"]
            ),
            "forced_write_route_accuracy": (
                totals["forced_write_route_correct"]
                / totals["forced_write_route_total"]
            ),
            "router_gradient_audit": router_gradient,
            "progress_schema": TRAIN_STEP_SCHEMA,
            "progress_sha256": progress_sha256,
            "distributed": {
                "backend": context.backend,
                "control_backend": context.control_backend,
                "world_size": context.world_size,
                "local_batch_size": local_batch_size,
                "local_microbatch_size": local_microbatch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "global_batch_size": global_batch_size,
                "gradient_synchronization": "sum",
                "unused_gradient_policy": (
                    "global_active_union_zero_fill_rank_missing_skip_global_inactive"
                ),
                "gradient_clip_order": "after_sum_before_adamw",
                "answer_loss_normalization": (
                    "global_supervised_answer_token_count"
                ),
                "route_loss_normalization": "global_row_count_after_layer_mean",
                "online_memory_state": (
                    "rank_local_reset_after_each_microbatch_never_reduced"
                ),
                "schedule_sha256": schedule_sha256,
                "ordered_row_ids_sha256": distributed.canonical_sha256(row_ids),
                "trainable_metadata": list(trainable_metadata),
                "trainable_metadata_sha256": trainable_metadata_sha256,
                "trainable_names_sha256": trainable_names_sha256,
                "collective_evidence": dict(collective_evidence),
                "collective_evidence_by_step": [
                    dict(value) for value in collective_evidence_by_step
                ],
                "rank_devices": [dict(value) for value in context.rank_devices],
                "rank_memory": list(validated_rank_memory),
                "final_adapter_state_sha256": adapter_hashes[0],
                "final_optimizer_state_sha256": optimizer_hashes[0],
                "step_evidence": step_evidence,
            },
        }

    return _run_consensused_local_phase(
        context,
        phase="distributed-training-result-preparation",
        operation=prepare_result,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _local_gradient_validation_passed(
    value: Any,
    *,
    trainable_names_sha256: Any,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    parameter_tensors = value.get("parameter_tensors")
    active_tensors = value.get("active_gradient_tensors")
    missing_tensors = value.get("missing_gradient_tensors")
    return (
        value.get("passed") is True
        and type(parameter_tensors) is int
        and parameter_tensors > 0
        and value.get("parameter_names_sha256") == trainable_names_sha256
        and type(active_tensors) is int
        and active_tensors >= 0
        and type(missing_tensors) is int
        and missing_tensors >= 0
        and active_tensors + missing_tensors == parameter_tensors
        and _is_sha256(value.get("active_names_sha256"))
        and _is_sha256(value.get("missing_names_sha256"))
        and value.get("nonfinite_gradient_tensors") == 0
        and value.get("nonfinite_names_sha256")
        == distributed.canonical_sha256([])
        and value.get("nonfinite_preview") == []
        and value.get("non_fp32_gradient_tensors") == 0
        and value.get("non_fp32_names_sha256")
        == distributed.canonical_sha256([])
        and value.get("non_fp32_preview") == []
    )


def _active_union_collective_passed(
    value: Any,
    *,
    trainable_names: Sequence[str] | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    trainable_tensors = value.get("trainable_parameter_tensors")
    active_tensors = value.get("gradient_tensors")
    global_active = value.get("global_active_parameter_indices")
    global_inactive = value.get("global_inactive_parameter_indices")
    per_rank = value.get("per_rank_active_gradients")
    materialized = value.get("materialized_zero_gradient_tensors_by_rank")
    if (
        type(trainable_tensors) is not int
        or trainable_tensors <= 0
        or not _is_sha256(value.get("trainable_names_sha256"))
        or type(active_tensors) is not int
        or active_tensors <= 0
        or not isinstance(global_active, list)
        or not isinstance(global_inactive, list)
        or any(type(index) is not int for index in global_active)
        or any(type(index) is not int for index in global_inactive)
        or len(global_active) != active_tensors
        or len(global_active) + len(global_inactive) != trainable_tensors
        or len(set(global_active)) != len(global_active)
        or len(set(global_inactive)) != len(global_inactive)
        or set(global_active) & set(global_inactive)
        or global_active != sorted(global_active)
        or global_inactive != sorted(global_inactive)
        or set(global_active) | set(global_inactive) != set(range(trainable_tensors))
        or not _is_sha256(value.get("global_active_names_sha256"))
        or not _is_sha256(value.get("global_inactive_names_sha256"))
        or not isinstance(per_rank, list)
        or len(per_rank) != distributed.REQUIRED_WORLD_SIZE
        or any(not isinstance(record, Mapping) for record in per_rank)
        or not isinstance(materialized, list)
        or len(materialized) != distributed.REQUIRED_WORLD_SIZE
        or type(value.get("collective_buckets")) is not int
        or value["collective_buckets"] <= 0
        or type(value.get("all_reduce_bytes")) is not int
        or value["all_reduce_bytes"] <= 0
        or not _is_sha256(value.get("bucket_plan_sha256"))
    ):
        return False
    if trainable_names is not None:
        ordered_trainable = list(trainable_names)
        if len(ordered_trainable) != trainable_tensors:
            return False
        resolved_active = [ordered_trainable[index] for index in global_active]
        resolved_inactive = [ordered_trainable[index] for index in global_inactive]
        if (
            len(set(ordered_trainable)) != len(ordered_trainable)
            or value.get("trainable_names_sha256")
            != distributed.canonical_sha256(ordered_trainable)
            or value.get("global_active_names_sha256")
            != distributed.canonical_sha256(resolved_active)
            or value.get("global_inactive_names_sha256")
            != distributed.canonical_sha256(resolved_inactive)
        ):
            return False
    for rank, record in enumerate(per_rank):
        local_active_tensors = record.get("active_gradient_tensors")
        if (
            record.get("rank") != rank
            or type(local_active_tensors) is not int
            or not 0 <= local_active_tensors <= active_tensors
            or not _is_sha256(record.get("active_names_sha256"))
            or type(materialized[rank]) is not int
            or materialized[rank] != active_tensors - local_active_tensors
        ):
            return False
    return True


def _distributed_preflight_step_passed_unchecked(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    ranks = value.get("ranks")
    global_row_ids = value.get("global_row_ids")
    if not isinstance(ranks, list) or not isinstance(global_row_ids, list):
        return False
    if any(not isinstance(rank, Mapping) for rank in ranks):
        return False
    if len(ranks) != distributed.REQUIRED_WORLD_SIZE or len(global_row_ids) != (
        distributed.REQUIRED_GLOBAL_BATCH_SIZE
    ):
        return False
    ordered_ranks = sorted(
        ranks,
        key=lambda rank: rank.get("rank", -1) if isinstance(rank, Mapping) else -1,
    )
    if [rank.get("rank") for rank in ordered_ranks] != list(
        range(distributed.REQUIRED_WORLD_SIZE)
    ):
        return False
    for rank in ordered_ranks:
        local_rows = rank.get("local_row_ids")
        local_microbatch_rows = rank.get("local_microbatch_row_ids")
        local_digests = rank.get("local_online_state_sha256")
        local_microbatch_digests = rank.get(
            "local_online_state_sha256_by_microbatch"
        )
        gradient_validation = rank.get("gradient_validation")
        gradient_collective = rank.get("gradient_collective")
        if (
            not isinstance(local_rows, list)
            or len(local_rows) != distributed.REQUIRED_LOCAL_BATCH_SIZE
            or not all(isinstance(row_id, str) and row_id for row_id in local_rows)
            or not isinstance(local_microbatch_rows, list)
            or len(local_microbatch_rows)
            != distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS
            or any(
                not isinstance(microbatch, list)
                or len(microbatch) != distributed.REQUIRED_LOCAL_MICROBATCH_SIZE
                for microbatch in local_microbatch_rows
            )
            or [
                row_id
                for microbatch in local_microbatch_rows
                for row_id in microbatch
            ]
            != local_rows
            or not isinstance(local_digests, list)
            or len(local_digests) != distributed.REQUIRED_LOCAL_BATCH_SIZE
            or not all(_is_sha256(digest) for digest in local_digests)
            or not isinstance(local_microbatch_digests, list)
            or len(local_microbatch_digests)
            != distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS
            or any(
                not isinstance(microbatch, list)
                or len(microbatch) != distributed.REQUIRED_LOCAL_MICROBATCH_SIZE
                for microbatch in local_microbatch_digests
            )
            or [
                digest
                for microbatch in local_microbatch_digests
                for digest in microbatch
            ]
            != local_digests
            or rank.get("online_state_reset_count")
            != distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS
            or type(rank.get("local_answer_tokens")) is not int
            or rank["local_answer_tokens"] <= 0
            or rank.get("local_route_rows") != distributed.REQUIRED_LOCAL_BATCH_SIZE
            or not _is_sha256(rank.get("adapter_state_sha256"))
            or not _is_sha256(rank.get("optimizer_state_sha256"))
            or not _is_sha256(rank.get("trainable_metadata_sha256"))
            or not _is_sha256(rank.get("trainable_names_sha256"))
            or not _local_gradient_validation_passed(
                gradient_validation,
                trainable_names_sha256=rank.get("trainable_names_sha256"),
            )
            or not _active_union_collective_passed(gradient_collective)
            or gradient_collective.get("trainable_parameter_tensors")
            != gradient_validation.get("parameter_tensors")
            or gradient_collective.get("trainable_names_sha256")
            != rank.get("trainable_names_sha256")
        ):
            return False
        collective_rank = gradient_collective["per_rank_active_gradients"][
            rank["rank"]
        ]
        if (
            collective_rank.get("active_gradient_tensors")
            != gradient_validation.get("active_gradient_tensors")
            or collective_rank.get("active_names_sha256")
            != gradient_validation.get("active_names_sha256")
        ):
            return False
    local_rows = [
        row_id
        for rank in ordered_ranks
        for row_id in rank["local_row_ids"]
    ]
    local_state_digests = [
        digest
        for rank in ordered_ranks
        for digest in rank["local_online_state_sha256"]
    ]
    adapter_hashes = {
        rank.get("adapter_state_sha256") for rank in ordered_ranks
    }
    optimizer_hashes = {
        rank.get("optimizer_state_sha256") for rank in ordered_ranks
    }
    metadata_hashes = {
        rank.get("trainable_metadata_sha256") for rank in ordered_ranks
    }
    names_hashes = {rank.get("trainable_names_sha256") for rank in ordered_ranks}
    collective_hashes = {
        distributed.canonical_sha256(rank["gradient_collective"])
        for rank in ordered_ranks
    }
    return (
        value.get("phase_order") == list(DISTRIBUTED_STEP_PHASE_ORDER)
        and type(value.get("global_answer_tokens")) is int
        and value["global_answer_tokens"] > 0
        and value.get("global_answer_tokens")
        == sum(rank.get("local_answer_tokens", 0) for rank in ordered_ranks)
        and value.get("global_route_rows")
        == sum(rank.get("local_route_rows", 0) for rank in ordered_ranks)
        and value.get("global_route_rows")
        == distributed.REQUIRED_GLOBAL_BATCH_SIZE
        and local_rows == global_row_ids
        and len(set(local_rows)) == distributed.REQUIRED_GLOBAL_BATCH_SIZE
        and len(local_state_digests) == distributed.REQUIRED_GLOBAL_BATCH_SIZE
        and all(_is_sha256(digest) for digest in local_state_digests)
        and len(set(local_state_digests)) > 1
        and len(adapter_hashes) == 1
        and _is_sha256(next(iter(adapter_hashes), None))
        and len(optimizer_hashes) == 1
        and _is_sha256(next(iter(optimizer_hashes), None))
        and value.get("adapter_state_sha256") == next(iter(adapter_hashes), None)
        and value.get("optimizer_state_sha256")
        == next(iter(optimizer_hashes), None)
        and len(metadata_hashes) == 1
        and value.get("trainable_metadata_sha256")
        == next(iter(metadata_hashes), None)
        and len(names_hashes) == 1
        and value.get("trainable_names_sha256")
        == next(iter(names_hashes), None)
        and len(collective_hashes) == 1
    )


def _distributed_preflight_step_passed(value: Any) -> bool:
    try:
        return _distributed_preflight_step_passed_unchecked(value)
    except Exception:
        return False


def _build_distributed_preflight_gate(
    training: Mapping[str, Any],
) -> dict[str, Any]:
    distributed_evidence = training.get("distributed")
    if not isinstance(distributed_evidence, Mapping):
        distributed_evidence = {}
    rank_devices = distributed_evidence.get("rank_devices")
    if not isinstance(rank_devices, list):
        rank_devices = []
    rank_memory = distributed_evidence.get("rank_memory")
    if not isinstance(rank_memory, list):
        rank_memory = []
    step_evidence = distributed_evidence.get("step_evidence")
    if not isinstance(step_evidence, list):
        step_evidence = []
    initialization = distributed_evidence.get("initialization")
    if not isinstance(initialization, Mapping):
        initialization = {}
    rank_immutability = distributed_evidence.get("rank_input_immutability")
    if not isinstance(rank_immutability, list):
        rank_immutability = []
    collective_evidence = distributed_evidence.get("collective_evidence")
    if not isinstance(collective_evidence, Mapping):
        collective_evidence = {}
    collective_evidence_by_step = distributed_evidence.get(
        "collective_evidence_by_step"
    )
    if not isinstance(collective_evidence_by_step, list):
        collective_evidence_by_step = []
    trainable_metadata = distributed_evidence.get("trainable_metadata")
    if not isinstance(trainable_metadata, list):
        trainable_metadata = []
    router_gradient_audit = training.get("router_gradient_audit")
    if not isinstance(router_gradient_audit, Mapping):
        router_gradient_audit = {}
    training_dataset_audit = training.get("training_dataset_audit")

    headroom_by_rank: list[dict[str, int]] = []
    memory_evidence_passed = len(rank_memory) == distributed.REQUIRED_WORLD_SIZE
    for rank, record in enumerate(rank_memory):
        after = record.get("after_training") if isinstance(record, Mapping) else None
        if not isinstance(after, Mapping):
            memory_evidence_passed = False
            continue
        required = (
            "process_rank",
            "total_bytes",
            "free_bytes",
            "peak_reserved_bytes",
        )
        if any(type(after.get(field)) is not int for field in required):
            memory_evidence_passed = False
            continue
        isolated_headroom = after["total_bytes"] - after["peak_reserved_bytes"]
        conservative_headroom = min(isolated_headroom, after["free_bytes"])
        headroom_by_rank.append(
            {
                "rank": rank,
                "process_rank": after["process_rank"],
                "peak_reserved_bytes": after["peak_reserved_bytes"],
                "isolated_headroom_bytes": isolated_headroom,
                "observed_free_bytes": after["free_bytes"],
                "conservative_headroom_bytes": conservative_headroom,
            }
        )
        if (
            after["process_rank"] != rank
            or isolated_headroom < 0
            or conservative_headroom < MINIMUM_DISTRIBUTED_HEADROOM_BYTES
        ):
            memory_evidence_passed = False

    device_ranks = [
        value.get("process_rank") for value in rank_devices
        if isinstance(value, Mapping)
    ]
    device_uuids = [
        value.get("device_uuid") for value in rank_devices
        if isinstance(value, Mapping)
    ]
    device_pids = [
        value.get("pid") for value in rank_devices if isinstance(value, Mapping)
    ]
    after_broadcast = initialization.get("hashes_after_broadcast")
    if not isinstance(after_broadcast, list):
        after_broadcast = []
    final_adapter_hash = distributed_evidence.get("final_adapter_state_sha256")
    final_optimizer_hash = distributed_evidence.get("final_optimizer_state_sha256")
    source_hashes = {
        value.get("source_snapshot_sha256")
        for value in rank_immutability
        if isinstance(value, Mapping)
    }
    model_hashes = {
        value.get("model_snapshot_sha256")
        for value in rank_immutability
        if isinstance(value, Mapping)
    }
    step_numbers = [
        step.get("step") if isinstance(step, Mapping) else None
        for step in step_evidence
    ]
    trainable_metadata_valid = bool(trainable_metadata) and all(
        isinstance(value, Mapping)
        and isinstance(value.get("name"), str)
        and value.get("name")
        and value.get("requires_grad") is True
        for value in trainable_metadata
    )
    computed_trainable_metadata_sha256 = (
        distributed.canonical_sha256(trainable_metadata)
        if trainable_metadata_valid
        else None
    )
    computed_trainable_names = (
        [value["name"] for value in trainable_metadata]
        if trainable_metadata_valid
        else []
    )
    computed_trainable_names_sha256 = (
        distributed.canonical_sha256(computed_trainable_names)
        if trainable_metadata_valid
        else None
    )
    collective_parameter_binding = (
        len(collective_evidence_by_step) == DISTRIBUTED_PREFLIGHT_STEPS
        and len(step_evidence) == DISTRIBUTED_PREFLIGHT_STEPS
        and isinstance(collective_evidence, Mapping)
        and dict(collective_evidence)
        == dict(collective_evidence_by_step[-1])
    )
    if collective_parameter_binding:
        for step, step_collective in zip(
            step_evidence, collective_evidence_by_step, strict=True
        ):
            if not isinstance(step, Mapping) or not isinstance(
                step_collective, Mapping
            ):
                collective_parameter_binding = False
                break
            ranks = step.get("ranks")
            if not isinstance(ranks, list) or any(
                not isinstance(rank, Mapping) for rank in ranks
            ):
                collective_parameter_binding = False
                break
            if (
                not _active_union_collective_passed(
                    step_collective,
                    trainable_names=computed_trainable_names,
                )
                or step_collective.get("trainable_names_sha256")
                != computed_trainable_names_sha256
                or step.get("trainable_metadata_sha256")
                != computed_trainable_metadata_sha256
                or step.get("trainable_names_sha256")
                != computed_trainable_names_sha256
                or any(
                    dict(rank.get("gradient_collective", {}))
                    != dict(step_collective)
                    for rank in ranks
                )
            ):
                collective_parameter_binding = False
                break
    checks = {
        "three_production_updates": (
            training.get("steps") == DISTRIBUTED_PREFLIGHT_STEPS
            and training.get("max_steps") == DISTRIBUTED_PREFLIGHT_STEPS
            and len(step_evidence) == DISTRIBUTED_PREFLIGHT_STEPS
            and step_numbers
            == list(range(1, DISTRIBUTED_PREFLIGHT_STEPS + 1))
        ),
        "four_rank_topology": (
            distributed_evidence.get("backend") == "nccl"
            and distributed_evidence.get("control_backend") == "gloo"
            and distributed_evidence.get("world_size")
            == distributed.REQUIRED_WORLD_SIZE
            and distributed_evidence.get("local_batch_size")
            == distributed.REQUIRED_LOCAL_BATCH_SIZE
            and distributed_evidence.get("local_microbatch_size")
            == distributed.REQUIRED_LOCAL_MICROBATCH_SIZE
            and distributed_evidence.get("gradient_accumulation_steps")
            == distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS
            and distributed_evidence.get("global_batch_size")
            == distributed.REQUIRED_GLOBAL_BATCH_SIZE
        ),
        "four_distinct_cuda_workers": (
            len(rank_devices) == distributed.REQUIRED_WORLD_SIZE
            and device_ranks == list(range(distributed.REQUIRED_WORLD_SIZE))
            and len(set(device_uuids)) == distributed.REQUIRED_WORLD_SIZE
            and len(set(device_pids)) == distributed.REQUIRED_WORLD_SIZE
        ),
        "complete_adapter_broadcast": (
            len(after_broadcast) == distributed.REQUIRED_WORLD_SIZE
            and len(set(after_broadcast)) == 1
            and _is_sha256(next(iter(after_broadcast), None))
            and initialization.get("synchronized_adapter_state_sha256")
            == next(iter(after_broadcast), None)
            and isinstance(initialization.get("broadcast"), Mapping)
            and type(
                initialization["broadcast"].get("parameter_tensors")
            ) is int
            and initialization["broadcast"]["parameter_tensors"] > 0
            and _is_sha256(
                initialization.get("complete_adapter_metadata_sha256")
            )
            and _is_sha256(initialization.get("complete_adapter_names_sha256"))
            and initialization["broadcast"].get("parameter_names_sha256")
            == initialization.get("complete_adapter_names_sha256")
            and _is_sha256(
                initialization["broadcast"].get("bucket_plan_sha256")
            )
            and type(initialization["broadcast"].get("broadcast_bytes")) is int
            and initialization["broadcast"]["broadcast_bytes"] > 0
        ),
        "global_objective_and_row_ownership": (
            len(step_evidence) == DISTRIBUTED_PREFLIGHT_STEPS
            and all(_distributed_preflight_step_passed(step) for step in step_evidence)
        ),
        "collective_contract": (
            distributed_evidence.get("gradient_synchronization") == "sum"
            and distributed_evidence.get("unused_gradient_policy")
            == "global_active_union_zero_fill_rank_missing_skip_global_inactive"
            and distributed_evidence.get("gradient_clip_order")
            == "after_sum_before_adamw"
            and distributed_evidence.get("answer_loss_normalization")
            == "global_supervised_answer_token_count"
            and distributed_evidence.get("route_loss_normalization")
            == "global_row_count_after_layer_mean"
            and distributed_evidence.get("online_memory_state")
            == "rank_local_reset_after_each_microbatch_never_reduced"
        ),
        "collective_parameter_binding": (
            trainable_metadata_valid
            and distributed_evidence.get("trainable_metadata_sha256")
            == computed_trainable_metadata_sha256
            and distributed_evidence.get("trainable_names_sha256")
            == computed_trainable_names_sha256
            and collective_parameter_binding
        ),
        "final_replica_consensus": (
            _is_sha256(final_adapter_hash)
            and _is_sha256(final_optimizer_hash)
            and bool(step_evidence)
            and step_evidence[-1].get("adapter_state_sha256")
            == final_adapter_hash
            and step_evidence[-1].get("optimizer_state_sha256")
            == final_optimizer_hash
        ),
        "adapter_updated": training.get("adapter_changed") is True,
        "router_gradients": router_gradient_audit.get(
            "all_ranks_all_modules_finite_nonzero"
        )
        is True,
        "rank_input_immutability": (
            len(rank_immutability) == distributed.REQUIRED_WORLD_SIZE
            and len(source_hashes) == 1
            and len(model_hashes) == 1
            and _is_sha256(next(iter(source_hashes), None))
            and _is_sha256(next(iter(model_hashes), None))
        ),
        "communication_inclusive_memory_headroom": memory_evidence_passed,
        "rank_zero_progress_evidence": _is_sha256(training.get("progress_sha256")),
        "compositional_training_dataset": (
            isinstance(training_dataset_audit, Mapping)
            and validate_production_training_contract(
                training_dataset_audit,
                schedule_mode="preflight",
            )
        ),
    }
    return {
        "schema": DISTRIBUTED_PREFLIGHT_GATE_SCHEMA,
        "minimum_headroom_bytes": MINIMUM_DISTRIBUTED_HEADROOM_BYTES,
        "headroom_by_rank": headroom_by_rank,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def build_distributed_preflight_gate(
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate preflight evidence without accepting or crashing on malformed data."""

    try:
        gate = _build_distributed_preflight_gate(training)
    except Exception as error:
        checks = {"evidence_well_formed": False}
        return {
            "schema": DISTRIBUTED_PREFLIGHT_GATE_SCHEMA,
            "minimum_headroom_bytes": MINIMUM_DISTRIBUTED_HEADROOM_BYTES,
            "headroom_by_rank": [],
            "checks": checks,
            "failed_checks": ["evidence_well_formed"],
            "evidence_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "passed": False,
        }
    checks = {"evidence_well_formed": True, **gate["checks"]}
    gate["checks"] = checks
    gate["failed_checks"] = sorted(
        name for name, passed in checks.items() if passed is not True
    )
    gate["passed"] = all(passed is True for passed in checks.values())
    return gate


def _preflight_code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "natural_runner": Path(__file__).resolve(strict=True),
        "distributed_primitives": Path(distributed.__file__).resolve(strict=True),
        "shared_training_runtime": Path(runtime.__file__).resolve(strict=True),
        "natural_source_builder": Path(source.__file__).resolve(strict=True),
        "delta_api": Path(delta_core.__file__).resolve(strict=True),
        "delta_implementation": Path(delta_impl.__file__).resolve(strict=True),
    }
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": source.sha256_file(path),
        }
        for name, path in paths.items()
    }


def load_pristine_base_model(
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
) -> torch.nn.Module:
    """Load frozen Gemma without ever attaching a Delta-Mem wrapper."""

    model = runtime.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    runtime._disable_training_cache(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if list(iter_delta_mem_modules(model)):
        raise RuntimeError("Pristine frozen base unexpectedly contains Delta-Mem modules")
    return model


@dataclass(frozen=True)
class NaturalRecord:
    record_id: str
    semantic_slot: int
    physical_slot: int
    key_text: str
    value_json: str
    write_text: str


@dataclass(frozen=True)
class NaturalQuery:
    query_id: str
    target_slot: int
    target_record_id: str
    address_text: str
    read_prompt: str
    gold_json: str
    expected_json_by_condition: Mapping[str, str]
    rewrite_records: tuple[NaturalRecord, ...]
    record_payload_sha256_by_condition: Mapping[str, str]
    binding_absent_from_training: Mapping[str, bool]
    shared_correct_runtime_state_group: str


@dataclass(frozen=True)
class NaturalEpisode:
    episode_id: str
    split: str
    task: str
    passage_components: tuple[str, ...]
    records_by_condition: Mapping[str, tuple[NaturalRecord, ...]]
    queries: tuple[NaturalQuery, ...]


@dataclass(frozen=True)
class NaturalMemoryExample:
    row_id: str
    memory_state_id: str
    source_split: str
    source_mapping_offset: int
    condition: str
    write_records: tuple[dict[str, Any], ...]
    write_slots: tuple[int, ...]
    read_input_ids: tuple[int, ...]
    read_attention_mask: tuple[int, ...]
    query_mask: tuple[bool, ...]
    answer_mask: tuple[bool, ...]
    labels: tuple[int, ...]
    target_slot: int | None
    expected_answer_token_ids: tuple[int, ...]
    expected_value: str
    target_slot_rewrite_selection: dict[str, Any] | None
    episode_id: str
    task: str
    semantic_target_slot: int
    write_record_ids: tuple[str, ...]
    write_semantic_slots: tuple[int, ...]
    write_value_jsons: tuple[str, ...]
    record_payload_sha256: str
    binding_absent_from_training: bool | None
    query_prefix_length: int


@dataclass(frozen=True)
class ProfileBundle:
    profile: str
    train_episodes: tuple[NaturalEpisode, ...]
    evaluation_episodes: tuple[NaturalEpisode, ...]
    evaluation_split: str
    development_manifest: Mapping[str, Any]
    sealed_manifest: Mapping[str, Any] | None
    source_paths: tuple[Path, ...]
    model_binding: Mapping[str, Any]
    eligibility: Mapping[str, Any]


@dataclass(frozen=True)
class GateThresholds:
    answer_exact_min: float = 0.80
    route_accuracy_min: float = 0.95
    rewrite_output_change_min: float = 0.80


def configure_hf_mirror(endpoint: str | None = None) -> str:
    requested = endpoint or os.environ.get("HF_ENDPOINT") or HF_MIRROR_ENDPOINT
    if requested.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {requested!r}"
        )
    current = os.environ.get("HF_ENDPOINT")
    if current is not None and current.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {current!r}"
        )
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    return HF_MIRROR_ENDPOINT


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _require_sequence(value: Any, description: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{description} must be an array")
    return value


def _canonical_value(value: Any) -> str:
    return source.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return source.sha256_text(source.canonical_json(value))


def _adapt_record(value: Any, *, location: str) -> NaturalRecord:
    record = _require_mapping(value, location)
    semantic_slot = record.get("slot_id")
    physical_slot = record.get("physical_index")
    if type(semantic_slot) is not int or semantic_slot not in range(RECORDS_PER_EPISODE):
        raise ValueError(f"{location} has an invalid semantic slot")
    if type(physical_slot) is not int or physical_slot not in range(RECORDS_PER_EPISODE):
        raise ValueError(f"{location} has an invalid physical slot")
    value_json = str(record.get("value_json", ""))
    if value_json != _canonical_value(record.get("value")):
        raise ValueError(f"{location} value_json is not canonical")
    result = NaturalRecord(
        record_id=str(record.get("record_id", "")),
        semantic_slot=semantic_slot,
        physical_slot=physical_slot,
        key_text=str(record.get("key_text", "")),
        value_json=value_json,
        write_text=str(record.get("write_text", "")),
    )
    if not all((result.record_id, result.key_text, result.value_json, result.write_text)):
        raise ValueError(f"{location} omits a required record field")
    if result.key_text not in result.write_text or result.value_json not in result.write_text:
        raise ValueError(f"{location} write text does not bind its key and value")
    return result


def _validate_record_set(
    records: Sequence[NaturalRecord],
    *,
    location: str,
    allow_empty: bool = False,
) -> tuple[NaturalRecord, ...]:
    result = tuple(records)
    if allow_empty and not result:
        return result
    if len(result) != RECORDS_PER_EPISODE:
        raise ValueError(f"{location} must contain exactly four records")
    if {record.semantic_slot for record in result} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError(f"{location} does not cover all semantic slots")
    if {record.physical_slot for record in result} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError(f"{location} does not cover all physical slots")
    if len({record.record_id for record in result}) != RECORDS_PER_EPISODE:
        raise ValueError(f"{location} has duplicate record IDs")
    return result


def _record_at_semantic_slot(
    records: Sequence[NaturalRecord], slot: int
) -> NaturalRecord:
    matches = [record for record in records if record.semantic_slot == slot]
    if len(matches) != 1:
        raise ValueError(f"State has {len(matches)} records for semantic slot {slot}")
    return matches[0]


def adapt_episode(raw_value: Mapping[str, Any]) -> NaturalEpisode:
    """Validate the source schema and isolate all generator-specific field access."""

    raw = _require_mapping(raw_value, "episode")
    if raw.get("schema") != source.SCHEMA:
        raise ValueError(
            f"Natural episode schema must be {source.SCHEMA}, not {raw.get('schema')!r}"
        )
    episode_id = str(raw.get("episode_id", ""))
    split = str(raw.get("split", ""))
    task = str(raw.get("task", ""))
    if not episode_id or split not in source.SPLITS or not task:
        raise ValueError("Natural episode identity is invalid")

    state_variants = _require_mapping(raw.get("state_variants"), "state_variants")
    required_states = {
        "correct_state",
        "donor_state",
        "value_swap",
        "shuffled_slots",
        "no_state",
    }
    if set(state_variants) != required_states:
        raise ValueError("Natural episode state variants differ from the v2 contract")

    records_by_condition: dict[str, tuple[NaturalRecord, ...]] = {}
    raw_records_by_condition: dict[str, Sequence[Any]] = {}
    payload_by_condition: dict[str, str] = {}
    for condition in required_states:
        variant = _require_mapping(state_variants[condition], f"state {condition}")
        raw_records = _require_sequence(
            variant.get("records"), f"state {condition} records"
        )
        raw_records_by_condition[condition] = raw_records
        expected_payload = str(variant.get("record_payload_sha256", ""))
        actual_payload = _sha256_json(raw_records)
        if expected_payload != actual_payload:
            raise ValueError(f"State {condition} record payload hash differs")
        payload_by_condition[condition] = actual_payload
        adapted = tuple(
            _adapt_record(record, location=f"{episode_id}:{condition}:{index}")
            for index, record in enumerate(raw_records)
        )
        records_by_condition[condition] = _validate_record_set(
            adapted,
            location=f"{episode_id}:{condition}",
            allow_empty=condition == "no_state",
        )

    correct_records = records_by_condition["correct_state"]
    raw_canonical_records = _require_sequence(raw.get("records"), "episode records")
    if source.canonical_json(raw_canonical_records) != source.canonical_json(
        raw_records_by_condition["correct_state"]
    ):
        raise ValueError("Episode records differ from correct_state")
    correct_identity = {
        record.semantic_slot: (record.record_id, record.key_text)
        for record in correct_records
    }
    for condition in ("donor_state", "value_swap", "shuffled_slots"):
        identity = {
            record.semantic_slot: (record.record_id, record.key_text)
            for record in records_by_condition[condition]
        }
        if identity != correct_identity:
            raise ValueError(f"State {condition} changed a semantic key")
    for condition in ("donor_state", "value_swap"):
        unchanged = [
            slot
            for slot in range(RECORDS_PER_EPISODE)
            if _record_at_semantic_slot(
                records_by_condition[condition], slot
            ).value_json
            == _record_at_semantic_slot(correct_records, slot).value_json
        ]
        if unchanged:
            raise ValueError(f"State {condition} left values unchanged at {unchanged}")
    shuffled = records_by_condition["shuffled_slots"]
    if {
        (record.semantic_slot, record.value_json) for record in shuffled
    } != {
        (record.semantic_slot, record.value_json) for record in correct_records
    }:
        raise ValueError("shuffled_slots changed a semantic value")
    donor_components = tuple(
        str(value)
        for value in _require_sequence(
            raw.get("donor_source_component_ids"),
            "donor_source_component_ids",
        )
    )
    if (
        len(donor_components) != RECORDS_PER_EPISODE
        or len(set(donor_components)) != RECORDS_PER_EPISODE
    ):
        raise ValueError(
            "Natural donor state is not backed by four distinct external components"
        )

    query_deltas = _require_mapping(
        raw.get("query_counterfactual_records"),
        "query_counterfactual_records",
    )
    raw_queries = _require_sequence(raw.get("queries"), "queries")
    if len(raw_queries) != RECORDS_PER_EPISODE:
        raise ValueError("Natural episode must contain exactly four queries")
    queries: list[NaturalQuery] = []
    for query_index, raw_query_value in enumerate(raw_queries):
        raw_query = _require_mapping(raw_query_value, f"query {query_index}")
        target_slot = raw_query.get("target_slot")
        if type(target_slot) is not int or target_slot not in range(RECORDS_PER_EPISODE):
            raise ValueError(f"Query {query_index} has an invalid target slot")
        correct_target = _record_at_semantic_slot(correct_records, target_slot)
        if raw_query.get("target_record_id") != correct_target.record_id:
            raise ValueError(f"Query {query_index} target record differs")
        if raw_query.get("address_text") != correct_target.key_text:
            raise ValueError(f"Query {query_index} address differs from its key")
        if raw_query.get("answer_absent_from_read_prompt") is not True:
            raise ValueError(f"Query {query_index} does not assert answer absence")
        gold_json = str(raw_query.get("gold_json", ""))
        if gold_json != correct_target.value_json or gold_json != _canonical_value(
            raw_query.get("gold")
        ):
            raise ValueError(f"Query {query_index} gold value differs")
        read_prompt = str(raw_query.get("read_prompt", ""))
        if (
            not read_prompt
            or correct_target.key_text not in read_prompt
            or gold_json in read_prompt
            or "memory_value:" in read_prompt
        ):
            raise ValueError(f"Query {query_index} read prompt leaks an answer")

        expected_raw = _require_mapping(
            raw_query.get("expected_by_state"),
            f"query {query_index} expected_by_state",
        )
        if set(expected_raw) != set(CONDITIONS):
            raise ValueError(f"Query {query_index} expected conditions differ")
        expected = {
            condition: _canonical_value(expected_raw[condition])
            for condition in CONDITIONS
        }
        for condition in ("correct_state", "donor_state", "value_swap", "shuffled_slots"):
            condition_target = _record_at_semantic_slot(
                records_by_condition[condition], target_slot
            )
            if expected[condition] != condition_target.value_json:
                raise ValueError(
                    f"Query {query_index} expectation differs for {condition}"
                )
        if expected["no_state"] != gold_json or expected["pristine_frozen_base"] != gold_json:
            raise ValueError(f"Query {query_index} control expectation differs")

        delta_group = _require_mapping(
            query_deltas.get(str(target_slot)),
            f"query {query_index} counterfactual group",
        )
        if delta_group.get("base_state") != "correct_state":
            raise ValueError(f"Query {query_index} rewrite base differs")
        rewrite = _require_mapping(
            delta_group.get("target_slot_rewrite"),
            f"query {query_index} target_slot_rewrite",
        )
        if rewrite.get("replace_slot") != target_slot:
            raise ValueError(f"Query {query_index} rewrite slot differs")
        raw_replacement = _require_mapping(
            rewrite.get("replacement_record"),
            f"query {query_index} replacement record",
        )
        replacement = _adapt_record(
            raw_replacement,
            location=f"{episode_id}:query-{target_slot}:replacement",
        )
        if (
            replacement.semantic_slot != target_slot
            or replacement.record_id != correct_target.record_id
            or replacement.key_text != correct_target.key_text
            or replacement.physical_slot != correct_target.physical_slot
        ):
            raise ValueError(f"Query {query_index} rewrite changed target identity")
        if replacement.value_json == correct_target.value_json:
            raise ValueError(f"Query {query_index} rewrite did not change the value")
        if expected["target_slot_rewrite"] != replacement.value_json:
            raise ValueError(f"Query {query_index} rewrite expectation differs")
        rewrite_records = list(correct_records)
        replacement_index = next(
            index
            for index, record in enumerate(rewrite_records)
            if record.semantic_slot == target_slot
        )
        rewrite_records[replacement_index] = replacement
        rewrite_records_tuple = _validate_record_set(
            rewrite_records,
            location=f"{episode_id}:query-{target_slot}:rewrite-state",
        )
        rewrite_payload = _sha256_json(
            [
                raw_replacement
                if int(record.get("slot_id", -1)) == target_slot
                else record
                for record in raw_records_by_condition["correct_state"]
            ]
        )
        if rewrite.get("result_record_payload_sha256") != rewrite_payload:
            raise ValueError(f"Query {query_index} rewrite payload hash differs")

        payloads_raw = _require_mapping(
            raw_query.get("record_payload_sha256_by_condition"),
            f"query {query_index} record payload hashes",
        )
        expected_payloads = {
            **payload_by_condition,
            "target_slot_rewrite": rewrite_payload,
        }
        if set(payloads_raw) != set(expected_payloads) or any(
            payloads_raw[name] != digest
            for name, digest in expected_payloads.items()
        ):
            raise ValueError(f"Query {query_index} condition payload hashes differ")

        binding_absence_raw = _require_mapping(
            raw_query.get("binding_absent_from_training"),
            f"query {query_index} training-binding audit",
        )
        binding_absence = {
            str(name): value is True for name, value in binding_absence_raw.items()
        }
        if split != "train" and not all(binding_absence.values()):
            raise ValueError(f"Query {query_index} overlaps a training binding")
        shared_group = str(raw_query.get("shared_correct_runtime_state_group", ""))
        if not shared_group:
            raise ValueError(f"Query {query_index} omits its correct-state group")
        queries.append(
            NaturalQuery(
                query_id=str(raw_query.get("query_id", "")),
                target_slot=target_slot,
                target_record_id=correct_target.record_id,
                address_text=correct_target.key_text,
                read_prompt=read_prompt,
                gold_json=gold_json,
                expected_json_by_condition=expected,
                rewrite_records=rewrite_records_tuple,
                record_payload_sha256_by_condition=expected_payloads,
                binding_absent_from_training=binding_absence,
                shared_correct_runtime_state_group=shared_group,
            )
        )
        if not queries[-1].query_id:
            raise ValueError(f"Query {query_index} omits query_id")
    if {query.target_slot for query in queries} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError("Natural episode queries do not cover all four target slots")
    if len({query.query_id for query in queries}) != RECORDS_PER_EPISODE:
        raise ValueError("Natural episode has duplicate query IDs")
    if len({query.shared_correct_runtime_state_group for query in queries}) != 1:
        raise ValueError("Natural episode correct queries do not share one state group")

    components = tuple(str(value) for value in _require_sequence(
        raw.get("passage_components"), "passage_components"
    ))
    if len(components) != RECORDS_PER_EPISODE or len(set(components)) != len(components):
        raise ValueError("Natural episode passage components are invalid")
    return NaturalEpisode(
        episode_id=episode_id,
        split=split,
        task=task,
        passage_components=components,
        records_by_condition=records_by_condition,
        queries=tuple(sorted(queries, key=lambda query: query.target_slot)),
    )


def _flat_ints(value: Any, description: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{description} unexpectedly contains a batch")
        value = value[0]
    result = tuple(int(item) for item in value)
    if not result:
        raise ValueError(f"{description} is empty")
    return result


def _flat_offsets(value: Any, description: str) -> tuple[tuple[int, int], ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        if len(value) != 1:
            raise ValueError(f"{description} unexpectedly contains a batch")
        value = value[0]
    result = tuple((int(start), int(end)) for start, end in value)
    if not result:
        raise ValueError(f"{description} is empty")
    return result


def _render_user_chat(
    tokenizer: Any,
    content: str,
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str) or content not in rendered:
        raise ValueError("Tokenizer chat template did not preserve the user content")
    return rendered


def _tokenize_offsets(tokenizer: Any, text: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = _flat_ints(encoded["input_ids"], "input_ids")
    attention_mask = _flat_ints(encoded.get("attention_mask", [1] * len(input_ids)), "attention_mask")
    offsets = _flat_offsets(encoded["offset_mapping"], "offset_mapping")
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(offsets):
        raise ValueError("Tokenizer IDs, attention mask, and offsets are misaligned")
    if not all(attention_mask):
        raise ValueError("Unpadded example tokenization contains masked tokens")
    return input_ids, attention_mask, offsets


def _span_mask(
    offsets: Sequence[tuple[int, int]],
    *,
    start: int,
    end: int,
    description: str,
) -> tuple[bool, ...]:
    if start < 0 or end <= start:
        raise ValueError(f"{description} character span is invalid")
    selected = tuple(
        token_end > token_start and token_end > start and token_start < end
        for token_start, token_end in offsets
    )
    if not any(selected):
        raise ValueError(f"{description} selected no tokens")
    return selected


def encode_record(tokenizer: Any, record: NaturalRecord) -> dict[str, Any]:
    rendered = _render_user_chat(
        tokenizer,
        record.write_text,
        add_generation_prompt=False,
    )
    content_start = rendered.find(record.write_text)
    key_local = record.write_text.find(record.key_text)
    value_local = record.write_text.find(record.value_json)
    if key_local < 0 or value_local < 0:
        raise ValueError(f"Record {record.record_id} write spans are absent")
    input_ids, attention_mask, offsets = _tokenize_offsets(tokenizer, rendered)
    key_mask = _span_mask(
        offsets,
        start=content_start + key_local,
        end=content_start + key_local + len(record.key_text),
        description=f"record {record.record_id} key",
    )
    value_mask = _span_mask(
        offsets,
        start=content_start + value_local,
        end=content_start + value_local + len(record.value_json),
        description=f"record {record.record_id} value",
    )
    if any(left and right for left, right in zip(key_mask, value_mask, strict=True)):
        raise ValueError(f"Record {record.record_id} key and value masks overlap")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "key_mask": key_mask,
        "value_mask": value_mask,
        "record_id": record.record_id,
        "semantic_slot": record.semantic_slot,
        "physical_slot": record.physical_slot,
        "value_json": record.value_json,
    }


def encode_query_read(
    tokenizer: Any,
    query: NaturalQuery,
    expected_value_json: str,
) -> dict[str, Any]:
    if expected_value_json in query.read_prompt:
        raise ValueError(f"Query {query.query_id} leaks its expected answer")
    prefix = _render_user_chat(
        tokenizer,
        query.read_prompt,
        add_generation_prompt=True,
    )
    if expected_value_json in prefix:
        raise ValueError(f"Query {query.query_id} rendered prefix leaks its answer")
    rendered = prefix + expected_value_json
    input_ids, attention_mask, offsets = _tokenize_offsets(tokenizer, rendered)
    crossing = [
        (start, end)
        for start, end in offsets
        if start < len(prefix) < end
    ]
    if crossing:
        raise ValueError(
            f"Query {query.query_id} has a tokenizer token crossing the prefix/answer boundary"
        )
    address_local = query.read_prompt.find(query.address_text)
    content_start = prefix.find(query.read_prompt)
    query_mask = _span_mask(
        offsets,
        start=content_start + address_local,
        end=content_start + address_local + len(query.address_text),
        description=f"query {query.query_id} address",
    )
    answer_mask = _span_mask(
        offsets,
        start=len(prefix),
        end=len(rendered),
        description=f"query {query.query_id} answer",
    )
    answer_positions = [index for index, selected in enumerate(answer_mask) if selected]
    if answer_positions != list(range(answer_positions[0], answer_positions[-1] + 1)):
        raise ValueError(f"Query {query.query_id} answer tokens are not contiguous")
    if any(answer_mask[: answer_positions[0]]) or any(
        query_mask[answer_positions[0] :]
    ):
        raise ValueError(f"Query {query.query_id} masks cross the answer boundary")
    answer_ids = tuple(input_ids[index] for index in answer_positions)
    if tokenizer.decode(list(answer_ids), skip_special_tokens=True).strip() == "":
        raise ValueError(
            f"Query {query.query_id} canonical JSON answer decodes to empty text"
        )
    if any(left and right for left, right in zip(query_mask, answer_mask, strict=True)):
        raise ValueError(f"Query {query.query_id} route and answer masks overlap")
    labels = tuple(
        token if selected else -100
        for token, selected in zip(input_ids, answer_mask, strict=True)
    )
    prefix_length = answer_positions[0]
    if any(answer_mask[:prefix_length]) or not any(query_mask[:prefix_length]):
        raise ValueError(f"Query {query.query_id} prefix masks are invalid")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_mask": query_mask,
        "answer_mask": answer_mask,
        "labels": labels,
        "expected_answer_token_ids": tuple(
            token for token, selected in zip(input_ids, answer_mask, strict=True) if selected
        ),
        "query_prefix_length": prefix_length,
    }


def _records_for_query(
    episode: NaturalEpisode,
    query: NaturalQuery,
    condition: str,
) -> tuple[NaturalRecord, ...]:
    if condition == "target_slot_rewrite":
        return query.rewrite_records
    if condition in CONTROL_CONDITIONS:
        return ()
    return episode.records_by_condition[condition]


def build_condition_examples(
    episodes: Sequence[NaturalEpisode],
    tokenizer: Any,
    condition: str,
) -> list[NaturalMemoryExample]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown natural condition {condition!r}")
    examples: list[NaturalMemoryExample] = []
    for episode in episodes:
        for query in episode.queries:
            records = _records_for_query(episode, query, condition)
            encoded_records = tuple(encode_record(tokenizer, record) for record in records)
            read = encode_query_read(
                tokenizer,
                query,
                query.expected_json_by_condition[condition],
            )
            if records:
                target_record = _record_at_semantic_slot(records, query.target_slot)
                target_slot: int | None = target_record.physical_slot
            else:
                target_slot = None
            if condition == "correct_state":
                memory_state_id = query.shared_correct_runtime_state_group
            elif condition == "target_slot_rewrite":
                memory_state_id = f"{episode.episode_id}:{condition}:q{query.target_slot}"
            else:
                memory_state_id = f"{episode.episode_id}:{condition}"
            absent = query.binding_absent_from_training.get(condition)
            examples.append(
                NaturalMemoryExample(
                    row_id=query.query_id,
                    memory_state_id=memory_state_id,
                    source_split=episode.split,
                    source_mapping_offset=0,
                    condition=condition,
                    write_records=encoded_records,
                    write_slots=tuple(record.physical_slot for record in records),
                    read_input_ids=read["input_ids"],
                    read_attention_mask=read["attention_mask"],
                    query_mask=read["query_mask"],
                    answer_mask=read["answer_mask"],
                    labels=read["labels"],
                    target_slot=target_slot,
                    expected_answer_token_ids=read["expected_answer_token_ids"],
                    expected_value=query.expected_json_by_condition[condition],
                    target_slot_rewrite_selection=(
                        {"semantic_target_slot": query.target_slot}
                        if condition == "target_slot_rewrite"
                        else None
                    ),
                    episode_id=episode.episode_id,
                    task=episode.task,
                    semantic_target_slot=query.target_slot,
                    write_record_ids=tuple(record.record_id for record in records),
                    write_semantic_slots=tuple(record.semantic_slot for record in records),
                    write_value_jsons=tuple(record.value_json for record in records),
                    record_payload_sha256=query.record_payload_sha256_by_condition.get(
                        condition,
                        _sha256_json([]),
                    ),
                    binding_absent_from_training=absent,
                    query_prefix_length=read["query_prefix_length"],
                )
            )
    return examples


def build_training_examples(
    episodes: Sequence[NaturalEpisode],
    tokenizer: Any,
    training_conditions: Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
) -> list[NaturalMemoryExample]:
    examples: list[NaturalMemoryExample] = []
    for condition in _parse_training_conditions(training_conditions):
        for example in build_condition_examples(episodes, tokenizer, condition):
            suffix = f"::training-condition={condition}"
            if suffix in example.row_id:
                raise ValueError("Source query ID collides with the training ID policy")
            examples.append(replace(example, row_id=example.row_id + suffix))
    return examples


def audit_training_dataset(
    examples: Sequence[NaturalMemoryExample],
    training_conditions: str | Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
) -> dict[str, Any]:
    """Prove exact condition/task balance and paired counterfactual coverage."""

    selected_conditions = _parse_training_conditions(training_conditions)
    if not examples:
        raise ValueError("Training dataset audit requires at least one example")
    tasks = tuple(sorted({example.task for example in examples}))
    if not tasks or any(not task for task in tasks):
        raise ValueError("Training dataset audit requires named tasks")

    row_ids = [example.row_id for example in examples]
    unique_row_ids = len(set(row_ids)) == len(row_ids)
    observed_conditions = {example.condition for example in examples}
    condition_set_exact = observed_conditions == set(selected_conditions)
    condition_task_counts = Counter(
        (example.condition, example.task) for example in examples
    )
    expected_strata = {
        (condition, task) for condition in selected_conditions for task in tasks
    }
    strata_exact = set(condition_task_counts) == expected_strata
    stratum_sizes = set(condition_task_counts.values())
    strata_balanced = strata_exact and len(stratum_sizes) == 1

    source_query_conditions: dict[str, list[str]] = defaultdict(list)
    source_query_examples: dict[str, list[NaturalMemoryExample]] = defaultdict(list)
    row_id_policy_passed = True
    for example in examples:
        suffix = f"::training-condition={example.condition}"
        if not example.row_id.endswith(suffix):
            row_id_policy_passed = False
            continue
        source_query_id = example.row_id[: -len(suffix)]
        source_query_conditions[source_query_id].append(example.condition)
        source_query_examples[source_query_id].append(example)
    complete_condition_families = sum(
        Counter(conditions) == Counter(selected_conditions)
        for conditions in source_query_conditions.values()
    )
    family_total = len(source_query_conditions)
    paired_condition_coverage = (
        row_id_policy_passed
        and family_total > 0
        and complete_condition_families == family_total
        and len(examples) == family_total * len(selected_conditions)
    )

    family_invariant_failures: list[str] = []
    for source_query_id, family in source_query_examples.items():
        if Counter(example.condition for example in family) != Counter(
            selected_conditions
        ):
            family_invariant_failures.append(source_query_id)
            continue
        signatures = {
            source.canonical_json(
                {
                    "source_split": example.source_split,
                    "episode_id": example.episode_id,
                    "task": example.task,
                    "source_mapping_offset": example.source_mapping_offset,
                    "semantic_target_slot": example.semantic_target_slot,
                    "query_prefix_length": example.query_prefix_length,
                    "read_input_prefix": list(
                        example.read_input_ids[: example.query_prefix_length]
                    ),
                    "read_attention_prefix": list(
                        example.read_attention_mask[: example.query_prefix_length]
                    ),
                    "query_mask_prefix": list(
                        example.query_mask[: example.query_prefix_length]
                    ),
                }
            )
            for example in family
        }
        if len(signatures) != 1:
            family_invariant_failures.append(source_query_id)
    family_invariants_passed = not family_invariant_failures

    payloads = [_training_example_payload(example) for example in examples]
    row_id_set_sha256 = _sha256_json(sorted(row_ids))
    ordered_training_examples_sha256 = _sha256_json(payloads)

    answer_tokens = Counter()
    rows_by_condition = Counter()
    rows_by_task = Counter()
    for example in examples:
        answer_tokens[(example.condition, example.task)] += len(
            example.expected_answer_token_ids
        )
        rows_by_condition[example.condition] += 1
        rows_by_task[example.task] += 1

    passed = (
        unique_row_ids
        and condition_set_exact
        and strata_balanced
        and paired_condition_coverage
        and family_invariants_passed
    )
    return {
        "schema": TRAINING_DATASET_AUDIT_SCHEMA,
        "training_conditions": list(selected_conditions),
        "tasks": list(tasks),
        "rows": len(examples),
        "unique_row_ids": unique_row_ids,
        "row_id_policy": TRAINING_ROW_ID_POLICY,
        "row_id_policy_passed": row_id_policy_passed,
        "sampling_policy": TRAINING_SAMPLING_POLICY,
        "payload_digest_policy": TRAINING_PAYLOAD_DIGEST_POLICY,
        "family_invariant_policy": TRAINING_FAMILY_INVARIANT_POLICY,
        "condition_set_exact": condition_set_exact,
        "condition_task_strata_exact": strata_exact,
        "condition_task_strata_balanced": strata_balanced,
        "rows_per_condition_task": {
            condition: {
                task: condition_task_counts[(condition, task)] for task in tasks
            }
            for condition in selected_conditions
        },
        "answer_tokens_per_condition_task": {
            condition: {
                task: answer_tokens[(condition, task)] for task in tasks
            }
            for condition in selected_conditions
        },
        "rows_by_condition": {
            condition: rows_by_condition[condition]
            for condition in selected_conditions
        },
        "rows_by_task": {task: rows_by_task[task] for task in tasks},
        "source_query_condition_families": family_total,
        "complete_source_query_condition_families": complete_condition_families,
        "paired_condition_coverage": paired_condition_coverage,
        "family_invariants_passed": family_invariants_passed,
        "family_invariant_failure_count": len(family_invariant_failures),
        "training_row_id_set_sha256": row_id_set_sha256,
        "ordered_training_examples_sha256": ordered_training_examples_sha256,
        "passed": passed,
    }


def _training_example_payload(example: NaturalMemoryExample) -> dict[str, Any]:
    """Return the encoded fields that determine one training objective row."""

    return {
        "row_id": example.row_id,
        "memory_state_id": example.memory_state_id,
        "source_split": example.source_split,
        "source_mapping_offset": example.source_mapping_offset,
        "condition": example.condition,
        "write_records": list(example.write_records),
        "write_slots": list(example.write_slots),
        "read_input_ids": list(example.read_input_ids),
        "read_attention_mask": list(example.read_attention_mask),
        "query_mask": list(example.query_mask),
        "answer_mask": list(example.answer_mask),
        "labels": list(example.labels),
        "target_slot": example.target_slot,
        "expected_answer_token_ids": list(example.expected_answer_token_ids),
        "expected_value": example.expected_value,
        "target_slot_rewrite_selection": example.target_slot_rewrite_selection,
        "episode_id": example.episode_id,
        "task": example.task,
        "semantic_target_slot": example.semantic_target_slot,
        "write_record_ids": list(example.write_record_ids),
        "write_semantic_slots": list(example.write_semantic_slots),
        "write_value_jsons": list(example.write_value_jsons),
        "record_payload_sha256": example.record_payload_sha256,
        "binding_absent_from_training": example.binding_absent_from_training,
        "query_prefix_length": example.query_prefix_length,
    }


def training_dataset_audit_checks(audit: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute the self-contained structural claims in a dataset audit."""

    conditions = audit.get("training_conditions")
    tasks = audit.get("tasks")
    rows = audit.get("rows")
    condition_task_counts = audit.get("rows_per_condition_task")
    answer_token_counts = audit.get("answer_tokens_per_condition_task")
    rows_by_condition = audit.get("rows_by_condition")
    rows_by_task = audit.get("rows_by_task")
    valid_conditions = (
        isinstance(conditions, list)
        and bool(conditions)
        and all(isinstance(condition, str) and condition for condition in conditions)
        and len(set(conditions)) == len(conditions)
        and set(conditions).issubset(POSITIVE_CONDITIONS)
    )
    valid_tasks = (
        isinstance(tasks, list)
        and bool(tasks)
        and tasks == sorted(tasks)
        and all(isinstance(task, str) and task for task in tasks)
        and len(set(tasks)) == len(tasks)
    )
    valid_rows = type(rows) is int and rows > 0
    expected_keys = (
        {(condition, task) for condition in conditions for task in tasks}
        if valid_conditions and valid_tasks
        else set()
    )
    observed_counts: dict[tuple[str, str], Any] = {}
    observed_tokens: dict[tuple[str, str], Any] = {}
    nested_counts_valid = isinstance(condition_task_counts, Mapping)
    nested_tokens_valid = isinstance(answer_token_counts, Mapping)
    if nested_counts_valid:
        for condition, task_counts in condition_task_counts.items():
            if not isinstance(task_counts, Mapping):
                nested_counts_valid = False
                continue
            for task, count in task_counts.items():
                observed_counts[(condition, task)] = count
    if nested_tokens_valid:
        for condition, task_counts in answer_token_counts.items():
            if not isinstance(task_counts, Mapping):
                nested_tokens_valid = False
                continue
            for task, count in task_counts.items():
                observed_tokens[(condition, task)] = count
    counts_shape = (
        nested_counts_valid
        and set(observed_counts) == expected_keys
        and all(type(value) is int and value > 0 for value in observed_counts.values())
    )
    tokens_shape = (
        nested_tokens_valid
        and set(observed_tokens) == expected_keys
        and all(type(value) is int and value > 0 for value in observed_tokens.values())
    )
    rows_total = counts_shape and sum(observed_counts.values()) == rows
    strata_balanced = (
        counts_shape
        and bool(observed_counts)
        and len(set(observed_counts.values())) == 1
    )
    rows_by_condition_valid = (
        isinstance(rows_by_condition, Mapping)
        and valid_conditions
        and set(rows_by_condition) == set(conditions)
        and all(
            type(rows_by_condition[condition]) is int
            and rows_by_condition[condition] > 0
            for condition in conditions
        )
        and all(
            rows_by_condition[condition]
            == sum(observed_counts.get((condition, task), -1) for task in tasks)
            for condition in conditions
        )
    )
    rows_by_task_valid = (
        isinstance(rows_by_task, Mapping)
        and valid_tasks
        and set(rows_by_task) == set(tasks)
        and all(
            type(rows_by_task[task]) is int and rows_by_task[task] > 0
            for task in tasks
        )
        and all(
            rows_by_task[task]
            == sum(observed_counts.get((condition, task), -1) for condition in conditions)
            for task in tasks
        )
    )
    family_total = audit.get("source_query_condition_families")
    complete_families = audit.get("complete_source_query_condition_families")
    family_counts_valid = (
        type(family_total) is int
        and family_total > 0
        and type(complete_families) is int
        and complete_families == family_total
        and valid_conditions
        and valid_rows
        and rows == family_total * len(conditions)
    )
    return {
        "schema": audit.get("schema") == TRAINING_DATASET_AUDIT_SCHEMA,
        "conditions_valid": valid_conditions,
        "tasks_valid": valid_tasks,
        "rows_valid": valid_rows,
        "unique_row_ids": audit.get("unique_row_ids") is True,
        "row_id_policy": audit.get("row_id_policy") == TRAINING_ROW_ID_POLICY,
        "row_id_policy_passed": audit.get("row_id_policy_passed") is True,
        "sampling_policy": audit.get("sampling_policy") == TRAINING_SAMPLING_POLICY,
        "payload_digest_policy": audit.get("payload_digest_policy")
        == TRAINING_PAYLOAD_DIGEST_POLICY,
        "family_invariant_policy": audit.get("family_invariant_policy")
        == TRAINING_FAMILY_INVARIANT_POLICY,
        "condition_set_exact": audit.get("condition_set_exact") is True,
        "condition_task_strata_exact": audit.get("condition_task_strata_exact") is True
        and counts_shape,
        "condition_task_strata_balanced": audit.get(
            "condition_task_strata_balanced"
        )
        is True
        and strata_balanced,
        "answer_token_counts_valid": tokens_shape,
        "rows_total_consistent": rows_total,
        "rows_by_condition_consistent": rows_by_condition_valid,
        "rows_by_task_consistent": rows_by_task_valid,
        "paired_condition_coverage": audit.get("paired_condition_coverage") is True
        and family_counts_valid,
        "family_invariants_passed": audit.get("family_invariants_passed") is True
        and audit.get("family_invariant_failure_count") == 0,
        "training_row_id_set_sha256": _is_sha256(
            audit.get("training_row_id_set_sha256")
        ),
        "ordered_training_examples_sha256": _is_sha256(
            audit.get("ordered_training_examples_sha256")
        ),
        "passed": audit.get("passed") is True,
    }


def production_training_dataset_checks(
    audit: Mapping[str, Any],
) -> dict[str, bool]:
    basic = training_dataset_audit_checks(audit)
    rows_per_condition_task = audit.get("rows_per_condition_task")
    expected_rows = {
        condition: {
            task: PRODUCTION_ROWS_PER_CONDITION_TASK for task in PRODUCTION_TASKS
        }
        for condition in SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS
    }
    checks = {
        "audit_passed": all(basic.values()),
        "conditions_exact": audit.get("training_conditions")
        == list(SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS),
        "tasks_exact": audit.get("tasks") == list(PRODUCTION_TASKS),
        "rows_exact": audit.get("rows") == PRODUCTION_TRAINING_ROWS,
        "rows_per_condition_task_exact": rows_per_condition_task == expected_rows,
        "source_query_families_exact": audit.get(
            "source_query_condition_families"
        )
        == PRODUCTION_ROWS_PER_CONDITION_TASK * len(PRODUCTION_TASKS),
        "complete_source_query_families_exact": audit.get(
            "complete_source_query_condition_families"
        )
        == PRODUCTION_ROWS_PER_CONDITION_TASK * len(PRODUCTION_TASKS),
    }
    checks.update({f"audit.{name}": passed for name, passed in basic.items()})
    return checks


def bind_production_training_contract(
    audit: Mapping[str, Any],
    *,
    epochs: int,
    global_batch_size: int,
    requested_max_steps: int | None,
    schedule_mode: str,
    expected_complete_schedule: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Attach and validate the exact data/schedule contract used by a run."""

    if schedule_mode not in {"complete", "preflight"}:
        raise ValueError(f"Unknown training schedule mode: {schedule_mode}")
    expected_epochs, expected_global_batch, expected_complete_updates = (
        expected_complete_schedule or CURRENT_PRODUCTION_COMPLETE_SCHEDULE
    )
    bound = dict(audit)
    complete_epoch_updates = (
        audit["rows"] * epochs // global_batch_size
        if type(audit.get("rows")) is int
        and audit["rows"] % global_batch_size == 0
        else None
    )
    bound["schedule_contract"] = {
        "epochs": epochs,
        "global_batch_size": global_batch_size,
        "rows_divide_global_batch": complete_epoch_updates is not None,
        "complete_epoch_updates": complete_epoch_updates,
        "requested_max_steps": requested_max_steps,
        "complete_epoch_schedule_requested": (
            requested_max_steps is None
            or requested_max_steps == complete_epoch_updates
        ),
    }
    dataset_checks = production_training_dataset_checks(bound)
    bound["production_dataset_contract_checks"] = dataset_checks
    bound["production_dataset_contract_passed"] = all(dataset_checks.values())
    expected_schedule = (
        schedule_mode == "preflight"
        and epochs == expected_epochs
        and global_batch_size == expected_global_batch
        and requested_max_steps == DISTRIBUTED_PREFLIGHT_STEPS
    ) or (
        schedule_mode == "complete"
        and epochs == expected_epochs
        and global_batch_size == expected_global_batch
        and requested_max_steps == expected_complete_updates
    )
    schedule = bound["schedule_contract"]
    schedule_checks = {
        "schedule_mode": schedule_mode in {"complete", "preflight"},
        "epochs_exact": epochs == expected_epochs,
        "global_batch_exact": global_batch_size == expected_global_batch,
        "rows_divide_global_batch": schedule["rows_divide_global_batch"] is True,
        "complete_epoch_updates_exact": (
            complete_epoch_updates == expected_complete_updates
        ),
        "requested_max_steps_exact": expected_schedule,
        "complete_epoch_schedule_requested": schedule[
            "complete_epoch_schedule_requested"
        ]
        is (schedule_mode == "complete"),
    }
    bound["schedule_mode"] = schedule_mode
    bound["schedule_contract_checks"] = schedule_checks
    bound["schedule_contract_passed"] = all(schedule_checks.values())
    bound["production_contract_checks"] = {
        "dataset": bound["production_dataset_contract_passed"],
        "schedule": bound["schedule_contract_passed"],
    }
    bound["production_contract_passed"] = all(
        bound["production_contract_checks"].values()
    )
    return bound


def _validate_production_training_contract_for_schedule(
    audit: Mapping[str, Any],
    *,
    schedule_mode: str,
    expected_complete_schedule: tuple[int, int, int],
) -> bool:
    if not isinstance(audit, Mapping):
        return False
    expected_epochs, expected_global_batch, expected_complete_updates = (
        expected_complete_schedule
    )
    try:
        bound = bind_production_training_contract(
            audit,
            epochs=expected_epochs,
            global_batch_size=expected_global_batch,
            requested_max_steps=(
                DISTRIBUTED_PREFLIGHT_STEPS
                if schedule_mode == "preflight"
                else expected_complete_updates
            ),
            schedule_mode=schedule_mode,
            expected_complete_schedule=expected_complete_schedule,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        bound.get("production_dataset_contract_checks")
        == audit.get("production_dataset_contract_checks")
        and bound.get("production_dataset_contract_passed")
        == audit.get("production_dataset_contract_passed")
        and bound.get("schedule_contract") == audit.get("schedule_contract")
        and bound.get("schedule_mode") == audit.get("schedule_mode")
        and bound.get("schedule_contract_checks") == audit.get("schedule_contract_checks")
        and bound.get("schedule_contract_passed")
        == audit.get("schedule_contract_passed")
        and bound.get("production_contract_checks")
        == audit.get("production_contract_checks")
        and bound.get("production_contract_passed")
        == audit.get("production_contract_passed")
        and audit.get("production_contract_passed") is True
    )


def validate_production_training_contract(
    audit: Mapping[str, Any], *, schedule_mode: str
) -> bool:
    """Validate only the current production schedule for a new launch."""

    return _validate_production_training_contract_for_schedule(
        audit,
        schedule_mode=schedule_mode,
        expected_complete_schedule=CURRENT_PRODUCTION_COMPLETE_SCHEDULE,
    )


def validate_retained_production_training_contract(
    audit: Mapping[str, Any], *, schedule_mode: str
) -> bool:
    """Validate an immutable current or explicitly retained proof schedule."""

    if schedule_mode != "complete":
        return validate_production_training_contract(
            audit,
            schedule_mode=schedule_mode,
        )
    return any(
        _validate_production_training_contract_for_schedule(
            audit,
            schedule_mode=schedule_mode,
            expected_complete_schedule=schedule,
        )
        for schedule in RETAINED_PRODUCTION_COMPLETE_SCHEDULES
    )


def _read_json_file(path: Path, description: str) -> Mapping[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{description} is not a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {description}: {resolved}") from error
    return _require_mapping(value, description)


def _verify_replication_receipt(
    payload: Mapping[str, Any],
    *,
    receipt_field: str,
    payload_scope: str,
    description: str,
) -> str:
    receipt = _require_mapping(payload.get(receipt_field), f"{description} receipt")
    unsigned = dict(payload)
    unsigned.pop(receipt_field, None)
    payload_sha256 = _sha256_json(unsigned)
    if (
        receipt.get("algorithm") != "sha256"
        or receipt.get("payload_scope") != payload_scope
        or receipt.get("payload_sha256") != payload_sha256
    ):
        raise ValueError(f"{description} self-receipt is invalid")
    return payload_sha256


def validate_replication_authorization(
    *,
    source_manifest: Path,
    profile: str,
    seed: int,
    replication_protocol: Path | None,
    replication_amendment: Path | None,
    replication_id: str | None,
) -> Mapping[str, Any] | None:
    """Authorize only the non-default seeds frozen before replication."""

    authorization_arguments = (
        replication_protocol,
        replication_amendment,
        replication_id,
    )
    if not any(value is not None for value in authorization_arguments):
        if profile in FORMAL_PROFILES and seed != PRODUCTION_SEED:
            raise ValueError(
                "A non-default formal seed requires replication authorization"
            )
        return None
    if not all(value is not None for value in authorization_arguments):
        raise ValueError(
            "Replication authorization requires protocol, amendment, and ID"
        )
    if profile not in FORMAL_PROFILES:
        raise ValueError("Replication authorization is restricted to formal profiles")

    protocol_path = (replication_protocol or Path()).expanduser().resolve(strict=True)
    protocol = _read_json_file(protocol_path, "replication protocol")
    protocol_file_sha256 = source.sha256_file(protocol_path)
    protocol_payload_sha256 = _verify_replication_receipt(
        protocol,
        receipt_field="protocol_receipt",
        payload_scope="canonical_protocol_without_receipt",
        description="Replication protocol",
    )
    protocol_runner_sha256 = _require_mapping(
        protocol.get("source_code_sha256"),
        "replication protocol source code bindings",
    ).get("runner")
    if (
        protocol.get("schema") != REPLICATION_PROTOCOL_SCHEMA
        or protocol_file_sha256
        != PREREGISTERED_REPLICATION_PROTOCOL_FILE_SHA256
        or protocol_payload_sha256
        != PREREGISTERED_REPLICATION_PROTOCOL_PAYLOAD_SHA256
        or protocol_runner_sha256 != PREREGISTERED_REPLICATION_RUNNER_SHA256
    ):
        raise ValueError("Replication protocol differs from the preregistered contract")

    amendment_path = (replication_amendment or Path()).expanduser().resolve(
        strict=True
    )
    amendment = _read_json_file(amendment_path, "replication amendment")
    amendment_payload_sha256 = _verify_replication_receipt(
        amendment,
        receipt_field="amendment_receipt",
        payload_scope="canonical_amendment_without_receipt",
        description="Replication amendment",
    )
    original_protocol = _require_mapping(
        amendment.get("original_protocol"),
        "replication amendment original protocol binding",
    )
    expected_original_protocol = {
        "git_commit": PREREGISTERED_REPLICATION_PROTOCOL_COMMIT,
        "file_sha256": PREREGISTERED_REPLICATION_PROTOCOL_FILE_SHA256,
        "payload_sha256": PREREGISTERED_REPLICATION_PROTOCOL_PAYLOAD_SHA256,
    }
    runner_change = _require_mapping(
        amendment.get("runner_change"),
        "replication amendment runner change",
    )
    current_runner_sha256 = source.sha256_file(Path(__file__).resolve())
    scope = _require_mapping(
        amendment.get("scope"),
        "replication amendment scope",
    )
    expected_scope = {
        "classification": "infrastructure_only",
        "data_changed": False,
        "gate_changed": False,
        "hyperparameters_changed": False,
        "training_math_changed": False,
    }
    replications = _require_sequence(
        protocol.get("replications"),
        "replication protocol replications",
    )
    protocol_replication_ids = [
        replication.get("id")
        for replication in replications
        if isinstance(replication, Mapping)
    ]
    if (
        amendment.get("schema") != REPLICATION_AMENDMENT_SCHEMA
        or dict(original_protocol) != expected_original_protocol
        or runner_change.get("old_sha256")
        != PREREGISTERED_REPLICATION_RUNNER_SHA256
        or runner_change.get("new_sha256") != current_runner_sha256
        or dict(scope) != expected_scope
        or amendment.get("authorized_replication_ids")
        != protocol_replication_ids
        or amendment.get("discovered_before_training_output") is not True
    ):
        raise ValueError("Replication amendment does not authorize this runner")

    matches = [
        replication
        for replication in replications
        if isinstance(replication, Mapping)
        and replication.get("id") == replication_id
    ]
    if len(matches) != 1:
        raise ValueError("Replication ID is not uniquely preregistered")
    replication = matches[0]
    if replication.get("training_seed") != seed:
        raise ValueError("Training seed differs from the preregistered replication")
    split_seed = replication.get("split_seed")
    if type(split_seed) is not int:
        raise ValueError("Preregistered replication split seed is invalid")

    manifest_path = source_manifest.expanduser().resolve(strict=True)
    manifest = _read_json_file(manifest_path, "replication source manifest")
    if (
        manifest.get("schema") != source.SCHEMA
        or manifest.get("hf_endpoint") != HF_MIRROR_ENDPOINT
        or not source.verify_manifest_receipt(manifest)
    ):
        raise ValueError("Replication source manifest receipt is invalid")
    benchmark_split_seed = _require_mapping(
        manifest.get("benchmark_contract"),
        "replication benchmark contract",
    ).get("split_seed")
    policy_split_seed = _require_mapping(
        manifest.get("split_policy"),
        "replication split policy",
    ).get("seed")
    if benchmark_split_seed != split_seed or policy_split_seed != split_seed:
        raise ValueError("Source manifest split seed differs from the replication")

    return {
        "schema": REPLICATION_AUTHORIZATION_SCHEMA,
        "replication_id": replication_id,
        "split_seed": split_seed,
        "training_seed": seed,
        "protocol_file": protocol_path.name,
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_payload_sha256": protocol_payload_sha256,
        "protocol_git_commit": PREREGISTERED_REPLICATION_PROTOCOL_COMMIT,
        "amendment_file": amendment_path.name,
        "amendment_file_sha256": source.sha256_file(amendment_path),
        "amendment_payload_sha256": amendment_payload_sha256,
        "runner_original_sha256": PREREGISTERED_REPLICATION_RUNNER_SHA256,
        "runner_authorized_sha256": current_runner_sha256,
        "scope": dict(scope),
    }


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        requested = raw_path.expanduser()
        if requested.is_symlink():
            raise ValueError(f"Bound artifact must not be a symbolic link: {requested}")
        path = requested.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Bound artifact is not a regular file: {path}")
        key = str(path)
        if key in snapshot:
            continue
        snapshot[key] = {
            "bytes": path.stat().st_size,
            "sha256": source.sha256_file(path),
        }
    return dict(sorted(snapshot.items()))


def assert_snapshot_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    *,
    description: str,
) -> dict[str, dict[str, Any]]:
    paths = [Path(path) for path in before]
    try:
        after = snapshot_files(paths)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} artifacts changed or disappeared") from error
    normalized_before = {
        str(path): dict(fingerprint) for path, fingerprint in before.items()
    }
    if after != normalized_before:
        changed = sorted(
            path
            for path in set(after) | set(normalized_before)
            if after.get(path) != normalized_before.get(path)
        )
        raise ValueError(f"{description} artifacts changed: {changed}")
    return after


def _validate_manifest_common(
    manifest: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    if manifest.get("schema") != source.SCHEMA:
        raise ValueError(f"Natural manifest schema differs at {path}")
    if manifest.get("hf_endpoint") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"Natural manifest does not bind the HF mirror at {path}")
    if not source.verify_manifest_receipt(manifest):
        raise ValueError(f"Natural manifest self-receipt is invalid at {path}")
    split_audit = _require_mapping(manifest.get("split_audit"), "split_audit")
    signature_audit = _require_mapping(
        manifest.get("signature_audit"), "signature_audit"
    )
    if (
        split_audit.get("passage_disjoint") is not True
        or split_audit.get("normalized_units_passage_disjoint") is not True
        or split_audit.get("normalized_signature_cross_split_overlap_count") != 0
        or signature_audit.get("signature_components_atomic") is not True
    ):
        raise ValueError(f"Natural manifest split isolation failed at {path}")
    binding = _require_mapping(manifest.get("model_binding"), "model_binding")
    if (
        binding.get("weights_bound") is not True
        or binding.get("hf_endpoint") != HF_MIRROR_ENDPOINT
        or not isinstance(binding.get("binding_sha256"), str)
    ):
        raise ValueError(f"Natural manifest lacks a formal local model binding at {path}")


def _validate_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    profile: str,
) -> str:
    _validate_manifest_common(manifest, path=manifest_path)
    materialized = tuple(manifest.get("materialized_splits", ()))
    build_profile = manifest.get("build_profile")
    if profile in {"train", "development"}:
        if build_profile != "development" or materialized != (
            "train",
            "development",
        ):
            raise ValueError("Train/development run requires an exact development package")
        if (manifest_path.parent / "sealed_validation.jsonl").exists():
            raise ValueError("Development package unexpectedly exposes sealed validation")
        return "train" if profile == "train" else "development"
    if profile == "sealed_validation":
        if build_profile != "sealed_validation" or materialized != (
            "sealed_validation",
        ):
            raise ValueError("Sealed run requires an exact sealed-validation package")
        for forbidden in ("train.jsonl", "development.jsonl"):
            if (manifest_path.parent / forbidden).exists():
                raise ValueError(f"Sealed package unexpectedly exposes {forbidden}")
        sealed_lock = _require_mapping(manifest.get("sealed_lock"), "sealed_lock")
        lock_receipt = _require_mapping(sealed_lock.get("receipt"), "sealed lock receipt")
        if (
            lock_receipt.get("schema") != source.SEALED_LOCK_SCHEMA
            or lock_receipt.get("configuration_frozen") is not True
            or lock_receipt.get("benchmark_contract_sha256")
            != manifest.get("benchmark_contract_sha256")
        ):
            raise ValueError("Sealed package does not contain a valid frozen lock")
        return "sealed_validation"
    raise ValueError(f"Unknown natural profile {profile!r}")


def _load_split(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    split: str,
) -> tuple[tuple[NaturalEpisode, ...], Path]:
    output_hashes = _require_mapping(manifest.get("output_sha256"), "output_sha256")
    expected_hash = output_hashes.get(split)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Manifest does not bind selected split {split}")
    requested_path = manifest_path.parent / f"{split}.jsonl"
    if requested_path.is_symlink():
        raise ValueError(f"Selected split must not be a symbolic link: {requested_path}")
    path = requested_path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Selected split is not a regular file: {path}")
    before_hash = source.sha256_file(path)
    if before_hash != expected_hash:
        raise ValueError(f"Selected split hash differs before read: {split}")
    episodes: list[NaturalEpisode] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            episode = adapt_episode(raw)
            if episode.split != split:
                raise ValueError(f"Episode split differs at {path}:{line_number}")
            episodes.append(episode)
    if source.sha256_file(path) != before_hash:
        raise ValueError(f"Selected split changed while being read: {split}")
    if not episodes:
        raise ValueError(f"Selected split is empty: {split}")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError(f"Selected split has duplicate episode IDs: {split}")
    expected_count = _require_mapping(
        manifest.get("episode_audit"), "episode_audit"
    ).get("episodes_by_split", {}).get(split)
    if expected_count != len(episodes):
        raise ValueError(f"Selected split count differs: {split}")
    return tuple(episodes), path


def load_profile_bundle(
    source_manifest: Path,
    *,
    profile: str,
) -> ProfileBundle:
    """Open only the JSONL files authorized by the selected run profile."""

    configure_hf_mirror()
    if profile not in PROFILES:
        raise ValueError(f"Unknown natural profile {profile!r}")
    requested_manifest = source_manifest.expanduser()
    if requested_manifest.is_symlink():
        raise ValueError(
            f"Natural source manifest must not be a symbolic link: {requested_manifest}"
        )
    manifest_path = requested_manifest.resolve(strict=True)
    manifest = _read_json_file(manifest_path, "natural source manifest")
    evaluation_split = _validate_profile_manifest(
        manifest,
        manifest_path=manifest_path,
        profile=profile,
    )
    selected_paths: list[Path] = [manifest_path]
    if profile == "sealed_validation":
        train_episodes: tuple[NaturalEpisode, ...] = ()
        evaluation_episodes, evaluation_path = _load_split(
            manifest,
            manifest_path=manifest_path,
            split="sealed_validation",
        )
        selected_paths.append(evaluation_path)
        development_manifest: Mapping[str, Any] = {}
        sealed_manifest: Mapping[str, Any] | None = manifest
    else:
        train_episodes, train_path = _load_split(
            manifest,
            manifest_path=manifest_path,
            split="train",
        )
        selected_paths.append(train_path)
        if profile == "train":
            evaluation_episodes = train_episodes
        else:
            evaluation_episodes, evaluation_path = _load_split(
                manifest,
                manifest_path=manifest_path,
                split="development",
            )
            selected_paths.append(evaluation_path)
        development_manifest = manifest
        sealed_manifest = None

    train_components = {
        component
        for episode in train_episodes
        for component in episode.passage_components
    }
    evaluation_components = {
        component
        for episode in evaluation_episodes
        for component in episode.passage_components
    }
    component_overlap = (
        train_components & evaluation_components if profile != "train" else set()
    )
    if component_overlap:
        raise ValueError("Loaded train and evaluation passage components overlap")
    heldout_queries = [
        query
        for episode in evaluation_episodes
        if episode.split != "train"
        for query in episode.queries
    ]
    heldout_novel = all(
        query.binding_absent_from_training
        and all(query.binding_absent_from_training.values())
        for query in heldout_queries
    )
    if heldout_queries and not heldout_novel:
        raise ValueError("Evaluation bindings overlap training")
    tasks = sorted({episode.task for episode in evaluation_episodes})
    eligibility = {
        "profile": profile,
        "evaluation_split": evaluation_split,
        "optimizer_authorized": profile != "sealed_validation",
        "opened_splits": (
            ["train"]
            if profile == "train"
            else ["train", "development"]
            if profile == "development"
            else ["sealed_validation"]
        ),
        "train_episodes": len(train_episodes),
        "evaluation_episodes": len(evaluation_episodes),
        "evaluation_tasks": tasks,
        "passage_component_overlap_count": len(component_overlap),
        "heldout_binding_novel": heldout_novel,
        "manifest_receipt_valid": True,
        "passed": not component_overlap and (not heldout_queries or heldout_novel),
    }
    return ProfileBundle(
        profile=profile,
        train_episodes=train_episodes,
        evaluation_episodes=evaluation_episodes,
        evaluation_split=evaluation_split,
        development_manifest=development_manifest,
        sealed_manifest=sealed_manifest,
        source_paths=tuple(selected_paths),
        model_binding=_require_mapping(manifest["model_binding"], "model_binding"),
        eligibility=eligibility,
    )


def resolve_model_artifacts(
    model_binding: Mapping[str, Any],
    *,
    model_path: Path | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    declared_path = model_binding.get("local_model_path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("Manifest does not declare a local model path")
    declared_requested = Path(declared_path).expanduser()
    if declared_requested.is_symlink():
        raise ValueError("Manifest local model path must not be a symbolic link")
    declared = declared_requested.resolve(strict=True)
    runtime_requested = declared_requested if model_path is None else model_path.expanduser()
    if runtime_requested.is_symlink():
        raise ValueError("Runtime model path must not be a symbolic link")
    resolved = runtime_requested.resolve(strict=True)
    if resolved != declared or not resolved.is_dir():
        raise ValueError("Runtime model path differs from the manifest binding")
    artifacts = _require_mapping(
        model_binding.get("local_artifacts"), "local model artifacts"
    )
    if not artifacts:
        raise ValueError("Manifest model binding contains no local artifacts")
    paths: list[Path] = []
    for name, raw_fingerprint in sorted(artifacts.items()):
        if Path(name).name != name:
            raise ValueError(f"Model artifact name is not local: {name!r}")
        fingerprint = _require_mapping(raw_fingerprint, f"model artifact {name}")
        requested_artifact = resolved / name
        if requested_artifact.is_symlink():
            raise ValueError(f"Model artifact must not be a symbolic link: {name}")
        path = requested_artifact.resolve(strict=True)
        if path.parent != resolved or not path.is_file():
            raise ValueError(f"Model artifact is invalid: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": source.sha256_file(path)}
        if actual != dict(fingerprint):
            raise ValueError(f"Model artifact hash differs: {name}")
        paths.append(path)
    return resolved, tuple(paths)


def select_complete_episodes(
    episodes: Sequence[NaturalEpisode], limit: int | None
) -> tuple[NaturalEpisode, ...]:
    if limit is None:
        return tuple(episodes)
    if limit <= 0:
        raise ValueError("Episode limit must be positive")
    return tuple(episodes[:limit])


def _batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _evaluation_batches(
    examples: Sequence[NaturalMemoryExample],
    *,
    condition: str,
    batch_size: int,
) -> Iterable[Sequence[NaturalMemoryExample]]:
    if condition not in SHARED_WRITE_CONDITIONS:
        yield from _batches(examples, batch_size)
        return
    if batch_size < RECORDS_PER_EPISODE:
        raise ValueError(
            "Shared-write evaluation batch size must hold one complete family"
        )
    families: dict[str, list[NaturalMemoryExample]] = {}
    for example in examples:
        families.setdefault(example.memory_state_id, []).append(example)
    ordered_families: list[tuple[NaturalMemoryExample, ...]] = []
    for state_id, raw_family in families.items():
        family = tuple(
            sorted(raw_family, key=lambda example: example.semantic_target_slot)
        )
        if [example.semantic_target_slot for example in family] != list(
            range(RECORDS_PER_EPISODE)
        ):
            raise ValueError(f"{condition} state family {state_id} is incomplete")
        ordered_families.append(family)
    families_per_batch = max(1, batch_size // RECORDS_PER_EPISODE)
    for family_batch in _batches(ordered_families, families_per_batch):
        yield tuple(example for family in family_batch for example in family)


def _control_state_absence_evidence(
    row_ids: Sequence[str],
    *,
    before_read_state_names: Sequence[str],
    after_read_state_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    before = list(before_read_state_names)
    after = list(after_read_state_names)
    return {
        row_id: {
            "projected_kv_state_names_before_read": before,
            "projected_kv_state_names_after_read": after,
            "projected_kv_state_absent_before_read": not before,
            "projected_kv_state_absent_after_read": not after,
        }
        for row_id in row_ids
    }


def _projected_kv_state_names(model: torch.nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    for module_name, module in iter_delta_mem_modules(model):
        for attribute in (
            "projected_kv_keys",
            "projected_kv_values",
            "projected_kv_occupied",
            "projected_kv_surprise",
        ):
            if getattr(module, attribute, None) is not None:
                names.append(f"{module_name}.{attribute}")
    return tuple(sorted(names))


def _decode_prediction(
    tokenizer: Any,
    token_ids: Sequence[int],
    expected_json: str,
) -> dict[str, Any]:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()
    parsed_json: Any = None
    parse_valid = False
    canonical = None
    try:
        parsed_json = json.loads(text)
        canonical = source.canonical_json(parsed_json)
        parse_valid = True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "text": text,
        "json_parse_valid": parse_valid,
        "canonical_json": canonical,
        "structured_json_exact": canonical == expected_json,
    }


def evaluate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[NaturalMemoryExample],
    *,
    condition: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    greedy: bool,
) -> dict[str, Any]:
    if not examples or any(example.condition != condition for example in examples):
        raise ValueError(f"Evaluation examples do not match condition {condition}")
    module_names = [name for name, _ in iter_delta_mem_modules(model)]
    if condition == "pristine_frozen_base" and module_names:
        raise RuntimeError(
            "Pristine frozen-base evaluation must use a model without Delta-Mem attached"
        )
    answer_exact: list[bool] = []
    structured_exact: list[bool] = []
    greedy_exact: list[bool] = []
    greedy_structured_exact: list[bool] = []
    token_correct = 0
    token_total = 0
    route_correct = 0
    route_total = 0
    route_by_layer = {
        name: {"correct": 0, "total": 0} for name in module_names
    }
    answer_predictions_by_row: dict[str, dict[str, Any]] = {}
    route_predictions_by_row: dict[str, dict[str, int]] = {}
    state_digest_by_row: dict[str, str] = {}
    control_state_absence_by_row: dict[str, dict[str, Any]] = {}
    occupancy_correct = 0
    occupancy_total = 0
    write_route_correct = 0
    write_route_total = 0
    absent_modules = 0
    possible_modules = 0
    task_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "teacher_forced_exact": 0,
            "teacher_forced_structured_exact": 0,
            "greedy_exact": 0,
            "greedy_structured_exact": 0,
            "route_correct": 0,
            "route_total": 0,
        }
    )

    model.eval()
    with torch.no_grad():
        for raw_batch in _evaluation_batches(
            list(examples), condition=condition, batch_size=batch_size
        ):
            batch = collate_examples(
                raw_batch,
                pad_token_id=pad_token_id,
                device=device,
            )
            write_audit = _write_episode_batch(model, batch, dtype=dtype)
            occupancy_correct += write_audit["full_occupancy_count"]
            occupancy_total += write_audit["full_occupancy_total"]
            write_route_correct += write_audit["forced_write_route_match_count"]
            write_route_total += write_audit["forced_write_route_total"]
            if batch.write_records:
                digests = _state_digests(model, len(batch.examples))
                for example, digest in zip(batch.examples, digests, strict=True):
                    state_digest_by_row[example.row_id] = digest
            before_read_state_names = (
                _projected_kv_state_names(model)
                if condition in CONTROL_CONDITIONS
                else ()
            )

            logits, route_logits = _read_episode_batch(model, batch, dtype=dtype)
            exact, batch_token_correct, batch_token_total = (
                _answer_exact_predictions(logits, batch.labels)
            )
            teacher_predictions, expected_rows = _answer_prediction_token_ids(
                logits, batch.labels
            )
            generated_rows = (
                _greedy_answer_predictions(
                    model,
                    batch,
                    pad_token_id=pad_token_id,
                    dtype=dtype,
                )
                if greedy
                else [None] * len(batch.examples)
            )
            if condition in CONTROL_CONDITIONS:
                evidence = _control_state_absence_evidence(
                    [example.row_id for example in batch.examples],
                    before_read_state_names=before_read_state_names,
                    after_read_state_names=_projected_kv_state_names(model),
                )
                overlap = set(control_state_absence_by_row) & set(evidence)
                if overlap:
                    raise ValueError(
                        "Duplicate control state-absence rows: "
                        + ", ".join(sorted(overlap))
                    )
                control_state_absence_by_row.update(evidence)

            batch_row_predictions = {
                example.row_id: {} for example in batch.examples
            }
            possible_modules += len(module_names) * len(batch.examples)
            absent_modules += (len(module_names) - len(route_logits)) * len(
                batch.examples
            )
            if condition in CONTROL_CONDITIONS:
                if route_logits:
                    raise RuntimeError(
                        f"{condition} unexpectedly exposed projected-KV routes"
                    )
            else:
                if set(route_logits) != set(module_names) or not route_logits:
                    raise RuntimeError(
                        f"{condition} omitted projected-KV route evidence"
                    )
                for name, layer_logits in route_logits.items():
                    selected = selected_route_logits(layer_logits, batch.query_mask)
                    predictions = selected.argmax(dim=-1)
                    matches = predictions.eq(batch.target_slots)
                    matched = int(matches.sum().item())
                    route_correct += matched
                    route_total += len(batch.examples)
                    route_by_layer[name]["correct"] += matched
                    route_by_layer[name]["total"] += len(batch.examples)
                    for example, predicted, is_match in zip(
                        batch.examples,
                        predictions.detach().cpu().tolist(),
                        matches.detach().cpu().tolist(),
                        strict=True,
                    ):
                        batch_row_predictions[example.row_id][name] = int(predicted)
                        task_counts[example.task]["route_correct"] += int(is_match)
                        task_counts[example.task]["route_total"] += 1
                route_predictions_by_row.update(batch_row_predictions)

            for example, predicted, expected, is_exact, generated in zip(
                batch.examples,
                teacher_predictions,
                expected_rows,
                exact,
                generated_rows,
                strict=True,
            ):
                if example.row_id in answer_predictions_by_row:
                    raise ValueError(f"Duplicate evaluation row ID: {example.row_id}")
                if expected != example.expected_answer_token_ids:
                    raise RuntimeError(
                        f"Evaluation labels differ from {example.row_id} expectation"
                    )
                teacher_evidence = _decode_prediction(
                    tokenizer, predicted, example.expected_value
                )
                structured = bool(teacher_evidence["structured_json_exact"])
                structured_exact.append(structured)
                task = task_counts[example.task]
                task["rows"] += 1
                task["teacher_forced_exact"] += int(is_exact)
                task["teacher_forced_structured_exact"] += int(structured)
                greedy_evidence = None
                greedy_is_exact = None
                if generated is not None:
                    greedy_evidence = _decode_prediction(
                        tokenizer, generated, example.expected_value
                    )
                    greedy_is_exact = generated == example.expected_answer_token_ids
                    greedy_exact.append(greedy_is_exact)
                    greedy_structured = bool(
                        greedy_evidence["structured_json_exact"]
                    )
                    greedy_structured_exact.append(greedy_structured)
                    task["greedy_exact"] += int(greedy_is_exact)
                    task["greedy_structured_exact"] += int(greedy_structured)
                answer_predictions_by_row[example.row_id] = {
                    "episode_id": example.episode_id,
                    "task": example.task,
                    "expected_value_json": example.expected_value,
                    "expected_answer_token_ids": list(expected),
                    "teacher_forced_prediction_token_ids": list(predicted),
                    "teacher_forced_exact": bool(is_exact),
                    "teacher_forced_json": teacher_evidence,
                    "greedy_generated_token_ids": (
                        list(generated) if generated is not None else None
                    ),
                    "greedy_exact": greedy_is_exact,
                    "greedy_json": greedy_evidence,
                }
            answer_exact.extend(exact)
            token_correct += batch_token_correct
            token_total += batch_token_total
            reset_delta_mem_states(model)

    per_task: dict[str, dict[str, Any]] = {}
    for task, counts in sorted(task_counts.items()):
        rows = counts["rows"]
        per_task[task] = {
            **counts,
            "teacher_forced_answer_exact_accuracy": (
                counts["teacher_forced_exact"] / rows
            ),
            "teacher_forced_structured_json_exact_accuracy": (
                counts["teacher_forced_structured_exact"] / rows
            ),
            "greedy_answer_exact_accuracy": (
                counts["greedy_exact"] / rows if greedy else None
            ),
            "greedy_structured_json_exact_accuracy": (
                counts["greedy_structured_exact"] / rows if greedy else None
            ),
            "semantic_route_accuracy": (
                counts["route_correct"] / counts["route_total"]
                if counts["route_total"]
                else None
            ),
        }
    layer_metrics = {
        name: {
            **counts,
            "accuracy": counts["correct"] / counts["total"]
            if counts["total"]
            else None,
        }
        for name, counts in route_by_layer.items()
    }
    return {
        "condition": condition,
        "rows": len(examples),
        "teacher_forced_answer_exact_count": sum(answer_exact),
        "teacher_forced_answer_exact_accuracy": sum(answer_exact) / len(answer_exact),
        "teacher_forced_structured_json_exact_count": sum(structured_exact),
        "teacher_forced_structured_json_exact_accuracy": (
            sum(structured_exact) / len(structured_exact)
        ),
        "teacher_forced_answer_token_correct": token_correct,
        "teacher_forced_answer_token_total": token_total,
        "teacher_forced_answer_token_accuracy": token_correct / token_total,
        "greedy_answer_evaluated": greedy,
        "greedy_answer_exact_count": sum(greedy_exact) if greedy else None,
        "greedy_answer_exact_accuracy": (
            sum(greedy_exact) / len(greedy_exact) if greedy else None
        ),
        "greedy_structured_json_exact_count": (
            sum(greedy_structured_exact) if greedy else None
        ),
        "greedy_structured_json_exact_accuracy": (
            sum(greedy_structured_exact) / len(greedy_structured_exact)
            if greedy
            else None
        ),
        "answer_predictions_by_row": answer_predictions_by_row,
        "semantic_route_correct": route_correct,
        "semantic_route_total": route_total,
        "semantic_route_accuracy": route_correct / route_total if route_total else None,
        "route_by_layer": layer_metrics,
        "route_predictions_by_row": route_predictions_by_row,
        "full_occupancy_count": occupancy_correct,
        "full_occupancy_total": occupancy_total,
        "full_occupancy_fraction": (
            occupancy_correct / occupancy_total if occupancy_total else None
        ),
        "forced_write_route_correct": write_route_correct,
        "forced_write_route_total": write_route_total,
        "forced_write_route_accuracy": (
            write_route_correct / write_route_total if write_route_total else None
        ),
        "route_absent_module_rows": absent_modules,
        "route_possible_module_rows": possible_modules,
        "route_absent_fraction": (
            absent_modules / possible_modules if possible_modules else 1.0
        ),
        "state_digest_by_row": state_digest_by_row,
        "runtime_state_absence_rows": len(control_state_absence_by_row),
        "runtime_state_absence_fraction": (
            sum(
                row["projected_kv_state_absent_before_read"]
                and row["projected_kv_state_absent_after_read"]
                for row in control_state_absence_by_row.values()
            )
            / len(control_state_absence_by_row)
            if control_state_absence_by_row
            else None
        ),
        "runtime_state_absence_by_row": control_state_absence_by_row,
        "delta_heads_disabled": condition == "pristine_frozen_base",
        "adapter_attached": bool(module_names),
        "attached_delta_mem_module_count": len(module_names),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "pristine_base_adapter_excluded": (
            condition == "pristine_frozen_base" and not module_names
        ),
        "per_task": per_task,
    }


def audit_correct_state_identity(
    examples: Sequence[NaturalMemoryExample],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not examples or any(example.condition != "correct_state" for example in examples):
        raise ValueError("Correct-state identity audit received other conditions")
    groups: dict[str, list[NaturalMemoryExample]] = defaultdict(list)
    for example in examples:
        groups[example.memory_state_id].append(example)
    state_digests = _require_mapping(
        evaluation.get("state_digest_by_row"), "correct state digests"
    )
    route_predictions = _require_mapping(
        evaluation.get("route_predictions_by_row"), "correct route predictions"
    )
    module_names = tuple(_require_mapping(
        evaluation.get("route_by_layer"), "correct route layers"
    ))
    identical_families = 0
    all_four_route_correct = 0
    route_family_total = 0
    families: dict[str, dict[str, Any]] = {}
    for group_id, raw_family in sorted(groups.items()):
        family = sorted(raw_family, key=lambda example: example.semantic_target_slot)
        if [example.semantic_target_slot for example in family] != list(
            range(RECORDS_PER_EPISODE)
        ):
            raise ValueError(f"Correct-state family {group_id} is incomplete")
        digests = [str(state_digests.get(example.row_id, "")) for example in family]
        if any(not digest for digest in digests):
            raise ValueError(f"Correct-state family {group_id} lacks tensor digests")
        identical = len(set(digests)) == 1
        identical_families += int(identical)
        layer_all_correct: dict[str, bool] = {}
        for module_name in module_names:
            matches = [
                int(_require_mapping(
                    route_predictions.get(example.row_id),
                    f"route row {example.row_id}",
                ).get(module_name, -1))
                == int(example.target_slot)
                for example in family
            ]
            passed = all(matches)
            layer_all_correct[module_name] = passed
            route_family_total += 1
            all_four_route_correct += int(passed)
        families[group_id] = {
            "row_ids": [example.row_id for example in family],
            "runtime_tensor_state_digests": digests,
            "runtime_byte_identical": identical,
            "layer_all_four_routes_correct": layer_all_correct,
        }
    family_count = len(families)
    return {
        "families": family_count,
        "runtime_byte_identical_families": identical_families,
        "runtime_byte_identical_state_fraction": identical_families / family_count,
        "family_layer_all_four_correct": all_four_route_correct,
        "family_layer_total": route_family_total,
        "family_layer_all_four_correct_fraction": (
            all_four_route_correct / route_family_total
        ),
        "families_by_id": families,
    }


def _validated_state_digests(
    examples: Sequence[NaturalMemoryExample],
    evaluation: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, str]:
    row_ids = [example.row_id for example in examples]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{condition} contains duplicate row IDs")
    raw_digests = _require_mapping(
        evaluation.get("state_digest_by_row"), f"{condition} state digests"
    )
    if set(raw_digests) != set(row_ids):
        raise ValueError(f"{condition} tensor-state digest rows differ")
    digests = {row_id: str(raw_digests[row_id]) for row_id in row_ids}
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests.values()
    ):
        raise ValueError(f"{condition} contains an invalid tensor-state digest")
    return digests


def audit_runtime_state_causality(
    examples_by_condition: Mapping[str, Sequence[NaturalMemoryExample]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind shared writes and counterfactual writes to their runtime tensors."""

    if set(examples_by_condition) != set(POSITIVE_CONDITIONS):
        raise ValueError("State-causality examples do not cover all positive conditions")
    if set(evaluations) != set(POSITIVE_CONDITIONS):
        raise ValueError("State-causality evaluations do not cover all positive conditions")

    examples = {
        condition: tuple(examples_by_condition[condition])
        for condition in POSITIVE_CONDITIONS
    }
    digests = {
        condition: _validated_state_digests(
            examples[condition], evaluations[condition], condition=condition
        )
        for condition in POSITIVE_CONDITIONS
    }
    correct_by_row = {example.row_id: example for example in examples["correct_state"]}
    if not correct_by_row:
        raise ValueError("State-causality audit received no correct-state rows")
    correct_rows = set(correct_by_row)
    for condition in POSITIVE_CONDITIONS:
        if any(example.condition != condition for example in examples[condition]):
            raise ValueError(f"State-causality audit received incorrect {condition} rows")
        if {example.row_id for example in examples[condition]} != correct_rows:
            raise ValueError(f"{condition} row pairing differs from correct_state")

    shared_families: dict[str, dict[str, Any]] = {}
    shared_family_total = 0
    identical_family_total = 0
    for condition in SHARED_WRITE_CONDITIONS:
        grouped: dict[str, list[NaturalMemoryExample]] = defaultdict(list)
        for example in examples[condition]:
            grouped[example.memory_state_id].append(example)
        condition_families: dict[str, Any] = {}
        for state_id, raw_family in sorted(grouped.items()):
            family = sorted(
                raw_family, key=lambda example: example.semantic_target_slot
            )
            if [example.semantic_target_slot for example in family] != list(
                range(RECORDS_PER_EPISODE)
            ):
                raise ValueError(f"{condition} state family {state_id} is incomplete")
            payloads = {example.record_payload_sha256 for example in family}
            if len(payloads) != 1:
                raise ValueError(
                    f"{condition} state family {state_id} does not share one write payload"
                )
            family_digests = [digests[condition][example.row_id] for example in family]
            identical = len(set(family_digests)) == 1
            shared_family_total += 1
            identical_family_total += int(identical)
            condition_families[state_id] = {
                "row_ids": [example.row_id for example in family],
                "record_payload_sha256": next(iter(payloads)),
                "runtime_tensor_state_digests": family_digests,
                "runtime_byte_identical": identical,
            }
        shared_families[condition] = condition_families

    counterfactual_by_condition: dict[str, Any] = {}
    counterfactual_pair_total = 0
    pair_contract_total = 0
    payload_difference_total = 0
    state_difference_total = 0
    for condition in COUNTERFACTUAL_STATE_CONDITIONS:
        condition_by_row = {
            example.row_id: example for example in examples[condition]
        }
        pairs: dict[str, Any] = {}
        for row_id in sorted(correct_rows):
            correct = correct_by_row[row_id]
            counterfactual = condition_by_row[row_id]
            pair_contract = (
                correct.row_id == counterfactual.row_id
                and correct.episode_id == counterfactual.episode_id
                and correct.task == counterfactual.task
                and correct.source_split == counterfactual.source_split
                and correct.semantic_target_slot
                == counterfactual.semantic_target_slot
                and correct.query_prefix_length
                == counterfactual.query_prefix_length
                and correct.read_input_ids[: correct.query_prefix_length]
                == counterfactual.read_input_ids[
                    : counterfactual.query_prefix_length
                ]
                and correct.query_mask[: correct.query_prefix_length]
                == counterfactual.query_mask[
                    : counterfactual.query_prefix_length
                ]
            )
            payload_differs = (
                correct.record_payload_sha256
                != counterfactual.record_payload_sha256
            )
            state_differs = (
                digests["correct_state"][row_id] != digests[condition][row_id]
            )
            counterfactual_pair_total += 1
            pair_contract_total += int(pair_contract)
            payload_difference_total += int(payload_differs)
            state_difference_total += int(state_differs)
            pairs[row_id] = {
                "episode_id": correct.episode_id,
                "semantic_target_slot": correct.semantic_target_slot,
                "pair_contract_passed": pair_contract,
                "correct_record_payload_sha256": correct.record_payload_sha256,
                "counterfactual_record_payload_sha256": (
                    counterfactual.record_payload_sha256
                ),
                "write_payload_differs": payload_differs,
                "correct_runtime_tensor_state_digest": (
                    digests["correct_state"][row_id]
                ),
                "counterfactual_runtime_tensor_state_digest": (
                    digests[condition][row_id]
                ),
                "runtime_tensor_state_differs": state_differs,
            }
        counterfactual_by_condition[condition] = {
            "rows": len(pairs),
            "write_payload_difference_fraction": (
                sum(pair["write_payload_differs"] for pair in pairs.values())
                / len(pairs)
            ),
            "pair_contract_passed_fraction": (
                sum(pair["pair_contract_passed"] for pair in pairs.values())
                / len(pairs)
            ),
            "runtime_tensor_state_difference_fraction": (
                sum(
                    pair["runtime_tensor_state_differs"]
                    for pair in pairs.values()
                )
                / len(pairs)
            ),
            "pairs_by_row": pairs,
        }

    return {
        "shared_write_conditions": list(SHARED_WRITE_CONDITIONS),
        "shared_state_families": shared_family_total,
        "runtime_byte_identical_families": identical_family_total,
        "runtime_byte_identical_state_fraction": (
            identical_family_total / shared_family_total
        ),
        "shared_families_by_condition": shared_families,
        "counterfactual_conditions": list(COUNTERFACTUAL_STATE_CONDITIONS),
        "counterfactual_pairs": counterfactual_pair_total,
        "counterfactual_pair_contract_passed_fraction": (
            pair_contract_total / counterfactual_pair_total
        ),
        "write_payload_difference_fraction": (
            payload_difference_total / counterfactual_pair_total
        ),
        "runtime_tensor_state_difference_fraction": (
            state_difference_total / counterfactual_pair_total
        ),
        "counterfactual_by_condition": counterfactual_by_condition,
    }


def audit_rewrite_output_change(
    correct_examples: Sequence[NaturalMemoryExample],
    rewrite_examples: Sequence[NaturalMemoryExample],
    correct_evaluation: Mapping[str, Any],
    rewrite_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    correct_by_row = {example.row_id: example for example in correct_examples}
    rewrite_by_row = {example.row_id: example for example in rewrite_examples}
    if (
        len(correct_by_row) != len(correct_examples)
        or set(correct_by_row) != set(rewrite_by_row)
    ):
        raise ValueError("Correct/rewrite row pairing differs")
    correct_predictions = _require_mapping(
        correct_evaluation.get("answer_predictions_by_row"),
        "correct answer predictions",
    )
    rewrite_predictions = _require_mapping(
        rewrite_evaluation.get("answer_predictions_by_row"),
        "rewrite answer predictions",
    )
    if set(correct_predictions) != set(correct_by_row) or set(rewrite_predictions) != set(
        correct_by_row
    ):
        raise ValueError("Correct/rewrite prediction rows differ")
    greedy = bool(correct_evaluation.get("greedy_answer_evaluated"))
    if bool(rewrite_evaluation.get("greedy_answer_evaluated")) != greedy:
        raise ValueError("Correct/rewrite greedy policy differs")

    pairs: dict[str, dict[str, Any]] = {}
    for row_id in sorted(correct_by_row):
        correct = correct_by_row[row_id]
        rewrite = rewrite_by_row[row_id]
        if correct.condition != "correct_state" or rewrite.condition != "target_slot_rewrite":
            raise ValueError("Rewrite audit received incorrect conditions")
        correct_prediction = _require_mapping(
            correct_predictions[row_id], f"correct prediction {row_id}"
        )
        rewrite_prediction = _require_mapping(
            rewrite_predictions[row_id], f"rewrite prediction {row_id}"
        )
        changed_record_indices = [
            index
            for index, (left, right) in enumerate(
                zip(correct.write_value_jsons, rewrite.write_value_jsons, strict=True)
            )
            if left != right
        ]
        target_index = correct.write_semantic_slots.index(
            correct.semantic_target_slot
        )
        pair_contract = (
            correct.semantic_target_slot == rewrite.semantic_target_slot
            and correct.write_record_ids == rewrite.write_record_ids
            and correct.write_semantic_slots == rewrite.write_semantic_slots
            and correct.write_slots == rewrite.write_slots
            and correct.target_slot == rewrite.target_slot
            and changed_record_indices == [target_index]
            and correct.read_input_ids[: correct.query_prefix_length]
            == rewrite.read_input_ids[: rewrite.query_prefix_length]
            and correct.query_mask[: correct.query_prefix_length]
            == rewrite.query_mask[: rewrite.query_prefix_length]
        )
        expected_answers_differ = (
            correct.expected_value != rewrite.expected_value
            and correct.expected_answer_token_ids != rewrite.expected_answer_token_ids
        )
        correct_teacher = tuple(
            int(token)
            for token in correct_prediction["teacher_forced_prediction_token_ids"]
        )
        rewrite_teacher = tuple(
            int(token)
            for token in rewrite_prediction["teacher_forced_prediction_token_ids"]
        )
        teacher_change = correct_teacher != rewrite_teacher
        teacher_joint_exact_change = (
            correct_prediction.get("teacher_forced_exact") is True
            and rewrite_prediction.get("teacher_forced_exact") is True
            and teacher_change
        )
        greedy_change = None
        greedy_joint_exact_change = None
        if greedy:
            correct_greedy = tuple(
                int(token)
                for token in correct_prediction["greedy_generated_token_ids"]
            )
            rewrite_greedy = tuple(
                int(token)
                for token in rewrite_prediction["greedy_generated_token_ids"]
            )
            greedy_change = correct_greedy != rewrite_greedy
            greedy_joint_exact_change = (
                correct_prediction.get("greedy_exact") is True
                and rewrite_prediction.get("greedy_exact") is True
                and greedy_change
            )
        pairs[row_id] = {
            "episode_id": correct.episode_id,
            "task": correct.task,
            "semantic_target_slot": correct.semantic_target_slot,
            "changed_write_record_indices": changed_record_indices,
            "target_write_record_only_changed": changed_record_indices == [target_index],
            "pair_contract_passed": pair_contract,
            "correct_expected_value_json": correct.expected_value,
            "rewrite_expected_value_json": rewrite.expected_value,
            "expected_answers_differ": expected_answers_differ,
            "teacher_forced_output_changed": teacher_change,
            "teacher_forced_joint_exact_output_flip": teacher_joint_exact_change,
            "greedy_output_changed": greedy_change,
            "greedy_joint_exact_output_flip": greedy_joint_exact_change,
        }

    def fraction(field: str) -> float:
        return sum(bool(pair[field]) for pair in pairs.values()) / len(pairs)

    per_task: dict[str, dict[str, Any]] = {}
    for task in sorted({pair["task"] for pair in pairs.values()}):
        task_pairs = [pair for pair in pairs.values() if pair["task"] == task]
        per_task[task] = {
            "rows": len(task_pairs),
            "pair_contract_passed_fraction": sum(
                bool(pair["pair_contract_passed"]) for pair in task_pairs
            )
            / len(task_pairs),
            "expected_answers_differ_fraction": sum(
                bool(pair["expected_answers_differ"]) for pair in task_pairs
            )
            / len(task_pairs),
            "teacher_forced_output_change_fraction": sum(
                bool(pair["teacher_forced_output_changed"]) for pair in task_pairs
            )
            / len(task_pairs),
            "teacher_forced_joint_exact_output_flip_fraction": sum(
                bool(pair["teacher_forced_joint_exact_output_flip"])
                for pair in task_pairs
            )
            / len(task_pairs),
            "greedy_output_change_fraction": (
                sum(bool(pair["greedy_output_changed"]) for pair in task_pairs)
                / len(task_pairs)
                if greedy
                else None
            ),
            "greedy_joint_exact_output_flip_fraction": (
                sum(
                    bool(pair["greedy_joint_exact_output_flip"])
                    for pair in task_pairs
                )
                / len(task_pairs)
                if greedy
                else None
            ),
        }
    return {
        "rows": len(pairs),
        "pair_contract_passed_fraction": fraction("pair_contract_passed"),
        "expected_answers_differ_fraction": fraction("expected_answers_differ"),
        "teacher_forced_output_change_fraction": fraction(
            "teacher_forced_output_changed"
        ),
        "teacher_forced_joint_exact_output_flip_fraction": fraction(
            "teacher_forced_joint_exact_output_flip"
        ),
        "greedy_answer_evaluated": greedy,
        "greedy_output_change_fraction": (
            fraction("greedy_output_changed") if greedy else None
        ),
        "greedy_joint_exact_output_flip_fraction": (
            fraction("greedy_joint_exact_output_flip") if greedy else None
        ),
        "per_task": per_task,
        "pairs_by_row": pairs,
    }


def audit_control_equivalence(
    no_state_evaluation: Mapping[str, Any],
    pristine_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    no_state = _require_mapping(
        no_state_evaluation.get("answer_predictions_by_row"),
        "no-state predictions",
    )
    pristine = _require_mapping(
        pristine_evaluation.get("answer_predictions_by_row"),
        "pristine predictions",
    )
    if set(no_state) != set(pristine) or not no_state:
        raise ValueError("No-state/pristine prediction rows differ")
    greedy = bool(no_state_evaluation.get("greedy_answer_evaluated"))
    if bool(pristine_evaluation.get("greedy_answer_evaluated")) != greedy:
        raise ValueError("No-state/pristine greedy policy differs")
    rows: dict[str, dict[str, Any]] = {}
    for row_id in sorted(no_state):
        left = _require_mapping(no_state[row_id], f"no-state row {row_id}")
        right = _require_mapping(pristine[row_id], f"pristine row {row_id}")
        teacher_equal = left.get("teacher_forced_prediction_token_ids") == right.get(
            "teacher_forced_prediction_token_ids"
        )
        greedy_equal = (
            left.get("greedy_generated_token_ids")
            == right.get("greedy_generated_token_ids")
            if greedy
            else None
        )
        rows[row_id] = {
            "teacher_forced_outputs_equal": teacher_equal,
            "greedy_outputs_equal": greedy_equal,
        }
    return {
        "rows": len(rows),
        "teacher_forced_output_equivalence_fraction": sum(
            row["teacher_forced_outputs_equal"] for row in rows.values()
        )
        / len(rows),
        "greedy_answer_evaluated": greedy,
        "greedy_output_equivalence_fraction": (
            sum(bool(row["greedy_outputs_equal"]) for row in rows.values())
            / len(rows)
            if greedy
            else None
        ),
        "rows_by_id": rows,
    }


def audit_trainable_parameters(
    model: torch.nn.Module,
    *,
    expected_trainable_names: Sequence[str] | None = None,
    allow_zero: bool = False,
) -> dict[str, Any]:
    actual = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    allowed: list[str] = []
    for module_name, module in iter_delta_mem_modules(model):
        predicate = getattr(module, "is_trainable_parameter", None)
        for sub_name, parameter in module.named_parameters():
            if sub_name.startswith("base."):
                continue
            if predicate is None or predicate(sub_name):
                allowed.append(f"{module_name}.{sub_name}")
    allowed = sorted(set(allowed))
    expected = (
        sorted(str(name) for name in expected_trainable_names)
        if expected_trainable_names is not None
        else allowed
    )
    only_delta_mem = set(actual).issubset(set(allowed))
    expected_binding = actual == expected
    passed = only_delta_mem and expected_binding and (allow_zero or bool(actual))
    return {
        "actual_trainable_names": actual,
        "allowed_delta_mem_trainable_names": allowed,
        "expected_trainable_names": expected,
        "only_delta_mem_parameters_trainable": only_delta_mem,
        "trainable_name_binding_passed": expected_binding,
        "nonempty_trainable_set": bool(actual),
        "allow_zero": allow_zero,
        "passed": passed,
    }


def _minimum_task_metric(
    evaluation: Mapping[str, Any],
    field: str,
) -> float | None:
    per_task = evaluation.get("per_task")
    if isinstance(per_task, Mapping) and per_task:
        values = [
            float(metrics[field])
            for metrics in per_task.values()
            if isinstance(metrics, Mapping) and metrics.get(field) is not None
        ]
        if values:
            return min(values)
    value = evaluation.get(field)
    return None if value is None else float(value)


def build_gate(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    state_identity: Mapping[str, Any],
    state_causality: Mapping[str, Any],
    rewrite_audit: Mapping[str, Any],
    control_equivalence: Mapping[str, Any],
    profile_eligibility: Mapping[str, Any],
    trainable_audit: Mapping[str, Any],
    immutability_passed: bool,
    training: Mapping[str, Any] | None = None,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """Return a deliberately conjunctive causal acceptance gate."""

    if set(evaluations) != set(CONDITIONS):
        raise ValueError("Gate evaluations do not cover all seven conditions")
    checks: dict[str, bool] = {}
    formal_profile = profile_eligibility.get("profile") in FORMAL_PROFILES

    for condition in POSITIVE_CONDITIONS:
        evaluation = evaluations[condition]
        answer_metric = _minimum_task_metric(
            evaluation, "teacher_forced_structured_json_exact_accuracy"
        )
        route_metric = _minimum_task_metric(evaluation, "semantic_route_accuracy")
        checks[f"{condition}.structured_json_exact_min"] = (
            answer_metric is not None and answer_metric >= thresholds.answer_exact_min
        )
        checks[f"{condition}.semantic_route_min"] = (
            route_metric is not None and route_metric >= thresholds.route_accuracy_min
        )
        checks[f"{condition}.full_occupancy"] = (
            evaluation.get("full_occupancy_fraction") == 1.0
        )
        checks[f"{condition}.forced_write_route"] = (
            evaluation.get("forced_write_route_accuracy") == 1.0
        )
        if evaluation.get("greedy_answer_evaluated"):
            greedy_metric = _minimum_task_metric(
                evaluation, "greedy_structured_json_exact_accuracy"
            )
            checks[f"{condition}.greedy_structured_json_exact_min"] = (
                greedy_metric is not None
                and greedy_metric >= thresholds.answer_exact_min
            )

    for condition in CONTROL_CONDITIONS:
        evaluation = evaluations[condition]
        checks[f"{condition}.zero_writes"] = (
            evaluation.get("full_occupancy_total") == 0
            and evaluation.get("forced_write_route_total") == 0
        )
        checks[f"{condition}.routes_absent"] = (
            evaluation.get("semantic_route_total") == 0
            and evaluation.get("route_absent_fraction") == 1.0
        )
        state_absence_by_row = evaluation.get("runtime_state_absence_by_row")
        checks[f"{condition}.runtime_state_absent"] = (
            evaluation.get("runtime_state_absence_rows")
            == evaluation.get("rows")
            and evaluation.get("runtime_state_absence_fraction") == 1.0
            and isinstance(state_absence_by_row, Mapping)
            and len(state_absence_by_row) == evaluation.get("rows")
            and not bool(evaluation.get("state_digest_by_row"))
        )
    checks["formal.greedy_answer_evaluation"] = (
        not formal_profile
        or all(
            evaluation.get("greedy_answer_evaluated") is True
            for evaluation in evaluations.values()
        )
    )
    checks["pristine_frozen_base.heads_disabled"] = (
        evaluations["pristine_frozen_base"].get("delta_heads_disabled") is True
    )
    checks["pristine_frozen_base.adapter_excluded"] = (
        evaluations["pristine_frozen_base"].get(
            "pristine_base_adapter_excluded"
        )
        is True
        and evaluations["pristine_frozen_base"].get("adapter_attached") is False
        and evaluations["pristine_frozen_base"].get(
            "attached_delta_mem_module_count"
        )
        == 0
        and evaluations["pristine_frozen_base"].get("trainable_parameter_count")
        == 0
    )

    checks["correct_state.runtime_byte_identical"] = (
        state_identity.get("runtime_byte_identical_state_fraction") == 1.0
    )
    checks["correct_state.all_four_routes"] = (
        state_identity.get("family_layer_all_four_correct_fraction") == 1.0
    )
    checks["positive_states.shared_write_identity"] = (
        state_causality.get("runtime_byte_identical_state_fraction") == 1.0
    )
    checks["counterfactual_states.write_payloads_differ"] = (
        state_causality.get("write_payload_difference_fraction") == 1.0
    )
    checks["counterfactual_states.pair_contract"] = (
        state_causality.get("counterfactual_pair_contract_passed_fraction")
        == 1.0
    )
    checks["counterfactual_states.runtime_tensors_differ"] = (
        state_causality.get("runtime_tensor_state_difference_fraction") == 1.0
    )
    checks["rewrite.expected_answers_differ"] = (
        rewrite_audit.get("expected_answers_differ_fraction") == 1.0
    )
    checks["rewrite.pair_contract"] = (
        rewrite_audit.get("pair_contract_passed_fraction") == 1.0
    )
    checks["rewrite.teacher_output_change"] = (
        rewrite_audit.get("teacher_forced_output_change_fraction", 0.0)
        >= thresholds.rewrite_output_change_min
    )
    checks["rewrite.teacher_joint_exact_flip"] = (
        rewrite_audit.get("teacher_forced_joint_exact_output_flip_fraction", 0.0)
        >= thresholds.rewrite_output_change_min
    )
    if rewrite_audit.get("greedy_answer_evaluated"):
        checks["rewrite.greedy_output_change"] = (
            rewrite_audit.get("greedy_output_change_fraction", 0.0)
            >= thresholds.rewrite_output_change_min
        )
        checks["rewrite.greedy_joint_exact_flip"] = (
            rewrite_audit.get("greedy_joint_exact_output_flip_fraction", 0.0)
            >= thresholds.rewrite_output_change_min
        )
    checks["no_state.pristine_equivalence"] = (
        control_equivalence.get("teacher_forced_output_equivalence_fraction") == 1.0
        and (
            not control_equivalence.get("greedy_answer_evaluated")
            or control_equivalence.get("greedy_output_equivalence_fraction") == 1.0
        )
    )
    checks["profile.eligibility"] = profile_eligibility.get("passed") is True
    checks["model.only_delta_mem_trainable"] = (
        trainable_audit.get("only_delta_mem_parameters_trainable") is True
        and trainable_audit.get("passed") is True
    )
    checks["source_and_model.immutable"] = bool(immutability_passed)
    if training is not None:
        training_dataset_audit = training.get("training_dataset_audit", {})
        try:
            audit_checks = (
                training_dataset_audit_checks(training_dataset_audit)
                if isinstance(training_dataset_audit, Mapping)
                else {}
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            audit_checks = {}
        checks["training.dataset_audit"] = bool(audit_checks) and all(
            audit_checks.values()
        )
        if formal_profile:
            checks["training.compositional_production_dataset"] = (
                isinstance(training_dataset_audit, Mapping)
                and validate_production_training_contract(
                    training_dataset_audit,
                    schedule_mode="complete",
                )
            )
    if training is not None and training.get("optimizer_skipped") is not True:
        checks["training.adapter_changed"] = training.get("adapter_changed") is True
        checks["training.router_gradient"] = training.get(
            "router_gradient_audit", {}
        ).get("all_modules_finite_nonzero") is True

    return {
        "schema": ACCEPTANCE_SCHEMA,
        "thresholds": {
            "answer_exact_min": thresholds.answer_exact_min,
            "route_accuracy_min": thresholds.route_accuracy_min,
            "rewrite_output_change_min": thresholds.rewrite_output_change_min,
        },
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def _json_payload_hash(path: Path) -> str:
    return source.sha256_text(
        source.canonical_json(_read_json_file(path, f"JSON artifact {path.name}"))
    )


def snapshot_directory_files(path: Path) -> dict[str, dict[str, Any]]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"Artifact directory must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Artifact directory is invalid: {resolved}")
    result: dict[str, dict[str, Any]] = {}
    for artifact in sorted(resolved.iterdir(), key=lambda value: value.name):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Artifact directory contains a non-file entry: {artifact}")
        result[artifact.name] = {
            "bytes": artifact.stat().st_size,
            "sha256": source.sha256_file(artifact),
        }
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source.sha256_file(path)


def validate_sealed_lock_chain(
    sealed_manifest: Mapping[str, Any],
    development_run_dir: Path,
    *,
    adapter_path: Path,
) -> dict[str, Any]:
    """Validate the sealed lock against the immutable development receipt."""

    sealed_lock = _require_mapping(sealed_manifest.get("sealed_lock"), "sealed_lock")
    lock = _require_mapping(sealed_lock.get("receipt"), "sealed lock receipt")
    requested_run_dir = development_run_dir.expanduser()
    if requested_run_dir.is_symlink():
        raise ValueError("Development run directory must not be a symbolic link")
    run_dir = requested_run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError(f"Development run directory is invalid: {run_dir}")
    protocol_path = run_dir / "protocol.json"
    training_path = run_dir / "training_configuration.json"
    evaluation_path = run_dir / "evaluation.json"
    receipt_path = run_dir / "run_receipt.json"
    for path in (protocol_path, training_path, evaluation_path, receipt_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Sealed lock chain artifact is missing: {path}")
    development_protocol = _read_json_file(protocol_path, "development protocol")
    development_training = _read_json_file(
        training_path, "development training configuration"
    )
    development_evaluation = _read_json_file(
        evaluation_path, "development evaluation"
    )
    if (
        development_protocol.get("schema") != PROTOCOL_SCHEMA
        or development_training.get("schema") != TRAINING_CONFIGURATION_SCHEMA
        or development_evaluation.get("schema") != EVALUATION_SCHEMA
    ):
        raise ValueError("Development run uses an incompatible protocol schema")
    if (
        development_protocol.get("runner_schema") != RUN_SCHEMA
        or development_protocol.get("profile") != "development"
        or development_training.get("profile") != "development"
        or development_evaluation.get("profile") != "development"
    ):
        raise ValueError("Development artifacts do not share the development profile")

    protocol_hash = _sha256_json(development_protocol)
    training_hash = _sha256_json(development_training)
    evaluation_hash = _sha256_json(development_evaluation)
    protocol_audit = _require_mapping(
        development_protocol.get("training_dataset_audit"),
        "development protocol training dataset audit",
    )
    training_dataset_audit = _require_mapping(
        development_training.get("training_dataset_audit"),
        "development training configuration dataset audit",
    )
    evaluation_training = _require_mapping(
        development_evaluation.get("training"),
        "development evaluation training evidence",
    )
    evaluation_audit = _require_mapping(
        evaluation_training.get("training_dataset_audit"),
        "development evaluation training dataset audit",
    )
    if not (
        protocol_audit == training_dataset_audit == evaluation_audit
    ):
        raise ValueError(
            "Development artifacts contain different training dataset audits"
        )
    if not validate_retained_production_training_contract(
        training_dataset_audit,
        schedule_mode="complete",
    ):
        raise ValueError("Development training dataset audit is not production-valid")
    training_dataset_audit_hash = _sha256_json(training_dataset_audit)

    receipt = _read_json_file(receipt_path, "development run receipt")
    if receipt.get("schema") != RUN_SCHEMA:
        raise ValueError("Development run receipt schema is incompatible")
    unsigned_receipt = dict(receipt)
    recorded_receipt_hash = unsigned_receipt.pop("run_receipt_sha256", None)
    if recorded_receipt_hash != _signed_payload(
        unsigned_receipt, "run_receipt_sha256"
    )["run_receipt_sha256"]:
        raise ValueError("Development run receipt signature is invalid")
    if receipt.get("profile") != "development":
        raise ValueError("Sealed lock chain does not point to a development run")

    evaluation_gate = _require_mapping(
        development_evaluation.get("gate"), "development evaluation gate"
    )
    receipt_gate = _require_mapping(receipt.get("gate"), "development receipt gate")
    if evaluation_gate != receipt_gate:
        raise ValueError("Development receipt gate differs from the evaluation gate")
    gate_checks = _require_mapping(
        evaluation_gate.get("checks"), "development evaluation gate checks"
    )
    genuine_gate_pass = (
        evaluation_gate.get("schema") == ACCEPTANCE_SCHEMA
        and evaluation_gate.get("passed") is True
        and evaluation_gate.get("failed_checks") == []
        and bool(gate_checks)
        and all(value is True for value in gate_checks.values())
        and receipt.get("gate_passed") is True
    )
    if not genuine_gate_pass:
        raise ValueError("Development evaluation does not prove a passing gate")

    manifest_payload_hash = receipt.get("source_manifest_payload_sha256")
    if not _is_sha256(manifest_payload_hash):
        raise ValueError("Development receipt has an invalid manifest payload digest")
    expected_benchmark_hash = sealed_manifest.get("benchmark_contract_sha256")
    required_lock_digests = (
        "benchmark_contract_sha256",
        "development_manifest_payload_sha256",
        "runner_protocol_sha256",
        "training_configuration_sha256",
        "training_dataset_audit_sha256",
        "evaluation_sha256",
        "development_run_receipt_sha256",
        "adapter_files_sha256",
    )
    if (
        lock.get("schema") != source.SEALED_LOCK_SCHEMA
        or lock.get("configuration_frozen") is not True
        or lock.get("development_gate_passed") is not True
        or not _is_sha256(expected_benchmark_hash)
        or lock.get("benchmark_contract_sha256") != expected_benchmark_hash
        or any(not _is_sha256(lock.get(name)) for name in required_lock_digests)
        or sealed_lock.get("receipt_sha256") != _sha256_json(lock)
    ):
        raise ValueError("Sealed lock receipt is invalid or not frozen")
    required = {
        "development_manifest_payload_sha256": manifest_payload_hash,
        "runner_protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_hash,
        "training_dataset_audit_sha256": training_dataset_audit_hash,
        "evaluation_sha256": evaluation_hash,
    }
    for name, value in required.items():
        if lock.get(name) != value:
            raise ValueError(f"Sealed lock chain mismatch: {name}")
    if (
        receipt.get("protocol_sha256") != protocol_hash
        or receipt.get("training_configuration_sha256") != training_hash
        or receipt.get("training_dataset_audit_sha256")
        != training_dataset_audit_hash
        or receipt.get("evaluation_sha256") != evaluation_hash
    ):
        raise ValueError("Development receipt does not bind its protocol files")
    if lock.get("development_run_receipt_sha256") != recorded_receipt_hash:
        raise ValueError("Sealed lock does not bind the signed development receipt")
    requested_adapter = adapter_path.expanduser()
    if requested_adapter.is_symlink():
        raise ValueError("Sealed adapter path must not be a symbolic link")
    adapter = requested_adapter.resolve(strict=True)
    if not adapter.is_dir():
        raise ValueError(f"Sealed adapter path is invalid: {adapter}")
    adapter_files = snapshot_directory_files(adapter)
    if not adapter_files:
        raise ValueError("Sealed adapter directory is empty")
    if receipt.get("adapter_files") != adapter_files:
        raise ValueError("Sealed adapter artifacts differ from the development receipt")
    adapter_files_sha256 = _sha256_json(adapter_files)
    if (
        receipt.get("adapter_files_sha256") != adapter_files_sha256
        or lock.get("adapter_files_sha256") != adapter_files_sha256
    ):
        raise ValueError("Sealed lock does not bind the exact adapter artifacts")
    return {
        "development_run_dir": str(run_dir),
        "protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_hash,
        "training_dataset_audit_sha256": training_dataset_audit_hash,
        "evaluation_sha256": evaluation_hash,
        "development_manifest_payload_sha256": manifest_payload_hash,
        "adapter_path": str(adapter),
        "adapter_files": adapter_files,
        "adapter_files_sha256": adapter_files_sha256,
        "development_run_receipt_sha256": recorded_receipt_hash,
        "development_protocol": development_protocol,
        "development_training_configuration": development_training,
        "development_evaluation": development_evaluation,
        "passed": True,
    }


def _parse_layers(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        layers = tuple(int(layer) for layer in value)
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("Target layers must be a nonempty unique list of nonnegative integers")
    return layers


def _parse_training_conditions(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        conditions = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        conditions = tuple(str(condition) for condition in value)
    if not conditions:
        raise ValueError("Training conditions must be a nonempty list")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Training conditions must be unique")
    invalid = sorted(set(conditions) - set(POSITIVE_CONDITIONS))
    if invalid:
        raise ValueError(
            "Training conditions must be positive memory conditions: "
            + ", ".join(invalid)
        )
    return conditions


def run_experiment(
    *,
    source_manifest: Path,
    output_dir: Path,
    profile: str = "development",
    model_path: Path | None = None,
    adapter_path: Path | None = None,
    development_run_dir: Path | None = None,
    replication_protocol: Path | None = None,
    replication_amendment: Path | None = None,
    replication_id: str | None = None,
    seed: int = PRODUCTION_SEED,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    epochs: int = PRODUCTION_EPOCHS,
    max_steps: int | None = PRODUCTION_UPDATES,
    batch_size: int = distributed.REQUIRED_GLOBAL_BATCH_SIZE,
    eval_batch_size: int = PRODUCTION_EVAL_BATCH_SIZE,
    learning_rate: float = PRODUCTION_LEARNING_RATE,
    answer_weight: float = PRODUCTION_ANSWER_WEIGHT,
    route_weight: float = PRODUCTION_ROUTE_WEIGHT,
    hard_negative_margin: float = PRODUCTION_HARD_NEGATIVE_MARGIN,
    hard_negative_weight: float = PRODUCTION_HARD_NEGATIVE_WEIGHT,
    max_grad_norm: float = PRODUCTION_MAX_GRAD_NORM,
    device_name: str = "cuda",
    dtype_name: str = PRODUCTION_DTYPE,
    attn_implementation: str = PRODUCTION_ATTN_IMPLEMENTATION,
    target_layers: Sequence[int] = DEFAULT_TARGET_LAYERS,
    training_conditions: Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
    rank: int = PRODUCTION_ADAPTER_RANK,
    key_dim: int = PRODUCTION_KEY_DIM,
    temperature: float = PRODUCTION_TEMPERATURE,
    greedy: bool = True,
    answer_exact_min: float = PRODUCTION_ANSWER_EXACT_MIN,
    route_accuracy_min: float = PRODUCTION_ROUTE_ACCURACY_MIN,
    rewrite_output_change_min: float = PRODUCTION_REWRITE_OUTPUT_CHANGE_MIN,
    distributed_context: distributed.DistributedTrainingContext | None = None,
    capture_distributed_step_evidence: bool = False,
    distributed_preflight: bool = False,
) -> dict[str, Any]:
    """Run a train/development screen or a sealed, optimizer-free evaluation."""

    configure_hf_mirror()
    replication_authorization = validate_replication_authorization(
        source_manifest=source_manifest,
        profile=profile,
        seed=seed,
        replication_protocol=replication_protocol,
        replication_amendment=replication_amendment,
        replication_id=replication_id,
    )
    if profile == "sealed_validation" and adapter_path is None:
        raise ValueError("Sealed validation requires a frozen development adapter")
    if profile != "sealed_validation" and adapter_path is not None:
        raise ValueError("Training profiles cannot inject a pre-trained adapter")
    if profile == "sealed_validation" and development_run_dir is None:
        raise ValueError("Sealed validation requires the development run receipt directory")
    if profile in FORMAL_PROFILES and not greedy:
        raise ValueError(
            "Formal development and sealed-validation runs require greedy evaluation"
        )
    if profile in {"development", "sealed_validation"} and (
        train_limit is not None or eval_limit is not None
    ):
        raise ValueError(
            "Formal development and sealed-validation runs require complete splits"
        )
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied")
    if epochs <= 0 or batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("Training and evaluation sizes must be positive")
    if hard_negative_margin < 0.0 or hard_negative_weight < 0.0:
        raise ValueError("Hard-negative margin and weight must be nonnegative")
    if eval_batch_size < RECORDS_PER_EPISODE:
        raise ValueError(
            "Evaluation batch size must hold a complete four-query state family"
        )
    thresholds = GateThresholds(
        answer_exact_min=answer_exact_min,
        route_accuracy_min=route_accuracy_min,
        rewrite_output_change_min=rewrite_output_change_min,
    )
    if not all(
        0.0 <= value <= 1.0
        for value in (
            answer_exact_min,
            route_accuracy_min,
            rewrite_output_change_min,
        )
    ):
        raise ValueError("Gate thresholds must be fractions in [0, 1]")
    if profile == "development":
        if distributed_context is None:
            raise ValueError(
                "Formal development training requires four-rank torchrun"
            )
        if (
            distributed_context.world_size != distributed.REQUIRED_WORLD_SIZE
            or batch_size != distributed.REQUIRED_GLOBAL_BATCH_SIZE
            or batch_size // distributed_context.world_size
            != distributed.REQUIRED_LOCAL_BATCH_SIZE
            or distributed.REQUIRED_LOCAL_MICROBATCH_SIZE != 2
            or distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS != 2
        ):
            raise ValueError(
                "Formal development requires world size 4, local optimizer batch 4, "
                "local microbatch 2, accumulation 2, and global batch 16"
            )
    if profile == "sealed_validation" and distributed_context is not None:
        raise ValueError("Sealed validation is a single-process evaluation")
    if capture_distributed_step_evidence and distributed_context is None:
        raise ValueError("Distributed step evidence requires torchrun")
    if distributed_preflight:
        if profile != "development" or distributed_context is None:
            raise ValueError(
                "Distributed preflight requires four-rank development torchrun"
            )
        if max_steps != DISTRIBUTED_PREFLIGHT_STEPS:
            raise ValueError(
                f"Distributed preflight requires exactly {DISTRIBUTED_PREFLIGHT_STEPS} updates"
            )
        capture_distributed_step_evidence = True
    elif profile == "development" and max_steps != PRODUCTION_UPDATES:
        raise ValueError(
            f"Formal development requires exactly {PRODUCTION_UPDATES} updates"
        )
    if profile == "development":
        production_configuration = {
            "seed": seed
            == (
                replication_authorization.get("training_seed")
                if replication_authorization is not None
                else PRODUCTION_SEED
            ),
            "epochs": epochs == PRODUCTION_EPOCHS,
            "eval_batch_size": eval_batch_size == PRODUCTION_EVAL_BATCH_SIZE,
            "learning_rate": learning_rate == PRODUCTION_LEARNING_RATE,
            "answer_weight": answer_weight == PRODUCTION_ANSWER_WEIGHT,
            "route_weight": route_weight == PRODUCTION_ROUTE_WEIGHT,
            "hard_negative_margin": (
                hard_negative_margin == PRODUCTION_HARD_NEGATIVE_MARGIN
            ),
            "hard_negative_weight": (
                hard_negative_weight == PRODUCTION_HARD_NEGATIVE_WEIGHT
            ),
            "max_grad_norm": max_grad_norm == PRODUCTION_MAX_GRAD_NORM,
            "dtype": dtype_name == PRODUCTION_DTYPE,
            "attn_implementation": (
                attn_implementation == PRODUCTION_ATTN_IMPLEMENTATION
            ),
            "target_layers": tuple(target_layers) == DEFAULT_TARGET_LAYERS,
            "training_conditions": (
                tuple(training_conditions) == DEFAULT_TRAINING_CONDITIONS
            ),
            "adapter_rank": rank == PRODUCTION_ADAPTER_RANK,
            "key_dim": key_dim == PRODUCTION_KEY_DIM,
            "temperature": temperature == PRODUCTION_TEMPERATURE,
            "answer_exact_min": answer_exact_min == PRODUCTION_ANSWER_EXACT_MIN,
            "route_accuracy_min": (
                route_accuracy_min == PRODUCTION_ROUTE_ACCURACY_MIN
            ),
            "rewrite_output_change_min": (
                rewrite_output_change_min
                == PRODUCTION_REWRITE_OUTPUT_CHANGE_MIN
            ),
        }
        mismatches = [
            name for name, matches in production_configuration.items() if not matches
        ]
        if mismatches:
            raise ValueError(
                "Formal development configuration differs from the profiled "
                "production contract: " + ", ".join(mismatches)
            )

    def prepare_run_inputs() -> tuple[
        Path,
        ProfileBundle,
        Path,
        tuple[Path, ...],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any] | None,
        torch.device,
        torch.dtype,
        tuple[int, ...],
        tuple[str, ...],
        Any,
    ]:
        requested_output = output_dir.expanduser()
        if requested_output.is_symlink():
            raise ValueError(
                f"Natural run output must not be a symbolic link: {requested_output}"
            )
        prepared_output = requested_output.resolve()
        if prepared_output.exists():
            raise ValueError(f"Natural run output must be fresh: {prepared_output}")
        prepared_bundle = load_profile_bundle(source_manifest, profile=profile)
        prepared_model_root, prepared_model_artifacts = resolve_model_artifacts(
            prepared_bundle.model_binding,
            model_path=model_path,
        )
        prepared_source_before = snapshot_files(prepared_bundle.source_paths)
        prepared_model_before = snapshot_files(prepared_model_artifacts)
        prepared_sealed_chain: Mapping[str, Any] | None = None
        if profile == "sealed_validation":
            prepared_sealed_chain = validate_sealed_lock_chain(
                prepared_bundle.sealed_manifest
                or prepared_bundle.development_manifest,
                development_run_dir or Path(),
                adapter_path=adapter_path or Path(),
            )
        prepared_device = (
            distributed_context.device
            if distributed_context is not None
            else torch.device(device_name)
        )
        if prepared_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if prepared_device.type == "cuda":
            torch.cuda.set_device(prepared_device)
            torch.backends.cuda.matmul.allow_tf32 = True
        prepared_dtype = _dtype(dtype_name)
        prepared_layers = _parse_layers(target_layers)
        prepared_training_conditions = _parse_training_conditions(
            training_conditions
        )
        prepared_delta_config = build_delta_config(
            target_layers=prepared_layers,
            rank=rank,
            key_dim=key_dim,
            temperature=temperature,
        )
        runtime.set_seed(seed)
        return (
            prepared_output,
            prepared_bundle,
            prepared_model_root,
            prepared_model_artifacts,
            prepared_source_before,
            prepared_model_before,
            prepared_sealed_chain,
            prepared_device,
            prepared_dtype,
            prepared_layers,
            prepared_training_conditions,
            prepared_delta_config,
        )

    if distributed_context is None:
        prepared_run_inputs = prepare_run_inputs()
    else:
        prepared_run_inputs = _run_consensused_local_phase(
            distributed_context,
            phase="run-input-preparation",
            operation=prepare_run_inputs,
        )
    (
        resolved_output,
        bundle,
        model_root,
        model_artifact_paths,
        source_before,
        model_before,
        sealed_chain,
        device,
        dtype,
        layers,
        selected_training_conditions,
        delta_config,
    ) = prepared_run_inputs
    if distributed_context is not None:
        run_input_binding_sha256 = _run_consensused_local_phase(
            distributed_context,
            phase="run-input-binding-preparation",
            operation=lambda: distributed.canonical_sha256(
                {
                    "resolved_output": str(resolved_output),
                    "model_root": str(model_root),
                    "model_artifact_paths": [
                        str(path) for path in model_artifact_paths
                    ],
                    "source_files": source_before,
                    "model_files": model_before,
                    "model_binding_sha256": bundle.model_binding.get(
                        "binding_sha256"
                    ),
                    "profile_eligibility": bundle.eligibility,
                    "delta_mem_config": delta_config.to_dict(),
                }
            ),
        )
        distributed.require_consensus(
            distributed_context,
            run_input_binding_sha256,
            description="run input binding",
        )
    model_source = {"model": {"path": str(model_root)}}
    load_error: BaseException | None = None
    try:
        model, tokenizer, replaced_layers, trainable_names, checkpointed_mlps = (
            _load_model_and_tokenizer(
                model_source,
                device=device,
                dtype=dtype,
                attn_implementation=attn_implementation,
                delta_config=delta_config,
            )
        )
    except BaseException as error:
        load_error = error
    if distributed_context is not None:
        distributed.phase_consensus(
            distributed_context,
            phase="model-and-adapter-load",
            error=load_error,
        )
    if load_error is not None:
        raise load_error
    def prepare_trainable_audit() -> Mapping[str, Any]:
        if profile == "sealed_validation":
            loaded_delta_config = load_delta_mem_adapter(model, adapter_path or Path())
            if loaded_delta_config.to_dict() != delta_config.to_dict():
                raise ValueError(
                    "Sealed adapter configuration differs from the frozen requested "
                    "configuration"
                )
            for parameter in model.parameters():
                parameter.requires_grad = False
            prepared_audit = audit_trainable_parameters(
                model,
                expected_trainable_names=(),
                allow_zero=True,
            )
        else:
            prepared_audit = audit_trainable_parameters(
                model,
                expected_trainable_names=trainable_names,
            )
        if not prepared_audit["passed"]:
            raise ValueError("Only Delta-Mem parameters may be trainable")
        return prepared_audit

    if distributed_context is None:
        trainable_audit = prepare_trainable_audit()
    else:
        trainable_audit = _run_consensused_local_phase(
            distributed_context,
            phase="initial-trainable-audit",
            operation=prepare_trainable_audit,
        )

    distributed_initialization: Mapping[str, Any] | None = None
    if distributed_context is not None:
        def prepare_adapter_initialization() -> tuple[
            tuple[tuple[str, torch.nn.Parameter], ...],
            tuple[dict[str, Any], ...],
            str,
            str,
        ]:
            prepared_parameters = _named_adapter_parameters(model)
            prepared_metadata = distributed.named_tensor_metadata(prepared_parameters)
            prepared_metadata_hash = distributed.canonical_sha256(prepared_metadata)
            prepared_state_hash = _state_dict_sha256(
                snapshot_delta_mem_weights(model)
            )
            return (
                prepared_parameters,
                prepared_metadata,
                prepared_metadata_hash,
                prepared_state_hash,
            )

        (
            adapter_parameters,
            adapter_metadata,
            adapter_metadata_sha256,
            adapter_hash_before_broadcast,
        ) = _run_consensused_local_phase(
            distributed_context,
            phase="adapter-initialization-preparation",
            operation=prepare_adapter_initialization,
        )
        distributed.require_consensus(
            distributed_context,
            adapter_metadata_sha256,
            description="complete adapter parameter metadata",
        )
        hashes_before_broadcast = distributed.gather_objects(
            distributed_context,
            adapter_hash_before_broadcast,
        )
        broadcast_error: BaseException | None = None
        try:
            broadcast_evidence = distributed.broadcast_named_parameters(
                distributed_context, adapter_parameters
            )
        except BaseException as error:
            broadcast_error = error
        distributed.phase_consensus(
            distributed_context,
            phase="complete-adapter-initialization-broadcast",
            error=broadcast_error,
        )
        if broadcast_error is not None:
            raise broadcast_error
        synchronized_adapter_hash = _run_consensused_local_phase(
            distributed_context,
            phase="broadcast-adapter-hash-preparation",
            operation=lambda: _state_dict_sha256(
                snapshot_delta_mem_weights(model)
            ),
        )
        hashes_after_broadcast = distributed.require_consensus(
            distributed_context,
            synchronized_adapter_hash,
            description="broadcast adapter initialization",
        )
        distributed_initialization = {
            "source_rank": 0,
            "complete_adapter_metadata_sha256": adapter_metadata_sha256,
            "complete_adapter_names_sha256": distributed.canonical_sha256(
                [value["name"] for value in adapter_metadata]
            ),
            "hashes_before_broadcast": list(hashes_before_broadcast),
            "hashes_after_broadcast": list(hashes_after_broadcast),
            "synchronized_adapter_state_sha256": hashes_after_broadcast[0],
            "broadcast": dict(broadcast_evidence),
        }

    def prepare_episode_selection() -> tuple[str, list[NaturalEpisode], list[NaturalEpisode]]:
        prepared_adapter_hash = _state_dict_sha256(
            snapshot_delta_mem_weights(model)
        )
        prepared_train_episodes = select_complete_episodes(
            bundle.train_episodes, train_limit
        )
        prepared_eval_episodes = select_complete_episodes(
            bundle.evaluation_episodes, eval_limit
        )
        if profile != "sealed_validation" and not prepared_train_episodes:
            raise ValueError("Training profile selected no training episodes")
        if not prepared_eval_episodes:
            raise ValueError("Selected profile has no evaluation episodes")
        return (
            prepared_adapter_hash,
            prepared_train_episodes,
            prepared_eval_episodes,
        )

    if distributed_context is None:
        pre_adapter_hash, train_episodes, eval_episodes = prepare_episode_selection()
    else:
        (
            pre_adapter_hash,
            train_episodes,
            eval_episodes,
        ) = _run_consensused_local_phase(
            distributed_context,
            phase="episode-selection-preparation",
            operation=prepare_episode_selection,
        )
        distributed.require_consensus(
            distributed_context,
            pre_adapter_hash,
            description="pre-training adapter state",
        )

    if distributed_context is None:
        resolved_output.mkdir(parents=True, exist_ok=False)
    else:
        output_error: BaseException | None = None
        if distributed_context.is_primary:
            try:
                resolved_output.mkdir(parents=True, exist_ok=False)
            except BaseException as error:
                output_error = error
        distributed.phase_consensus(
            distributed_context,
            phase="rank-zero-output-creation",
            error=output_error,
        )
        if output_error is not None:
            raise output_error
    progress_path = resolved_output / "training_progress.jsonl"
    if profile == "sealed_validation":
        development_training_configuration = _require_mapping(
            (sealed_chain or {}).get("development_training_configuration"),
            "locked development training configuration",
        )
        training_dataset_audit = dict(
            _require_mapping(
                development_training_configuration.get("training_dataset_audit"),
                "locked training dataset audit",
            )
        )
        training: dict[str, Any] = {
            "optimizer_skipped": True,
            "steps": 0,
            "adapter_changed": None,
            "router_gradient_audit": {"all_modules_finite_nonzero": True},
            "training_dataset_audit": training_dataset_audit,
        }
        adapter_files = dict((sealed_chain or {})["adapter_files"])
    else:
        if distributed_context is None:
            training_examples = build_training_examples(
                train_episodes,
                tokenizer,
                selected_training_conditions,
            )
        else:
            training_examples = _run_consensused_local_phase(
                distributed_context,
                phase="training-example-construction",
                operation=lambda: build_training_examples(
                    train_episodes,
                    tokenizer,
                    selected_training_conditions,
                ),
            )
            ordered_training_examples_sha256 = _run_consensused_local_phase(
                distributed_context,
                phase="ordered-training-example-hash-preparation",
                operation=lambda: distributed.canonical_sha256(
                    [example.row_id for example in training_examples]
                ),
            )
            distributed.require_consensus(
                distributed_context,
                ordered_training_examples_sha256,
                description="ordered training examples",
            )
        training_dataset_audit = audit_training_dataset(
            training_examples,
            selected_training_conditions,
        )
        if profile == "development":
            training_dataset_audit = bind_production_training_contract(
                training_dataset_audit,
                epochs=epochs,
                global_batch_size=batch_size,
                requested_max_steps=max_steps,
                schedule_mode=("preflight" if distributed_preflight else "complete"),
            )
            if not validate_production_training_contract(
                training_dataset_audit,
                schedule_mode=("preflight" if distributed_preflight else "complete"),
            ):
                failed_dataset_checks = sorted(
                    name for name, passed in training_dataset_audit.get(
                        "production_dataset_contract_checks", {}
                    ).items()
                    if not passed
                )
                raise ValueError(
                    "Formal development training dataset differs from the "
                    "compositional production contract: "
                    + ", ".join(failed_dataset_checks)
                )
        else:
            basic_checks = training_dataset_audit_checks(training_dataset_audit)
            if not all(basic_checks.values()):
                raise ValueError(
                    "Training dataset audit is inconsistent: "
                    + ", ".join(
                        name for name, passed in basic_checks.items() if not passed
                    )
                )
            complete_epoch_updates = (
                len(training_examples) * epochs // batch_size
                if len(training_examples) % batch_size == 0
                else None
            )
            training_dataset_audit["schedule_contract"] = {
                "epochs": epochs,
                "global_batch_size": batch_size,
                "rows_divide_global_batch": complete_epoch_updates is not None,
                "complete_epoch_updates": complete_epoch_updates,
                "requested_max_steps": max_steps,
                "complete_epoch_schedule_requested": (
                    max_steps is None or max_steps == complete_epoch_updates
                ),
            }
        if distributed_context is not None:
            distributed.require_consensus(
                distributed_context,
                distributed.canonical_sha256(training_dataset_audit),
                description="training dataset audit",
            )
        if distributed_context is None:
            training = dict(
                train_model(
                    model,
                    training_examples,
                    seed=seed,
                    epochs=epochs,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    answer_weight=answer_weight,
                    route_weight=route_weight,
                    hard_negative_margin=hard_negative_margin,
                    hard_negative_weight=hard_negative_weight,
                    max_grad_norm=max_grad_norm,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=device,
                    dtype=dtype,
                    progress_path=progress_path,
                    training_conditions=selected_training_conditions,
                )
            )
        else:
            def run_distributed_training() -> dict[str, Any]:
                prepared_training = dict(
                    train_model_distributed(
                        model,
                        training_examples,
                        context=distributed_context,
                        seed=seed,
                        epochs=epochs,
                        max_steps=max_steps,
                        global_batch_size=batch_size,
                        learning_rate=learning_rate,
                        answer_weight=answer_weight,
                        route_weight=route_weight,
                        hard_negative_margin=hard_negative_margin,
                        hard_negative_weight=hard_negative_weight,
                        max_grad_norm=max_grad_norm,
                        pad_token_id=int(tokenizer.pad_token_id),
                        dtype=dtype,
                        progress_path=progress_path,
                        training_conditions=selected_training_conditions,
                        capture_step_evidence=capture_distributed_step_evidence,
                    )
                )
                prepared_training["distributed"]["initialization"] = dict(
                    distributed_initialization or {}
                )
                return prepared_training

            training = _run_consensused_local_phase(
                distributed_context,
                phase="distributed-training-completion",
                operation=run_distributed_training,
            )
        training["training_dataset_audit"] = training_dataset_audit

        def prepare_post_training_evidence() -> tuple[
            Mapping[str, Any], str, bool, Mapping[str, Any], Mapping[str, Any]
        ]:
            prepared_audit = audit_trainable_parameters(
                model,
                expected_trainable_names=trainable_names,
            )
            if not prepared_audit["passed"]:
                raise ValueError("Training changed the trainable-parameter boundary")
            prepared_adapter_hash = _state_dict_sha256(
                snapshot_delta_mem_weights(model)
            )
            prepared_adapter_changed = prepared_adapter_hash != pre_adapter_hash
            if not prepared_adapter_changed:
                raise RuntimeError("Training produced no Delta-Mem parameter update")
            prepared_source_after = assert_snapshot_unchanged(
                source_before,
                description="Natural distributed training source"
                if distributed_context is not None
                else "Natural training source",
            )
            prepared_model_after = assert_snapshot_unchanged(
                model_before,
                description="Natural distributed training model"
                if distributed_context is not None
                else "Natural training model",
            )
            return (
                prepared_audit,
                prepared_adapter_hash,
                prepared_adapter_changed,
                prepared_source_after,
                prepared_model_after,
            )

        if distributed_context is None:
            (
                post_train_audit,
                post_adapter_hash,
                adapter_changed,
                rank_source_after,
                rank_model_after,
            ) = prepare_post_training_evidence()
        else:
            (
                post_train_audit,
                post_adapter_hash,
                adapter_changed,
                rank_source_after,
                rank_model_after,
            ) = _run_consensused_local_phase(
                distributed_context,
                phase="post-training-evidence-preparation",
                operation=prepare_post_training_evidence,
            )
        training["adapter_changed"] = adapter_changed
        training["adapter_state_sha256_before"] = pre_adapter_hash
        training["adapter_state_sha256_after"] = post_adapter_hash

        if distributed_context is None:
            save_delta_mem_adapter(model, resolved_output / "adapter", delta_config)
            adapter_files = snapshot_directory_files(resolved_output / "adapter")
        else:
            post_training_binding = _run_consensused_local_phase(
                distributed_context,
                phase="post-training-binding-preparation",
                operation=lambda: {
                    "post_train_audit_sha256": distributed.canonical_sha256(
                        post_train_audit
                    ),
                    "adapter_state_sha256_before": pre_adapter_hash,
                    "adapter_state_sha256_after": post_adapter_hash,
                    "adapter_changed": adapter_changed,
                },
            )
            distributed.require_consensus(
                distributed_context,
                post_training_binding,
                description="post-training adapter and trainable audit",
            )
            local_immutability_evidence = _run_consensused_local_phase(
                distributed_context,
                phase="rank-input-immutability-evidence-preparation",
                operation=lambda: {
                    "rank": distributed_context.process_rank,
                    "source_snapshot_sha256": distributed.canonical_sha256(
                        rank_source_after
                    ),
                    "model_snapshot_sha256": distributed.canonical_sha256(
                        rank_model_after
                    ),
                },
            )
            rank_immutability = distributed.gather_objects(
                distributed_context,
                local_immutability_evidence,
            )

            def validate_rank_immutability() -> list[Mapping[str, Any]]:
                if len(rank_immutability) != distributed_context.world_size or any(
                    not isinstance(value, Mapping) for value in rank_immutability
                ):
                    raise distributed.DistributedTrainingError(
                        "Input immutability evidence is malformed"
                    )
                ordered = sorted(
                    rank_immutability, key=lambda value: value.get("rank", -1)
                )
                if [value.get("rank") for value in ordered] != list(
                    range(distributed_context.world_size)
                ):
                    raise distributed.DistributedTrainingError(
                        "Input immutability rank identities differ"
                    )
                if len(
                    {value.get("source_snapshot_sha256") for value in ordered}
                ) != 1 or len(
                    {value.get("model_snapshot_sha256") for value in ordered}
                ) != 1:
                    raise distributed.DistributedTrainingError(
                        "Input immutability snapshots differ across ranks"
                    )
                return list(ordered)

            training["distributed"]["rank_input_immutability"] = (
                _run_consensused_local_phase(
                    distributed_context,
                    phase="rank-input-immutability-validation",
                    operation=validate_rank_immutability,
                )
            )
            if distributed_preflight:
                preflight_error: BaseException | None = None
                preflight_receipt: Mapping[str, Any] | None = None
                if distributed_context.is_primary:
                    try:
                        preflight_configuration = {
                            "profile": profile,
                            "seed": seed,
                            "epochs": epochs,
                            "optimizer_updates": max_steps,
                            "world_size": distributed_context.world_size,
                            "local_batch_size": (
                                batch_size // distributed_context.world_size
                            ),
                            "local_microbatch_size": (
                                distributed.REQUIRED_LOCAL_MICROBATCH_SIZE
                            ),
                            "gradient_accumulation_steps": (
                                distributed.REQUIRED_GRADIENT_ACCUMULATION_STEPS
                            ),
                            "global_batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "answer_weight": answer_weight,
                            "route_weight": route_weight,
                            "hard_negative_margin": hard_negative_margin,
                            "hard_negative_weight": hard_negative_weight,
                            "max_grad_norm": max_grad_norm,
                            "training_conditions": list(
                                selected_training_conditions
                            ),
                            "target_layers": list(layers),
                            "adapter_rank": rank,
                            "key_dim": key_dim,
                            "temperature": temperature,
                            "dtype": dtype_name,
                            "attn_implementation": attn_implementation,
                            "thresholds": {
                                "answer_exact_min": answer_exact_min,
                                "route_accuracy_min": route_accuracy_min,
                                "rewrite_output_change_min": (
                                    rewrite_output_change_min
                                ),
                            },
                            "delta_mem_config": delta_config.to_dict(),
                            "replication_authorization": replication_authorization,
                        }
                        preflight_gate = build_distributed_preflight_gate(training)
                        preflight_receipt = _signed_payload(
                            {
                                "schema": DISTRIBUTED_PREFLIGHT_SCHEMA,
                                "status": (
                                    "passed" if preflight_gate["passed"] else "failed"
                                ),
                                "gate_passed": preflight_gate["passed"],
                                "hf_endpoint": os.environ.get("HF_ENDPOINT"),
                                "source_manifest_path": str(
                                    source_manifest.expanduser().resolve(strict=True)
                                ),
                                "source_manifest_file_sha256": source.sha256_file(
                                    source_manifest.expanduser().resolve(strict=True)
                                ),
                                "source_manifest_payload_sha256": (
                                    bundle.development_manifest["manifest_receipt"][
                                        "payload_sha256"
                                    ]
                                ),
                                "model_binding_sha256": bundle.model_binding[
                                    "binding_sha256"
                                ],
                                "configuration": preflight_configuration,
                                "training": training,
                                "gate": preflight_gate,
                                "source_files_before": source_before,
                                "source_files_after": rank_source_after,
                                "model_files_before": model_before,
                                "model_files_after": rank_model_after,
                                "code_bindings": _preflight_code_bindings(),
                            },
                            "preflight_receipt_sha256",
                        )
                        _write_json(
                            resolved_output / "preflight_receipt.json",
                            preflight_receipt,
                        )
                        if not preflight_gate["passed"]:
                            preflight_error = RuntimeError(
                                "Distributed production preflight failed: "
                                + ", ".join(preflight_gate["failed_checks"])
                            )
                    except BaseException as error:
                        preflight_error = error
                distributed.phase_consensus(
                    distributed_context,
                    phase="rank-zero-preflight-receipt",
                    error=preflight_error,
                )
                if preflight_error is not None:
                    raise preflight_error
                was_primary = distributed_context.is_primary
                worker_rank = distributed_context.process_rank
                distributed.destroy_distributed_training(distributed_context)
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not was_primary:
                    return {
                        "output_dir": str(resolved_output),
                        "distributed_worker_rank": worker_rank,
                        "training_complete": True,
                    }
                return {
                    "output_dir": str(resolved_output),
                    "preflight_receipt": preflight_receipt,
                    "gate": dict(preflight_receipt or {}).get("gate"),
                }
            save_error: BaseException | None = None
            adapter_files = {}
            if distributed_context.is_primary:
                try:
                    save_delta_mem_adapter(
                        model, resolved_output / "adapter", delta_config
                    )
                    adapter_files = snapshot_directory_files(
                        resolved_output / "adapter"
                    )
                except BaseException as error:
                    save_error = error
            distributed.phase_consensus(
                distributed_context,
                phase="rank-zero-adapter-save",
                error=save_error,
            )
            if save_error is not None:
                raise save_error
            was_primary = distributed_context.is_primary
            worker_rank = distributed_context.process_rank
            distributed.destroy_distributed_training(distributed_context)
            if not was_primary:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return {
                    "output_dir": str(resolved_output),
                    "distributed_worker_rank": worker_rank,
                    "training_complete": True,
                }

    evaluation_examples: dict[str, list[NaturalMemoryExample]] = {
        condition: build_condition_examples(eval_episodes, tokenizer, condition)
        for condition in CONDITIONS
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        if condition == "pristine_frozen_base":
            continue
        evaluations[condition] = evaluate_condition(
            model,
            tokenizer,
            evaluation_examples[condition],
            condition=condition,
            batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            dtype=dtype,
            greedy=greedy,
        )

    del model
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    pristine_model = load_pristine_base_model(
        model_root,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    try:
        evaluations["pristine_frozen_base"] = evaluate_condition(
            pristine_model,
            tokenizer,
            evaluation_examples["pristine_frozen_base"],
            condition="pristine_frozen_base",
            batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            dtype=dtype,
            greedy=greedy,
        )
    finally:
        del pristine_model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    correct_identity = audit_correct_state_identity(
        evaluation_examples["correct_state"], evaluations["correct_state"]
    )
    state_causality = audit_runtime_state_causality(
        {
            condition: evaluation_examples[condition]
            for condition in POSITIVE_CONDITIONS
        },
        {condition: evaluations[condition] for condition in POSITIVE_CONDITIONS},
    )
    rewrite_audit = audit_rewrite_output_change(
        evaluation_examples["correct_state"],
        evaluation_examples["target_slot_rewrite"],
        evaluations["correct_state"],
        evaluations["target_slot_rewrite"],
    )
    control_equivalence = audit_control_equivalence(
        evaluations["no_state"], evaluations["pristine_frozen_base"]
    )

    source_after = assert_snapshot_unchanged(
        source_before,
        description="Natural source",
    )
    model_after = assert_snapshot_unchanged(
        model_before,
        description="Local model",
    )
    if isinstance(training.get("distributed"), Mapping):
        distributed_training = _require_mapping(
            training["distributed"], "distributed training evidence"
        )
        training_topology: Mapping[str, Any] = {
            "mode": "raw_model_replicas_explicit_gradient_sum",
            "backend": distributed_training["backend"],
            "control_backend": distributed_training["control_backend"],
            "world_size": distributed_training["world_size"],
            "local_batch_size": distributed_training["local_batch_size"],
            "local_microbatch_size": distributed_training[
                "local_microbatch_size"
            ],
            "gradient_accumulation_steps": distributed_training[
                "gradient_accumulation_steps"
            ],
            "global_batch_size": distributed_training["global_batch_size"],
            "gradient_synchronization": distributed_training[
                "gradient_synchronization"
            ],
            "gradient_clip_order": distributed_training["gradient_clip_order"],
            "answer_loss_normalization": distributed_training[
                "answer_loss_normalization"
            ],
            "route_loss_normalization": distributed_training[
                "route_loss_normalization"
            ],
            "online_memory_state": distributed_training["online_memory_state"],
        }
    elif sealed_chain is not None:
        locked_topology_protocol = _require_mapping(
            sealed_chain.get("development_protocol"),
            "locked development protocol",
        )
        training_topology = _require_mapping(
            locked_topology_protocol.get("training_topology"),
            "locked development training topology",
        )
    else:
        training_topology = {
            "mode": "single_process",
            "world_size": 1,
            "local_batch_size": batch_size,
            "global_batch_size": batch_size,
        }
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "runner_schema": RUN_SCHEMA,
        "source_schema": source.SCHEMA,
        "profile": profile,
        "replication_authorization": replication_authorization,
        "conditions": list(CONDITIONS),
        "training_conditions": list(selected_training_conditions),
        "training_dataset_audit": training_dataset_audit,
        "opened_splits": list(bundle.eligibility["opened_splits"]),
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "model_binding_sha256": bundle.model_binding["binding_sha256"],
        "target_layers": list(layers),
        "rank": rank,
        "key_dim": key_dim,
        "temperature": temperature,
        "hard_negative_margin": hard_negative_margin,
        "hard_negative_weight": hard_negative_weight,
        "route_objective": (
            "layer_mean(cross_entropy)"
            if hard_negative_weight == 0.0
            else (
                "layer_mean(cross_entropy + hard_negative_weight * "
                "relu(hard_negative_margin + hardest_wrong_logit - target_logit))"
            )
        ),
        "eval_batch_size": eval_batch_size,
        "greedy_answer_evaluation": greedy,
        "dtype": dtype_name,
        "attn_implementation": attn_implementation,
        "write_read_cache_policy": "every model invocation passes use_cache=False; writes and reads are separate",
        "query_encoding_policy": "Gemma chat-template address-only prefix and canonical JSON label tokenized as one full string with offset-derived disjoint masks and boundary-crossing rejection",
        "answer_logit_policy": ANSWER_LOGIT_POLICY,
        "shared_state_batching_policy": SHARED_STATE_BATCHING_POLICY,
        "training_topology": dict(training_topology),
        "sealed_chain": sealed_chain,
        "thresholds": {
            "answer_exact_min": answer_exact_min,
            "route_accuracy_min": route_accuracy_min,
            "rewrite_output_change_min": rewrite_output_change_min,
        },
    }
    if sealed_chain is not None:
        locked_protocol = _require_mapping(
            sealed_chain.get("development_protocol"), "locked development protocol"
        )
        frozen_fields = (
            "replication_authorization",
            "conditions",
            "training_conditions",
            "training_dataset_audit",
            "hf_endpoint",
            "model_binding_sha256",
            "target_layers",
            "rank",
            "key_dim",
            "temperature",
            "hard_negative_margin",
            "hard_negative_weight",
            "route_objective",
            "eval_batch_size",
            "greedy_answer_evaluation",
            "dtype",
            "attn_implementation",
            "write_read_cache_policy",
            "query_encoding_policy",
            "answer_logit_policy",
            "shared_state_batching_policy",
            "training_topology",
            "thresholds",
        )
        mismatches = [
            field
            for field in frozen_fields
            if locked_protocol.get(field) != protocol.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Sealed protocol differs from frozen development fields: "
                + ", ".join(mismatches)
            )
    training_configuration = {
        "schema": TRAINING_CONFIGURATION_SCHEMA,
        "profile": profile,
        "replication_authorization": replication_authorization,
        "seed": seed,
        "training_conditions": list(selected_training_conditions),
        "training_dataset_audit": training_dataset_audit,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size_semantics": "global_training_rows_per_optimizer_update",
        "batch_size": batch_size,
        "training_topology": dict(training_topology),
        "eval_batch_size": eval_batch_size,
        "greedy_answer_evaluation": greedy,
        "learning_rate": learning_rate,
        "answer_weight": answer_weight,
        "route_weight": route_weight,
        "hard_negative_margin": hard_negative_margin,
        "hard_negative_weight": hard_negative_weight,
        "max_grad_norm": max_grad_norm,
        "device": device_name,
        "dtype": dtype_name,
        "thresholds": dict(protocol["thresholds"]),
    }
    gate = build_gate(
        evaluations,
        state_identity=correct_identity,
        state_causality=state_causality,
        rewrite_audit=rewrite_audit,
        control_equivalence=control_equivalence,
        profile_eligibility=bundle.eligibility,
        trainable_audit=trainable_audit,
        immutability_passed=source_before == source_after and model_before == model_after,
        training=training,
        thresholds=thresholds,
    )
    protocol_hash = _sha256_json(protocol)
    training_configuration_hash = _sha256_json(training_configuration)
    evaluation_payload = {
        "schema": EVALUATION_SCHEMA,
        "profile": profile,
        "training": training,
        "conditions": evaluations,
        "correct_state_identity": correct_identity,
        "runtime_state_causality": state_causality,
        "rewrite_audit": rewrite_audit,
        "control_equivalence": control_equivalence,
        "trainable_audit": trainable_audit,
        "gate": gate,
    }
    _write_json(resolved_output / "protocol.json", protocol)
    _write_json(
        resolved_output / "training_configuration.json",
        training_configuration,
    )
    _write_json(resolved_output / "evaluation.json", evaluation_payload)
    adapter_files_sha256 = _sha256_json(adapter_files)
    receipt = {
        "schema": RUN_SCHEMA,
        "profile": profile,
        "gate_passed": gate["passed"],
        "source_manifest_payload_sha256": (
            (bundle.development_manifest or bundle.sealed_manifest or {})
            .get("manifest_receipt", {})
            .get("payload_sha256")
        ),
        "source_files_before": source_before,
        "source_files_after": source_after,
        "model_files_before": model_before,
        "model_files_after": model_after,
        "protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_configuration_hash,
        "training_dataset_audit_sha256": _sha256_json(training_dataset_audit),
        "evaluation_sha256": _sha256_json(evaluation_payload),
        "replaced_layers": list(replaced_layers),
        "trainable_names": list(trainable_names),
        "checkpointed_frozen_mlps": list(checkpointed_mlps),
        "adapter_files": adapter_files,
        "adapter_files_sha256": adapter_files_sha256,
        "gate": gate,
    }
    receipt = _signed_payload(receipt, "run_receipt_sha256")
    _write_json(resolved_output / "run_receipt.json", receipt)
    if profile == "development" and gate["passed"]:
        sealed_lock_receipt = {
            "schema": source.SEALED_LOCK_SCHEMA,
            "configuration_frozen": True,
            "development_gate_passed": True,
            "benchmark_contract_sha256": bundle.development_manifest[
                "benchmark_contract_sha256"
            ],
            "development_manifest_payload_sha256": bundle.development_manifest[
                "manifest_receipt"
            ]["payload_sha256"],
            "runner_protocol_sha256": protocol_hash,
            "training_configuration_sha256": training_configuration_hash,
            "training_dataset_audit_sha256": _sha256_json(
                training_dataset_audit
            ),
            "evaluation_sha256": _sha256_json(evaluation_payload),
            "development_run_receipt_sha256": receipt["run_receipt_sha256"],
            "adapter_files_sha256": adapter_files_sha256,
        }
        _write_json(resolved_output / "sealed_lock_receipt.json", sealed_lock_receipt)
    if not gate["passed"]:
        raise RuntimeError(
            "Natural memory gate failed: " + ", ".join(gate["failed_checks"])
        )
    return {
        "output_dir": str(resolved_output),
        "receipt": receipt,
        "evaluation": evaluation_payload,
        "gate": gate,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="development")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--replication-protocol", type=Path)
    parser.add_argument("--replication-amendment", type=Path)
    parser.add_argument("--replication-id")
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--epochs", type=int, default=PRODUCTION_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=PRODUCTION_UPDATES)
    parser.add_argument(
        "--batch-size", type=int, default=distributed.REQUIRED_GLOBAL_BATCH_SIZE
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=PRODUCTION_EVAL_BATCH_SIZE
    )
    parser.add_argument(
        "--learning-rate", type=float, default=PRODUCTION_LEARNING_RATE
    )
    parser.add_argument("--answer-weight", type=float, default=PRODUCTION_ANSWER_WEIGHT)
    parser.add_argument("--route-weight", type=float, default=PRODUCTION_ROUTE_WEIGHT)
    parser.add_argument(
        "--hard-negative-margin",
        type=float,
        default=PRODUCTION_HARD_NEGATIVE_MARGIN,
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=PRODUCTION_HARD_NEGATIVE_WEIGHT,
    )
    parser.add_argument("--max-grad-norm", type=float, default=PRODUCTION_MAX_GRAD_NORM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--target-layers", default=",".join(map(str, DEFAULT_TARGET_LAYERS)))
    parser.add_argument(
        "--training-conditions",
        default=",".join(DEFAULT_TRAINING_CONDITIONS),
        help=(
            "Comma-separated positive memory conditions; the formal compositional "
            "contract uses all five"
        ),
    )
    parser.add_argument("--rank", type=int, default=PRODUCTION_ADAPTER_RANK)
    parser.add_argument("--key-dim", type=int, default=PRODUCTION_KEY_DIM)
    parser.add_argument("--temperature", type=float, default=PRODUCTION_TEMPERATURE)
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    parser.add_argument(
        "--answer-exact-min", type=float, default=PRODUCTION_ANSWER_EXACT_MIN
    )
    parser.add_argument(
        "--route-accuracy-min", type=float, default=PRODUCTION_ROUTE_ACCURACY_MIN
    )
    parser.add_argument(
        "--rewrite-output-change-min",
        type=float,
        default=PRODUCTION_REWRITE_OUTPUT_CHANGE_MIN,
    )
    parser.add_argument(
        "--capture-distributed-step-evidence",
        action="store_true",
        help="Record per-rank adapter, optimizer, and CUDA evidence at every step",
    )
    parser.add_argument(
        "--distributed-preflight",
        action="store_true",
        help="Run exactly three production training steps and emit a signed receipt",
    )
    parser.set_defaults(greedy=True)
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    try:
        result = run_experiment(
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
            profile=args.profile,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            development_run_dir=args.development_run_dir,
            replication_protocol=args.replication_protocol,
            replication_amendment=args.replication_amendment,
            replication_id=args.replication_id,
            seed=args.seed,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            answer_weight=args.answer_weight,
            route_weight=args.route_weight,
            hard_negative_margin=args.hard_negative_margin,
            hard_negative_weight=args.hard_negative_weight,
            max_grad_norm=args.max_grad_norm,
            device_name=args.device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
            target_layers=_parse_layers(args.target_layers),
            training_conditions=_parse_training_conditions(
                args.training_conditions
            ),
            rank=args.rank,
            key_dim=args.key_dim,
            temperature=args.temperature,
            greedy=args.greedy,
            answer_exact_min=args.answer_exact_min,
            route_accuracy_min=args.route_accuracy_min,
            rewrite_output_change_min=args.rewrite_output_change_min,
            distributed_context=context,
            capture_distributed_step_evidence=(
                args.capture_distributed_step_evidence
            ),
            distributed_preflight=args.distributed_preflight,
        )
    finally:
        distributed.destroy_distributed_training(context)
    if "distributed_worker_rank" in result:
        print(
            json.dumps(
                {
                    "output_dir": result["output_dir"],
                    "distributed_worker_rank": result[
                        "distributed_worker_rank"
                    ],
                    "training_complete": result["training_complete"],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {"output_dir": result["output_dir"], "gate": result["gate"]},
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
