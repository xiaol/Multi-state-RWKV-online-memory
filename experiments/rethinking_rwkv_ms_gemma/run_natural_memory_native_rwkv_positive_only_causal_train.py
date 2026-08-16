#!/usr/bin/env python3
"""Train stable RWKV readouts only through the correct native state path."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

from deltamem.core.delta import reset_delta_mem_states, snapshot_delta_mem_weights
from experiments.rethinking_rwkv_ms_gemma import (
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_train as benchmark_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stopgrad_router_causal_train as stopgrad,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_positive_only_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_positive_only_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_positive_only_causal_train_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_positive_only_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "50e96d1882335108a9031747b577858fdd59ffb5a3cb95c99008204552417bfd"
)
SEED = 73
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.05
HELDOUT_ORDINALS = (
    134, 1308, 434, 696, 1187, 861, 253, 87,
    412, 947, 1442, 541, 599, 470, 60, 761,
    804, 146, 1093, 910, 225, 121, 362, 817,
    329, 103, 567, 339, 1257, 585, 616, 521,
)
HELDOUT_PAYLOAD_SHA256 = (
    "6ceb493d95d55186a8d1ba97a7030ba0c8179c6e4a1e0bb7e7da9d4d10b5e824"
)
SPSA_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_spsa_gate_causal_train_v1/"
    "result.json"
)
SPSA_RESULT_FILE_SHA256 = (
    "1850d6a68d10e86a35b81e8d16199d243a638eb4be02a67cc5b822f5ca019571"
)
SPSA_RESULT_RECEIPT = (
    "6b9e9d25aa84fa1a0af217227d7e572abcdc9da3711d56d8094db74ef5d388d1"
)
SELECTED_CANDIDATE = stopgrad.SELECTED_CANDIDATE
FILTER_NONFINITE_ROWS = False
MIN_ACCEPTED_ROWS_PER_UPDATE = causal_train.GLOBAL_BATCH_SIZE
MAX_TOTAL_REJECTED_ROWS = 0


def accumulate_finite_row_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: dict[str, torch.Tensor],
) -> Mapping[str, Any]:
    """Copy one finite row backward into buffers without retaining bad values."""
    validation = distributed.validate_local_gradients(named_trainable)
    if validation["non_fp32_gradient_tensors"]:
        raise RuntimeError(f"Positive-only row gradients are not FP32: {validation!r}")
    if validation["active_gradient_tensors"] == 0:
        raise RuntimeError(f"Positive-only row has no active gradients: {validation!r}")
    if validation["nonfinite_gradient_tensors"]:
        return validation
    with torch.no_grad():
        for name, parameter in named_trainable:
            if parameter.grad is None:
                continue
            if name not in clean_gradients:
                clean_gradients[name] = parameter.grad.detach().clone()
            else:
                clean_gradients[name].add_(parameter.grad.detach())
    return validation


def materialize_clean_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: Mapping[str, torch.Tensor],
    *,
    scale: float,
) -> None:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Positive-only clean-gradient scale must be finite and positive")
    with torch.no_grad():
        for name, parameter in named_trainable:
            gradient = clean_gradients.get(name)
            parameter.grad = None if gradient is None else gradient.mul(scale)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Positive-only protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Positive-only protocol payload differs")
    if shared.sha256_file(SPSA_RESULT) != SPSA_RESULT_FILE_SHA256:
        raise ValueError("SPSA result binding differs")
    result = json.loads(SPSA_RESULT.read_text(encoding="utf-8"))
    result_receipt = result.get("receipt", {})
    unsigned_result = dict(result)
    unsigned_result.pop("receipt", None)
    if (
        shared.canonical_sha256(unsigned_result) != SPSA_RESULT_RECEIPT
        or result_receipt.get("payload_sha256") != SPSA_RESULT_RECEIPT
        or result.get("status") != "spsa_gate_heldout_failed_generation_blocked"
        or result.get("training_passed") is not True
        or result.get("passed") is not False
    ):
        raise ValueError("SPSA failure does not authorize positive-only training")
    endpoint = protocol.get("heldout_causal_endpoint", {})
    training = protocol.get("training", {})
    if (
        endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("max_gradient_norm") != MAX_GRAD_NORM
        or training.get("optimizer_updates") != shared.UPDATES
        or training.get("control_branch_backward_calls") != 0
    ):
        raise ValueError("Positive-only training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Positive-only training may not open protected data")
    return protocol


def evaluate_forward_only_controls(
    model: torch.nn.Module,
    target_batch: evolution.NativeFullRowBatch,
    donor_batch: evolution.NativeFullRowBatch,
) -> Mapping[str, Any]:
    modules = causal_train.ordered_modules(model)
    with torch.no_grad():
        evolution._native_write(model, target_batch, dtype=torch.bfloat16)
        correct_state = causal_train.capture_online_state_references(modules)
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
    values = {
        "zero": contrast.detached_answer_ce(zero_logits, target_batch.labels),
        "donor": contrast.detached_answer_ce(donor_logits, target_batch.labels),
        "layer_permuted": contrast.detached_answer_ce(
            permuted_logits,
            target_batch.labels,
        ),
    }
    finite = all(
        bool(torch.isfinite(logits).all().item())
        for logits in (zero_logits, donor_logits, permuted_logits)
    )
    del correct_state, donor_state, zero_logits, donor_logits, permuted_logits
    return {
        "condition_ce_and_tokens": values,
        "projected_carrier_fixed": bool(
            donor_carrier_fixed and permuted_carrier_fixed
        ),
        "all_control_logits_finite": finite,
    }


def train_positive_only(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[contrast.ContrastScheduleStep],
    *,
    updates: int,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    initial_state = snapshot_delta_mem_weights(model)
    initial_adapter_sha256 = runtime._state_dict_sha256(initial_state)
    initial_recurrent_sha256 = benchmark_train.state_subset_sha256(
        initial_state,
        recurrent_only=True,
    )
    initial_trainable_sha256 = causal_train.trainable_subset_sha256(named_trainable)
    for description, value in (
        ("initial positive-only adapter", initial_adapter_sha256),
        ("initial positive-only recurrent subset", initial_recurrent_sha256),
        ("initial positive-only trainable subset", initial_trainable_sha256),
    ):
        distributed.require_consensus(context, value, description=description)

    progress_path = output_dir / "training_progress.jsonl"
    total_metrics = [0.0] * 10
    minimum_gradient_norm = math.inf
    maximum_global_inactive = 0
    first_update_gradient_audit: Mapping[str, Any] | None = None
    projected_carrier_fixed_every_row = True
    all_control_logits_finite = True
    total_accepted_gradient_rows = 0
    total_rejected_gradient_rows = 0
    minimum_accepted_rows_per_update = causal_train.GLOBAL_BATCH_SIZE
    rejected_source_ordinals: list[int] = []
    started = time.time()
    for schedule_step in schedule[:updates]:
        local_start = context.process_rank * causal_train.LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + causal_train.LOCAL_ROWS
        ]
        if len(local_sources) != causal_train.LOCAL_ROWS:
            raise RuntimeError("Positive-only local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        local_metrics = [0.0] * 10
        clean_gradients: dict[str, torch.Tensor] = {}
        local_row_gradient_evidence: list[dict[str, Any]] = []
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
            _, correct_logits = evolution.checkpointed_native_write_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            correct_ce, correct_tokens = contrast.detached_answer_ce(
                correct_logits,
                target_batch.labels,
            )
            controls = evaluate_forward_only_controls(
                model,
                target_batch,
                donor_batch,
            )
            condition_values = controls["condition_ce_and_tokens"]
            zero_ce, zero_tokens = condition_values["zero"]
            donor_ce, donor_tokens = condition_values["donor"]
            permuted_ce, permuted_tokens = condition_values["layer_permuted"]
            if len({correct_tokens, zero_tokens, donor_tokens, permuted_tokens}) != 1:
                raise RuntimeError("Positive-only condition token counts differ")
            _, chunks = causal_train.backward_logits(
                correct_logits,
                target_batch.labels,
                coefficient=1.0,
            )
            if FILTER_NONFINITE_ROWS:
                row_gradient_validation = accumulate_finite_row_gradients(
                    named_trainable,
                    clean_gradients,
                )
                accepted = row_gradient_validation["passed"] is True
                local_row_gradient_evidence.append(
                    {
                        "rank": context.process_rank,
                        "source_ordinal": source_ordinal,
                        "accepted": accepted,
                        "gradient_validation": row_gradient_validation,
                    }
                )
                optimizer.zero_grad(set_to_none=True)
            values = (
                correct_ce,
                zero_ce - correct_ce,
                donor_ce - correct_ce,
                permuted_ce - correct_ce,
                float((zero_ce - correct_ce) < causal_train.MARGIN),
                float((donor_ce - correct_ce) < causal_train.MARGIN),
                float((permuted_ce - correct_ce) < causal_train.MARGIN),
                float(controls["projected_carrier_fixed"]),
                float(controls["all_control_logits_finite"]),
                float(chunks),
            )
            local_metrics = [
                total + value for total, value in zip(local_metrics, values)
            ]
            del target_batch, donor_batch, correct_logits, controls
            reset_delta_mem_states(model)
            evolution.release_native_row_allocator_cache(context.device)
        metric_tensor = contrast.gate._prepare_distributed_scalar_sums(
            context,
            local_metrics,
        )
        metrics = contrast.gate._distributed_scalar_sums(context, metric_tensor)
        if (
            metrics[7] != causal_train.GLOBAL_BATCH_SIZE
            or metrics[8] != causal_train.GLOBAL_BATCH_SIZE
        ):
            raise RuntimeError("Positive-only forward control audit failed")
        row_filter_evidence: Mapping[str, Any] | None = None
        if FILTER_NONFINITE_ROWS:
            gathered_row_evidence = distributed.gather_objects(
                context,
                local_row_gradient_evidence,
            )
            rank_rows = [list(value) for value in gathered_row_evidence]
            all_row_evidence = [row for rows_on_rank in rank_rows for row in rows_on_rank]
            accepted_rows = sum(row["accepted"] is True for row in all_row_evidence)
            rejected_rows = [
                int(row["source_ordinal"])
                for row in all_row_evidence
                if row["accepted"] is not True
            ]
            total_accepted_gradient_rows += accepted_rows
            total_rejected_gradient_rows += len(rejected_rows)
            rejected_source_ordinals.extend(rejected_rows)
            minimum_accepted_rows_per_update = min(
                minimum_accepted_rows_per_update,
                accepted_rows,
            )
            filter_error: BaseException | None = None
            if len(all_row_evidence) != causal_train.GLOBAL_BATCH_SIZE:
                filter_error = RuntimeError("Positive-only row-filter evidence is incomplete")
            elif accepted_rows < MIN_ACCEPTED_ROWS_PER_UPDATE:
                filter_error = RuntimeError(
                    "Positive-only row filter accepted too few rows: "
                    f"{accepted_rows} < {MIN_ACCEPTED_ROWS_PER_UPDATE}"
                )
            elif total_rejected_gradient_rows > MAX_TOTAL_REJECTED_ROWS:
                filter_error = RuntimeError(
                    "Positive-only row filter rejected too many total rows: "
                    f"{total_rejected_gradient_rows} > {MAX_TOTAL_REJECTED_ROWS}"
                )
            distributed.phase_consensus(
                context,
                phase=f"positive-only-step-{schedule_step.step}-row-filter",
                error=filter_error,
            )
            gradient_rescale = causal_train.GLOBAL_BATCH_SIZE / accepted_rows
            materialize_clean_gradients(
                named_trainable,
                clean_gradients,
                scale=gradient_rescale,
            )
            row_filter_evidence = {
                "enabled": True,
                "accepted_rows": accepted_rows,
                "rejected_rows": len(rejected_rows),
                "rejected_source_ordinals": rejected_rows,
                "gradient_rescale": gradient_rescale,
                "rank_rows": rank_rows,
            }
        else:
            total_accepted_gradient_rows += causal_train.GLOBAL_BATCH_SIZE
        local_gradient_validation = distributed.validate_local_gradients(named_trainable)
        validation_error: BaseException | None = None
        if local_gradient_validation["passed"] is not True:
            validation_error = RuntimeError(
                f"Positive-only local gradients are invalid: {local_gradient_validation!r}"
            )
        distributed.phase_consensus(
            context,
            phase=f"positive-only-step-{schedule_step.step}-gradient-validation",
            error=validation_error,
        )
        collective = distributed.sum_gradients(context, named_trainable)
        inactive = len(collective["global_inactive_parameter_indices"])
        maximum_global_inactive = max(maximum_global_inactive, inactive)
        if inactive:
            raise RuntimeError("Positive-only optimizer has inactive parameters")
        gradient_audit = None
        if schedule_step.step == 1:
            gradient_audit = recurrent_calibration.audit_recurrent_readout_gradients(
                named_trainable
            )
            if gradient_audit["passed"] is not True:
                raise RuntimeError("Positive-only recurrent gradients are invalid")
            first_update_gradient_audit = gradient_audit
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        gradient_norm_value = float(gradient_norm.detach().float().item())
        if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0.0:
            raise RuntimeError("Positive-only gradient norm is invalid")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm_value)
        optimizer.step()
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": causal_train.GLOBAL_BATCH_SIZE,
            "mean_correct_ce": metrics[0] / causal_train.GLOBAL_BATCH_SIZE,
            "mean_zero_minus_correct_ce": metrics[1] / causal_train.GLOBAL_BATCH_SIZE,
            "mean_donor_minus_correct_ce": metrics[2] / causal_train.GLOBAL_BATCH_SIZE,
            "mean_layer_permuted_minus_correct_ce": (
                metrics[3] / causal_train.GLOBAL_BATCH_SIZE
            ),
            "active_zero_rows": int(metrics[4]),
            "active_donor_rows": int(metrics[5]),
            "active_layer_permuted_rows": int(metrics[6]),
            "projected_carrier_fixed_rows": int(metrics[7]),
            "finite_control_rows": int(metrics[8]),
            "checkpointed_ce_chunks": int(metrics[9]),
            "correct_branch_backward_calls": causal_train.GLOBAL_BATCH_SIZE,
            "control_branch_backward_calls": 0,
            "gradient_norm_before_clip": gradient_norm_value,
            "gradient_collective_sha256": shared.canonical_sha256(collective),
            "local_gradient_validation": local_gradient_validation,
            "row_filter": row_filter_evidence,
            "gradient_audit": gradient_audit,
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
        }
        if context.is_primary:
            causal_train.append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "correct_ce": round(metrics[0] / causal_train.GLOBAL_BATCH_SIZE, 6),
                        "zero_margin": round(metrics[1] / causal_train.GLOBAL_BATCH_SIZE, 6),
                        "donor_margin": round(metrics[2] / causal_train.GLOBAL_BATCH_SIZE, 6),
                        "permuted_margin": round(metrics[3] / causal_train.GLOBAL_BATCH_SIZE, 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_metrics = [
            total + value for total, value in zip(total_metrics, metrics)
        ]
        projected_carrier_fixed_every_row = (
            projected_carrier_fixed_every_row
            and metrics[7] == causal_train.GLOBAL_BATCH_SIZE
        )
        all_control_logits_finite = (
            all_control_logits_finite
            and metrics[8] == causal_train.GLOBAL_BATCH_SIZE
        )

    final_state = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = runtime._state_dict_sha256(final_state)
    final_recurrent_sha256 = benchmark_train.state_subset_sha256(
        final_state,
        recurrent_only=True,
    )
    final_trainable_sha256 = causal_train.trainable_subset_sha256(named_trainable)
    for description, value in (
        ("final positive-only adapter", final_adapter_sha256),
        ("final positive-only recurrent subset", final_recurrent_sha256),
        ("final positive-only trainable subset", final_trainable_sha256),
    ):
        distributed.require_consensus(context, value, description=description)
    denominator = updates * causal_train.GLOBAL_BATCH_SIZE
    return {
        "optimization": "correct_state_ce_only",
        "updates": updates,
        "rows": denominator,
        "elapsed_seconds": time.time() - started,
        "mean_correct_ce": total_metrics[0] / denominator,
        "mean_zero_minus_correct_ce": total_metrics[1] / denominator,
        "mean_donor_minus_correct_ce": total_metrics[2] / denominator,
        "mean_layer_permuted_minus_correct_ce": total_metrics[3] / denominator,
        "active_zero_fraction": total_metrics[4] / denominator,
        "active_donor_fraction": total_metrics[5] / denominator,
        "active_layer_permuted_fraction": total_metrics[6] / denominator,
        "minimum_gradient_norm_before_clip": minimum_gradient_norm,
        "maximum_global_inactive_parameter_tensors": maximum_global_inactive,
        "projected_carrier_fixed_every_row": projected_carrier_fixed_every_row,
        "all_control_logits_finite": all_control_logits_finite,
        "correct_branch_backward_calls": denominator,
        "control_branch_backward_calls": 0,
        "row_filter": {
            "enabled": FILTER_NONFINITE_ROWS,
            "minimum_required_accepted_rows_per_update": (
                MIN_ACCEPTED_ROWS_PER_UPDATE
            ),
            "maximum_total_rejected_rows": MAX_TOTAL_REJECTED_ROWS,
            "minimum_accepted_rows_per_update": minimum_accepted_rows_per_update,
            "accepted_gradient_rows": total_accepted_gradient_rows,
            "rejected_gradient_rows": total_rejected_gradient_rows,
            "rejected_source_ordinals": rejected_source_ordinals,
        },
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "initial_recurrent_subset_sha256": initial_recurrent_sha256,
        "final_recurrent_subset_sha256": final_recurrent_sha256,
        "recurrent_subset_changed": initial_recurrent_sha256 != final_recurrent_sha256,
        "initial_trainable_subset_sha256": initial_trainable_sha256,
        "final_trainable_subset_sha256": final_trainable_sha256,
        "trainable_subset_changed": initial_trainable_sha256 != final_trainable_sha256,
        "first_update_gradient_audit": first_update_gradient_audit,
        "first_update_recurrent_gradient_audit": first_update_gradient_audit,
        "progress_sha256": shared.sha256_file(progress_path) if context.is_primary else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }


@contextmanager
def training_bindings() -> Iterator[None]:
    with stopgrad.training_bindings():
        bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SEED": SEED,
            "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "positive_only_heldout_passed_generation_authorized",
            "FAIL_STATUS": "positive_only_heldout_failed_generation_blocked",
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": train_positive_only,
            "validate_protocol": validate_protocol,
        }
        previous = {name: getattr(shared, name) for name in bindings}
        previous_learning_rate = causal_train.LEARNING_RATE
        previous_max_grad_norm = causal_train.MAX_GRAD_NORM
        try:
            for name, value in bindings.items():
                setattr(shared, name, value)
            causal_train.LEARNING_RATE = LEARNING_RATE
            causal_train.MAX_GRAD_NORM = MAX_GRAD_NORM
            yield
        finally:
            causal_train.LEARNING_RATE = previous_learning_rate
            causal_train.MAX_GRAD_NORM = previous_max_grad_norm
            for name, value in previous.items():
                setattr(shared, name, value)


def validate_calibration_result() -> Mapping[str, Any]:
    return shared.validate_calibration_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
