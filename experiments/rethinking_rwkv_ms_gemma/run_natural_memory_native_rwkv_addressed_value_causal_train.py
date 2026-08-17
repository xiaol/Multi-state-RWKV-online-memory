#!/usr/bin/env python3
"""Train addressed RWKV values against fixed-carrier causal controls."""

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
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    attach_delta_mem,
    iter_delta_mem_modules,
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
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_value_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_addressed_value_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_addressed_value_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_value_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "15f84baa45e3c5c1aa8c9d7e3c1a20a936824103684911a90b21966ff81be23a"
)
CALIBRATION_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_addressed_value_calibration_v1/result.json"
)
CALIBRATION_RESULT_FILE_SHA256 = (
    "80e7e65e1b4bdc55c0200067a0aa038843fa719727fd6fee627f3eeffafd61cf"
)
CALIBRATION_RESULT_RECEIPT = (
    "63d32f841f3cc1c9471a4caa0f0067273ecd4f7c5bc4ca0645fd33903dd87efa"
)
SELECTED_CANDIDATE = calibration.SELECTED_CANDIDATE
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 60
GLOBAL_BATCH_SIZE = 8
LOCAL_ROWS = 2
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 8
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
FIRST_UPDATE_GRADIENT_AUDITOR = (
    recurrent_calibration.audit_recurrent_readout_gradients
)
FILTER_NONFINITE_ROWS = False
MIN_ACCEPTED_ROWS_PER_UPDATE = GLOBAL_BATCH_SIZE
MAX_TOTAL_REJECTED_ROWS = 0
OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = False
SERIALIZE_CONTROL_BRANCH_GRAPHS = False
RECURRENT_ATTRIBUTES = (
    "delta_state",
    "rwkv_ms_positions",
    "rwkv_ms_previous_source",
)
PROJECTED_ATTRIBUTES = (
    "projected_kv_keys",
    "projected_kv_values",
    "projected_kv_occupied",
    "projected_kv_surprise",
)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Addressed causal training output must be fresh: {path}")
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


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed causal training protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Addressed causal training protocol payload differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "calibration_protocol_payload_sha256": calibration.PROTOCOL_PAYLOAD_SHA256,
        "calibration_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_addressed_value_calibration_v1/result.json"
        ),
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": "calibration_passed_causal_training_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if authorization != required_authorization:
        raise ValueError("Addressed causal training authorization differs")
    training = protocol.get("training", {})
    required_training = {
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "global_batch_rows": GLOBAL_BATCH_SIZE,
        "local_rows_per_rank": LOCAL_ROWS,
        "preflight_optimizer_updates": PREFLIGHT_UPDATES,
        "screen_optimizer_updates": TRAIN_UPDATES,
        "contrast_weight_per_active_control": CONTRAST_WEIGHT,
        "contrast_margin": MARGIN,
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("Addressed causal training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Addressed causal training may not authorize protected data")
    return protocol


def validate_calibration_result() -> Mapping[str, Any]:
    if sha256_file(CALIBRATION_RESULT) != CALIBRATION_RESULT_FILE_SHA256:
        raise ValueError("Addressed calibration result file hash differs")
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed calibration result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if (
        digest != CALIBRATION_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or result.get("schema") != calibration.SCHEMA
        or result.get("status") != "calibration_passed_causal_training_authorized"
        or result.get("passed") is not True
        or result.get("causal_training_authorized") is not True
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Addressed calibration did not authorize training")
    return result


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    Any,
    Any,
    tuple[tuple[str, torch.nn.Parameter], ...],
    Mapping[str, Any],
]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=False,
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
    delta_config = screen.build_config()
    replaced = attach_delta_mem(model, delta_config)
    named_trainable, trainable_audit = benchmark_train.configure_trainable_parameters(
        model,
        architecture="hybrid_candidate",
    )
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    wrappers_valid = all(
        module.memory_backend == "rwkv_ms"
        and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_write_mode == "recurrent"
        and module.rwkv_ms_hybrid_mode == "addressed_value"
        and module.rwkv_ms_hybrid_gain == 0.03125
        for _, module in modules
    )
    audit = {
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "all_wrappers_addressed_value": wrappers_valid,
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "backbone_dtype": "bfloat16",
        "trainable_master_dtype": "float32",
        "trainables": trainable_audit,
    }
    if (
        len(replaced) != preflight.EXPECTED_LAYERS
        or len(modules) != preflight.EXPECTED_LAYERS
        or not wrappers_valid
    ):
        raise RuntimeError(f"Addressed causal attachment failed: {audit!r}")
    return model, tokenizer, delta_config, named_trainable, audit


def ordered_modules(
    model: torch.nn.Module,
) -> tuple[tuple[str, Any], ...]:
    modules = tuple(iter_delta_mem_modules(model))
    ordered = tuple(
        sorted(
            modules,
            key=lambda item: int(item[0].split(".layers.", 1)[1].split(".", 1)[0]),
        )
    )
    if len(ordered) != preflight.EXPECTED_LAYERS:
        raise ValueError("Addressed causal training requires 42 wrapped layers")
    return ordered


def capture_online_state_references(
    modules: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    captured: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in modules:
        values: dict[str, torch.Tensor] = {}
        for attribute in (*RECURRENT_ATTRIBUTES, *PROJECTED_ATTRIBUTES):
            value = getattr(module, attribute)
            if value is None:
                raise RuntimeError(
                    f"Addressed causal write omitted {name}.{attribute}"
                )
            values[attribute] = value
        captured[name] = values
    return captured


def install_intervened_state(
    modules: Sequence[tuple[str, Any]],
    *,
    projected: Mapping[str, Mapping[str, torch.Tensor]],
    recurrent: Mapping[str, Mapping[str, torch.Tensor]],
    rotate_recurrent_layers: bool,
) -> bool:
    module_names = [name for name, _ in modules]
    for index, (name, module) in enumerate(modules):
        recurrent_name = (
            module_names[(index + 1) % len(module_names)]
            if rotate_recurrent_layers
            else name
        )
        for attribute in PROJECTED_ATTRIBUTES:
            setattr(module, attribute, projected[name][attribute])
        for attribute in RECURRENT_ATTRIBUTES:
            setattr(module, attribute, recurrent[recurrent_name][attribute])
    return all(
        getattr(module, attribute) is projected[name][attribute]
        for name, module in modules
        for attribute in PROJECTED_ATTRIBUTES
    )


def checkpointed_intervened_write_read(
    model: torch.nn.Module,
    target_batch: evolution.NativeFullRowBatch,
    *,
    donor_batch: evolution.NativeFullRowBatch | None,
    rotate_recurrent_layers: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Mapping[str, bool]]:
    modules = ordered_modules(model)
    audit = {"projected_carrier_references_fixed": True}

    def write_read(*tensors: torch.Tensor) -> torch.Tensor:
        target = evolution.NativeFullRowBatch(
            examples=target_batch.examples,
            write_input_ids=tensors[0],
            write_attention_mask=tensors[1],
            read_input_ids=tensors[2],
            read_attention_mask=tensors[3],
            labels=target_batch.labels,
        )
        evolution._native_write(model, target, dtype=dtype)
        correct = capture_online_state_references(modules)
        if donor_batch is None:
            recurrent = correct
        else:
            donor = evolution.NativeFullRowBatch(
                examples=target_batch.examples,
                write_input_ids=tensors[4],
                write_attention_mask=tensors[5],
                read_input_ids=tensors[2],
                read_attention_mask=tensors[3],
                labels=target_batch.labels,
            )
            evolution._native_write(model, donor, dtype=dtype)
            recurrent = capture_online_state_references(modules)
        audit["projected_carrier_references_fixed"] = (
            audit["projected_carrier_references_fixed"]
            and install_intervened_state(
                modules,
                projected=correct,
                recurrent=recurrent,
                rotate_recurrent_layers=rotate_recurrent_layers,
            )
        )
        return evolution._native_read(model, target, dtype=dtype)

    inputs = [
        target_batch.write_input_ids,
        target_batch.write_attention_mask,
        target_batch.read_input_ids,
        target_batch.read_attention_mask,
    ]
    if donor_batch is not None:
        inputs.extend(
            [donor_batch.write_input_ids, donor_batch.write_attention_mask]
        )
    logits = checkpoint(write_read, *inputs, use_reentrant=False)
    return logits, audit


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
        raise RuntimeError("Addressed causal signed loss is non-finite")
    scaled.backward()
    del loss_sum, mean_ce, scaled
    return tokens, chunks


def evaluate_intervened_condition_without_grad(
    model: torch.nn.Module,
    target_batch: evolution.NativeFullRowBatch,
    *,
    donor_batch: evolution.NativeFullRowBatch | None,
    rotate_recurrent_layers: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[float, int, Mapping[str, bool]]:
    logits: torch.Tensor | None = None
    try:
        with torch.no_grad():
            logits, audit = checkpointed_intervened_write_read(
                model,
                target_batch,
                donor_batch=donor_batch,
                rotate_recurrent_layers=rotate_recurrent_layers,
                dtype=dtype,
            )
            mean_ce, tokens = contrast.detached_answer_ce(
                logits,
                target_batch.labels,
            )
        return mean_ce, tokens, audit
    finally:
        del logits
        reset_delta_mem_states(model)
        evolution.release_native_row_allocator_cache(device)


def backward_serialized_intervened_condition(
    model: torch.nn.Module,
    target_batch: evolution.NativeFullRowBatch,
    *,
    donor_batch: evolution.NativeFullRowBatch | None,
    rotate_recurrent_layers: bool,
    coefficient: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int, Mapping[str, bool]]:
    logits: torch.Tensor | None = None
    try:
        logits, audit = checkpointed_intervened_write_read(
            model,
            target_batch,
            donor_batch=donor_batch,
            rotate_recurrent_layers=rotate_recurrent_layers,
            dtype=dtype,
        )
        tokens, chunks = backward_logits(
            logits,
            target_batch.labels,
            coefficient=coefficient,
        )
        return tokens, chunks, audit
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


def accumulate_finite_row_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: dict[str, torch.Tensor],
) -> Mapping[str, Any]:
    validation = distributed.validate_local_gradients(named_trainable)
    if validation["non_fp32_gradient_tensors"]:
        raise RuntimeError(f"Causal row gradients are not FP32: {validation!r}")
    if validation["active_gradient_tensors"] == 0:
        raise RuntimeError(f"Causal row has no active gradients: {validation!r}")
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
        raise ValueError("Causal clean-gradient scale must be finite and positive")
    with torch.no_grad():
        for name, parameter in named_trainable:
            gradient = clean_gradients.get(name)
            parameter.grad = None if gradient is None else gradient.mul(scale)


def train(
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
    initial_trainable_sha256 = trainable_subset_sha256(named_trainable)
    for description, value in (
        ("initial adapter", initial_adapter_sha256),
        ("initial recurrent subset", initial_recurrent_sha256),
        ("initial trainable subset", initial_trainable_sha256),
    ):
        distributed.require_consensus(context, value, description=description)

    progress_path = output_dir / "training_progress.jsonl"
    total_correct_ce = 0.0
    total_zero_margin = 0.0
    total_donor_margin = 0.0
    total_permuted_margin = 0.0
    total_active_zero = 0.0
    total_active_donor = 0.0
    total_active_permuted = 0.0
    minimum_gradient_norm = math.inf
    maximum_global_inactive = 0
    first_update_gradient_audit: Mapping[str, Any] | None = None
    projected_carrier_fixed_every_row = True
    total_accepted_gradient_rows = 0
    total_rejected_gradient_rows = 0
    minimum_accepted_rows_per_update = GLOBAL_BATCH_SIZE
    rejected_source_ordinals: list[int] = []
    optimizer_state_cpu_offload_steps = 0
    optimizer_state_cpu_offload_tensors = 0
    optimizer_state_cpu_offload_bytes = 0
    started = time.time()
    for schedule_step in schedule[:updates]:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        if len(local_sources) != LOCAL_ROWS:
            raise RuntimeError("Addressed causal local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        optimizer_state_offload = None
        optimizer_state_restore = None
        if OFFLOAD_OPTIMIZER_STATE_DURING_ROWS and optimizer.state:
            optimizer_state_offload = evolution.move_optimizer_state(
                optimizer,
                device=torch.device("cpu"),
            )
            if optimizer_state_offload.tensors <= 0:
                raise RuntimeError("Causal optimizer state offload moved no tensors")
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
        local_metrics = [0.0] * 14
        clean_gradients: dict[str, torch.Tensor] = {}
        local_row_gradient_evidence: list[dict[str, Any]] = []
        for source_ordinal in local_sources:
            target = rows[source_ordinal].example
            source_offset = schedule_step.source_ordinals.index(source_ordinal)
            donor_ordinal = schedule_step.donor_ordinals[source_offset]
            donor = rows[donor_ordinal].example
            target_batch = evolution.collate_native_examples(
                [target],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            donor_batch = contrast.build_donor_batch(
                target_batch,
                donor,
                device=context.device,
            )
            _, correct_logits = evolution.checkpointed_native_write_read(
                model,
                target_batch,
                dtype=torch.bfloat16,
            )
            donor_logits: torch.Tensor | None = None
            permuted_logits: torch.Tensor | None = None
            if SERIALIZE_CONTROL_BRANCH_GRAPHS:
                donor_ce, donor_tokens, donor_audit = (
                    evaluate_intervened_condition_without_grad(
                        model,
                        target_batch,
                        donor_batch=donor_batch,
                        rotate_recurrent_layers=False,
                        dtype=torch.bfloat16,
                        device=context.device,
                    )
                )
                permuted_ce, permuted_tokens, permuted_audit = (
                    evaluate_intervened_condition_without_grad(
                        model,
                        target_batch,
                        donor_batch=None,
                        rotate_recurrent_layers=True,
                        dtype=torch.bfloat16,
                        device=context.device,
                    )
                )
            else:
                donor_logits, donor_audit = checkpointed_intervened_write_read(
                    model,
                    target_batch,
                    donor_batch=donor_batch,
                    rotate_recurrent_layers=False,
                    dtype=torch.bfloat16,
                )
                permuted_logits, permuted_audit = checkpointed_intervened_write_read(
                    model,
                    target_batch,
                    donor_batch=None,
                    rotate_recurrent_layers=True,
                    dtype=torch.bfloat16,
                )
            zero_ce, zero_tokens = contrast.evaluate_condition_ce(
                model,
                target_batch,
                no_state=True,
                dtype=torch.bfloat16,
            )
            correct_ce, correct_tokens = contrast.detached_answer_ce(
                correct_logits,
                target_batch.labels,
            )
            if not SERIALIZE_CONTROL_BRANCH_GRAPHS:
                if donor_logits is None or permuted_logits is None:
                    raise RuntimeError("Causal control graph construction failed")
                donor_ce, donor_tokens = contrast.detached_answer_ce(
                    donor_logits,
                    target_batch.labels,
                )
                permuted_ce, permuted_tokens = contrast.detached_answer_ce(
                    permuted_logits,
                    target_batch.labels,
                )
            if len({zero_tokens, correct_tokens, donor_tokens, permuted_tokens}) != 1:
                raise RuntimeError("Addressed causal answer token counts differ")
            zero_margin = zero_ce - correct_ce
            donor_margin = donor_ce - correct_ce
            permuted_margin = permuted_ce - correct_ce
            active_zero = zero_margin < MARGIN
            active_donor = donor_margin < MARGIN
            active_permuted = permuted_margin < MARGIN
            positive_coefficient = 1.0 + CONTRAST_WEIGHT * sum(
                (active_zero, active_donor, active_permuted)
            )
            _, correct_chunks = backward_logits(
                correct_logits,
                target_batch.labels,
                coefficient=positive_coefficient,
            )
            chunks = correct_chunks
            backward_carrier_fixed = True
            if SERIALIZE_CONTROL_BRANCH_GRAPHS:
                del correct_logits
                reset_delta_mem_states(model)
                evolution.release_native_row_allocator_cache(context.device)
                if active_donor:
                    donor_backward_tokens, donor_chunks, donor_backward_audit = (
                        backward_serialized_intervened_condition(
                            model,
                            target_batch,
                            donor_batch=donor_batch,
                            rotate_recurrent_layers=False,
                            coefficient=-CONTRAST_WEIGHT,
                            dtype=torch.bfloat16,
                            device=context.device,
                        )
                    )
                    if donor_backward_tokens != donor_tokens:
                        raise RuntimeError(
                            "Serialized donor answer token count changed"
                        )
                    backward_carrier_fixed = bool(
                        backward_carrier_fixed
                        and donor_backward_audit[
                            "projected_carrier_references_fixed"
                        ]
                    )
                    chunks += donor_chunks
                if active_permuted:
                    (
                        permuted_backward_tokens,
                        permuted_chunks,
                        permuted_backward_audit,
                    ) = backward_serialized_intervened_condition(
                        model,
                        target_batch,
                        donor_batch=None,
                        rotate_recurrent_layers=True,
                        coefficient=-CONTRAST_WEIGHT,
                        dtype=torch.bfloat16,
                        device=context.device,
                    )
                    if permuted_backward_tokens != permuted_tokens:
                        raise RuntimeError(
                            "Serialized layer-permuted answer token count changed"
                        )
                    backward_carrier_fixed = bool(
                        backward_carrier_fixed
                        and permuted_backward_audit[
                            "projected_carrier_references_fixed"
                        ]
                    )
                    chunks += permuted_chunks
            else:
                if active_donor:
                    if donor_logits is None:
                        raise RuntimeError("Causal donor graph is missing")
                    _, donor_chunks = backward_logits(
                        donor_logits,
                        target_batch.labels,
                        coefficient=-CONTRAST_WEIGHT,
                    )
                    chunks += donor_chunks
                if active_permuted:
                    if permuted_logits is None:
                        raise RuntimeError("Causal layer-permuted graph is missing")
                    _, permuted_chunks = backward_logits(
                        permuted_logits,
                        target_batch.labels,
                        coefficient=-CONTRAST_WEIGHT,
                    )
                    chunks += permuted_chunks
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
            carrier_fixed = bool(
                donor_audit["projected_carrier_references_fixed"]
                and permuted_audit["projected_carrier_references_fixed"]
                and backward_carrier_fixed
            )
            projected_carrier_fixed_every_row = (
                projected_carrier_fixed_every_row and carrier_fixed
            )
            values = (
                correct_ce,
                zero_ce,
                donor_ce,
                permuted_ce,
                zero_margin,
                donor_margin,
                permuted_margin,
                float(active_zero),
                float(active_donor),
                float(active_permuted),
                float(correct_tokens),
                float(chunks),
                float(carrier_fixed),
                1.0,
            )
            local_metrics = [
                total + value for total, value in zip(local_metrics, values)
            ]
            if not SERIALIZE_CONTROL_BRANCH_GRAPHS:
                del correct_logits
            del target_batch, donor_batch, donor_logits, permuted_logits
            reset_delta_mem_states(model)
            evolution.release_native_row_allocator_cache(context.device)

        metric_tensor = contrast.gate._prepare_distributed_scalar_sums(
            context,
            local_metrics,
        )
        metrics = contrast.gate._distributed_scalar_sums(context, metric_tensor)
        if metrics[13] != GLOBAL_BATCH_SIZE or metrics[12] != GLOBAL_BATCH_SIZE:
            raise RuntimeError("Addressed causal projected-carrier audit failed")
        if updates == PREFLIGHT_UPDATES and any(metrics[index] < 1 for index in (7, 8, 9)):
            raise RuntimeError("Addressed causal preflight found an inactive control family")
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
            if len(all_row_evidence) != GLOBAL_BATCH_SIZE:
                filter_error = RuntimeError("Causal row-filter evidence is incomplete")
            elif accepted_rows < MIN_ACCEPTED_ROWS_PER_UPDATE:
                filter_error = RuntimeError(
                    "Causal row filter accepted too few rows: "
                    f"{accepted_rows} < {MIN_ACCEPTED_ROWS_PER_UPDATE}"
                )
            elif total_rejected_gradient_rows > MAX_TOTAL_REJECTED_ROWS:
                filter_error = RuntimeError(
                    "Causal row filter rejected too many total rows: "
                    f"{total_rejected_gradient_rows} > {MAX_TOTAL_REJECTED_ROWS}"
                )
            distributed.phase_consensus(
                context,
                phase=f"causal-step-{schedule_step.step}-row-filter",
                error=filter_error,
            )
            gradient_rescale = GLOBAL_BATCH_SIZE / accepted_rows
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
            total_accepted_gradient_rows += GLOBAL_BATCH_SIZE
        local_gradient_validation = distributed.validate_local_gradients(named_trainable)
        if local_gradient_validation["passed"] is not True:
            raise RuntimeError(
                "Addressed causal local gradients are invalid: "
                f"{local_gradient_validation!r}"
            )
        collective = distributed.sum_gradients(context, named_trainable)
        inactive = len(collective["global_inactive_parameter_indices"])
        maximum_global_inactive = max(maximum_global_inactive, inactive)
        if inactive:
            raise RuntimeError("Addressed causal optimizer has inactive parameters")
        gradient_audit = None
        if schedule_step.step == 1:
            gradient_audit = FIRST_UPDATE_GRADIENT_AUDITOR(named_trainable)
            if gradient_audit["passed"] is not True:
                raise RuntimeError("Addressed causal first-update gradients are invalid")
            first_update_gradient_audit = gradient_audit
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        gradient_norm_value = float(gradient_norm.detach().float().item())
        if not bool(torch.isfinite(gradient_norm).item()) or gradient_norm_value <= 0.0:
            raise RuntimeError("Addressed causal gradient norm is invalid")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm_value)
        if optimizer_state_offload is not None:
            evolution.release_native_row_allocator_cache(context.device)
            optimizer_state_restore = evolution.move_optimizer_state(
                optimizer,
                device=context.device,
            )
            if optimizer_state_restore != optimizer_state_offload:
                raise RuntimeError("Causal optimizer state restore audit differs")
        optimizer.step()
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "mean_correct_ce": metrics[0] / GLOBAL_BATCH_SIZE,
            "mean_zero_ce": metrics[1] / GLOBAL_BATCH_SIZE,
            "mean_donor_ce": metrics[2] / GLOBAL_BATCH_SIZE,
            "mean_layer_permuted_ce": metrics[3] / GLOBAL_BATCH_SIZE,
            "mean_zero_minus_correct_ce": metrics[4] / GLOBAL_BATCH_SIZE,
            "mean_donor_minus_correct_ce": metrics[5] / GLOBAL_BATCH_SIZE,
            "mean_layer_permuted_minus_correct_ce": metrics[6] / GLOBAL_BATCH_SIZE,
            "active_zero_rows": int(metrics[7]),
            "active_donor_rows": int(metrics[8]),
            "active_layer_permuted_rows": int(metrics[9]),
            "answer_target_tokens": int(metrics[10]),
            "checkpointed_ce_chunks": int(metrics[11]),
            "projected_carrier_fixed_rows": int(metrics[12]),
            "gradient_norm_before_clip": gradient_norm_value,
            "gradient_collective_sha256": canonical_sha256(collective),
            "local_gradient_validation": local_gradient_validation,
            "row_filter": row_filter_evidence,
            "optimizer_state_cpu_offload": {
                "enabled": OFFLOAD_OPTIMIZER_STATE_DURING_ROWS,
                "tensors": (
                    0
                    if optimizer_state_offload is None
                    else optimizer_state_offload.tensors
                ),
                "bytes": (
                    0
                    if optimizer_state_offload is None
                    else optimizer_state_offload.bytes
                ),
                "restored_before_optimizer_step": (
                    optimizer_state_offload is None
                    or optimizer_state_restore == optimizer_state_offload
                ),
            },
            "control_branch_graph_serialization": {
                "enabled": SERIALIZE_CONTROL_BRANCH_GRAPHS,
                "metric_only_forwards_without_grad": (
                    2 * GLOBAL_BATCH_SIZE
                    if SERIALIZE_CONTROL_BRANCH_GRAPHS
                    else 0
                ),
                "active_control_graphs_recreated": (
                    int(metrics[8] + metrics[9])
                    if SERIALIZE_CONTROL_BRANCH_GRAPHS
                    else 0
                ),
                "maximum_simultaneous_autograd_graphs_per_rank": (
                    1 if SERIALIZE_CONTROL_BRANCH_GRAPHS else 3
                ),
            },
            "gradient_audit": gradient_audit,
            "recurrent_gradient_audit": gradient_audit,
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
        }
        if context.is_primary:
            append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "correct_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "zero_margin": round(metrics[4] / GLOBAL_BATCH_SIZE, 6),
                        "donor_margin": round(metrics[5] / GLOBAL_BATCH_SIZE, 6),
                        "permuted_margin": round(metrics[6] / GLOBAL_BATCH_SIZE, 6),
                        "active": [int(metrics[7]), int(metrics[8]), int(metrics[9])],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_correct_ce += metrics[0]
        total_zero_margin += metrics[4]
        total_donor_margin += metrics[5]
        total_permuted_margin += metrics[6]
        total_active_zero += metrics[7]
        total_active_donor += metrics[8]
        total_active_permuted += metrics[9]

    final_state = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = runtime._state_dict_sha256(final_state)
    final_recurrent_sha256 = benchmark_train.state_subset_sha256(
        final_state,
        recurrent_only=True,
    )
    final_trainable_sha256 = trainable_subset_sha256(named_trainable)
    distributed.require_consensus(
        context,
        final_adapter_sha256,
        description="final addressed causal adapter",
    )
    distributed.require_consensus(
        context,
        final_recurrent_sha256,
        description="final addressed causal recurrent subset",
    )
    distributed.require_consensus(
        context,
        final_trainable_sha256,
        description="final addressed causal trainable subset",
    )
    return {
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": updates * GLOBAL_BATCH_SIZE,
        "mean_correct_ce": total_correct_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_zero_minus_correct_ce": total_zero_margin / (
            updates * GLOBAL_BATCH_SIZE
        ),
        "mean_donor_minus_correct_ce": total_donor_margin / (
            updates * GLOBAL_BATCH_SIZE
        ),
        "mean_layer_permuted_minus_correct_ce": total_permuted_margin / (
            updates * GLOBAL_BATCH_SIZE
        ),
        "active_zero_fraction": total_active_zero / (updates * GLOBAL_BATCH_SIZE),
        "active_donor_fraction": total_active_donor / (updates * GLOBAL_BATCH_SIZE),
        "active_layer_permuted_fraction": total_active_permuted
        / (updates * GLOBAL_BATCH_SIZE),
        "minimum_gradient_norm_before_clip": minimum_gradient_norm,
        "maximum_global_inactive_parameter_tensors": maximum_global_inactive,
        "projected_carrier_fixed_every_row": projected_carrier_fixed_every_row,
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
        "optimizer_state_cpu_offload": {
            "enabled": OFFLOAD_OPTIMIZER_STATE_DURING_ROWS,
            "steps": optimizer_state_cpu_offload_steps,
            "maximum_tensors_per_rank": optimizer_state_cpu_offload_tensors,
            "maximum_bytes_per_rank": optimizer_state_cpu_offload_bytes,
            "restored_before_every_optimizer_step": True,
        },
        "control_branch_graph_serialization": {
            "enabled": SERIALIZE_CONTROL_BRANCH_GRAPHS,
            "metric_only_forwards_without_grad": (
                2 * updates * GLOBAL_BATCH_SIZE
                if SERIALIZE_CONTROL_BRANCH_GRAPHS
                else 0
            ),
            "maximum_simultaneous_autograd_graphs_per_rank": (
                1 if SERIALIZE_CONTROL_BRANCH_GRAPHS else 3
            ),
        },
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "initial_recurrent_subset_sha256": initial_recurrent_sha256,
        "final_recurrent_subset_sha256": final_recurrent_sha256,
        "recurrent_subset_changed": (
            initial_recurrent_sha256 != final_recurrent_sha256
        ),
        "initial_trainable_subset_sha256": initial_trainable_sha256,
        "final_trainable_subset_sha256": final_trainable_sha256,
        "trainable_subset_changed": (
            initial_trainable_sha256 != final_trainable_sha256
        ),
        "first_update_gradient_audit": first_update_gradient_audit,
        "first_update_recurrent_gradient_audit": first_update_gradient_audit,
        "progress_sha256": sha256_file(progress_path) if context.is_primary else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Addressed causal training requires exactly four ranks")
    if updates not in (PREFLIGHT_UPDATES, TRAIN_UPDATES):
        raise ValueError("Addressed causal updates must be 1 or 8")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    calibration_result = validate_calibration_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Addressed causal pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Addressed causal output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="addressed-causal-output-freshness",
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
        phase="addressed-causal-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, delta_config, named_trainable, model_audit = load_model(
        base_model,
        device=context.device,
    )
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    donor_mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    schedule, schedule_payload = contrast.build_schedule(
        rows,
        donor_mapping,
        donor_deltas,
    )
    if (
        canonical_sha256(schedule_payload) != contrast.FULL_SCHEDULE_SHA256
        or canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256
    ):
        raise RuntimeError("Addressed causal schedule binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": calibration_result["status"],
        "selected_candidate": SELECTED_CANDIDATE,
        "seed": SEED,
        "updates": updates,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "training_schedule_sha256": canonical_sha256(schedule_payload),
        "schedule_prefix_sha256": canonical_sha256(schedule_payload[:updates]),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="addressed causal input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="addressed-causal-input-binding-save",
        error=binding_error,
    )
    training = train(
        model,
        rows,
        schedule,
        updates=updates,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
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
        screen.four_distinct_a100s(context.rank_devices)
        and training["initial_adapter_sha256"] != training["final_adapter_sha256"]
        and training["recurrent_subset_changed"] is True
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and training["projected_carrier_fixed_every_row"] is True
        and training["first_update_recurrent_gradient_audit"]["passed"] is True
    )
    save_error: BaseException | None = None
    result: dict[str, Any] = {}
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
                    else "training_complete_open_evaluation_authorized"
                    if passed
                    else "training_failed_evaluation_blocked"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "seed": SEED,
                "updates": updates,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_native_evaluation_authorized": (
                    passed and updates == TRAIN_UPDATES
                ),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "calibration_result_file_sha256": sha256_file(
                        CALIBRATION_RESULT
                    ),
                    "contrast_runner_sha256": sha256_file(Path(contrast.__file__)),
                    "distributed_sha256": sha256_file(Path(distributed.__file__)),
                    "delta_impl_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                    "rwkv_write_scan_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/kernels/rwkv_ms_write_scan.py"
                    ),
                    "rwkv_write_scan_cuda_sha256": sha256_file(
                        PROJECT_ROOT / "deltamem/kernels/rwkv_ms_write_scan_cuda.cu"
                    ),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="addressed-causal-result-save",
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
    parser.add_argument("--updates", type=int, required=True, choices=(1, 8))
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Addressed causal training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
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
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
