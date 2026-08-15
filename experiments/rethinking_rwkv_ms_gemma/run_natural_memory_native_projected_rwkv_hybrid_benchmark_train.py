#!/usr/bin/env python3
"""Train one locked arm of the paired projected-KV/RWKV native benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    attach_delta_mem,
    freeze_non_delta_mem_params,
    iter_delta_mem_modules,
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
    run_natural_memory_native_projected_rwkv_hybrid_bf16_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as recurrent_preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_projected_rwkv_hybrid_benchmark_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "b782fdf56386402e0a6f1147b5a9e5e608e1d231c770e174499a63f8e81d3dfd"
)
CALIBRATION_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_projected_rwkv_hybrid_bf16_calibration_v1/result.json"
)
CALIBRATION_RESULT_FILE_SHA256 = (
    "1d7808f4760a47c40923ee61668cf40b8033629e1399f8e5cee618cd1b6ee5a0"
)
CALIBRATION_RESULT_RECEIPT = (
    "5f48f8d67a2593a2e98d7a64a15b41c1979d627d3ddd7c60e0a4b6648053627d"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
ARCHITECTURES = ("projected_control", "hybrid_candidate")
SEEDS = (57, 58, 59)
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 8
LOCAL_ROWS = 2
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 32
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
RECURRENT_PARAMETER_MARKERS = (
    ".hrm_rwkv7_core.",
    ".beta_proj",
    ".beta_bias",
    ".lambda_proj",
    ".lambda_bias",
)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return recurrent_preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Matched benchmark output must be fresh: {path}")
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
        raise ValueError("Matched hybrid benchmark protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Matched hybrid benchmark protocol payload hash differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "calibration_protocol_payload_sha256": calibration.PROTOCOL_PAYLOAD_SHA256,
        "calibration_result_file": (
            "local_artifacts/"
            "natural_memory_native_projected_rwkv_hybrid_bf16_calibration_v1/"
            "result.json"
        ),
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": "calibration_passed_native_benchmark_authorized",
        "selected_candidate": calibration.SELECTED_CANDIDATE,
    }
    if authorization != required_authorization:
        raise ValueError("Matched hybrid benchmark authorization binding differs")
    training = protocol.get("training", {})
    expected_training = {
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "seeds": list(SEEDS),
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "global_batch_rows": GLOBAL_BATCH_SIZE,
        "local_rows_per_rank": LOCAL_ROWS,
        "optimizer_updates": TRAIN_UPDATES,
        "contrast_weight": CONTRAST_WEIGHT,
        "contrast_margin": MARGIN,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("Matched hybrid benchmark training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Matched hybrid benchmark may not authorize protected data")
    return protocol


def validate_calibration_result() -> Mapping[str, Any]:
    if sha256_file(CALIBRATION_RESULT) != CALIBRATION_RESULT_FILE_SHA256:
        raise ValueError("Matched benchmark calibration result file hash differs")
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Matched benchmark calibration result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "schema": calibration.SCHEMA,
        "status": "calibration_passed_native_benchmark_authorized",
        "passed": True,
        "native_benchmark_authorized": True,
        "protocol_payload_sha256": calibration.PROTOCOL_PAYLOAD_SHA256,
        "protected_splits_opened": [],
    }
    if (
        digest != CALIBRATION_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(key) != value for key, value in required.items())
    ):
        raise ValueError("Matched benchmark calibration did not authorize training")
    return result


def build_config(architecture: str):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown matched benchmark architecture: {architecture}")
    return replace(
        calibration.build_config(),
        memory_readout_mode=(
            "projected_kv_slots"
            if architecture == "projected_control"
            else "projected_kv_rwkv_hybrid"
        ),
    )


def is_recurrent_only_parameter(name: str) -> bool:
    return any(marker in name for marker in RECURRENT_PARAMETER_MARKERS)


def configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    architecture: str,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    freeze_non_delta_mem_params(model)
    recurrent_frozen: list[str] = []
    if architecture == "projected_control":
        for name, parameter in model.named_parameters():
            if is_recurrent_only_parameter(name):
                parameter.requires_grad_(False)
                recurrent_frozen.append(name)
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    selected_names = [name for name, _ in selected]
    recurrent_selected = [name for name in selected_names if is_recurrent_only_parameter(name)]
    passed = (
        bool(selected)
        and (
            (architecture == "projected_control" and not recurrent_selected)
            or (
                architecture == "hybrid_candidate"
                and len(
                    [
                        name
                        for name in recurrent_selected
                        if name.endswith(recurrent_calibration.RECURRENT_READOUT_SUFFIX)
                    ]
                )
                == recurrent_preflight.EXPECTED_LAYERS
            )
        )
    )
    audit = {
        "architecture": architecture,
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": canonical_sha256(selected_names),
        "recurrent_only_trainable_tensors": len(recurrent_selected),
        "recurrent_only_trainable_names_sha256": canonical_sha256(recurrent_selected),
        "recurrent_only_frozen_tensors": len(recurrent_frozen),
        "recurrent_only_frozen_names_sha256": canonical_sha256(sorted(recurrent_frozen)),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Matched benchmark trainable isolation failed: {audit!r}")
    return selected, audit


def state_subset_sha256(
    state: Mapping[str, torch.Tensor],
    *,
    recurrent_only: bool,
) -> str:
    selected = {
        name: tensor
        for name, tensor in state.items()
        if is_recurrent_only_parameter(name) == recurrent_only
    }
    if not selected:
        raise ValueError("Matched benchmark adapter state subset is empty")
    return runtime._state_dict_sha256(selected)


def load_model(
    base_model: Path,
    *,
    architecture: str,
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
    delta_config = build_config(architecture)
    replaced = attach_delta_mem(model, delta_config)
    named_trainable, trainable_audit = configure_trainable_parameters(
        model,
        architecture=architecture,
    )
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    expected_readout = (
        "projected_kv_slots"
        if architecture == "projected_control"
        else "projected_kv_rwkv_hybrid"
    )
    wrappers_valid = all(
        module.memory_backend == "rwkv_ms"
        and module.memory_readout_mode == expected_readout
        and module.rwkv_ms_write_mode == "recurrent"
        and module.rwkv_ms_hybrid_mode == "scalar_gate"
        and module.rwkv_ms_hybrid_gain == 0.03125
        for _, module in modules
    )
    audit = {
        "architecture": architecture,
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "expected_readout": expected_readout,
        "all_wrappers_valid": wrappers_valid,
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "backbone_dtype": "bfloat16",
        "trainable_master_dtype": "float32",
        "trainables": trainable_audit,
    }
    if (
        len(replaced) != recurrent_preflight.EXPECTED_LAYERS
        or len(modules) != recurrent_preflight.EXPECTED_LAYERS
        or not wrappers_valid
    ):
        raise RuntimeError(f"Matched benchmark attachment failed: {audit!r}")
    return model, tokenizer, delta_config, named_trainable, audit


def four_distinct_a100s(rank_devices: Sequence[Mapping[str, Any]]) -> bool:
    return (
        len(rank_devices) == WORLD_SIZE
        and len({str(device.get("device_uuid")) for device in rank_devices}) == WORLD_SIZE
        and all("A100" in str(device.get("device_name")) for device in rank_devices)
    )


def train(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[contrast.ContrastScheduleStep],
    *,
    architecture: str,
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
    initial_shared_sha256 = state_subset_sha256(initial_state, recurrent_only=False)
    initial_recurrent_sha256 = state_subset_sha256(initial_state, recurrent_only=True)
    initial_recurrent_readout_sha256 = (
        calibration.recurrent_readout_sha256(named_trainable)
        if architecture == "hybrid_candidate"
        else None
    )
    for description, value in (
        ("initial full adapter", initial_adapter_sha256),
        ("initial shared adapter", initial_shared_sha256),
        ("initial recurrent subset", initial_recurrent_sha256),
    ):
        distributed.require_consensus(context, value, description=description)

    progress_path = output_dir / "training_progress.jsonl"
    total_positive_ce = 0.0
    total_donor_ce = 0.0
    total_margin = 0.0
    total_active = 0.0
    minimum_gradient_norm = math.inf
    maximum_global_inactive_parameter_tensors = 0
    first_recurrent_gradient_audit: Mapping[str, Any] | None = None
    started = time.time()
    for schedule_step in schedule[:updates]:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        if len(local_sources) != LOCAL_ROWS:
            raise RuntimeError("Matched benchmark local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        local_positive_ce = 0.0
        local_donor_ce = 0.0
        local_margin = 0.0
        local_active = 0.0
        local_dropped = 0.0
        local_correct = 0.0
        local_positive_tokens = 0.0
        local_donor_tokens = 0.0
        local_ce_chunks = 0.0
        local_occupied = 0.0
        for source_ordinal in local_sources:
            target = rows[source_ordinal].example
            source_offset = schedule_step.source_ordinals.index(source_ordinal)
            donor_ordinal = schedule_step.donor_ordinals[source_offset]
            donor = rows[donor_ordinal].example
            no_state = source_ordinal in schedule_step.no_state_ordinals
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
            evolution.release_native_row_allocator_cache(context.device)
            positive_probe_ce, positive_tokens = contrast.evaluate_condition_ce(
                model,
                target_batch,
                no_state=no_state,
                dtype=torch.bfloat16,
            )
            evolution.release_native_row_allocator_cache(context.device)
            donor_probe_ce, donor_tokens = contrast.evaluate_condition_ce(
                model,
                donor_batch,
                no_state=False,
                dtype=torch.bfloat16,
            )
            margin_value = donor_probe_ce - positive_probe_ce
            active = margin_value < MARGIN
            positive_coefficient = 1.0 + (CONTRAST_WEIGHT if active else 0.0)
            evolution.release_native_row_allocator_cache(context.device)
            _, chunks, occupancy = contrast.backward_condition(
                model,
                target_batch,
                no_state=no_state,
                coefficient=positive_coefficient,
                dtype=torch.bfloat16,
            )
            local_ce_chunks += chunks
            local_occupied += occupancy
            if active:
                evolution.release_native_row_allocator_cache(context.device)
                donor_train_ce, chunks, occupancy = contrast.backward_condition(
                    model,
                    donor_batch,
                    no_state=False,
                    coefficient=-CONTRAST_WEIGHT,
                    dtype=torch.bfloat16,
                )
                if not math.isfinite(donor_train_ce):
                    raise RuntimeError("Matched benchmark donor train CE is non-finite")
                local_ce_chunks += chunks
                local_occupied += occupancy
            local_positive_ce += positive_probe_ce
            local_donor_ce += donor_probe_ce
            local_margin += margin_value
            local_active += float(active)
            local_dropped += float(no_state)
            local_correct += float(not no_state)
            local_positive_tokens += positive_tokens
            local_donor_tokens += donor_tokens
            del target_batch, donor_batch
            evolution.release_native_row_allocator_cache(context.device)

        scalar_tensor = contrast.gate._prepare_distributed_scalar_sums(
            context,
            (
                local_positive_ce,
                local_donor_ce,
                local_margin,
                local_active,
                local_dropped,
                local_correct,
                local_positive_tokens,
                local_donor_tokens,
                local_ce_chunks,
                local_occupied,
            ),
        )
        metrics = contrast.gate._distributed_scalar_sums(context, scalar_tensor)
        if metrics[4] != 2 or metrics[5] != 6:
            raise RuntimeError("Matched benchmark state-dropout balance differs")
        if updates == PREFLIGHT_UPDATES and metrics[3] < 1:
            raise RuntimeError("Matched benchmark preflight found no active hinge")
        local_gradient_validation = distributed.validate_local_gradients(named_trainable)
        if local_gradient_validation["passed"] is not True:
            raise RuntimeError("Matched benchmark produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        global_inactive_parameter_tensors = len(
            collective["global_inactive_parameter_indices"]
        )
        maximum_global_inactive_parameter_tensors = max(
            maximum_global_inactive_parameter_tensors,
            global_inactive_parameter_tensors,
        )
        if global_inactive_parameter_tensors:
            raise RuntimeError(
                "Matched benchmark optimizer contains globally inactive parameters"
            )
        recurrent_gradient_audit = None
        if architecture == "hybrid_candidate" and schedule_step.step == 1:
            recurrent_gradient_audit = (
                recurrent_calibration.audit_recurrent_readout_gradients(named_trainable)
            )
            if recurrent_gradient_audit["passed"] is not True:
                raise RuntimeError(
                    "Matched hybrid first update did not activate every recurrent readout"
                )
            first_recurrent_gradient_audit = recurrent_gradient_audit
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        grad_norm_value = float(grad_norm.detach().float().item())
        if not bool(torch.isfinite(grad_norm).item()) or grad_norm_value <= 0.0:
            raise RuntimeError("Matched benchmark gradient norm is invalid")
        minimum_gradient_norm = min(minimum_gradient_norm, grad_norm_value)
        optimizer.step()
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "architecture": architecture,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "world_size": context.world_size,
            "correct_state_rows": int(metrics[5]),
            "no_state_rows": int(metrics[4]),
            "active_hinge_rows": int(metrics[3]),
            "mean_positive_probe_ce": metrics[0] / GLOBAL_BATCH_SIZE,
            "mean_donor_probe_ce": metrics[1] / GLOBAL_BATCH_SIZE,
            "mean_donor_minus_positive_ce": metrics[2] / GLOBAL_BATCH_SIZE,
            "positive_target_tokens": int(metrics[6]),
            "donor_target_tokens": int(metrics[7]),
            "checkpointed_ce_chunks": int(metrics[8]),
            "written_condition_occupancy_rows": int(metrics[9]),
            "gradient_norm_before_clip": grad_norm_value,
            "gradient_collective_sha256": canonical_sha256(collective),
            "local_gradient_validation": local_gradient_validation,
            "recurrent_gradient_audit": recurrent_gradient_audit,
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
            "no_state_ordinals": sorted(schedule_step.no_state_ordinals),
        }
        if context.is_primary:
            append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "architecture": architecture,
                        "step": schedule_step.step,
                        "positive_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "donor_ce": round(metrics[1] / GLOBAL_BATCH_SIZE, 6),
                        "margin": round(metrics[2] / GLOBAL_BATCH_SIZE, 6),
                        "active": int(metrics[3]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_positive_ce += metrics[0]
        total_donor_ce += metrics[1]
        total_margin += metrics[2]
        total_active += metrics[3]

    final_state = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = runtime._state_dict_sha256(final_state)
    final_shared_sha256 = state_subset_sha256(final_state, recurrent_only=False)
    final_recurrent_sha256 = state_subset_sha256(final_state, recurrent_only=True)
    final_recurrent_readout_sha256 = (
        calibration.recurrent_readout_sha256(named_trainable)
        if architecture == "hybrid_candidate"
        else None
    )
    for description, value in (
        ("final full adapter", final_adapter_sha256),
        ("final shared adapter", final_shared_sha256),
        ("final recurrent subset", final_recurrent_sha256),
    ):
        distributed.require_consensus(context, value, description=description)
    if final_adapter_sha256 == initial_adapter_sha256:
        raise RuntimeError("Matched benchmark training did not change the adapter")
    recurrent_changed = final_recurrent_sha256 != initial_recurrent_sha256
    if architecture == "projected_control" and recurrent_changed:
        raise RuntimeError("Projected control changed its frozen recurrent subset")
    if architecture == "hybrid_candidate" and (
        not recurrent_changed
        or initial_recurrent_readout_sha256 == final_recurrent_readout_sha256
    ):
        raise RuntimeError("Matched hybrid did not change recurrent RWKV weights")
    return {
        "architecture": architecture,
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": updates * GLOBAL_BATCH_SIZE,
        "mean_positive_probe_ce": total_positive_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_probe_ce": total_donor_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_minus_positive_ce": total_margin / (updates * GLOBAL_BATCH_SIZE),
        "active_hinge_fraction": total_active / (updates * GLOBAL_BATCH_SIZE),
        "minimum_gradient_norm_before_clip": minimum_gradient_norm,
        "maximum_global_inactive_parameter_tensors": (
            maximum_global_inactive_parameter_tensors
        ),
        "initial_adapter_sha256": initial_adapter_sha256,
        "final_adapter_sha256": final_adapter_sha256,
        "initial_shared_adapter_sha256": initial_shared_sha256,
        "final_shared_adapter_sha256": final_shared_sha256,
        "initial_recurrent_subset_sha256": initial_recurrent_sha256,
        "final_recurrent_subset_sha256": final_recurrent_sha256,
        "recurrent_subset_changed": recurrent_changed,
        "initial_recurrent_readout_sha256": initial_recurrent_readout_sha256,
        "final_recurrent_readout_sha256": final_recurrent_readout_sha256,
        "first_update_recurrent_gradient_audit": first_recurrent_gradient_audit,
        "progress_sha256": sha256_file(progress_path) if context.is_primary else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    architecture: str,
    seed: int,
    updates: int,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Matched hybrid benchmark training requires exactly four ranks")
    if architecture not in ARCHITECTURES or seed not in SEEDS:
        raise ValueError("Matched hybrid benchmark architecture or seed is not locked")
    if updates not in (PREFLIGHT_UPDATES, TRAIN_UPDATES):
        raise ValueError("Matched hybrid benchmark updates must be 1 or 32")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    calibration_result = validate_calibration_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != recurrent_preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Matched benchmark pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Matched benchmark output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="matched-hybrid-output-freshness",
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
        phase="matched-hybrid-output-creation",
        error=creation_error,
    )

    set_seed(seed)
    model, tokenizer, delta_config, named_trainable, model_audit = load_model(
        base_model,
        architecture=architecture,
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
        or canonical_sha256(
            [rows[index].example.row_sha256 for step in schedule for index in step.source_ordinals]
        )
        != contrast.SELECTED_ROWS_SHA256
    ):
        raise RuntimeError("Matched benchmark locked schedule binding differs")
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "calibration_result_file_sha256": CALIBRATION_RESULT_FILE_SHA256,
        "calibration_result_receipt": CALIBRATION_RESULT_RECEIPT,
        "calibration_status": calibration_result["status"],
        "architecture": architecture,
        "seed": seed,
        "updates": updates,
        "world_size": context.world_size,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "base_model": str(base_model),
        "base_config_sha256": recurrent_preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "training_selected_rows_sha256": contrast.SELECTED_ROWS_SHA256,
        "training_schedule_sha256": contrast.FULL_SCHEDULE_SHA256,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "runner_sha256": sha256_file(Path(__file__)),
        "protected_splits_opened": [],
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="matched hybrid input binding",
    )
    binding_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            binding_error = error
    distributed.phase_consensus(
        context,
        phase="matched-hybrid-input-binding-save",
        error=binding_error,
    )
    training = train(
        model,
        rows,
        schedule,
        architecture=architecture,
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
        four_distinct_a100s(context.rank_devices)
        and training["initial_adapter_sha256"] != training["final_adapter_sha256"]
        and training["maximum_global_inactive_parameter_tensors"] == 0
        and (
            (
                architecture == "projected_control"
                and training["recurrent_subset_changed"] is False
            )
            or (
                architecture == "hybrid_candidate"
                and training["recurrent_subset_changed"] is True
                and training["first_update_recurrent_gradient_audit"]["passed"]
                is True
            )
        )
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
                    "preflight_passed" if updates == PREFLIGHT_UPDATES and passed
                    else "training_complete_evaluation_pending" if passed
                    else "training_failed"
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "architecture": architecture,
                "seed": seed,
                "updates": updates,
                "input_binding": input_binding,
                "training": training,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "protocol_file_sha256": sha256_file(PROTOCOL),
                    "calibration_result_file_sha256": sha256_file(CALIBRATION_RESULT),
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
        phase="matched-hybrid-result-save",
        error=save_error,
    )
    del model, rows
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": passed,
        "architecture": architecture,
        "seed": seed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--updates",
        type=int,
        choices=(PREFLIGHT_UPDATES, TRAIN_UPDATES),
        default=PREFLIGHT_UPDATES,
    )
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(
        args.device,
        required_world_size=WORLD_SIZE,
        timeout_seconds=1800,
    )
    if context is None:
        raise ValueError("Matched hybrid benchmark training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            architecture=args.architecture,
            seed=args.seed,
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
                "architecture": args.architecture,
                "seed": args.seed,
                "updates": args.updates,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": (
                    result.get("receipt", {}).get("payload_sha256")
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
