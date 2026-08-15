#!/usr/bin/env python3
"""Run the locked four-GPU BF16 materiality gate for recurrent RWKV-MS."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as torch_dist
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
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_bf16_calibration.v1"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DATASET_ROOT = Path(
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_development_v1"
)
WORLD_SIZE = 4
SEED = 57
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0
MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA = 1e-3
CALIBRATION_SOURCE_ORDINALS = (718, 1149, 918, 76)
CALIBRATION_ROWS_PAYLOAD_SHA256 = (
    "b77caeb963647569648244a8927de9693347bb159de48f5a21d59738169cfe5a"
)
RECURRENT_READOUT_SUFFIX = ".hrm_rwkv7_core.output.weight"


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def calibration_rows(
    rows: Sequence[contrast.SceneContrastRow],
) -> tuple[
    tuple[contrast.SceneContrastRow, ...],
    tuple[contrast.SceneContrastRow, ...],
    list[dict[str, Any]],
]:
    if len(CALIBRATION_SOURCE_ORDINALS) != WORLD_SIZE:
        raise ValueError("Calibration requires exactly one source row per rank")
    mapping, deltas, _ = contrast.build_donor_mapping(rows)
    sources: list[contrast.SceneContrastRow] = []
    donors: list[contrast.SceneContrastRow] = []
    payload: list[dict[str, Any]] = []
    for rank, source_ordinal in enumerate(CALIBRATION_SOURCE_ORDINALS):
        donor_ordinal = mapping[source_ordinal]
        source_row = rows[source_ordinal]
        donor_row = rows[donor_ordinal]
        if source_row.assistant_identity == donor_row.assistant_identity:
            raise ValueError("Calibration donor must have a different gold answer")
        sources.append(source_row)
        donors.append(donor_row)
        payload.append(
            {
                "rank": rank,
                "source_ordinal": source_ordinal,
                "source_row_sha256": source_row.example.row_sha256,
                "source_write_tokens": len(source_row.example.write_input_ids),
                "source_read_tokens": len(source_row.example.read_input_ids),
                "donor_ordinal": donor_ordinal,
                "donor_row_sha256": donor_row.example.row_sha256,
                "donor_write_tokens": len(donor_row.example.write_input_ids),
                "absolute_write_token_delta": deltas[source_ordinal],
            }
        )
    if canonical_sha256(payload) != CALIBRATION_ROWS_PAYLOAD_SHA256:
        raise ValueError("Calibration source/donor row binding differs")
    return tuple(sources), tuple(donors), payload


def materiality_metrics(
    correct_logits: torch.Tensor,
    comparison_logits: torch.Tensor,
) -> Mapping[str, Any]:
    if correct_logits.shape != comparison_logits.shape or not correct_logits.numel():
        raise ValueError("Calibration condition logits are misaligned")
    difference = (correct_logits.float() - comparison_logits.float()).abs()
    maximum = float(difference.max().item())
    return {
        "max_abs_logit_delta": maximum,
        "mean_abs_logit_delta": float(difference.mean().item()),
        "changed_logit_fraction": float(
            difference.ne(0).sum().item() / difference.numel()
        ),
        "materiality_minimum": MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA,
        "passed": maximum >= MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA,
    }


def audit_recurrent_readout_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    selected = [
        (name, parameter)
        for name, parameter in named_trainable
        if name.endswith(RECURRENT_READOUT_SUFFIX)
    ]
    rows: list[dict[str, Any]] = []
    for name, parameter in selected:
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        norm = (
            0.0
            if gradient is None or not finite
            else float(gradient.detach().float().norm().item())
        )
        rows.append(
            {
                "name": name,
                "gradient_present": gradient is not None,
                "gradient_finite": finite,
                "gradient_l2_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        )
    passed = (
        len(rows) == preflight.EXPECTED_LAYERS
        and all(row["gradient_finite"] for row in rows)
        and all(row["gradient_nonzero"] for row in rows)
    )
    return {
        "parameter_family": "hrm_rwkv7_core.output.weight",
        "parameter_tensors": len(rows),
        "parameter_names_sha256": canonical_sha256(
            [row["name"] for row in rows]
        ),
        "minimum_l2_norm": min(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "maximum_l2_norm": max(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "all_42_finite_nonzero": passed,
        "layers": rows,
        "passed": passed,
    }


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
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
    replaced = attach_delta_mem(model, preflight.build_config())
    trainable_names = freeze_non_delta_mem_params(model)
    runtime._promote_trainable_parameters_to_fp32(model)
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    audit = {
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_names_sha256": canonical_sha256(
            sorted(trainable_names)
        ),
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "backbone_dtype": "bfloat16",
        "trainable_master_dtype": "float32",
        "all_wrappers_recurrent_delta_readout": all(
            module.memory_backend == "rwkv_ms"
            and module.memory_readout_mode == "delta"
            for _, module in modules
        ),
    }
    if (
        len(replaced) != preflight.EXPECTED_LAYERS
        or len(modules) != preflight.EXPECTED_LAYERS
        or not audit["all_wrappers_recurrent_delta_readout"]
    ):
        raise RuntimeError(f"Calibration recurrent attachment failed: {audit!r}")
    return model, tokenizer, audit


def write_read_logits(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    reset_delta_mem_states(model)
    set_delta_mem_write_enabled(model, True)
    with runtime._autocast_context(batch.write_input_ids.device, dtype):
        model(
            input_ids=batch.write_input_ids,
            attention_mask=batch.write_attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    set_delta_mem_write_enabled(model, False)
    predictor_indices = runtime._answer_predictor_indices(batch.labels)
    with runtime._autocast_context(batch.read_input_ids.device, dtype):
        outputs = model(
            input_ids=batch.read_input_ids,
            attention_mask=batch.read_attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=predictor_indices,
        )
    return outputs.logits


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    reset_delta_mem_states(model)
    set_delta_mem_write_enabled(model, True)
    with torch.inference_mode(), runtime._autocast_context(
        input_ids.device, torch.bfloat16
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    state = get_delta_mem_online_state(model)
    if not state:
        raise RuntimeError("Calibration write produced no recurrent online state")
    return state


def read_logits(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    state: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
    reset_delta_mem_states(model)
    if state is not None:
        load_delta_mem_online_state(model, state)
    set_delta_mem_write_enabled(model, False)
    predictor_indices = runtime._answer_predictor_indices(batch.labels)
    with torch.inference_mode(), runtime._autocast_context(
        batch.read_input_ids.device, torch.bfloat16
    ):
        outputs = model(
            input_ids=batch.read_input_ids,
            attention_mask=batch.read_attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=predictor_indices,
        )
    return outputs.logits.detach().float().cpu()


def adapter_sha256(model: torch.nn.Module) -> str:
    return runtime._state_dict_sha256(snapshot_delta_mem_weights(model))


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("BF16 recurrent calibration requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = preflight.validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Pinned Gemma base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Calibration output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="recurrent-bf16-calibration-output-freshness",
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
        phase="recurrent-bf16-calibration-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    sources, donors, row_payload = calibration_rows(rows)
    source = sources[context.process_rank]
    donor = donors[context.process_rank]
    batch = evolution.collate_native_examples(
        [source.example],
        pad_token_id=int(tokenizer.pad_token_id),
        device=context.device,
    )
    donor_batch = contrast.build_donor_batch(
        batch,
        donor.example,
        device=context.device,
    )
    named_trainable = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    initial_adapter_sha256 = adapter_sha256(model)
    distributed.require_consensus(
        context,
        initial_adapter_sha256,
        description="recurrent calibration initial adapter state",
    )

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = write_read_logits(model, batch, dtype=torch.bfloat16)
    answer_loss_sum, local_answer_tokens = distributed.answer_loss_sum_and_count(
        logits,
        batch.labels,
    )
    global_tokens_tensor = torch.tensor(
        [local_answer_tokens],
        device=context.device,
        dtype=torch.long,
    )
    torch_dist.all_reduce(global_tokens_tensor, op=torch_dist.ReduceOp.SUM)
    global_answer_tokens = int(global_tokens_tensor.item())
    if global_answer_tokens <= 0:
        raise RuntimeError("Calibration global answer-token count is invalid")
    loss = answer_loss_sum / global_answer_tokens
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Calibration loss is non-finite")
    loss.backward()
    local_gradient_validation = distributed.validate_local_gradients(
        named_trainable
    )
    if local_gradient_validation["passed"] is not True:
        raise RuntimeError(
            f"Calibration local gradient validation failed: {local_gradient_validation!r}"
        )
    collective = distributed.sum_gradients(context, named_trainable)
    recurrent_gradient_audit = audit_recurrent_readout_gradients(named_trainable)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_trainable],
        MAX_GRAD_NORM,
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError("Calibration gradient norm is non-finite")
    optimizer.step()
    post_update_adapter_sha256 = adapter_sha256(model)
    distributed.require_consensus(
        context,
        post_update_adapter_sha256,
        description="recurrent calibration post-update adapter state",
    )

    model.eval()
    correct_state = write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_state = write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    correct_logits = read_logits(model, batch, correct_state)
    zero_logits = read_logits(model, batch, None)
    donor_logits = read_logits(model, batch, donor_state)
    correct_zero = materiality_metrics(correct_logits, zero_logits)
    correct_donor = materiality_metrics(correct_logits, donor_logits)
    local_evidence = {
        **row_payload[context.process_rank],
        "answer_target_tokens": local_answer_tokens,
        "local_loss_sum": float(answer_loss_sum.detach().float().item()),
        "local_gradient_validation": local_gradient_validation,
        "post_update_correct_vs_zero": correct_zero,
        "post_update_correct_vs_matched_donor": correct_donor,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)
    checks = {
        "four_distinct_a100_ranks": (
            len(context.rank_devices) == WORLD_SIZE
            and all("A100" in str(device["device_name"]) for device in context.rank_devices)
        ),
        "all_42_recurrent_readout_gradients_finite_nonzero": (
            recurrent_gradient_audit["passed"] is True
        ),
        "optimizer_update_changed_adapter": (
            initial_adapter_sha256 != post_update_adapter_sha256
        ),
        "post_update_correct_vs_zero_material_on_all_ranks": all(
            evidence["post_update_correct_vs_zero"]["passed"]
            for evidence in rank_evidence
        ),
        "post_update_correct_vs_matched_donor_material_on_all_ranks": all(
            evidence["post_update_correct_vs_matched_donor"]["passed"]
            for evidence in rank_evidence
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "calibration_passed_benchmark_training_authorized"
            if passed
            else "calibration_failed_benchmark_training_blocked"
        ),
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": preflight.PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "calibration_rows_payload_sha256": CALIBRATION_ROWS_PAYLOAD_SHA256,
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "learning_rate": LEARNING_RATE,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "max_grad_norm": MAX_GRAD_NORM,
        "logical_global_batch_rows": WORLD_SIZE,
        "optimizer_updates": 1,
        "global_answer_tokens": global_answer_tokens,
        "model_audit": model_audit,
        "initial_adapter_sha256": initial_adapter_sha256,
        "post_update_adapter_sha256": post_update_adapter_sha256,
        "gradient_norm_before_clip": float(gradient_norm.detach().float().item()),
        "gradient_collective": collective,
        "recurrent_readout_gradient_audit": recurrent_gradient_audit,
        "rank_devices": list(context.rank_devices),
        "rank_evidence": list(rank_evidence),
        "materiality_minimum": MIN_POST_UPDATE_MAX_ABS_LOGIT_DELTA,
        "benchmark_training_authorized": passed,
        "failure_rule": (
            "The three-seed matched native benchmark remains blocked unless this "
            "calibration passes every check."
        ),
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "protocol_file_sha256": sha256_file(preflight.PROTOCOL),
            "preflight_runner_sha256": sha256_file(Path(preflight.__file__)),
            "delta_impl_sha256": sha256_file(PROJECT_ROOT / "deltamem/core/delta_impl.py"),
            "rwkv_core_sha256": sha256_file(PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"),
            "distributed_sha256": sha256_file(Path(distributed.__file__)),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-bf16-calibration-result-save",
        error=save_error,
    )
    del model, optimizer, batch, donor_batch, rows
    gc.collect()
    torch.cuda.empty_cache()
    return result


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
        raise ValueError("BF16 recurrent calibration requires four-rank torchrun")
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
                "result_receipt": result["receipt"]["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
