#!/usr/bin/env python3
"""Run the locked one-update BF16 calibration for the selected RWKV hybrid."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    iter_delta_mem_modules,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
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


SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_bf16_calibration.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_projected_rwkv_hybrid_bf16_calibration_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "5039c320c96356495b2ebfbf0dacfc79299d60224b8b772883eecf300266b3da"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_projected_rwkv_hybrid_screen_v1/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "b0db9ad28249c274135ded09f2ade23b79750bc83882290b0a6a002585e29b36"
)
SCREEN_RESULT_RECEIPT = (
    "627821345ae6a9f41c6274aacec5ce10e3f34883fc4165de19685483f624edc5"
)
SELECTED_CANDIDATE = {
    "candidate_id": "scalar_gate_g003125",
    "hybrid_mode": "scalar_gate",
    "hybrid_gain": 0.03125,
}
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = recurrent_calibration.BASE_MODEL
DATASET_ROOT = recurrent_calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 57
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Hybrid BF16 calibration protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Hybrid BF16 calibration protocol payload hash differs")
    authorization = protocol.get("authorization_basis", {})
    expected_authorization = {
        "screen_protocol_payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
        "screen_result_file": (
            "local_artifacts/"
            "natural_memory_native_projected_rwkv_hybrid_screen_v1/result.json"
        ),
        "screen_result_file_sha256": SCREEN_RESULT_FILE_SHA256,
        "screen_result_receipt": SCREEN_RESULT_RECEIPT,
        "screen_status": "screen_passed_one_update_calibration_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if authorization != expected_authorization:
        raise ValueError("Hybrid calibration authorization binding differs")
    architecture = protocol.get("architecture", {})
    required_architecture = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "projected_kv_key_dim": 64,
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
        "rwkv_ms_hybrid_mode": SELECTED_CANDIDATE["hybrid_mode"],
        "rwkv_ms_hybrid_gain": SELECTED_CANDIDATE["hybrid_gain"],
        "backbone_dtype": "bfloat16",
    }
    mismatches = [
        key
        for key, expected in required_architecture.items()
        if architecture.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Hybrid BF16 calibration architecture differs: " + ", ".join(mismatches)
        )
    training = protocol.get("training", {})
    if (
        training.get("optimizer_updates") != 1
        or training.get("logical_global_batch_rows") != WORLD_SIZE
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("hf_endpoint") != HF_MIRROR_ENDPOINT
    ):
        raise ValueError("Hybrid BF16 calibration training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Hybrid calibration may not authorize protected data")
    return protocol


def validate_screen_result() -> Mapping[str, Any]:
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Hybrid screen result file hash differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Hybrid screen result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != SCREEN_RESULT_RECEIPT or receipt.get("payload_sha256") != digest:
        raise ValueError("Hybrid screen result receipt differs")
    selected = result.get("selected_candidate")
    if not isinstance(selected, Mapping):
        raise ValueError("Hybrid screen selected candidate is missing")
    selected_identity = {
        key: selected.get(key)
        for key in ("candidate_id", "hybrid_mode", "hybrid_gain")
    }
    if (
        result.get("schema") != screen.SCHEMA
        or result.get("status")
        != "screen_passed_one_update_calibration_authorized"
        or result.get("passed") is not True
        or result.get("one_update_calibration_authorized") is not True
        or result.get("protocol_payload_sha256") != screen.PROTOCOL_PAYLOAD_SHA256
        or result.get("protected_splits_opened") != []
        or selected_identity != SELECTED_CANDIDATE
        or selected.get("passed") is not True
    ):
        raise ValueError("Hybrid screen did not authorize the locked candidate")
    return result


def build_config():
    return replace(
        screen.build_config(),
        rwkv_ms_hybrid_mode=str(SELECTED_CANDIDATE["hybrid_mode"]),
        rwkv_ms_hybrid_gain=float(SELECTED_CANDIDATE["hybrid_gain"]),
    )


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
    replaced = attach_delta_mem(model, build_config())
    trainable_names = freeze_non_delta_mem_params(model)
    runtime._promote_trainable_parameters_to_fp32(model)
    checkpointed_mlps = runtime.checkpoint_frozen_mlp_activations(model)
    modules = tuple(iter_delta_mem_modules(model))
    all_selected_hybrid = all(
        module.memory_backend == "rwkv_ms"
        and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_write_mode == "recurrent"
        and module.rwkv_ms_hybrid_mode == SELECTED_CANDIDATE["hybrid_mode"]
        and module.rwkv_ms_hybrid_gain == SELECTED_CANDIDATE["hybrid_gain"]
        for _, module in modules
    )
    audit = {
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_names_sha256": canonical_sha256(sorted(trainable_names)),
        "checkpointed_frozen_mlps": len(checkpointed_mlps),
        "backbone_dtype": "bfloat16",
        "trainable_master_dtype": "float32",
        "all_wrappers_selected_hybrid": all_selected_hybrid,
    }
    if (
        len(replaced) != preflight.EXPECTED_LAYERS
        or len(modules) != preflight.EXPECTED_LAYERS
        or not all_selected_hybrid
    ):
        raise RuntimeError(f"Hybrid calibration attachment failed: {audit!r}")
    return model, tokenizer, audit


def recurrent_readout_sha256(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> str:
    selected = {
        name: parameter.detach().cpu()
        for name, parameter in named_trainable
        if name.endswith(recurrent_calibration.RECURRENT_READOUT_SUFFIX)
    }
    if len(selected) != preflight.EXPECTED_LAYERS:
        raise ValueError("Expected one recurrent output tensor per wrapped layer")
    return runtime._state_dict_sha256(selected)


def four_distinct_a100s(rank_devices: Sequence[Mapping[str, Any]]) -> bool:
    return (
        len(rank_devices) == WORLD_SIZE
        and len({str(device.get("device_uuid")) for device in rank_devices}) == WORLD_SIZE
        and all("A100" in str(device.get("device_name")) for device in rank_devices)
    )


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Hybrid BF16 calibration requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    screen_result = validate_screen_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Pinned Gemma base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Hybrid calibration output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="hybrid-bf16-calibration-output-freshness",
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
        phase="hybrid-bf16-calibration-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    sources, donors, row_payload = recurrent_calibration.calibration_rows(rows)
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
    initial_adapter_sha256 = runtime._state_dict_sha256(
        snapshot_delta_mem_weights(model)
    )
    initial_recurrent_readout_sha256 = recurrent_readout_sha256(named_trainable)
    distributed.require_consensus(
        context,
        initial_adapter_sha256,
        description="hybrid calibration initial adapter state",
    )
    distributed.require_consensus(
        context,
        initial_recurrent_readout_sha256,
        description="hybrid calibration initial recurrent output weights",
    )

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = recurrent_calibration.write_read_logits(
        model,
        batch,
        dtype=torch.bfloat16,
    )
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
        raise RuntimeError("Hybrid calibration global answer-token count is invalid")
    loss = answer_loss_sum / global_answer_tokens
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Hybrid calibration loss is non-finite")
    loss.backward()
    local_gradient_validation = distributed.validate_local_gradients(named_trainable)
    if local_gradient_validation["passed"] is not True:
        raise RuntimeError(
            "Hybrid calibration local gradient validation failed: "
            f"{local_gradient_validation!r}"
        )
    gradient_collective = distributed.sum_gradients(context, named_trainable)
    recurrent_gradient_audit = recurrent_calibration.audit_recurrent_readout_gradients(
        named_trainable
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_trainable],
        MAX_GRAD_NORM,
    )
    gradient_norm_value = float(gradient_norm.detach().float().item())
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError("Hybrid calibration gradient norm is non-finite")
    optimizer.step()
    post_update_adapter_sha256 = runtime._state_dict_sha256(
        snapshot_delta_mem_weights(model)
    )
    post_update_recurrent_readout_sha256 = recurrent_readout_sha256(named_trainable)
    distributed.require_consensus(
        context,
        post_update_adapter_sha256,
        description="hybrid calibration post-update adapter state",
    )
    distributed.require_consensus(
        context,
        post_update_recurrent_readout_sha256,
        description="hybrid calibration post-update recurrent output weights",
    )

    model.eval()
    correct_written_state = screen.write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_written_state = screen.write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    correct_state = screen.combine_state(correct_written_state, correct_written_state)
    zero_recurrent_state = screen.combine_state(correct_written_state, None)
    donor_recurrent_state = screen.combine_state(
        correct_written_state,
        donor_written_state,
    )
    projected_hashes = {
        "correct": runtime._state_dict_sha256(screen.projected_state(correct_state)),
        "zero_recurrent": runtime._state_dict_sha256(
            screen.projected_state(zero_recurrent_state)
        ),
        "matched_donor_recurrent": runtime._state_dict_sha256(
            screen.projected_state(donor_recurrent_state)
        ),
    }
    projected_carrier_fixed = len(set(projected_hashes.values())) == 1
    if not projected_carrier_fixed:
        raise RuntimeError("Projected carrier changed across recurrent interventions")
    projected_only_logits = screen.read_logits(
        model,
        batch,
        correct_state,
        readout_mode="projected_kv_slots",
    )
    post_update_candidate = screen.local_candidate_evidence(
        model,
        batch,
        correct_state,
        zero_recurrent_state,
        donor_recurrent_state,
        SELECTED_CANDIDATE,
        projected_only_logits,
    )
    local_evidence = {
        **row_payload[context.process_rank],
        "answer_target_tokens": local_answer_tokens,
        "local_loss_sum": float(answer_loss_sum.detach().float().item()),
        "local_gradient_validation": local_gradient_validation,
        "projected_carrier_sha256": projected_hashes["correct"],
        "projected_carrier_fixed_across_conditions": projected_carrier_fixed,
        "post_update_candidate": post_update_candidate,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)
    checks = {
        "four_distinct_a100_ranks": four_distinct_a100s(context.rank_devices),
        "screen_result_binding_valid": True,
        "all_42_recurrent_output_gradients_finite_nonzero": (
            recurrent_gradient_audit["passed"] is True
        ),
        "global_gradient_norm_finite_nonzero": (
            bool(torch.isfinite(gradient_norm).item()) and gradient_norm_value > 0.0
        ),
        "optimizer_update_changed_adapter": (
            initial_adapter_sha256 != post_update_adapter_sha256
        ),
        "optimizer_update_changed_recurrent_output_weights": (
            initial_recurrent_readout_sha256 != post_update_recurrent_readout_sha256
        ),
        "projected_carrier_fixed_on_all_ranks": all(
            evidence["projected_carrier_fixed_across_conditions"]
            for evidence in rank_evidence
        ),
        "zero_recurrent_exactly_equals_projected_only_on_all_ranks": all(
            evidence["post_update_candidate"]["checks"][
                "zero_recurrent_exactly_equals_projected_only"
            ]
            for evidence in rank_evidence
        ),
        "post_update_correct_vs_zero_material_on_all_ranks": all(
            evidence["post_update_candidate"]["checks"]["correct_vs_zero_material"]
            for evidence in rank_evidence
        ),
        "post_update_correct_vs_matched_donor_material_on_all_ranks": all(
            evidence["post_update_candidate"]["checks"][
                "correct_vs_matched_donor_material"
            ]
            for evidence in rank_evidence
        ),
        "post_update_correct_vs_projected_bounded_on_all_ranks": all(
            evidence["post_update_candidate"]["checks"][
                "correct_vs_projected_bounded"
            ]
            for evidence in rank_evidence
        ),
        "all_post_update_condition_logits_finite_on_all_ranks": all(
            evidence["post_update_candidate"]["checks"][
                "all_condition_logits_finite"
            ]
            for evidence in rank_evidence
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "calibration_passed_native_benchmark_authorized"
            if passed
            else "calibration_failed_native_benchmark_blocked"
        ),
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "screen_result_binding": {
            "path": str(SCREEN_RESULT),
            "file_sha256": SCREEN_RESULT_FILE_SHA256,
            "receipt": SCREEN_RESULT_RECEIPT,
            "selected_candidate": SELECTED_CANDIDATE,
            "screen_status": screen_result["status"],
        },
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "calibration_rows_payload_sha256": (
            recurrent_calibration.CALIBRATION_ROWS_PAYLOAD_SHA256
        ),
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
        "initial_recurrent_readout_sha256": initial_recurrent_readout_sha256,
        "post_update_recurrent_readout_sha256": (
            post_update_recurrent_readout_sha256
        ),
        "gradient_norm_before_clip": gradient_norm_value,
        "gradient_collective": gradient_collective,
        "recurrent_output_gradient_audit": recurrent_gradient_audit,
        "rank_devices": list(context.rank_devices),
        "rank_evidence": list(rank_evidence),
        "materiality_minimum": screen.MIN_MATERIAL_LOGIT_DELTA,
        "bounded_logit_delta_maximum": screen.MAX_BOUNDED_LOGIT_DELTA,
        "native_benchmark_authorized": passed,
        "failure_rule": (
            "The matched open native benchmark remains blocked unless this "
            "calibration passes every gate."
        ),
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "screen_result_file_sha256": sha256_file(SCREEN_RESULT),
            "screen_runner_sha256": sha256_file(Path(screen.__file__)),
            "delta_impl_sha256": sha256_file(
                PROJECT_ROOT / "deltamem/core/delta_impl.py"
            ),
            "rwkv_core_sha256": sha256_file(
                PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
            ),
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
        phase="hybrid-bf16-calibration-result-save",
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
        raise ValueError("Hybrid BF16 calibration requires four-rank torchrun")
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
                "selected_candidate": SELECTED_CANDIDATE["candidate_id"],
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
