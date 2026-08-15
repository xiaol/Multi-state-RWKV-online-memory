#!/usr/bin/env python3
"""Run one-update BF16 calibration for the addressed RWKV value path."""

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
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import snapshot_delta_mem_weights  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_bf16_calibration as hybrid_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as recurrent_calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
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


SCHEMA = "rwkv_ms_natural_memory_native_addressed_value_calibration.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_addressed_value_calibration_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "6e9951cb636cc645357acfa8cc550d3c34c07f5b3d9e17125da0c7ad1a1c54b4"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_value_screen_v1/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "d6206d155a8fc967fae83a6de17a9e513f8b416142694c9fdf1f49d8806a18e1"
)
SCREEN_RESULT_RECEIPT = (
    "e985f1a507cd459e3266c79a7ceb2c62afe2e259f0ea0f64d732536955de1773"
)
SELECTED_CANDIDATE = {
    "candidate_id": "addressed_value_g003125",
    "hybrid_mode": "addressed_value",
    "hybrid_gain": 0.03125,
}
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = screen.BASE_MODEL
DATASET_ROOT = screen.DATASET_ROOT
WORLD_SIZE = 4
SEED = 57
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Addressed-value calibration output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-value calibration protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Addressed-value calibration protocol payload differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "screen_protocol_payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
        "screen_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_addressed_value_screen_v1/result.json"
        ),
        "screen_result_file_sha256": SCREEN_RESULT_FILE_SHA256,
        "screen_result_receipt": SCREEN_RESULT_RECEIPT,
        "screen_status": "screen_passed_causal_calibration_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if authorization != required_authorization:
        raise ValueError("Addressed-value calibration authorization differs")
    training = protocol.get("training", {})
    required_training = {
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "max_gradient_norm": MAX_GRAD_NORM,
        "optimizer_updates": 1,
        "logical_global_batch_rows": WORLD_SIZE,
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("Addressed-value calibration training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Addressed-value calibration may not authorize protected data")
    return protocol


def validate_screen_result() -> Mapping[str, Any]:
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Addressed-value screen result file hash differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-value screen result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    selected = result.get("selected_candidate")
    selected_identity = (
        None
        if not isinstance(selected, Mapping)
        else {
            key: selected.get(key)
            for key in ("candidate_id", "hybrid_mode", "hybrid_gain")
        }
    )
    if (
        digest != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or result.get("schema") != screen.SCHEMA
        or result.get("status") != "screen_passed_causal_calibration_authorized"
        or result.get("passed") is not True
        or result.get("causal_gradient_calibration_authorized") is not True
        or result.get("protected_splits_opened") != []
        or selected_identity != SELECTED_CANDIDATE
    ):
        raise ValueError("Addressed-value screen did not authorize calibration")
    return result


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, audit = screen.load_model(base_model, device=device)
    return model, tokenizer, audit


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Addressed-value calibration requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    screen_result = validate_screen_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Addressed-value pinned base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Addressed-value calibration output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="addressed-value-calibration-output-freshness",
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
        phase="addressed-value-calibration-output-creation",
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
    initial_recurrent_readout_sha256 = hybrid_calibration.recurrent_readout_sha256(
        named_trainable
    )
    distributed.require_consensus(
        context,
        initial_adapter_sha256,
        description="addressed-value calibration initial adapter",
    )
    distributed.require_consensus(
        context,
        initial_recurrent_readout_sha256,
        description="addressed-value calibration initial recurrent readout",
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
        raise RuntimeError("Addressed-value calibration answer-token count is invalid")
    loss = answer_loss_sum / global_answer_tokens
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Addressed-value calibration loss is non-finite")
    loss.backward()
    local_gradient_validation = distributed.validate_local_gradients(named_trainable)
    if local_gradient_validation["passed"] is not True:
        raise RuntimeError("Addressed-value calibration gradients are invalid")
    gradient_collective = distributed.sum_gradients(context, named_trainable)
    recurrent_gradient_audit = recurrent_calibration.audit_recurrent_readout_gradients(
        named_trainable
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_trainable],
        MAX_GRAD_NORM,
    )
    gradient_norm_value = float(gradient_norm.detach().float().item())
    if not bool(torch.isfinite(gradient_norm).item()) or gradient_norm_value <= 0.0:
        raise RuntimeError("Addressed-value calibration gradient norm is invalid")
    optimizer.step()
    post_update_adapter_sha256 = runtime._state_dict_sha256(
        snapshot_delta_mem_weights(model)
    )
    post_update_recurrent_readout_sha256 = (
        hybrid_calibration.recurrent_readout_sha256(named_trainable)
    )
    distributed.require_consensus(
        context,
        post_update_adapter_sha256,
        description="addressed-value calibration updated adapter",
    )
    distributed.require_consensus(
        context,
        post_update_recurrent_readout_sha256,
        description="addressed-value calibration updated recurrent readout",
    )

    model.eval()
    correct_written = hybrid_screen.write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_written = hybrid_screen.write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    correct_state = hybrid_screen.combine_state(correct_written, correct_written)
    states = {
        "correct": correct_state,
        "zero_recurrent": hybrid_screen.combine_state(correct_written, None),
        "matched_donor_recurrent": hybrid_screen.combine_state(
            correct_written, donor_written
        ),
        "layer_permuted_recurrent": screen.permute_recurrent_state(
            correct_state,
            screen.ordered_module_names(model),
        ),
        "zero_projected_values": screen.zero_projected_values(correct_state),
        "empty_memory": screen.empty_state(correct_state),
    }
    projected_hashes = {
        name: screen.projected_hash(state)
        for name, state in states.items()
        if name
        in {
            "correct",
            "zero_recurrent",
            "matched_donor_recurrent",
            "layer_permuted_recurrent",
        }
    }
    projected_carrier_fixed = len(set(projected_hashes.values())) == 1
    candidate_evidence = screen.local_candidate_evidence(
        model,
        batch,
        states,
        SELECTED_CANDIDATE,
    )
    local_evidence = {
        **row_payload[context.process_rank],
        "answer_target_tokens": local_answer_tokens,
        "local_loss_sum": float(answer_loss_sum.detach().float().item()),
        "local_gradient_validation": local_gradient_validation,
        "projected_carrier_hashes": projected_hashes,
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
        "post_update_candidate": candidate_evidence,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)
    checks = {
        "four_distinct_a100_ranks": screen.four_distinct_a100s(context.rank_devices),
        "screen_result_binding_valid": True,
        "all_42_recurrent_output_gradients_finite_nonzero": (
            recurrent_gradient_audit["passed"] is True
        ),
        "global_gradient_norm_finite_nonzero": gradient_norm_value > 0.0,
        "optimizer_update_changed_adapter": (
            initial_adapter_sha256 != post_update_adapter_sha256
        ),
        "optimizer_update_changed_recurrent_output_weights": (
            initial_recurrent_readout_sha256
            != post_update_recurrent_readout_sha256
        ),
        "projected_carrier_fixed_on_all_ranks": all(
            rank["projected_carrier_fixed_across_recurrent_interventions"]
            for rank in rank_evidence
        ),
        "zero_recurrent_exactly_equals_empty_memory_on_all_ranks": all(
            rank["post_update_candidate"]["checks"][
                "zero_recurrent_exactly_equals_empty_memory"
            ]
            for rank in rank_evidence
        ),
        "correct_exactly_equals_zero_projected_values_on_all_ranks": all(
            rank["post_update_candidate"]["checks"][
                "correct_exactly_equals_zero_projected_values"
            ]
            for rank in rank_evidence
        ),
        "correct_vs_zero_material_on_all_ranks": all(
            rank["post_update_candidate"]["checks"]["correct_vs_zero_material"]
            for rank in rank_evidence
        ),
        "correct_vs_matched_donor_material_on_all_ranks": all(
            rank["post_update_candidate"]["checks"][
                "correct_vs_matched_donor_material"
            ]
            for rank in rank_evidence
        ),
        "correct_vs_layer_permuted_material_on_all_ranks": all(
            rank["post_update_candidate"]["checks"][
                "correct_vs_layer_permuted_material"
            ]
            for rank in rank_evidence
        ),
        "all_post_update_logits_finite_on_all_ranks": all(
            rank["post_update_candidate"]["checks"][
                "all_condition_logits_finite"
            ]
            for rank in rank_evidence
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "calibration_passed_causal_training_authorized"
            if passed
            else "calibration_failed_causal_training_blocked"
        ),
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "screen_result_binding": {
            "path": str(SCREEN_RESULT),
            "file_sha256": SCREEN_RESULT_FILE_SHA256,
            "receipt": SCREEN_RESULT_RECEIPT,
            "status": screen_result["status"],
            "selected_candidate": SELECTED_CANDIDATE,
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
        "max_gradient_norm": MAX_GRAD_NORM,
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
        "causal_training_authorized": passed,
        "native_benchmark_authorized": False,
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
        phase="addressed-value-calibration-result-save",
        error=save_error,
    )
    del model, optimizer, logits, batch, donor_batch, rows
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
        raise ValueError("Addressed-value calibration requires four-rank torchrun")
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
