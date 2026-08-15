#!/usr/bin/env python3
"""Run one-update calibration for chunk-aligned addressed RWKV values."""

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
    run_natural_memory_native_rwkv_addressed_value_screen as addressed_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_chunk_addressed_value_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_chunk_addressed_value_calibration.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_chunk_addressed_value_calibration_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "86c63e684e01470f0e03395297ce09f4eab42994e64058acd51bf0aab4cd8051"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_screen_v2/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "f2d7a33b4e3e6af61011c88d0a65dbd45d1e0f0a220bcc044caef3095760e0dc"
)
SCREEN_RESULT_RECEIPT = (
    "59eb70e7e2ae6ca846eb05c61b447f5332485d0bde82c7b531e80a0f959c2b42"
)
SELECTED_CANDIDATE = {
    "candidate_id": "chunk_addressed_value_g003125",
    "hybrid_mode": "chunk_addressed_value",
    "hybrid_gain": 0.03125,
}
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = screen.BASE_MODEL
DATASET_ROOT = screen.DATASET_ROOT
WORLD_SIZE = 4
SEED = 61
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Chunk-addressed calibration output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Chunk-addressed calibration protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Chunk-addressed calibration protocol payload differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "screen_protocol_payload_sha256": screen.PROTOCOL_PAYLOAD_SHA256,
        "screen_result_file": (
            "local_artifacts/"
            "natural_memory_native_rwkv_chunk_addressed_value_screen_v2/"
            "result.json"
        ),
        "screen_result_file_sha256": SCREEN_RESULT_FILE_SHA256,
        "screen_result_receipt": SCREEN_RESULT_RECEIPT,
        "screen_status": "screen_passed_causal_calibration_authorized",
        "selected_candidate": SELECTED_CANDIDATE,
    }
    if authorization != required_authorization:
        raise ValueError("Chunk-addressed calibration authorization differs")
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
        raise ValueError("Chunk-addressed calibration training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Chunk-addressed calibration may not authorize protected data")
    return protocol


def validate_screen_result() -> Mapping[str, Any]:
    if sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Chunk-addressed screen result file hash differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Chunk-addressed screen result receipt is missing")
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
    required = {
        "schema": screen.SCHEMA,
        "status": "screen_passed_causal_calibration_authorized",
        "passed": True,
        "causal_gradient_calibration_authorized": True,
        "protected_splits_opened": [],
    }
    if (
        digest != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(key) != value for key, value in required.items())
        or selected_identity != SELECTED_CANDIDATE
    ):
        raise ValueError("Chunk-addressed screen did not authorize calibration")
    return result


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Chunk-addressed calibration requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    screen_result = validate_screen_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Chunk-addressed pinned base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Chunk-addressed calibration output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="chunk-addressed-calibration-output-freshness",
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
        phase="chunk-addressed-calibration-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = screen.load_model(
        base_model,
        device=context.device,
    )
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
        description="chunk-addressed calibration initial adapter",
    )
    distributed.require_consensus(
        context,
        initial_recurrent_readout_sha256,
        description="chunk-addressed calibration initial recurrent readout",
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
        raise RuntimeError("Chunk-addressed calibration answer-token count is invalid")
    loss = answer_loss_sum / global_answer_tokens
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Chunk-addressed calibration loss is non-finite")
    loss.backward()
    local_gradient_validation = distributed.validate_local_gradients(named_trainable)
    if local_gradient_validation["passed"] is not True:
        raise RuntimeError("Chunk-addressed calibration gradients are invalid")
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
        raise RuntimeError("Chunk-addressed calibration gradient norm is invalid")
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
        description="chunk-addressed calibration updated adapter",
    )
    distributed.require_consensus(
        context,
        post_update_recurrent_readout_sha256,
        description="chunk-addressed calibration updated recurrent readout",
    )

    model.eval()
    correct_written = screen.write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_written = screen.write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    module_names = addressed_screen.ordered_module_names(model)
    correct_alignment = screen.chunk_alignment_evidence(
        correct_written,
        module_names,
    )
    donor_alignment = screen.chunk_alignment_evidence(
        donor_written,
        module_names,
    )
    correct_state = hybrid_screen.combine_state(correct_written, correct_written)
    states = {
        "correct": correct_state,
        "zero_recurrent": hybrid_screen.combine_state(correct_written, None),
        "matched_donor_recurrent": hybrid_screen.combine_state(
            correct_written,
            donor_written,
        ),
        "layer_permuted_recurrent": addressed_screen.permute_recurrent_state(
            correct_state,
            module_names,
        ),
        "zero_projected_values": addressed_screen.zero_projected_values(
            correct_state
        ),
        "empty_memory": addressed_screen.empty_state(correct_state),
    }
    projected_hashes = {
        name: addressed_screen.projected_hash(state)
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
    candidate_evidence = addressed_screen.local_candidate_evidence(
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
        "correct_chunk_alignment": correct_alignment,
        "donor_chunk_alignment": donor_alignment,
        "projected_carrier_hashes": projected_hashes,
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
        "post_update_candidate": candidate_evidence,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)
    alignment_conditions = ("correct_chunk_alignment", "donor_chunk_alignment")
    checks = {
        "four_distinct_a100_ranks": addressed_screen.four_distinct_a100s(
            context.rank_devices
        ),
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
        "projected_and_recurrent_occupied_slots_match_on_all_ranks": all(
            rank[condition]["all_layer_occupied_slots_match"]
            for rank in rank_evidence
            for condition in alignment_conditions
        ),
        "at_least_two_aligned_slots_occupied_on_all_ranks": all(
            rank[condition]["at_least_two_aligned_slots_on_every_layer"]
            for rank in rank_evidence
            for condition in alignment_conditions
        ),
        "projected_values_exactly_zero_on_all_ranks": all(
            rank[condition]["projected_values_exactly_zero_on_every_layer"]
            for rank in rank_evidence
            for condition in alignment_conditions
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
        phase="chunk-addressed-calibration-result-save",
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
        raise ValueError("Chunk-addressed calibration requires four-rank torchrun")
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
