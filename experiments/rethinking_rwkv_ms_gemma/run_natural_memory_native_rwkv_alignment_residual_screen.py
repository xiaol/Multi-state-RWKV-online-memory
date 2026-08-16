#!/usr/bin/env python3
"""Screen alignment-gated recurrent residuals with causal state controls."""

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
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
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
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as state_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as router_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_alignment_residual_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_alignment_residual_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "be6abb289bcc4b2326b0bd14cfedabbccae17486098cc4601d4457f92057668b"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_vector_gate_eval_v1/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "7edadd1ed61b85157d4604eac75ab8da8d04f79c0ee3503d7f6535fe1f003471"
)
PRIOR_RESULT_RECEIPT = (
    "9fcbbd11ba502fdab77bee6c1177a5f5296cd4ff6ebecd0acdb3ce02b4cd10af"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 75
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
SELECTED_CANDIDATE = {
    "candidate_id": "alignment_residual_t16_k2_gate025_g0125",
    "hybrid_mode": "alignment_residual",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
PASS_STATUS = "alignment_residual_screen_passed_training_authorized"
FAIL_STATUS = "alignment_residual_screen_failed_training_blocked"
MODEL_AUDIT_KEY = "all_wrappers_alignment_residual_content_gated"
PRIOR_RESULT_CODE_BINDING_KEY = "vector_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Alignment-residual protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("hybrid_mode") != "alignment_residual"
        or architecture.get("hybrid_gain") != 0.125
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Alignment-residual screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Vector-gate result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    receipt_prior = unsigned_prior.pop("receipt", {})
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or receipt_prior.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "vector_gate_native_gain_without_full_causal_pass"
        or prior.get("passed") is not False
    ):
        raise ValueError("Vector-gate partial result does not authorize redesign")
    return protocol


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=top2_screen.build_config(SELECTED_CANDIDATE),
    )
    router_screen.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == SELECTED_CANDIDATE["hybrid_mode"]
        and module.rwkv_ms_hybrid_gain == SELECTED_CANDIDATE["hybrid_gain"]
        and module.rwkv_ms_read_temperature
        == SELECTED_CANDIDATE["read_temperature"]
        and module.rwkv_ms_read_top_k == SELECTED_CANDIDATE["read_top_k"]
        and module.rwkv_ms_detach_read_scores
        is SELECTED_CANDIDATE["detach_read_scores"]
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        MODEL_AUDIT_KEY: configured,
    }
    if not configured:
        raise RuntimeError(f"Alignment-residual attachment failed: {audit!r}")
    return model, tokenizer, audit


def local_evidence(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    projected_only = hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_slots",
    )
    condition_logits = {
        condition: hybrid_screen.read_logits(
            model,
            batch,
            state,
            readout_mode="projected_kv_rwkv_hybrid",
            hybrid_mode=str(SELECTED_CANDIDATE["hybrid_mode"]),
            hybrid_gain=float(SELECTED_CANDIDATE["hybrid_gain"]),
        )
        for condition, state in states.items()
    }
    comparisons = {
        "correct_vs_zero": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["zero"]
        ),
        "correct_vs_matched_donor": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["matched_donor"]
        ),
        "correct_vs_layer_permuted": hybrid_screen.compare_logits(
            condition_logits["correct"], condition_logits["layer_permuted"]
        ),
        "correct_vs_projected_only": hybrid_screen.compare_logits(
            condition_logits["correct"], projected_only
        ),
        "zero_vs_projected_only": hybrid_screen.compare_logits(
            condition_logits["zero"], projected_only
        ),
    }
    checks = {
        "zero_recurrent_exactly_equals_projected_only": torch.equal(
            condition_logits["zero"], projected_only
        ),
        "correct_vs_zero_material": (
            comparisons["correct_vs_zero"]["max_abs_logit_delta"]
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_matched_donor_material": (
            comparisons["correct_vs_matched_donor"]["max_abs_logit_delta"]
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_layer_permuted_material": (
            comparisons["correct_vs_layer_permuted"]["max_abs_logit_delta"]
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_projected_bounded": (
            comparisons["correct_vs_projected_only"]["max_abs_logit_delta"]
            <= MAX_BOUNDED_LOGIT_DELTA
        ),
        "all_condition_logits_finite": all(
            metrics["all_finite"] for metrics in comparisons.values()
        ),
    }
    return {
        "candidate": SELECTED_CANDIDATE,
        "checks": checks,
        "passed": all(checks.values()),
        "comparisons": comparisons,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Alignment-residual screen requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Pinned Gemma base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Alignment-residual screen output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="alignment-residual-output-freshness",
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
        phase="alignment-residual-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    adapter_sha256 = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    distributed.require_consensus(
        context,
        adapter_sha256,
        description="alignment-residual initial adapter state",
    )
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    sources, donors, row_payload = calibration.calibration_rows(rows)
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
    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    correct_recurrent, correct_projected = state_helper.split_state(
        correct_written, module_names
    )
    donor_recurrent, _ = state_helper.split_state(donor_written, module_names)
    states = {
        "correct": state_helper.merge_state(correct_recurrent, correct_projected),
        "zero": state_helper.merge_state(
            state_helper.zero_recurrent_state(correct_recurrent), correct_projected
        ),
        "matched_donor": state_helper.merge_state(
            donor_recurrent, correct_projected
        ),
        "layer_permuted": state_helper.merge_state(
            state_helper.permute_recurrent_state(correct_recurrent, module_names),
            correct_projected,
        ),
    }
    projected_hashes = {
        condition: runtime._state_dict_sha256(hybrid_screen.projected_state(state))
        for condition, state in states.items()
    }
    projected_carrier_fixed = len(set(projected_hashes.values())) == 1
    if not projected_carrier_fixed:
        raise RuntimeError("Projected carrier changed across recurrent controls")
    evidence = local_evidence(model, batch, states)
    local_result = {
        **row_payload[context.process_rank],
        **evidence,
        "projected_carrier_sha256": projected_hashes["correct"],
        "projected_carrier_fixed_across_conditions": projected_carrier_fixed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_result)
    checks = {
        "four_distinct_a100_ranks": (
            len(context.rank_devices) == WORLD_SIZE
            and all("A100" in device["device_name"] for device in context.rank_devices)
        ),
        "projected_carrier_fixed_on_all_ranks": all(
            row["projected_carrier_fixed_across_conditions"]
            for row in rank_evidence
        ),
        "candidate_passed_on_all_ranks": all(
            row["passed"] for row in rank_evidence
        ),
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "selected_candidate": SELECTED_CANDIDATE,
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "initial_adapter_sha256": adapter_sha256,
        "rank_evidence": list(rank_evidence),
        "training_authorized": passed,
        "native_generation_authorized": False,
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(RUNNER_BINDING_PATH),
            "shared_screen_runner_sha256": sha256_file(Path(__file__)),
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "delta_impl_sha256": sha256_file(
                PROJECT_ROOT / "deltamem/core/delta_impl.py"
            ),
            PRIOR_RESULT_CODE_BINDING_KEY: sha256_file(PRIOR_RESULT),
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
        phase="alignment-residual-result-save",
        error=save_error,
    )
    del model, batch, donor_batch, rows
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
        raise ValueError("Alignment-residual screen requires four-rank torchrun")
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
