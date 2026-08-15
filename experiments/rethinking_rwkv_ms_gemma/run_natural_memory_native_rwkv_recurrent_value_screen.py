#!/usr/bin/env python3
"""Screen RWKV internally routed recurrent values on four GPUs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import statistics
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

from deltamem.core.delta import iter_delta_mem_modules, snapshot_delta_mem_weights  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
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
    run_natural_memory_native_rwkv_addressed_value_screen as addressed_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_value_screen.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_recurrent_value_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "ec425e2f2858281002d879736de31410ba468f927d234d0a60adeca928d9f968"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_chunk_addressed_value_eval_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "962605bb4898d9acb09c1f5a76da5994b38d660bce060a9fa862bad9deda7272"
)
PRIOR_RESULT_RECEIPT = (
    "41befd55a2b6f88ae9c6b14f79db326e7c152cb9435d8e9e7c050503ab056d0e"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 62
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 16.0
CANDIDATES = (
    {
        "candidate_id": "recurrent_value_g003125",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
    },
    {
        "candidate_id": "recurrent_value_g00625",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.0625,
    },
    {
        "candidate_id": "recurrent_value_g0125",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.125,
    },
    {
        "candidate_id": "recurrent_value_g025",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.25,
    },
)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Recurrent-value screen output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Recurrent-value screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Recurrent-value screen protocol payload differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "prior_result_file_sha256": PRIOR_RESULT_FILE_SHA256,
        "prior_result_receipt": PRIOR_RESULT_RECEIPT,
        "prior_status": "chunk_addressed_value_native_gain_not_established",
    }
    if any(
        authorization.get(key) != value
        for key, value in required_authorization.items()
    ):
        raise ValueError("Recurrent-value screen authorization differs")
    if protocol.get("candidate_grid") != list(CANDIDATES):
        raise ValueError("Recurrent-value candidate grid differs")
    architecture = protocol.get("architecture", {})
    required_architecture = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "rwkv_ms_hybrid_mode": "recurrent_value",
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
        "backbone_dtype": "bfloat16",
    }
    if any(
        architecture.get(key) != value
        for key, value in required_architecture.items()
    ):
        raise ValueError("Recurrent-value screen architecture differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Recurrent-value screen may not authorize protected data")
    return protocol


def validate_prior_result() -> Mapping[str, Any]:
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Chunk-addressed result file hash differs")
    result = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Chunk-addressed result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "status": "chunk_addressed_value_native_gain_not_established",
        "passed": False,
        "native_recurrent_causal_gain_established": False,
    }
    if (
        digest != PRIOR_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(key) != value for key, value in required.items())
    ):
        raise ValueError("Chunk-addressed failure does not authorize redesign")
    return result


def build_config():
    return replace(
        hybrid_screen.build_config(),
        rwkv_ms_hybrid_mode="recurrent_value",
        rwkv_ms_hybrid_gain=float(CANDIDATES[0]["hybrid_gain"]),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
    )
    hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="recurrent_value",
        hybrid_gain=float(CANDIDATES[0]["hybrid_gain"]),
    )
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "recurrent_value"
        and module.rwkv_ms_hybrid_gain == float(CANDIDATES[0]["hybrid_gain"])
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_recurrent_value": configured,
    }
    if not configured:
        raise RuntimeError(f"Recurrent-value attachment failed: {audit!r}")
    return model, tokenizer, audit


def zero_projected_bundle(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: (
            torch.zeros_like(value)
            if key.endswith(hybrid_screen.PROJECTED_SUFFIXES)
            else value.detach().cpu().clone()
        )
        for key, value in state.items()
    }


def local_candidate_evidence(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    states: Mapping[str, Mapping[str, torch.Tensor]],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    logits = {
        name: addressed_screen.condition_logits(model, batch, state, candidate)
        for name, state in states.items()
    }
    correct_zero = hybrid_screen.compare_logits(
        logits["correct"],
        logits["zero_recurrent"],
    )
    correct_donor = hybrid_screen.compare_logits(
        logits["correct"],
        logits["matched_donor_recurrent"],
    )
    correct_permuted = hybrid_screen.compare_logits(
        logits["correct"],
        logits["layer_permuted_recurrent"],
    )
    correct_empty = hybrid_screen.compare_logits(
        logits["correct"],
        logits["empty_memory"],
    )
    projected_invariance = hybrid_screen.compare_logits(
        logits["correct"],
        logits["zero_projected_bundle"],
    )
    zero_empty = hybrid_screen.compare_logits(
        logits["zero_recurrent"],
        logits["empty_memory"],
    )
    checks = {
        "zero_recurrent_exactly_equals_empty_memory": torch.equal(
            logits["zero_recurrent"],
            logits["empty_memory"],
        ),
        "correct_exactly_equals_zero_projected_bundle": torch.equal(
            logits["correct"],
            logits["zero_projected_bundle"],
        ),
        "correct_vs_zero_material": (
            float(correct_zero["max_abs_logit_delta"])
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_matched_donor_material": (
            float(correct_donor["max_abs_logit_delta"])
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_layer_permuted_material": (
            float(correct_permuted["max_abs_logit_delta"])
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_empty_bounded": (
            float(correct_empty["max_abs_logit_delta"])
            <= MAX_BOUNDED_LOGIT_DELTA
        ),
        "all_condition_logits_finite": all(
            torch.isfinite(value).all().item() for value in logits.values()
        ),
    }
    return {
        **dict(candidate),
        "checks": checks,
        "passed": all(checks.values()),
        "correct_vs_zero_recurrent": correct_zero,
        "correct_vs_matched_donor_recurrent": correct_donor,
        "correct_vs_layer_permuted_recurrent": correct_permuted,
        "correct_vs_empty_memory": correct_empty,
        "correct_vs_zero_projected_bundle": projected_invariance,
        "zero_recurrent_vs_empty_memory": zero_empty,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Recurrent-value screen requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    prior_result = validate_prior_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Recurrent-value pinned base config differs")
    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Recurrent-value screen output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="recurrent-value-screen-output-freshness",
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
        phase="recurrent-value-screen-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    adapter_sha256 = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    distributed.require_consensus(
        context,
        adapter_sha256,
        description="recurrent-value initial adapter state",
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
            addressed_screen.ordered_module_names(model),
        ),
        "zero_projected_bundle": zero_projected_bundle(correct_state),
        "empty_memory": addressed_screen.empty_state(correct_state),
    }
    carrier_hashes = {
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
    projected_carrier_fixed = len(set(carrier_hashes.values())) == 1
    if not projected_carrier_fixed:
        raise RuntimeError("Projected carrier changed across recurrent interventions")
    local_candidates = [
        local_candidate_evidence(model, batch, states, candidate)
        for candidate in CANDIDATES
    ]
    local_evidence = {
        **row_payload[context.process_rank],
        "projected_carrier_hashes": carrier_hashes,
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
        "candidates": local_candidates,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        rank_rows = [rank["candidates"][candidate_index] for rank in rank_evidence]
        checks = {
            "projected_carrier_fixed_on_all_ranks": all(
                rank["projected_carrier_fixed_across_recurrent_interventions"]
                for rank in rank_evidence
            ),
            "zero_recurrent_exactly_equals_empty_memory_on_all_ranks": all(
                row["checks"]["zero_recurrent_exactly_equals_empty_memory"]
                for row in rank_rows
            ),
            "correct_exactly_equals_zero_projected_bundle_on_all_ranks": all(
                row["checks"]["correct_exactly_equals_zero_projected_bundle"]
                for row in rank_rows
            ),
            "correct_vs_zero_material_on_all_ranks": all(
                row["checks"]["correct_vs_zero_material"] for row in rank_rows
            ),
            "correct_vs_matched_donor_material_on_all_ranks": all(
                row["checks"]["correct_vs_matched_donor_material"]
                for row in rank_rows
            ),
            "correct_vs_layer_permuted_material_on_all_ranks": all(
                row["checks"]["correct_vs_layer_permuted_material"]
                for row in rank_rows
            ),
            "correct_vs_empty_bounded_on_all_ranks": all(
                row["checks"]["correct_vs_empty_bounded"] for row in rank_rows
            ),
            "all_condition_logits_finite_on_all_ranks": all(
                row["checks"]["all_condition_logits_finite"] for row in rank_rows
            ),
        }
        correct_empty_deltas = [
            float(row["correct_vs_empty_memory"]["max_abs_logit_delta"])
            for row in rank_rows
        ]
        candidate_results.append(
            {
                **dict(candidate),
                "checks": checks,
                "passed": all(checks.values()),
                "worst_rank_correct_vs_empty_max_abs_logit_delta": max(
                    correct_empty_deltas
                ),
                "median_rank_correct_vs_empty_max_abs_logit_delta": statistics.median(
                    correct_empty_deltas
                ),
                "minimum_rank_correct_vs_zero_max_abs_logit_delta": min(
                    float(row["correct_vs_zero_recurrent"]["max_abs_logit_delta"])
                    for row in rank_rows
                ),
                "minimum_rank_correct_vs_donor_max_abs_logit_delta": min(
                    float(
                        row["correct_vs_matched_donor_recurrent"][
                            "max_abs_logit_delta"
                        ]
                    )
                    for row in rank_rows
                ),
                "minimum_rank_correct_vs_layer_permuted_max_abs_logit_delta": min(
                    float(
                        row["correct_vs_layer_permuted_recurrent"][
                            "max_abs_logit_delta"
                        ]
                    )
                    for row in rank_rows
                ),
                "rank_results": rank_rows,
            }
        )
    selected = addressed_screen.select_candidate(candidate_results)
    checks = {
        "four_distinct_a100_ranks": addressed_screen.four_distinct_a100s(
            context.rank_devices
        ),
        "candidate_selected": selected is not None,
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "screen_passed_causal_calibration_authorized"
            if passed
            else "screen_failed_recurrent_value_training_blocked"
        ),
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "authorization_binding": {
            "prior_result": str(PRIOR_RESULT),
            "prior_result_file_sha256": PRIOR_RESULT_FILE_SHA256,
            "prior_result_receipt": PRIOR_RESULT_RECEIPT,
            "prior_result_status": prior_result["status"],
        },
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "base_model": str(base_model),
        "base_config_sha256": preflight.EXPECTED_BASE_CONFIG_SHA256,
        "dataset_root": str(dataset_root),
        "scene_fit_file_sha256": contrast.SCENE_FILE_SHA256,
        "calibration_rows_payload_sha256": calibration.CALIBRATION_ROWS_PAYLOAD_SHA256,
        "seed": SEED,
        "world_size": WORLD_SIZE,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "model_audit": model_audit,
        "initial_adapter_sha256": adapter_sha256,
        "rank_devices": list(context.rank_devices),
        "candidate_results": candidate_results,
        "selected_candidate": selected,
        "rank_evidence": list(rank_evidence),
        "causal_gradient_calibration_authorized": passed,
        "native_benchmark_authorized": False,
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "protocol_file_sha256": sha256_file(PROTOCOL),
            "prior_result_file_sha256": sha256_file(PRIOR_RESULT),
            "delta_impl_sha256": sha256_file(
                PROJECT_ROOT / "deltamem/core/delta_impl.py"
            ),
            "rwkv_core_sha256": sha256_file(
                PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
            ),
            "addressed_screen_helper_sha256": sha256_file(
                Path(addressed_screen.__file__)
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
        phase="recurrent-value-screen-result-save",
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
        raise ValueError("Recurrent-value screen requires four-rank torchrun")
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
                "selected_candidate": (
                    None
                    if result["selected_candidate"] is None
                    else result["selected_candidate"]["candidate_id"]
                ),
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
