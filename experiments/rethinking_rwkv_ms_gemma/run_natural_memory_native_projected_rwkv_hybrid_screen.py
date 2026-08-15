#!/usr/bin/env python3
"""Screen bounded projected-slot plus recurrent RWKV hybrids on four GPUs."""

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
    run_natural_memory_native_recurrent_rwkv_bf16_calibration as calibration,
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


SCHEMA = "rwkv_ms_natural_memory_native_projected_rwkv_hybrid_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_projected_rwkv_hybrid_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "d1902e8a82163acfda151d5c80d8cb0d5bfa4a4b3406a9b5a6fe779654c94dca"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 57
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
PROJECTED_SUFFIXES = (
    ".__projected_kv_keys",
    ".__projected_kv_values",
    ".__projected_kv_occupied",
    ".__projected_kv_surprise",
)
CANDIDATES = (
    {"candidate_id": "scalar_gate_g003125", "hybrid_mode": "scalar_gate", "hybrid_gain": 0.03125},
    {"candidate_id": "vector_gate_g003125", "hybrid_mode": "vector_gate", "hybrid_gain": 0.03125},
    {"candidate_id": "residual_g003125", "hybrid_mode": "residual", "hybrid_gain": 0.03125},
    {"candidate_id": "scalar_gate_g00625", "hybrid_mode": "scalar_gate", "hybrid_gain": 0.0625},
    {"candidate_id": "vector_gate_g00625", "hybrid_mode": "vector_gate", "hybrid_gain": 0.0625},
    {"candidate_id": "residual_g00625", "hybrid_mode": "residual", "hybrid_gain": 0.0625},
    {"candidate_id": "scalar_gate_g0125", "hybrid_mode": "scalar_gate", "hybrid_gain": 0.125},
    {"candidate_id": "vector_gate_g0125", "hybrid_mode": "vector_gate", "hybrid_gain": 0.125},
    {"candidate_id": "residual_g0125", "hybrid_mode": "residual", "hybrid_gain": 0.125},
)


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
        raise ValueError("Hybrid screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Hybrid screen protocol payload hash differs")
    if protocol.get("candidate_grid") != list(CANDIDATES):
        raise ValueError("Hybrid screen candidate grid differs")
    architecture = protocol.get("architecture", {})
    required = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "projected_kv_key_dim": 64,
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
        "backbone_dtype": "bfloat16",
    }
    mismatches = [
        key for key, expected in required.items() if architecture.get(key) != expected
    ]
    if mismatches:
        raise ValueError("Hybrid screen architecture differs: " + ", ".join(mismatches))
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Hybrid screen may not authorize protected data")
    return protocol


def build_config():
    return replace(
        preflight.build_config(),
        memory_readout_mode="projected_kv_rwkv_hybrid",
        projected_kv_key_dim=64,
        projected_kv_temperature=16.0,
        projected_kv_update_cosine_threshold=0.95,
        rwkv_ms_hybrid_mode=str(CANDIDATES[0]["hybrid_mode"]),
        rwkv_ms_hybrid_gain=float(CANDIDATES[0]["hybrid_gain"]),
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
    modules = tuple(iter_delta_mem_modules(model))
    audit = {
        "wrapped_layers": len(modules),
        "replaced_layers": len(replaced),
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_names_sha256": canonical_sha256(sorted(trainable_names)),
        "all_wrappers_hybrid": all(
            module.memory_backend == "rwkv_ms"
            and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_write_mode == "recurrent"
            for _, module in modules
        ),
    }
    if (
        len(replaced) != preflight.EXPECTED_LAYERS
        or len(modules) != preflight.EXPECTED_LAYERS
        or not audit["all_wrappers_hybrid"]
    ):
        raise RuntimeError(f"Hybrid attachment failed: {audit!r}")
    return model, tokenizer, audit


def configure_readout(
    model: torch.nn.Module,
    *,
    readout_mode: str,
    hybrid_mode: str = "residual",
    hybrid_gain: float = 0.0,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.memory_readout_mode = readout_mode
        module.rwkv_ms_hybrid_mode = hybrid_mode
        module.rwkv_ms_hybrid_gain = float(hybrid_gain)


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="residual",
        hybrid_gain=0.0,
    )
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
    audit_hybrid_state(state)
    return state


def audit_hybrid_state(state: Mapping[str, torch.Tensor]) -> None:
    projected = [name for name in state if name.endswith(PROJECTED_SUFFIXES)]
    recurrent_matrices = [
        name
        for name in state
        if not name.endswith(PROJECTED_SUFFIXES)
        and not name.endswith(".__rwkv_ms_positions")
        and not name.endswith(".__rwkv_ms_previous_source")
    ]
    positions = [name for name in state if name.endswith(".__rwkv_ms_positions")]
    previous = [name for name in state if name.endswith(".__rwkv_ms_previous_source")]
    expected = preflight.EXPECTED_LAYERS
    if (
        len(projected) != expected * len(PROJECTED_SUFFIXES)
        or len(recurrent_matrices) != expected
        or len(positions) != expected
        or len(previous) != expected
    ):
        raise RuntimeError(
            "Hybrid state is incomplete: "
            f"projected={len(projected)} matrices={len(recurrent_matrices)} "
            f"positions={len(positions)} previous={len(previous)}"
        )
    if any(torch.count_nonzero(state[name]).item() == 0 for name in recurrent_matrices):
        raise RuntimeError("Hybrid recurrent matrix state contains an empty layer")


def projected_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
        if name.endswith(PROJECTED_SUFFIXES)
    }


def combine_state(
    carrier: Mapping[str, torch.Tensor],
    recurrent: Mapping[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if recurrent is not None and set(carrier) != set(recurrent):
        raise ValueError("Correct and donor hybrid state keys differ")
    combined: dict[str, torch.Tensor] = {}
    for name, tensor in carrier.items():
        source = tensor if name.endswith(PROJECTED_SUFFIXES) else (
            None if recurrent is None else recurrent[name]
        )
        combined[name] = (
            torch.zeros_like(tensor)
            if source is None
            else source.detach().cpu().clone()
        )
    return combined


def read_logits(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    state: Mapping[str, torch.Tensor],
    *,
    readout_mode: str,
    hybrid_mode: str = "residual",
    hybrid_gain: float = 0.0,
) -> torch.Tensor:
    configure_readout(
        model,
        readout_mode=readout_mode,
        hybrid_mode=hybrid_mode,
        hybrid_gain=hybrid_gain,
    )
    reset_delta_mem_states(model)
    load_delta_mem_online_state(model, dict(state))
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


def compare_logits(
    reference: torch.Tensor,
    comparison: torch.Tensor,
) -> Mapping[str, Any]:
    metrics = dict(calibration.materiality_metrics(reference, comparison))
    metrics["all_finite"] = bool(
        torch.isfinite(reference).all().item()
        and torch.isfinite(comparison).all().item()
    )
    return metrics


def local_candidate_evidence(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    correct_state: Mapping[str, torch.Tensor],
    zero_recurrent_state: Mapping[str, torch.Tensor],
    donor_recurrent_state: Mapping[str, torch.Tensor],
    candidate: Mapping[str, Any],
    projected_only_logits: torch.Tensor,
) -> Mapping[str, Any]:
    mode = str(candidate["hybrid_mode"])
    gain = float(candidate["hybrid_gain"])
    zero_logits = read_logits(
        model,
        batch,
        zero_recurrent_state,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=mode,
        hybrid_gain=gain,
    )
    correct_logits = read_logits(
        model,
        batch,
        correct_state,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=mode,
        hybrid_gain=gain,
    )
    donor_logits = read_logits(
        model,
        batch,
        donor_recurrent_state,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=mode,
        hybrid_gain=gain,
    )
    correct_zero = compare_logits(correct_logits, zero_logits)
    correct_donor = compare_logits(correct_logits, donor_logits)
    correct_projected = compare_logits(correct_logits, projected_only_logits)
    zero_projected = compare_logits(zero_logits, projected_only_logits)
    checks = {
        "zero_recurrent_exactly_equals_projected_only": torch.equal(
            zero_logits,
            projected_only_logits,
        ),
        "correct_vs_zero_material": (
            float(correct_zero["max_abs_logit_delta"])
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_matched_donor_material": (
            float(correct_donor["max_abs_logit_delta"])
            >= MIN_MATERIAL_LOGIT_DELTA
        ),
        "correct_vs_projected_bounded": (
            float(correct_projected["max_abs_logit_delta"])
            <= MAX_BOUNDED_LOGIT_DELTA
        ),
        "all_condition_logits_finite": all(
            bool(metrics["all_finite"])
            for metrics in (
                correct_zero,
                correct_donor,
                correct_projected,
                zero_projected,
            )
        ),
    }
    return {
        **dict(candidate),
        "checks": checks,
        "passed": all(checks.values()),
        "correct_vs_zero_recurrent": correct_zero,
        "correct_vs_matched_donor_recurrent": correct_donor,
        "correct_vs_projected_only": correct_projected,
        "zero_recurrent_vs_projected_only": zero_projected,
    }


def select_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    passing = [result for result in candidate_results if result["passed"] is True]
    if not passing:
        return None
    return min(
        passing,
        key=lambda result: (
            float(result["hybrid_gain"]),
            float(result["worst_rank_correct_vs_projected_max_abs_logit_delta"]),
            str(result["candidate_id"]),
        ),
    )


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Hybrid screen requires exactly four ranks")
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
        freshness_error = ValueError(f"Hybrid screen output must be fresh: {resolved_output}")
    distributed.phase_consensus(
        context,
        phase="hybrid-screen-output-freshness",
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
        phase="hybrid-screen-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    adapter_sha256 = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    distributed.require_consensus(
        context,
        adapter_sha256,
        description="hybrid screen initial adapter state",
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
    correct_written_state = write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_written_state = write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    correct_state = combine_state(correct_written_state, correct_written_state)
    zero_recurrent_state = combine_state(correct_written_state, None)
    donor_recurrent_state = combine_state(correct_written_state, donor_written_state)
    projected_hashes = {
        "correct": runtime._state_dict_sha256(projected_state(correct_state)),
        "zero_recurrent": runtime._state_dict_sha256(projected_state(zero_recurrent_state)),
        "matched_donor_recurrent": runtime._state_dict_sha256(projected_state(donor_recurrent_state)),
    }
    projected_carrier_fixed = len(set(projected_hashes.values())) == 1
    if not projected_carrier_fixed:
        raise RuntimeError("Projected carrier changed across recurrent interventions")

    projected_only_logits = read_logits(
        model,
        batch,
        correct_state,
        readout_mode="projected_kv_slots",
    )
    local_candidates = [
        local_candidate_evidence(
            model,
            batch,
            correct_state,
            zero_recurrent_state,
            donor_recurrent_state,
            candidate,
            projected_only_logits,
        )
        for candidate in CANDIDATES
    ]
    local_evidence = {
        **row_payload[context.process_rank],
        "projected_carrier_sha256": projected_hashes["correct"],
        "projected_carrier_fixed_across_conditions": projected_carrier_fixed,
        "candidates": local_candidates,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        rank_rows = [rank["candidates"][candidate_index] for rank in rank_evidence]
        candidate_checks = {
            "projected_carrier_fixed_on_all_ranks": all(
                rank["projected_carrier_fixed_across_conditions"]
                for rank in rank_evidence
            ),
            "zero_recurrent_exactly_equals_projected_only_on_all_ranks": all(
                row["checks"]["zero_recurrent_exactly_equals_projected_only"]
                for row in rank_rows
            ),
            "correct_vs_zero_material_on_all_ranks": all(
                row["checks"]["correct_vs_zero_material"] for row in rank_rows
            ),
            "correct_vs_matched_donor_material_on_all_ranks": all(
                row["checks"]["correct_vs_matched_donor_material"]
                for row in rank_rows
            ),
            "correct_vs_projected_bounded_on_all_ranks": all(
                row["checks"]["correct_vs_projected_bounded"] for row in rank_rows
            ),
            "all_condition_logits_finite_on_all_ranks": all(
                row["checks"]["all_condition_logits_finite"] for row in rank_rows
            ),
        }
        candidate_results.append(
            {
                **dict(candidate),
                "checks": candidate_checks,
                "passed": all(candidate_checks.values()),
                "worst_rank_correct_vs_projected_max_abs_logit_delta": max(
                    float(row["correct_vs_projected_only"]["max_abs_logit_delta"])
                    for row in rank_rows
                ),
                "median_rank_correct_vs_projected_max_abs_logit_delta": statistics.median(
                    float(row["correct_vs_projected_only"]["max_abs_logit_delta"])
                    for row in rank_rows
                ),
                "minimum_rank_correct_vs_donor_max_abs_logit_delta": min(
                    float(row["correct_vs_matched_donor_recurrent"]["max_abs_logit_delta"])
                    for row in rank_rows
                ),
                "rank_results": rank_rows,
            }
        )
    selected = select_candidate(candidate_results)
    passed = selected is not None
    checks = {
        "four_distinct_a100_ranks": (
            len(context.rank_devices) == WORLD_SIZE
            and all("A100" in str(device["device_name"]) for device in context.rank_devices)
        ),
        "candidate_selected": passed,
    }
    passed = passed and all(checks.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "screen_passed_one_update_calibration_authorized"
            if passed
            else "screen_failed_hybrid_training_blocked"
        ),
        "passed": passed,
        "checks": checks,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
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
        "one_update_calibration_authorized": passed,
        "native_benchmark_authorized": False,
        "protected_splits_opened": [],
        "code_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "protocol_file_sha256": sha256_file(PROTOCOL),
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
        phase="hybrid-screen-result-save",
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
        raise ValueError("Hybrid screen requires four-rank torchrun")
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
