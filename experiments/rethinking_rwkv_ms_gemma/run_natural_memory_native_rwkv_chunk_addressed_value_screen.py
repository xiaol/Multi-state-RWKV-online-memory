#!/usr/bin/env python3
"""Screen exact projected-key/RWKV-chunk slot alignment on four GPUs."""

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


SCHEMA = "rwkv_ms_natural_memory_native_chunk_addressed_value_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_chunk_addressed_value_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "527f9512ab5cb7af4151d3b7f1c073a6744d1697527e8f4c62eb97a0be2f6b50"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_addressed_value_eval_batched_2way_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "67e54650ae6ed6dd26402b51c1a264d0f631f1f88d07c127c7a083579484fce1"
)
PRIOR_RESULT_RECEIPT = (
    "0aedbca3e4e98ab3d4325e8b291324a247ee06e1624152f7c20fd29c03f21a2d"
)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = calibration.BASE_MODEL
DATASET_ROOT = calibration.DATASET_ROOT
WORLD_SIZE = 4
SEED = 61
CANDIDATES = (
    {
        "candidate_id": "chunk_addressed_value_g003125",
        "hybrid_mode": "chunk_addressed_value",
        "hybrid_gain": 0.03125,
    },
    {
        "candidate_id": "chunk_addressed_value_g00625",
        "hybrid_mode": "chunk_addressed_value",
        "hybrid_gain": 0.0625,
    },
    {
        "candidate_id": "chunk_addressed_value_g0125",
        "hybrid_mode": "chunk_addressed_value",
        "hybrid_gain": 0.125,
    },
    {
        "candidate_id": "chunk_addressed_value_g025",
        "hybrid_mode": "chunk_addressed_value",
        "hybrid_gain": 0.25,
    },
)


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Chunk-addressed screen output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Chunk-addressed screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Chunk-addressed screen protocol payload hash differs")
    authorization = protocol.get("authorization_basis", {})
    required_authorization = {
        "prior_result_file_sha256": PRIOR_RESULT_FILE_SHA256,
        "prior_result_receipt": PRIOR_RESULT_RECEIPT,
        "prior_status": "addressed_value_native_gain_not_established",
    }
    if any(
        authorization.get(key) != expected
        for key, expected in required_authorization.items()
    ):
        raise ValueError("Chunk-addressed screen authorization differs")
    if protocol.get("candidate_grid") != list(CANDIDATES):
        raise ValueError("Chunk-addressed candidate grid differs")
    architecture = protocol.get("architecture", {})
    required_architecture = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "rwkv_ms_hybrid_mode": "chunk_addressed_value",
        "projected_kv_key_dim": 64,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
        "backbone_dtype": "bfloat16",
    }
    if any(
        architecture.get(key) != expected
        for key, expected in required_architecture.items()
    ):
        raise ValueError("Chunk-addressed screen architecture differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Chunk-addressed screen may not authorize protected data")
    return protocol


def validate_prior_result() -> Mapping[str, Any]:
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Addressed-value result file hash differs")
    result = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-value result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    required = {
        "status": "addressed_value_native_gain_not_established",
        "passed": False,
        "native_recurrent_causal_gain_established": False,
    }
    if (
        digest != PRIOR_RESULT_RECEIPT
        or receipt.get("payload_sha256") != digest
        or any(result.get(key) != expected for key, expected in required.items())
    ):
        raise ValueError("Addressed-value near-miss does not authorize redesign")
    return result


def build_config():
    return replace(
        hybrid_screen.build_config(),
        rwkv_ms_hybrid_mode="chunk_addressed_value",
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
        hybrid_mode="chunk_addressed_value",
        hybrid_gain=float(CANDIDATES[0]["hybrid_gain"]),
    )
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "chunk_addressed_value"
        and module.rwkv_ms_hybrid_gain == float(CANDIDATES[0]["hybrid_gain"])
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_chunk_addressed_value": configured,
    }
    if not configured:
        raise RuntimeError(f"Chunk-addressed attachment failed: {audit!r}")
    return model, tokenizer, audit


def chunk_alignment_evidence(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> Mapping[str, Any]:
    layer_evidence = []
    for name in module_names:
        recurrent = state[name]
        projected_occupied = state[f"{name}.__projected_kv_occupied"].to(
            dtype=torch.bool
        )
        recurrent_occupied = recurrent.ne(0).any(dim=(-1, -2)).any(dim=1)
        projected_values = state[f"{name}.__projected_kv_values"]
        layer_evidence.append(
            {
                "name": name,
                "occupied_slots_match": torch.equal(
                    projected_occupied,
                    recurrent_occupied,
                ),
                "minimum_projected_occupied_slots": int(
                    projected_occupied.sum(dim=-1).min().item()
                ),
                "minimum_recurrent_occupied_slots": int(
                    recurrent_occupied.sum(dim=-1).min().item()
                ),
                "projected_values_exactly_zero": (
                    torch.count_nonzero(projected_values).item() == 0
                ),
            }
        )
    return {
        "all_layer_occupied_slots_match": all(
            layer["occupied_slots_match"] for layer in layer_evidence
        ),
        "at_least_two_aligned_slots_on_every_layer": all(
            layer["minimum_projected_occupied_slots"] >= 2
            and layer["minimum_recurrent_occupied_slots"] >= 2
            for layer in layer_evidence
        ),
        "projected_values_exactly_zero_on_every_layer": all(
            layer["projected_values_exactly_zero"] for layer in layer_evidence
        ),
        "layers": layer_evidence,
    }


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="chunk_addressed_value",
        hybrid_gain=float(CANDIDATES[0]["hybrid_gain"]),
    )
    hybrid_screen.reset_delta_mem_states(model)
    hybrid_screen.set_delta_mem_write_enabled(model, True)
    with torch.inference_mode(), runtime._autocast_context(
        input_ids.device,
        torch.bfloat16,
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    state = hybrid_screen.get_delta_mem_online_state(model)
    hybrid_screen.audit_hybrid_state(state)
    return state


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    base_model: Path = BASE_MODEL,
    dataset_root: Path = DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Chunk-addressed screen requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_MIRROR_ENDPOINT}")
    protocol = validate_protocol()
    prior_result = validate_prior_result()
    base_model = base_model.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    if sha256_file(base_model / "config.json") != preflight.EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError("Chunk-addressed pinned base config differs")

    resolved_output = output_dir.expanduser().resolve()
    freshness_error: BaseException | None = None
    if context.is_primary and resolved_output.exists():
        freshness_error = ValueError(
            f"Chunk-addressed screen output must be fresh: {resolved_output}"
        )
    distributed.phase_consensus(
        context,
        phase="chunk-addressed-screen-output-freshness",
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
        phase="chunk-addressed-screen-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, model_audit = load_model(base_model, device=context.device)
    adapter_sha256 = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    distributed.require_consensus(
        context,
        adapter_sha256,
        description="chunk-addressed initial adapter state",
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
    correct_written = write_state(
        model,
        batch.write_input_ids,
        batch.write_attention_mask,
    )
    donor_written = write_state(
        model,
        donor_batch.write_input_ids,
        donor_batch.write_attention_mask,
    )
    module_names = addressed_screen.ordered_module_names(model)
    correct_alignment = chunk_alignment_evidence(correct_written, module_names)
    donor_alignment = chunk_alignment_evidence(donor_written, module_names)
    correct_state = hybrid_screen.combine_state(correct_written, correct_written)
    zero_recurrent = hybrid_screen.combine_state(correct_written, None)
    donor_recurrent = hybrid_screen.combine_state(correct_written, donor_written)
    layer_permuted = addressed_screen.permute_recurrent_state(
        correct_state,
        module_names,
    )
    states = {
        "correct": correct_state,
        "zero_recurrent": zero_recurrent,
        "matched_donor_recurrent": donor_recurrent,
        "layer_permuted_recurrent": layer_permuted,
        "zero_projected_values": addressed_screen.zero_projected_values(
            correct_state
        ),
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
        raise RuntimeError("Chunk-addressed carrier changed across interventions")

    local_candidates = [
        addressed_screen.local_candidate_evidence(
            model,
            batch,
            states,
            candidate,
        )
        for candidate in CANDIDATES
    ]
    local_evidence = {
        **row_payload[context.process_rank],
        "correct_chunk_alignment": correct_alignment,
        "donor_chunk_alignment": donor_alignment,
        "projected_carrier_hashes": carrier_hashes,
        "projected_carrier_fixed_across_recurrent_interventions": (
            projected_carrier_fixed
        ),
        "candidates": local_candidates,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(context.device)),
    }
    rank_evidence = distributed.gather_objects(context, local_evidence)
    alignment_on_all_ranks = all(
        rank[condition]["all_layer_occupied_slots_match"]
        for rank in rank_evidence
        for condition in ("correct_chunk_alignment", "donor_chunk_alignment")
    )
    two_slots_on_all_ranks = all(
        rank[condition]["at_least_two_aligned_slots_on_every_layer"]
        for rank in rank_evidence
        for condition in ("correct_chunk_alignment", "donor_chunk_alignment")
    )
    zero_values_on_all_ranks = all(
        rank[condition]["projected_values_exactly_zero_on_every_layer"]
        for rank in rank_evidence
        for condition in ("correct_chunk_alignment", "donor_chunk_alignment")
    )

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        rank_rows = [rank["candidates"][candidate_index] for rank in rank_evidence]
        checks = {
            "projected_and_recurrent_occupied_slots_match_on_all_ranks": (
                alignment_on_all_ranks
            ),
            "at_least_two_aligned_slots_occupied_on_all_ranks": (
                two_slots_on_all_ranks
            ),
            "projected_values_exactly_zero_on_all_ranks": zero_values_on_all_ranks,
            "projected_carrier_fixed_on_all_ranks": all(
                rank["projected_carrier_fixed_across_recurrent_interventions"]
                for rank in rank_evidence
            ),
            "zero_recurrent_exactly_equals_empty_memory_on_all_ranks": all(
                row["checks"]["zero_recurrent_exactly_equals_empty_memory"]
                for row in rank_rows
            ),
            "correct_exactly_equals_zero_projected_values_on_all_ranks": all(
                row["checks"]["correct_exactly_equals_zero_projected_values"]
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
            else "screen_failed_chunk_addressed_training_blocked"
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
        phase="chunk-addressed-screen-result-save",
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
        raise ValueError("Chunk-addressed screen requires four-rank torchrun")
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
