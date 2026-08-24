#!/usr/bin/env python3
"""Screen a causal prefix-depth accumulator for virtual-KV routing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import (
    screen_natural_memory_native_rwkv_compatibility_bias as local_screen,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_cumulative_compatibility_bias_protocol_v1.json"
)
PARENT_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_compatibility_bias_v1/result.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_cumulative_compatibility_bias_v1"
)
DISTRIBUTED_SOURCE = SCRIPT_DIR / "natural_memory_distributed.py"
LOCAL_SCREEN_SOURCE = (
    SCRIPT_DIR / "screen_natural_memory_native_rwkv_compatibility_bias.py"
)
WORLD_SIZE = 4
ROWS = 32
MODULES = 42
ADDRESS_DIM = 64
ANCHORS = (5, 11, 17, 23)
PREFIXES = {
    5: (5,),
    11: (5, 11),
    17: (5, 11, 17),
    23: (5, 11, 17, 23),
}
HF_ENDPOINT = "https://hf-mirror.com"
RESULT_SCHEMA = (
    "rwkv_ms_natural_memory_native_rwkv_cumulative_compatibility_bias_screen.v1"
)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    local_screen._validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Cumulative compatibility-bias protocol",
    )
    prefixes = {
        int(layer): tuple(values)
        for layer, values in protocol.get("architecture", {}).get("prefixes", {}).items()
    }
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_cumulative_compatibility_bias_protocol.v1"
        or protocol.get("architecture", {}).get("anchor_layers") != list(ANCHORS)
        or prefixes != PREFIXES
        or protocol.get("execution", {}).get("world_size") != WORLD_SIZE
        or protocol.get("execution", {}).get("hf_endpoint") != HF_ENDPOINT
        or protocol.get("frozen_inputs", {}).get("retrieval_rows") != ROWS
        or protocol.get("data_lifecycle", {}).get(
            "already_open_retrieval_tensors_only"
        )
        is not True
    ):
        raise ValueError("Cumulative compatibility-bias protocol contract differs")
    return protocol


def validate_parent_result(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    local_screen._validate_receipt(
        parent,
        scope="canonical_result_without_receipt",
        description="Local compatibility-bias parent result",
    )
    authorization = protocol["authorization_basis"]
    if (
        local_screen.sha256_file(PARENT_RESULT)
        != authorization["local_bias_result_sha256"]
        or parent.get("receipt", {}).get("payload_sha256")
        != authorization["local_bias_result_receipt"]
        or parent.get("status") != authorization["local_bias_status"]
        or parent.get("analysis", {}).get("passed_layers") != 2
        or parent.get("passed") is not False
    ):
        raise ValueError("Local compatibility-bias parent boundary differs")
    return parent


def _evaluate_local(
    records: Sequence[Mapping[str, Any]],
    module_names: Sequence[str],
    maps: Mapping[str, Any],
    *,
    process_rank: int,
    device: torch.device,
) -> list[Mapping[str, Any]]:
    by_source = {int(row["source_index"]): row for row in records}
    candidates = local_screen._candidate_sources(records)
    permutation = torch.tensor(
        local_screen.CANDIDATE_PERMUTATION, dtype=torch.long, device=device
    )
    local_results = []
    for source in sorted(by_source):
        if source % WORLD_SIZE != process_rank:
            continue
        row = by_source[source]
        candidate_sources = candidates[source]
        per_anchor = {}
        for anchor in ANCHORS:
            local_scores = []
            permuted_local_scores = []
            layer_permuted_local_scores = []
            zero_local_scores = []
            for layer in PREFIXES[anchor]:
                name = module_names[layer]
                query = torch.tensor(
                    row["causal_prompt_boundary_receptance32"][layer],
                    dtype=torch.float32,
                    device=device,
                )
                candidate_addresses = torch.tensor(
                    [
                        by_source[candidate]["write_address_full64"][layer]
                        for candidate in candidate_sources
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                scores = local_screen._score(query, candidate_addresses, maps[name])
                local_scores.append(scores)
                permuted_local_scores.append(
                    local_screen._score(
                        query,
                        candidate_addresses.index_select(0, permutation),
                        maps[name],
                    )
                )
                layer_permuted_address = torch.tensor(
                    row["write_address_full64"][(layer - 1) % MODULES],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                layer_permuted_local_scores.append(
                    local_screen._score(
                        query, layer_permuted_address, maps[name]
                    )[0]
                )
                zero_address = torch.zeros(
                    ADDRESS_DIM, dtype=torch.float32, device=device
                ).unsqueeze(0)
                zero_local_scores.append(
                    local_screen._score(query, zero_address, maps[name])[0]
                )
            local_score_tensor = torch.stack(local_scores)
            scores = local_score_tensor.mean(dim=0)
            permuted_scores = torch.stack(permuted_local_scores).mean(dim=0)
            layer_permuted_score = torch.stack(layer_permuted_local_scores).mean()
            zero_score = torch.stack(zero_local_scores).mean()
            correct = scores[0]
            strongest_wrong = scores[1:].max()
            per_anchor[str(anchor)] = {
                "prefix_layers": list(PREFIXES[anchor]),
                "local_scores": [
                    [float(value) for value in component.detach().cpu()]
                    for component in local_score_tensor
                ],
                "scores": [float(value) for value in scores.detach().cpu()],
                "correct_over_strongest_wrong_margin": float(
                    (correct - strongest_wrong).detach().cpu()
                ),
                "correct_over_matched_donor_margin": float(
                    (correct - scores[1]).detach().cpu()
                ),
                "correct_over_layer_permuted_margin": float(
                    (correct - layer_permuted_score).detach().cpu()
                ),
                "strict_correct_top1": bool((correct > strongest_wrong).item()),
                "candidate_permutation_exact": bool(
                    torch.equal(permuted_scores, scores.index_select(0, permutation))
                ),
                "zero_address_raw_bias_exact_zero": bool(zero_score.item() == 0.0),
                "zero_address_slot_eligible": False,
                "finite": bool(
                    torch.isfinite(local_score_tensor).all().item()
                    and torch.isfinite(scores).all().item()
                    and torch.isfinite(layer_permuted_score).item()
                    and torch.isfinite(zero_score).item()
                ),
            }
        local_results.append(
            {
                "source_index": source,
                "donor_source_index": int(row["donor_source_index"]),
                "candidate_sources": list(candidate_sources),
                "per_anchor": per_anchor,
            }
        )
    return local_results


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Cumulative compatibility-bias screen requires four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
    if (
        len(context.rank_devices) != WORLD_SIZE
        or len({device["device_uuid"] for device in context.rank_devices})
        != WORLD_SIZE
        or not all("A100" in device["device_name"] for device in context.rank_devices)
    ):
        raise RuntimeError(
            "Cumulative compatibility-bias screen requires four distinct A100s"
        )
    protocol = validate_protocol()
    validate_parent_result(protocol)
    frozen = protocol["frozen_inputs"]
    if (
        local_screen.sha256_file(LOCAL_SCREEN_SOURCE)
        != frozen["local_screen_source_sha256"]
        or local_screen.sha256_file(DISTRIBUTED_SOURCE)
        != frozen["distributed_source_sha256"]
    ):
        raise ValueError("Cumulative compatibility-bias source binding differs")
    records = local_screen._load_records(protocol)
    module_names, maps = local_screen._load_maps(protocol, context.device)
    input_digest = local_screen.canonical_sha256(records)
    distributed.require_consensus(
        context, input_digest, description="cumulative compatibility-bias inputs"
    )
    local_rows = _evaluate_local(
        records,
        module_names,
        maps,
        process_rank=context.process_rank,
        device=context.device,
    )
    gathered = distributed.gather_objects(context, local_rows)
    rows = sorted(
        [row for rank_rows in gathered for row in rank_rows],
        key=lambda row: int(row["source_index"]),
    )
    if len(rows) != ROWS or len({row["source_index"] for row in rows}) != ROWS:
        raise RuntimeError(
            "Cumulative compatibility-bias distributed coverage differs"
        )
    analysis = local_screen._aggregate(rows, protocol)
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "cumulative_compatibility_bias_passed_live_mechanics_protocol_authorized"
            if analysis["passed"]
            else "cumulative_compatibility_bias_failed_family_retired"
        ),
        "passed": analysis["passed"],
        "protocol_file_sha256": local_screen.sha256_file(PROTOCOL),
        "protocol_receipt": protocol["receipt"]["payload_sha256"],
        "runner_sha256": local_screen.sha256_file(Path(__file__).resolve()),
        "source_bindings": {
            "alignment": local_screen.sha256_file(local_screen.ALIGNMENT_SOURCE),
            "local_screen": local_screen.sha256_file(LOCAL_SCREEN_SOURCE),
            "distributed": local_screen.sha256_file(DISTRIBUTED_SOURCE),
        },
        "parent_result_sha256": local_screen.sha256_file(PARENT_RESULT),
        "parent_result_receipt": protocol["authorization_basis"][
            "local_bias_result_receipt"
        ],
        "input_root": str(local_screen.INPUT_ROOT.relative_to(SCRIPT_DIR)),
        "input_records_digest": input_digest,
        "map_file_sha256": local_screen.sha256_file(local_screen.MAP_FILE),
        "frozen_map_digest": frozen["frozen_map_digest"],
        "hardware": {
            "world_size": context.world_size,
            "devices": list(context.rank_devices),
            "four_distinct_a100s": True,
            "hf_endpoint": os.environ["HF_ENDPOINT"],
        },
        "architecture": dict(protocol["architecture"]),
        "candidate_bank": dict(protocol["candidate_bank"]),
        "analysis": analysis,
        "rows": rows,
        "retrieval_evaluation_calls": 1,
        "fit_shards_read": False,
        "mechanics_or_causal_bytes_opened": False,
        "generation_or_native_benchmark_bytes_opened": False,
        "model_or_adapter_parameters_updated": False,
        "full_bandwidth_feedback_installed": False,
        "native_gain_claimed": False,
        "sota_claimed": False,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": local_screen.canonical_sha256(result),
    }
    distributed.require_consensus(
        context,
        local_screen.canonical_sha256(result),
        description="cumulative compatibility-bias result",
    )
    if context.is_primary:
        if output_dir.exists():
            raise ValueError(
                f"Cumulative compatibility-bias output must be fresh: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dist.barrier(group=context.control_group)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    context = distributed.initialize_distributed_training(
        arguments.device, required_world_size=WORLD_SIZE
    )
    if context is None:
        raise RuntimeError(
            "Cumulative compatibility-bias screen must run under torchrun"
        )
    try:
        result = run(context=context, output_dir=arguments.output_dir)
        if context.is_primary:
            print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    finally:
        distributed.destroy_distributed_training(context)


if __name__ == "__main__":
    raise SystemExit(main())
