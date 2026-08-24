#!/usr/bin/env python3
"""Screen an explicit RWKV compatibility bias on frozen retrieval tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import rwkv_continuous_write_alignment as alignment


SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_compatibility_bias_protocol_v1.json"
INPUT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_continuous_write_retrieval_v1"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_compatibility_bias_v1"
)
MAP_FILE = INPUT_ROOT / "continuous-write-maps.pt"
ALIGNMENT_SOURCE = SCRIPT_DIR / "rwkv_continuous_write_alignment.py"
WORLD_SIZE = 4
ROWS = 32
MODULES = 42
ADDRESS_DIM = 64
STATE_DIM = 32
ANCHORS = (5, 11, 17, 23)
CANDIDATE_PERMUTATION = (2, 0, 3, 1)
HF_ENDPOINT = "https://hf-mirror.com"
SHARD_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval_shard.v1"
FEATURE_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_retrieval_feature.v1"
MAP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_continuous_write_maps.v1"
RESULT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_compatibility_bias_screen.v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_receipt(
    payload: Mapping[str, Any], *, scope: str, description: str
) -> None:
    unsigned = dict(payload)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": scope,
        "payload_sha256": canonical_sha256(unsigned),
    }
    if receipt != expected:
        raise ValueError(f"{description} receipt differs")


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_receipt(
        protocol,
        scope="canonical_protocol_without_receipt",
        description="Compatibility-bias protocol",
    )
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_compatibility_bias_protocol.v1"
        or protocol.get("architecture", {}).get("anchor_layers") != list(ANCHORS)
        or protocol.get("execution", {}).get("world_size") != WORLD_SIZE
        or protocol.get("execution", {}).get("hf_endpoint") != HF_ENDPOINT
        or protocol.get("frozen_inputs", {}).get("retrieval_rows") != ROWS
        or protocol.get("data_lifecycle", {}).get(
            "already_open_retrieval_tensors_only"
        )
        is not True
    ):
        raise ValueError("Compatibility-bias protocol contract differs")
    return protocol


def _load_records(protocol: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expected_shards = protocol["frozen_inputs"]["retrieval_shards"]
    records: list[Mapping[str, Any]] = []
    for rank, expected in enumerate(expected_shards):
        path = INPUT_ROOT / str(expected["path"])
        if sha256_file(path) != expected["sha256"]:
            raise ValueError("Compatibility-bias retrieval shard hash differs")
        shard = json.loads(path.read_text(encoding="utf-8"))
        _validate_receipt(
            shard,
            scope="canonical_feature_shard_without_receipt",
            description=f"Compatibility-bias retrieval shard {rank}",
        )
        if (
            shard.get("receipt", {}).get("payload_sha256") != expected["receipt"]
            or shard.get("schema") != SHARD_SCHEMA
            or shard.get("split") != "retrieval"
            or shard.get("rank") != rank
            or shard.get("world_size") != WORLD_SIZE
            or shard.get("assignment") != "source_index_modulo_4"
            or shard.get("mechanics_or_causal_rows_opened") is not False
        ):
            raise ValueError("Compatibility-bias retrieval shard contract differs")
        for row in shard.get("rows", []):
            address = torch.tensor(row.get("write_address_full64"))
            receptance = torch.tensor(row.get("causal_prompt_boundary_receptance32"))
            if (
                row.get("schema") != FEATURE_SCHEMA
                or row.get("split") != "retrieval"
                or row.get("capture_rank") != rank
                or int(row.get("source_index", -1)) % WORLD_SIZE != rank
                or tuple(address.shape) != (MODULES, ADDRESS_DIM)
                or tuple(receptance.shape) != (MODULES, STATE_DIM)
                or not bool(torch.isfinite(address).all())
                or not bool(torch.isfinite(receptance).all())
                or row.get("read_writes_enabled") is not False
                or row.get("model_output_changed_by_observer") is not False
                or row.get("binder_or_feedback_installed") is not False
            ):
                raise ValueError("Compatibility-bias retrieval row contract differs")
            records.append(row)
    records.sort(key=lambda row: int(row["source_index"]))
    by_source = {int(row["source_index"]): row for row in records}
    if len(records) != ROWS or len(by_source) != ROWS:
        raise ValueError("Compatibility-bias retrieval coverage differs")
    for source, row in by_source.items():
        donor = int(row["donor_source_index"])
        donor_row = by_source.get(donor)
        if (
            donor_row is None
            or donor == source
            or int(donor_row["donor_source_index"]) != source
            or row.get("donor_row_sha256") != donor_row.get("row_sha256")
        ):
            raise ValueError("Compatibility-bias donor binding differs")
    return records


def _map_digest(
    maps: Mapping[str, alignment.FrozenMapWeights], module_names: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for name in module_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        for tensor in (maps[name].down, maps[name].up):
            value = tensor.detach().cpu().float().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_maps(
    protocol: Mapping[str, Any], device: torch.device
) -> tuple[list[str], dict[str, alignment.FrozenMapWeights]]:
    frozen = protocol["frozen_inputs"]
    if (
        sha256_file(MAP_FILE) != frozen["map_file_sha256"]
        or sha256_file(ALIGNMENT_SOURCE) != frozen["alignment_source_sha256"]
    ):
        raise ValueError("Compatibility-bias map dependency hash differs")
    payload = torch.load(MAP_FILE, map_location="cpu", weights_only=False)
    module_names = list(payload.get("module_names", []))
    expected_names = [
        f"model.language_model.layers.{layer}.self_attn" for layer in range(MODULES)
    ]
    if (
        payload.get("schema") != MAP_SCHEMA
        or module_names != expected_names
        or payload.get("rank") != 16
        or payload.get("ridge") != 1.0
        or payload.get("address_dim") != ADDRESS_DIM
        or payload.get("state_dim") != STATE_DIM
        or payload.get("frozen_map_digest") != frozen["frozen_map_digest"]
    ):
        raise ValueError("Compatibility-bias map contract differs")
    maps = {
        name: alignment.FrozenMapWeights(
            down=payload["maps"][name]["down"].float().to(device),
            up=payload["maps"][name]["up"].float().to(device),
        )
        for name in module_names
    }
    if _map_digest(maps, module_names) != frozen["frozen_map_digest"]:
        raise ValueError("Compatibility-bias frozen map digest differs")
    return module_names, maps


def _candidate_sources(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[int, int, int, int]]:
    by_source = {int(row["source_index"]): row for row in records}
    pairs = sorted(
        {
            tuple(sorted((source, int(row["donor_source_index"]))))
            for source, row in by_source.items()
        }
    )
    candidates: dict[int, tuple[int, int, int, int]] = {}
    for source, row in by_source.items():
        donor = int(row["donor_source_index"])
        own = tuple(sorted((source, donor)))
        distractors = [pair[0] for pair in pairs if pair != own]
        candidates[source] = (source, donor, distractors[0], distractors[1])
    return candidates


def _score(
    query: torch.Tensor,
    addresses: torch.Tensor,
    weights: alignment.FrozenMapWeights,
) -> torch.Tensor:
    direction = alignment.mapped_direction(addresses, weights)
    normalized_query = query / query.square().mean().clamp_min(1e-12).sqrt()
    return (direction * normalized_query.unsqueeze(0)).mean(dim=-1)


def _evaluate_local(
    records: Sequence[Mapping[str, Any]],
    module_names: Sequence[str],
    maps: Mapping[str, alignment.FrozenMapWeights],
    *,
    process_rank: int,
    device: torch.device,
) -> list[Mapping[str, Any]]:
    by_source = {int(row["source_index"]): row for row in records}
    candidates = _candidate_sources(records)
    permutation = torch.tensor(CANDIDATE_PERMUTATION, device=device)
    local_results = []
    for source in sorted(by_source):
        if source % WORLD_SIZE != process_rank:
            continue
        row = by_source[source]
        candidate_sources = candidates[source]
        per_anchor = {}
        for layer in ANCHORS:
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
            scores = _score(query, candidate_addresses, maps[name])
            permuted_scores = _score(
                query, candidate_addresses.index_select(0, permutation), maps[name]
            )
            layer_permuted_address = torch.tensor(
                row["write_address_full64"][(layer - 1) % MODULES],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            layer_permuted_score = _score(
                query, layer_permuted_address, maps[name]
            )[0]
            zero_address = torch.zeros(
                ADDRESS_DIM, dtype=torch.float32, device=device
            )
            zero_score = _score(query, zero_address.unsqueeze(0), maps[name])[0]
            correct = scores[0]
            strongest_wrong = scores[1:].max()
            per_anchor[str(layer)] = {
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
                "zero_address_slot_eligible": bool(zero_address.ne(0.0).any().item()),
                "finite": bool(
                    torch.isfinite(scores).all().item()
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


def _aggregate(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    gates = protocol["required_gates"]["per_anchor"]
    per_anchor = {}
    for layer in ANCHORS:
        values = [row["per_anchor"][str(layer)] for row in rows]
        top1 = sum(value["strict_correct_top1"] for value in values) / len(values)
        margins = [value["correct_over_strongest_wrong_margin"] for value in values]
        donor_margins = [
            value["correct_over_matched_donor_margin"] for value in values
        ]
        layer_margins = [
            value["correct_over_layer_permuted_margin"] for value in values
        ]
        checks = {
            "strict_correct_top1_fraction": top1
            >= gates["strict_correct_top1_fraction"],
            "correct_over_strongest_wrong_mean_margin": (
                sum(margins) / len(margins)
                >= gates["correct_over_strongest_wrong_mean_margin"]
            ),
            "correct_over_matched_donor_positive_fraction": (
                sum(margin > 0.0 for margin in donor_margins) / len(donor_margins)
                >= gates["correct_over_matched_donor_positive_fraction"]
            ),
            "correct_over_layer_permuted_positive_fraction": (
                sum(margin > 0.0 for margin in layer_margins) / len(layer_margins)
                >= gates["correct_over_layer_permuted_positive_fraction"]
            ),
            "finite": all(value["finite"] for value in values),
            "candidate_permutation_exact": all(
                value["candidate_permutation_exact"] for value in values
            ),
            "zero_address_raw_bias_exact_zero": all(
                value["zero_address_raw_bias_exact_zero"] for value in values
            ),
            "zero_address_slot_eligible": not any(
                value["zero_address_slot_eligible"] for value in values
            ),
        }
        per_anchor[str(layer)] = {
            "strict_correct_top1_fraction": top1,
            "correct_over_strongest_wrong_mean_margin": sum(margins) / len(margins),
            "correct_over_strongest_wrong_minimum_margin": min(margins),
            "correct_over_matched_donor_positive_fraction": (
                sum(margin > 0.0 for margin in donor_margins) / len(donor_margins)
            ),
            "correct_over_matched_donor_mean_margin": sum(donor_margins)
            / len(donor_margins),
            "correct_over_layer_permuted_positive_fraction": (
                sum(margin > 0.0 for margin in layer_margins) / len(layer_margins)
            ),
            "correct_over_layer_permuted_mean_margin": sum(layer_margins)
            / len(layer_margins),
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed_layers = sum(value["passed"] for value in per_anchor.values())
    return {
        "per_anchor": per_anchor,
        "passed_layers": passed_layers,
        "required_passed_layers": protocol["required_gates"][
            "minimum_anchor_layers_passing"
        ],
        "passed": passed_layers
        >= protocol["required_gates"]["minimum_anchor_layers_passing"],
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Compatibility-bias screen requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != HF_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {HF_ENDPOINT}")
    if (
        len(context.rank_devices) != WORLD_SIZE
        or len({device["device_uuid"] for device in context.rank_devices})
        != WORLD_SIZE
        or not all("A100" in device["device_name"] for device in context.rank_devices)
    ):
        raise RuntimeError("Compatibility-bias screen requires four distinct A100s")
    protocol = validate_protocol()
    protocol_hash = sha256_file(PROTOCOL)
    records = _load_records(protocol)
    module_names, maps = _load_maps(protocol, context.device)
    input_digest = canonical_sha256(records)
    distributed.require_consensus(
        context, input_digest, description="compatibility-bias input records"
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
        raise RuntimeError("Compatibility-bias distributed result coverage differs")
    analysis = _aggregate(rows, protocol)
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "compatibility_bias_passed_live_mechanics_protocol_authorized"
            if analysis["passed"]
            else "compatibility_bias_failed_frozen_local_route_retired"
        ),
        "passed": analysis["passed"],
        "protocol_file_sha256": protocol_hash,
        "protocol_receipt": protocol["receipt"]["payload_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "alignment_source_sha256": sha256_file(ALIGNMENT_SOURCE),
        "input_root": str(INPUT_ROOT.relative_to(SCRIPT_DIR)),
        "input_records_digest": input_digest,
        "map_file_sha256": sha256_file(MAP_FILE),
        "frozen_map_digest": protocol["frozen_inputs"]["frozen_map_digest"],
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
        "payload_sha256": canonical_sha256(result),
    }
    result_digest = canonical_sha256(result)
    distributed.require_consensus(
        context, result_digest, description="compatibility-bias result"
    )
    if context.is_primary:
        if output_dir.exists():
            raise ValueError(f"Compatibility-bias output must be fresh: {output_dir}")
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
        raise RuntimeError("Compatibility-bias screen must run under torchrun")
    try:
        result = run(context=context, output_dir=arguments.output_dir)
        if context.is_primary:
            print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    finally:
        distributed.destroy_distributed_training(context)


if __name__ == "__main__":
    raise SystemExit(main())
