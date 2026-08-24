#!/usr/bin/env python3
"""Screen nonlinear co-rotated address keys on frozen open-split captures."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_rwkv_continuous_write_open_fit as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_address_decoded_reconstruction as address_decode,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_virtual_kv_identity_v5"
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_corotated_key_v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_corotated_key_protocol_v1.json"
MATERIALIZATION = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_continuous_write_open_fit_v1"
BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
LAYERS = (5, 11, 17, 23)
SEEDS = (223, 227, 229)
STEPS = 800
HIDDEN = 256
LEARNING_RATE = 1e-3
TEMPERATURE = 8.0
FIT_ROWS = 64
RETRIEVAL_ROWS = 32
SLOTS = 4


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


def _validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", None)
    expected = {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": canonical_sha256(unsigned),
    }
    if receipt != expected:
        raise ValueError("co-rotated key protocol receipt differs")
    if (
        protocol.get("schema")
        != "rwkv_ms_natural_memory_native_rwkv_corotated_key_protocol.v1"
        or protocol.get("architecture", {}).get("anchor_layers") != list(LAYERS)
        or protocol.get("training", {}).get("seeds") != list(SEEDS)
        or protocol.get("data_lifecycle", {}).get(
            "already_captured_open_tensors_only"
        )
        is not True
    ):
        raise ValueError("co-rotated key protocol contract differs")
    return protocol


def _load_records(split: str) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for path_value in sorted(glob.glob(str(CAPTURE_ROOT / f"{split}-shard-*.pt"))):
        path = Path(path_value)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("split") != split or payload.get("world_size") != 4:
            raise ValueError("co-rotated key input shard contract differs")
        for record in payload.get("records", []):
            source = int(record["source_index"])
            if source in records:
                raise ValueError("co-rotated key input repeats a source")
            records[source] = record
    expected = FIT_ROWS if split == "fit" else RETRIEVAL_ROWS
    if len(records) != expected:
        raise ValueError(f"co-rotated key {split} coverage differs")
    return records


def _active_address(record: Mapping[str, Any], layer: int) -> torch.Tensor:
    occupied = record["features"]["occupied"][layer, 0]
    active = occupied.nonzero(as_tuple=False)
    if active.size(0) != 1:
        raise ValueError("co-rotated key rows require one occupied slot")
    return record["features"]["keys"][layer, 0, active[0, 0]].float()


def _query(record: Mapping[str, Any], layer: int) -> torch.Tensor:
    query = record["attention"][layer]["query"][0, :, 0].float()
    shape = record["module_shapes"][layer]
    kv_heads = int(shape["num_kv_heads"])
    query_heads = int(shape["num_query_heads"])
    head_dim = int(shape["head_dim"])
    return query.reshape(kv_heads, query_heads // kv_heads, head_dim).mean(dim=1)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _predictor_positions(split: str) -> Mapping[int, int]:
    manifest = address_decode._load_manifest_only(MATERIALIZATION)
    rows = materializer._read_bundle(MATERIALIZATION, manifest, split)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    examples = address_decode.retrieval._encode_rows(tokenizer, rows)
    positions = {}
    for source, example in examples.items():
        labels = torch.tensor(example.labels).unsqueeze(0)
        _, predictor = address_decode.retrieval.first_prompt_boundary(labels)
        positions[int(source)] = int(predictor)
    return positions


def _unrotated_queries(
    records: Mapping[int, Mapping[str, Any]],
    *,
    split: str,
    layer: int,
) -> Mapping[int, torch.Tensor]:
    positions = _predictor_positions(split)
    config = AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True).text_config
    layer_type = str(config.layer_types[layer])
    rotary = Gemma4TextRotaryEmbedding(config, layer_type=layer_type)
    result = {}
    for source, record in records.items():
        post_rope = _query(record, layer)
        position_ids = torch.tensor([[positions[source]]], dtype=torch.long)
        cos, sin = rotary(post_rope.unsqueeze(0), position_ids, layer_type=layer_type)
        cos = cos[0, 0].float()
        sin = sin[0, 0].float()
        result[source] = post_rope * cos - _rotate_half(post_rope) * sin
    return result


def _candidate_sources(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[int, tuple[int, int, int, int]]:
    ordered = sorted(records)
    groups = sorted(
        {frozenset((source, int(records[source]["donor_source_index"]))) for source in ordered},
        key=lambda group: min(group),
    )
    result = {}
    for source in ordered:
        donor = int(records[source]["donor_source_index"])
        own = frozenset((source, donor))
        distractors = [min(group) for group in groups if group != own]
        result[source] = (source, donor, distractors[0], distractors[1])
    return result


class KeyMap(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(64, HIDDEN, bias=False)
        self.up = nn.Linear(HIDDEN, output_dim, bias=False)

    def forward(self, address: torch.Tensor) -> torch.Tensor:
        normalized = address / address.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        return self.up(F.silu(self.down(normalized)))


def _dataset(
    records: Mapping[int, Mapping[str, Any]], layer: int, *, split: str
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = _candidate_sources(records)
    queries_by_source = _unrotated_queries(records, split=split, layer=layer)
    addresses = []
    queries = []
    for source in sorted(records):
        addresses.append(
            torch.stack(
                [_active_address(records[candidate], layer) for candidate in candidates[source]]
            )
        )
        queries.append(queries_by_source[source])
    return torch.stack(addresses), torch.stack(queries)


def _logits(model: KeyMap, addresses: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    rows, slots, _ = addresses.shape
    kv_heads, head_dim = queries.shape[1:]
    keys = model(addresses.reshape(rows * slots, -1)).reshape(
        rows, slots, kv_heads, head_dim
    )
    keys = keys / keys.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    normalized_query = queries / queries.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    return (normalized_query[:, None] * keys).mean(dim=-1).mean(dim=-1) * TEMPERATURE


def _metrics(logits: torch.Tensor) -> Mapping[str, float]:
    correct = logits[:, 0]
    wrong = logits[:, 1:].max(dim=-1).values
    margin = correct - wrong
    return {
        "strict_top1": float(margin.gt(0).float().mean().item()),
        "mean_margin": float(margin.mean().item()),
        "minimum_margin": float(margin.min().item()),
        "maximum_margin": float(margin.max().item()),
    }


def _train_one(
    fit_addresses: torch.Tensor,
    fit_queries: torch.Tensor,
    retrieval_addresses: torch.Tensor,
    retrieval_queries: torch.Tensor,
    *,
    seed: int,
) -> tuple[KeyMap, Mapping[str, Any]]:
    torch.manual_seed(seed)
    output_dim = fit_queries.size(1) * fit_queries.size(2)
    model = KeyMap(output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    targets = torch.zeros(fit_addresses.size(0), dtype=torch.long)
    for _ in range(STEPS):
        logits = _logits(model, fit_addresses, fit_queries)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        fit_logits = _logits(model, fit_addresses, fit_queries)
        retrieval_logits = _logits(model, retrieval_addresses, retrieval_queries)
    return model.eval(), {
        "seed": seed,
        "steps": STEPS,
        "final_fit_loss": float(F.cross_entropy(fit_logits, targets).item()),
        "fit": _metrics(fit_logits),
        "retrieval": _metrics(retrieval_logits),
    }


def run(output_dir: Path) -> Mapping[str, Any]:
    protocol = _validate_protocol()
    fit_records = _load_records("fit")
    retrieval_records = _load_records("retrieval")
    if output_dir.exists():
        raise ValueError(f"co-rotated key output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    per_layer = {}
    artifacts = []
    for layer in LAYERS:
        fit_addresses, fit_queries = _dataset(fit_records, layer, split="fit")
        retrieval_addresses, retrieval_queries = _dataset(
            retrieval_records, layer, split="retrieval"
        )
        seed_runs = []
        for seed in SEEDS:
            model, metrics = _train_one(
                fit_addresses,
                fit_queries,
                retrieval_addresses,
                retrieval_queries,
                seed=seed,
            )
            artifact = output_dir / f"layer-{layer}-seed-{seed}.pt"
            torch.save(
                {
                    "schema": "rwkv_ms_corotated_virtual_key_map.v1",
                    "layer": layer,
                    "seed": seed,
                    "hidden": HIDDEN,
                    "temperature": TEMPERATURE,
                    "state_dict": model.state_dict(),
                },
                artifact,
            )
            artifacts.append({"path": artifact.name, "sha256": sha256_file(artifact)})
            seed_runs.append(metrics)
        seed_top1 = [run["retrieval"]["strict_top1"] for run in seed_runs]
        seed_margins = [run["retrieval"]["mean_margin"] for run in seed_runs]
        per_layer[str(layer)] = {
            "seeds": seed_runs,
            "minimum_seed_strict_top1": min(seed_top1),
            "mean_seed_strict_top1": sum(seed_top1) / len(seed_top1),
            "minimum_seed_mean_margin": min(seed_margins),
            "passed": min(seed_top1) >= 0.75 and min(seed_margins) >= 0.05,
        }
    passed_layers = sum(layer["passed"] is True for layer in per_layer.values())
    result = {
        "schema": "rwkv_ms_natural_memory_native_rwkv_corotated_key_screen.v1",
        "status": (
            "corotated_nonlinear_key_passed_mechanics_protocol_authorized"
            if passed_layers >= 3
            else "corotated_nonlinear_key_failed_family_retired"
        ),
        "passed": passed_layers >= 3,
        "capture_root": str(CAPTURE_ROOT.relative_to(SCRIPT_DIR)),
        "capture_result_sha256": sha256_file(CAPTURE_ROOT / "result.json"),
        "capture_result_receipt": "75209e6996b5e17035435757a0f9e3a0a6a3299faa2198e52ce7288b064c737a",
        "protocol_receipt": protocol["receipt"]["payload_sha256"],
        "fit_rows": FIT_ROWS,
        "retrieval_rows": RETRIEVAL_ROWS,
        "layers": list(LAYERS),
        "seeds": list(SEEDS),
        "architecture": {
            "address_dim": 64,
            "hidden": HIDDEN,
            "activation": "silu",
            "bias": False,
            "key_normalization": "per-KV-head RMS sphere",
            "query_normalization": "per-KV-head RMS sphere",
            "deployment_position": "unrotated key followed by current-query-position RoPE",
            "co_rotation_logit_shift_invariant_by_construction": True,
        },
        "training": {
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "temperature": TEMPERATURE,
            "objective": "four-way candidate cross entropy on FIT only",
        },
        "per_layer": per_layer,
        "passed_layers": passed_layers,
        "required_passed_layers": 3,
        "artifacts": artifacts,
        "already_captured_open_tensors_only": True,
        "mechanics_causal_generation_or_native_bytes_opened": False,
        "model_parameters_updated": False,
        "full_bandwidth_feedback_installed": False,
        "native_gain_claimed": False,
        "sota_claimed": False,
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0 if result.get("passed") else 1)
