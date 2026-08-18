#!/usr/bin/env python3
"""Probe whether aligned-vector RWKV states expose a layer-aware scalar identity.

This is deliberately a read-only screen.  It never changes the adapter or the
projected carrier; it records per-layer recurrent-state norms for the locked
220-row native endpoint and its donor/permutation controls.  The output is
used to decide whether a tiny state-conditioned abstention head is worth a
generation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from common import load_model_and_tokenizer  # noqa: E402
from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_benchmark_eval as base,
)


WORLD_SIZE = 4
SCHEMA = "rwkv_ms_native_state_scalar_probe.v1"
DATASET_RELATIVE_PATH = "v4-scene-boundary-detection/train_derived_development.jsonl"
DATASET_SHA256 = base.DATASET_SHA256
ADAPTER_CONFIG_SHA256 = "39dd450d660cd139f34f2aeb5ca1f7a068ad41cd4be684069107f21195d41a1e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def recurrent_features(
    state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> list[float]:
    recurrent, _ = base.split_state(state, module_names)
    values: list[float] = []
    for name in module_names:
        tensor = recurrent[name].float()
        values.append(float(torch.linalg.vector_norm(tensor).item()))
    return values


def write_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def run(args: argparse.Namespace) -> int:
    if args.shard_index < 0 or args.shard_index >= WORLD_SIZE:
        raise ValueError("state scalar probe requires one of four shards")
    base_model = args.base_model.expanduser().resolve(strict=True)
    adapter_dir = args.adapter_dir.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    dataset_file = dataset_root / DATASET_RELATIVE_PATH
    if sha256_file(dataset_file) != DATASET_SHA256:
        raise ValueError("native development dataset hash differs")
    if sha256_file(adapter_dir / "delta_mem_config.json") != ADAPTER_CONFIG_SHA256:
        raise ValueError("precision adapter config differs")

    model, tokenizer = load_model_and_tokenizer(
        base_model=str(base_model),
        memory_dir=str(adapter_dir),
        device=args.device,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )
    metadata = base.raw_line_metadata(dataset_file)
    rows = base.parse_authorized_rows(metadata)
    counts = base.prompt_token_counts(tokenizer, rows)
    mapping = base.build_donor_mapping(rows, counts)
    by_index = {int(row["source_index"]): row for row in rows}
    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    if len(module_names) != 42:
        raise ValueError(f"expected 42 wrapped layers, found {len(module_names)}")
    output = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}.jsonl"
    existing = {
        int(json.loads(line)["source_index"])
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if output.is_file() else set()
    shard_rows = [
        row for row in rows if int(row["source_index"]) % WORLD_SIZE == args.shard_index
    ]
    binding = {
        "schema": SCHEMA,
        "world_size": WORLD_SIZE,
        "shard_index": args.shard_index,
        "authorized_rows": base.EVALUATION_ROWS,
        "authorized_rows_payload_sha256": base.AUTHORIZED_ROWS_PAYLOAD_SHA256,
        "dataset_file": str(dataset_file),
        "dataset_sha256": DATASET_SHA256,
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
        "module_names_sha256": canonical_sha256(list(module_names)),
        "conditions": ["correct", "zero", "matched_donor", "layer_permuted"],
    }
    binding_path = args.output_root.expanduser().resolve() / f"shard-{args.shard_index}.binding.json"
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    if binding_path.is_file():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
            raise ValueError("state scalar probe binding differs on resume")
    else:
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")

    for ordinal, row in enumerate(shard_rows, start=1):
        source = int(row["source_index"])
        if source in existing:
            continue
        donor = by_index[mapping[source]]
        correct = base.prime_state(model, tokenizer, row, device=args.device)
        donor_state = base.prime_state(model, tokenizer, donor, device=args.device)
        correct_features = recurrent_features(correct, module_names)
        donor_features = recurrent_features(donor_state, module_names)
        # A cyclic layer permutation is exactly the endpoint intervention.
        permuted = [correct_features[(i + 1) % len(correct_features)] for i in range(len(correct_features))]
        record = {
            "schema": SCHEMA,
            "source_index": source,
            "row_sha256": row["row_sha256"],
            "donor_source_index": int(donor["source_index"]),
            "donor_row_sha256": donor["row_sha256"],
            "gold": sorted(base.recovery.strict_gold_boundaries(row["gold"])),
            "correct": correct_features,
            "zero": [0.0] * len(correct_features),
            "matched_donor": donor_features,
            "layer_permuted": permuted,
        }
        write_record(output, record)
        print(
            f"STATE_SCALAR_PROBE shard={args.shard_index} row={source} "
            f"ordinal={ordinal}/{len(shard_rows)}",
            flush=True,
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
