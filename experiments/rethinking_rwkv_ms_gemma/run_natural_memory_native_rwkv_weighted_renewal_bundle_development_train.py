#!/usr/bin/env python3
"""Train the frozen multi-anchor value bundle with weighted source renewal."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.cumulative_rwkv_residual import SourceBoundMultiAnchorBundleFFN, SourceCumulativeResidualRouter
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_rwkv_source_multi_anchor_bundle_development_train as multi
from experiments.rethinking_rwkv_ms_gemma import materialize_natural_memory_native_rwkv_weighted_renewal_bundle_development as development_materializer


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_weighted_renewal_bundle_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_weighted_renewal_bundle_development_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "403373a6ea5ee0c2893bb09bed4e58145daea16d99b9d3f2888276e2a5e9437e"
PROTOCOL_PAYLOAD_SHA256 = "1f43584b8440a2e2ae9ea97cb7ebcc74602e8b79168603321079cb56820c74df"
DEFAULT_MATERIALIZATION = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_weighted_renewal_bundle_development_v1"
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_weighted_renewal_bundle_development_train_v1"
SEED = 20260828
SPLIT_SALT = "rwkv-source-weighted-renewal-bundle-open-pair-split-v1:"
TRAIN_PAIRS = 24
HELDOUT_PAIRS = 16
TRAIN_ROWS = 48
HELDOUT_ROWS = 32
UPDATES = 48
ROUTE_WEIGHTS = (0.25, 0.25, 0.5)


def validate_protocol() -> Mapping[str, Any]:
    if multi.base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Weighted renewal protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    multi.base.validate_receipt(protocol, scope="canonical_protocol_without_receipt", description="Weighted renewal protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("open_development_only") is not True:
        raise ValueError("Weighted renewal protocol identity differs")
    if protocol.get("receipt", {}).get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Weighted renewal protocol receipt differs")
    if tuple(protocol.get("route_aggregation", {}).get("weights", ())) != ROUTE_WEIGHTS:
        raise ValueError("Weighted renewal route weights differ")
    if protocol.get("split") != {
        "heldout_pairs": HELDOUT_PAIRS,
        "manifest_sha256": development_materializer.sha256_bytes((DEFAULT_MATERIALIZATION / "manifest.json").read_bytes()),
        "payload_sha256": "b145c49366faa297040281368d5d3a711d221f9e7116be76ad0ff8e1156826ae",
        "train_pairs": TRAIN_PAIRS,
    }:
        raise ValueError("Weighted renewal split binding differs")
    if protocol.get("protected_mechanics_authorized") is not False or protocol.get("protected_causal_authorized") is not False or protocol.get("native_benchmark_authorized") is not False:
        raise ValueError("Weighted renewal access policy differs")
    return protocol


def make_router(maps: Mapping[int, Any], device: torch.device) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    outer_ffn = SourceBoundMultiAnchorBundleFFN(
        state_dim=multi.base.NATIVE_READ_DIM,
        hidden_dim=multi.base.HIDDEN_DIM,
        anchor_count=len(multi.base.ANCHORS),
        bottleneck_dim=multi.base.BOTTLENECK_DIM,
    )
    return SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=multi.base.ANCHORS,
        compatibility_scale=multi.base.COMPATIBILITY_SCALE,
        residual_gain=multi.base.RESIDUAL_GAIN,
        required_receptance_calls=2,
        route_weights=ROUTE_WEIGHTS,
        outer_ffn=outer_ffn,
    ).to(device)


def configure() -> None:
    multi.configure()
    base = multi.base
    base.SCHEMA = SCHEMA
    base.STEP_SCHEMA = STEP_SCHEMA
    base.INPUT_SCHEMA = INPUT_SCHEMA
    base.SPLIT_SCHEMA = SPLIT_SCHEMA
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    base.DEFAULT_MATERIALIZATION = DEFAULT_MATERIALIZATION
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.SEED = SEED
    base.SPLIT_SALT = SPLIT_SALT
    base.TRAIN_PAIRS = TRAIN_PAIRS
    base.HELDOUT_PAIRS = HELDOUT_PAIRS
    base.TRAIN_ROWS = TRAIN_ROWS
    base.HELDOUT_ROWS = HELDOUT_ROWS
    base.UPDATES = UPDATES
    base.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        "548388d866d59e6cfec2425f60d982db4ca8166a9e8cc42c33847d09d442cb5f"
    )
    base.development_materializer = development_materializer
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return multi.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
