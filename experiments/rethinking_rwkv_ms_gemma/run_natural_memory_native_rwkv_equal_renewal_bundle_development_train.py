#!/usr/bin/env python3
"""Equal-weight control for the weighted renewal split."""

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

SCHEMA = "rwkv_ms_natural_memory_native_rwkv_equal_renewal_bundle_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_equal_renewal_bundle_development_protocol_v1.json"
PROTOCOL_FILE_SHA256 = "dfcc73d4bdd1f20e872e588afdc091226d549ecf7dfd230a0f1a6f65a63cb5d8"
PROTOCOL_PAYLOAD_SHA256 = "39523b5736b1a1416214af3252ba1209fa5b7e9bdda8394551e17f67d9e8c3e2"
DEFAULT_MATERIALIZATION = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_weighted_renewal_bundle_development_v1"
DEFAULT_OUTPUT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_equal_renewal_bundle_development_train_v1"
SEED = 20260828
SPLIT_SALT = "rwkv-source-weighted-renewal-bundle-open-pair-split-v1:"
TRAIN_PAIRS = 24
HELDOUT_PAIRS = 16
TRAIN_ROWS = 48
HELDOUT_ROWS = 32
UPDATES = 48
ROUTE_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
SPLIT_PAYLOAD_SHA256 = "ac742277d66f3f199b54c8a7a6a5db071f09789d5941275f03c42a9f3c2e6c09"
TARGET_PAYLOAD_SHA256 = "e0fca0d7b85ae27d4c161d5f60c464e00690c6184511d978bf320593b95fc573"


def validate_protocol() -> Mapping[str, Any]:
    if multi.base.sha256_file(PROTOCOL) != PROTOCOL_FILE_SHA256:
        raise ValueError("Equal renewal protocol file hash differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    multi.base.validate_receipt(protocol, scope="canonical_protocol_without_receipt", description="Equal renewal protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("open_development_only") is not True:
        raise ValueError("Equal renewal protocol identity differs")
    if protocol.get("receipt", {}).get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("Equal renewal protocol receipt differs")
    if tuple(protocol.get("route_aggregation", {}).get("weights", ())) != ROUTE_WEIGHTS:
        raise ValueError("Equal renewal route weights differ")
    if protocol.get("split", {}).get("payload_sha256") != SPLIT_PAYLOAD_SHA256:
        raise ValueError("Equal renewal split differs")
    return protocol


def make_router(maps: Mapping[int, Any], device: torch.device) -> SourceCumulativeResidualRouter:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return SourceCumulativeResidualRouter(
        maps=maps,
        anchor_layers=multi.base.ANCHORS,
        compatibility_scale=multi.base.COMPATIBILITY_SCALE,
        residual_gain=multi.base.RESIDUAL_GAIN,
        required_receptance_calls=2,
        route_weights=ROUTE_WEIGHTS,
        outer_ffn=SourceBoundMultiAnchorBundleFFN(
            state_dim=multi.base.NATIVE_READ_DIM,
            hidden_dim=multi.base.HIDDEN_DIM,
            anchor_count=len(multi.base.ANCHORS),
            bottleneck_dim=multi.base.BOTTLENECK_DIM,
        ),
    ).to(device)


def configure() -> None:
    multi.configure()
    base = multi.base
    for name, value in {
        "SCHEMA": SCHEMA, "STEP_SCHEMA": STEP_SCHEMA, "INPUT_SCHEMA": INPUT_SCHEMA,
        "SPLIT_SCHEMA": SPLIT_SCHEMA, "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "PROTOCOL": PROTOCOL, "PROTOCOL_FILE_SHA256": PROTOCOL_FILE_SHA256,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "DEFAULT_MATERIALIZATION": DEFAULT_MATERIALIZATION, "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "SEED": SEED, "SPLIT_SALT": SPLIT_SALT, "TRAIN_PAIRS": TRAIN_PAIRS,
        "HELDOUT_PAIRS": HELDOUT_PAIRS, "TRAIN_ROWS": TRAIN_ROWS,
        "HELDOUT_ROWS": HELDOUT_ROWS, "UPDATES": UPDATES,
        "DISCRIMINATIVE_TARGET_PAYLOAD_SHA256": TARGET_PAYLOAD_SHA256,
    }.items():
        setattr(base, name, value)
    base.development_materializer = development_materializer
    base.validate_protocol = validate_protocol
    base.make_router = make_router
    base.__file__ = str(Path(__file__).resolve())


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return multi.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
