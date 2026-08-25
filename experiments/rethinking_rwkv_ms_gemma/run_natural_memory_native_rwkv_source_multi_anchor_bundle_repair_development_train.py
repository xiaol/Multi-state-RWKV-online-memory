#!/usr/bin/env python3
"""Train the source-identity bundle on the second fresh open reservation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    materialize_natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development as development_materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_source_multi_anchor_bundle_development_train as multi,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development.v1"
STEP_SCHEMA = f"{SCHEMA}.step"
INPUT_SCHEMA = f"{SCHEMA}.input"
SPLIT_SCHEMA = f"{SCHEMA}.split"
PROTOCOL_SCHEMA = f"{SCHEMA}.protocol"
PROTOCOL = SCRIPT_DIR / (
    "natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development_protocol_v1.json"
)
PROTOCOL_FILE_SHA256 = (
    "8e00291d70adc361ea591069c2e251c6dd7552afc4208b667ee8fd8ea0c15e09"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "7487c72aacce71537341215053e867c25b49fca9f539c9cd4f22abd8ad44c88e"
)
DEFAULT_MATERIALIZATION = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development_v1"
)
DEFAULT_OUTPUT = SCRIPT_DIR / (
    "local_artifacts/"
    "natural_memory_native_rwkv_source_multi_anchor_bundle_repair_development_train_v1"
)
SEED = 20260829
SPLIT_SALT = "rwkv-source-multi-anchor-bundle-open-repair-v1:"
SPLIT_PAYLOAD_SHA256 = (
    "2ff976006e28f7feb06a995935942a18e0f0b5a1f1296b9ef90a9fc904ba918b"
)
DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
    "b617872a2546408ddfdb0d4b86875c4a167d32ec53cbcab4aabb78ba99f80b7d"
)


def configure() -> None:
    multi.SCHEMA = SCHEMA
    multi.STEP_SCHEMA = STEP_SCHEMA
    multi.INPUT_SCHEMA = INPUT_SCHEMA
    multi.SPLIT_SCHEMA = SPLIT_SCHEMA
    multi.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    multi.PROTOCOL = PROTOCOL
    multi.PROTOCOL_FILE_SHA256 = PROTOCOL_FILE_SHA256
    multi.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    multi.DEFAULT_MATERIALIZATION = DEFAULT_MATERIALIZATION
    multi.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    multi.SEED = SEED
    multi.SPLIT_SALT = SPLIT_SALT
    multi.SPLIT_PAYLOAD_SHA256 = SPLIT_PAYLOAD_SHA256
    multi.DISCRIMINATIVE_TARGET_PAYLOAD_SHA256 = (
        DISCRIMINATIVE_TARGET_PAYLOAD_SHA256
    )
    multi.development_materializer = development_materializer
    multi.configure()
    multi.base.development_materializer = development_materializer
    multi.base.validate_protocol = multi.validate_protocol


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return multi.base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
