#!/usr/bin/env python3
"""Evaluate addressed-affine RWKV fusion on native generation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_affine_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_vector_gate_eval as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_affine_eval.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_addressed_affine_eval_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_addressed_affine_generation_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "050a63b99bc5259bcb74053e435d4c207410fe4bc88e490a2bfde421d20b8a56"
TRAINING_PROTOCOL_PAYLOAD_SHA256 = training.PROTOCOL_PAYLOAD_SHA256
TRAIN_RESULT_FILE_SHA256 = (
    "096e20bb01abbe86689745379b12b3f1b8d5de32c7be0ba682793855a85e0e2d"
)
TRAIN_RESULT_RECEIPT = (
    "c74dc75bee63bc3cae65671c0978f92969bb08e2e69d6fa1c8ea3c28c232a4e6"
)
ADAPTER_CONFIG_SHA256 = (
    "28830f91f455c77f6b565cacb3069262d991d2030e3564961e0c957f13c8245f"
)
TRAIN_RESULT_STATUS = training.PASS_STATUS
TRAINING_UPDATES = training.UPDATES
SELECTED_CANDIDATE = training.SELECTED_CANDIDATE
RUNTIME_HYBRID_MODE = "addressed_affine"
SERIALIZED_HYBRID_MODE = "recurrent_value"
RUNNER_BINDING_PATH = Path(__file__)
HF_MIRROR_ENDPOINT = base.HF_MIRROR_ENDPOINT
WORLD_SIZE = base.WORLD_SIZE
SEED = training.SEED
EVALUATION_ROWS = base.EVALUATION_ROWS
AUTHORIZED_ROWS_PAYLOAD_SHA256 = base.AUTHORIZED_ROWS_PAYLOAD_SHA256
DATASET_RELATIVE_PATH = base.DATASET_RELATIVE_PATH
DATASET_SHA256 = base.DATASET_SHA256
STATE_CONDITIONS = base.STATE_CONDITIONS
CONDITIONS = base.CONDITIONS
BATCH_CONDITIONS = base.BATCH_CONDITIONS


def canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


@contextmanager
def base_bindings() -> Iterator[None]:
    bindings = {
        "SCHEMA": SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "TRAINING_PROTOCOL_PAYLOAD_SHA256": TRAINING_PROTOCOL_PAYLOAD_SHA256,
        "TRAIN_RESULT_FILE_SHA256": TRAIN_RESULT_FILE_SHA256,
        "TRAIN_RESULT_RECEIPT": TRAIN_RESULT_RECEIPT,
        "ADAPTER_CONFIG_SHA256": ADAPTER_CONFIG_SHA256,
        "TRAIN_RESULT_STATUS": TRAIN_RESULT_STATUS,
        "TRAINING_UPDATES": TRAINING_UPDATES,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "RUNTIME_HYBRID_MODE": RUNTIME_HYBRID_MODE,
        "SERIALIZED_HYBRID_MODE": SERIALIZED_HYBRID_MODE,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "SEED": SEED,
        "training": training,
    }
    previous = {name: getattr(base, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def validate_protocol() -> Mapping[str, Any]:
    with base_bindings():
        return base.validate_protocol()


def validate_train_result(
    result_path: Path,
    *,
    adapter_dir: Path,
) -> Mapping[str, Any]:
    with base_bindings():
        return base.validate_train_result(result_path, adapter_dir=adapter_dir)


def restore_and_assert_addressed_affine_modules(model) -> None:
    with base_bindings():
        base._restore_and_assert_vector_gate_modules(model)


@contextmanager
def evaluation_bindings() -> Iterator[None]:
    with base_bindings(), base.evaluation_bindings():
        yield


def run(args: argparse.Namespace) -> int:
    with base_bindings():
        return base.run(args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partitions-per-shard", type=int, default=1)
    parser.add_argument("--base-model", type=Path, default=training.shared.BASE_MODEL)
    parser.add_argument("--dataset-root", type=Path, default=training.shared.DATASET_ROOT)
    parser.add_argument("--train-result", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
