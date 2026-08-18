#!/usr/bin/env python3
"""Evaluate the precision-trained aligned-vector gate under native controls."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_vector_gate_eval as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_eval.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_precision_unlikelihood_eval_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_aligned_vector_gate_precision_unlikelihood_generation_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "3d31d3831fa9e741870aea7a23d13c8f74c92c16339e62f8f90ba3412958c70e"
TRAINING_PROTOCOL_PAYLOAD_SHA256 = training.PROTOCOL_PAYLOAD_SHA256
TRAIN_RESULT_FILE_SHA256 = "dd93ffad256ec887dc2c688f0de46485735f8a6ba3f0913fb7d735f7292d7412"
TRAIN_RESULT_RECEIPT = "c765c24c89a4b0569d8b703cdaead6494a0290030abac84e6608a91ee5cc26d8"
ADAPTER_CONFIG_SHA256 = "39dd450d660cd139f34f2aeb5ca1f7a068ad41cd4be684069107f21195d41a1e"
TRAIN_RESULT_STATUS = "aligned_vector_gate_precision_unlikelihood_training_passed_generation_authorized"
TRAINING_UPDATES = training.UPDATES
SELECTED_CANDIDATE = training.aligned.SELECTED_CANDIDATE
RUNTIME_HYBRID_MODE = "aligned_vector_gate"
SERIALIZED_HYBRID_MODE = "recurrent_value"
RUNNER_BINDING_PATH = Path(__file__)
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
WORLD_SIZE = 4
SEED = training.SEED
EVALUATION_ROWS = base.EVALUATION_ROWS
AUTHORIZED_ROWS_PAYLOAD_SHA256 = base.AUTHORIZED_ROWS_PAYLOAD_SHA256
DATASET_RELATIVE_PATH = base.DATASET_RELATIVE_PATH
DATASET_SHA256 = base.DATASET_SHA256
STATE_CONDITIONS = base.STATE_CONDITIONS
CONDITIONS = base.CONDITIONS
BATCH_CONDITIONS = base.BATCH_CONDITIONS


def canonical_sha256(value: Any) -> str:
    return training.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return training.sha256_file(path)


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


def validate_train_result(result_path: Path, *, adapter_dir: Path) -> Mapping[str, Any]:
    with base_bindings():
        return base.validate_train_result(result_path, adapter_dir=adapter_dir)


@contextmanager
def evaluation_bindings() -> Iterator[None]:
    with base_bindings(), base.evaluation_bindings():
        yield


def run(args: Any) -> int:
    with base_bindings():
        return base.run(args)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    parser = base.parse_args(argv)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
