#!/usr/bin/env python3
"""Train recurrent RWKV readouts after rejecting non-finite gradient rows."""

from __future__ import annotations

from contextlib import contextmanager
import json
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

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_positive_only_causal_train as positive,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_positive_only_filtered_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_positive_only_filtered_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_positive_only_filtered_causal_train_input.v1"
)
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_positive_only_filtered_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "131ff0c1108e97ac9d619fd5fcb0dbf4fe29b680c4acfb9e1ba25a1c9e56450e"
)
SEED = positive.SEED
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
FAILED_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_positive_only_causal_train_failed_step2_v1/"
    "input_binding.json"
)
FAILED_INPUT_BINDING_SHA256 = (
    "42e5487bf77436200d5903437f299d9071ab7f817e7e83ac3f4e14d197fed616"
)
FAILED_PROGRESS = FAILED_INPUT_BINDING.parent / "training_progress.jsonl"
FAILED_PROGRESS_SHA256 = (
    "3d3dc7af06a9f228ec044163d9bb6b43daf1fa2dd2ab66575b7b9369009d6560"
)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Filtered positive-only protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Filtered positive-only protocol payload differs")
    if shared.sha256_file(FAILED_INPUT_BINDING) != FAILED_INPUT_BINDING_SHA256:
        raise ValueError("Positive-only failed input binding differs")
    if shared.sha256_file(FAILED_PROGRESS) != FAILED_PROGRESS_SHA256:
        raise ValueError("Positive-only failed progress binding differs")
    endpoint = protocol.get("heldout_causal_endpoint", {})
    training = protocol.get("training", {})
    if (
        endpoint.get("source_ordinals") != list(positive.HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != positive.HELDOUT_PAYLOAD_SHA256
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or training.get("gradient_rescale") != "8 / accepted_rows"
        or training.get("optimizer_updates") != shared.UPDATES
    ):
        raise ValueError("Filtered positive-only training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Filtered positive-only training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with positive.training_bindings():
        positive_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "FILTER_NONFINITE_ROWS": True,
            "MIN_ACCEPTED_ROWS_PER_UPDATE": MIN_ACCEPTED_ROWS_PER_UPDATE,
            "MAX_TOTAL_REJECTED_ROWS": MAX_TOTAL_REJECTED_ROWS,
        }
        shared_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SEED": SEED,
            "HELDOUT_ORDINALS": positive.HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": positive.HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "positive_only_filtered_heldout_passed_generation_authorized",
            "FAIL_STATUS": "positive_only_filtered_heldout_failed_generation_blocked",
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": positive.train_positive_only,
            "validate_protocol": validate_protocol,
        }
        previous_positive = {
            name: getattr(positive, name) for name in positive_bindings
        }
        previous_shared = {name: getattr(shared, name) for name in shared_bindings}
        try:
            for name, value in positive_bindings.items():
                setattr(positive, name, value)
            for name, value in shared_bindings.items():
                setattr(shared, name, value)
            yield
        finally:
            for name, value in previous_shared.items():
                setattr(shared, name, value)
            for name, value in previous_positive.items():
                setattr(positive, name, value)


def validate_calibration_result() -> Mapping[str, Any]:
    return shared.validate_calibration_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
