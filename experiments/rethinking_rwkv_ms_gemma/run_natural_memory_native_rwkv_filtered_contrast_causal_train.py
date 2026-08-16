#!/usr/bin/env python3
"""Train causal-contrast RWKV readouts with isolated finite row gradients."""

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

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_positive_only_causal_train as positive,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_filtered_contrast_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_filtered_contrast_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_filtered_contrast_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_filtered_contrast_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "676f1485244d541310feef5ce1a3de07e6dc06b00d340bb9e85e3be30aa6b189"
)
SEED = positive.SEED
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
HELDOUT_ORDINALS = (
    439, 635, 1312, 47, 244, 823, 1054, 576,
    1104, 382, 1299, 819, 607, 296, 498, 1285,
    578, 548, 558, 1315, 375, 35, 739, 728,
    1415, 425, 205, 149, 76, 100, 801, 1226,
)
HELDOUT_PAYLOAD_SHA256 = (
    "3002b98cbbc27018ef80a6dcd14b9d7992676715b57f3158a9d1342ff5360b7b"
)
FILTERED_POSITIVE_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_positive_only_filtered_causal_train_v1/result.json"
)
FILTERED_POSITIVE_RESULT_FILE_SHA256 = (
    "cd5cde7bae45a15b558611a324f9dd83c50c656014c8d3943f487654e36ded33"
)
FILTERED_POSITIVE_RESULT_RECEIPT = (
    "c713578b478e72763efe10cc68dcde45b5d98b3902d11d3456e45e6c14c4833d"
)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Filtered-contrast protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Filtered-contrast protocol payload differs")
    if (
        shared.sha256_file(FILTERED_POSITIVE_RESULT)
        != FILTERED_POSITIVE_RESULT_FILE_SHA256
    ):
        raise ValueError("Filtered positive-only result binding differs")
    prior = json.loads(FILTERED_POSITIVE_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    if (
        shared.canonical_sha256(unsigned_prior) != FILTERED_POSITIVE_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != FILTERED_POSITIVE_RESULT_RECEIPT
        or prior.get("status")
        != "positive_only_filtered_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or prior.get("passed") is not False
    ):
        raise ValueError("Filtered positive-only failure does not authorize training")
    endpoint = protocol.get("heldout_causal_endpoint", {})
    training = protocol.get("training", {})
    if (
        endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or training.get("contrast_weight_per_active_control")
        != causal_train.CONTRAST_WEIGHT
        or training.get("contrast_margin") != causal_train.MARGIN
        or training.get("optimizer_updates") != shared.UPDATES
    ):
        raise ValueError("Filtered-contrast training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Filtered-contrast training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with positive.training_bindings():
        causal_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
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
            "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
            "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "filtered_contrast_heldout_passed_generation_authorized",
            "FAIL_STATUS": "filtered_contrast_heldout_failed_generation_blocked",
            "REQUIRE_RECURRENT_SUBSET_CHANGED": True,
            "TRAINING_FUNCTION": causal_train.train,
            "validate_protocol": validate_protocol,
        }
        previous_causal = {name: getattr(causal_train, name) for name in causal_bindings}
        previous_shared = {name: getattr(shared, name) for name in shared_bindings}
        try:
            for name, value in causal_bindings.items():
                setattr(causal_train, name, value)
            for name, value in shared_bindings.items():
                setattr(shared, name, value)
            yield
        finally:
            for name, value in previous_shared.items():
                setattr(shared, name, value)
            for name, value in previous_causal.items():
                setattr(causal_train, name, value)


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
