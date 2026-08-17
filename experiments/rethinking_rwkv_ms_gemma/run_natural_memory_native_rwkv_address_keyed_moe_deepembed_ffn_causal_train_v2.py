#!/usr/bin/env python3
"""Retry address-keyed DeepEmbed training with stabilized RMS gradients."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train as base,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "13e8b88cde6800b2f7963c02f8eed7b2d2648be1d647d7d2f6f83ab5d6c1bb0c"
)
V1_PROTOCOL_PAYLOAD_SHA256 = (
    "f7cdda7a31f48128b56289166a85fed55824d59c89889b0a18e14371b94f0400"
)
FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v1/failure.json"
)
FAILURE_FILE_SHA256 = (
    "4a8aaf60a8151db0b9a678bcc2916c0f7ebef7fe829f9adfbb4a93488f736612"
)
FAILURE_RECEIPT = (
    "9230c945c9221bd66e081474b2a6d68f2e3a5c533f26e6a6615dab69f189c88d"
)
SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train.v2"
)
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_step.v2"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_input.v2"
)
SEED = 107
PASS_STATUS = (
    "address_keyed_moe_deepembed_ffn_stabilized_heldout_passed_generation_authorized"
)
FAIL_STATUS = (
    "address_keyed_moe_deepembed_ffn_stabilized_heldout_failed_generation_blocked"
)


def validate_failure() -> Mapping[str, Any]:
    failure = base.validate_signed_result(
        FAILURE,
        file_sha256=FAILURE_FILE_SHA256,
        receipt_sha256=FAILURE_RECEIPT,
    )
    if (
        failure.get("status")
        != "address_keyed_moe_deepembed_ffn_training_nonfinite_step1_endpoint_unopened"
        or failure.get("protocol_payload_sha256") != V1_PROTOCOL_PAYLOAD_SHA256
        or failure.get("completed_optimizer_updates") != 0
        or failure.get("failed_optimizer_update") != 1
        or failure.get("row_filter", {}).get("accepted_rows") != 0
        or failure.get("heldout_causal_endpoint_opened") is not False
        or failure.get("native_generation_opened") is not False
        or failure.get("protected_splits_opened") != []
    ):
        raise ValueError("The v1 gradient failure does not authorize a stabilized retry")
    return failure


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Stabilized causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = base.SHARED_TRAINER.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != base.screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != base.SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt")
        != base.SCREEN_RESULT_RECEIPT
        or authorization.get("candidate_result_file_sha256")
        != base.CANDIDATE_RESULT_FILE_SHA256
        or authorization.get("candidate_result_receipt")
        != base.CANDIDATE_RESULT_RECEIPT
        or authorization.get("v1_failure_file_sha256") != FAILURE_FILE_SHA256
        or authorization.get("v1_failure_receipt") != FAILURE_RECEIPT
        or authorization.get("selected_candidate") != base.SELECTED_CANDIDATE
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain")
        != float(base.SELECTED_CANDIDATE["write_address_gain"])
        or architecture.get("rms_epsilon_inside_sqrt") != 1e-12
        or architecture.get("expected_trainable_parameter_tensors")
        != base.EXPECTED_TRAINABLE_TENSORS
        or architecture.get("learned_write_parameter_tensors") != 0
        or architecture.get("outer_ffn_layers") != list(base.OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != base.UPDATES
        or training.get("contrast_weight_per_active_control")
        != base.CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != base.MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != base.MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions")
        != base.ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(base.HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != base.HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Stabilized address-keyed causal protocol differs")
    base.validate_screen_result()
    validate_failure()
    return protocol


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "SEED": SEED,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "validate_protocol": validate_protocol,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        with base.bindings():
            previous_runner_binding = base.SHARED_TRAINER.RUNNER_BINDING_PATH
            base.SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            try:
                yield
            finally:
                base.SHARED_TRAINER.RUNNER_BINDING_PATH = previous_runner_binding
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.SHARED_TRAINER.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.SHARED_TRAINER.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
