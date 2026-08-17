#!/usr/bin/env python3
"""Retry the exact-RMS DeepEmbed hybrid with audited Adam-state offload."""

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
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v3
    as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v4.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "a70c9eb52c7266bda416b896fa8aa1ca0c3573e0d4bf917fe1fd963dcc81cbbf"
)
V3_PROTOCOL_PAYLOAD_SHA256 = (
    "dc697cc94926c094f1c4c2c537b9288b30eabede3ff5d8f019aa09a33792aa77"
)
V3_FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v3/failure.json"
)
V3_FAILURE_FILE_SHA256 = (
    "7d2d938f4c6b2bc21e4d14d507df17329d303e781db67460cdcae4d5bb3c2d14"
)
V3_FAILURE_RECEIPT = (
    "a1bda5854dc5eb5a95326a30cdad4404f763943e328dd4cd5af527dc24db7979"
)
CAUSAL_TRAINER_AT_FAILURE_SHA256 = (
    "ceac32bb3d810637995da35d630f1190e5ac72ff5587750087dde483bce5686b"
)
CAUSAL_TRAINER_WITH_OFFLOAD_SHA256 = (
    "28c6d7ae38de52e4e7f05e21e67b22d7ec068b2632f43a579982f2ef432f0179"
)
SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train.v4"
)
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_step.v4"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_input.v4"
)
SEED = 108
PASS_STATUS = (
    "address_keyed_moe_deepembed_ffn_exact_rms_offload_heldout_passed_generation_authorized"
)
FAIL_STATUS = (
    "address_keyed_moe_deepembed_ffn_exact_rms_offload_heldout_failed_generation_blocked"
)


def validate_v3_failure() -> Mapping[str, Any]:
    failure = base.base.base.validate_signed_result(
        V3_FAILURE,
        file_sha256=V3_FAILURE_FILE_SHA256,
        receipt_sha256=V3_FAILURE_RECEIPT,
    )
    if (
        failure.get("status")
        != "address_keyed_moe_deepembed_ffn_training_cuda_oom_step6_endpoint_unopened"
        or failure.get("protocol_payload_sha256") != V3_PROTOCOL_PAYLOAD_SHA256
        or failure.get("completed_optimizer_updates") != 5
        or failure.get("failed_optimizer_update") != 6
        or failure.get("completed_training_evidence", {}).get("accepted_rows") != 40
        or failure.get("failure", {}).get("exception_type")
        != "torch.OutOfMemoryError"
        or failure.get("heldout_causal_endpoint_opened") is not False
        or failure.get("native_generation_opened") is not False
        or failure.get("protected_splits_opened") != []
    ):
        raise ValueError("The signed v3 execution failure does not authorize offload retry")
    return failure


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Optimizer-offload causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = base.base.base.SHARED_TRAINER.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != base.base.base.screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != base.base.base.SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt")
        != base.base.base.SCREEN_RESULT_RECEIPT
        or authorization.get("candidate_result_file_sha256")
        != base.base.base.CANDIDATE_RESULT_FILE_SHA256
        or authorization.get("candidate_result_receipt")
        != base.base.base.CANDIDATE_RESULT_RECEIPT
        or authorization.get("v3_protocol_payload_sha256") != V3_PROTOCOL_PAYLOAD_SHA256
        or authorization.get("v3_failure_file_sha256") != V3_FAILURE_FILE_SHA256
        or authorization.get("v3_failure_receipt") != V3_FAILURE_RECEIPT
        or authorization.get("causal_trainer_at_failure_sha256")
        != CAUSAL_TRAINER_AT_FAILURE_SHA256
        or authorization.get("causal_trainer_with_offload_sha256")
        != CAUSAL_TRAINER_WITH_OFFLOAD_SHA256
        or authorization.get("selected_candidate") != base.base.base.SELECTED_CANDIDATE
        or frozen.get("seed") != SEED
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain")
        != float(base.base.base.SELECTED_CANDIDATE["write_address_gain"])
        or architecture.get("rms_epsilon_inside_sqrt") != 1e-12
        or architecture.get("expected_trainable_parameter_tensors")
        != base.base.base.EXPECTED_TRAINABLE_TENSORS
        or architecture.get("outer_ffn_layers")
        != list(base.base.base.OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != base.base.base.UPDATES
        or training.get("contrast_weight_per_active_control")
        != base.base.base.CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != base.base.base.MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != base.base.base.MAX_TOTAL_REJECTED_ROWS
        or training.get("optimizer_state_cpu_offload_enabled") is not True
        or endpoint.get("candidate_rows_after_exclusions")
        != base.base.base.ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals")
        != list(base.base.base.HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != base.base.base.HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Optimizer-offload address-keyed causal protocol differs")
    base.base.base.validate_screen_result()
    base.validate_v2_failure()
    base.validate_gradient_diagnostics()
    validate_v3_failure()
    if base.base.base.SHARED_TRAINER.sha256_file(PROJECT_ROOT / "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_value_causal_train.py") != CAUSAL_TRAINER_WITH_OFFLOAD_SHA256:
        raise ValueError("Optimizer-offload causal trainer source differs")
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
    previous_offload = causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = True
        with base.bindings():
            previous_runner_binding = base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH
            base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            try:
                yield
            finally:
                base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = previous_runner_binding
    finally:
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = previous_offload
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.base.SHARED_TRAINER.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.base.SHARED_TRAINER.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
