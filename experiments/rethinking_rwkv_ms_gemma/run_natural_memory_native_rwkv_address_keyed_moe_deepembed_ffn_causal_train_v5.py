#!/usr/bin/env python3
"""Retry DeepEmbed training with serialized active-control autograd graphs."""

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
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v4
    as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v5.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "749d5a6f6edf2ac8ee259d1494916529a2046621a52736ad02960760a8ed4ddd"
)
V4_PROTOCOL_PAYLOAD_SHA256 = (
    "a70c9eb52c7266bda416b896fa8aa1ca0c3573e0d4bf917fe1fd963dcc81cbbf"
)
V4_FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v4_r2/failure.json"
)
V4_FAILURE_FILE_SHA256 = (
    "306d3c448cdd9d075cb388cfe841c63d089ac337194ce2cbd8ad94ef68e36034"
)
V4_FAILURE_RECEIPT = (
    "64a21070e6a3713ed3e4cfd863e887fbe5537203f8d34a3e6f43651544b994ad"
)
CAUSAL_TRAINER_AT_FAILURE_SHA256 = (
    "28c6d7ae38de52e4e7f05e21e67b22d7ec068b2632f43a579982f2ef432f0179"
)
CAUSAL_TRAINER_WITH_SERIALIZED_GRAPHS_SHA256 = (
    "3036e7c75c1dedd31ab7f3d8aa79126c849a5c64e5e37395bbe3e2c43822fbc7"
)
SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train.v5"
)
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_step.v5"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_input.v5"
)
SEED = 108
PASS_STATUS = (
    "address_keyed_moe_deepembed_ffn_serialized_graphs_heldout_passed_generation_authorized"
)
FAIL_STATUS = (
    "address_keyed_moe_deepembed_ffn_serialized_graphs_heldout_failed_generation_blocked"
)


def validate_v4_failure() -> Mapping[str, Any]:
    failure = base.base.base.base.validate_signed_result(
        V4_FAILURE,
        file_sha256=V4_FAILURE_FILE_SHA256,
        receipt_sha256=V4_FAILURE_RECEIPT,
    )
    evidence = failure.get("completed_training_evidence", {})
    if (
        failure.get("status")
        != "address_keyed_moe_deepembed_ffn_training_cuda_oom_step6_endpoint_unopened"
        or failure.get("protocol_payload_sha256") != V4_PROTOCOL_PAYLOAD_SHA256
        or failure.get("completed_optimizer_updates") != 5
        or failure.get("failed_optimizer_update") != 6
        or evidence.get("accepted_rows") != 40
        or evidence.get("rejected_rows") != 0
        or evidence.get("optimizer_state_cpu_offload_tensors_per_rank") != 1170
        or evidence.get("optimizer_state_restored_before_completed_updates") is not True
        or failure.get("failure", {}).get("exception_type")
        != "torch.OutOfMemoryError"
        or failure.get("failure", {}).get("source_ordinal") != 899
        or failure.get("heldout_causal_endpoint_opened") is not False
        or failure.get("native_generation_opened") is not False
        or failure.get("protected_splits_opened") != []
    ):
        raise ValueError(
            "The signed v4 execution failure does not authorize graph serialization"
        )
    return failure


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Serialized-graph causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = base.base.base.base.SHARED_TRAINER.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    candidate = base.base.base.base.SELECTED_CANDIDATE
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != base.base.base.base.screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != base.base.base.base.SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt")
        != base.base.base.base.SCREEN_RESULT_RECEIPT
        or authorization.get("candidate_result_file_sha256")
        != base.base.base.base.CANDIDATE_RESULT_FILE_SHA256
        or authorization.get("candidate_result_receipt")
        != base.base.base.base.CANDIDATE_RESULT_RECEIPT
        or authorization.get("v4_protocol_payload_sha256")
        != V4_PROTOCOL_PAYLOAD_SHA256
        or authorization.get("v4_failure_file_sha256") != V4_FAILURE_FILE_SHA256
        or authorization.get("v4_failure_receipt") != V4_FAILURE_RECEIPT
        or authorization.get("causal_trainer_at_failure_sha256")
        != CAUSAL_TRAINER_AT_FAILURE_SHA256
        or authorization.get("causal_trainer_with_serialized_control_graphs_sha256")
        != CAUSAL_TRAINER_WITH_SERIALIZED_GRAPHS_SHA256
        or authorization.get("selected_candidate") != candidate
        or frozen.get("seed") != SEED
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain")
        != float(candidate["write_address_gain"])
        or architecture.get("rms_epsilon_inside_sqrt") != 1e-12
        or architecture.get("expected_trainable_parameter_tensors")
        != base.base.base.base.EXPECTED_TRAINABLE_TENSORS
        or architecture.get("outer_ffn_layers")
        != list(base.base.base.base.OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != base.base.base.base.UPDATES
        or training.get("contrast_weight_per_active_control")
        != base.base.base.base.CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != base.base.base.base.MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != base.base.base.base.MAX_TOTAL_REJECTED_ROWS
        or training.get("optimizer_state_cpu_offload_enabled") is not True
        or training.get("control_branch_graph_serialization_enabled") is not True
        or training.get("maximum_simultaneous_autograd_graphs_per_rank") != 1
        or endpoint.get("candidate_rows_after_exclusions")
        != base.base.base.base.ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals")
        != list(base.base.base.base.HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != base.base.base.base.HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Serialized-graph address-keyed causal protocol differs")
    base.base.base.base.validate_screen_result()
    base.base.validate_v2_failure()
    base.base.validate_gradient_diagnostics()
    base.validate_v3_failure()
    validate_v4_failure()
    trainer = (
        PROJECT_ROOT
        / "experiments/rethinking_rwkv_ms_gemma/run_natural_memory_native_rwkv_addressed_value_causal_train.py"
    )
    if (
        base.base.base.base.SHARED_TRAINER.sha256_file(trainer)
        != CAUSAL_TRAINER_WITH_SERIALIZED_GRAPHS_SHA256
    ):
        raise ValueError("Serialized-graph causal trainer source differs")
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
    previous_serialization = causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = True
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = True
        with base.bindings():
            previous_runner_binding = (
                base.base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH
            )
            base.base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            try:
                yield
            finally:
                base.base.base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = (
                    previous_runner_binding
                )
    finally:
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = previous_serialization
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = previous_offload
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.base.base.SHARED_TRAINER.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.base.base.SHARED_TRAINER.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
