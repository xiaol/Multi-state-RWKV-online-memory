#!/usr/bin/env python3
"""Train the stabilized address-keyed attention plus sparse DeepEmbed hybrid."""

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
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v2
    as base,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v3.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "dc697cc94926c094f1c4c2c537b9288b30eabede3ff5d8f019aa09a33792aa77"
)
V2_PROTOCOL_PAYLOAD_SHA256 = (
    "13e8b88cde6800b2f7963c02f8eed7b2d2648be1d647d7d2f6f83ab5d6c1bb0c"
)
V2_PROTOCOL_FILE_SHA256 = (
    "ea121241f579110b2298fa2449219739e7467142c59d82c3acaf275f127361d9"
)
V2_FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v2/failure.json"
)
V2_FAILURE_FILE_SHA256 = (
    "22dfd178417732896e077b6842cca5c084a52b91472162db56f9ebb338dfea2c"
)
V2_FAILURE_RECEIPT = (
    "38b38d20778eed349058ee8256ea43c0f13781497e05358d9a9edf8b1c8bbe1c"
)
PRE_FIX_DIAGNOSTIC_DIR = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_gradient_diagnostic_v1"
)
PRE_FIX_DIAGNOSTIC_RANK_SHA256 = (
    "e29cc71568de10d45f139abfed99386e662b32c9a43aedb9eae58a6c6fe8cfe5"
)
POST_FIX_DIAGNOSTIC_DIR = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_gradient_diagnostic_v6"
)
POST_FIX_DIAGNOSTIC_INPUT_SHA256 = (
    "82f23678d294881d5adf39936d621dbdf20dad50a58f50338dbdd3f55b556407"
)
POST_FIX_DIAGNOSTIC_RANK_SHA256 = (
    "7652e4956e3ec27233644827de0ec2928cf4e6fa20fb968bb660f10de6970d70"
)
DELTA_IMPL_SHA256 = (
    "88c495a417fcff62b295b70971f1c02991aac3962f35dae3afa5affb0a808788"
)
SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train.v3"
)
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_step.v3"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_input.v3"
)
SEED = 108
PASS_STATUS = (
    "address_keyed_moe_deepembed_ffn_exact_rms_heldout_passed_generation_authorized"
)
FAIL_STATUS = (
    "address_keyed_moe_deepembed_ffn_exact_rms_heldout_failed_generation_blocked"
)


def validate_v2_failure() -> Mapping[str, Any]:
    failure = base.base.validate_signed_result(
        V2_FAILURE,
        file_sha256=V2_FAILURE_FILE_SHA256,
        receipt_sha256=V2_FAILURE_RECEIPT,
    )
    diagnostic = failure.get("gradient_diagnostic", {})
    post_fix = failure.get("post_fix_diagnostic", {})
    if (
        failure.get("status")
        != "address_keyed_moe_deepembed_ffn_training_nonfinite_step1_endpoint_unopened"
        or failure.get("protocol_payload_sha256") != V2_PROTOCOL_PAYLOAD_SHA256
        or failure.get("completed_optimizer_updates") != 0
        or failure.get("row_filter", {}).get("accepted_rows") != 0
        or failure.get("row_filter", {}).get("rejected_rows") != 8
        or diagnostic.get("first_correct_backward_nonfinite_tensors") != 379
        or diagnostic.get("active_trainable_tensors") != 390
        or post_fix.get("finite_rows") != 8
        or post_fix.get("active_finite_trainable_tensors_per_rank") != 390
        or post_fix.get("nonfinite_trainable_tensors_after_each_branch") != 0
        or post_fix.get("optimizer_updates_completed") != 0
        or failure.get("heldout_causal_endpoint_opened") is not False
        or failure.get("native_generation_opened") is not False
        or failure.get("protected_splits_opened") != []
    ):
        raise ValueError("The signed v2 failure does not authorize the exact-RMS retry")
    return failure


def validate_gradient_diagnostics() -> None:
    sha256_file = base.base.SHARED_TRAINER.sha256_file
    for rank in range(4):
        pre_fix_path = PRE_FIX_DIAGNOSTIC_DIR / f"gradient_diagnostic_rank{rank}.jsonl"
        post_fix_path = POST_FIX_DIAGNOSTIC_DIR / f"gradient_diagnostic_rank{rank}.jsonl"
        if sha256_file(pre_fix_path) != PRE_FIX_DIAGNOSTIC_RANK_SHA256:
            raise ValueError(f"Pre-fix gradient diagnostic differs on rank {rank}")
        if sha256_file(post_fix_path) != POST_FIX_DIAGNOSTIC_RANK_SHA256:
            raise ValueError(f"Post-fix gradient diagnostic differs on rank {rank}")
        events = [
            json.loads(line)
            for line in post_fix_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        branch_events = [event for event in events if event.get("event") == "after_backward"]
        row_events = [event for event in events if event.get("event") == "row_complete"]
        if (
            len(branch_events) != 6
            or len(row_events) != 2
            or any(event.get("logits_finite") is not True for event in branch_events)
            or any(event.get("active_gradient_tensors") != 390 for event in events)
            or any(event.get("nonfinite_gradient_tensors") != 0 for event in events)
        ):
            raise ValueError(f"Post-fix gradient diagnostic failed on rank {rank}")
    if (
        sha256_file(POST_FIX_DIAGNOSTIC_DIR / "input_binding.json")
        != POST_FIX_DIAGNOSTIC_INPUT_SHA256
    ):
        raise ValueError("Post-fix diagnostic input binding differs")
    if sha256_file(PROJECT_ROOT / "deltamem/core/delta_impl.py") != DELTA_IMPL_SHA256:
        raise ValueError("The exact-RMS implementation differs from the diagnosed source")


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Exact-RMS causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = base.base.SHARED_TRAINER.canonical_sha256(unsigned)
    authorization = protocol.get("authorization_basis", {})
    frozen = protocol.get("frozen_inputs", {})
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != base.base.screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != base.base.SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt")
        != base.base.SCREEN_RESULT_RECEIPT
        or authorization.get("candidate_result_file_sha256")
        != base.base.CANDIDATE_RESULT_FILE_SHA256
        or authorization.get("candidate_result_receipt")
        != base.base.CANDIDATE_RESULT_RECEIPT
        or authorization.get("v2_protocol_file_sha256")
        != V2_PROTOCOL_FILE_SHA256
        or authorization.get("v2_protocol_payload_sha256")
        != V2_PROTOCOL_PAYLOAD_SHA256
        or authorization.get("v2_failure_file_sha256")
        != V2_FAILURE_FILE_SHA256
        or authorization.get("v2_failure_receipt") != V2_FAILURE_RECEIPT
        or authorization.get("pre_fix_gradient_diagnostic_rank_sha256")
        != PRE_FIX_DIAGNOSTIC_RANK_SHA256
        or authorization.get("post_fix_gradient_diagnostic_rank_sha256")
        != POST_FIX_DIAGNOSTIC_RANK_SHA256
        or authorization.get("post_fix_gradient_diagnostic_input_sha256")
        != POST_FIX_DIAGNOSTIC_INPUT_SHA256
        or authorization.get("post_fix_delta_impl_sha256") != DELTA_IMPL_SHA256
        or authorization.get("selected_candidate") != base.base.SELECTED_CANDIDATE
        or frozen.get("seed") != SEED
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain")
        != float(base.base.SELECTED_CANDIDATE["write_address_gain"])
        or architecture.get("rms_epsilon_inside_sqrt") != 1e-12
        or architecture.get("expected_trainable_parameter_tensors")
        != base.base.EXPECTED_TRAINABLE_TENSORS
        or architecture.get("learned_write_parameter_tensors") != 0
        or architecture.get("outer_ffn_layers") != list(base.base.OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != base.base.UPDATES
        or training.get("contrast_weight_per_active_control")
        != base.base.CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != base.base.MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != base.base.MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions")
        != base.base.ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(base.base.HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != base.base.HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Exact-RMS address-keyed causal protocol differs")
    base.base.validate_screen_result()
    validate_v2_failure()
    validate_gradient_diagnostics()
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
            previous_runner_binding = base.base.SHARED_TRAINER.RUNNER_BINDING_PATH
            base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            try:
                yield
            finally:
                base.base.SHARED_TRAINER.RUNNER_BINDING_PATH = previous_runner_binding
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.SHARED_TRAINER.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.SHARED_TRAINER.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
