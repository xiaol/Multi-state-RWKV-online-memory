#!/usr/bin/env python3
"""Train address-bound recurrent writes with a trainable RWKV MoE controller."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_bound_moe_controller_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_controller_causal_train as moe,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_route_agreement_causal_train as base,
)

PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_bound_moe_controller_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "b1054b0a9827996a28fc7cb6c052eac7279503230acbb0082d8adc9743fff10d"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_bound_moe_controller_screen_v1/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "5817b709fa7eaaa56ea2686a71b7a807185b01522ad4937d6ad9f501524fcd30"
)
SCREEN_RESULT_RECEIPT = (
    "e9bdd521ce213713caea92154d08ed903bc403b2c6337e329fe53435aa81bf4a"
)
SCHEMA = "rwkv_ms_natural_memory_native_address_bound_moe_controller_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_bound_moe_controller_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_bound_moe_controller_causal_train_input.v1"
)
SEED = 96
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 19
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (545, 765, 582, 718, 863, 1432, 1349, 1436, 29, 586, 456, 596)
HELDOUT_PAYLOAD_SHA256 = (
    "97b043c06f2981446705b7c8aba3574d11d52a25ff99fa3ce242343f475cd417"
)
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "address_bound_moe_controller_heldout_passed_generation_authorized"
FAIL_STATUS = "address_bound_moe_controller_heldout_failed_generation_blocked"
MOE_SUFFIXES = moe.MOE_SUFFIXES


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = screen.load_model(
        base_model,
        device=device,
    )
    screen.bound.candidate_helper.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "address_bound_moe_controller"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        and module.rwkv_ms_write_mode == "recurrent"
        and hasattr(module, "rwkv_moe_bias")
        for _, module in modules
    )
    audit = {**dict(inherited_audit), "all_wrappers_address_bound_moe_controller": configured}
    if not configured:
        raise RuntimeError(f"Address-bound MoE causal attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if base.affine_train.shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Address-bound MoE screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound MoE screen receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        base.affine_train.shared.canonical_sha256(unsigned) != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != SCREEN_RESULT_RECEIPT
        or result.get("status") != screen.PASS_STATUS
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("native_generation_authorized") is not False
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Address-bound MoE screen did not authorize training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound MoE causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    authorization = protocol.get("authorization_basis", {})
    digest = base.affine_train.shared.canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256") != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update") != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows") != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions") != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-bound MoE causal protocol differs")
    validate_screen_result()
    return protocol


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCREEN_RESULT": SCREEN_RESULT,
        "SCREEN_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
        "SCREEN_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "CONTRAST_WEIGHT": CONTRAST_WEIGHT,
        "MIN_ACCEPTED_ROWS_PER_UPDATE": MIN_ACCEPTED_ROWS_PER_UPDATE,
        "MAX_TOTAL_REJECTED_ROWS": MAX_TOTAL_REJECTED_ROWS,
        "ENDPOINT_CANDIDATE_ROWS": ENDPOINT_CANDIDATE_ROWS,
        "TRAINING_PREFIX_SHA256": TRAINING_PREFIX_SHA256,
        "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "screen": screen,
        "load_model": load_model,
        "validate_screen_result": validate_screen_result,
        "validate_protocol": validate_protocol,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        with base.training_bindings():
            previous_shared_screen = base.affine_train.shared.screen
            previous_trainable_configurer = base.affine_train.shared.TRAINABLE_CONFIGURER
            base.affine_train.shared.screen = screen
            base.affine_train.shared.TRAINABLE_CONFIGURER = moe.configure_moe_parameters
            try:
                yield
            finally:
                base.affine_train.shared.screen = previous_shared_screen
                base.affine_train.shared.TRAINABLE_CONFIGURER = previous_trainable_configurer
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.affine_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.affine_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
