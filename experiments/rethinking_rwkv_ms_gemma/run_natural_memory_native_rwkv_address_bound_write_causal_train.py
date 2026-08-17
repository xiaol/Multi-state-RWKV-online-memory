#!/usr/bin/env python3
"""Train recurrent RWKV writes bound to projected memory addresses."""

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
    run_natural_memory_native_rwkv_addressed_route_agreement_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_bound_write_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as candidate_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)

PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_bound_write_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "053c85e5e3f7c44db6c0f242962979e1d2bfeb4af39d027b2f8c36f72e4907ed"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_bound_write_screen_v4/"
    "result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "fc2e53e07d0b797322276e0a12b94ac684b6d6256fa0fed083fc17fda7480d14"
)
SCREEN_RESULT_RECEIPT = (
    "f1c91a769f100bcae94f5006f06ae662a8877603c48ad7ab5b047819d4e706ee"
)
SCHEMA = "rwkv_ms_natural_memory_native_address_bound_write_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_bound_write_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_bound_write_causal_train_input.v1"
)
SEED = 94
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 91
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (
    488, 742, 106, 102, 231, 1046, 406, 1049,
    1223, 1267, 461, 1310, 12, 477, 976, 565,
    1026, 1300, 1125, 1085, 598, 1352, 98, 854,
    1323, 1120, 621, 191, 1334, 1419, 440, 848,
)
HELDOUT_PAYLOAD_SHA256 = (
    "96c62e69cb1b9128cc57472bc4885d3328694ea89a7d0ea3ff790fdfb9083459"
)
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "address_bound_write_heldout_passed_generation_authorized"
FAIL_STATUS = "address_bound_write_heldout_failed_generation_blocked"


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
    candidate_helper.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == "address_bound_write"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.rwkv_ms_write_mode == "recurrent"
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_address_bound_write_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Address-bound-write causal attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if base.affine_train.shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Address-bound-write screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound-write screen receipt is missing")
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
        raise ValueError("Address-bound-write screen did not authorize training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound-write causal protocol receipt is missing")
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
        or authorization.get("screen_result_file_sha256")
        != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions")
        != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-bound-write causal protocol differs")
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
            previous_trainable_configurer = (
                base.affine_train.shared.TRAINABLE_CONFIGURER
            )
            base.affine_train.shared.screen = screen
            base.affine_train.shared.TRAINABLE_CONFIGURER = (
                stable.configure_stable_readout_parameters
            )
            try:
                yield
            finally:
                base.affine_train.shared.screen = previous_shared_screen
                base.affine_train.shared.TRAINABLE_CONFIGURER = (
                    previous_trainable_configurer
                )
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
