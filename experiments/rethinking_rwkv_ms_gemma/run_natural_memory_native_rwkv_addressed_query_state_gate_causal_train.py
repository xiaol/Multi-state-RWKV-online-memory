#!/usr/bin/env python3
"""Train the supervised query/state-gated addressed RWKV hybrid."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_route_agreement_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_query_state_gate_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as candidate_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_query_state_gate_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "38a6c2125cdb0cbe1447d4343be0d9efa18adc6aa211b16e725dd2da14dbf70a"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_query_state_gate_screen_v1/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "3b9f4c4bcae8bb05acfd1e752b1ad60b4a9db633bcacec50c96b97e4d18e3c7b"
)
SCREEN_RESULT_RECEIPT = (
    "ab0691924065eeef754df3b07080df5df120e908ec14eafda5e1eb188554946a"
)
SCHEMA = "rwkv_ms_natural_memory_native_addressed_query_state_gate_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_query_state_gate_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_query_state_gate_causal_train_input.v1"
)
SEED = 85
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 475
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (
    304, 1083, 960, 1270, 159, 497, 1395, 284,
    323, 704, 1214, 1204, 1313, 1396, 533, 911,
    1061, 575, 833, 1314, 834, 326, 1213, 1066,
    920, 939, 888, 708, 949, 637, 177, 7,
)
HELDOUT_PAYLOAD_SHA256 = (
    "cd9010857a42e3d38da0278135a65e5be1a1976df7bbfb0e203f8b3362d4f72f"
)
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "addressed_query_state_gate_heldout_passed_generation_authorized"
FAIL_STATUS = "addressed_query_state_gate_heldout_failed_generation_blocked"


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=top2_screen.build_config(candidate),
    )
    candidate_helper.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "addressed_query_state_gate"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_addressed_query_state_gate_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Query-state-gate attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if base.affine_train.shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Query-state-gate screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Query-state-gate screen receipt is missing")
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
        raise ValueError("Query-state-gate screen did not authorize training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Query-state-gate causal protocol receipt is missing")
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
        or training.get("maximum_total_rejected_rows") != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions") != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Query-state-gate causal protocol differs")
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
            yield
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
