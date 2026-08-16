#!/usr/bin/env python3
"""Screen an alignment-verified vector-FiLM RWKV hybrid."""

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
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_aligned_vector_gate_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_aligned_vector_gate_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "3cbd932f8a906cd97b6f42c92188f5e32b1179a7d4f88a8e30384b88da2e9b43"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_alignment_residual_causal_train_v1/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "82bd5a9e115ba825caec8dfc1149f6e04393f8370e4185d22306ea12dca1b6c4"
)
PRIOR_RESULT_RECEIPT = (
    "5f9a2b9ce10e5cd55537cb1bebdfe249494627dc3709e8006402c21e44b867f5"
)
PRIOR_PROTOCOL_PAYLOAD_SHA256 = (
    "023a970f0c3d65c48e6bf3563bd6c54a7dc9e86f618ca8d891208b1fbf74f9b4"
)
SEED = 77
SELECTED_CANDIDATE = {
    "candidate_id": "aligned_vector_gate_t16_k2_gate025_g0125",
    "hybrid_mode": "aligned_vector_gate",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
PASS_STATUS = "aligned_vector_gate_screen_passed_training_authorized"
FAIL_STATUS = "aligned_vector_gate_screen_failed_training_blocked"


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Aligned-vector-gate protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("hybrid_mode") != "aligned_vector_gate"
        or architecture.get("hybrid_gain") != 0.125
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
        or authorization.get("prior_protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Aligned-vector-gate screen protocol differs")
    if shared.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Alignment-residual result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    receipt_prior = unsigned_prior.pop("receipt", {})
    if (
        shared.canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or receipt_prior.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or prior.get("status")
        != "alignment_residual_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or prior.get("passed") is not False
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Alignment-residual failure does not authorize redesign")
    return protocol


@contextmanager
def screen_bindings() -> Iterator[None]:
    bindings = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "PRIOR_RESULT": PRIOR_RESULT,
        "PRIOR_RESULT_FILE_SHA256": PRIOR_RESULT_FILE_SHA256,
        "PRIOR_RESULT_RECEIPT": PRIOR_RESULT_RECEIPT,
        "SEED": SEED,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MODEL_AUDIT_KEY": "all_wrappers_aligned_vector_gate_content_gated",
        "PRIOR_RESULT_CODE_BINDING_KEY": "alignment_result_file_sha256",
        "RUNNER_BINDING_PATH": Path(__file__),
        "validate_protocol": validate_protocol,
    }
    previous = {name: getattr(shared, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with screen_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with screen_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
