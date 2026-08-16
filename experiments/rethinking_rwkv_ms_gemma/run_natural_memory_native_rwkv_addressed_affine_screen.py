#!/usr/bin/env python3
"""Screen a projected-addressed RWKV affine fusion."""

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


SCHEMA = "rwkv_ms_natural_memory_native_addressed_affine_screen.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_addressed_affine_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "483fec0fa7b0547237c13aa931397b0c944dd375d6efb4404206f820b0ac7394"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_addressed_vector_gate_causal_train_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "e9a115cd7864e0c31738478c0393aed21c6db40e3ab1adccc50ba76cb8a898e4"
)
PRIOR_RESULT_RECEIPT = (
    "e13f1a45139e28178b0e7ca28b3b647d42a427a23180bc2b7f695fda9dc109c3"
)
PRIOR_PROTOCOL_PAYLOAD_SHA256 = (
    "7ecbd8f9572a0d2f6be7eceee47cc6c62a82cd0ba8b58f942844238309769f37"
)
SEED = 82
SELECTED_CANDIDATE = {
    "candidate_id": "addressed_affine_t16_k2_gate025_g0125_r025",
    "hybrid_mode": "addressed_affine",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
PASS_STATUS = "addressed_affine_screen_passed_training_authorized"
FAIL_STATUS = "addressed_affine_screen_failed_training_blocked"


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-affine protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("hybrid_mode") != "addressed_affine"
        or architecture.get("hybrid_gain") != 0.125
        or architecture.get("recurrent_residual_ratio") != 0.25
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
        or authorization.get("prior_protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Addressed-affine screen protocol differs")
    if shared.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Addressed-vector causal result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    margins = prior.get("heldout_causal_endpoint", {}).get("mean_ce_margins", {})
    if (
        shared.canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or prior.get("status")
        != "addressed_vector_gate_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or prior.get("passed") is not False
        or prior.get("open_native_generation_authorized") is not False
        or margins.get("donor_minus_correct", 0.0) >= 0.0
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Addressed-vector donor failure does not authorize redesign")
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
        "MODEL_AUDIT_KEY": "all_wrappers_addressed_affine_content_gated",
        "PRIOR_RESULT_CODE_BINDING_KEY": "addressed_vector_result_file_sha256",
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
