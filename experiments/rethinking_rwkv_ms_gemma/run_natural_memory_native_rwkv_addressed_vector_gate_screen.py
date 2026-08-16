#!/usr/bin/env python3
"""Screen a projected-addressed RWKV vector-FiLM controller."""

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


SCHEMA = "rwkv_ms_natural_memory_native_addressed_vector_gate_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_vector_gate_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "763a029133c7125dffa586d80d8fcb129954a9dca2b5b9a98a448be334bdeb6d"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_aligned_vector_gate_specificity_eval_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "05342b71790f8584c5f856765896670f60cbed2a3f8090ae1eaeaa50a4ecf747"
)
PRIOR_RESULT_RECEIPT = (
    "b2c3e76cebc7744021393dd5028d568e8ea334b2008414e6ce0a014ccc9c8c65"
)
PRIOR_PROTOCOL_PAYLOAD_SHA256 = (
    "7ccad9ace385918324e59c16a0c60844ee467564cf8968f79910b9bb6d3ac1ff"
)
SEED = 80
SELECTED_CANDIDATE = {
    "candidate_id": "addressed_vector_gate_t16_k2_gate025_g0125",
    "hybrid_mode": "addressed_vector_gate",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
PASS_STATUS = "addressed_vector_gate_screen_passed_training_authorized"
FAIL_STATUS = "addressed_vector_gate_screen_failed_training_blocked"


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Addressed-vector protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("hybrid_mode") != "addressed_vector_gate"
        or architecture.get("hybrid_gain") != 0.125
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
        or authorization.get("prior_protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Addressed-vector screen protocol differs")
    if shared.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Aligned-vector native result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    margins = prior.get("causal_margins", {})
    if (
        shared.canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or prior.get("status")
        != "aligned_vector_gate_native_gain_not_established"
        or prior.get("passed") is not False
        or prior.get("native_recurrent_causal_gain_established") is not False
        or margins.get("correct_minus_matched_donor_micro_f1", 0.0) >= 0.0
        or margins.get("correct_minus_layer_permuted_micro_f1", 0.0) >= 0.0
        or any(
            prior.get("scope", {}).get(key) is not False
            for key in (
                "publisher_validation_predictions_opened",
                "publisher_test_opened",
                "hard32_opened",
                "strength_holdout_opened",
            )
        )
    ):
        raise ValueError("Aligned-vector native failure does not authorize redesign")
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
        "MODEL_AUDIT_KEY": "all_wrappers_addressed_vector_gate_content_gated",
        "PRIOR_RESULT_CODE_BINDING_KEY": "aligned_native_result_file_sha256",
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
