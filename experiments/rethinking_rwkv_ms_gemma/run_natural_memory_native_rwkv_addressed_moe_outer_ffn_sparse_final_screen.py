#!/usr/bin/env python3
"""Run the final bounded four-anchor outer-FFN screen."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_scaled_screen as base,
)

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_sparse_final_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_final_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "22ad5ee06e0dced49b51c6cbd84c3a6f44956e7bb6927aec03e441ae1bcd92b7"
)
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_scaled_screen_v1/result.json"
PRIOR_RESULT_FILE_SHA256 = "832b40ec199d80e2e93cab86f35c3405120213d5ef87074be05b3454b3787ea8"
PRIOR_RESULT_RECEIPT = "f40ed5bf23dbab372a671f5ef7cee0796bb9f7af7036c722e54c9373e32f82fc"
SEED = 99
OUTER_FFN_GAIN = 1.0 / 32768.0
PASS_STATUS = "addressed_moe_outer_ffn_sparse_final_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_sparse_final_screen_failed_branch_stopped"
SELECTED_CANDIDATE = {
    **base.SELECTED_CANDIDATE,
    "candidate_id": "addressed_moe_outer_ffn_sparse_t16_k2_ag03125_fg000030517578125_l10_21_31_41",
    "outer_ffn_gain": OUTER_FFN_GAIN,
}
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_outer_ffn_sparse_final"
PRIOR_RESULT_CODE_BINDING_KEY = "outer_ffn_sparse_scaled_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)
BASE_BUILD_CONFIG = base.build_config


def canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Final sparse outer-FFN protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("outer_ffn_gain") != OUTER_FFN_GAIN
        or architecture.get("outer_ffn_layers") != list(base.OUTER_FFN_LAYERS)
        or authorization.get("prior_result_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Final sparse outer-FFN protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Scaled sparse outer-FFN result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    evidence = prior.get("rank_evidence", [])
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "addressed_moe_outer_ffn_sparse_scaled_screen_failed_training_blocked"
        or len(evidence) != 4
        or not all(
            row.get("checks", {}).get("outer_ffn_vs_attention_only_bounded")
            is False
            and row.get("checks", {}).get("correct_vs_projected_bounded") is True
            for row in evidence
        )
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Scaled sparse failure does not authorize final rescaling")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return BASE_BUILD_CONFIG(candidate)


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "PRIOR_RESULT": PRIOR_RESULT,
        "PRIOR_RESULT_FILE_SHA256": PRIOR_RESULT_FILE_SHA256,
        "PRIOR_RESULT_RECEIPT": PRIOR_RESULT_RECEIPT,
        "SEED": SEED,
        "OUTER_FFN_GAIN": OUTER_FFN_GAIN,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MODEL_AUDIT_KEY": MODEL_AUDIT_KEY,
        "PRIOR_RESULT_CODE_BINDING_KEY": PRIOR_RESULT_CODE_BINDING_KEY,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "validate_protocol": validate_protocol,
        "build_config": build_config,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
