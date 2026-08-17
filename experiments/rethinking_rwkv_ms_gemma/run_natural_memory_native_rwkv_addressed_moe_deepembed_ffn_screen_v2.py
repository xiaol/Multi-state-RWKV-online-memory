#!/usr/bin/env python3
"""Screen BF16-resolvable DeepEmbed FFN gains with a bounded attention path."""

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
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_screen.v2"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = "c4602ff981199cc9312422c88884082facbb8b9d33eb61597a6135f0b7fe157d"
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v1/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "a21eaa2a051492ccfb9ef8701f632a03a45bf3d89ec296a0ed3614de642ae236"
)
PRIOR_RESULT_RECEIPT = (
    "72c40ca277df0eb8072701052d12081eadeb7f31a24d2c7b493fa3ca7e25f9e9"
)
SEED = 102
PASS_STATUS = "addressed_moe_deepembed_ffn_bf16_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_deepembed_ffn_bf16_screen_failed_training_blocked"
CANDIDATES = (
    {
        "candidate_id": "deepembed_ffn_bf16_t16_k2_ag015625_fg0078125",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.015625,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 128.0,
    },
    {
        "candidate_id": "deepembed_ffn_bf16_t16_k2_ag015625_fg015625",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.015625,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 64.0,
    },
    {
        "candidate_id": "deepembed_ffn_bf16_t16_k2_ag015625_fg03125",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.015625,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 32.0,
    },
)
RUNNER_BINDING_PATH = Path(__file__)


def validate_prior_result() -> Mapping[str, Any]:
    if base.sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("DeepEmbed v1 result file differs")
    result = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("DeepEmbed v1 result receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        base.canonical_sha256(unsigned) != PRIOR_RESULT_RECEIPT
        or receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or result.get("status")
        != "addressed_moe_deepembed_ffn_screen_failed_training_blocked"
        or result.get("passed") is not False
        or result.get("selected_candidate") is not None
        or result.get("training_authorized") is not False
        or result.get("protected_splits_opened") != []
        or any(row.get("passed") is not False for row in result.get("candidate_results", []))
    ):
        raise ValueError("DeepEmbed v1 failure does not authorize BF16 calibration")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("DeepEmbed BF16 protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        base.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("candidate_grid") != list(CANDIDATES)
        or architecture.get("hybrid_mode") != "addressed_moe_deepembed_ffn"
        or architecture.get("attention_hybrid_gain") != 0.015625
        or architecture.get("outer_ffn_gains")
        != [candidate["outer_ffn_gain"] for candidate in CANDIDATES]
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("DeepEmbed BF16 screen protocol differs")
    validate_prior_result()
    return protocol


@contextmanager
def screen_bindings() -> Iterator[None]:
    names = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SEED": SEED,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "CANDIDATES": CANDIDATES,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "validate_protocol": validate_protocol,
    }
    previous = {name: getattr(base, name) for name in names}
    try:
        for name, value in names.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with screen_bindings():
        return base.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with screen_bindings():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
