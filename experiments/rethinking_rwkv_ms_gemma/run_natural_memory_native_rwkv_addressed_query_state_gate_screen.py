#!/usr/bin/env python3
"""Screen the supervised query/state-gated addressed RWKV hybrid."""

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

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_sharp_router_screen as candidate_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_query_state_gate_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_query_state_gate_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "b12bce10ca2efd35119caef95460ba73c522a2d83246caedb85632b0009b5847"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_route_agreement_causal_train_v1/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "fa665edfa75620de412c12f85e453cfdbb4f6f19fc15088fc0babc3bd1be2ca8"
)
PRIOR_RESULT_RECEIPT = (
    "d4f2ff8f105fbba9d29c64d1e5e56c33e067f7a674babf4805be40144aa9f622"
)
PRIOR_PROTOCOL_PAYLOAD_SHA256 = (
    "eb8e01a83e44092559efcf914401cbeb26fb1990b655d623083c8c7260e8d716"
)
SEED = 85
SELECTED_CANDIDATE = {
    "candidate_id": "addressed_query_state_gate_t16_k2_gate025_g0125",
    "hybrid_mode": "addressed_query_state_gate",
    "hybrid_gain": 0.125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
PASS_STATUS = "addressed_query_state_gate_screen_passed_training_authorized"
FAIL_STATUS = "addressed_query_state_gate_screen_failed_training_blocked"
MODEL_AUDIT_KEY = "all_wrappers_addressed_query_state_gate_content_gated"
PRIOR_RESULT_CODE_BINDING_KEY = "addressed_route_agreement_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Query-state-gate protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or architecture.get("hybrid_mode") != "addressed_query_state_gate"
        or architecture.get("hybrid_gain") != 0.125
        or architecture.get("read_temperature") != 16.0
        or architecture.get("read_top_k") != 2
        or authorization.get("prior_result_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
        or authorization.get("prior_protocol_payload_sha256")
        != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Query-state-gate screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Route-agreement causal result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    margins = prior.get("heldout_causal_endpoint", {}).get("mean_ce_margins", {})
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("protocol_payload_sha256") != PRIOR_PROTOCOL_PAYLOAD_SHA256
        or prior.get("status")
        != "addressed_route_agreement_heldout_failed_generation_blocked"
        or prior.get("passed") is not False
        or margins.get("donor_minus_correct", 0.0) >= 0.0
        or prior.get("open_native_generation_authorized") is not False
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Route-agreement failure does not authorize redesign")
    return protocol


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=top2_screen.build_config(SELECTED_CANDIDATE),
    )
    candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == "addressed_query_state_gate"
        and module.rwkv_ms_hybrid_gain == 0.125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        for _, module in modules
    )
    audit = {**dict(inherited_audit), MODEL_AUDIT_KEY: configured}
    if not configured:
        raise RuntimeError(f"Query-state-gate attachment failed: {audit!r}")
    return model, tokenizer, audit


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
        "MODEL_AUDIT_KEY": MODEL_AUDIT_KEY,
        "PRIOR_RESULT_CODE_BINDING_KEY": PRIOR_RESULT_CODE_BINDING_KEY,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "validate_protocol": validate_protocol,
        "load_model": load_model,
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
