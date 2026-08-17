#!/usr/bin/env python3
"""Screen the bounded addressed/global RWKV mixture-of-experts controller."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

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

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_controller_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_controller_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "e0b5d8f370957e3c7bb340047be718088e8e83e5484a40df55aeceb3f8435c6f"
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_query_state_gate_causal_train_v1/result.json"
PRIOR_RESULT_FILE_SHA256 = "7bc255f1784c2e36df9ef2abd53903e634d3164e51f29631911fa2235b736343"
PRIOR_RESULT_RECEIPT = "a4f9bb3943a9e47cfd8457d58aea52ba9403998b5eddd440ae8d657b1d6b0512"
SEED = 91
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
PASS_STATUS = "addressed_moe_controller_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_controller_screen_failed_training_blocked"
CANDIDATES = (
    {
        "candidate_id": "addressed_moe_controller_t16_k2_g03125",
        "hybrid_mode": "addressed_moe_controller",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
    },
)
SELECTED_CANDIDATE = CANDIDATES[0]
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_controller"
PRIOR_RESULT_CODE_BINDING_KEY = "addressed_query_state_gate_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)
BASE_HYBRID_LOAD_MODEL = hybrid_screen.load_model


def canonical_sha256(value: Any) -> str:
    return hybrid_screen.distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return hybrid_screen.preflight.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("MoE screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("hybrid_mode") != "addressed_moe_controller"
        or architecture.get("hybrid_gain") != 0.03125
        or authorization.get("prior_result_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("MoE screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Query-state-gate result file binding differs")
    return protocol


def build_config(
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> Any:
    return replace(
        top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="addressed_moe_controller",
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = BASE_HYBRID_LOAD_MODEL(
        base_model,
        device=device,
        delta_config=build_config(),
    )
    candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == "addressed_moe_controller"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and hasattr(module, "rwkv_moe_bias")
        for _, module in modules
    )
    audit = {**dict(inherited_audit), MODEL_AUDIT_KEY: configured}
    if not configured:
        raise RuntimeError(f"MoE attachment failed: {audit!r}")
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
        "MIN_MATERIAL_LOGIT_DELTA": MIN_MATERIAL_LOGIT_DELTA,
        "MAX_BOUNDED_LOGIT_DELTA": MAX_BOUNDED_LOGIT_DELTA,
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
