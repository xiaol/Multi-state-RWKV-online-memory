#!/usr/bin/env python3
"""Screen differentiable top-2 RWKV routing with learned abstention."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_screen as recurrent_screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as shared,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_top2_abstention_screen.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_top2_abstention_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "26b0bd889f6ab28167d286dcd90117ad225a27c1a80c5dc7c629d147d6a18976"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_sharp_router_calibration_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "714b9cd3817e028057e5897ec456b104329a7a9516b728947a8504bfac31f268"
)
PRIOR_RESULT_RECEIPT = (
    "743163d5b6211864b2d65465349c8e597564cd24b42e9c7f32af00d66c5ffa69"
)
AUTHORIZATION_BASIS = {
    "prior_result_file": (
        "local_artifacts/"
        "natural_memory_native_rwkv_sharp_router_calibration_v1/result.json"
    ),
    "prior_result_file_sha256": PRIOR_RESULT_FILE_SHA256,
    "prior_result_receipt": PRIOR_RESULT_RECEIPT,
    "prior_status": "calibration_failed_causal_training_blocked",
    "prior_outcome": (
        "Hard top-1 forward with full-softmax straight-through backward made "
        "1224 of 1260 active trainable gradients non-finite on every "
        "calibration rank."
    ),
    "architectural_response": (
        "Remove the straight-through estimator, retain sparse differentiable "
        "top-2 routing, and add a learned content gate as an explicit "
        "abstention path."
    ),
}
PRIOR_REQUIRED_RESULT = {
    "status": "calibration_failed_causal_training_blocked",
    "passed": False,
    "failure_phase": "local_backward_gradient_validation",
    "causal_training_authorized": False,
    "protected_splits_opened": [],
}
SEED = 66
CANDIDATES = (
    {
        "candidate_id": "recurrent_value_t16_k2_gate025",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
    },
    {
        "candidate_id": "recurrent_value_t16_k2_gate050",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.5,
    },
    {
        "candidate_id": "recurrent_value_t8_k2_gate025",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
        "read_temperature": 8.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
    },
    {
        "candidate_id": "recurrent_value_t16_k3_gate025",
        "hybrid_mode": "recurrent_value",
        "hybrid_gain": 0.03125,
        "read_temperature": 16.0,
        "read_top_k": 3,
        "fusion_gate_probability": 0.25,
    },
)


def build_config(candidate: Mapping[str, Any] = CANDIDATES[0]) -> Any:
    return replace(
        recurrent_screen.build_config(),
        memory_fusion_mode="content_gated_add",
        memory_fusion_gate_init=float(candidate["fusion_gate_probability"]),
        rwkv_ms_hybrid_gain=float(candidate["hybrid_gain"]),
        rwkv_ms_read_temperature=float(candidate["read_temperature"]),
        rwkv_ms_read_top_k=int(candidate["read_top_k"]),
        rwkv_ms_detach_read_scores=bool(candidate.get("detach_read_scores", False)),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = CANDIDATES[0],
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=build_config(candidate),
    )
    hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=str(candidate["hybrid_mode"]),
        hybrid_gain=float(candidate["hybrid_gain"]),
    )
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "recurrent_value"
        and module.memory_fusion_mode == "content_gated_add"
        and hasattr(module, "memory_fusion_bias")
        for _, module in modules
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_recurrent_value_content_gated": configured,
    }
    if not configured:
        raise RuntimeError(f"Top-2 abstention attachment failed: {audit!r}")
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
        "AUTHORIZATION_BASIS": AUTHORIZATION_BASIS,
        "PRIOR_REQUIRED_RESULT": PRIOR_REQUIRED_RESULT,
        "SEED": SEED,
        "CANDIDATES": CANDIDATES,
        "RUNNER_BINDING_PATH": Path(__file__),
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


def validate_protocol() -> Mapping[str, Any]:
    with screen_bindings():
        return shared.validate_protocol()


def validate_prior_result() -> Mapping[str, Any]:
    with screen_bindings():
        return shared.validate_prior_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with screen_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with screen_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
