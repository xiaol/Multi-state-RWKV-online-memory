#!/usr/bin/env python3
"""Screen address-bound RWKV writes with an addressed/global MoE controller."""

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
    run_natural_memory_native_rwkv_address_bound_write_screen as bound,
)

SCHEMA = "rwkv_ms_natural_memory_native_address_bound_moe_controller_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_bound_moe_controller_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "dbb12dffeddaeb4f1ec4ba24b663ecdead3c8571770383ebd5550a5b578bb0fa"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_bound_write_causal_train_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "6227f5ad9f252ec9709c627da9d188394c88f7ec7f4b5704ce174032acaf3d45"
)
PRIOR_RESULT_RECEIPT = (
    "438205f95fc7b4b41d0828a6c5a163aa2e09eb3098253f07399d7c1c921d440c"
)
SEED = 95
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
PASS_STATUS = "address_bound_moe_controller_screen_passed_training_authorized"
FAIL_STATUS = "address_bound_moe_controller_screen_failed_training_blocked"
SELECTED_CANDIDATE = {
    "candidate_id": "address_bound_moe_controller_t16_k2_g03125",
    "hybrid_mode": "address_bound_moe_controller",
    "hybrid_gain": 0.03125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
MODEL_AUDIT_KEY = "all_wrappers_address_bound_moe_controller"
PRIOR_RESULT_CODE_BINDING_KEY = "address_bound_write_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return bound.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return bound.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound MoE screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or architecture.get("hybrid_mode") != "address_bound_moe_controller"
        or architecture.get("hybrid_gain") != 0.03125
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-bound MoE screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Address-bound-write result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    endpoint = prior.get("heldout_causal_endpoint", {})
    trainable = prior.get("input_binding", {}).get("trainable_audit", {})
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "address_bound_write_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or endpoint.get("checks", {}).get("donor_minus_correct_mean_ce_positive")
        is not False
        or endpoint.get("checks", {}).get(
            "layer_permuted_minus_correct_mean_ce_positive"
        )
        is not False
        or trainable.get("parameter_tensors") != 210
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Address-bound-write failure does not authorize MoE screen")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return replace(
        bound.top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="address_bound_moe_controller",
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = bound.BASE_HYBRID_LOAD_MODEL(
        base_model,
        device=device,
        delta_config=build_config(),
    )
    bound.candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.memory_readout_mode == "projected_kv_rwkv_hybrid"
        and module.rwkv_ms_hybrid_mode == "address_bound_moe_controller"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.rwkv_ms_write_mode == "recurrent"
        and hasattr(module, "rwkv_moe_bias")
        for _, module in modules
    )
    audit = {**dict(inherited_audit), MODEL_AUDIT_KEY: configured}
    if not configured:
        raise RuntimeError(f"Address-bound MoE attachment failed: {audit!r}")
    return model, tokenizer, audit


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    bound.hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="address_bound_moe_controller",
        hybrid_gain=0.03125,
    )
    bound.hybrid_screen.reset_delta_mem_states(model)
    bound.hybrid_screen.set_delta_mem_write_enabled(model, True)
    with torch.inference_mode(), bound.hybrid_screen.runtime._autocast_context(
        input_ids.device,
        torch.bfloat16,
    ):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    state = bound.hybrid_screen.get_delta_mem_online_state(model)
    bound.hybrid_screen.audit_hybrid_state(state)
    return state


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
        "local_evidence": bound.local_evidence,
    }
    previous = {name: getattr(bound.shared, name) for name in bindings}
    previous_write_state = bound.hybrid_screen.write_state
    try:
        for name, value in bindings.items():
            setattr(bound.shared, name, value)
        bound.hybrid_screen.write_state = write_state
        yield
    finally:
        bound.hybrid_screen.write_state = previous_write_state
        for name, value in previous.items():
            setattr(bound.shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with screen_bindings():
        return bound.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with screen_bindings():
        return bound.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
