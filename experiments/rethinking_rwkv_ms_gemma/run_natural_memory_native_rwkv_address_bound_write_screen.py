#!/usr/bin/env python3
"""Screen recurrent writes bound to projected memory addresses."""

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

SCHEMA = "rwkv_ms_natural_memory_native_address_bound_write_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_address_bound_write_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "78bfefd54d7cbca8ee69fea1084ce48fa552a6f9a68c1e8919ae65f1264854f2"
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_causal_train_v4/result.json"
PRIOR_RESULT_FILE_SHA256 = "6b4f835e487eb01bdc4058013b00df0ec4e364e7a8bf19dada182c59e4e18df2"
PRIOR_RESULT_RECEIPT = "aaec56edc52ab77a207617685e8dc3c8ead1b550614763012fe0595e222e9e29"
SEED = 93
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
PASS_STATUS = "address_bound_write_screen_passed_training_authorized"
FAIL_STATUS = "address_bound_write_screen_failed_training_blocked"
SELECTED_CANDIDATE = {
    "candidate_id": "address_bound_write_t16_k2_g003125",
    "hybrid_mode": "address_bound_write",
    "hybrid_gain": 0.03125,
    "read_temperature": 16.0,
    "read_top_k": 2,
    "fusion_gate_probability": 0.25,
    "detach_read_scores": True,
}
MODEL_AUDIT_KEY = "all_wrappers_address_bound_write"
PRIOR_RESULT_CODE_BINDING_KEY = "addressed_moe_controller_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)
BASE_HYBRID_LOAD_MODEL = hybrid_screen.load_model
BASE_LOCAL_EVIDENCE = shared.local_evidence
BASE_WRITE_STATE = hybrid_screen.write_state


def canonical_sha256(value: Any) -> str:
    return hybrid_screen.distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return hybrid_screen.preflight.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-bound-write screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or architecture.get("hybrid_mode") != "address_bound_write"
        or architecture.get("hybrid_gain") != 0.03125
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-bound-write screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Corrected MoE result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    endpoint = prior.get("heldout_causal_endpoint", {})
    trainable = prior.get("input_binding", {}).get("trainable_audit", {})
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "addressed_moe_controller_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or endpoint.get("checks", {}).get(
            "layer_permuted_minus_correct_mean_ce_positive"
        )
        is not False
        or trainable.get("moe_router_trainable_tensors") != 168
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Corrected MoE failure does not authorize write redesign")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return replace(
        top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="address_bound_write",
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
        and module.rwkv_ms_hybrid_mode == "address_bound_write"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.rwkv_ms_write_mode == "recurrent"
        for _, module in modules
    )
    audit = {**dict(inherited_audit), MODEL_AUDIT_KEY: configured}
    if not configured:
        raise RuntimeError(f"Address-bound-write attachment failed: {audit!r}")
    return model, tokenizer, audit


def local_evidence(
    model: torch.nn.Module,
    batch: Any,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    evidence = dict(BASE_LOCAL_EVIDENCE(model, batch, states))
    checks = dict(evidence["checks"])
    correct = states["correct"]
    matched_layers = 0
    total_layers = 0
    for name, _ in iter_delta_mem_modules(model):
        occupied = correct[f"{name}.__projected_kv_occupied"].to(torch.bool)
        recurrent = correct[name]
        recurrent_occupied = recurrent.ne(0).any(dim=(1, 3, 4))
        total_layers += 1
        matched_layers += int(torch.equal(occupied, recurrent_occupied))
    checks["projected_recurrent_write_slots_identical"] = (
        total_layers == 42 and matched_layers == total_layers
    )
    evidence["checks"] = checks
    evidence["passed"] = all(checks.values())
    evidence["write_binding_audit"] = {
        "matched_layers": matched_layers,
        "total_layers": total_layers,
    }
    return evidence


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="address_bound_write",
        hybrid_gain=0.03125,
    )
    hybrid_screen.reset_delta_mem_states(model)
    hybrid_screen.set_delta_mem_write_enabled(model, True)
    with torch.inference_mode(), hybrid_screen.runtime._autocast_context(
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
    state = hybrid_screen.get_delta_mem_online_state(model)
    hybrid_screen.audit_hybrid_state(state)
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
        "local_evidence": local_evidence,
    }
    previous = {name: getattr(shared, name) for name in bindings}
    previous_write_state = hybrid_screen.write_state
    try:
        for name, value in bindings.items():
            setattr(shared, name, value)
        hybrid_screen.write_state = write_state
        yield
    finally:
        hybrid_screen.write_state = previous_write_state
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
