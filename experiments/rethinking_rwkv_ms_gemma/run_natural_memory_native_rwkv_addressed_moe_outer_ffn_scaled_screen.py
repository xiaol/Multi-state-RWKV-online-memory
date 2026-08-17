#!/usr/bin/env python3
"""Screen the layer-safe scaled addressed-MoE outer-FFN hybrid."""

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
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_screen as outer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_scaled_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "6bd5aa2f3d4d2d2aa190b9db8aca74245408a35994327a73b70411822c63e672"
)
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_screen_v1/"
    "result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "b8ea91e7ecee62573d383393d736fff8dfdf6fa5a2a701debaf000c8361ca2c3"
)
PRIOR_RESULT_RECEIPT = (
    "ff23246156174653dca227a3c0cd186124498aba8887725157a69ab0c1484f4f"
)
SEED = 98
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
OUTER_FFN_GAIN = 1.0 / 2048.0
PASS_STATUS = "addressed_moe_outer_ffn_scaled_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_scaled_screen_failed_training_blocked"
SELECTED_CANDIDATE = {
    **outer.SELECTED_CANDIDATE,
    "candidate_id": "addressed_moe_outer_ffn_t16_k2_ag03125_fg00048828125",
    "outer_ffn_gain": OUTER_FFN_GAIN,
}
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_outer_ffn_scaled"
PRIOR_RESULT_CODE_BINDING_KEY = "outer_ffn_unscaled_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Scaled outer-FFN screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("hybrid_mode") != "addressed_moe_outer_ffn"
        or architecture.get("attention_hybrid_gain") != 0.03125
        or architecture.get("outer_ffn_gain") != OUTER_FFN_GAIN
        or authorization.get("prior_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Scaled outer-FFN screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Unscaled outer-FFN result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    rank_evidence = prior.get("rank_evidence", [])
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "addressed_moe_outer_ffn_screen_failed_training_blocked"
        or prior.get("passed") is not False
        or prior.get("training_authorized") is not False
        or len(rank_evidence) != 4
        or not all(
            row.get("checks", {}).get("correct_vs_projected_bounded") is False
            and row.get("checks", {}).get(
                "zero_recurrent_exactly_equals_projected_only"
            )
            is True
            and row.get("checks", {}).get("all_condition_logits_finite") is True
            for row in rank_evidence
        )
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Unscaled outer-FFN failure does not authorize rescaling")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return replace(
        outer.top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="addressed_moe_outer_ffn",
        rwkv_ms_outer_ffn_gain=float(candidate["outer_ffn_gain"]),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = outer.BASE_HYBRID_LOAD_MODEL(
        base_model,
        device=device,
        delta_config=build_config(),
    )
    outer.candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    family_counts = {
        family: sum(hasattr(module, family) for _, module in modules)
        for family in outer.OUTER_FFN_FAMILIES
    }
    hook_count = sum(
        module._post_feedforward_norm_hook_handle is not None
        for _, module in modules
    )
    configured = (
        len(modules) == 42
        and hook_count == 42
        and all(count == 42 for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "addressed_moe_outer_ffn"
            and module.rwkv_ms_hybrid_gain == 0.03125
            and module.rwkv_ms_outer_ffn_gain == OUTER_FFN_GAIN
            and module.rwkv_ms_read_temperature == 16.0
            and module.rwkv_ms_read_top_k == 2
            and module.rwkv_ms_detach_read_scores is True
            and module.memory_fusion_mode == "content_gated_add"
            and hasattr(module, "rwkv_moe_bias")
            for _, module in modules
        )
    )
    audit = {
        **dict(inherited_audit),
        MODEL_AUDIT_KEY: configured,
        "outer_ffn_gain": OUTER_FFN_GAIN,
        "outer_ffn_hook_count": hook_count,
        "outer_ffn_family_counts": family_counts,
    }
    if not configured:
        raise RuntimeError(f"Scaled outer-FFN attachment failed: {audit!r}")
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
