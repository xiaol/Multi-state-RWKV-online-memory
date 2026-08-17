#!/usr/bin/env python3
"""Screen sparse depth-anchor placement for the outer-FFN hybrid."""

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
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_scaled_screen as outer,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_top2_abstention_screen as top2_screen,
)

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_sparse_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "3e163f77ca4cc74bf9f9db85f32994970691ee228eb24629c1799a71a8d3b859"
)
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train_v4/failure.json"
PRIOR_RESULT_FILE_SHA256 = "97a7ed46d6a7258ae1216077ac7ef6e632b635ac5334ea3aab66b6a587262b77"
PRIOR_RESULT_RECEIPT = "0dc8b7e793b353a5f9ec5baea71732c2220f305bb6dfedb04288845d379c6091"
SEED = 99
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
OUTER_FFN_GAIN = 1.0 / 2048.0
OUTER_FFN_LAYERS = (10, 21, 31, 41)
PASS_STATUS = "addressed_moe_outer_ffn_sparse_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_sparse_screen_failed_training_blocked"
SELECTED_CANDIDATE = {
    **outer.SELECTED_CANDIDATE,
    "candidate_id": "addressed_moe_outer_ffn_sparse_t16_k2_ag03125_fg00048828125_l10_21_31_41",
    "outer_ffn_gain": OUTER_FFN_GAIN,
    "outer_ffn_layers": list(OUTER_FFN_LAYERS),
}
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_outer_ffn_sparse"
PRIOR_RESULT_CODE_BINDING_KEY = "outer_ffn_all_layer_failure_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Sparse outer-FFN screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("hybrid_mode") != "addressed_moe_outer_ffn"
        or architecture.get("outer_ffn_layers") != list(OUTER_FFN_LAYERS)
        or authorization.get("prior_failure_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_failure_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Sparse outer-FFN screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("All-layer outer-FFN failure file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status") != "addressed_moe_outer_ffn_training_oom_endpoint_unopened"
        or prior.get("heldout_causal_endpoint_opened") is not False
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("All-layer outer-FFN failure does not authorize sparse redesign")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return replace(
        top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="addressed_moe_outer_ffn",
        rwkv_ms_outer_ffn_gain=float(candidate["outer_ffn_gain"]),
        rwkv_ms_outer_ffn_layers=tuple(candidate["outer_ffn_layers"]),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = outer.outer.BASE_HYBRID_LOAD_MODEL(
        base_model,
        device=device,
        delta_config=build_config(),
    )
    outer.outer.candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    family_counts = {
        family: sum(hasattr(module, family) for _, module in modules)
        for family in outer.outer.OUTER_FFN_FAMILIES
    }
    enabled_layers = tuple(
        module.layer_idx for _, module in modules if module.rwkv_ms_outer_ffn_enabled
    )
    hook_count = sum(
        module._post_feedforward_norm_hook_handle is not None
        for _, module in modules
    )
    configured = (
        len(modules) == 42
        and hook_count == 42
        and enabled_layers == OUTER_FFN_LAYERS
        and all(count == 4 for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "addressed_moe_outer_ffn"
            and module.rwkv_ms_hybrid_gain == 0.03125
            and module.rwkv_ms_outer_ffn_gain == OUTER_FFN_GAIN
            and module.rwkv_ms_outer_ffn_layers == OUTER_FFN_LAYERS
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
        "outer_ffn_layers": list(OUTER_FFN_LAYERS),
        "outer_ffn_hook_count": hook_count,
        "outer_ffn_active_layers": list(enabled_layers),
        "outer_ffn_family_counts": family_counts,
    }
    if not configured:
        raise RuntimeError(f"Sparse outer-FFN attachment failed: {audit!r}")
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
