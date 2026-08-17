#!/usr/bin/env python3
"""Screen a memory-safe four-anchor DeepEmbed FFN placement."""

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
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_sparse_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "38f5a4f2a451b9ae6360d6718a6696a19c24f222a0febaf88e5e794073a022f7"
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_screen_v2/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "4d14fc15e7648d2403faea48c057e41f895a55d5ea15b3c56d9220c5b816f1d8"
)
PRIOR_RESULT_RECEIPT = (
    "5dae1e213ae2f8808b7d9278fb1f9033caa03f54f05321f99aabfe16c8490e02"
)
MEMORY_FAILURE = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train_v4/failure.json"
)
MEMORY_FAILURE_FILE_SHA256 = (
    "97a7ed46d6a7258ae1216077ac7ef6e632b635ac5334ea3aab66b6a587262b77"
)
MEMORY_FAILURE_RECEIPT = (
    "0dc8b7e793b353a5f9ec5baea71732c2220f305bb6dfedb04288845d379c6091"
)
SEED = 103
OUTER_FFN_LAYERS = (10, 21, 31, 41)
PASS_STATUS = "addressed_moe_deepembed_ffn_sparse_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_deepembed_ffn_sparse_screen_failed_training_blocked"
CANDIDATES = (
    {
        "candidate_id": "deepembed_ffn_sparse_t16_k2_ag015625_fg0078125_l10_21_31_41",
        "hybrid_mode": "addressed_moe_deepembed_ffn",
        "hybrid_gain": 0.015625,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 128.0,
        "outer_ffn_layers": list(OUTER_FFN_LAYERS),
    },
)
RUNNER_BINDING_PATH = Path(__file__)


def validate_signed_result(
    path: Path,
    *,
    file_sha256: str,
    receipt_sha256: str,
) -> Mapping[str, Any]:
    if base.sha256_file(path) != file_sha256:
        raise ValueError(f"Signed prerequisite file differs: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Signed prerequisite receipt is missing: {path}")
    unsigned = dict(value)
    unsigned.pop("receipt")
    if (
        base.canonical_sha256(unsigned) != receipt_sha256
        or receipt.get("payload_sha256") != receipt_sha256
    ):
        raise ValueError(f"Signed prerequisite receipt differs: {path}")
    return value


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Sparse DeepEmbed protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        base.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("candidate_grid") != list(CANDIDATES)
        or architecture.get("hybrid_mode") != "addressed_moe_deepembed_ffn"
        or architecture.get("outer_ffn_layers") != list(OUTER_FFN_LAYERS)
        or authorization.get("deepembed_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("deepembed_result_receipt") != PRIOR_RESULT_RECEIPT
        or authorization.get("memory_failure_file_sha256")
        != MEMORY_FAILURE_FILE_SHA256
        or authorization.get("memory_failure_receipt") != MEMORY_FAILURE_RECEIPT
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Sparse DeepEmbed screen protocol differs")
    prior = validate_signed_result(
        PRIOR_RESULT,
        file_sha256=PRIOR_RESULT_FILE_SHA256,
        receipt_sha256=PRIOR_RESULT_RECEIPT,
    )
    failure = validate_signed_result(
        MEMORY_FAILURE,
        file_sha256=MEMORY_FAILURE_FILE_SHA256,
        receipt_sha256=MEMORY_FAILURE_RECEIPT,
    )
    if (
        prior.get("status")
        != "addressed_moe_deepembed_ffn_bf16_screen_passed_training_authorized"
        or prior.get("passed") is not True
        or prior.get("selected_candidate", {}).get("outer_ffn_gain") != 1.0 / 128.0
        or prior.get("training_authorized") is not True
        or prior.get("protected_splits_opened") != []
        or failure.get("status") != "addressed_moe_outer_ffn_training_oom_endpoint_unopened"
        or failure.get("completed_optimizer_updates") != 5
        or failure.get("heldout_causal_endpoint_opened") is not False
        or failure.get("protected_splits_opened") != []
    ):
        raise ValueError("Sparse DeepEmbed prerequisites do not authorize redesign")
    return protocol


def build_config(candidate: Mapping[str, Any]) -> Any:
    return replace(
        base.top2_screen.build_config(candidate),
        rwkv_ms_hybrid_mode="addressed_moe_deepembed_ffn",
        rwkv_ms_outer_ffn_gain=float(candidate["outer_ffn_gain"]),
        rwkv_ms_outer_ffn_layers=tuple(candidate["outer_ffn_layers"]),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    candidate = base.ACTIVE_CANDIDATE
    model, tokenizer, inherited_audit = base.hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=build_config(candidate),
    )
    base.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    deepembed_pre_hooks = sum(
        module._deepembed_ffn_pre_hook_handle is not None
        for _, module in modules
    )
    deepembed_down_hooks = sum(
        module._deepembed_ffn_down_pre_hook_handle is not None
        for _, module in modules
    )
    family_counts = {
        family: sum(hasattr(module, family) for _, module in modules)
        for family in (
            "rwkv_outer_ffn_down_weight",
            "rwkv_outer_ffn_gate_weight",
            "rwkv_outer_ffn_up_weight",
        )
    }
    enabled_layers = tuple(
        module.layer_idx for _, module in modules if module.rwkv_ms_outer_ffn_enabled
    )
    configured = (
        len(modules) == base.preflight.EXPECTED_LAYERS
        and deepembed_pre_hooks == base.preflight.EXPECTED_LAYERS
        and deepembed_down_hooks == base.preflight.EXPECTED_LAYERS
        and enabled_layers == OUTER_FFN_LAYERS
        and all(count == len(OUTER_FFN_LAYERS) for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "addressed_moe_deepembed_ffn"
            and module.rwkv_ms_hybrid_gain == float(candidate["hybrid_gain"])
            and module.rwkv_ms_outer_ffn_gain == float(candidate["outer_ffn_gain"])
            and module.rwkv_ms_outer_ffn_layers == OUTER_FFN_LAYERS
            and module.rwkv_ms_read_temperature == float(candidate["read_temperature"])
            and module.rwkv_ms_read_top_k == int(candidate["read_top_k"])
            and module.rwkv_ms_detach_read_scores is True
            and module.memory_fusion_mode == "content_gated_add"
            and hasattr(module, "rwkv_moe_bias")
            for _, module in modules
        )
    )
    audit = {
        **dict(inherited_audit),
        "all_wrappers_addressed_moe_deepembed_ffn": configured,
        "deepembed_ffn_pre_hook_count": deepembed_pre_hooks,
        "deepembed_ffn_down_hook_count": deepembed_down_hooks,
        "deepembed_ffn_family_counts": family_counts,
        "deepembed_ffn_active_layers": list(enabled_layers),
        "deepembed_ffn_gain": float(candidate["outer_ffn_gain"]),
    }
    if not configured:
        raise RuntimeError(f"Sparse DeepEmbed attachment failed: {audit!r}")
    return model, tokenizer, audit


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
        "load_model": load_model,
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
