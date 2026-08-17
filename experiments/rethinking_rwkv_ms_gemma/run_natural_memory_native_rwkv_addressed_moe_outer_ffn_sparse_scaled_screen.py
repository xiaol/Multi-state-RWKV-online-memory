#!/usr/bin/env python3
"""Screen the directly ablated, tighter four-anchor outer FFN."""

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
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_screen as sparse,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_alignment_residual_screen as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_projected_rwkv_hybrid_screen as hybrid_screen,
)

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_sparse_scaled_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_scaled_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "97daef8ccc57e4159f2575d3d6c7387ce5b34c0674ec55974e03605c1363021f"
)
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_screen_v2/result.json"
PRIOR_RESULT_FILE_SHA256 = "d2e25b829278a62538f7118a891133044ac5fa8d942aa125ee0a2874448a37ba"
PRIOR_RESULT_RECEIPT = "0c67b752b9357fb53755e98634d69e23cadf1e802e23c2f4f330dc195bf0d8f4"
SEED = 99
MIN_MATERIAL_LOGIT_DELTA = 1e-3
MAX_BOUNDED_LOGIT_DELTA = 2.0
MAX_OUTER_ONLY_LOGIT_DELTA = 0.5
OUTER_FFN_GAIN = 1.0 / 8192.0
OUTER_FFN_LAYERS = sparse.OUTER_FFN_LAYERS
PASS_STATUS = "addressed_moe_outer_ffn_sparse_scaled_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_sparse_scaled_screen_failed_training_blocked"
SELECTED_CANDIDATE = {
    **sparse.SELECTED_CANDIDATE,
    "candidate_id": "addressed_moe_outer_ffn_sparse_t16_k2_ag03125_fg0001220703125_l10_21_31_41",
    "outer_ffn_gain": OUTER_FFN_GAIN,
}
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_outer_ffn_sparse_scaled"
PRIOR_RESULT_CODE_BINDING_KEY = "outer_ffn_sparse_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)
BASE_LOCAL_EVIDENCE = shared.local_evidence


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Scaled sparse outer-FFN protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("outer_ffn_gain") != OUTER_FFN_GAIN
        or architecture.get("outer_ffn_layers") != list(OUTER_FFN_LAYERS)
        or authorization.get("prior_result_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Scaled sparse outer-FFN protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Sparse outer-FFN result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    failures = [
        row.get("checks", {}).get("correct_vs_projected_bounded") is False
        for row in prior.get("rank_evidence", [])
    ]
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status") != sparse.FAIL_STATUS
        or prior.get("passed") is not False
        or failures.count(True) != 1
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Sparse outer-FFN bound failure does not authorize rescaling")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return replace(
        sparse.build_config(candidate),
        rwkv_ms_outer_ffn_gain=float(candidate["outer_ffn_gain"]),
    )


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = sparse.outer.outer.BASE_HYBRID_LOAD_MODEL(
        base_model,
        device=device,
        delta_config=build_config(),
    )
    sparse.outer.outer.candidate_helper.configure_candidate(model, SELECTED_CANDIDATE)
    modules = tuple(iter_delta_mem_modules(model))
    family_counts = {
        family: sum(hasattr(module, family) for _, module in modules)
        for family in sparse.outer.outer.OUTER_FFN_FAMILIES
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
            module.rwkv_ms_hybrid_mode == "addressed_moe_outer_ffn"
            and module.rwkv_ms_hybrid_gain == 0.03125
            and module.rwkv_ms_outer_ffn_gain == OUTER_FFN_GAIN
            and module.rwkv_ms_outer_ffn_layers == OUTER_FFN_LAYERS
            and module.rwkv_ms_read_temperature == 16.0
            and module.rwkv_ms_read_top_k == 2
            and module.rwkv_ms_detach_read_scores is True
            and hasattr(module, "rwkv_moe_bias")
            for _, module in modules
        )
    )
    audit = {
        **dict(inherited_audit),
        MODEL_AUDIT_KEY: configured,
        "outer_ffn_gain": OUTER_FFN_GAIN,
        "outer_ffn_hook_count": hook_count,
        "outer_ffn_active_layers": list(enabled_layers),
        "outer_ffn_family_counts": family_counts,
    }
    if not configured:
        raise RuntimeError(f"Scaled sparse outer-FFN attachment failed: {audit!r}")
    return model, tokenizer, audit


def local_evidence(
    model: torch.nn.Module,
    batch: Any,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    evidence = dict(BASE_LOCAL_EVIDENCE(model, batch, states))
    outer_logits = hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="addressed_moe_outer_ffn",
        hybrid_gain=0.03125,
    )
    attention_only_logits = hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="addressed_moe_controller",
        hybrid_gain=0.03125,
    )
    comparison = hybrid_screen.compare_logits(outer_logits, attention_only_logits)
    checks = dict(evidence["checks"])
    checks["outer_ffn_vs_attention_only_material"] = (
        comparison["max_abs_logit_delta"] >= MIN_MATERIAL_LOGIT_DELTA
    )
    checks["outer_ffn_vs_attention_only_bounded"] = (
        comparison["max_abs_logit_delta"] <= MAX_OUTER_ONLY_LOGIT_DELTA
    )
    comparisons = dict(evidence["comparisons"])
    comparisons["outer_ffn_vs_attention_only"] = comparison
    return {
        **evidence,
        "checks": checks,
        "comparisons": comparisons,
        "passed": all(checks.values()),
    }


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
