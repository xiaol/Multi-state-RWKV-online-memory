#!/usr/bin/env python3
"""Screen the sparse outer FFN with a same-mode zero-gain ablation."""

from __future__ import annotations

from contextlib import contextmanager
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
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_scaled_screen as base,
)

SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_outer_ffn_gain_ablation_screen.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_outer_ffn_gain_ablation_screen_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "56e535300a4ab635e4b8d5de1820ce0ff3a0e174bff00c7a3c2dc34d03418ff3"
)
PRIOR_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_outer_ffn_sparse_final_screen_v2/result.json"
PRIOR_RESULT_FILE_SHA256 = "8d9edd05895be493be80d573c94b4635a55a99d49005b59ef7458f6f528af1cd"
PRIOR_RESULT_RECEIPT = "d93e6fe2d373a7570f8290fcd63f6a0ce7b5b947f0aff09cece5ec0fe0ebea05"
SEED = 99
OUTER_FFN_GAIN = 1.0 / 8192.0
PASS_STATUS = "addressed_moe_outer_ffn_gain_ablation_screen_passed_training_authorized"
FAIL_STATUS = "addressed_moe_outer_ffn_gain_ablation_screen_failed_branch_stopped"
SELECTED_CANDIDATE = {
    **base.SELECTED_CANDIDATE,
    "candidate_id": "addressed_moe_outer_ffn_gain_ablation_t16_k2_ag03125_fg0001220703125_l10_21_31_41",
    "outer_ffn_gain": OUTER_FFN_GAIN,
}
MODEL_AUDIT_KEY = "all_wrappers_addressed_moe_outer_ffn_gain_ablation"
PRIOR_RESULT_CODE_BINDING_KEY = "outer_ffn_invalid_ablation_result_file_sha256"
RUNNER_BINDING_PATH = Path(__file__)
BASE_BUILD_CONFIG = base.build_config
BASE_LOCAL_EVIDENCE = base.BASE_LOCAL_EVIDENCE


def canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Gain-ablation screen protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
        or architecture.get("outer_ffn_gain") != OUTER_FFN_GAIN
        or architecture.get("outer_ffn_layers") != list(base.OUTER_FFN_LAYERS)
        or authorization.get("prior_result_file_sha256") != PRIOR_RESULT_FILE_SHA256
        or authorization.get("prior_result_receipt") != PRIOR_RESULT_RECEIPT
    ):
        raise ValueError("Gain-ablation screen protocol differs")
    if sha256_file(PRIOR_RESULT) != PRIOR_RESULT_FILE_SHA256:
        raise ValueError("Invalid-ablation result file binding differs")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    unsigned_prior = dict(prior)
    prior_receipt = unsigned_prior.pop("receipt", {})
    direct = [
        row.get("comparisons", {}).get("outer_ffn_vs_attention_only", {}).get(
            "max_abs_logit_delta"
        )
        for row in prior.get("rank_evidence", [])
    ]
    if (
        canonical_sha256(unsigned_prior) != PRIOR_RESULT_RECEIPT
        or prior_receipt.get("payload_sha256") != PRIOR_RESULT_RECEIPT
        or prior.get("status")
        != "addressed_moe_outer_ffn_sparse_final_screen_failed_branch_stopped"
        or len(direct) != 4
        or not all(value is not None and value > 1.0 for value in direct)
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("Invalid mode-switch ablation does not authorize correction")
    return protocol


def build_config(candidate: Mapping[str, Any] = SELECTED_CANDIDATE) -> Any:
    return BASE_BUILD_CONFIG(candidate)


def local_evidence(
    model: torch.nn.Module,
    batch: Any,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    evidence = dict(BASE_LOCAL_EVIDENCE(model, batch, states))
    outer_logits = base.hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="addressed_moe_outer_ffn",
        hybrid_gain=0.03125,
    )
    modules = tuple(iter_delta_mem_modules(model))
    previous_gains = tuple(module.rwkv_ms_outer_ffn_gain for _, module in modules)
    try:
        for _, module in modules:
            module.rwkv_ms_outer_ffn_gain = 0.0
        zero_gain_logits = base.hybrid_screen.read_logits(
            model,
            batch,
            states["correct"],
            readout_mode="projected_kv_rwkv_hybrid",
            hybrid_mode="addressed_moe_outer_ffn",
            hybrid_gain=0.03125,
        )
    finally:
        for (_, module), gain in zip(modules, previous_gains):
            module.rwkv_ms_outer_ffn_gain = gain
    comparison = base.hybrid_screen.compare_logits(outer_logits, zero_gain_logits)
    checks = dict(evidence["checks"])
    checks["outer_ffn_gain_vs_zero_gain_material"] = (
        comparison["max_abs_logit_delta"] >= base.MIN_MATERIAL_LOGIT_DELTA
    )
    checks["outer_ffn_gain_vs_zero_gain_bounded"] = (
        comparison["max_abs_logit_delta"] <= base.MAX_OUTER_ONLY_LOGIT_DELTA
    )
    comparisons = dict(evidence["comparisons"])
    comparisons["outer_ffn_gain_vs_zero_gain"] = comparison
    return {
        **evidence,
        "checks": checks,
        "comparisons": comparisons,
        "passed": all(checks.values()),
    }


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "PRIOR_RESULT": PRIOR_RESULT,
        "PRIOR_RESULT_FILE_SHA256": PRIOR_RESULT_FILE_SHA256,
        "PRIOR_RESULT_RECEIPT": PRIOR_RESULT_RECEIPT,
        "SEED": SEED,
        "OUTER_FFN_GAIN": OUTER_FFN_GAIN,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "MODEL_AUDIT_KEY": MODEL_AUDIT_KEY,
        "PRIOR_RESULT_CODE_BINDING_KEY": PRIOR_RESULT_CODE_BINDING_KEY,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "validate_protocol": validate_protocol,
        "build_config": build_config,
        "local_evidence": local_evidence,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
