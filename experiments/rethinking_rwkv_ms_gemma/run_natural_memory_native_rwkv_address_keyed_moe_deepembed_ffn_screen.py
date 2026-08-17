#!/usr/bin/env python3
"""Screen projected-address-conditioned RWKV writes with sparse DeepEmbed."""

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

from deltamem.core.delta import (  # noqa: E402
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen as base,
)


SCHEMA = "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_screen.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_screen_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = "8f6219f0c2c1f7b19ac17ed98940c5524a7a4395fb6eded1950ac8f030746895"
PRIOR_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_v1/result.json"
)
PRIOR_RESULT_FILE_SHA256 = (
    "5067878c838b55ad953b606563ce0e21a290efe55d054bc3136fad9239b488ef"
)
PRIOR_RESULT_RECEIPT = (
    "980123e096fb4125ebe0c8da98e25a0333404df4772131bf2b732b95effa4af7"
)
SEED = 105
WRITE_ADDRESS_GAIN = 0.25
PASS_STATUS = "address_keyed_moe_deepembed_ffn_screen_passed_training_authorized"
FAIL_STATUS = "address_keyed_moe_deepembed_ffn_screen_failed_training_blocked"
CANDIDATES = (
    {
        "candidate_id": "address_keyed_deepembed_t16_k2_ag015625_wag025_fg0078125",
        "hybrid_mode": "address_keyed_moe_deepembed_ffn",
        "hybrid_gain": 1.0 / 64.0,
        "write_address_gain": WRITE_ADDRESS_GAIN,
        "read_temperature": 16.0,
        "read_top_k": 2,
        "fusion_gate_probability": 0.25,
        "detach_read_scores": True,
        "outer_ffn_gain": 1.0 / 128.0,
        "outer_ffn_layers": list(base.OUTER_FFN_LAYERS),
    },
)
RUNNER_BINDING_PATH = Path(__file__)
BASE_BUILD_CONFIG = base.build_config
BASE_LOCAL_EVIDENCE = base.base.local_evidence


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-keyed DeepEmbed protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    architecture = protocol.get("architecture", {})
    authorization = protocol.get("authorization_basis", {})
    if (
        base.base.canonical_sha256(unsigned) != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("candidate_grid") != list(CANDIDATES)
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain") != WRITE_ADDRESS_GAIN
        or architecture.get("outer_ffn_layers") != list(base.OUTER_FFN_LAYERS)
        or authorization.get("deepembed_causal_result_file_sha256")
        != PRIOR_RESULT_FILE_SHA256
        or authorization.get("deepembed_causal_result_receipt")
        != PRIOR_RESULT_RECEIPT
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-keyed DeepEmbed screen protocol differs")
    prior = base.validate_signed_result(
        PRIOR_RESULT,
        file_sha256=PRIOR_RESULT_FILE_SHA256,
        receipt_sha256=PRIOR_RESULT_RECEIPT,
    )
    endpoint = prior.get("heldout_causal_endpoint", {})
    if (
        prior.get("status")
        != "addressed_moe_deepembed_ffn_sparse_heldout_failed_generation_blocked"
        or prior.get("training_passed") is not True
        or endpoint.get("checks", {}).get("donor_minus_correct_mean_ce_positive")
        is not False
        or endpoint.get("checks", {}).get(
            "layer_permuted_minus_correct_mean_ce_positive"
        )
        is not True
        or prior.get("protected_splits_opened") != []
    ):
        raise ValueError("DeepEmbed donor-identity failure does not authorize redesign")
    return protocol


def build_config(candidate: Mapping[str, Any]) -> Any:
    return replace(
        BASE_BUILD_CONFIG(candidate),
        rwkv_ms_hybrid_mode="address_keyed_moe_deepembed_ffn",
        rwkv_ms_write_address_gain=float(candidate["write_address_gain"]),
    )


def configure_candidate(
    model: torch.nn.Module,
    candidate: Mapping[str, Any],
) -> None:
    base.base.configure_candidate(model, candidate)
    for _, module in iter_delta_mem_modules(model):
        module.rwkv_ms_write_address_gain = float(candidate["write_address_gain"])


def load_model(
    base_model: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    candidate = base.base.ACTIVE_CANDIDATE
    model, tokenizer, inherited_audit = base.base.hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=build_config(candidate),
    )
    configure_candidate(model, candidate)
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
        len(modules) == base.base.preflight.EXPECTED_LAYERS
        and deepembed_pre_hooks == base.base.preflight.EXPECTED_LAYERS
        and deepembed_down_hooks == base.base.preflight.EXPECTED_LAYERS
        and enabled_layers == base.OUTER_FFN_LAYERS
        and all(count == len(base.OUTER_FFN_LAYERS) for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode
            == "address_keyed_moe_deepembed_ffn"
            and module.rwkv_ms_hybrid_gain == float(candidate["hybrid_gain"])
            and module.rwkv_ms_write_address_gain
            == float(candidate["write_address_gain"])
            and module.projected_kv_key_dim % module.state_read_dim == 0
            and module.rwkv_ms_outer_ffn_gain
            == float(candidate["outer_ffn_gain"])
            and module.rwkv_ms_outer_ffn_layers == base.OUTER_FFN_LAYERS
            and module.rwkv_ms_read_temperature
            == float(candidate["read_temperature"])
            and module.rwkv_ms_read_top_k == int(candidate["read_top_k"])
            and module.rwkv_ms_detach_read_scores is True
            and module.rwkv_ms_write_mode == "recurrent"
            and module.memory_fusion_mode == "content_gated_add"
            and hasattr(module, "rwkv_moe_bias")
            for _, module in modules
        )
    )
    audit = {
        **dict(inherited_audit),
        # The delegated screen currently looks up this legacy key.
        "all_wrappers_addressed_moe_deepembed_ffn": configured,
        "all_wrappers_address_keyed_moe_deepembed_ffn": configured,
        "deepembed_ffn_pre_hook_count": deepembed_pre_hooks,
        "deepembed_ffn_down_hook_count": deepembed_down_hooks,
        "deepembed_ffn_family_counts": family_counts,
        "deepembed_ffn_active_layers": list(enabled_layers),
        "write_address_gain": float(candidate["write_address_gain"]),
    }
    if not configured:
        raise RuntimeError(f"Address-keyed DeepEmbed attachment failed: {audit!r}")
    return model, tokenizer, audit


def write_state(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    candidate = base.base.ACTIVE_CANDIDATE
    base.base.hybrid_screen.configure_readout(
        model,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode="address_keyed_moe_deepembed_ffn",
        hybrid_gain=float(candidate["hybrid_gain"]),
    )
    reset_delta_mem_states(model)
    set_delta_mem_write_enabled(model, True)
    with torch.inference_mode(), base.base.runtime._autocast_context(
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
    state = get_delta_mem_online_state(model)
    base.base.hybrid_screen.audit_hybrid_state(state)
    return state


def _set_write_address_gain(
    model: torch.nn.Module,
    gain: float,
) -> list[float]:
    previous = []
    for _, module in iter_delta_mem_modules(model):
        previous.append(float(module.rwkv_ms_write_address_gain))
        module.rwkv_ms_write_address_gain = float(gain)
    return previous


def _restore_write_address_gain(
    model: torch.nn.Module,
    previous: Sequence[float],
) -> None:
    for (_, module), gain in zip(iter_delta_mem_modules(model), previous):
        module.rwkv_ms_write_address_gain = float(gain)


def local_evidence(
    model: torch.nn.Module,
    batch: Any,
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> Mapping[str, Any]:
    evidence = dict(BASE_LOCAL_EVIDENCE(model, batch, states))
    previous = _set_write_address_gain(model, 0.0)
    try:
        gain_zero_written = write_state(
            model,
            batch.write_input_ids,
            batch.write_attention_mask,
        )
    finally:
        _restore_write_address_gain(model, previous)

    module_names = tuple(name for name, _ in iter_delta_mem_modules(model))
    correct_recurrent, correct_projected = base.base.state_helper.split_state(
        states["correct"],
        module_names,
    )
    gain_zero_recurrent, gain_zero_projected = base.base.state_helper.split_state(
        gain_zero_written,
        module_names,
    )
    gain_zero_state = base.base.state_helper.merge_state(
        gain_zero_recurrent,
        correct_projected,
    )
    candidate = base.base.ACTIVE_CANDIDATE
    correct_logits = base.base.hybrid_screen.read_logits(
        model,
        batch,
        states["correct"],
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=str(candidate["hybrid_mode"]),
        hybrid_gain=float(candidate["hybrid_gain"]),
    )
    gain_zero_logits = base.base.hybrid_screen.read_logits(
        model,
        batch,
        gain_zero_state,
        readout_mode="projected_kv_rwkv_hybrid",
        hybrid_mode=str(candidate["hybrid_mode"]),
        hybrid_gain=float(candidate["hybrid_gain"]),
    )
    comparison = base.base.hybrid_screen.compare_logits(
        correct_logits,
        gain_zero_logits,
    )
    correct_recurrent_sha = base.base.runtime._state_dict_sha256(correct_recurrent)
    gain_zero_recurrent_sha = base.base.runtime._state_dict_sha256(
        gain_zero_recurrent
    )
    projected_fixed = (
        base.base.runtime._state_dict_sha256(correct_projected)
        == base.base.runtime._state_dict_sha256(gain_zero_projected)
    )
    comparisons = dict(evidence["comparisons"])
    comparisons["correct_vs_write_address_gain_zero"] = comparison
    checks = dict(evidence["checks"])
    checks.update(
        {
            "write_address_changes_recurrent_state": (
                correct_recurrent_sha != gain_zero_recurrent_sha
            ),
            "write_address_preserves_projected_carrier": projected_fixed,
            "write_address_ablation_material": (
                comparison["max_abs_logit_delta"]
                >= base.base.MIN_MATERIAL_LOGIT_DELTA
            ),
        }
    )
    evidence.update(
        {
            "checks": checks,
            "passed": all(checks.values()),
            "comparisons": comparisons,
            "write_address_audit": {
                "correct_recurrent_sha256": correct_recurrent_sha,
                "gain_zero_recurrent_sha256": gain_zero_recurrent_sha,
            },
        }
    )
    return evidence


@contextmanager
def screen_bindings() -> Iterator[None]:
    names = {
        "SCHEMA": SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "PRIOR_RESULT": PRIOR_RESULT,
        "PRIOR_RESULT_FILE_SHA256": PRIOR_RESULT_FILE_SHA256,
        "PRIOR_RESULT_RECEIPT": PRIOR_RESULT_RECEIPT,
        "SEED": SEED,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "CANDIDATES": CANDIDATES,
        "RUNNER_BINDING_PATH": RUNNER_BINDING_PATH,
        "validate_protocol": validate_protocol,
        "build_config": build_config,
        "load_model": load_model,
    }
    previous = {name: getattr(base, name) for name in names}
    previous_local_evidence = base.base.local_evidence
    previous_write_state = base.base.hybrid_screen.write_state
    try:
        for name, value in names.items():
            setattr(base, name, value)
        base.base.local_evidence = local_evidence
        base.base.hybrid_screen.write_state = write_state
        yield
    finally:
        base.base.hybrid_screen.write_state = previous_write_state
        base.base.local_evidence = previous_local_evidence
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
