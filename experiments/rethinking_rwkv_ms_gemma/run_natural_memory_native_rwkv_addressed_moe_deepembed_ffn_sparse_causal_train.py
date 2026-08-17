#!/usr/bin/env python3
"""Train the sparse DeepEmbed ChannelMix recurrent hybrid on four GPUs."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_outer_ffn_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "a55cba6449be6d6e2e7678aa309b955a8ad00fea329dc01c719ec7d43794b4e8"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_screen_v1/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "079be7f01b2c8e53199c1db4efeda4d66e4428b8fd2dc5ad4f41a1bbf61a3844"
)
SCREEN_RESULT_RECEIPT = (
    "bfdb7ef9d40683a87404243811548764cc4a7bdec8c3993d5409505ace04d275"
)
SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_sparse_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_sparse_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_addressed_moe_deepembed_ffn_sparse_causal_train_input.v1"
)
SEED = 104
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 11
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (545, 765, 582, 863, 1432, 1349, 1436, 29, 586, 456, 596)
HELDOUT_PAYLOAD_SHA256 = (
    "647603d5035441eb28321bc5f167b502f32b1e393b93119dbdb1c8072057e902"
)
CALIBRATION_SOURCE_DONOR_PAIRS = ((718, 1149), (1149, 918), (918, 76), (76, 918))
SELECTED_CANDIDATE = screen.CANDIDATES[0]
PASS_STATUS = "addressed_moe_deepembed_ffn_sparse_heldout_passed_generation_authorized"
FAIL_STATUS = "addressed_moe_deepembed_ffn_sparse_heldout_failed_generation_blocked"
EXPECTED_ATTENTION_TENSORS = 378
EXPECTED_OUTER_FFN_TENSORS = 12
EXPECTED_TRAINABLE_TENSORS = EXPECTED_ATTENTION_TENSORS + EXPECTED_OUTER_FFN_TENSORS


def configure_sparse_deepembed_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_names: list[str] = []
    family_counts = {suffix: 0 for suffix in base.OUTER_FFN_SUFFIXES}
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".projected_kv_key_proj"):
            projected_router_frozen.append(name)
        for suffix in base.OUTER_FFN_SUFFIXES:
            if name.endswith(suffix):
                parameter.requires_grad_(True)
                selected_names.append(name)
                family_counts[suffix] += 1
                break
    stable.runtime._promote_trainable_parameters_to_fp32(model)
    selected = stable.distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    ordered_names = [name for name, _ in selected]
    attention_counts = family_counts.copy()
    for suffix in base.OUTER_FFN_SUFFIXES[-3:]:
        attention_counts.pop(suffix)
    passed = (
        len(selected) == EXPECTED_TRAINABLE_TENSORS
        and ordered_names == sorted(selected_names)
        and len(projected_router_frozen) == 42
        and all(count == 42 for count in attention_counts.values())
        and all(
            family_counts[suffix] == len(screen.OUTER_FFN_LAYERS)
            for suffix in base.OUTER_FFN_SUFFIXES[-3:]
        )
    )
    audit = {
        "architecture": "addressed_moe_deepembed_ffn_sparse_channelmix",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": stable.shared.canonical_sha256(ordered_names),
        "projected_router_frozen_tensors": len(projected_router_frozen),
        "projected_router_frozen_names_sha256": stable.shared.canonical_sha256(
            sorted(projected_router_frozen)
        ),
        "recurrent_output_trainable_tensors": family_counts[
            ".hrm_rwkv7_core.output.weight"
        ],
        "attention_trainable_tensors": sum(attention_counts.values()),
        "moe_router_trainable_tensors": sum(
            family_counts[suffix]
            for suffix in base.moe.MOE_SUFFIXES[len(stable.STABLE_SUFFIXES) :]
        ),
        "outer_ffn_trainable_tensors": sum(
            family_counts[suffix] for suffix in base.OUTER_FFN_SUFFIXES[-3:]
        ),
        "outer_ffn_active_layers": list(screen.OUTER_FFN_LAYERS),
        "family_counts": family_counts,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Sparse DeepEmbed trainable isolation failed: {audit!r}")
    return selected, audit


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = screen.base.hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=screen.build_config(candidate),
    )
    screen.base.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    deepembed_pre_hooks = sum(
        module._deepembed_ffn_pre_hook_handle is not None for _, module in modules
    )
    deepembed_down_hooks = sum(
        module._deepembed_ffn_down_pre_hook_handle is not None for _, module in modules
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
        len(modules) == 42
        and deepembed_pre_hooks == 42
        and deepembed_down_hooks == 42
        and enabled_layers == screen.OUTER_FFN_LAYERS
        and all(count == len(screen.OUTER_FFN_LAYERS) for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "addressed_moe_deepembed_ffn"
            and module.rwkv_ms_hybrid_gain == float(candidate["hybrid_gain"])
            and module.rwkv_ms_outer_ffn_gain == float(candidate["outer_ffn_gain"])
            and module.rwkv_ms_outer_ffn_layers == screen.OUTER_FFN_LAYERS
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
        "all_wrappers_addressed_moe_deepembed_ffn_sparse_causal": configured,
        "deepembed_ffn_pre_hook_count": deepembed_pre_hooks,
        "deepembed_ffn_down_hook_count": deepembed_down_hooks,
        "deepembed_ffn_family_counts": family_counts,
        "deepembed_ffn_active_layers": list(enabled_layers),
    }
    if not configured:
        raise RuntimeError(f"Sparse DeepEmbed causal attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if base.base.affine_train.shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("Sparse DeepEmbed screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Sparse DeepEmbed screen receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        base.base.affine_train.shared.canonical_sha256(unsigned)
        != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != SCREEN_RESULT_RECEIPT
        or result.get("status") != screen.PASS_STATUS
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("native_generation_authorized") is not False
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("Sparse DeepEmbed screen did not authorize causal training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Sparse DeepEmbed causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    authorization = protocol.get("authorization_basis", {})
    architecture = protocol.get("architecture", {})
    digest = base.base.affine_train.shared.canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256") != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or architecture.get("expected_trainable_parameter_tensors")
        != EXPECTED_TRAINABLE_TENSORS
        or architecture.get("outer_ffn_layers") != list(screen.OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows") != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions") != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or endpoint.get("excluded_calibration_source_donor_pairs")
        != [list(pair) for pair in CALIBRATION_SOURCE_DONOR_PAIRS]
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Sparse DeepEmbed causal protocol differs")
    validate_screen_result()
    return protocol


@contextmanager
def bindings() -> Iterator[None]:
    overrides = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCREEN_RESULT": SCREEN_RESULT,
        "SCREEN_RESULT_FILE_SHA256": SCREEN_RESULT_FILE_SHA256,
        "SCREEN_RESULT_RECEIPT": SCREEN_RESULT_RECEIPT,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "CONTRAST_WEIGHT": CONTRAST_WEIGHT,
        "MIN_ACCEPTED_ROWS_PER_UPDATE": MIN_ACCEPTED_ROWS_PER_UPDATE,
        "MAX_TOTAL_REJECTED_ROWS": MAX_TOTAL_REJECTED_ROWS,
        "ENDPOINT_CANDIDATE_ROWS": ENDPOINT_CANDIDATE_ROWS,
        "TRAINING_PREFIX_SHA256": TRAINING_PREFIX_SHA256,
        "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "screen": screen,
        "load_model": load_model,
        "validate_screen_result": validate_screen_result,
        "validate_protocol": validate_protocol,
        "configure_outer_ffn_parameters": configure_sparse_deepembed_parameters,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        with base.bindings():
            yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.affine_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.affine_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
