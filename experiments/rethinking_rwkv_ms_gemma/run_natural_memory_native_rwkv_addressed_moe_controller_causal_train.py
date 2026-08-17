#!/usr/bin/env python3
"""Train the bounded addressed/global RWKV mixture-of-experts controller."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch

from deltamem.core.delta import iter_delta_mem_modules
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_route_agreement_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_moe_controller_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as candidate_helper,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_addressed_moe_controller_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "e186599406c5f7916070a0cc1a08db11ee127c0aabbae74e971107a2d72367e5"
SCREEN_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_addressed_moe_controller_screen_v4/result.json"
SCREEN_RESULT_FILE_SHA256 = "35bc8ce19fefed7d1accf2a682e02ebb7722635ce36f0585770455beb4c71c5a"
SCREEN_RESULT_RECEIPT = "04ea2779d83e0f8048786eae50761acf17f8135c01f924bb3f563c4dd0a3888a"
SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_controller_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_controller_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_addressed_moe_controller_causal_train_input.v1"
SEED = 91
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 777
TRAINING_PREFIX_SHA256 = "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
HELDOUT_ORDINALS = (
    1143, 174, 1208, 441, 180, 79, 1365, 919,
    1028, 1042, 81, 391, 310, 1394, 215, 1241,
    950, 1127, 121, 364, 339, 370, 1330, 332,
    1010, 1079, 1391, 832, 1423, 84, 1187, 368,
)
HELDOUT_PAYLOAD_SHA256 = "9622cb94f26f42fb80a86cfff3d8688a817832cf95e28ff3ca10f13ca882a4d2"
SELECTED_CANDIDATE = screen.SELECTED_CANDIDATE
PASS_STATUS = "addressed_moe_controller_heldout_passed_generation_authorized"
FAIL_STATUS = "addressed_moe_controller_heldout_failed_generation_blocked"
MOE_SUFFIXES = stable.STABLE_SUFFIXES + (
    ".rwkv_moe_hidden_weight",
    ".rwkv_moe_addressed_weight",
    ".rwkv_moe_global_weight",
    ".rwkv_moe_bias",
)


def configure_moe_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_names: list[str] = []
    family_counts = {suffix: 0 for suffix in MOE_SUFFIXES}
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".projected_kv_key_proj"):
            projected_router_frozen.append(name)
        for suffix in MOE_SUFFIXES:
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
    passed = (
        len(selected) == 378
        and ordered_names == sorted(selected_names)
        and len(projected_router_frozen) == 42
        and all(count == 42 for count in family_counts.values())
    )
    audit = {
        "architecture": "addressed_moe_controller_trainable_router",
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
        "moe_router_trainable_tensors": sum(
            family_counts[suffix] for suffix in MOE_SUFFIXES[len(stable.STABLE_SUFFIXES) :]
        ),
        "family_counts": family_counts,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"MoE trainable isolation failed: {audit!r}")
    return selected, audit


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = screen.load_model(base_model, device=device)
    candidate_helper.configure_candidate(model, candidate)
    modules = tuple(iter_delta_mem_modules(model))
    configured = all(
        module.rwkv_ms_hybrid_mode == "addressed_moe_controller"
        and module.rwkv_ms_hybrid_gain == 0.03125
        and module.rwkv_ms_read_temperature == 16.0
        and module.rwkv_ms_read_top_k == 2
        and module.rwkv_ms_detach_read_scores is True
        and module.memory_fusion_mode == "content_gated_add"
        and hasattr(module, "rwkv_moe_bias")
        for _, module in modules
    )
    audit = {**dict(inherited_audit), "all_wrappers_addressed_moe_controller": configured}
    if not configured:
        raise RuntimeError(f"MoE causal attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_screen_result() -> Mapping[str, Any]:
    if base.affine_train.shared.sha256_file(SCREEN_RESULT) != SCREEN_RESULT_FILE_SHA256:
        raise ValueError("MoE screen result file differs")
    result = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("MoE screen receipt is missing")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        base.affine_train.shared.canonical_sha256(unsigned) != SCREEN_RESULT_RECEIPT
        or receipt.get("payload_sha256") != SCREEN_RESULT_RECEIPT
        or result.get("status") != screen.PASS_STATUS
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("native_generation_authorized") is not False
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("MoE screen did not authorize training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("MoE causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    authorization = protocol.get("authorization_basis", {})
    digest = base.affine_train.shared.canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256") != screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256") != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update") != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows") != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions") != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("MoE causal protocol differs")
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
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        with base.training_bindings():
            previous_shared_screen = base.affine_train.shared.screen
            previous_trainable_configurer = (
                base.affine_train.shared.TRAINABLE_CONFIGURER
            )
            base.affine_train.shared.screen = screen
            base.affine_train.shared.TRAINABLE_CONFIGURER = configure_moe_parameters
            try:
                yield
            finally:
                base.affine_train.shared.screen = previous_shared_screen
                base.affine_train.shared.TRAINABLE_CONFIGURER = (
                    previous_trainable_configurer
                )
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.affine_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.affine_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
