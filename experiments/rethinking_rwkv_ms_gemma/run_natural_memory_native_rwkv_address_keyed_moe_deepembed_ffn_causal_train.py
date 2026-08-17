#!/usr/bin/env python3
"""Train the address-keyed sparse DeepEmbed recurrent hybrid on four GPUs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_screen as screen,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_addressed_moe_deepembed_ffn_sparse_causal_train as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_stable_readout_causal_train as stable,
)


PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "f7cdda7a31f48128b56289166a85fed55824d59c89889b0a18e14371b94f0400"
)
SCREEN_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_screen_v2/result.json"
)
SCREEN_RESULT_FILE_SHA256 = (
    "8b9cfcdacd52a38b31a4a739c04e6f92be5267dbf119415adc4e69a4d7744012"
)
SCREEN_RESULT_RECEIPT = (
    "de1a4b8f8c213abfbc8778fe9c66c7f22266527f5747aaafed43ae1b1aa14f4b"
)
CANDIDATE_RESULT = (
    SCREEN_RESULT.parent
    / "address_keyed_deepembed_t16_k2_ag015625_wag025_fg0078125/result.json"
)
CANDIDATE_RESULT_FILE_SHA256 = (
    "e7e3c7d1d38e0260878efa3f9f095a4a4b43681ad33d107983b2a1597630092b"
)
CANDIDATE_RESULT_RECEIPT = (
    "0989150f19958293a3e89344311552b7434e9a92fb9a6bab554d8b4c14a93a5e"
)
SCHEMA = "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train.v1"
STEP_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_step.v1"
)
INPUT_SCHEMA = (
    "rwkv_ms_natural_memory_native_address_keyed_moe_deepembed_ffn_causal_train_input.v1"
)
SEED = 106
UPDATES = 16
CONTRAST_WEIGHT = 1.0
MIN_ACCEPTED_ROWS_PER_UPDATE = 6
MAX_TOTAL_REJECTED_ROWS = 8
ENDPOINT_CANDIDATE_ROWS = 11
TRAINING_PREFIX_SHA256 = (
    "121c2eea4904545b2f8c44ef2e4b16e905ca76809af9c34de5b7f848825872fd"
)
HELDOUT_ORDINALS = (666, 532, 397, 107, 1230, 1077, 1389, 1075, 78, 465, 1251)
HELDOUT_PAYLOAD_SHA256 = (
    "49049954f8c3aebcc94f5fcf4cc72e80f651e156c682d94aa08a21d49ed84b1f"
)
SELECTED_CANDIDATE = screen.CANDIDATES[0]
OUTER_FFN_LAYERS = tuple(int(value) for value in SELECTED_CANDIDATE["outer_ffn_layers"])
PASS_STATUS = "address_keyed_moe_deepembed_ffn_heldout_passed_generation_authorized"
FAIL_STATUS = "address_keyed_moe_deepembed_ffn_heldout_failed_generation_blocked"
EXPECTED_ATTENTION_TENSORS = 378
EXPECTED_OUTER_FFN_TENSORS = 12
EXPECTED_TRAINABLE_TENSORS = EXPECTED_ATTENTION_TENSORS + EXPECTED_OUTER_FFN_TENSORS
BASE_NATIVE_WRITE = evolution._native_write
SHARED_TRAINER = base.base.base.affine_train.shared


def configure_keyed_deepembed_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_names: list[str] = []
    family_counts = {suffix: 0 for suffix in base.base.OUTER_FFN_SUFFIXES}
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".projected_kv_key_proj"):
            projected_router_frozen.append(name)
        for suffix in base.base.OUTER_FFN_SUFFIXES:
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
    for suffix in base.base.OUTER_FFN_SUFFIXES[-3:]:
        attention_counts.pop(suffix)
    passed = (
        len(selected) == EXPECTED_TRAINABLE_TENSORS
        and ordered_names == sorted(selected_names)
        and len(projected_router_frozen) == 42
        and all(count == 42 for count in attention_counts.values())
        and all(
            family_counts[suffix] == len(OUTER_FFN_LAYERS)
            for suffix in base.base.OUTER_FFN_SUFFIXES[-3:]
        )
    )
    audit = {
        "architecture": "address_keyed_moe_deepembed_ffn_sparse_channelmix",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": stable.shared.canonical_sha256(ordered_names),
        "projected_router_frozen_tensors": len(projected_router_frozen),
        "projected_router_frozen_names_sha256": stable.shared.canonical_sha256(
            sorted(projected_router_frozen)
        ),
        "learned_write_parameter_tensors": 0,
        "recurrent_output_trainable_tensors": family_counts[
            ".hrm_rwkv7_core.output.weight"
        ],
        "attention_trainable_tensors": sum(attention_counts.values()),
        "moe_router_trainable_tensors": sum(
            family_counts[suffix]
            for suffix in base.base.moe.MOE_SUFFIXES[len(stable.STABLE_SUFFIXES) :]
        ),
        "outer_ffn_trainable_tensors": sum(
            family_counts[suffix] for suffix in base.base.OUTER_FFN_SUFFIXES[-3:]
        ),
        "outer_ffn_active_layers": list(OUTER_FFN_LAYERS),
        "family_counts": family_counts,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Address-keyed trainable isolation failed: {audit!r}")
    return selected, audit


def configure_keyed_runtime(
    model: torch.nn.Module,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "address_keyed_moe_deepembed_ffn"
        module.rwkv_ms_hybrid_gain = float(candidate["hybrid_gain"])
        module.rwkv_ms_write_address_gain = float(candidate["write_address_gain"])
        module.rwkv_ms_outer_ffn_gain = float(candidate["outer_ffn_gain"])
        module.rwkv_ms_read_temperature = float(candidate["read_temperature"])
        module.rwkv_ms_read_top_k = int(candidate["read_top_k"])
        module.rwkv_ms_detach_read_scores = bool(candidate["detach_read_scores"])


def keyed_native_write(
    model: torch.nn.Module,
    batch: Any,
    *,
    dtype: torch.dtype,
) -> Mapping[str, Any]:
    configure_keyed_runtime(model)
    result = BASE_NATIVE_WRITE(model, batch, dtype=dtype)
    configured = all(
        module.rwkv_ms_hybrid_mode == "address_keyed_moe_deepembed_ffn"
        and module.rwkv_ms_write_address_gain
        == float(SELECTED_CANDIDATE["write_address_gain"])
        for _, module in iter_delta_mem_modules(model)
    )
    if not configured:
        raise RuntimeError("Native write escaped the locked address-keyed mode")
    return result


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] = SELECTED_CANDIDATE,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    model, tokenizer, inherited_audit = screen.base.base.hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=screen.build_config(candidate),
    )
    screen.configure_candidate(model, candidate)
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
        and enabled_layers == OUTER_FFN_LAYERS
        and all(count == len(OUTER_FFN_LAYERS) for count in family_counts.values())
        and all(
            module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_hybrid_mode == "address_keyed_moe_deepembed_ffn"
            and module.rwkv_ms_hybrid_gain == float(candidate["hybrid_gain"])
            and module.rwkv_ms_write_address_gain
            == float(candidate["write_address_gain"])
            and module.rwkv_ms_write_mode == "recurrent"
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
        "all_wrappers_address_keyed_moe_deepembed_ffn_causal": configured,
        "deepembed_ffn_pre_hook_count": deepembed_pre_hooks,
        "deepembed_ffn_down_hook_count": deepembed_down_hooks,
        "deepembed_ffn_family_counts": family_counts,
        "deepembed_ffn_active_layers": list(enabled_layers),
        "write_address_gain": float(candidate["write_address_gain"]),
        "native_write_mode_locked": True,
    }
    if not configured:
        raise RuntimeError(f"Address-keyed causal attachment failed: {audit!r}")
    return model, tokenizer, audit


def validate_signed_result(
    path: Path,
    *,
    file_sha256: str,
    receipt_sha256: str,
) -> Mapping[str, Any]:
    if base.base.base.affine_train.shared.sha256_file(path) != file_sha256:
        raise ValueError(f"Signed screen result differs: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"Signed screen receipt is missing: {path}")
    unsigned = dict(result)
    unsigned.pop("receipt")
    if (
        base.base.base.affine_train.shared.canonical_sha256(unsigned)
        != receipt_sha256
        or receipt.get("payload_sha256") != receipt_sha256
    ):
        raise ValueError(f"Signed screen receipt differs: {path}")
    return result


def validate_screen_result() -> Mapping[str, Any]:
    result = validate_signed_result(
        SCREEN_RESULT,
        file_sha256=SCREEN_RESULT_FILE_SHA256,
        receipt_sha256=SCREEN_RESULT_RECEIPT,
    )
    candidate_result = validate_signed_result(
        CANDIDATE_RESULT,
        file_sha256=CANDIDATE_RESULT_FILE_SHA256,
        receipt_sha256=CANDIDATE_RESULT_RECEIPT,
    )
    if (
        result.get("status") != screen.PASS_STATUS
        or result.get("passed") is not True
        or result.get("training_authorized") is not True
        or result.get("native_generation_authorized") is not False
        or result.get("selected_candidate") != SELECTED_CANDIDATE
        or result.get("selected_candidate_result_receipt")
        != CANDIDATE_RESULT_RECEIPT
        or result.get("protected_splits_opened") != []
        or candidate_result.get("passed") is not True
        or candidate_result.get("selected_candidate") != SELECTED_CANDIDATE
        or candidate_result.get("checks", {}).get("candidate_passed_on_all_ranks")
        is not True
        or candidate_result.get("protected_splits_opened") != []
    ):
        raise ValueError("Address-keyed screen did not authorize causal training")
    return result


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Address-keyed causal protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    authorization = protocol.get("authorization_basis", {})
    architecture = protocol.get("architecture", {})
    digest = base.base.base.affine_train.shared.canonical_sha256(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or authorization.get("screen_protocol_payload_sha256")
        != screen.PROTOCOL_PAYLOAD_SHA256
        or authorization.get("screen_result_file_sha256")
        != SCREEN_RESULT_FILE_SHA256
        or authorization.get("screen_result_receipt") != SCREEN_RESULT_RECEIPT
        or authorization.get("candidate_result_file_sha256")
        != CANDIDATE_RESULT_FILE_SHA256
        or authorization.get("candidate_result_receipt")
        != CANDIDATE_RESULT_RECEIPT
        or authorization.get("selected_candidate") != SELECTED_CANDIDATE
        or architecture.get("hybrid_mode")
        != "address_keyed_moe_deepembed_ffn"
        or architecture.get("write_address_gain")
        != float(SELECTED_CANDIDATE["write_address_gain"])
        or architecture.get("expected_trainable_parameter_tensors")
        != EXPECTED_TRAINABLE_TENSORS
        or architecture.get("learned_write_parameter_tensors") != 0
        or architecture.get("outer_ffn_layers") != list(OUTER_FFN_LAYERS)
        or training.get("optimizer_updates") != UPDATES
        or training.get("contrast_weight_per_active_control") != CONTRAST_WEIGHT
        or training.get("minimum_accepted_rows_per_update")
        != MIN_ACCEPTED_ROWS_PER_UPDATE
        or training.get("maximum_total_rejected_rows")
        != MAX_TOTAL_REJECTED_ROWS
        or endpoint.get("candidate_rows_after_exclusions")
        != ENDPOINT_CANDIDATE_ROWS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256")
        != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Address-keyed causal protocol differs")
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
        "CALIBRATION_SOURCE_DONOR_PAIRS": (),
        "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
        "screen": screen,
        "load_model": load_model,
        "validate_screen_result": validate_screen_result,
        "validate_protocol": validate_protocol,
        "configure_sparse_deepembed_parameters": configure_keyed_deepembed_parameters,
    }
    previous = {name: getattr(base, name) for name in overrides}
    previous_native_write = evolution._native_write
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        evolution._native_write = keyed_native_write
        with base.bindings():
            previous_runner_binding = SHARED_TRAINER.RUNNER_BINDING_PATH
            SHARED_TRAINER.RUNNER_BINDING_PATH = Path(__file__)
            try:
                yield
            finally:
                SHARED_TRAINER.RUNNER_BINDING_PATH = previous_runner_binding
    finally:
        evolution._native_write = previous_native_write
        for name, value in previous.items():
            setattr(base, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    with bindings():
        return base.base.base.affine_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with bindings():
        return base.base.base.affine_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
