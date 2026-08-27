#!/usr/bin/env python3
"""Train an identity-bound RWKV adapter in Gemma's native PLE path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import get_delta_mem_state_dict, iter_delta_mem_modules  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_identity_bound_deepembed_causal_train as base,
)


shared = base.identity_train.shared
stable = base.deepembed.stable
binder = base.binder
value_identity = base.value_identity
causal_train = base.causal_train
evolution = base.evolution
deepembed = base.deepembed

PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_ple_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "675fc968ff0ec258b2e2352cd2535a9a34f0c2f6deae086dc35f88b2dd55934c"
SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_ple_causal_train_input.v1"
SEED = 20260831
UPDATES = 8
IDENTITY_MARGIN = 0.2
IDENTITY_WEIGHT = 1.0
HELDOUT_ORDINALS = base.HELDOUT_ORDINALS
HELDOUT_PAYLOAD_SHA256 = base.HELDOUT_PAYLOAD_SHA256
PASS_STATUS = "rwkv_ple_heldout_passed_generation_authorized"
FAIL_STATUS = "rwkv_ple_heldout_failed_generation_blocked"
INSTALL_MODE = "ple_gate"
PLE_RANK = 4
PLE_GAIN = 0.125
DISTRIBUTED_TIMEOUT_SECONDS = 1200
EXPECTED_PLE_TENSORS = 84
EXPECTED_BINDER_TENSORS = 168
EXPECTED_TRAINABLE_TENSORS = EXPECTED_PLE_TENSORS + EXPECTED_BINDER_TENSORS
WARMSTART_ADAPTER = base.WARMSTART_ADAPTER
WARMSTART_RESULT = base.WARMSTART_RESULT
WARMSTART_RESULT_SHA256 = base.WARMSTART_RESULT_SHA256
WARMSTART_RESULT_RECEIPT = base.WARMSTART_RESULT_RECEIPT
WARMSTART_CONFIG_SHA256 = base.WARMSTART_CONFIG_SHA256
CURRENT_MODEL: torch.nn.Module | None = None
BASE_NATIVE_WRITE = evolution._native_write


def _canonical(value: Any) -> str:
    return shared.canonical_sha256(value)


def _sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def _validate_warmstart() -> None:
    if _sha256_file(WARMSTART_RESULT) != WARMSTART_RESULT_SHA256:
        raise ValueError("PLE warm-start result differs")
    result = json.loads(WARMSTART_RESULT.read_text(encoding="utf-8"))
    unsigned = dict(result)
    unsigned.pop("receipt", None)
    if (
        _canonical(unsigned) != WARMSTART_RESULT_RECEIPT
        or result.get("receipt", {}).get("payload_sha256") != WARMSTART_RESULT_RECEIPT
        or result.get("passed") is not True
        or result.get("open_native_generation_authorized") is not True
        or result.get("protected_splits_opened") != []
    ):
        raise ValueError("PLE warm-start is not authorized")
    if _sha256_file(WARMSTART_ADAPTER / "delta_mem_config.json") != WARMSTART_CONFIG_SHA256:
        raise ValueError("PLE warm-start config differs")


def _ple_config() -> Any:
    return replace(
        deepembed.screen.build_config(deepembed.SELECTED_CANDIDATE),
        delta_heads=(),
        rwkv_ms_hybrid_mode="address_keyed_moe_ple",
        rwkv_ms_outer_ffn_gain=0.0,
        rwkv_ms_outer_ffn_layers=(),
        rwkv_ms_ple_rank=PLE_RANK,
        rwkv_ms_ple_gain=PLE_GAIN,
        rwkv_ms_write_address_value_adapter=False,
    )


def configure_runtime(model: torch.nn.Module) -> None:
    candidate = deepembed.SELECTED_CANDIDATE
    for _, module in iter_delta_mem_modules(model):
        module.memory_readout_mode = "projected_kv_rwkv_hybrid"
        module.rwkv_ms_hybrid_mode = "address_keyed_moe_ple"
        module.rwkv_ms_hybrid_gain = float(candidate["hybrid_gain"])
        module.rwkv_ms_write_address_gain = float(candidate["write_address_gain"])
        module.rwkv_ms_read_temperature = float(candidate["read_temperature"])
        module.rwkv_ms_read_top_k = int(candidate["read_top_k"])
        module.rwkv_ms_detach_read_scores = bool(candidate["detach_read_scores"])
        module.rwkv_ms_ple_rank = PLE_RANK
        module.rwkv_ms_ple_gain = PLE_GAIN


def _warmstart_common_parameters(model: torch.nn.Module) -> Mapping[str, Any]:
    source_state = torch.load(
        WARMSTART_ADAPTER / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = get_delta_mem_state_dict(model)
    ple_keys = {
        key
        for key in expected
        if key.endswith(".rwkv_ple_down_weight") or key.endswith(".rwkv_ple_up_weight")
    }
    common = {key: value for key, value in source_state.items() if key in expected}
    missing = sorted(set(expected) - ple_keys - set(common))
    if missing:
        raise ValueError(f"PLE warm-start is missing common tensors: {missing[:8]}")
    extra = sorted(set(source_state) - set(expected))
    unexpected_extra = [
        key
        for key in extra
        if not key.endswith(
            (
                ".rwkv_outer_ffn_down_weight",
                ".rwkv_outer_ffn_gate_weight",
                ".rwkv_outer_ffn_up_weight",
            )
        )
    ]
    if unexpected_extra:
        raise ValueError(f"PLE warm-start has unexpected extra tensors: {unexpected_extra[:8]}")
    for key, value in common.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(
                f"PLE warm-start shape mismatch for {key}: "
                f"source={tuple(value.shape)} target={tuple(expected[key].shape)}"
            )
        module_name, parameter_name = key.rsplit(".", 1)
        parameter = dict(model.named_modules())[module_name]._parameters[parameter_name]
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return {
        "source_result_sha256": WARMSTART_RESULT_SHA256,
        "source_result_receipt": WARMSTART_RESULT_RECEIPT,
        "source_config_sha256": WARMSTART_CONFIG_SHA256,
        "common_parameter_tensors": len(common),
        "ple_parameter_tensors_left_at_identity": len(ple_keys),
        "ignored_deepembed_parameter_tensors": len(extra),
        "ignored_parameter_suffixes": sorted(
            set(key.rsplit(".", 1)[-1] for key in extra)
        ),
    }


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    del candidate
    global CURRENT_MODEL
    _validate_warmstart()
    model, tokenizer, inherited_audit = deepembed.screen.base.base.hybrid_screen.load_model(
        base_model,
        device=device,
        delta_config=_ple_config(),
    )
    configure_runtime(model)
    warmstart_audit = _warmstart_common_parameters(model)
    capture_audit = value_identity.install(model)
    binder_audit = binder.install(model, device=device, mode=INSTALL_MODE)
    modules = tuple(iter_delta_mem_modules(model))
    configured = (
        len(modules) == 42
        and all(
            module.rwkv_ms_hybrid_mode == "address_keyed_moe_ple"
            and module.memory_readout_mode == "projected_kv_rwkv_hybrid"
            and module.rwkv_ms_write_mode == "recurrent"
            and module._ple_input_projection_hook_handle is not None
            and module._deepembed_ffn_pre_hook_handle is None
            and module._deepembed_ffn_down_pre_hook_handle is None
            and module._post_feedforward_norm_hook_handle is None
            and hasattr(module, "rwkv_ple_down_weight")
            and hasattr(module, "rwkv_ple_up_weight")
            for _, module in modules
        )
    )
    if not configured:
        raise RuntimeError("PLE attachment topology is not isolated")
    CURRENT_MODEL = model
    base.CURRENT_MODEL = model
    return model, tokenizer, {
        **dict(inherited_audit),
        "rwkv_ple_attachment": "native_gemma_per_layer_projection_input",
        "rwkv_ple_layers": [module.layer_idx for _, module in modules],
        "rwkv_ple_rank": PLE_RANK,
        "rwkv_ple_gain": PLE_GAIN,
        "rwkv_ple_hooks": len(modules),
        "warmstart": warmstart_audit,
        "projected_value_capture": capture_audit,
        "identity_binder": binder_audit,
        "identity_binding": INSTALL_MODE,
        "configured": configured,
    }


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ple_names: list[str] = []
    binder_names: list[str] = []
    projected_router_names: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith((".rwkv_ple_down_weight", ".rwkv_ple_up_weight")):
            parameter.requires_grad_(True)
            ple_names.append(name)
        elif "rwkv_identity_binder" in name:
            parameter.requires_grad_(True)
            binder_names.append(name)
        elif name.endswith(".projected_kv_key_proj"):
            projected_router_names.append(name)
    stable.runtime._promote_trainable_parameters_to_fp32(model)
    selected = stable.distributed.stable_named_parameters(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    )
    ordered_names = [name for name, _ in selected]
    expected_names = sorted((*ple_names, *binder_names))
    binder_parameter_names = binder.parameter_names(model)
    audit = {
        "architecture": "address_keyed_rwkv_native_ple_identity_gate",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": _canonical(ordered_names),
        "ple_parameter_tensors": len(ple_names),
        "identity_binder_parameter_tensors": len(binder_names),
        "identity_binder_parameter_names": len(binder_parameter_names),
        "projected_router_frozen_tensors": len(projected_router_names),
        "expected_trainable_tensors": EXPECTED_TRAINABLE_TENSORS,
        "warm_start_parameters_frozen": True,
        "passed": bool(
            len(ple_names) == EXPECTED_PLE_TENSORS
            and len(binder_names) == EXPECTED_BINDER_TENSORS
            and len(binder_parameter_names) == EXPECTED_BINDER_TENSORS
            and len(selected) == EXPECTED_TRAINABLE_TENSORS
            and ordered_names == expected_names
            and len(projected_router_names) == 42
        ),
    }
    if audit["passed"] is not True:
        raise RuntimeError(f"PLE trainable isolation failed: {audit!r}")
    return selected, audit


def _binder_score_tensor(captured: Any, labels: torch.Tensor) -> torch.Tensor:
    if CURRENT_MODEL is None:
        raise RuntimeError("PLE identity score requested before model load")
    return binder.score_tensor(captured, labels, CURRENT_MODEL)


def _audit_joint_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    allowed_suffixes = (
        ".rwkv_ple_down_weight",
        ".rwkv_ple_up_weight",
    )
    if not named_trainable or any(
        "rwkv_identity_binder" not in name and not name.endswith(allowed_suffixes)
        for name, _ in named_trainable
    ):
        raise RuntimeError("PLE gradient audit received unexpected parameters")
    finite_values: list[int] = []
    active_values: list[int] = []
    for _, parameter in named_trainable:
        gradient = parameter.grad
        finite = bool(
            gradient is not None
            and gradient.dtype == torch.float32
            and torch.isfinite(gradient).all().item()
        )
        finite_values.append(int(finite))
        active_values.append(int(finite and bool(gradient.abs().gt(0).any().item())))
    device = named_trainable[0][1].device
    finite_tensor = torch.tensor(finite_values, device=device, dtype=torch.int32)
    active_tensor = torch.tensor(active_values, device=device, dtype=torch.int32)
    torch.distributed.all_reduce(finite_tensor)
    torch.distributed.all_reduce(active_tensor)
    return {
        "trainable_tensors": len(named_trainable),
        "global_finite_fp32_tensors": int(finite_tensor.gt(0).sum().item()),
        "global_finite_nonzero_tensors": int(active_tensor.gt(0).sum().item()),
        "passed": bool(
            finite_tensor.gt(0).all().item() and active_tensor.gt(0).all().item()
        ),
    }


def ple_native_write(
    model: torch.nn.Module,
    batch: Any,
    *,
    dtype: torch.dtype,
) -> Mapping[str, Any]:
    configure_runtime(model)
    result = BASE_NATIVE_WRITE(model, batch, dtype=dtype)
    if not all(
        module.rwkv_ms_hybrid_mode == "address_keyed_moe_ple"
        for _, module in iter_delta_mem_modules(model)
    ):
        raise RuntimeError("Native write escaped the locked PLE mode")
    return result


@contextmanager
def _no_deepembed_bindings() -> Iterator[None]:
    yield


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt", None)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    digest = _canonical(unsigned)
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != digest
        or protocol.get("schema") != SCHEMA
        or architecture.get("hybrid_mode") != "address_keyed_moe_ple"
        or architecture.get("native_target") != "per_layer_projection_input"
        or architecture.get("deepembed_hooks") != 0
        or architecture.get("ple_hooks") != 42
        or architecture.get("projected_carrier_fixed") is not True
        or architecture.get("zero_effect_initialization") is not True
        or training.get("optimizer_updates") != UPDATES
        or training.get("distributed_timeout_seconds") != DISTRIBUTED_TIMEOUT_SECONDS
        or training.get("identity_margin") != IDENTITY_MARGIN
        or training.get("joint_trainable_parameter_tensors") != EXPECTED_TRAINABLE_TENSORS
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("RWKV PLE causal protocol differs")
    _validate_warmstart()
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    global CURRENT_MODEL
    previous = {
        "load_model": base.load_model,
        "configure_trainable_parameters": base.configure_trainable_parameters,
        "validate_protocol": base.validate_protocol,
        "PROTOCOL": base.PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": base.PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": base.SCHEMA,
        "STEP_SCHEMA": base.STEP_SCHEMA,
        "INPUT_SCHEMA": base.INPUT_SCHEMA,
        "SEED": base.SEED,
        "UPDATES": base.UPDATES,
        "IDENTITY_MARGIN": base.IDENTITY_MARGIN,
        "IDENTITY_WEIGHT": base.IDENTITY_WEIGHT,
        "HELDOUT_ORDINALS": base.HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": base.HELDOUT_PAYLOAD_SHA256,
        "PASS_STATUS": base.PASS_STATUS,
        "FAIL_STATUS": base.FAIL_STATUS,
        "_binder_score_tensor": base._binder_score_tensor,
        "_audit_identity_gradients": base._audit_identity_gradients,
    }
    previous_deepembed_bindings = deepembed.bindings
    previous_native_write = evolution._native_write
    previous_initialize_distributed_training = (
        shared.distributed.initialize_distributed_training
    )
    previous_score = value_identity.score_tensor
    previous_clear = value_identity.clear
    base.load_model = load_model
    base.configure_trainable_parameters = configure_trainable_parameters
    base.validate_protocol = validate_protocol
    base._binder_score_tensor = _binder_score_tensor
    base._audit_identity_gradients = _audit_joint_gradients
    deepembed.bindings = _no_deepembed_bindings
    evolution._native_write = ple_native_write

    def initialize_with_extended_timeout(device_name: str, **kwargs: Any) -> Any:
        kwargs["timeout_seconds"] = DISTRIBUTED_TIMEOUT_SECONDS
        return previous_initialize_distributed_training(device_name, **kwargs)

    shared.distributed.initialize_distributed_training = initialize_with_extended_timeout
    for name, value in {
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "SEED": SEED,
        "UPDATES": UPDATES,
        "IDENTITY_MARGIN": IDENTITY_MARGIN,
        "IDENTITY_WEIGHT": IDENTITY_WEIGHT,
        "HELDOUT_ORDINALS": HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": HELDOUT_PAYLOAD_SHA256,
        "PASS_STATUS": PASS_STATUS,
        "FAIL_STATUS": FAIL_STATUS,
    }.items():
        setattr(base, name, value)
    value_identity.score_tensor = _binder_score_tensor

    def clear_with_binder(model: torch.nn.Module) -> None:
        previous_clear(model)
        binder.clear_runtime(model)

    value_identity.clear = clear_with_binder
    try:
        with base.training_bindings():
            yield
    finally:
        value_identity.score_tensor = previous_score
        value_identity.clear = previous_clear
        evolution._native_write = previous_native_write
        shared.distributed.initialize_distributed_training = (
            previous_initialize_distributed_training
        )
        deepembed.bindings = previous_deepembed_bindings
        for name, value in previous.items():
            setattr(base, name, value)
        CURRENT_MODEL = None
        base.CURRENT_MODEL = None


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
