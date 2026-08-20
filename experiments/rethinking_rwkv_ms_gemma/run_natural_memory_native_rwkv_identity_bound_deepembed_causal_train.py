#!/usr/bin/env python3
"""Train a learned projected-value/RWKV identity gate on the DeepEmbed route."""

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

from deltamem.core.delta import load_delta_mem_adapter  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    identity_bound_deepembed as binder,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train as deepembed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_rwkv_projected_value_identity_causal_train as identity_train,
)


shared = deepembed.base.base.base.affine_train.shared
causal_train = deepembed.base.base.base.affine_train.causal_train
contrast = shared.contrast
distributed = shared.distributed
evolution = shared.evolution
value_identity = identity_train.value_identity
IDENTITY_METRICS = identity_train.base._identity_metrics

PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_identity_bound_deepembed_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "e52bef493088d48ec71744c1e5bfb4c569df865cf7c6de9f6572418a133432df"
TRAINING_PREFIX_SHA256 = "108b83ce2f2dd590c6ad45c7d46affeb4fd01afddee08dce35b2d5d18219876d"
SCHEMA = "rwkv_ms_natural_memory_native_identity_bound_deepembed_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_identity_bound_deepembed_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_identity_bound_deepembed_causal_train_input.v1"
SEED = 117
UPDATES = 8
IDENTITY_MARGIN = 0.2
IDENTITY_WEIGHT = 1.0
HELDOUT_ORDINALS = (
    1002, 1431, 1161, 1128, 1189, 1232, 1437, 1220,
    718, 805, 1331, 546, 472, 973, 101, 1154,
)
HELDOUT_PAYLOAD_SHA256 = "94a7ce9e5c1ee6d649daa3d377e5433956ac891fc1ec13625f7e89b34f06d75b"
PASS_STATUS = "identity_bound_deepembed_heldout_passed_generation_authorized"
FAIL_STATUS = "identity_bound_deepembed_heldout_failed_generation_blocked"
WARMSTART_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1/adapter"
WARMSTART_RESULT = SCRIPT_DIR / "local_artifacts/natural_memory_native_rwkv_address_keyed_moe_deepembed_ffn_causal_train_v5_r1/result.json"
WARMSTART_RESULT_SHA256 = "95376ed78da98cf36183146ce56a3623988e94645723c9b34aee0510e0457545"
WARMSTART_RESULT_RECEIPT = "7afee3fd1d88c7db91c86dd3f7febfd80656a35d54971fd824623a29883dba8e"
WARMSTART_CONFIG_SHA256 = "69d18784bb400fb51f38d8e073ed6acb83be54428bfd18644d9e4b833933be44"
CURRENT_MODEL: torch.nn.Module | None = None


def _accumulate_cpu_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: dict[str, torch.Tensor],
) -> Mapping[str, Any]:
    validation = distributed.validate_local_gradients(named_trainable)
    if validation["non_fp32_gradient_tensors"]:
        raise RuntimeError(f"Identity-bound row gradients are not FP32: {validation!r}")
    if validation["active_gradient_tensors"] == 0:
        raise RuntimeError(f"Identity-bound row has no active gradients: {validation!r}")
    if validation["nonfinite_gradient_tensors"]:
        return validation
    with torch.no_grad():
        for name, parameter in named_trainable:
            if parameter.grad is None:
                continue
            contribution = parameter.grad.detach().float().cpu()
            if name not in clean_gradients:
                clean_gradients[name] = contribution.clone()
            else:
                clean_gradients[name].add_(contribution)
    return validation


def _materialize_cpu_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    clean_gradients: Mapping[str, torch.Tensor],
    *,
    scale: float,
) -> None:
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
        raise ValueError("Identity-bound gradient scale must be finite and positive")
    with torch.no_grad():
        for name, parameter in named_trainable:
            gradient = clean_gradients.get(name)
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(device=parameter.device, dtype=parameter.dtype).mul(scale)
            )


def _audit_identity_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    if not named_trainable or any("rwkv_identity_binder" not in name for name, _ in named_trainable):
        raise RuntimeError("Identity-bound gradient audit received unexpected parameters")
    finite_values = []
    active_values = []
    for _, parameter in named_trainable:
        gradient = parameter.grad
        finite = bool(
            gradient is not None
            and gradient.dtype == torch.float32
            and torch.isfinite(gradient).all().item()
        )
        finite_values.append(int(finite))
        active_values.append(int(finite and bool(gradient.abs().gt(0).any().item())))
    finite_tensor = torch.tensor(finite_values, device=named_trainable[0][1].device, dtype=torch.int32)
    active_tensor = torch.tensor(active_values, device=named_trainable[0][1].device, dtype=torch.int32)
    torch.distributed.all_reduce(finite_tensor)
    torch.distributed.all_reduce(active_tensor)
    return {
        "trainable_tensors": len(named_trainable),
        "global_finite_fp32_tensors": int(finite_tensor.gt(0).sum().item()),
        "global_finite_nonzero_tensors": int(active_tensor.gt(0).sum().item()),
        "passed": bool(finite_tensor.gt(0).all().item() and active_tensor.gt(0).all().item()),
    }


def _canonical(value: Any) -> str:
    return distributed.canonical_sha256(value)


def _sha256_file(path: Path) -> str:
    return deepembed.base.base.base.affine_train.shared.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt", {})
    unsigned = dict(protocol)
    unsigned.pop("receipt", None)
    digest = _canonical(unsigned)
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    frozen_inputs = protocol.get("frozen_inputs", {})
    endpoint = protocol.get("heldout_causal_endpoint", {})
    if (
        digest != PROTOCOL_PAYLOAD_SHA256
        or receipt.get("payload_sha256") != digest
        or architecture.get("hybrid_mode") != "address_keyed_moe_deepembed_ffn"
        or architecture.get("identity_binder") != "learned_projected_value_rwkv_pair_score"
        or architecture.get("forward_output_changed") is not True
        or training.get("optimizer_updates") != UPDATES
        or training.get("identity_margin") != IDENTITY_MARGIN
        or training.get("identity_weight") != IDENTITY_WEIGHT
        or frozen_inputs.get("eight_update_schedule_prefix_sha256") != TRAINING_PREFIX_SHA256
        or endpoint.get("source_ordinals") != list(HELDOUT_ORDINALS)
        or endpoint.get("source_donor_payload_sha256") != HELDOUT_PAYLOAD_SHA256
        or protocol.get("protected_splits_opened_by_this_protocol") != []
    ):
        raise ValueError("Identity-bound DeepEmbed protocol differs")
    if _sha256_file(WARMSTART_RESULT) != WARMSTART_RESULT_SHA256:
        raise ValueError("Identity-bound DeepEmbed warm-start result differs")
    warmstart = json.loads(WARMSTART_RESULT.read_text(encoding="utf-8"))
    unsigned_warmstart = dict(warmstart)
    unsigned_warmstart.pop("receipt", None)
    if (
        _canonical(unsigned_warmstart) != WARMSTART_RESULT_RECEIPT
        or warmstart.get("receipt", {}).get("payload_sha256") != WARMSTART_RESULT_RECEIPT
        or warmstart.get("passed") is not True
        or warmstart.get("open_native_generation_authorized") is not True
        or warmstart.get("protected_splits_opened") != []
    ):
        raise ValueError("Identity-bound DeepEmbed warm-start is not authorized")
    if _sha256_file(WARMSTART_ADAPTER / "delta_mem_config.json") != WARMSTART_CONFIG_SHA256:
        raise ValueError("Identity-bound DeepEmbed warm-start config differs")
    return protocol


def _tracked_build_donor_batch(*args: Any, **kwargs: Any) -> Any:
    donor_batch = identity_train._ORIGINAL_BUILD_DONOR_BATCH(*args, **kwargs)
    deepembed._pending_donor_batch = donor_batch
    return donor_batch


def load_model(
    base_model: Path,
    *,
    device: torch.device,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    global CURRENT_MODEL
    model, tokenizer, inherited_audit = deepembed.load_model(base_model, device=device)
    load_delta_mem_adapter(model, WARMSTART_ADAPTER)
    capture_audit = value_identity.install(model)
    binder_audit = binder.install(model, device=device)
    CURRENT_MODEL = model
    return model, tokenizer, {
        **dict(inherited_audit),
        "warm_start_result_sha256": WARMSTART_RESULT_SHA256,
        "warm_start_result_receipt": WARMSTART_RESULT_RECEIPT,
        "warm_start_config_sha256": WARMSTART_CONFIG_SHA256,
        "projected_value_capture": capture_audit,
        "identity_binder": binder_audit,
    }


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    selected_base, inherited_audit = deepembed.configure_keyed_deepembed_parameters(model)
    for name, parameter in selected_base:
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "rwkv_identity_binder" in name:
            parameter.requires_grad_(True)
    deepembed.stable.runtime._promote_trainable_parameters_to_fp32(model)
    selected = deepembed.stable.distributed.stable_named_parameters(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    )
    binder_names = binder.parameter_names(model)
    audit = {
        **dict(inherited_audit),
        "architecture": "identity_bound_address_keyed_deepembed_ffn",
        "warm_start_trainable_parameters": len(selected_base),
        "identity_binder_parameter_tensors": len(binder_names),
        "identity_binder_parameter_elements": sum(
            parameter.numel()
            for name, parameter in selected
            if "rwkv_identity_binder" in name
        ),
        "parameter_tensors": len(selected),
        "parameter_names_sha256": _canonical([name for name, _ in selected]),
        "warm_start_parameters_frozen_during_identity": True,
        "passed": bool(binder_names and len(selected) > 0),
    }
    if audit["passed"] is not True:
        raise RuntimeError(f"Identity-bound trainable isolation failed: {audit!r}")
    return selected, audit


def _binder_score_tensor(captured: Any, labels: torch.Tensor) -> torch.Tensor:
    if CURRENT_MODEL is None:
        raise RuntimeError("Identity-bound score requested before model load")
    return binder.score_tensor(captured, labels, CURRENT_MODEL)


@contextmanager
def _deepembed_training_bindings() -> Iterator[None]:
    with deepembed.bindings():
        previous_prefix = shared.TRAINING_PREFIX_SHA256
        previous_offload = causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS
        previous_serialization = causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS
        previous_filter = causal_train.FILTER_NONFINITE_ROWS
        previous_accumulate = causal_train.accumulate_finite_row_gradients
        previous_materialize = causal_train.materialize_clean_gradients
        previous_auditor = causal_train.FIRST_UPDATE_GRADIENT_AUDITOR
        shared.TRAINING_PREFIX_SHA256 = TRAINING_PREFIX_SHA256
        causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = True
        causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = True
        causal_train.FILTER_NONFINITE_ROWS = True
        causal_train.accumulate_finite_row_gradients = _accumulate_cpu_gradients
        causal_train.materialize_clean_gradients = _materialize_cpu_gradients
        causal_train.FIRST_UPDATE_GRADIENT_AUDITOR = _audit_identity_gradients
        try:
            yield
        finally:
            causal_train.materialize_clean_gradients = previous_materialize
            causal_train.accumulate_finite_row_gradients = previous_accumulate
            causal_train.FIRST_UPDATE_GRADIENT_AUDITOR = previous_auditor
            causal_train.FILTER_NONFINITE_ROWS = previous_filter
            causal_train.SERIALIZE_CONTROL_BRANCH_GRAPHS = previous_serialization
            causal_train.OFFLOAD_OPTIMIZER_STATE_DURING_ROWS = previous_offload
            shared.TRAINING_PREFIX_SHA256 = previous_prefix


def _prepare_compatibility() -> None:
    deepembed._pending_donor_batch = None
    deepembed._pending_identity = None
    deepembed._identity_metrics = IDENTITY_METRICS
    deepembed._ORIGINAL_CHECKPOINTED_WRITE_READ = evolution.checkpointed_native_write_read
    deepembed._ORIGINAL_BACKWARD_LOGITS = causal_train.backward_logits
    deepembed._ORIGINAL_TRAIN = causal_train.train
    deepembed._ORIGINAL_BUILD_DONOR_BATCH = contrast.build_donor_batch
    deepembed._ORIGINAL_ENDPOINT = shared.evaluate_heldout_causal_endpoint
    deepembed._tracked_build_donor_batch = _tracked_build_donor_batch
    deepembed.training_bindings = _deepembed_training_bindings
    identity_train.base = deepembed
    identity_train.shared = shared
    identity_train.causal_train = causal_train
    identity_train.contrast = contrast
    identity_train.distributed = distributed
    identity_train._ORIGINAL_CHECKPOINTED_WRITE_READ = evolution.checkpointed_native_write_read
    identity_train._ORIGINAL_BACKWARD_LOGITS = causal_train.backward_logits
    identity_train._ORIGINAL_TRAIN = causal_train.train
    identity_train._ORIGINAL_BUILD_DONOR_BATCH = contrast.build_donor_batch
    identity_train._ORIGINAL_ENDPOINT = shared.evaluate_heldout_causal_endpoint
    identity_train.load_model = load_model
    identity_train.configure_trainable_parameters = configure_trainable_parameters
    identity_train.validate_protocol = validate_protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    global CURRENT_MODEL
    _prepare_compatibility()
    previous_score = value_identity.score_tensor
    previous_clear = value_identity.clear
    value_identity.score_tensor = _binder_score_tensor
    def clear_with_binder(model: torch.nn.Module) -> None:
        previous_clear(model)
        binder.clear_runtime(model)
    value_identity.clear = clear_with_binder
    previous_constants = {
        "PROTOCOL": identity_train.PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": identity_train.PROTOCOL_PAYLOAD_SHA256,
        "SCHEMA": identity_train.SCHEMA,
        "STEP_SCHEMA": identity_train.STEP_SCHEMA,
        "INPUT_SCHEMA": identity_train.INPUT_SCHEMA,
        "SEED": identity_train.SEED,
        "UPDATES": identity_train.UPDATES,
        "IDENTITY_MARGIN": identity_train.IDENTITY_MARGIN,
        "IDENTITY_WEIGHT": identity_train.IDENTITY_WEIGHT,
        "HELDOUT_ORDINALS": identity_train.HELDOUT_ORDINALS,
        "HELDOUT_PAYLOAD_SHA256": identity_train.HELDOUT_PAYLOAD_SHA256,
        "PASS_STATUS": identity_train.PASS_STATUS,
        "FAIL_STATUS": identity_train.FAIL_STATUS,
    }
    try:
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
            setattr(identity_train, name, value)
        with identity_train.training_bindings():
            yield
    finally:
        value_identity.score_tensor = previous_score
        value_identity.clear = previous_clear
        for name, value in previous_constants.items():
            setattr(identity_train, name, value)
        CURRENT_MODEL = None


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return identity_train.shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return identity_train.shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
