#!/usr/bin/env python3
"""Train only per-layer acceptance biases for frozen recurrent RWKV memory."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_stopgrad_router_causal_train as stopgrad,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_gate_bias_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_gate_bias_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_gate_bias_causal_train_input.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_rwkv_gate_bias_causal_train_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "accba1d0148556606042baa001ee8555794b4f110c8feb7df1d76330fd002290"
)
SEED = 71
LEARNING_RATE = 1e-2
MAX_GRAD_NORM = 0.05
GATE_BIAS_SUFFIX = ".memory_fusion_bias"
FAILED_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_stopgrad_router_causal_train_failed_step2_v1/"
    "input_binding.json"
)
FAILED_INPUT_BINDING_SHA256 = (
    "f14833e823bf0bf36f3ca87ec6ac24f88e6f5ed42a7fed27856cbaa4a362be8d"
)
FAILED_PROGRESS = FAILED_INPUT_BINDING.parent / "training_progress.jsonl"
FAILED_PROGRESS_SHA256 = (
    "32b48fe81ca6bb2d708b9f4202525c6eb6e587113e2929b238406aea2486e086"
)
SELECTED_CANDIDATE = stopgrad.SELECTED_CANDIDATE


def is_gate_bias_parameter(name: str) -> bool:
    return name.endswith(GATE_BIAS_SUFFIX)


def configure_gate_bias_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_names: list[str] = []
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".projected_kv_key_proj"):
            projected_router_frozen.append(name)
        if is_gate_bias_parameter(name):
            parameter.requires_grad_(True)
            selected_names.append(name)
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    ordered_names = [name for name, _ in selected]
    passed = (
        len(selected) == 42
        and ordered_names == sorted(selected_names)
        and all(parameter.numel() == 1 for _, parameter in selected)
        and len(projected_router_frozen) == 42
    )
    audit = {
        "architecture": "recurrent_value_top2_gate_bias_only",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": shared.canonical_sha256(ordered_names),
        "projected_router_frozen_tensors": len(projected_router_frozen),
        "projected_router_frozen_names_sha256": shared.canonical_sha256(
            sorted(projected_router_frozen)
        ),
        "gate_bias_trainable_tensors": len(selected),
        "all_other_adapter_parameters_frozen": True,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Gate-bias trainable isolation failed: {audit!r}")
    return selected, audit


def audit_gate_bias_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    selected = [
        (name, parameter)
        for name, parameter in named_trainable
        if is_gate_bias_parameter(name)
    ]
    rows: list[dict[str, Any]] = []
    for name, parameter in selected:
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        norm = (
            0.0
            if gradient is None or not finite
            else float(gradient.detach().float().norm().item())
        )
        rows.append(
            {
                "name": name,
                "gradient_present": gradient is not None,
                "gradient_finite": finite,
                "gradient_l2_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        )
    passed = (
        len(rows) == 42
        and all(row["gradient_finite"] for row in rows)
        and all(row["gradient_nonzero"] for row in rows)
    )
    return {
        "parameter_family": "memory_fusion_bias",
        "parameter_tensors": len(rows),
        "parameter_names_sha256": shared.canonical_sha256(
            [row["name"] for row in rows]
        ),
        "minimum_l2_norm": min(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "maximum_l2_norm": max(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "all_42_finite_nonzero": passed,
        "layers": rows,
        "passed": passed,
    }


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Gate-bias protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Gate-bias protocol payload differs")
    if shared.sha256_file(FAILED_INPUT_BINDING) != FAILED_INPUT_BINDING_SHA256:
        raise ValueError("Failed stop-gradient input binding differs")
    if shared.sha256_file(FAILED_PROGRESS) != FAILED_PROGRESS_SHA256:
        raise ValueError("Failed stop-gradient progress binding differs")
    architecture = protocol.get("architecture", {})
    training = protocol.get("training", {})
    if (
        architecture.get("trainable_parameter_families")
        != ["memory_fusion_bias"]
        or architecture.get("trainable_parameter_tensors") != 42
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("optimizer_updates") != shared.UPDATES
    ):
        raise ValueError("Gate-bias training contract differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Gate-bias training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    with stopgrad.training_bindings():
        shared_bindings = {
            "SCHEMA": SCHEMA,
            "STEP_SCHEMA": STEP_SCHEMA,
            "INPUT_SCHEMA": INPUT_SCHEMA,
            "PROTOCOL": PROTOCOL,
            "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
            "SEED": SEED,
            "SELECTED_CANDIDATE": SELECTED_CANDIDATE,
            "RUNNER_BINDING_PATH": Path(__file__),
            "PASS_STATUS": "gate_bias_heldout_gate_passed_generation_authorized",
            "FAIL_STATUS": "gate_bias_heldout_gate_failed_generation_blocked",
            "TRAINABLE_CONFIGURER": configure_gate_bias_parameters,
            "REQUIRE_RECURRENT_SUBSET_CHANGED": False,
            "validate_protocol": validate_protocol,
        }
        previous_shared = {
            name: getattr(shared, name) for name in shared_bindings
        }
        previous_learning_rate = causal_train.LEARNING_RATE
        previous_max_grad_norm = causal_train.MAX_GRAD_NORM
        previous_auditor = causal_train.FIRST_UPDATE_GRADIENT_AUDITOR
        try:
            for name, value in shared_bindings.items():
                setattr(shared, name, value)
            causal_train.LEARNING_RATE = LEARNING_RATE
            causal_train.MAX_GRAD_NORM = MAX_GRAD_NORM
            causal_train.FIRST_UPDATE_GRADIENT_AUDITOR = audit_gate_bias_gradients
            yield
        finally:
            causal_train.LEARNING_RATE = previous_learning_rate
            causal_train.MAX_GRAD_NORM = previous_max_grad_norm
            causal_train.FIRST_UPDATE_GRADIENT_AUDITOR = previous_auditor
            for name, value in previous_shared.items():
                setattr(shared, name, value)


def validate_calibration_result() -> Mapping[str, Any]:
    return shared.validate_calibration_result()


def run(**kwargs: Any) -> Mapping[str, Any]:
    with training_bindings():
        return shared.run(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with training_bindings():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
