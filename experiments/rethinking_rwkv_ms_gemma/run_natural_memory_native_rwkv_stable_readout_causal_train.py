#!/usr/bin/env python3
"""Train only stable RWKV readout and abstention parameters."""

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
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "rwkv_ms_natural_memory_native_stable_readout_causal_train.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_stable_readout_causal_train_step.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_stable_readout_causal_train_input.v1"
PROTOCOL = (
    SCRIPT_DIR / "natural_memory_native_rwkv_stable_readout_causal_train_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "771ff4a69ddabf843b9ad7e470bbc272388f4934de3a0a42f503e50787679e25"
)
SEED = 69
LEARNING_RATE = 5e-5
MAX_GRAD_NORM = 0.05
FAILED_INPUT_BINDING = (
    SCRIPT_DIR
    / "local_artifacts/"
    "natural_memory_native_rwkv_top2_abstention_causal_train_failed_step2_diagnostic_v1/"
    "input_binding.json"
)
FAILED_INPUT_BINDING_SHA256 = (
    "eaca3f40d2cb28f43ec862cf5fa4afabf970a693666ea094125ab91795822f07"
)
FAILED_PROGRESS = FAILED_INPUT_BINDING.parent / "training_progress.jsonl"
FAILED_PROGRESS_SHA256 = (
    "db0d95ecd9c2b41efc205fe7b002e852f2804c45fad666d3b9e8b474ad6953a5"
)
STABLE_SUFFIXES = (
    ".hrm_rwkv7_core.output.weight",
    ".delta_o_proj",
    ".memory_fusion_hidden_weight",
    ".memory_fusion_read_weight",
    ".memory_fusion_bias",
)


def is_stable_readout_parameter(name: str) -> bool:
    return name.endswith(STABLE_SUFFIXES)


def configure_stable_readout_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected_names: list[str] = []
    family_counts = {suffix: 0 for suffix in STABLE_SUFFIXES}
    projected_router_frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".projected_kv_key_proj"):
            projected_router_frozen.append(name)
        if is_stable_readout_parameter(name):
            parameter.requires_grad_(True)
            selected_names.append(name)
            for suffix in STABLE_SUFFIXES:
                if name.endswith(suffix):
                    family_counts[suffix] += 1
                    break
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
        len(selected) == 210
        and ordered_names == sorted(selected_names)
        and len(projected_router_frozen) == 42
        and all(count == 42 for count in family_counts.values())
    )
    audit = {
        "architecture": "recurrent_value_top2_stable_readout_only",
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": shared.canonical_sha256(ordered_names),
        "projected_router_frozen_tensors": len(projected_router_frozen),
        "projected_router_frozen_names_sha256": shared.canonical_sha256(
            sorted(projected_router_frozen)
        ),
        "recurrent_output_trainable_tensors": family_counts[
            ".hrm_rwkv7_core.output.weight"
        ],
        "family_counts": family_counts,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Stable readout trainable isolation failed: {audit!r}")
    return selected, audit


def validate_protocol() -> Mapping[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Stable readout protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    digest = shared.canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Stable readout protocol payload differs")
    if shared.sha256_file(FAILED_INPUT_BINDING) != FAILED_INPUT_BINDING_SHA256:
        raise ValueError("Failed full-core input binding differs")
    if shared.sha256_file(FAILED_PROGRESS) != FAILED_PROGRESS_SHA256:
        raise ValueError("Failed full-core progress binding differs")
    if protocol.get("protected_splits_opened_by_this_protocol") != []:
        raise ValueError("Stable readout training may not open protected data")
    return protocol


@contextmanager
def training_bindings() -> Iterator[None]:
    shared_bindings = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "SEED": SEED,
        "RUNNER_BINDING_PATH": Path(__file__),
        "PASS_STATUS": "stable_readout_heldout_gate_passed_generation_authorized",
        "FAIL_STATUS": "stable_readout_heldout_gate_failed_generation_blocked",
        "TRAINABLE_CONFIGURER": configure_stable_readout_parameters,
        "validate_protocol": validate_protocol,
    }
    previous_shared = {
        name: getattr(shared, name) for name in shared_bindings
    }
    previous_learning_rate = causal_train.LEARNING_RATE
    previous_max_grad_norm = causal_train.MAX_GRAD_NORM
    try:
        for name, value in shared_bindings.items():
            setattr(shared, name, value)
        causal_train.LEARNING_RATE = LEARNING_RATE
        causal_train.MAX_GRAD_NORM = MAX_GRAD_NORM
        yield
    finally:
        causal_train.LEARNING_RATE = previous_learning_rate
        causal_train.MAX_GRAD_NORM = previous_max_grad_norm
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
