from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_gate_bias_causal_train as training,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_causal_train as shared,
)


def gate_bias_parameters() -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (
            f"model.language_model.layers.{index}.self_attn.memory_fusion_bias",
            torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32)),
        )
        for index in range(42)
    )


def test_protocol_and_failure_bindings_validate() -> None:
    protocol = training.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == training.PROTOCOL_PAYLOAD_SHA256
    assert protocol["architecture"]["trainable_parameter_families"] == [
        "memory_fusion_bias"
    ]


def test_gate_bias_parameter_family_is_exact() -> None:
    assert training.is_gate_bias_parameter(
        "model.language_model.layers.0.self_attn.memory_fusion_bias"
    )
    assert not training.is_gate_bias_parameter(
        "model.language_model.layers.0.self_attn.memory_fusion_read_weight"
    )
    assert not training.is_gate_bias_parameter(
        "model.language_model.layers.0.self_attn.hrm_rwkv7_core.output.weight"
    )


def test_gate_bias_gradient_audit_requires_every_layer_finite_nonzero() -> None:
    named = gate_bias_parameters()
    for _, parameter in named:
        parameter.grad = torch.ones_like(parameter)

    passing = training.audit_gate_bias_gradients(named)
    assert passing["parameter_tensors"] == 42
    assert passing["all_42_finite_nonzero"] is True
    assert passing["passed"] is True

    named[-1][1].grad = torch.zeros_like(named[-1][1])
    failing = training.audit_gate_bias_gradients(named)
    assert failing["all_42_finite_nonzero"] is False
    assert failing["passed"] is False


def test_training_bindings_select_gate_only_contract_and_restore() -> None:
    original_configurer = shared.TRAINABLE_CONFIGURER
    original_requirement = shared.REQUIRE_RECURRENT_SUBSET_CHANGED
    original_auditor = causal_train.FIRST_UPDATE_GRADIENT_AUDITOR

    with training.training_bindings():
        assert shared.TRAINABLE_CONFIGURER is training.configure_gate_bias_parameters
        assert shared.REQUIRE_RECURRENT_SUBSET_CHANGED is False
        assert causal_train.FIRST_UPDATE_GRADIENT_AUDITOR is training.audit_gate_bias_gradients
        assert causal_train.LEARNING_RATE == training.LEARNING_RATE

    assert shared.TRAINABLE_CONFIGURER is original_configurer
    assert shared.REQUIRE_RECURRENT_SUBSET_CHANGED is original_requirement
    assert causal_train.FIRST_UPDATE_GRADIENT_AUDITOR is original_auditor
