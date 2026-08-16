from __future__ import annotations

import math

import torch

from deltamem.tests.test_projected_rwkv_hybrid_contract import _module
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_top2_abstention_screen as screen,
)


def test_protocol_and_failed_calibration_binding_validate() -> None:
    protocol = screen.validate_protocol()
    prior = screen.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert prior["status"] == "calibration_failed_causal_training_blocked"
    assert prior["checks"]["all_local_gradients_finite_fp32"] is False


def test_top2_abstention_candidate_configures_router_and_gate() -> None:
    module = _module(hybrid_mode="recurrent_value")
    module.memory_fusion_mode = "content_gated_add"
    module.register_parameter("memory_fusion_bias", torch.nn.Parameter(torch.zeros(1)))
    model = torch.nn.Module()
    model.add_module("attn", module)

    shared.configure_candidate(model, screen.CANDIDATES[0])

    assert module.rwkv_ms_read_temperature == 16.0
    assert module.rwkv_ms_read_top_k == 2
    assert torch.allclose(
        module.memory_fusion_bias.detach(),
        torch.tensor([math.log(0.25 / 0.75)]),
    )


def test_screen_bindings_are_scoped() -> None:
    original_protocol = shared.PROTOCOL

    with screen.screen_bindings():
        assert shared.PROTOCOL == screen.PROTOCOL
        assert shared.load_model is screen.load_model

    assert shared.PROTOCOL == original_protocol
