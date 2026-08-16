from __future__ import annotations

import torch

from deltamem.tests.test_projected_rwkv_hybrid_contract import _module
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_screen as screen,
)


def test_protocol_and_recurrent_failure_binding_validate() -> None:
    protocol = screen.validate_protocol()
    prior = screen.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert prior["status"] == "recurrent_value_native_gain_not_established"
    assert prior["native_recurrent_causal_gain_established"] is False


def test_candidate_grid_orders_soft_before_hard_routing() -> None:
    assert [candidate["candidate_id"] for candidate in screen.CANDIDATES] == [
        "recurrent_value_t4_k0",
        "recurrent_value_t8_k0",
        "recurrent_value_t16_k0",
        "recurrent_value_t1_k1",
    ]


def test_configure_candidate_sets_internal_router_controls() -> None:
    module = _module(hybrid_mode="recurrent_value")
    model = torch.nn.Module()
    model.add_module("attn", module)

    screen.configure_candidate(model, screen.CANDIDATES[1])

    assert module.rwkv_ms_hybrid_mode == "recurrent_value"
    assert module.rwkv_ms_read_temperature == 8.0
    assert module.rwkv_ms_read_top_k == 0
