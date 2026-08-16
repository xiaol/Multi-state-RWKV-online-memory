from __future__ import annotations

import torch

from deltamem.tests.test_projected_rwkv_hybrid_contract import _module
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_calibration as shared,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_sharp_router_calibration as calibration,
)


def test_protocol_and_sharp_screen_binding_validate() -> None:
    protocol = calibration.validate_protocol()
    screen_result = calibration.validate_screen_result()

    assert (
        protocol["receipt"]["payload_sha256"]
        == calibration.PROTOCOL_PAYLOAD_SHA256
    )
    assert screen_result["status"] == "screen_passed_causal_calibration_authorized"
    assert {
        name: screen_result["selected_candidate"][name]
        for name in calibration.SELECTED_CANDIDATE
    } == calibration.SELECTED_CANDIDATE


def test_calibration_bindings_are_scoped() -> None:
    original_protocol = shared.PROTOCOL

    with calibration.calibration_bindings():
        assert shared.PROTOCOL == calibration.PROTOCOL
        assert shared.screen is calibration.screen

    assert shared.PROTOCOL == original_protocol


def test_selected_candidate_is_active_for_training_forward() -> None:
    module = _module(hybrid_mode="recurrent_value")
    model = torch.nn.Module()
    model.add_module("attn", module)

    with calibration.calibration_bindings():
        shared.configure_selected_candidate(model)

    assert module.rwkv_ms_hybrid_mode == "recurrent_value"
    assert module.rwkv_ms_hybrid_gain == 0.03125
    assert module.rwkv_ms_read_temperature == 1.0
    assert module.rwkv_ms_read_top_k == 1
