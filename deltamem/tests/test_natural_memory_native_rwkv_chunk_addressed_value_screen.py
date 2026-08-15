from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_chunk_addressed_value_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_chunk_addressed_value_screen as screen,
)


def test_protocol_and_addressed_near_miss_binding_validate() -> None:
    protocol = screen.validate_protocol()
    prior = screen.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert prior["status"] == "addressed_value_native_gain_not_established"
    assert prior["native_recurrent_causal_gain_established"] is False


def test_build_config_uses_chunk_addressed_value() -> None:
    config = screen.build_config()

    assert config.memory_readout_mode == "projected_kv_rwkv_hybrid"
    assert config.rwkv_ms_hybrid_mode == "chunk_addressed_value"
    assert config.rwkv_ms_hybrid_gain == 0.03125


def test_calibration_protocol_and_screen_binding_validate() -> None:
    protocol = calibration.validate_protocol()
    result = calibration.validate_screen_result()

    assert protocol["receipt"]["payload_sha256"] == calibration.PROTOCOL_PAYLOAD_SHA256
    assert result["receipt"]["payload_sha256"] == calibration.SCREEN_RESULT_RECEIPT


def test_chunk_alignment_evidence_requires_matching_nonzero_slots() -> None:
    module_names = ("layer.0", "layer.1")
    state: dict[str, torch.Tensor] = {}
    for name in module_names:
        recurrent = torch.zeros(1, 1, 4, 2, 2)
        recurrent[:, :, :2] = 1.0
        state[name] = recurrent
        state[f"{name}.__projected_kv_occupied"] = torch.tensor(
            [[True, True, False, False]]
        )
        state[f"{name}.__projected_kv_values"] = torch.zeros(1, 4, 2)

    evidence = screen.chunk_alignment_evidence(state, module_names)

    assert evidence["all_layer_occupied_slots_match"] is True
    assert evidence["at_least_two_aligned_slots_on_every_layer"] is True
    assert evidence["projected_values_exactly_zero_on_every_layer"] is True


def test_chunk_alignment_evidence_rejects_misaligned_slot() -> None:
    recurrent = torch.zeros(1, 1, 4, 2, 2)
    recurrent[:, :, :2] = 1.0
    state = {
        "layer.0": recurrent,
        "layer.0.__projected_kv_occupied": torch.tensor(
            [[True, False, True, False]]
        ),
        "layer.0.__projected_kv_values": torch.zeros(1, 4, 2),
    }

    evidence = screen.chunk_alignment_evidence(state, ("layer.0",))

    assert evidence["all_layer_occupied_slots_match"] is False
