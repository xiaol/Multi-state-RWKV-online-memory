from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_recurrent_value_screen as screen,
)


def test_protocol_and_chunk_failure_binding_validate() -> None:
    protocol = screen.validate_protocol()
    prior = screen.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert prior["status"] == "chunk_addressed_value_native_gain_not_established"
    assert prior["native_recurrent_causal_gain_established"] is False


def test_build_config_uses_internal_recurrent_value_router() -> None:
    config = screen.build_config()

    assert config.memory_readout_mode == "projected_kv_rwkv_hybrid"
    assert config.rwkv_ms_hybrid_mode == "recurrent_value"
    assert config.rwkv_ms_hybrid_gain == 0.03125


def test_zero_projected_bundle_preserves_recurrence() -> None:
    state = {
        "layer.0": torch.ones(1, 1, 4, 2, 2),
        "layer.0.__rwkv_ms_positions": torch.ones(1, dtype=torch.long),
        "layer.0.__rwkv_ms_previous_source": torch.ones(1, 2),
        "layer.0.__projected_kv_keys": torch.ones(1, 4, 2),
        "layer.0.__projected_kv_values": torch.ones(1, 4, 2),
        "layer.0.__projected_kv_occupied": torch.ones(1, 4, dtype=torch.bool),
        "layer.0.__projected_kv_surprise": torch.ones(1, 4),
    }

    zeroed = screen.zero_projected_bundle(state)

    assert torch.equal(zeroed["layer.0"], state["layer.0"])
    assert torch.count_nonzero(zeroed["layer.0.__projected_kv_keys"]).item() == 0
    assert torch.count_nonzero(zeroed["layer.0.__projected_kv_values"]).item() == 0
    assert torch.count_nonzero(zeroed["layer.0.__projected_kv_occupied"]).item() == 0
    assert torch.count_nonzero(zeroed["layer.0.__projected_kv_surprise"]).item() == 0
