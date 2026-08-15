from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_projected_rwkv_hybrid_screen as screen,
)


def _state(matrix_value: float, projected_value: float) -> dict[str, torch.Tensor]:
    name = "model.layers.0.self_attn"
    return {
        name: torch.full((1, 1, 2, 2, 2), matrix_value),
        f"{name}.__rwkv_ms_positions": torch.tensor([int(matrix_value)]),
        f"{name}.__rwkv_ms_previous_source": torch.full((1, 2), matrix_value),
        f"{name}.__projected_kv_keys": torch.full((1, 2, 2), projected_value),
        f"{name}.__projected_kv_values": torch.full((1, 2, 2), projected_value),
        f"{name}.__projected_kv_occupied": torch.ones(1, 2, dtype=torch.bool),
        f"{name}.__projected_kv_surprise": torch.full((1, 2), projected_value),
    }


def test_hybrid_screen_protocol_and_runtime_config_are_bound() -> None:
    protocol = screen.validate_protocol()
    config = screen.build_config()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert protocol["candidate_grid"] == list(screen.CANDIDATES)
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert config.memory_readout_mode == "projected_kv_rwkv_hybrid"
    assert config.memory_backend == "rwkv_ms"
    assert config.projected_kv_key_dim == 64
    assert config.rwkv_ms_write_mode == "recurrent"


def test_recurrent_interventions_hold_projected_carrier_fixed() -> None:
    correct = _state(1.0, 3.0)
    donor = _state(2.0, 9.0)

    correct_combined = screen.combine_state(correct, correct)
    donor_combined = screen.combine_state(correct, donor)
    zero_combined = screen.combine_state(correct, None)

    assert screen.runtime._state_dict_sha256(
        screen.projected_state(correct_combined)
    ) == screen.runtime._state_dict_sha256(screen.projected_state(donor_combined))
    assert screen.runtime._state_dict_sha256(
        screen.projected_state(correct_combined)
    ) == screen.runtime._state_dict_sha256(screen.projected_state(zero_combined))
    assert torch.equal(
        donor_combined["model.layers.0.self_attn"],
        donor["model.layers.0.self_attn"],
    )
    assert torch.count_nonzero(zero_combined["model.layers.0.self_attn"]).item() == 0
    assert torch.count_nonzero(
        zero_combined["model.layers.0.self_attn.__rwkv_ms_previous_source"]
    ).item() == 0


def test_selection_prefers_lowest_gain_then_smallest_bounded_effect() -> None:
    rows = [
        {
            "candidate_id": "large",
            "hybrid_gain": 0.125,
            "worst_rank_correct_vs_projected_max_abs_logit_delta": 0.01,
            "passed": True,
        },
        {
            "candidate_id": "small_b",
            "hybrid_gain": 0.03125,
            "worst_rank_correct_vs_projected_max_abs_logit_delta": 0.2,
            "passed": True,
        },
        {
            "candidate_id": "small_a",
            "hybrid_gain": 0.03125,
            "worst_rank_correct_vs_projected_max_abs_logit_delta": 0.1,
            "passed": True,
        },
    ]

    selected = screen.select_candidate(rows)

    assert selected is rows[2]
