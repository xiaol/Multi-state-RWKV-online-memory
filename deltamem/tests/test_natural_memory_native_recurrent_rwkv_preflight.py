from __future__ import annotations

import copy

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_recurrent_rwkv_preflight as preflight,
)


def _state(module_names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer, name in enumerate(module_names):
        state[name] = torch.full((1, 1, 4, 2, 2), float(layer))
        state[f"{name}.__rwkv_ms_positions"] = torch.tensor([layer])
        state[f"{name}.__rwkv_ms_previous_source"] = torch.full(
            (1, 2), float(layer)
        )
    return state


def test_recurrent_protocol_receipt_and_mechanism_are_bound() -> None:
    protocol = preflight.validate_protocol()

    assert protocol["receipt"]["payload_sha256"] == (
        preflight.PROTOCOL_PAYLOAD_SHA256
    )
    assert protocol["paired_architectures"]["recurrent_candidate"] == {
        "candidate_id": "fresh_recurrent_rwkv_ms",
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "delta",
        "rwkv_ms_num_states": 4,
        "rwkv_ms_chunk_size": 128,
        "rwkv_ms_boundary_mode": "fixed_chunk",
        "rwkv_ms_write_mode": "recurrent",
        "rwkv_ms_semantics_version": 2,
        "state_dtype": "float32",
    }
    assert protocol["protected_splits_opened_by_this_protocol"] == []
    assert protocol["runtime_preflight"]["preflight_dtype"] == "float32"
    assert protocol["training"]["bf16_calibration_gate"][
        "required_before_benchmark_training"
    ] is True


def test_layer_permutation_rotates_complete_recurrent_bundles() -> None:
    names = tuple(f"model.layers.{layer}.self_attn" for layer in range(3))
    state = _state(names)

    permuted = preflight.permute_recurrent_state(state, names)

    for layer, name in enumerate(names):
        source_layer = (layer + 1) % len(names)
        assert torch.equal(
            permuted[name],
            state[f"model.layers.{source_layer}.self_attn"],
        )
        assert torch.equal(
            permuted[f"{name}.__rwkv_ms_positions"],
            state[
                f"model.layers.{source_layer}.self_attn.__rwkv_ms_positions"
            ],
        )
        assert torch.equal(
            permuted[f"{name}.__rwkv_ms_previous_source"],
            state[
                f"model.layers.{source_layer}.self_attn.__rwkv_ms_previous_source"
            ],
        )


def test_layer_permutation_rejects_incomplete_state() -> None:
    names = tuple(f"model.layers.{layer}.self_attn" for layer in range(2))
    state = _state(names)
    state.pop(f"{names[0]}.__rwkv_ms_previous_source")

    with pytest.raises(ValueError, match="unexpected keys"):
        preflight.permute_recurrent_state(state, names)


def test_protocol_validation_rejects_receipt_drift(monkeypatch, tmp_path) -> None:
    protocol = copy.deepcopy(preflight.validate_protocol())
    protocol["training"]["seeds"] = [1]
    path = tmp_path / "protocol.json"
    path.write_text(preflight.json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(preflight, "PROTOCOL", path)

    with pytest.raises(ValueError, match="payload hash differs"):
        preflight.validate_protocol()
