from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_rwkv_addressed_value_eval as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_calibration as calibration,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_causal_train as causal_train,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_eval as addressed_eval,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_addressed_value_screen as screen,
)


def _module_names() -> tuple[str, ...]:
    return tuple(
        f"model.language_model.layers.{layer}.self_attn"
        for layer in range(42)
    )


def _state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer, name in enumerate(_module_names()):
        state[name] = torch.full((1, 1, 4, 2, 2), float(layer + 1))
        state[f"{name}.__rwkv_ms_positions"] = torch.tensor([layer + 1])
        state[f"{name}.__rwkv_ms_previous_source"] = torch.full(
            (1, 2), float(layer + 1)
        )
        state[f"{name}.__projected_kv_keys"] = torch.full(
            (1, 4, 2), float(layer + 1)
        )
        state[f"{name}.__projected_kv_values"] = torch.full(
            (1, 4, 2), float(layer + 1)
        )
        state[f"{name}.__projected_kv_occupied"] = torch.ones(
            1, 4, dtype=torch.bool
        )
        state[f"{name}.__projected_kv_surprise"] = torch.full(
            (1, 4), float(layer + 1)
        )
    return state


def test_protocol_and_prior_failure_binding_validate() -> None:
    protocol = screen.validate_protocol()
    prior = screen.validate_prior_result()

    assert protocol["receipt"]["payload_sha256"] == screen.PROTOCOL_PAYLOAD_SHA256
    assert prior["recurrent_rwkv_causal_attribution_established"] is False


def test_build_config_uses_addressed_value_bottleneck() -> None:
    config = screen.build_config()

    assert config.memory_readout_mode == "projected_kv_rwkv_hybrid"
    assert config.rwkv_ms_hybrid_mode == "addressed_value"
    assert config.rwkv_ms_hybrid_gain == 0.03125


def test_layer_permutation_preserves_projected_carrier() -> None:
    state = _state()
    names = _module_names()

    permuted = screen.permute_recurrent_state(state, names)

    assert set(permuted) == set(state)
    assert torch.equal(
        permuted[f"{names[0]}.__projected_kv_values"],
        state[f"{names[0]}.__projected_kv_values"],
    )
    assert torch.equal(permuted[names[0]], state[names[1]])
    assert torch.equal(permuted[names[-1]], state[names[0]])


def test_projected_value_and_empty_interventions_are_exact() -> None:
    state = _state()

    values_zeroed = screen.zero_projected_values(state)
    empty = screen.empty_state(state)

    assert torch.count_nonzero(
        values_zeroed[f"{_module_names()[0]}.__projected_kv_values"]
    ).item() == 0
    assert torch.equal(values_zeroed[_module_names()[0]], state[_module_names()[0]])
    assert all(torch.count_nonzero(value).item() == 0 for value in empty.values())


def test_candidate_selection_prefers_lowest_passing_gain() -> None:
    selected = screen.select_candidate(
        [
            {
                "candidate_id": "high",
                "hybrid_gain": 0.125,
                "passed": True,
                "worst_rank_correct_vs_empty_max_abs_logit_delta": 1.0,
            },
            {
                "candidate_id": "low",
                "hybrid_gain": 0.03125,
                "passed": True,
                "worst_rank_correct_vs_empty_max_abs_logit_delta": 2.0,
            },
        ]
    )

    assert selected is not None
    assert selected["candidate_id"] == "low"


def test_calibration_protocol_binds_selected_screen_result() -> None:
    protocol = calibration.validate_protocol()
    result = calibration.validate_screen_result()

    assert protocol["receipt"]["payload_sha256"] == calibration.PROTOCOL_PAYLOAD_SHA256
    assert result["receipt"]["payload_sha256"] == calibration.SCREEN_RESULT_RECEIPT


def test_causal_training_protocol_binds_calibration() -> None:
    protocol = causal_train.validate_protocol()
    result = causal_train.validate_calibration_result()

    assert protocol["receipt"]["payload_sha256"] == causal_train.PROTOCOL_PAYLOAD_SHA256
    assert result["receipt"]["payload_sha256"] == causal_train.CALIBRATION_RESULT_RECEIPT


def _reference_bundle(value: float) -> dict[str, torch.Tensor]:
    return {
        attribute: torch.full((1,), value)
        for attribute in (
            *causal_train.RECURRENT_ATTRIBUTES,
            *causal_train.PROJECTED_ATTRIBUTES,
        )
    }


def test_training_intervention_keeps_projected_references() -> None:
    modules = (
        ("layer.0", SimpleNamespace()),
        ("layer.1", SimpleNamespace()),
    )
    projected = {"layer.0": _reference_bundle(1.0), "layer.1": _reference_bundle(2.0)}
    donor = {"layer.0": _reference_bundle(3.0), "layer.1": _reference_bundle(4.0)}

    fixed = causal_train.install_intervened_state(
        modules,
        projected=projected,
        recurrent=donor,
        rotate_recurrent_layers=False,
    )

    assert fixed
    for name, module in modules:
        assert module.projected_kv_keys is projected[name]["projected_kv_keys"]
        assert module.delta_state is donor[name]["delta_state"]


def test_training_layer_permutation_rotates_only_recurrence() -> None:
    modules = (
        ("layer.0", SimpleNamespace()),
        ("layer.1", SimpleNamespace()),
    )
    correct = {"layer.0": _reference_bundle(1.0), "layer.1": _reference_bundle(2.0)}

    fixed = causal_train.install_intervened_state(
        modules,
        projected=correct,
        recurrent=correct,
        rotate_recurrent_layers=True,
    )

    assert fixed
    assert modules[0][1].projected_kv_values is correct["layer.0"]["projected_kv_values"]
    assert modules[0][1].delta_state is correct["layer.1"]["delta_state"]
    assert modules[1][1].delta_state is correct["layer.0"]["delta_state"]


def test_evaluation_training_binding_validates() -> None:
    result_path = (
        causal_train.SCRIPT_DIR
        / "local_artifacts/natural_memory_native_rwkv_addressed_value_causal_train_v1/result.json"
    )
    adapter_dir = result_path.parent / "adapter"

    result = addressed_eval.validate_train_result(
        result_path,
        adapter_dir=adapter_dir,
    )

    assert result["open_native_evaluation_authorized"] is True


def test_evaluation_empty_state_zeros_every_carrier() -> None:
    state = _state()

    empty = addressed_eval.zero_state(state)

    assert all(torch.count_nonzero(value).item() == 0 for value in empty.values())


def test_evaluation_stacks_conditions_in_locked_order() -> None:
    states = {
        condition: {"state": torch.full((1, 2), float(index))}
        for index, condition in enumerate(addressed_eval.CONDITIONS)
    }

    stacked = addressed_eval.stack_condition_states(states)

    assert stacked["state"].shape == (len(addressed_eval.CONDITIONS), 2)
    assert torch.equal(
        stacked["state"][:, 0],
        torch.arange(len(addressed_eval.CONDITIONS), dtype=torch.float32),
    )


def test_analyzer_aggregates_micro_f1() -> None:
    records = {
        1: {"score": {"tp": 2, "fp": 1, "fn": 0, "covered": True}},
        2: {"score": {"tp": 0, "fp": 1, "fn": 2, "covered": False}},
    }

    metrics = analyzer.aggregate_condition(records)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 2
    assert metrics["fn"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["micro_f1"] == 0.5
