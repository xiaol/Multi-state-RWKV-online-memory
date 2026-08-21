from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_continuous_write_mechanics as mechanics,
)


def _state(value: float) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for index in range(mechanics.MODULES):
        name = f"layer.{index}"
        result[name] = {
            "delta_state": torch.full((1, 2, 2), value + index),
            "rwkv_ms_positions": torch.tensor([[index]], dtype=torch.long),
            "rwkv_ms_previous_source": torch.full((1, 2), value),
        }
    return result


def test_norm_random_address_is_deterministic_and_norm_matched() -> None:
    address = torch.arange(
        1,
        mechanics.MODULES * mechanics.ADDRESS_DIM + 1,
        dtype=torch.float32,
    ).reshape(mechanics.MODULES, mechanics.ADDRESS_DIM)

    first = mechanics.norm_random_address(address, seed=17)
    second = mechanics.norm_random_address(address, seed=17)

    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), address.norm(dim=-1), atol=1e-4)
    assert not torch.equal(first, address)


def test_norm_random_recurrent_preserves_each_module_norm_and_metadata() -> None:
    state = _state(2.0)

    randomized = mechanics.norm_random_recurrent(state, seed=23)

    for name in state:
        assert torch.allclose(
            randomized[name]["delta_state"].float().norm(),
            state[name]["delta_state"].float().norm(),
        )
        assert torch.equal(
            randomized[name]["rwkv_ms_positions"],
            state[name]["rwkv_ms_positions"],
        )
        assert torch.equal(
            randomized[name]["rwkv_ms_previous_source"],
            state[name]["rwkv_ms_previous_source"],
        )


def test_layer_roll_recurrent_uses_previous_layer_tuple() -> None:
    state = _state(3.0)
    names = tuple(state)

    rolled = mechanics.layer_roll_recurrent(state, names)

    for index, name in enumerate(names):
        source = names[(index - 1) % len(names)]
        for attribute in mechanics.RECURRENT_ATTRIBUTES:
            assert torch.equal(rolled[name][attribute], state[source][attribute])
            assert rolled[name][attribute] is not state[source][attribute]


def test_effective_override_requires_byte_exact_consumption() -> None:
    requested = torch.arange(
        mechanics.MODULES * mechanics.ADDRESS_DIM, dtype=torch.float32
    ).reshape(mechanics.MODULES, mechanics.ADDRESS_DIM)

    assert mechanics._require_effective_address_match(requested.clone(), requested)
    changed = requested.clone()
    changed[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="effective address differs"):
        mechanics._require_effective_address_match(changed, requested)


def test_normalized_state_distance_uses_symmetric_bounded_denominator() -> None:
    reference = _state(0.0)
    candidate = _state(0.0)
    for name in reference:
        reference[name]["delta_state"].zero_()
        candidate[name]["delta_state"].fill_(1.0)

    distance = mechanics.normalized_delta_state_l2(
        reference, candidate, tuple(reference)
    )

    assert distance == pytest.approx(1.0)


def test_unconditional_read_basis_observer_counts_every_invocation() -> None:
    module = SimpleNamespace(
        rwkv_continuous_mechanics_read_basis_invocations=0,
        rwkv_continuous_mechanics_original_read_basis=lambda state, source, mask: (
            state,
            source,
            state,
        ),
    )
    state = torch.ones(1)
    source = torch.ones(1)

    mechanics._observed_mechanics_read_basis(module, state, source, None)

    assert module.rwkv_continuous_mechanics_read_basis_invocations == 1


def _analysis_row(source_index: int, distance: float) -> dict[str, object]:
    integrity = {
        "natural_target_replay_exact": True,
        "zero_address_logits_exact_raw": True,
    }
    return {
        "source_index": source_index,
        "state_normalized_l2": {
            name: distance for name in mechanics.MATERIAL_COMPARISONS
        },
        "integrity": integrity,
        "read_diagnostics": {
            "raw_unconditioned": {
                "predictor_logit_changed_fraction": 1.0,
                "answer_ce_minus_correct": 0.25,
            }
        },
    }


def test_mechanics_analysis_applies_global_row_gates_once() -> None:
    rows = [_analysis_row(index, 0.1) for index in range(mechanics.ROWS)]

    result = mechanics.mechanics_analysis(rows)

    assert result["evaluation_calls"] == 1
    assert result["passed"] is True
    for name in mechanics.MATERIAL_COMPARISONS:
        assert result["aggregate"][name]["positive_row_fraction"] == 1.0


def test_mechanics_analysis_rejects_below_threshold_rows() -> None:
    rows = [_analysis_row(index, 0.1) for index in range(mechanics.ROWS)]
    rows[0]["state_normalized_l2"] = {
        name: 0.0 for name in mechanics.MATERIAL_COMPARISONS
    }
    rows[1]["state_normalized_l2"] = {
        name: 0.0 for name in mechanics.MATERIAL_COMPARISONS
    }

    result = mechanics.mechanics_analysis(rows)

    assert result["passed"] is False
    assert result["aggregate"][mechanics.MATERIAL_COMPARISONS[0]][
        "positive_row_fraction"
    ] == 30 / 32


def test_protocol_has_canonical_nonplaceholder_receipt() -> None:
    protocol_path = Path(mechanics.PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt")

    assert receipt["payload_sha256"] == mechanics.canonical_sha256(unsigned)
    assert receipt["payload_sha256"] == mechanics.PROTOCOL_PAYLOAD_SHA256
    assert mechanics.sha256_file(protocol_path) == mechanics.PROTOCOL_FILE_SHA256
    assert protocol["mechanics_gate"]["write_conditions"] == list(
        mechanics.WRITE_CONDITIONS
    )
    assert protocol["mechanics_gate"]["read_conditions"] == list(
        mechanics.READ_CONDITIONS
    )
    assert "PLACEHOLDER" not in protocol_path.read_text(encoding="utf-8")
    assert "TO_BE_FILLED" not in protocol_path.read_text(encoding="utf-8")


def test_shard_loader_rejects_rank_swapped_complete_coverage(tmp_path: Path) -> None:
    source_rows = [
        {
            "source_index": index,
            "row_sha256": f"row-{index}",
            "donor_source_index": (index + 1) % mechanics.ROWS,
            "donor_row_sha256": f"row-{(index + 1) % mechanics.ROWS}",
        }
        for index in range(mechanics.ROWS)
    ]
    binding = {"test": "binding"}
    assignments = [source_rows[rank :: mechanics.WORLD_SIZE] for rank in range(4)]
    assignments[0], assignments[1] = assignments[1], assignments[0]
    for rank, assigned in enumerate(assignments):
        mechanics._write_shard(
            tmp_path,
            rank=rank,
            rows=[_analysis_row(int(row["source_index"]), 0.1) for row in assigned],
            binding=binding,
            assigned_source_rows=assigned,
        )

    with pytest.raises(ValueError, match="shard contract differs"):
        mechanics._load_shards(tmp_path, source_rows, binding=binding)


@pytest.mark.parametrize(("passed", "expected"), ((True, 0), (False, 1)))
def test_main_returns_scientific_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected: int,
) -> None:
    base_model = tmp_path / "base"
    materialization = tmp_path / "materialization"
    base_model.mkdir()
    materialization.mkdir()
    monkeypatch.setattr(
        mechanics,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            base_model=base_model,
            materialization_root=materialization,
            output_dir=tmp_path / "output",
        ),
    )
    monkeypatch.setattr(mechanics, "run", lambda **kwargs: {"passed": passed})

    assert mechanics.main([]) == expected


def test_main_propagates_operational_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_model = tmp_path / "base"
    materialization = tmp_path / "materialization"
    base_model.mkdir()
    materialization.mkdir()
    monkeypatch.setattr(
        mechanics,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            base_model=base_model,
            materialization_root=materialization,
            output_dir=tmp_path / "output",
        ),
    )

    def fail(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("operational failure")

    monkeypatch.setattr(mechanics, "run", fail)
    with pytest.raises(RuntimeError, match="operational failure"):
        mechanics.main([])


def test_result_validator_rejects_status_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps({"receipt": {"payload_sha256": "launch"}}), encoding="utf-8"
    )
    monkeypatch.setattr(mechanics, "LAUNCH_BINDING", launch_path)
    result = {
        "schema": mechanics.SCHEMA,
        "status": "wrong",
        "passed": False,
        "protocol_payload_sha256": mechanics.PROTOCOL_PAYLOAD_SHA256,
        "protocol_file_sha256": mechanics.PROTOCOL_FILE_SHA256,
        "mechanics_evaluation_calls": 1,
        "causal_protocol_drafting_authorized": False,
        "causal_bytes_open_authorized": False,
        "causal_authorized": False,
        "model_or_adapter_training_authorized": False,
        "generation_authorized": False,
        "native_benchmark_authorized": False,
        "materialization_bundles_opened": ["mechanics"],
        "firewall": {
            "causal_path_statted_listed_hashed_or_opened_by_experiment_runner": False
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": mechanics.canonical_sha256(result),
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="result contract differs"):
        mechanics._validate_result(result_path)
