from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_rwkv_cumulative_virtual_kv_mechanics as mechanics,
)


PROTOCOL = Path(mechanics.PROTOCOL)


def synthetic_natural_cache(
    sources: tuple[int, ...], ordered_names: tuple[str, ...]
) -> dict[int, dict[str, object]]:
    result = {}
    for source in sources:
        per_module = {}
        active = source % mechanics.SLOTS
        for layer, name in enumerate(ordered_names):
            state = torch.zeros(1, 1, mechanics.SLOTS, mechanics.STATE_DIM, mechanics.STATE_DIM)
            keys = torch.zeros(1, mechanics.SLOTS, mechanics.ADDRESS_DIM)
            occupied = torch.zeros(1, mechanics.SLOTS, dtype=torch.bool)
            state[:, :, active] = float(source * 100 + layer + 1)
            keys[:, active] = float(source * 100 + layer + 1)
            occupied[:, active] = True
            per_module[name] = {
                "delta_state": state,
                "projected_kv_keys": keys,
                "projected_kv_occupied": occupied,
            }
        result[source] = {"state": per_module}
    return result


def test_cumulative_virtual_kv_mechanics_protocol_is_signed_and_sealed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    receipt = unsigned.pop("receipt")
    assert receipt == {
        "algorithm": "sha256",
        "payload_scope": "canonical_protocol_without_receipt",
        "payload_sha256": distributed.canonical_sha256(unsigned),
    }
    assert protocol["architecture"]["anchor_layers"] == list(mechanics.ANCHORS)
    assert protocol["architecture"]["compatibility_scale"] == mechanics.STATE_DIM
    assert protocol["mechanics_bundle_byte_read_authorized_by_this_protocol"] is True
    assert protocol["causal_bundle_byte_read_authorized"] is False
    assert protocol["data_lifecycle"]["generation_or_native_benchmark_authorized"] is False
    assert protocol["data_lifecycle"]["full_bandwidth_feedback_authorized"] is False
    mechanics.validate_protocol_contract(protocol)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("authorization_basis", "runtime_parent_commit"), "969a320"),
        (("frozen_inputs", "frozen_map_digest"), "wrong"),
        (("frozen_inputs", "conditioning_seed"), 152),
        (("value_builder", "probe_rank"), 7),
        (("candidate_and_control_bank", "joint_slot_permutation"), [0, 1, 2, 3]),
        (("required_gates", "cached_null_replay_max_absolute_logit_tolerance"), 1e-6),
        (("execution", "protected_bundle_byte_opens"), 4),
        (("data_lifecycle", "causal_bundle_byte_read_authorized"), True),
    ),
)
def test_protocol_contract_rejects_bound_field_drift(
    path: tuple[str, str], replacement: object
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol[path[0]][path[1]] = replacement
    with pytest.raises(ValueError, match="protocol contract differs"):
        mechanics.validate_protocol_contract(protocol)


def test_launch_contract_requires_exact_full_runtime_parent() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    launch = {
        "schema": protocol["launch_binding"]["schema"],
        "authorized_code_commit": "code-commit",
        "runtime_parent_commit": mechanics.RUNTIME_PARENT_COMMIT,
        "protocol_file_sha256": mechanics.sha256_file(PROTOCOL),
        "protocol_receipt": protocol["receipt"]["payload_sha256"],
        "dependency_bindings": mechanics.dependency_bindings(),
        "dependency_digest": mechanics.canonical_sha256(
            mechanics.dependency_bindings()
        ),
    }
    mechanics.validate_launch_contract(
        protocol,
        launch,
        head_parent="code-commit",
        code_parent=mechanics.RUNTIME_PARENT_COMMIT,
    )
    shortened = deepcopy(launch)
    shortened["runtime_parent_commit"] = mechanics.RUNTIME_PARENT_COMMIT[:7]
    with pytest.raises(ValueError, match="launch binding differs"):
        mechanics.validate_launch_contract(
            protocol,
            shortened,
            head_parent="code-commit",
            code_parent=mechanics.RUNTIME_PARENT_COMMIT,
        )


def test_output_preparation_and_primary_only_loader_contract(tmp_path: Path) -> None:
    output = tmp_path / "result"
    mechanics.prepare_output(SimpleNamespace(is_primary=False), output)
    assert not output.exists()
    mechanics.prepare_output(SimpleNamespace(is_primary=True), output)
    assert output.is_dir()
    with pytest.raises(ValueError, match="must be fresh"):
        mechanics.prepare_output(SimpleNamespace(is_primary=True), output)
    source = Path(mechanics.__file__).read_text(encoding="utf-8")
    assert source.count("mechanics._load_authorized_mechanics_bundle(") == 1


def test_control_banks_keep_native_components_separate() -> None:
    ordered_names = tuple(f"layer.{layer}" for layer in range(mechanics.MODULES))
    names_by_layer = {layer: ordered_names[layer] for layer in mechanics.ANCHORS}
    sources = (10, 11, 20, 30)
    cache = synthetic_natural_cache(sources, ordered_names)
    states, addresses, occupied, source_ids = mechanics.control_banks(
        cache,
        sources,
        names_by_layer,
        ordered_names,
        torch.device("cpu"),
    )
    permutation = torch.tensor(mechanics.SLOT_PERMUTATION)
    correct = mechanics.CONTROL_INDEX["correct_four_way"]
    permuted = mechanics.CONTROL_INDEX["joint_slot_permutation"]
    single = mechanics.CONTROL_INDEX["single_target"]
    state_only = mechanics.CONTROL_INDEX["matched_donor_state_only"]
    address_only = mechanics.CONTROL_INDEX["matched_donor_address_only"]
    zero_state = mechanics.CONTROL_INDEX["zero_state"]
    zero_address = mechanics.CONTROL_INDEX["zero_address"]
    for layer in mechanics.ANCHORS:
        assert tuple(states[layer].shape) == (
            len(mechanics.CONTROL_NAMES),
            1,
            mechanics.SLOTS,
            mechanics.STATE_DIM,
            mechanics.STATE_DIM,
        )
        assert occupied[layer][correct].all()
        assert torch.equal(
            states[layer][permuted], states[layer][correct].index_select(1, permutation)
        )
        assert torch.equal(
            addresses[layer][permuted],
            addresses[layer][correct].index_select(0, permutation),
        )
        assert torch.equal(
            source_ids[layer][permuted],
            source_ids[layer][correct].index_select(0, permutation),
        )
        assert occupied[layer][single].tolist() == [True, False, False, False]
        assert torch.equal(states[layer][single], states[layer][address_only])
        assert torch.equal(addresses[layer][single], addresses[layer][state_only])
        assert not torch.equal(states[layer][single], states[layer][state_only])
        assert not torch.equal(addresses[layer][single], addresses[layer][address_only])
        assert states[layer][zero_state].eq(0.0).all()
        assert addresses[layer][zero_address].eq(0.0).all()


def test_candidate_sources_are_component_disjoint_and_deterministic() -> None:
    rows = [
        {"source_index": source, "donor_source_index": donor}
        for source, donor in (
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (4, 5),
            (5, 4),
            (6, 7),
            (7, 6),
        )
    ]
    candidates = mechanics.candidate_sources(rows)
    assert candidates[0] == (0, 1, 2, 4)
    assert candidates[3] == (3, 2, 0, 4)
    for source, values in candidates.items():
        assert values[0] == source
        assert len(values) == len(set(values)) == mechanics.SLOTS


def test_aggregate_gates_cached_replay_and_keeps_full_parity_diagnostic() -> None:
    identity = {
        str(layer): {
            "strict_target_top1": True,
            "target_over_strongest_wrong_margin": 1.0,
            "target_over_matched_donor_margin": 1.0,
            "target_over_live_layer_roll_margin": 1.0,
            "correct_virtual_mass": 1.0,
        }
        for layer in mechanics.ANCHORS
    }
    branch_audit = {
        "one_real_position_appended_no_virtual_slots": True,
        "prefix_cache_bytes_unchanged": {
            str(layer): True for layer in mechanics.ANCHORS
        },
        "projected_carrier_bytes_unchanged": True,
        "rwkv_state_bytes_unchanged": True,
    }
    material_names = (
        "correct_vs_provider_off",
        "donor_state_only_vs_single_target",
        "donor_address_only_vs_single_target",
        "donor_both_vs_single_target",
        "layer_state_only_vs_single_target",
        "layer_address_only_vs_single_target",
        "layer_both_vs_single_target",
    )
    row = {
        "identity": identity,
        "comparisons": {
            **{name: {"material": True} for name in material_names},
            "full_null_vs_cached_null": {"maximum_absolute_delta": 2.0},
        },
        "audit": {
            "all_router_path_checks": True,
            "zero_controls_byte_exact_provider_off": {"zero": True},
            "joint_slot_permutation_final_logits_close": True,
            "cached_null_replay_close": True,
            "full_cached_null_diagnostic_close": False,
            "routed": branch_audit,
            "provider_off": branch_audit,
            "provider_off_replay": branch_audit,
        },
    }
    protocol = {
        "required_gates": {
            "identity_per_anchor": {
                "strict_target_top1_fraction": 1.0,
                "mean_target_over_strongest_wrong_margin": 1.0,
                "matched_donor_positive_fraction": 1.0,
                "live_layer_roll_positive_fraction": 1.0,
                "nonzero_virtual_mass_fraction": 1.0,
            },
            "minimum_anchor_layers_passing": len(mechanics.ANCHORS),
            "material_predictor_change_fraction": 1.0,
        }
    }

    result = mechanics.aggregate([row], protocol)

    assert result["passed"] is True
    assert result["invariants"]["all_cached_null_replays_close"] is True
    assert (
        result["diagnostics"][
            "all_full_cached_null_close_at_diagnostic_tolerance"
        ]
        is False
    )
    assert result["diagnostics"]["maximum_full_cached_null_logit_delta"] == 2.0


def test_independent_router_equation_rejects_self_consistent_tampering() -> None:
    compatibility_map = SimpleNamespace(
        down=torch.eye(2, dtype=torch.float32),
        up=torch.eye(2, dtype=torch.float32),
    )
    receptance = torch.tensor([[[[1.0, 1.0]]]])
    state = torch.ones(1, 1, 2, 2, 2)
    addresses = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
    occupied = torch.ones(1, 2, dtype=torch.bool)
    running = {
        "count": 0,
        "score_sum": torch.zeros(1, 2),
        "active": occupied.clone(),
    }
    expected = mechanics.independent_router_equation(
        compatibility_map=compatibility_map,
        receptance=receptance,
        state=state,
        addresses=addresses,
        occupied=occupied,
        running=running,
        compatibility_scale=2.0,
    )
    torch.testing.assert_close(expected["local_scores"], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(expected["attention_bias"], torch.tensor([[2.0, 0.0]]))
    pristine = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in expected.items()
    }
    assert all(mechanics.independent_equation_checks(pristine, expected).values())
    mutations = {
        "count": lambda value: value + 1,
        "local_scores": lambda value: value + 1.0,
        "accumulated_scores": lambda value: value + 1.0,
        "attention_bias": lambda value: value + 1.0,
        "active": lambda value: ~value,
    }
    for field, mutate in mutations.items():
        diagnostic = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in expected.items()
        }
        diagnostic[field] = mutate(diagnostic[field])
        checks = mechanics.independent_equation_checks(diagnostic, expected)
        assert not all(checks.values())
