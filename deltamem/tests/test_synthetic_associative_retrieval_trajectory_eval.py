from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_associative_retrieval_trajectory_eval as evaluator,
)


def _state(*, layers: int = 42) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "shared": torch.arange(4, dtype=torch.float32).reshape(4, 1),
    }
    for layer in range(layers):
        prefix = f"layer.{layer}"
        state[f"{prefix}.__projected_kv_keys"] = (
            torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) + layer
        )
        state[f"{prefix}.__projected_kv_values"] = (
            torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) + layer
        )
        state[f"{prefix}.__projected_kv_occupied"] = torch.ones(
            4, 2, dtype=torch.bool
        )
    return state


def _route_modules(*, expected: bool = True, absent: bool = False):
    modules = []
    for layer in evaluator.TARGET_LAYERS:
        routes = None
        if not absent:
            routes = torch.zeros(4, 2, 2)
            for row, slot in enumerate(evaluator.EXPECTED_QUERY_SLOTS):
                routes[row, 0, slot] = 1.0
            if not expected and layer == 17:
                routes[2, 0] = torch.tensor([1.0, 0.0])
        module = SimpleNamespace(
            layer_idx=layer,
            last_read_routes=routes,
            projected_kv_keys=None,
            projected_kv_values=None,
            projected_kv_occupied=None,
            projected_kv_surprise=None,
        )
        modules.append((f"layer.{layer}", module))
    return modules


def _condition(
    margins: list[float],
    *,
    route_passed: bool = True,
    absence_passed: bool | None = None,
) -> dict[str, object]:
    donor_logits = [0.0] * 4
    rows = evaluator.build_row_score_reports(
        margins,
        donor_logits,
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        ["t0", "t1", "t2", "t3"],
        ["d0", "d1", "d2", "d3"],
    )
    result: dict[str, object] = {
        "rows": rows,
        "route_audit": {"passed": route_passed},
    }
    if absence_passed is not None:
        result["absence_audit"] = {"passed": absence_passed}
    return result


def _passing_conditions() -> dict[str, dict[str, object]]:
    return {
        "correct": _condition([4.0, 3.5, 3.0, 2.5]),
        "donor": _condition([-3.0, -2.5, -2.0, -1.5]),
        "wrong_slot": _condition([-2.0, -1.5, -1.0, -0.5]),
        "no_write": _condition(
            [0.25, -0.25, 0.5, -0.5],
            route_passed=True,
            absence_passed=True,
        ),
    }


def _identical_write_audit(*, passed: bool = True) -> dict[str, object]:
    return {
        "passed": passed,
        "row_pairs": [[0, 2], [1, 3]],
        "mismatch_count": 0 if passed else 1,
    }


def test_artifact_snapshot_rejects_mutation(tmp_path: Path) -> None:
    files = {}
    for name in evaluator.CHECKPOINT_ARTIFACT_NAMES:
        path = tmp_path / name
        path.write_bytes(f"before:{name}".encode("ascii"))
        files[name] = path
    snapshot = evaluator.snapshot_artifacts(files)

    assert evaluator.verify_artifact_snapshot(snapshot) == snapshot
    files["delta_mem_adapter.pt"].write_bytes(b"after")

    with pytest.raises(ValueError, match="changed during evaluation"):
        evaluator.verify_artifact_snapshot(snapshot)


def test_wrong_slot_transform_flips_values_only() -> None:
    state = _state()
    original = {name: value.clone() for name, value in state.items()}

    transformed = evaluator.build_wrong_slot_state(state)

    evaluator.validate_wrong_slot_state_transform(state, transformed)
    for name, value in state.items():
        assert torch.equal(value, original[name])
        expected = value.flip(1) if name.endswith(".__projected_kv_values") else value
        assert torch.equal(transformed[name], expected)
    corrupted = {name: value.clone() for name, value in transformed.items()}
    corrupted["layer.0.__projected_kv_keys"].add_(1)
    with pytest.raises(ValueError, match="wrong tensor"):
        evaluator.validate_wrong_slot_state_transform(state, corrupted)


def test_donor_transform_is_exact_row_permutation() -> None:
    state = _state()

    donor = evaluator.build_donor_state(state)

    evaluator.validate_donor_state_transform(state, donor)
    order = torch.tensor(evaluator.DONOR_INDICES)
    for name, value in state.items():
        assert torch.equal(donor[name], value.index_select(0, order))
    corrupted = {name: value.clone() for name, value in donor.items()}
    corrupted["shared"][0] = -1
    with pytest.raises(ValueError, match="permutation differs"):
        evaluator.validate_donor_state_transform(state, corrupted)


def test_score_rows_report_winner_token_and_margin() -> None:
    rows = evaluator.build_row_score_reports(
        [3.0, -1.0, 2.5, 0.0],
        [1.0, 4.0, 1.5, 0.0],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        ["ta", "tb", "tc", "td"],
        ["da", "db", "dc", "dd"],
    )

    assert rows[0]["winning_role"] == "target"
    assert rows[0]["winning_token_id"] == 10
    assert rows[0]["winning_token_text"] == "ta"
    assert rows[0]["target_minus_donor_margin"] == 2.0
    assert rows[1]["winning_role"] == "donor"
    assert rows[1]["winning_token_id"] == 21
    assert rows[1]["target_minus_donor_margin"] == -5.0


def test_score_rows_treat_exact_tie_as_neither_winner() -> None:
    rows = evaluator.build_row_score_reports(
        [1.0] * 4,
        [1.0] * 4,
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        ["ta", "tb", "tc", "td"],
        ["da", "db", "dc", "dd"],
    )

    assert all(row["winning_role"] == "tie" for row in rows)
    assert all(row["winning_token_id"] is None for row in rows)
    assert all(row["winning_token_text"] is None for row in rows)


def test_expected_route_audit_requires_all_42_layers() -> None:
    passed = evaluator.audit_expected_read_routes(
        _route_modules(),
        [1, 1, 1, 1],
    )
    failed = evaluator.audit_expected_read_routes(
        _route_modules(expected=False),
        [1, 1, 1, 1],
    )

    assert passed["passed"] is True
    assert passed["layer_count"] == 42
    assert passed["intended_slot_match_count"] == 168
    assert passed["intended_slot_match_fraction"] == 1.0
    assert passed["exact_intended_layer_count"] == 42
    assert passed["query_separation_match_count"] == 84
    assert passed["query_separation_match_fraction"] == 1.0
    assert passed["same_query_consistency_match_count"] == 84
    assert passed["same_query_consistency_match_fraction"] == 1.0
    assert all(
        layer["target_predictor_slot_indices"] == [0, 0, 1, 1]
        for layer in passed["layers"]
    )
    assert failed["passed"] is False
    assert failed["intended_slot_match_count"] == 167
    assert failed["intended_slot_match_fraction"] == 167 / 168
    assert failed["exact_intended_layer_count"] == 41
    assert failed["query_separation_match_count"] == 83
    assert failed["query_separation_match_fraction"] == 83 / 84
    assert failed["same_query_consistency_match_count"] == 83
    assert failed["same_query_consistency_match_fraction"] == 83 / 84
    assert failed["layers"][17]["passed"] is False
    with pytest.raises(ValueError, match="42 layers"):
        evaluator.audit_expected_read_routes(
            _route_modules()[:-1],
            [1, 1, 1, 1],
        )


def test_route_audit_exposes_constant_slot_shortcut() -> None:
    modules = _route_modules()
    for _, module in modules:
        module.last_read_routes.zero_()
        module.last_read_routes[:, 0, 0] = 1.0

    audit = evaluator.audit_expected_read_routes(modules, [1, 1, 1, 1])

    assert audit["passed"] is False
    assert audit["intended_slot_match_count"] == 84
    assert audit["intended_slot_match_fraction"] == 0.5
    assert audit["exact_intended_layer_count"] == 0
    assert audit["query_separation_match_count"] == 0
    assert audit["query_separation_match_fraction"] == 0.0
    assert audit["same_query_consistency_match_count"] == 84
    assert audit["same_query_consistency_match_fraction"] == 1.0


def test_identical_write_audit_detects_state_row_drift() -> None:
    state = _state(layers=1)
    for tensor in state.values():
        tensor[2].copy_(tensor[0])
        tensor[3].copy_(tensor[1])

    passed = evaluator.audit_identical_write_rows(state)

    assert passed["passed"] is True
    assert passed["mismatch_count"] == 0
    state["layer.0.__projected_kv_values"][2, 0, 0].add_(1)
    failed = evaluator.audit_identical_write_rows(state)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1
    assert failed["mismatches"][0]["row_pair"] == [0, 2]


def test_no_write_audit_requires_absent_state_and_routes() -> None:
    modules = _route_modules(absent=True)

    passed = evaluator.audit_no_write_absence(modules, [1, 1, 1, 1])

    assert passed["passed"] is True
    assert passed["projected_kv_state_absent"] is True
    assert passed["read_routes_absent"] is True
    modules[4][1].projected_kv_values = torch.zeros(4, 2, 3)
    modules[9][1].last_read_routes = torch.zeros(4, 2, 2)
    failed = evaluator.audit_no_write_absence(modules, [1, 1, 1, 1])
    assert failed["passed"] is False
    assert failed["projected_kv_state_absent"] is False
    assert failed["read_routes_absent"] is False


def test_strict_checkpoint_gate_passes_only_complete_contract() -> None:
    conditions = _passing_conditions()

    gate = evaluator.build_checkpoint_gate(conditions, _identical_write_audit())

    assert gate["passed"] is True
    assert gate["causal_content_passed"] is True
    assert gate["semantic_addressing_passed"] is True
    assert all(gate["criteria"].values())
    assert gate["winner_counts"]["correct"] == {
        "target": 4,
        "donor": 0,
        "tie": 0,
    }


@pytest.mark.parametrize(
    ("condition", "replacement", "failed_criterion"),
    [
        ("correct", [4.0, 3.0, 2.0, 0.0], "correct_target_wins_all_four"),
        ("donor", [-2.0, -1.0, 0.25, -0.5], "donor_state_selects_donor_all_four"),
        ("wrong_slot", [-2.0, -1.0, 0.0, -0.5], "wrong_slot_selects_donor_all_four"),
    ],
)
def test_strict_checkpoint_gate_rejects_winner_or_tie_failures(
    condition: str,
    replacement: list[float],
    failed_criterion: str,
) -> None:
    conditions = _passing_conditions()
    conditions[condition] = _condition(replacement)

    gate = evaluator.build_checkpoint_gate(conditions, _identical_write_audit())

    assert gate["passed"] is False
    assert gate["causal_content_passed"] is False
    assert gate["semantic_addressing_passed"] is True
    assert gate["criteria"][failed_criterion] is False


def test_checkpoint_gate_keeps_route_failure_out_of_causal_verdict() -> None:
    conditions = _passing_conditions()
    conditions["wrong_slot"]["route_audit"] = {"passed": False}

    gate = evaluator.build_checkpoint_gate(conditions, _identical_write_audit())

    assert gate["passed"] is False
    assert gate["causal_content_passed"] is True
    assert gate["semantic_addressing_passed"] is False
    assert gate["criteria"]["wrong_slot_routes_match_all_42_layers"] is False


def test_causal_gate_rejects_margin_absence_and_identical_write_failures() -> None:
    conditions = _passing_conditions()
    conditions["no_write"] = _condition(
        [4.5, -0.25, 0.5, -0.5],
        route_passed=True,
        absence_passed=False,
    )

    gate = evaluator.build_checkpoint_gate(
        conditions,
        _identical_write_audit(passed=False),
    )

    assert gate["passed"] is False
    assert gate["causal_content_passed"] is False
    assert gate["semantic_addressing_passed"] is True
    assert gate["criteria"]["correct_margin_exceeds_all_controls_row_wise"] is False
    assert gate["criteria"]["no_write_state_and_routes_absent"] is False
    assert gate["criteria"]["identical_write_rows_produce_identical_state"] is False


def test_no_write_winner_count_is_diagnostic_not_causal_gate() -> None:
    conditions = _passing_conditions()
    conditions["no_write"] = _condition(
        [0.5, 0.5, 0.5, 0.5],
        route_passed=True,
        absence_passed=True,
    )

    gate = evaluator.build_checkpoint_gate(conditions, _identical_write_audit())

    assert gate["passed"] is True
    assert gate["causal_content_passed"] is True
    assert gate["diagnostics"]["no_write_cannot_solve_all_four"] is False


def test_trajectory_reports_passing_steps_and_selects_robust_best() -> None:
    results = []
    margins_by_step = {
        8: [1.0, 1.0, 1.0, 1.0],
        16: [3.0, 2.0, 2.0, 2.0],
        32: [2.5, 2.5, 2.5, 2.5],
        64: [5.0, 5.0, 5.0, 5.0],
    }
    for step in evaluator.CHECKPOINT_STEPS:
        causal_passed = step == 64
        results.append(
            {
                "step": step,
                "gate": {
                    "passed": False,
                    "causal_content_passed": causal_passed,
                    "semantic_addressing_passed": False,
                },
                "conditions": {
                    "correct": _condition(margins_by_step[step]),
                },
            }
        )

    summary = evaluator.summarize_trajectory(results)

    assert summary["causal_passing_steps"] == [64]
    assert summary["semantic_addressing_passing_steps"] == []
    assert summary["fully_passing_steps"] == []
    assert summary["all_steps_fully_passed"] is False
    assert summary["best_causal_step"] == 64
    assert summary["best_causal_step_score"] == {
        "minimum_correct_target_minus_donor_margin": 5.0,
        "mean_correct_target_minus_donor_margin": 5.0,
    }
    assert summary["decision"] == "causal_memory_content_but_semantic_router_failed"


def test_pairing_provenance_binds_source_rows_and_donors() -> None:
    row_records = []
    pairs = []
    target_ids = [10, 11, 12, 13]
    for source_index, donor_index in enumerate(evaluator.DONOR_INDICES):
        row_records.append(
            {
                "token_metadata": {
                    "target_label_position": 8,
                    "target_token_id": target_ids[source_index],
                }
            }
        )
        pairs.append(
            {
                "source_index": source_index,
                "donor_index": donor_index,
                "target_label_positions": [8],
                "target_token_ids": [target_ids[source_index]],
                "donor_token_ids": [target_ids[donor_index]],
            }
        )
    pairing = {
        "objective_version": "scene_state_identity_ce_v2",
        "pairing_locked": True,
        "locked_donor_indices": list(evaluator.DONOR_INDICES),
        "data_seed": evaluator.SEED,
        "splits": {
            "train": {
                "sample_count": 4,
                "pairing_locked": True,
                "locked_donor_indices": list(evaluator.DONOR_INDICES),
                "pairs": pairs,
            }
        },
    }
    pairing["manifest_sha256"] = evaluator.canonical_sha256(pairing)

    result = evaluator.validate_pairing_manifest(
        pairing,
        {"row_records": row_records},
    )

    assert result["validated"] is True
    corrupted = copy.deepcopy(pairing)
    corrupted["splits"]["train"]["pairs"][0]["donor_index"] = 3
    unsigned = dict(corrupted)
    unsigned.pop("manifest_sha256")
    corrupted["manifest_sha256"] = evaluator.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="donor index differs"):
        evaluator.validate_pairing_manifest(
            corrupted,
            {"row_records": row_records},
        )


def test_atomic_fresh_writer_never_replaces_output(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"
    payload = {"complete": True}

    written = evaluator.write_json_atomic_fresh(output, payload)

    assert json.loads(written.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="must be fresh"):
        evaluator.write_json_atomic_fresh(output, {"complete": False})
    assert json.loads(output.read_text(encoding="utf-8")) == payload
