from __future__ import annotations

from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v13_gate as gate
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v13_launch_contract as launch,
)


def _v13_donor_by_ordinal() -> dict[int, int]:
    result = {low: high for low, high in launch.FIRST_CYCLE_PAIRS}
    result.update({high: low for low, high in launch.FIRST_CYCLE_PAIRS})
    return result


def _v13_pairing() -> dict[str, Any]:
    donor_by_ordinal = _v13_donor_by_ordinal()
    return {
        "directed_pairs": [
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor_by_ordinal[ordinal],
                "target_stratum": (
                    "cross_cardinality_value"
                    if ordinal in {1, 14, 22, 26}
                    else "same_cardinality_value"
                ),
            }
            for ordinal in gate.VALUE14_ORDINALS
        ]
    }


def _v13_gate_records(*, zero_exact_rows: int) -> dict[str, list[dict[str, Any]]]:
    donor_by_ordinal = _v13_donor_by_ordinal()
    gold = {
        ordinal: {"boundaries": [ordinal]} for ordinal in gate.VALUE14_ORDINALS
    }
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in gate.CONDITIONS
    }
    for position, ordinal in enumerate(gate.VALUE14_ORDINALS):
        common = {
            "train_row_ordinal": ordinal,
            "gold": gold[ordinal],
        }
        records["state_only"].append(
            {
                **common,
                "condition": "state_only",
                "parsed_json": gold[ordinal],
                "raw_generation": f"correct-{ordinal}",
            }
        )
        records["state_only_donor"].append(
            {
                **common,
                "condition": "state_only_donor",
                "parsed_json": gold[donor_by_ordinal[ordinal]],
                "raw_generation": f"donor-{ordinal}",
            }
        )
        records["state_only_no_write"].append(
            {
                **common,
                "condition": "state_only_no_write",
                "parsed_json": (
                    gold[ordinal]
                    if position < zero_exact_rows
                    else {"boundaries": []}
                ),
                "raw_generation": "identical-zero-reset-output",
            }
        )
    return records


def _patch_selected_token_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {
        "bidirectional_identity_switch_rows": 14,
        "correct_state_beats_donor_state_on_source_token_rows": 14,
        "correct_state_prefers_source_token_rows": 14,
        "donor_state_prefers_donor_token_rows": 14,
        "correct_state_beats_zero_on_source_token_rows": 14,
    }
    monkeypatch.setattr(
        gate.v10.v9.v8,
        "build_value14_selected_token_evidence",
        lambda **_kwargs: {"overall": identity, "rows": []},
    )


def test_v13_gate_requirements_are_exact_semantic_value14_only() -> None:
    assert gate.GATE_REQUIREMENTS == {
        "correct_strict_exact_rows": 14,
        "donor_identity_strict_exact_rows": 14,
        "correct_donor_semantic_switch_rows": 14,
        "zero_strict_exact_rows_max": 13,
        "zero_reset_control_is_row_invariant": True,
    }
    assert gate.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert len(gate.VALUE14_ORDINALS) == 14


@pytest.mark.parametrize("zero_exact_rows", (13, 14))
def test_v13_gate_zero_no_write_must_not_reproduce_all_correct_rows(
    zero_exact_rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _v13_gate_records(zero_exact_rows=zero_exact_rows)

    for checkpoint_step in gate.CHECKPOINT_STEPS:
        result = gate.build_v13_gate(
            records_by_condition=records,
            pairing=_v13_pairing(),
            checkpoint_step=checkpoint_step,
        )
        generation = result["metrics"]["value14_generation"]
        assert result["checkpoint_step"] == checkpoint_step
        assert result["consumed_pair_presentations"] == checkpoint_step * 7
        assert result["evaluation_scope"] == "exact_value14_ordinals_only"
        assert generation["correct_strict_exact_rows"] == 14
        assert generation["donor_identity_strict_exact_rows"] == 14
        assert generation["correct_donor_semantic_switch_rows"] == 14
        assert generation["zero_strict_exact_rows"] == zero_exact_rows
        assert result["gates"]["value14_correct_semantic_exact"] is True
        assert result["gates"]["value14_donor_semantic_exact"] is True
        assert (
            result["gates"]["value14_correct_donor_semantic_switch"] is True
        )
        assert result["gates"]["zero_reset_control_is_row_invariant"] is True
        expected_pass = zero_exact_rows == 13
        assert result[
            "gates"
        ]["value14_zero_does_not_reproduce_correct_memory"] is expected_pass
        assert result["all_gates_passed"] is expected_pass
        assert result["candidate_authorized"] is expected_pass
        assert result["training_continuation_authorized"] is False


@pytest.mark.parametrize(
    ("condition", "gate_name"),
    (
        ("state_only", "value14_correct_semantic_exact"),
        ("state_only_donor", "value14_donor_semantic_exact"),
    ),
)
def test_v13_gate_rejects_one_missing_correct_or_donor_identity_exact_row(
    condition: str,
    gate_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _v13_gate_records(zero_exact_rows=13)
    records[condition][0]["parsed_json"] = {"boundaries": []}

    result = gate.build_v13_gate(
        records_by_condition=records,
        pairing=_v13_pairing(),
        checkpoint_step=1,
    )

    assert result["gates"][gate_name] is False
    assert result["all_gates_passed"] is False
    assert result["candidate_authorized"] is False


def test_v13_gate_requires_all_fourteen_correct_donor_semantic_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _v13_gate_records(zero_exact_rows=13)
    first = gate.VALUE14_ORDINALS[0]
    paired = _v13_donor_by_ordinal()[first]
    shared_gold = {"boundaries": [first]}
    for ordinal in (first, paired):
        position = gate.VALUE14_ORDINALS.index(ordinal)
        records["state_only"][position]["gold"] = shared_gold
        records["state_only"][position]["parsed_json"] = shared_gold
        records["state_only_donor"][position]["gold"] = shared_gold
        records["state_only_donor"][position]["parsed_json"] = shared_gold

    result = gate.build_v13_gate(
        records_by_condition=records,
        pairing=_v13_pairing(),
        checkpoint_step=1,
    )
    generation = result["metrics"]["value14_generation"]

    assert generation["correct_strict_exact_rows"] == 14
    assert generation["donor_identity_strict_exact_rows"] == 14
    assert generation["correct_donor_semantic_switch_rows"] == 12
    assert result["gates"]["value14_correct_semantic_exact"] is True
    assert result["gates"]["value14_donor_semantic_exact"] is True
    assert result["gates"]["value14_correct_donor_semantic_switch"] is False
    assert result["all_gates_passed"] is False
