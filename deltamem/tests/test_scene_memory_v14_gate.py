from __future__ import annotations

from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v14_gate as gate
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v14_launch_contract as launch,
)


def _donor_by_ordinal() -> dict[int, int]:
    result = {low: high for low, high in launch.FIRST_CYCLE_PAIRS}
    result.update({high: low for low, high in launch.FIRST_CYCLE_PAIRS})
    return result


def _pairing() -> dict[str, Any]:
    donor_by_ordinal = _donor_by_ordinal()
    return {
        "directed_pairs": [
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor_by_ordinal[ordinal],
                "target_stratum": "locked_value14_pair",
            }
            for ordinal in gate.VALUE14_ORDINALS
        ]
    }


def _records(*, zero_exact_rows: int = 13) -> dict[str, list[dict[str, Any]]]:
    donor_by_ordinal = _donor_by_ordinal()
    gold = {
        ordinal: {"boundaries": [ordinal]} for ordinal in gate.VALUE14_ORDINALS
    }
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in gate.CONDITIONS
    }
    for position, ordinal in enumerate(gate.VALUE14_ORDINALS):
        common = {"train_row_ordinal": ordinal, "gold": gold[ordinal]}
        records["state_only"].append(
            {
                **common,
                "condition": "state_only",
                "parsed_json": gold[ordinal],
                # Deliberately noncanonical and unique: semantic exactness is
                # parsed boundary-set equality, not token/prose equality.
                "raw_generation": f"prefix {position}: {{ boundaries: [{ordinal}] }}",
            }
        )
        records["state_only_donor"].append(
            {
                **common,
                "condition": "state_only_donor",
                "parsed_json": gold[donor_by_ordinal[ordinal]],
                "raw_generation": f"donor wording differs for row {ordinal}",
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
                "raw_generation": "row-invariant-zero-control",
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


def test_v14_gate_is_exact_semantic_value14_only() -> None:
    assert gate.GATE_REQUIREMENTS == {
        "correct_strict_exact_rows": 14,
        "donor_identity_strict_exact_rows": 14,
        "correct_donor_semantic_switch_rows": 14,
        "zero_strict_exact_rows_max": 13,
        "zero_reset_control_is_row_invariant": True,
    }
    assert gate.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert len(gate.VALUE14_ORDINALS) == 14
    assert gate.V14_OBJECTIVE["evaluation_rows"] == "exact_value14_ordinals_only"
    assert gate.V14_OBJECTIVE["raw_token_exact_role"] == "telemetry_only"


@pytest.mark.parametrize("checkpoint_step", gate.CHECKPOINT_STEPS)
def test_v14_gate_passes_on_semantics_despite_raw_token_differences(
    checkpoint_step: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _records()

    result = gate.build_v14_gate(
        records_by_condition=records,
        pairing=_pairing(),
        checkpoint_step=checkpoint_step,
    )

    generation = result["metrics"]["value14_generation"]
    assert len({row["raw_generation"] for row in records["state_only"]}) == 14
    assert generation["correct_strict_exact_rows"] == 14
    assert generation["donor_identity_strict_exact_rows"] == 14
    assert generation["correct_donor_semantic_switch_rows"] == 14
    assert generation["zero_strict_exact_rows"] == 13
    assert result["evaluation_scope"] == "exact_value14_ordinals_only"
    assert result["consumed_pair_presentations"] == checkpoint_step * 7
    assert all(result["gates"].values())
    assert result["all_gates_passed"] is True
    assert result["candidate_authorized"] is True
    assert result["hard32_authorized"] is True
    assert result["full170_authorized"] is False
    assert result["test_authorized"] is False
    assert result["other_benchmarks_authorized"] is False
    assert result["training_continuation_authorized"] is False


@pytest.mark.parametrize(
    ("condition", "gate_name"),
    (
        ("state_only", "value14_correct_semantic_exact"),
        ("state_only_donor", "value14_donor_semantic_exact"),
    ),
)
def test_v14_gate_requires_all_fourteen_correct_and_donor_identity_rows(
    condition: str,
    gate_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _records()
    records[condition][0]["parsed_json"] = {"boundaries": []}

    result = gate.build_v14_gate(
        records_by_condition=records,
        pairing=_pairing(),
        checkpoint_step=1,
    )

    assert result["gates"][gate_name] is False
    assert result["all_gates_passed"] is False
    assert result["hard32_authorized"] is False


def test_v14_gate_requires_fourteen_semantic_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _records()
    first = gate.VALUE14_ORDINALS[0]
    paired = _donor_by_ordinal()[first]
    shared_gold = {"boundaries": [first]}
    for ordinal in (first, paired):
        position = gate.VALUE14_ORDINALS.index(ordinal)
        records["state_only"][position]["gold"] = shared_gold
        records["state_only"][position]["parsed_json"] = shared_gold
        records["state_only_donor"][position]["gold"] = shared_gold
        records["state_only_donor"][position]["parsed_json"] = shared_gold

    result = gate.build_v14_gate(
        records_by_condition=records,
        pairing=_pairing(),
        checkpoint_step=1,
    )

    generation = result["metrics"]["value14_generation"]
    assert generation["correct_strict_exact_rows"] == 14
    assert generation["donor_identity_strict_exact_rows"] == 14
    assert generation["correct_donor_semantic_switch_rows"] == 12
    assert result["gates"]["value14_correct_donor_semantic_switch"] is False
    assert result["all_gates_passed"] is False


def test_v14_gate_requires_reciprocal_donor_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    pairing = _pairing()
    pairing["directed_pairs"][0]["donor_train_row_ordinal"] = (
        gate.VALUE14_ORDINALS[2]
    )

    with pytest.raises(gate.V14EvaluationContractError, match="donor mapping differs"):
        gate.build_v14_gate(
            records_by_condition=_records(),
            pairing=pairing,
            checkpoint_step=1,
        )


@pytest.mark.parametrize("zero_exact_rows", (13, 14))
def test_v14_zero_control_cannot_reproduce_every_memory(
    zero_exact_rows: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    result = gate.build_v14_gate(
        records_by_condition=_records(zero_exact_rows=zero_exact_rows),
        pairing=_pairing(),
        checkpoint_step=1,
    )

    expected = zero_exact_rows == 13
    assert result["gates"]["zero_reset_control_is_row_invariant"] is True
    assert result[
        "gates"
    ]["value14_zero_does_not_reproduce_correct_memory"] is expected
    assert result["all_gates_passed"] is expected


def test_v14_zero_control_must_be_raw_output_row_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_selected_token_evidence(monkeypatch)
    records = _records()
    records["state_only_no_write"][0]["raw_generation"] = "different zero output"

    result = gate.build_v14_gate(
        records_by_condition=records,
        pairing=_pairing(),
        checkpoint_step=1,
    )

    assert result["gates"]["zero_reset_control_is_row_invariant"] is False
    assert result["all_gates_passed"] is False
