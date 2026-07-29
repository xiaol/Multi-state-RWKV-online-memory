from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v8_gate as gate
from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as state_eval


def _pairing() -> tuple[dict[str, object], dict[int, int]]:
    value = list(gate.VALUE14_ORDINALS)
    presence = [ordinal for ordinal in range(32) if ordinal not in gate.VALUE14_SET]
    donor_by_ordinal: dict[int, int] = {}
    for ordinals in (value, presence):
        for offset in range(0, len(ordinals), 2):
            left, right = ordinals[offset : offset + 2]
            donor_by_ordinal[left] = right
            donor_by_ordinal[right] = left
    directed_pairs = []
    for value_offset, ordinal in enumerate(value):
        stratum = (
            "same_cardinality_value"
            if value_offset < 10
            else "cross_cardinality_value"
        )
        directed_pairs.append(
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor_by_ordinal[ordinal],
                "target_stratum": stratum,
            }
        )
    for ordinal in presence:
        directed_pairs.append(
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor_by_ordinal[ordinal],
                "target_stratum": "presence",
            }
        )
    return {"directed_pairs": directed_pairs}, donor_by_ordinal


def _semantic_report(
    *,
    ordinal: int,
    donor_ordinal: int,
    condition: str,
) -> dict[str, object]:
    if condition == "state_only":
        mean_nll, alternative_nll, margin = 0.1, 2.1, 2.0
    elif condition == "state_only_donor":
        mean_nll, alternative_nll, margin = 2.1, 0.1, -2.0
    else:
        mean_nll, alternative_nll, margin = 1.1, 1.1, 0.0
    return {
        "pair_target": {
            "mask_mode": gate.PAIR_TARGET_DECISION_MASK_MODE,
            "normalization": gate.PAIR_TARGET_DECISION_NLL_NORMALIZATION,
            "token_count": 1,
            "mean_nll": mean_nll,
            "alternative_target_mean_nll": alternative_nll,
            "selected_over_alternative_logprob_margin": margin,
            "selected_target_positions": [7],
            "selected_target_token_ids": [1000 + ordinal],
            "donor_target_token_ids": [1000 + donor_ordinal],
            "first_differing_semantic_ordinal": 0,
            "causal_prefix_sha256": f"causal-{ordinal}",
            "donor_source_index": donor_ordinal,
            "donor_row_sha256": f"row-{donor_ordinal}",
            "read_rendered_sha256": f"read-{ordinal}",
        }
    }


def _records(*, correct_value_exact: bool) -> tuple[dict[str, list[dict]], dict]:
    pairing, donor_by_ordinal = _pairing()
    gold = {ordinal: {"boundaries": [ordinal + 1]} for ordinal in range(32)}
    records: dict[str, list[dict]] = {condition: [] for condition in gate.CONDITIONS}
    for condition in gate.CONDITIONS:
        for ordinal in range(32):
            if condition == "state_only":
                parsed = gold[ordinal]
                if ordinal in gate.VALUE14_SET and not correct_value_exact:
                    parsed = {"boundaries": []}
            elif condition == "state_only_donor":
                parsed = gold[donor_by_ordinal[ordinal]]
            else:
                parsed = {"boundaries": []}
            strict = gate.score_prediction("scene", parsed, gold[ordinal])
            record = {
                "condition": condition,
                "train_row_ordinal": ordinal,
                "source_index": ordinal,
                "row_sha256": f"row-{ordinal}",
                "gold": gold[ordinal],
                "parsed_json": parsed,
                "raw_generation": (
                    "zero-invariant"
                    if condition == "state_only_no_write"
                    else f"{condition}-{ordinal}"
                ),
                "score_strict": strict,
                "score_recovered": gate.recovered_scene_score(parsed, gold[ordinal]),
                "hit_max_new_tokens": False,
                "input_tokens": 8,
                "output_tokens": 4,
                "elapsed_seconds": 0.01,
            }
            if ordinal in gate.VALUE14_SET:
                record["semantic_decision_nll"] = _semantic_report(
                    ordinal=ordinal,
                    donor_ordinal=donor_by_ordinal[ordinal],
                    condition=condition,
                )
            records[condition].append(record)
    return records, pairing


def test_v8_gate_passes_only_with_generation_identity_and_causal_evidence() -> None:
    records, pairing = _records(correct_value_exact=True)

    result = gate.build_v8_gate(
        records_by_condition=records,
        pairing=pairing,
    )

    assert result["status"] == "pass"
    assert result["all_gates_passed"] is True
    assert result["hard32_authorized"] is True
    assert result["full_answer_ce_used_for_gate"] is False
    assert result["full170_authorized"] is False
    assert result["test_authorized"] is False
    assert result["other_benchmarks_authorized"] is False


def test_v8_gate_rejects_selected_token_evidence_without_correct_generation() -> None:
    records, pairing = _records(correct_value_exact=False)

    result = gate.build_v8_gate(
        records_by_condition=records,
        pairing=pairing,
    )

    assert result["status"] == "fail"
    assert result["hard32_authorized"] is False
    assert result["gates"]["value14_correct_identity_generation"] is False


def test_v8_gate_is_unavailable_before_checkpoint_56(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-42"
    checkpoint.mkdir()

    with pytest.raises(
        gate.V8EvaluationContractError,
        match="gate is unavailable before step56",
    ):
        gate.validate_v8_checkpoint(checkpoint, input_contract={})


def test_v8_receipt_rejects_a_different_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_checkpoint = {"memory_dir": "/ssd/run/checkpoint-56", "global_step": 56}
    input_contract = {
        "artifacts": {},
        "v8_source_manifest_sha256": "a" * 64,
        "v8_schedule_entries_sha256": "b" * 64,
        "benchmark_lock": {"protected_benchmark": {}},
    }
    payload = {
        "schema": gate.GATE_RECEIPT_SCHEMA,
        "status": "pass",
        "contract": gate.GATE_CONTRACT,
        "task": gate.TASK_NAME,
        "objective": deepcopy(gate.V8_OBJECTIVE),
        "training_sources": {},
        "v8_source_manifest_sha256": "a" * 64,
        "v8_schedule_entries_sha256": "b" * 64,
        "benchmark_selection_lock": input_contract["benchmark_lock"],
        "checkpoint": {"memory_dir": "/ssd/other/checkpoint-56", "global_step": 56},
        "gate": {
            "status": "pass",
            "all_gates_passed": True,
            "hard32_authorized": True,
        },
    }
    payload["receipt_sha256"] = gate.self_hash_payload(
        payload,
        hash_field="receipt_sha256",
    )
    monkeypatch.setattr(
        gate,
        "validate_v8_checkpoint",
        lambda *args, **kwargs: current_checkpoint,
    )

    with pytest.raises(
        gate.V8EvaluationContractError,
        match="checkpoint binding differs",
    ):
        gate.validate_gate_receipt_for_checkpoint(
            payload,
            memory_dir=current_checkpoint["memory_dir"],
            input_contract=input_contract,
        )


def test_v8_hard32_files_are_untouched_when_train32_receipt_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected_accesses: list[tuple[Path, Path]] = []

    def reject_receipt(*args, **kwargs):
        raise gate.V8EvaluationContractError("receipt rejected")

    def protected_access(*, dataset_file: Path, selection_file: Path):
        protected_accesses.append((dataset_file, selection_file))
        raise AssertionError("protected Hard32 was accessed before authorization")

    monkeypatch.setattr(gate, "validate_gate_receipt_for_checkpoint", reject_receipt)
    monkeypatch.setattr(
        state_eval,
        "validate_historical_v6_hard32_artifacts",
        protected_access,
    )

    with pytest.raises(gate.V8EvaluationContractError, match="receipt rejected"):
        state_eval.validate_scene_v8_train32_hard32_authorization(
            Path("invalid-receipt.json"),
            memory_dir=Path("checkpoint-56"),
            dataset_file=Path("protected-holdout.jsonl"),
            selection_file=Path("protected-selection.json"),
        )

    assert protected_accesses == []
