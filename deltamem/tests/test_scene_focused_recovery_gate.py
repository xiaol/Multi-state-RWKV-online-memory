from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_focused_recovery_gate as gate
from experiments.rethinking_rwkv_ms_gemma import run_scene_state_eval as state_eval


def _record(
    *,
    condition: str,
    source_index: int,
    row_sha256: str,
    prediction: Any,
    split: str,
    gold: list[int] | None = None,
    donor_source_index: int | None = None,
    donor_row_sha256: str | None = None,
    shuffled_source_index: int | None = None,
    shuffled_row_sha256: str | None = None,
) -> dict[str, Any]:
    gold_payload = {"boundaries": [1] if gold is None else gold}
    return {
        "status": "ok",
        "condition": condition,
        "task": gate.TASK_NAME,
        "split": split,
        "source_index": source_index,
        "row_sha256": row_sha256,
        "gold": gold_payload,
        "parsed_json": prediction,
        "score_strict": state_eval.score_prediction(
            "scene", prediction, gold_payload
        ),
        "score_recovered": state_eval.recovered_scene_score(
            prediction, gold_payload
        ),
        "donor_source_index": donor_source_index,
        "donor_row_sha256": donor_row_sha256,
        "shuffled_source_index": shuffled_source_index,
        "shuffled_row_sha256": shuffled_row_sha256,
    }


def _bundle(*, stage: str, state_prediction: Any | None = None) -> dict[str, list[dict]]:
    if stage == "hard32":
        identities = [
            (index, gate.HARD32_ROW_HASHES[index])
            for index in gate.HARD32_ROW_INDICES
        ]
        split = "val"
    else:
        identities = [(41, "a" * 64), (73, "b" * 64)]
        split = "train"
    selected_indices = [index for index, _ in identities]
    correct = {"boundaries": [1]}
    wrong = {"boundaries": [2]}
    state_prediction = correct if state_prediction is None else state_prediction
    predictions = {
        "base_full": wrong,
        "no_write_full": wrong,
        "normal_full": correct,
        "state_only": state_prediction,
        "state_only_donor": wrong,
        "state_only_no_write": wrong,
    }
    records: dict[str, list[dict]] = {}
    for condition in gate.FOCUSED_CONDITIONS:
        condition_records = []
        for ordinal, (source_index, row_sha256) in enumerate(identities):
            donor_index = selected_indices[(ordinal + 1) % len(selected_indices)]
            condition_records.append(
                _record(
                    condition=condition,
                    source_index=source_index,
                    row_sha256=row_sha256,
                    prediction=predictions[condition],
                    split=split,
                    donor_source_index=(
                        donor_index if condition == "state_only_donor" else None
                    ),
                )
            )
        records[condition] = condition_records
    return records


def _write_results_dir(root: Path, records: dict[str, list[dict]]) -> None:
    root.mkdir()
    conditions = list(gate.FOCUSED_CONDITIONS)
    selection = [
        {"source_index": index, "row_sha256": gate.HARD32_ROW_HASHES[index]}
        for index in gate.HARD32_ROW_INDICES
    ]
    manifest = {
        "fingerprint": "f" * 64,
        "fingerprint_payload": {
            "task": gate.TASK_NAME,
            "split": "val",
            "conditions": conditions,
            "dataset_file": str(root / "val.jsonl"),
            "dataset_sha256": gate.OFFICIAL_SCENE_V4_VAL_SHA256,
            "selection": selection,
        },
        "evaluation_contract": {
            "name": "scene_v6_identity_hard32",
            "task": gate.TASK_NAME,
            "split": "val",
            "rows": 32,
            "conditions": conditions,
        },
    }
    summary_conditions = {}
    for condition, rows in records.items():
        path = root / f"{condition}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        strict_tp = sum(row["score_strict"]["tp"] for row in rows)
        strict_fp = sum(row["score_strict"]["fp"] for row in rows)
        strict_fn = sum(row["score_strict"]["fn"] for row in rows)
        denominator = 2 * strict_tp + strict_fp + strict_fn
        summary_conditions[condition] = {
            "strict": {
                "primary_metric": (
                    0.0 if denominator == 0 else 2 * strict_tp / denominator
                )
            }
        }
    summary = {
        "complete": True,
        "task": gate.TASK_NAME,
        "split": "val",
        "selected_source_indices": list(gate.HARD32_ROW_INDICES),
        "conditions": summary_conditions,
        "semantic_decision_evidence": {"positive_rows": 32},
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _protected_bundle(*, stage: str) -> dict[str, list[dict[str, Any]]]:
    if stage == "hard32":
        identities = [
            (index, gate.HARD32_ROW_HASHES[index])
            for index in gate.HARD32_ROW_INDICES
        ]
        split = "val"
    else:
        identities = [(index, f"{index + 1:064x}") for index in range(32)]
        split = "train"
    records: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in state_eval.SCENE_FOCUSED_CONDITIONS
    }
    for ordinal, (source_index, row_sha256) in enumerate(identities):
        gold = [1] if ordinal % 2 == 0 else [2]
        correct = {"boundaries": gold}
        wrong = {"boundaries": [2] if gold == [1] else [1]}
        donor_index, donor_hash = identities[ordinal ^ 1]
        shuffled_index, shuffled_hash = identities[(ordinal + 3) % len(identities)]
        for condition in state_eval.SCENE_FOCUSED_CONDITIONS:
            prediction = correct if condition in {"normal_full", "state_only"} else wrong
            records[condition].append(
                _record(
                    condition=condition,
                    source_index=source_index,
                    row_sha256=row_sha256,
                    prediction=prediction,
                    split=split,
                    gold=gold,
                    donor_source_index=(
                        donor_index if condition == "state_only_donor" else None
                    ),
                    donor_row_sha256=(
                        donor_hash if condition == "state_only_donor" else None
                    ),
                    shuffled_source_index=(
                        shuffled_index
                        if condition == "state_only_shuffled"
                        else None
                    ),
                    shuffled_row_sha256=(
                        shuffled_hash
                        if condition == "state_only_shuffled"
                        else None
                    ),
                )
            )
    return records


def _write_protected_results_dir(
    root: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    records = _protected_bundle(stage=stage)
    root.mkdir()
    reference = records["base_full"]
    selection = [
        {"source_index": row["source_index"], "row_sha256": row["row_sha256"]}
        for row in reference
    ]
    donor_mapping = []
    shuffled_mapping = []
    for ordinal, row in enumerate(reference):
        donor = reference[ordinal ^ 1]
        shuffled = reference[(ordinal + 3) % len(reference)]
        donor_mapping.append(
            {
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "donor_source_index": donor["source_index"],
                "donor_row_sha256": donor["row_sha256"],
            }
        )
        shuffled_mapping.append(
            {
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "shuffled_source_index": shuffled["source_index"],
                "shuffled_row_sha256": shuffled["row_sha256"],
            }
        )
    split = "train" if stage == "train_overfit" else "val"
    dataset_file = root / f"{split}.jsonl"
    dataset_sha256 = (
        "d" * 64
        if stage == "train_overfit"
        else gate.OFFICIAL_SCENE_V4_VAL_SHA256
    )
    contract: dict[str, Any] = {
        "name": (
            state_eval.SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT
            if stage == "train_overfit"
            else state_eval.SCENE_HARD_FAILURE_HARD32_CONTRACT
        ),
        "task": gate.TASK_NAME,
        "split": split,
        "rows": 32,
        "conditions": list(state_eval.SCENE_FOCUSED_CONDITIONS),
    }
    if stage == "train_overfit":
        contract["train_source"] = {
            "dataset": {
                "path": str(dataset_file),
                "sha256": dataset_sha256,
            },
            "selection": selection,
        }
    else:
        contract["train_selection_authorization"] = {
            "hard32_authorized": True,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        }
    fingerprint = {
        "task": gate.TASK_NAME,
        "split": split,
        "conditions": list(state_eval.SCENE_FOCUSED_CONDITIONS),
        "dataset_file": str(dataset_file),
        "dataset_sha256": dataset_sha256,
        "selection": selection,
        "state_only_donor_mapping": donor_mapping,
        "state_only_shuffled_mapping": shuffled_mapping,
        "evaluation_contract": contract,
    }
    manifest = {
        "fingerprint": state_eval.fingerprint_payload_sha256(fingerprint),
        "fingerprint_payload": fingerprint,
        "evaluation_contract": contract,
    }
    summary_conditions: dict[str, Any] = {}
    for condition, rows in records.items():
        path = root / f"{condition}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        tp = sum(row["score_strict"]["tp"] for row in rows)
        fp = sum(row["score_strict"]["fp"] for row in rows)
        fn = sum(row["score_strict"]["fn"] for row in rows)
        summary_conditions[condition] = {
            "strict": {
                "primary_metric": 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            }
        }
    summary = {
        "complete": True,
        "task": gate.TASK_NAME,
        "split": split,
        "selected_source_indices": [row["source_index"] for row in reference],
        "conditions": summary_conditions,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {"records": records, "manifest": manifest, "summary": summary}


def test_hard32_gate_requires_task_recovery_and_state_causality() -> None:
    report = gate.build_focused_recovery_gate(
        stage="hard32",
        records_by_condition=_bundle(stage="hard32"),
    )

    assert report["status"] == "pass"
    assert report["criterion"]["primary_metric"] == (
        "dataset_native_strict_boundaries_micro_f1"
    )
    assert report["base_failure_cohort"]["rows"] == 32
    assert len(report["base_failure_cohort"]["normal_full_recoveries"]) == 32
    assert len(report["base_failure_cohort"]["state_only_recoveries"]) == 32
    assert len(
        report["base_failure_cohort"]["write_specific_normal_recoveries"]
    ) == 32
    assert len(
        report["base_failure_cohort"]["identity_specific_state_recoveries"]
    ) == 32
    assert report["shuffled_state_control"]["condition"] == "state_only_donor"
    assert report["authorization"]["full_validation_authorized"] is False
    assert report["authorization"]["test_authorized"] is False


def test_semantic_nll_cannot_replace_failed_generation() -> None:
    records = _bundle(stage="hard32")
    for condition in ("normal_full", "state_only"):
        for record in records[condition]:
            prediction = {"boundaries": [2]}
            record["parsed_json"] = prediction
            record["score_strict"] = state_eval.score_prediction(
                "scene", prediction, record["gold"]
            )
            record["score_recovered"] = state_eval.recovered_scene_score(
                prediction, record["gold"]
            )

    report = gate.build_focused_recovery_gate(
        stage="hard32",
        records_by_condition=records,
        diagnostics={
            "semantic_decision_evidence": {
                "correct_better_than_donor_rows": 32,
                "correct_better_than_zero_rows": 32,
            },
            "training_loss": 0.01,
        },
    )

    assert report["status"] == "fail"
    assert report["all_gates_passed"] is False
    assert report["criterion"]["loss_logit_or_semantic_nll_can_satisfy_gate"] is False
    assert report["diagnostics_only"]["can_satisfy_gate"] is False
    assert report["gates"]["state_only_recovers_base_failures"]["passed"] is False


def test_recovered_parser_success_is_secondary_and_cannot_bypass_schema() -> None:
    noncanonical = [{"boundaries": ["P1"]}]
    records = _bundle(stage="hard32", state_prediction=noncanonical)
    report = gate.build_focused_recovery_gate(
        stage="hard32",
        records_by_condition=records,
    )

    state = report["condition_scores"]["state_only"]
    assert state["primary_metric"] == 0.0
    assert state["format"]["recovered_micro_f1_diagnostic"] == 1.0
    assert state["format"]["canonical_outputs"] == 0
    assert report["gates"]["state_only_canonical_output_coverage"]["passed"] is False
    assert report["status"] == "fail"


def test_train_overfit_pass_remains_diagnostic_and_authorizes_nothing() -> None:
    report = gate.build_focused_recovery_gate(
        stage="train_overfit",
        records_by_condition=_protected_bundle(stage="train_overfit"),
    )

    assert report["status"] == "diagnostic_pass"
    assert len(
        report["same_cardinality_reciprocal_switches"]["switched_pairs"]
    ) == 16
    assert report["authorization"] == {
        "hard32_authorized_by_this_report": False,
        "full_validation_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
        "reason": report["authorization"]["reason"],
    }


def test_train_overfit_requires_a_reciprocal_pair_to_switch_both_directions() -> None:
    records = _protected_bundle(stage="train_overfit")
    for ordinal in range(1, 32, 2):
        record = records["state_only"][ordinal]
        wrong = {"boundaries": [1] if record["gold"]["boundaries"] != [1] else [2]}
        record["parsed_json"] = wrong
        record["score_strict"] = state_eval.score_prediction(
            "scene", wrong, record["gold"]
        )
        record["score_recovered"] = state_eval.recovered_scene_score(
            wrong, record["gold"]
        )

    report = gate.build_focused_recovery_gate(
        stage="train_overfit",
        records_by_condition=records,
    )

    pair_gate = report["gates"][
        "same_cardinality_reciprocal_pair_switches_both_directions"
    ]
    assert pair_gate["passed"] is False
    assert report["same_cardinality_reciprocal_switches"]["switched_pairs"] == []
    assert report["status"] == "diagnostic_fail"


def test_train_overfit_rejects_validation_records() -> None:
    records = _bundle(stage="train_overfit")
    records["state_only"][0]["split"] = "val"

    with pytest.raises(gate.FocusedRecoveryContractError, match="record split"):
        gate.build_focused_recovery_gate(
            stage="train_overfit",
            records_by_condition=records,
        )


def test_gate_recomputes_scores_and_rejects_drift() -> None:
    records = _bundle(stage="train_overfit")
    records["normal_full"][0]["score_strict"]["tp"] = 999

    with pytest.raises(gate.FocusedRecoveryContractError, match="strict score differs"):
        gate.build_focused_recovery_gate(
            stage="train_overfit",
            records_by_condition=records,
        )


def test_gate_requires_all_six_conditions() -> None:
    records = _bundle(stage="train_overfit")
    del records["no_write_full"]

    with pytest.raises(gate.FocusedRecoveryContractError, match="six focused"):
        gate.build_focused_recovery_gate(
            stage="train_overfit",
            records_by_condition=records,
        )


def test_results_dir_binds_exact_hard32_and_recomputes_summary(tmp_path: Path) -> None:
    records = _bundle(stage="hard32")
    results_dir = tmp_path / "hard32"
    _write_results_dir(results_dir, records)

    report = gate.analyze_results_dir(results_dir, stage="hard32")

    assert report["status"] == "pass"
    assert report["input"]["evaluation_fingerprint"] == "f" * 64
    assert report["source_indices"] == list(gate.HARD32_ROW_INDICES)


def test_results_dir_rejects_hard32_selection_drift(tmp_path: Path) -> None:
    results_dir = tmp_path / "hard32"
    _write_results_dir(results_dir, _bundle(stage="hard32"))
    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint_payload"]["selection"][0]["row_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(gate.FocusedRecoveryContractError, match="selection differs"):
        gate.analyze_results_dir(results_dir, stage="hard32")


def test_protected_train_overfit_binds_exact_train32_and_seven_conditions(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "train32"
    _write_protected_results_dir(results_dir, stage="train_overfit")

    report = gate.analyze_results_dir(results_dir, stage="train_overfit")

    assert report["status"] == "diagnostic_pass"
    assert report["source_indices"] == list(range(32))
    assert list(report["condition_scores"]) == list(
        state_eval.SCENE_FOCUSED_CONDITIONS
    )
    assert report["shuffled_state_control"]["separate_shuffle_evaluated"] is True


def test_protected_hard32_requires_exact_seven_condition_contract(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "focused_hard32"
    _write_protected_results_dir(results_dir, stage="hard32")

    report = gate.analyze_results_dir(results_dir, stage="hard32")

    assert report["status"] == "pass"
    assert report["source_indices"] == list(gate.HARD32_ROW_INDICES)
    assert report["gates"][
        "correct_state_lifts_base_failure_f1_over_shuffled"
    ]["passed"] is True


def test_protected_results_reject_donor_mapping_record_drift(tmp_path: Path) -> None:
    results_dir = tmp_path / "train32"
    _write_protected_results_dir(results_dir, stage="train_overfit")
    donor_path = results_dir / "state_only_donor.jsonl"
    rows = [json.loads(line) for line in donor_path.read_text().splitlines()]
    rows[0]["donor_source_index"] = rows[2]["source_index"]
    donor_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        gate.FocusedRecoveryContractError,
        match="state_only_donor record differs",
    ):
        gate.analyze_results_dir(results_dir, stage="train_overfit")


def test_protected_results_reject_shuffled_mapping_record_drift(tmp_path: Path) -> None:
    results_dir = tmp_path / "train32"
    _write_protected_results_dir(results_dir, stage="train_overfit")
    shuffled_path = results_dir / "state_only_shuffled.jsonl"
    rows = [json.loads(line) for line in shuffled_path.read_text().splitlines()]
    rows[0]["shuffled_row_sha256"] = "0" * 64
    shuffled_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(
        gate.FocusedRecoveryContractError,
        match="state_only_shuffled record differs",
    ):
        gate.analyze_results_dir(results_dir, stage="train_overfit")
