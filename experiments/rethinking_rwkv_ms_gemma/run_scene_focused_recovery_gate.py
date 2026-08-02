#!/usr/bin/env python3
"""Gate one scene-memory experiment on benchmark recovery and state causality.

This module does not load a model. It consumes the historical six-condition
bundle or the protected experiment's seven-condition bundle from
``run_scene_state_eval.py`` and recomputes every score from the recorded
generation. Keeping generation in the existing evaluator avoids a second,
quietly divergent implementation of state priming or decoding.

Two evidence stages are supported:

* ``train_overfit`` proves only that a checkpoint can fit explicitly selected
  official-train failures. It never authorizes access to validation or test.
* ``hard32`` accepts only the frozen scene-v4-current Hard32 validation rows.
  It reports the held-out result but never authorizes full-170 validation,
  test, or another benchmark.

The primary task metric is the dataset-native strict boundary micro-F1: the
prediction must be a JSON object and its literal ``boundaries`` values are
compared as a set. Conservative format recovery is secondary telemetry.
Canonical formatting is also a separate required gate. Semantic NLL, loss,
and logit gaps are copied as diagnostics only and cannot satisfy a gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (  # noqa: E402
    HARD32_ROW_INDICES,
    HARD32_ROW_HASHES,
    OFFICIAL_SCENE_V4_VAL_SHA256,
    SCENE_FOCUSED_CONDITIONS as PROTECTED_FOCUSED_CONDITIONS,
    SCENE_HARD_FAILURE_HARD32_CONTRACT,
    SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
    SCENE_HARD_FAILURE_ROWS,
    TASK_NAME,
    fingerprint_payload_sha256,
    is_canonical_scene_prediction,
    recovered_scene_score,
    score_prediction,
    write_json_atomic,
)


SCHEMA = "rwkv_ms_scene_focused_recovery_gate.v1"
STAGES = ("train_overfit", "hard32")
PRIMARY_METRIC = "dataset_native_strict_boundaries_micro_f1"
FORMAT_GATE_RATIO = 31 / 32
HARD32_MIN_F1_LIFT = 0.05
HARD32_MIN_RECOVERY_ADVANTAGE = 3
MAX_PREDICTED_TO_GOLD_BOUNDARY_RATIO = 2.0
FOCUSED_CONDITIONS = (
    "base_full",
    "no_write_full",
    "normal_full",
    "state_only",
    "state_only_donor",
    "state_only_no_write",
)
OPTIONAL_SHUFFLED_CONDITION = "state_only_shuffled"
FORMAT_GATE_CONDITIONS = (
    "no_write_full",
    "normal_full",
    "state_only",
    "state_only_donor",
    "state_only_no_write",
)
HARD32_CONTRACT_NAMES = {
    "scene_v6_identity_hard32",
    "scene_v8_authorized_hard32",
    "scene_v14_authorized_hard32",
    "scene_v15_authorized_hard32",
    SCENE_HARD_FAILURE_HARD32_CONTRACT,
}
PROTECTED_CONTRACT_NAMES = {
    SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
    SCENE_HARD_FAILURE_HARD32_CONTRACT,
}
HARD32_BASE_SUCCESS_SENTINEL_IDENTITIES = (
    (
        56,
        "8c856ddd10d3f2172a7d4b87a8ab653c4b31d78ff3f29b91b299450e110a1375",
    ),
)
HARD32_BASE_FAILURE_IDENTITIES = tuple(
    (index, HARD32_ROW_HASHES[index])
    for index in HARD32_ROW_INDICES
    if index != HARD32_BASE_SUCCESS_SENTINEL_IDENTITIES[0][0]
)


def _valid_condition_set(conditions: Iterable[str]) -> bool:
    observed = set(conditions)
    required = set(FOCUSED_CONDITIONS)
    return observed == required or observed == required | {OPTIONAL_SHUFFLED_CONDITION}


class FocusedRecoveryContractError(ValueError):
    """Raised when an input could change the meaning of the focused gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedRecoveryContractError(message)


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FocusedRecoveryContractError(
            f"invalid JSON in {description}: {path}"
        ) from exc
    _require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def _load_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            _require(bool(raw.strip()), f"blank row in {description}:{line_number}")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise FocusedRecoveryContractError(
                    f"invalid JSON in {description}:{line_number}"
                ) from exc
            _require(
                isinstance(record, dict),
                f"row in {description}:{line_number} must be an object",
            )
            records.append(record)
    return records


def _condition_list(payload: Mapping[str, Any]) -> list[str] | None:
    direct = payload.get("conditions")
    if isinstance(direct, list):
        return [str(item) for item in direct]
    runtime = payload.get("runtime")
    if isinstance(runtime, Mapping) and isinstance(runtime.get("conditions"), list):
        return [str(item) for item in runtime["conditions"]]
    return None


def _manifest_fingerprint_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("fingerprint_payload")
    _require(isinstance(payload, Mapping), "manifest fingerprint_payload is missing")
    return payload


def _validate_manifest(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    fingerprint = _manifest_fingerprint_payload(manifest)
    split = fingerprint.get("split")
    expected_split = "train" if stage == "train_overfit" else "val"
    _require(split == expected_split, f"{stage} manifest split must be {expected_split}")

    task = fingerprint.get("task", summary.get("task", TASK_NAME))
    _require(task == TASK_NAME, f"focused gate supports only {TASK_NAME}")
    conditions = _condition_list(fingerprint)
    _require(conditions is not None, "manifest condition list is missing")
    contract = manifest.get("evaluation_contract")
    _require(isinstance(contract, Mapping), "evaluation contract is missing")
    contract_name = contract.get("name")
    protected_contract = contract_name in PROTECTED_CONTRACT_NAMES
    if protected_contract:
        _require(
            fingerprint.get("evaluation_contract") == contract,
            "protected evaluation contract differs from fingerprint payload",
        )
        _require(
            manifest.get("fingerprint")
            == fingerprint_payload_sha256(dict(fingerprint)),
            "protected evaluation fingerprint differs from payload",
        )
    _require(
        len(conditions) == len(set(conditions))
        and (
            conditions == list(PROTECTED_FOCUSED_CONDITIONS)
            if protected_contract
            else _valid_condition_set(conditions)
        ),
        "focused gate requires the six scene-state conditions and permits one "
        "optional shuffled-state condition",
    )

    dataset_file = fingerprint.get("dataset_file")
    _require(isinstance(dataset_file, str), "manifest dataset_file is missing")
    dataset_name = Path(dataset_file).name.lower()
    if stage == "train_overfit":
        _require(
            contract_name == SCENE_HARD_FAILURE_TRAIN_OVERFIT_CONTRACT,
            "train-overfit result was not produced under the protected train contract",
        )
        _require(
            not any(token in dataset_name for token in ("val", "test", "holdout")),
            "train-overfit input cannot name validation, test, or holdout data",
        )
        selection = fingerprint.get("selection")
        _require(
            isinstance(selection, list)
            and len(selection) == SCENE_HARD_FAILURE_ROWS
            and [row.get("source_index") for row in selection]
            == list(range(SCENE_HARD_FAILURE_ROWS)),
            "train-overfit manifest selection must be exact local rows 0..31",
        )
        _require(contract.get("task") == TASK_NAME, "train-overfit contract task differs")
        _require(contract.get("split") == "train", "train-overfit contract split differs")
        _require(
            contract.get("rows") == SCENE_HARD_FAILURE_ROWS,
            "train-overfit contract row count differs",
        )
        _require(
            contract.get("conditions") == list(PROTECTED_FOCUSED_CONDITIONS),
            "train-overfit contract conditions differ",
        )
        train_source = contract.get("train_source")
        _require(isinstance(train_source, Mapping), "train-overfit source binding is missing")
        source_selection = train_source.get("selection")
        source_dataset = train_source.get("dataset")
        _require(
            source_selection == selection,
            "train-overfit manifest selection differs from source manifest",
        )
        _require(
            isinstance(source_dataset, Mapping)
            and source_dataset.get("path") == dataset_file
            and source_dataset.get("sha256") == fingerprint.get("dataset_sha256"),
            "train-overfit dataset differs from source-manifest binding",
        )
    else:
        _require(dataset_name == "val.jsonl", "Hard32 requires official val.jsonl")
        _require(
            fingerprint.get("dataset_sha256") == OFFICIAL_SCENE_V4_VAL_SHA256,
            "Hard32 official validation SHA-256 differs",
        )
        selection = fingerprint.get("selection")
        _require(isinstance(selection, list), "Hard32 manifest selection is missing")
        expected_selection = [
            {"source_index": index, "row_sha256": HARD32_ROW_HASHES[index]}
            for index in HARD32_ROW_INDICES
        ]
        _require(selection == expected_selection, "Hard32 manifest selection differs")
        _require(
            contract.get("name") in HARD32_CONTRACT_NAMES,
            "Hard32 result was not produced under a protected Hard32 contract",
        )
        _require(contract.get("task") == TASK_NAME, "Hard32 contract task differs")
        _require(contract.get("split") == "val", "Hard32 contract split differs")
        _require(contract.get("rows") == 32, "Hard32 contract row count differs")
        contract_conditions = contract.get("conditions")
        _require(
            isinstance(contract_conditions, list)
            and len(contract_conditions) == len(set(contract_conditions))
            and (
                contract_conditions == list(PROTECTED_FOCUSED_CONDITIONS)
                if contract_name == SCENE_HARD_FAILURE_HARD32_CONTRACT
                else _valid_condition_set(str(item) for item in contract_conditions)
            ),
            "Hard32 contract conditions differ",
        )
        if contract_name == SCENE_HARD_FAILURE_HARD32_CONTRACT:
            authorization = contract.get("train_selection_authorization")
            _require(
                isinstance(authorization, Mapping)
                and authorization.get("hard32_authorized") is True
                and authorization.get("full170_authorized") is False
                and authorization.get("test_authorized") is False
                and authorization.get("other_benchmarks_authorized") is False,
                "focused Hard32 train-only authorization differs",
            )

    _require(summary.get("complete") is True, "evaluation summary is incomplete")
    summary_conditions = summary.get("conditions")
    _require(isinstance(summary_conditions, Mapping), "summary conditions are missing")
    _require(
        (
            len(summary_conditions) == len(PROTECTED_FOCUSED_CONDITIONS)
            and set(summary_conditions) == set(PROTECTED_FOCUSED_CONDITIONS)
            if protected_contract
            else _valid_condition_set(str(item) for item in summary_conditions)
        )
        and set(summary_conditions) == set(conditions),
        "summary condition set differs from the focused manifest",
    )


def _validate_and_recompute_record(
    record: Mapping[str, Any],
    *,
    condition: str,
    expected_split: str,
) -> dict[str, Any]:
    _require(record.get("status") == "ok", f"{condition} record status is not ok")
    _require(record.get("condition") == condition, f"{condition} record condition differs")
    _require(record.get("task") == TASK_NAME, f"{condition} record task differs")
    _require(record.get("split") == expected_split, f"{condition} record split differs")
    source_index = record.get("source_index", record.get("line_index"))
    _require(
        isinstance(source_index, int) and not isinstance(source_index, bool),
        f"{condition} source index is invalid",
    )
    row_sha256 = record.get("row_sha256")
    _require(
        isinstance(row_sha256, str) and len(row_sha256) == 64,
        f"{condition} row SHA-256 is invalid",
    )
    gold = record.get("gold")
    _require(isinstance(gold, Mapping), f"{condition} gold is missing")
    parsed = record.get("parsed_json")
    strict = score_prediction("scene", parsed, gold)
    recovered = recovered_scene_score(parsed, gold)
    _require(
        record.get("score_strict") == strict,
        f"{condition} strict score differs from recorded generation",
    )
    _require(
        record.get("score_recovered") == recovered,
        f"{condition} recovered score differs from recorded generation",
    )
    result = dict(record)
    result["source_index"] = source_index
    result["score_strict"] = strict
    result["score_recovered"] = recovered
    return result


def validate_records(
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    stage: str,
) -> dict[str, list[dict[str, Any]]]:
    _require(stage in STAGES, f"unsupported focused-gate stage: {stage}")
    _require(
        _valid_condition_set(str(item) for item in records_by_condition),
        "record bundle must contain the six focused conditions and may include "
        "state_only_shuffled",
    )
    split = "train" if stage == "train_overfit" else "val"
    normalized: dict[str, list[dict[str, Any]]] = {}
    condition_order = list(FOCUSED_CONDITIONS)
    if OPTIONAL_SHUFFLED_CONDITION in records_by_condition:
        condition_order.append(OPTIONAL_SHUFFLED_CONDITION)
    for condition in condition_order:
        records = [
            _validate_and_recompute_record(
                record,
                condition=condition,
                expected_split=split,
            )
            for record in records_by_condition[condition]
        ]
        _require(bool(records), f"{condition} has no records")
        indices = [int(record["source_index"]) for record in records]
        _require(len(indices) == len(set(indices)), f"{condition} has duplicate rows")
        normalized[condition] = records

    reference = normalized["base_full"]
    reference_identity = [
        (record["source_index"], record["row_sha256"], record["gold"])
        for record in reference
    ]
    for condition in condition_order[1:]:
        identity = [
            (record["source_index"], record["row_sha256"], record["gold"])
            for record in normalized[condition]
        ]
        _require(identity == reference_identity, f"{condition} row order or identity differs")

    if stage == "hard32":
        expected_identity = [
            (index, HARD32_ROW_HASHES[index]) for index in HARD32_ROW_INDICES
        ]
        actual_identity = [
            (int(record["source_index"]), str(record["row_sha256"]))
            for record in reference
        ]
        _require(actual_identity == expected_identity, "record bundle is not frozen Hard32")
    else:
        _require(len(reference) <= 32, "train-overfit diagnostic is capped at 32 rows")

    for record in normalized["state_only_donor"]:
        donor_index = record.get("donor_source_index")
        _require(
            isinstance(donor_index, int)
            and not isinstance(donor_index, bool)
            and donor_index != record["source_index"],
            "state_only_donor must carry a distinct predeclared donor state",
        )
    for record in normalized.get(OPTIONAL_SHUFFLED_CONDITION, []):
        shuffled_index = record.get(
            "shuffled_source_index", record.get("donor_source_index")
        )
        _require(
            isinstance(shuffled_index, int)
            and not isinstance(shuffled_index, bool)
            and shuffled_index != record["source_index"],
            "state_only_shuffled must carry a distinct predeclared shuffled state",
        )
    return normalized


def _validate_manifest_state_mappings(
    *,
    fingerprint: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    require_shuffled: bool,
    require_donor: bool,
) -> None:
    reference = records["base_full"]
    reference_by_source = {
        int(record["source_index"]): record for record in reference
    }

    donor_rows = fingerprint.get("state_only_donor_mapping")
    if donor_rows is None and not require_donor:
        return
    _require(isinstance(donor_rows, list), "manifest donor mapping is missing")
    _require(
        len(donor_rows) == len(reference),
        "manifest donor mapping is incomplete",
    )
    donor_by_source: dict[int, Mapping[str, Any]] = {}
    donor_targets: set[int] = set()
    for row in donor_rows:
        _require(isinstance(row, Mapping), "manifest donor mapping row is invalid")
        source_index = row.get("source_index")
        donor_index = row.get("donor_source_index")
        _require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and isinstance(donor_index, int)
            and not isinstance(donor_index, bool)
            and source_index in reference_by_source
            and donor_index in reference_by_source
            and source_index != donor_index
            and source_index not in donor_by_source
            and donor_index not in donor_targets,
            "manifest donor mapping is not a complete distinct bijection",
        )
        source = reference_by_source[source_index]
        donor = reference_by_source[donor_index]
        _require(
            row.get("row_sha256") == source["row_sha256"]
            and row.get("donor_row_sha256") == donor["row_sha256"]
            and source["gold"] != donor["gold"],
            "manifest donor mapping identity differs",
        )
        donor_by_source[source_index] = row
        donor_targets.add(donor_index)
    _require(
        set(donor_by_source) == set(reference_by_source)
        and donor_targets == set(reference_by_source),
        "manifest donor mapping coverage differs",
    )
    donor_records = {
        int(record["source_index"]): record
        for record in records["state_only_donor"]
    }
    for source_index, mapping in donor_by_source.items():
        record = donor_records[source_index]
        _require(
            record.get("donor_source_index") == mapping["donor_source_index"]
            and record.get("donor_row_sha256") == mapping["donor_row_sha256"],
            "state_only_donor record differs from manifest mapping",
        )

    shuffled_rows = fingerprint.get("state_only_shuffled_mapping")
    if shuffled_rows is None and not require_shuffled:
        return
    _require(isinstance(shuffled_rows, list), "manifest shuffled mapping is missing")
    _require(
        len(shuffled_rows) == len(reference),
        "manifest shuffled mapping is incomplete",
    )
    shuffled_by_source: dict[int, Mapping[str, Any]] = {}
    shuffled_targets: set[int] = set()
    for row in shuffled_rows:
        _require(isinstance(row, Mapping), "manifest shuffled mapping row is invalid")
        source_index = row.get("source_index")
        shuffled_index = row.get("shuffled_source_index")
        _require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and isinstance(shuffled_index, int)
            and not isinstance(shuffled_index, bool)
            and source_index in reference_by_source
            and shuffled_index in reference_by_source
            and source_index != shuffled_index
            and source_index not in shuffled_by_source
            and shuffled_index not in shuffled_targets,
            "manifest shuffled mapping is not a complete distinct bijection",
        )
        source = reference_by_source[source_index]
        shuffled = reference_by_source[shuffled_index]
        locked_donor = donor_by_source[source_index]["donor_source_index"]
        _require(
            shuffled_index != locked_donor
            and row.get("row_sha256") == source["row_sha256"]
            and row.get("shuffled_row_sha256") == shuffled["row_sha256"]
            and source["gold"] != shuffled["gold"],
            "manifest shuffled mapping identity differs",
        )
        shuffled_by_source[source_index] = row
        shuffled_targets.add(shuffled_index)
    _require(
        set(shuffled_by_source) == set(reference_by_source)
        and shuffled_targets == set(reference_by_source),
        "manifest shuffled mapping coverage differs",
    )
    shuffled_records = {
        int(record["source_index"]): record
        for record in records[OPTIONAL_SHUFFLED_CONDITION]
    }
    for source_index, mapping in shuffled_by_source.items():
        record = shuffled_records[source_index]
        _require(
            record.get("shuffled_source_index")
            == mapping["shuffled_source_index"]
            and record.get("shuffled_row_sha256")
            == mapping["shuffled_row_sha256"],
            "state_only_shuffled record differs from manifest mapping",
        )


def _micro_score(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    rows = list(records)
    tp = sum(int(record[field]["tp"]) for record in rows)
    fp = sum(int(record[field]["fp"]) for record in rows)
    fn = sum(int(record[field]["fn"]) for record in rows)
    denominator = 2 * tp + fp + fn
    return {
        "rows": len(rows),
        "micro_f1": 0.0 if denominator == 0 else (2 * tp) / denominator,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _strict_exact(record: Mapping[str, Any]) -> bool:
    score = record["score_strict"]
    return bool(
        score.get("schema_valid") is True
        and int(score.get("fp", 0)) == 0
        and int(score.get("fn", 0)) == 0
    )


def _frozen_base_outcome_exact(record: Mapping[str, Any]) -> bool:
    """Match the recovered-boundary evidence used to freeze Hard32 outcomes."""

    score = record["score_recovered"]
    gold = score.get("gold_boundaries")
    predicted = score.get("predicted_boundaries")
    return isinstance(gold, list) and isinstance(predicted, list) and predicted == gold


def _train_reciprocal_same_cardinality_switches(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Find value pairs whose correct state succeeds in both directions."""

    reference_by_source = {
        int(record["source_index"]): record for record in records["base_full"]
    }
    state_by_source = {
        int(record["source_index"]): record for record in records["state_only"]
    }
    donor_state_by_source = {
        int(record["source_index"]): record
        for record in records["state_only_donor"]
    }
    donor_by_source = {
        source_index: record.get("donor_source_index")
        for source_index, record in donor_state_by_source.items()
    }
    eligible: list[list[int]] = []
    switched: list[list[int]] = []
    visited: set[int] = set()
    for source_index in sorted(reference_by_source):
        if source_index in visited:
            continue
        donor_index = donor_by_source.get(source_index)
        if (
            not isinstance(donor_index, int)
            or isinstance(donor_index, bool)
            or donor_index not in reference_by_source
            or donor_by_source.get(donor_index) != source_index
        ):
            continue
        visited.update((source_index, donor_index))
        source_gold = reference_by_source[source_index].get("gold", {}).get(
            "boundaries"
        )
        donor_gold = reference_by_source[donor_index].get("gold", {}).get(
            "boundaries"
        )
        if (
            not isinstance(source_gold, list)
            or not isinstance(donor_gold, list)
            or not source_gold
            or len(source_gold) != len(donor_gold)
            or source_gold == donor_gold
        ):
            continue
        pair = sorted((source_index, donor_index))
        eligible.append(pair)
        if (
            _strict_exact(state_by_source[source_index])
            and _strict_exact(state_by_source[donor_index])
            and not _strict_exact(donor_state_by_source[source_index])
            and not _strict_exact(donor_state_by_source[donor_index])
        ):
            switched.append(pair)
    return {"eligible_pairs": eligible, "switched_pairs": switched}


def _task_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recovered = _micro_score(records, "score_recovered")
    strict = _micro_score(records, "score_strict")
    recovered_outputs = sum(
        record["score_recovered"].get("schema_recovered") is True for record in records
    )
    canonical_outputs = sum(
        is_canonical_scene_prediction(record.get("parsed_json")) for record in records
    )
    return {
        "metric_name": PRIMARY_METRIC,
        "primary_metric": strict["micro_f1"],
        "tp": strict["tp"],
        "fp": strict["fp"],
        "fn": strict["fn"],
        "predicted_boundary_count": strict["tp"] + strict["fp"],
        "gold_boundary_count": strict["tp"] + strict["fn"],
        "predicted_to_gold_boundary_ratio": (
            (strict["tp"] + strict["fp"]) / (strict["tp"] + strict["fn"])
            if strict["tp"] + strict["fn"]
            else 0.0
        ),
        "exact_rows": sum(_strict_exact(record) for record in records),
        "rows": len(records),
        "format": {
            "recovered_outputs": recovered_outputs,
            "recovered_coverage": recovered_outputs / len(records),
            "canonical_outputs": canonical_outputs,
            "canonical_coverage": canonical_outputs / len(records),
            "recovered_micro_f1_diagnostic": recovered["micro_f1"],
            "recovered_tp_diagnostic": recovered["tp"],
            "recovered_fp_diagnostic": recovered["fp"],
            "recovered_fn_diagnostic": recovered["fn"],
        },
    }


def _gate(
    *,
    value: float | int,
    operator: str,
    threshold: float | int,
    category: str,
) -> dict[str, Any]:
    numeric_value = float(value)
    numeric_threshold = float(threshold)
    _require(
        math.isfinite(numeric_value) and math.isfinite(numeric_threshold),
        "gate value is non-finite",
    )
    if operator == ">":
        passed = numeric_value > numeric_threshold
    elif operator == ">=":
        passed = numeric_value >= numeric_threshold
    elif operator == "<=":
        passed = numeric_value <= numeric_threshold
    else:
        raise FocusedRecoveryContractError(f"unsupported gate operator: {operator}")
    return {
        "category": category,
        "operator": operator,
        "threshold": threshold,
        "value": value,
        "passed": passed,
    }


def build_focused_recovery_gate(
    *,
    stage: str,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the benchmark-first gate from already generated condition records."""

    records = validate_records(records_by_condition, stage=stage)
    row_count = len(records["base_full"])
    condition_order = (
        list(PROTECTED_FOCUSED_CONDITIONS)
        if OPTIONAL_SHUFFLED_CONDITION in records
        else list(FOCUSED_CONDITIONS)
    )
    summaries = {
        condition: _task_summary(records[condition]) for condition in condition_order
    }
    base_outcome_exact = (
        _frozen_base_outcome_exact if stage == "hard32" else _strict_exact
    )
    base_failure_ordinals = [
        ordinal
        for ordinal, record in enumerate(records["base_full"])
        if not base_outcome_exact(record)
    ]
    base_success_ordinals = [
        ordinal
        for ordinal, record in enumerate(records["base_full"])
        if base_outcome_exact(record)
    ]
    if stage == "hard32":
        observed_failure_identities = tuple(
            (
                int(records["base_full"][ordinal]["source_index"]),
                str(records["base_full"][ordinal]["row_sha256"]),
            )
            for ordinal in base_failure_ordinals
        )
        observed_success_identities = tuple(
            (
                int(records["base_full"][ordinal]["source_index"]),
                str(records["base_full"][ordinal]["row_sha256"]),
            )
            for ordinal in base_success_ordinals
        )
        _require(
            observed_failure_identities == HARD32_BASE_FAILURE_IDENTITIES,
            "Hard32 frozen-base failure identities differ from the authoritative "
            "31-row cohort",
        )
        _require(
            observed_success_identities == HARD32_BASE_SUCCESS_SENTINEL_IDENTITIES,
            "Hard32 frozen-base success sentinel identity differs from the "
            "authoritative one-row cohort",
        )
    _require(base_failure_ordinals, "focused gate requires frozen-base failures")

    failure_records = {
        condition: [records[condition][ordinal] for ordinal in base_failure_ordinals]
        for condition in condition_order
    }
    failure_summaries = {
        condition: _task_summary(failure_records[condition])
        for condition in condition_order
    }

    def exact(condition: str, ordinal: int) -> bool:
        return _strict_exact(records[condition][ordinal])

    normal_recoveries = [
        ordinal for ordinal in base_failure_ordinals if exact("normal_full", ordinal)
    ]
    state_recoveries = [
        ordinal for ordinal in base_failure_ordinals if exact("state_only", ordinal)
    ]
    write_specific_normal_recoveries = [
        ordinal
        for ordinal in normal_recoveries
        if not exact("no_write_full", ordinal)
    ]
    identity_specific_state_recoveries = [
        ordinal
        for ordinal in state_recoveries
        if not exact("state_only_donor", ordinal)
        and not exact("state_only_no_write", ordinal)
        and (
            OPTIONAL_SHUFFLED_CONDITION not in records
            or not exact(OPTIONAL_SHUFFLED_CONDITION, ordinal)
        )
    ]
    no_write_recoveries = [
        ordinal
        for ordinal in base_failure_ordinals
        if exact("no_write_full", ordinal)
    ]
    zero_state_recoveries = [
        ordinal
        for ordinal in base_failure_ordinals
        if exact("state_only_no_write", ordinal)
    ]
    normal_regressions = [
        ordinal
        for ordinal in base_success_ordinals
        if not exact("normal_full", ordinal)
    ]
    state_regressions = [
        ordinal
        for ordinal in base_success_ordinals
        if not exact("state_only", ordinal)
    ]

    def source_indices(ordinals: Iterable[int]) -> list[int]:
        return [int(records["base_full"][ordinal]["source_index"]) for ordinal in ordinals]

    def row_identities(ordinals: Iterable[int]) -> list[dict[str, Any]]:
        return [
            {
                "source_index": int(records["base_full"][ordinal]["source_index"]),
                "row_sha256": str(records["base_full"][ordinal]["row_sha256"]),
            }
            for ordinal in ordinals
        ]

    full_f1 = {
        condition: float(summaries[condition]["primary_metric"])
        for condition in condition_order
    }
    failure_f1 = {
        condition: float(failure_summaries[condition]["primary_metric"])
        for condition in condition_order
    }

    def f1_deltas(f1: Mapping[str, float]) -> dict[str, float]:
        result = {
            "normal_full_minus_base_full": f1["normal_full"] - f1["base_full"],
            "normal_full_minus_no_write_full": (
                f1["normal_full"] - f1["no_write_full"]
            ),
            "state_only_minus_state_only_donor": (
                f1["state_only"] - f1["state_only_donor"]
            ),
            "state_only_minus_state_only_no_write": (
                f1["state_only"] - f1["state_only_no_write"]
            ),
        }
        if OPTIONAL_SHUFFLED_CONDITION in f1:
            result["state_only_minus_state_only_shuffled"] = (
                f1["state_only"] - f1[OPTIONAL_SHUFFLED_CONDITION]
            )
        return result

    full_f1_deltas = f1_deltas(full_f1)
    failure_f1_deltas = f1_deltas(failure_f1)
    deltas = full_f1_deltas if stage == "hard32" else failure_f1_deltas
    f1_lift_threshold = HARD32_MIN_F1_LIFT if stage == "hard32" else 0.0
    f1_lift_operator = ">=" if stage == "hard32" else ">"
    recovery_advantage_threshold = (
        HARD32_MIN_RECOVERY_ADVANTAGE if stage == "hard32" else 0
    )
    recovery_advantage_operator = ">=" if stage == "hard32" else ">"
    reciprocal_switches = (
        _train_reciprocal_same_cardinality_switches(records)
        if stage == "train_overfit"
        else None
    )
    gates: dict[str, dict[str, Any]] = {
        "contains_frozen_base_failures": _gate(
            value=len(base_failure_ordinals),
            operator=">",
            threshold=0,
            category="task",
        ),
        "normal_full_recovers_base_failures": _gate(
            value=len(normal_recoveries),
            operator=">",
            threshold=0,
            category="task",
        ),
        "state_only_recovers_base_failures": _gate(
            value=len(state_recoveries),
            operator=">",
            threshold=0,
            category="task",
        ),
        "normal_recovery_requires_state_writes": _gate(
            value=len(write_specific_normal_recoveries),
            operator=">",
            threshold=0,
            category="causality",
        ),
        "state_recovery_requires_correct_history": _gate(
            value=len(identity_specific_state_recoveries),
            operator=">",
            threshold=0,
            category="causality",
        ),
        "normal_full_lifts_base_failure_f1_over_base": _gate(
            value=deltas["normal_full_minus_base_full"],
            operator=f1_lift_operator,
            threshold=f1_lift_threshold,
            category="task",
        ),
        "normal_full_lifts_base_failure_f1_over_no_write": _gate(
            value=deltas["normal_full_minus_no_write_full"],
            operator=f1_lift_operator,
            threshold=f1_lift_threshold,
            category="causality",
        ),
        "correct_state_lifts_base_failure_f1_over_donor": _gate(
            value=deltas["state_only_minus_state_only_donor"],
            operator=f1_lift_operator,
            threshold=f1_lift_threshold,
            category="causality",
        ),
        "correct_state_lifts_base_failure_f1_over_zero": _gate(
            value=deltas["state_only_minus_state_only_no_write"],
            operator=f1_lift_operator,
            threshold=f1_lift_threshold,
            category="causality",
        ),
        "normal_recovery_advantage_over_no_write": _gate(
            value=len(normal_recoveries) - len(no_write_recoveries),
            operator=recovery_advantage_operator,
            threshold=recovery_advantage_threshold,
            category="causality",
        ),
        "correct_state_recovery_advantage_over_zero": _gate(
            value=len(state_recoveries) - len(zero_state_recoveries),
            operator=recovery_advantage_operator,
            threshold=recovery_advantage_threshold,
            category="causality",
        ),
        "normal_recoveries_exceed_base_success_regressions": _gate(
            value=len(normal_recoveries) - len(normal_regressions),
            operator=">",
            threshold=0,
            category="task",
        ),
        "state_recoveries_exceed_base_success_regressions": _gate(
            value=len(state_recoveries) - len(state_regressions),
            operator=">",
            threshold=0,
            category="task",
        ),
        "normal_full_predicted_boundary_density": _gate(
            value=summaries["normal_full"]["predicted_to_gold_boundary_ratio"],
            operator="<=",
            threshold=MAX_PREDICTED_TO_GOLD_BOUNDARY_RATIO,
            category="format",
        ),
        "state_only_predicted_boundary_density": _gate(
            value=summaries["state_only"]["predicted_to_gold_boundary_ratio"],
            operator="<=",
            threshold=MAX_PREDICTED_TO_GOLD_BOUNDARY_RATIO,
            category="format",
        ),
    }
    if reciprocal_switches is not None:
        gates["same_cardinality_reciprocal_pair_switches_both_directions"] = _gate(
            value=len(reciprocal_switches["switched_pairs"]),
            operator=">",
            threshold=0,
            category="causality",
        )
    if OPTIONAL_SHUFFLED_CONDITION in summaries:
        gates["correct_state_lifts_base_failure_f1_over_shuffled"] = _gate(
            value=deltas["state_only_minus_state_only_shuffled"],
            operator=f1_lift_operator,
            threshold=f1_lift_threshold,
            category="causality",
        )
    format_threshold = math.ceil(row_count * FORMAT_GATE_RATIO)
    format_conditions = list(FORMAT_GATE_CONDITIONS)
    if OPTIONAL_SHUFFLED_CONDITION in summaries:
        format_conditions.append(OPTIONAL_SHUFFLED_CONDITION)
    for condition in format_conditions:
        condition_format = summaries[condition]["format"]
        gates[f"{condition}_dataset_parser_coverage"] = _gate(
            value=int(condition_format["recovered_outputs"]),
            operator=">=",
            threshold=format_threshold,
            category="format",
        )
        gates[f"{condition}_canonical_output_coverage"] = _gate(
            value=int(condition_format["canonical_outputs"]),
            operator=">=",
            threshold=format_threshold,
            category="format",
        )

    passed = all(gate["passed"] for gate in gates.values())
    status = (
        ("diagnostic_pass" if passed else "diagnostic_fail")
        if stage == "train_overfit"
        else ("pass" if passed else "fail")
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "stage": stage,
        "task": TASK_NAME,
        "scope": "scene-v4-current-only",
        "rows": row_count,
        "source_indices": source_indices(range(row_count)),
        "criterion": {
            "primary_metric": PRIMARY_METRIC,
            "primary_cohort": (
                "all_32_frozen_hard32_rows"
                if stage == "hard32"
                else "rows_failed_by_frozen_base_full"
            ),
            "format_gate_ratio": FORMAT_GATE_RATIO,
            "loss_logit_or_semantic_nll_can_satisfy_gate": False,
        },
        "condition_scores": summaries,
        "f1_deltas": {
            "acceptance": deltas,
            "full_cohort": full_f1_deltas,
            "base_failure_diagnostic": failure_f1_deltas,
        },
        "base_failure_cohort": {
            "rows": len(base_failure_ordinals),
            "source_indices": source_indices(base_failure_ordinals),
            "row_identities": row_identities(base_failure_ordinals),
            "condition_scores": failure_summaries,
            "f1_deltas": failure_f1_deltas,
            "normal_full_recoveries": source_indices(normal_recoveries),
            "state_only_recoveries": source_indices(state_recoveries),
            "no_write_full_recoveries": source_indices(no_write_recoveries),
            "zero_state_recoveries": source_indices(zero_state_recoveries),
            "write_specific_normal_recoveries": source_indices(
                write_specific_normal_recoveries
            ),
            "identity_specific_state_recoveries": source_indices(
                identity_specific_state_recoveries
            ),
        },
        "base_success_sentinel": {
            "rows": len(base_success_ordinals),
            "source_indices": source_indices(base_success_ordinals),
            "row_identities": row_identities(base_success_ordinals),
            "normal_full_regressions": source_indices(normal_regressions),
            "state_only_regressions": source_indices(state_regressions),
        },
        "shuffled_state_control": {
            "condition": (
                OPTIONAL_SHUFFLED_CONDITION
                if OPTIONAL_SHUFFLED_CONDITION in summaries
                else "state_only_donor"
            ),
            "kind": (
                "predeclared_distinct_shuffled_state"
                if OPTIONAL_SHUFFLED_CONDITION in summaries
                else "predeclared_distinct_donor_state_permutation"
            ),
            "separate_shuffle_evaluated": OPTIONAL_SHUFFLED_CONDITION in summaries,
            "reason": (
                "When a separate shuffled output is absent, the deterministic, "
                "label-distinct donor permutation is the stronger shuffled-state "
                "control."
            ),
        },
        "same_cardinality_reciprocal_switches": reciprocal_switches,
        "gates": gates,
        "all_gates_passed": passed,
        "diagnostics_only": {
            "strict_metrics": {
                condition: summaries[condition]["format"]
                for condition in condition_order
            },
            "source_summary_diagnostics": dict(diagnostics or {}),
            "can_satisfy_gate": False,
        },
        "authorization": {
            "hard32_authorized_by_this_report": False,
            "full_validation_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
            "reason": (
                "Train-overfit evidence is diagnostic only and cannot open held-out "
                "data. A Hard32 report records an already authorized frozen evaluation "
                "and cannot open a broader split."
            ),
        },
    }
    return report


def analyze_results_dir(results_dir: Path, *, stage: str) -> dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    manifest = _load_json(results_dir / "manifest.json", description="evaluation manifest")
    summary = _load_json(results_dir / "summary.json", description="evaluation summary")
    _validate_manifest(manifest, summary, stage=stage)
    fingerprint = _manifest_fingerprint_payload(manifest)
    manifest_conditions = _condition_list(fingerprint)
    if manifest_conditions is None:
        raise FocusedRecoveryContractError("manifest condition list is missing")
    records = {
        condition: _load_jsonl(
            results_dir / f"{condition}.jsonl",
            description=f"{condition} output",
        )
        for condition in manifest_conditions
    }
    normalized_records = validate_records(records, stage=stage)
    contract = manifest.get("evaluation_contract")
    contract_name = contract.get("name") if isinstance(contract, Mapping) else None
    protected_contract = contract_name in PROTECTED_CONTRACT_NAMES
    _validate_manifest_state_mappings(
        fingerprint=fingerprint,
        records=normalized_records,
        require_donor=protected_contract,
        require_shuffled=protected_contract,
    )
    diagnostics = {
        key: summary.get(key)
        for key in (
            "semantic_decision_evidence",
            "scene_v6_identity_hard32_gate",
            "scene_v15_hard32_gate",
        )
        if summary.get(key) is not None
    }
    report = build_focused_recovery_gate(
        stage=stage,
        records_by_condition=normalized_records,
        diagnostics=diagnostics,
    )

    summary_indices = summary.get("selected_source_indices")
    if stage == "hard32":
        _require(
            summary_indices == list(HARD32_ROW_INDICES),
            "Hard32 summary source indices differ",
        )
    for condition in manifest_conditions:
        recorded = summary["conditions"][condition].get("strict", {})
        recorded_metric = recorded.get("primary_metric")
        _require(
            isinstance(recorded_metric, (int, float))
            and not isinstance(recorded_metric, bool),
            f"summary {condition} strict metric is missing",
        )
        recomputed = report["condition_scores"][condition]["primary_metric"]
        _require(
            math.isclose(float(recorded_metric), float(recomputed), abs_tol=1e-12),
            f"summary {condition} strict metric differs from records",
        )
    report["input"] = {
        "results_dir": str(results_dir),
        "evaluation_fingerprint": manifest.get("fingerprint"),
        "evaluation_contract": manifest.get("evaluation_contract"),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_results_dir(args.results_dir, stage=args.stage)
    output_file = (
        args.output_file.expanduser().resolve()
        if args.output_file is not None
        else args.results_dir.expanduser().resolve() / "focused_recovery_gate.json"
    )
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"focused gate output already exists: {output_file}")
    write_json_atomic(output_file, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
