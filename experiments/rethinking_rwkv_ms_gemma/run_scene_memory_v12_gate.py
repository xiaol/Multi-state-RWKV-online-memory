#!/usr/bin/env python3
"""Run the exact-Value14-only Scene Memory V12 candidate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deltamem import scene_boundary as scene_boundary_metric  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_memory_v10_gate as v10,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v12_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v12_warm_start as v12_warm,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v10_warm_start as v10_warm,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v9_launch_contract as v9_launch,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v9_warm_start as v9_warm,
)


GATE_CONTRACT = "scene_memory_v12_exact_value14_candidate_gate"
GATE_RECORD_SCHEMA = "rwkv_ms_scene_memory_v12_value14_gate_record.v1"
GATE_SUMMARY_SCHEMA = "rwkv_ms_scene_memory_v12_value14_gate_summary.v1"
GATE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v12_value14_gate_manifest.v1"
GATE_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v12_value14_gate_receipt.v1"
CANDIDATE_DESIGNATION_KIND = "scene_memory_v12_value14_candidate_receipt"
CONDITIONS = v10.CONDITIONS
VALUE14_ORDINALS = v10.VALUE14_ORDINALS
VALUE14_SET = frozenset(VALUE14_ORDINALS)
CHECKPOINT_STEPS = launch.CHECKPOINT_STEPS
HARD32_ACCESS_POLICY = launch.HARD32_ACCESS_POLICY
GATE_DEVICE = v10.GATE_DEVICE
GATE_DTYPE = v10.GATE_DTYPE
GATE_ATTN_IMPLEMENTATION = v10.GATE_ATTN_IMPLEMENTATION
GATE_NORMAL_FUSION_PROFILE = v10.GATE_NORMAL_FUSION_PROFILE
GATE_EXPECTED_MEMORY_LAYER_COUNT = v10.GATE_EXPECTED_MEMORY_LAYER_COUNT
GATE_MAX_NEW_TOKENS = v10.GATE_MAX_NEW_TOKENS

V10_DIAGNOSTIC_BASELINE = dict(launch.V10_DIAGNOSTIC_BASELINE_METRICS)
GATE_REQUIREMENTS = {
    "canonical_correct_outputs": 14,
    "correct_strict_exact_rows": 4,
    "donor_identity_strict_exact_rows": 4,
    "correct_strict_micro_f1": 0.3783783783783784,
    "bidirectional_identity_switch_rows": 8,
    "correct_state_beats_donor_state_on_source_token_rows": 14,
    "correct_state_prefers_source_token_rows": 11,
    "donor_state_prefers_donor_token_rows": 11,
    "correct_state_beats_zero_on_source_token_rows": 11,
    "zero_reset_control_is_row_invariant": True,
}
V12_OBJECTIVE = {
    **v10.V10_OBJECTIVE,
    "training_objective_version": launch.OBJECTIVE_VERSION,
    "progression_basis": "two_cycle_value14_semantic_boundary_margin_v1",
    "semantic_margin_mode": launch.SEMANTIC_MARGIN_MODE,
    "semantic_margin": launch.SEMANTIC_MARGIN,
    "parsed_exactness_mode": launch.PARSED_EXACTNESS_MODE,
    "failed_replay_mode": launch.FAILED_REPLAY_MODE,
    "failed_decision_alignment_mode": launch.FAILED_DECISION_ALIGNMENT_MODE,
    "exact_value_mask_mode": launch.EXACT_VALUE_MASK_MODE,
    "exact_retention_scope": launch.EXACT_RETENTION_SCOPE,
    "raw_token_exact_role": "telemetry_only",
    "full_answer_ce_optimization_weight": 0.0,
    "warm_start_source": "pinned_v8_checkpoint56_adapter_only",
    "v10_role": "frozen_diagnostic_only_never_warm_start",
    "evaluation_rows": "exact_value14_ordinals_only",
    "training_continuation": launch.TRAINING_CONTINUATION_POLICY,
}


class V12EvaluationContractError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V12EvaluationContractError(message)


canonical_sha256 = v10.canonical_sha256
self_hash_payload = v10.self_hash_payload
atomic_write_json = v10.atomic_write_json
atomic_write_canonical_json = v10.atomic_write_canonical_json
atomic_write_jsonl = v10.atomic_write_jsonl
_artifact_binding = v10._artifact_binding
_record_with_self_hash = v10._record_with_self_hash


def _run_path(path: Path | str, *, description: str, ssd_root: Path) -> Path:
    try:
        return launch.require_v12_run_path(
            path,
            description=description,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V12EvaluationContractError(
            f"V12 {description} must stay under the locked V12 run root: {exc}"
        ) from exc


def _gate_path(path: Path | str, *, description: str, ssd_root: Path) -> Path:
    try:
        return launch.require_v12_gate_path(
            path,
            description=description,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V12EvaluationContractError(
            f"V12 {description} must stay under the locked V12 gates root: {exc}"
        ) from exc


def validate_base_model_path(
    path: Path | str,
    *,
    ssd_root: Path = launch.SSD_ROOT,
    pinned_base_model: Path = launch.PINNED_BASE_MODEL,
) -> Path:
    try:
        _require(
            pinned_base_model == launch.PINNED_BASE_MODEL,
            "V12 pinned base model override is forbidden",
        )
        baseline = launch.validate_v10_diagnostic_baseline(ssd_root=ssd_root)
        identity = launch.validate_base_model_contract(
            base_model=Path(path),
            baseline=baseline,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V12EvaluationContractError(
            "V12 base model must equal the model and prompt artifacts in the "
            f"pinned V10 manifest: {exc}"
        ) from exc
    return Path(str(identity["path"]))


def _regular_file(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path | None = None,
) -> Path:
    raw = launch._lexically_guard_path(path, description=description)
    resolved = raw.resolve() if ssd_root is None else _gate_path(
        raw,
        description=description,
        ssd_root=ssd_root,
    )
    _require(resolved.is_file(), f"V12 {description} is missing: {resolved}")
    return resolved


def _load_json(path: Path | str, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V12EvaluationContractError(f"V12 {description} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"V12 {description} must be an object")
    return payload


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, description=description)
    records: list[dict[str, Any]] = []
    for row_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        _require(bool(line.strip()), f"V12 {description} contains a blank row")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V12EvaluationContractError(
                f"V12 {description} row {row_number} is invalid JSON"
            ) from exc
        _require(isinstance(payload, dict), f"V12 {description} row must be an object")
        records.append(payload)
    return records


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    recorded = payload.get(field)
    _require(isinstance(recorded, str), f"V12 {field} is missing")
    _require(
        recorded == self_hash_payload(payload, hash_field=field),
        f"V12 {field} differs",
    )
    return recorded


def validate_v12_train_inputs(
    *,
    data_root: Path = launch.DATA_ROOT,
    source_lock_path: Path = launch.SOURCE_LOCK,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    try:
        launch_data = launch.validate_data_contract(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
        base = v10.validate_v10_train_inputs(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
        baseline = launch.validate_v10_diagnostic_baseline(ssd_root=ssd_root)
    except Exception as exc:
        raise V12EvaluationContractError(f"V12 input contract failed: {exc}") from exc
    _require(
        launch_data["two_cycle_pairs"]
        == [list(pair) for pair in launch.TWO_CYCLE_PAIRS]
        and launch_data["two_cycle_pairs_sha256"]
        == launch.TWO_CYCLE_PAIRS_SHA256,
        "V12 two-cycle identity differs",
    )
    result = dict(base)
    result.update(
        {
            "contract": GATE_CONTRACT,
            "launch_data": launch_data,
            "checkpoint_steps": [1, 2],
            "presentation_checkpoint_steps": [7, 14],
            "optimizer_cycles": launch_data["optimizer_cycles"],
            "v10_diagnostic_baseline": baseline,
            "hard32_access": HARD32_ACCESS_POLICY,
        }
    )
    return result


def _validate_v12_objective_protocol(
    protocol: Mapping[str, Any],
    *,
    input_contract: Mapping[str, Any],
) -> None:
    try:
        launch._validate_checkpoint_protocol(
            protocol,
            data=input_contract["launch_data"],
        )
    except Exception as exc:
        raise V12EvaluationContractError(f"V12 objective protocol differs: {exc}") from exc
    artifacts = input_contract["artifacts"]
    expected_source = {
        "path": artifacts["v9_source_manifest"]["path"],
        "file_sha256": artifacts["v9_source_manifest"]["sha256"],
        "schema": launch.v10.v9.SOURCE_SCHEMA,
        "train_file": artifacts["train32"]["path"],
        "train_file_sha256": artifacts["train32"]["sha256"],
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": v10.v9.v8.v7.EPISODE_CONTRACT,
    }
    _require(
        protocol.get("scene_state_source_manifest") == expected_source,
        "V12 objective source identity differs",
    )


def validate_v12_checkpoint(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
    launch_receipt: Path | str,
    completion_receipt: Path | str,
    base_model_identity: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    warm = (
        launch.validate_warm_start_contract(ssd_root=ssd_root)
        if warm_contract is None
        else dict(warm_contract)
    )
    try:
        lineage = launch.validate_checkpoint_contract(
            Path(memory_dir),
            data=input_contract["launch_data"],
            warm=warm,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V12EvaluationContractError(f"V12 checkpoint contract failed: {exc}") from exc
    resolved = Path(str(lineage["checkpoint"]))
    checkpoint_step = int(lineage["checkpoint_step"])
    protocol = _load_json(
        resolved / "training_protocol.json",
        description="V12 training protocol",
    )
    _validate_v12_objective_protocol(protocol, input_contract=input_contract)
    pairing = _load_json(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="V12 checkpoint pairing manifest",
    )
    try:
        pairing_sha256 = v10.v9._validate_v9_pairing(
            pairing,
            protocol=protocol,
            input_contract=input_contract,
        )
    except Exception as exc:
        raise V12EvaluationContractError(f"V12 pairing contract failed: {exc}") from exc
    architecture = v10.v9.memory_architecture_contract(resolved)
    _require(
        architecture.get("target_layers") == list(range(42))
        and architecture.get("delta_heads") == ["q", "o"]
        and architecture.get("rank") == 4
        and architecture.get("rwkv_ms_semantics_version") == 2
        and architecture.get("memory_backend") == "rwkv_ms",
        "V12 checkpoint architecture differs",
    )
    artifacts = {
        name.removesuffix(".json").removesuffix(".pt").removesuffix(".pth"): (
            _artifact_binding(resolved / name, description=f"V12 checkpoint {name}")
        )
        for name in launch.REQUIRED_CHECKPOINT_ARTIFACTS
    }
    rng = [
        _artifact_binding(path, description=f"V12 checkpoint RNG {path.name}")
        for path in sorted(resolved.glob("rng_state*.pth"))
    ]
    lineage_path = resolved / str(lineage["lineage_filename"])
    checkpoint_contract = {
        "memory_dir": str(resolved),
        "global_step": checkpoint_step,
        "max_steps": 2,
        "consumed_pair_presentations": lineage["consumed_pair_presentations"],
        "cycle_pair_telemetry": lineage["cycle_pair_telemetry"],
        "row_objective_audit": lineage["row_objective_audit"],
        "row_objective_audit_file_sha256": lineage[
            "row_objective_audit_file_sha256"
        ],
        "artifacts": artifacts,
        "rng_state": rng,
        "training_protocol_canonical_sha256": lineage[
            "training_protocol_sha256"
        ],
        "pairing_manifest_sha256": pairing_sha256,
        "lineage": dict(lineage),
        "lineage_artifact": _artifact_binding(
            lineage_path,
            description="V12 checkpoint lineage",
        ),
        "architecture": architecture,
        "objective": dict(V12_OBJECTIVE),
        "warm_start_checkpoint": warm["warm_start_checkpoint"],
        "v10_diagnostic_checkpoint": input_contract["v10_diagnostic_baseline"][
            "checkpoint"
        ],
    }
    model_identity = (
        input_contract["v10_diagnostic_baseline"]["base_model_identity"]
        if base_model_identity is None
        else dict(base_model_identity)
    )
    try:
        provenance = launch.validate_training_provenance(
            checkpoint=resolved,
            checkpoint_contract=lineage,
            launch_receipt=Path(launch_receipt),
            completion_receipt=Path(completion_receipt),
            baseline=input_contract["v10_diagnostic_baseline"],
            base_model_identity=model_identity,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V12EvaluationContractError(
            f"V12 training provenance failed: {exc}"
        ) from exc
    checkpoint_contract["training_provenance"] = provenance
    return checkpoint_contract


def build_v12_gate(
    *,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    pairing: Mapping[str, Any],
    checkpoint_step: int = 1,
) -> dict[str, Any]:
    _require(
        checkpoint_step in CHECKPOINT_STEPS,
        "V12 gate accepts only checkpoint-1 or checkpoint-2",
    )
    _require(
        set(records_by_condition) == set(CONDITIONS),
        "V12 gate conditions differ",
    )
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        records = list(records_by_condition[condition])
        _require(
            len(records) == len(VALUE14_ORDINALS),
            f"V12 gate requires exactly 14 {condition} rows",
        )
        condition_rows: dict[int, Mapping[str, Any]] = {}
        for expected_ordinal, record in zip(VALUE14_ORDINALS, records, strict=True):
            ordinal = record.get("train_row_ordinal")
            _require(
                ordinal == expected_ordinal,
                "V12 gate rows must be exactly Value14 in canonical order",
            )
            _require(record.get("condition") == condition, "V12 gate condition differs")
            v10.v9.v8._strict_score(record)
            condition_rows[expected_ordinal] = record
        indexed[condition] = condition_rows

    correct = indexed["state_only"]
    donor = indexed["state_only_donor"]
    zero = indexed["state_only_no_write"]
    donor_by_ordinal = {
        int(entry["train_row_ordinal"]): int(entry["donor_train_row_ordinal"])
        for entry in pairing["directed_pairs"]
        if int(entry["train_row_ordinal"]) in VALUE14_SET
    }
    _require(
        set(donor_by_ordinal) == VALUE14_SET
        and all(
            donor_by_ordinal.get(donor_ordinal) == ordinal
            for ordinal, donor_ordinal in donor_by_ordinal.items()
        ),
        "V12 Value14 donor mapping differs",
    )
    scores = {
        ordinal: v10.v9.v8._strict_score(correct[ordinal])
        for ordinal in VALUE14_ORDINALS
    }
    donor_current_scores = {
        ordinal: v10.v9.v8._strict_score(donor[ordinal])
        for ordinal in VALUE14_ORDINALS
    }
    zero_scores = {
        ordinal: v10.v9.v8._strict_score(zero[ordinal])
        for ordinal in VALUE14_ORDINALS
    }
    donor_identity_scores = {
        ordinal: v10.v9.v8.score_prediction(
            "scene",
            donor[ordinal].get("parsed_json"),
            correct[donor_by_ordinal[ordinal]]["gold"],
        )
        for ordinal in VALUE14_ORDINALS
    }
    score_values = list(scores.values())
    donor_current_values = list(donor_current_scores.values())
    zero_values = list(zero_scores.values())
    donor_identity_values = list(donor_identity_scores.values())
    exact = sum(v10.v9.v8._strict_exact(score) for score in score_values)
    generation = {
        "rows": 14,
        "ordinals": list(VALUE14_ORDINALS),
        "canonical_correct_outputs": sum(
            v10.v9.v8.is_canonical_scene_prediction(
                correct[ordinal].get("parsed_json")
            )
            for ordinal in VALUE14_ORDINALS
        ),
        "correct_strict_exact_rows": exact,
        "correct_strict_micro_f1": v10.v9.v8._micro_f1(score_values),
        "donor_current_strict_exact_rows": sum(
            v10.v9.v8._strict_exact(score) for score in donor_current_values
        ),
        "donor_current_strict_micro_f1": v10.v9.v8._micro_f1(
            donor_current_values
        ),
        "donor_identity_strict_exact_rows": sum(
            v10.v9.v8._strict_exact(score) for score in donor_identity_values
        ),
        "donor_identity_strict_micro_f1": v10.v9.v8._micro_f1(
            donor_identity_values
        ),
        "zero_strict_exact_rows": sum(
            v10.v9.v8._strict_exact(score) for score in zero_values
        ),
        "zero_strict_micro_f1": v10.v9.v8._micro_f1(zero_values),
        "correct_minus_zero_strict_exact_rows": exact
        - sum(v10.v9.v8._strict_exact(score) for score in zero_values),
        "correct_beats_donor_current_rows": sum(
            v10.v9.v8._strictly_better(correct[ordinal], donor[ordinal])
            for ordinal in VALUE14_ORDINALS
        ),
        "correct_beats_zero_rows": sum(
            v10.v9.v8._strictly_better(correct[ordinal], zero[ordinal])
            for ordinal in VALUE14_ORDINALS
        ),
        "tp": sum(int(score["tp"]) for score in score_values),
        "fp": sum(int(score["fp"]) for score in score_values),
        "fn": sum(int(score["fn"]) for score in score_values),
    }
    selected = v10.v9.v8.build_value14_selected_token_evidence(
        indexed=indexed,
        pairing=pairing,
    )
    identity = selected["overall"]
    zero_invariant = len(
        {str(zero[ordinal].get("raw_generation")) for ordinal in VALUE14_ORDINALS}
    ) == 1
    gates = {
        "value14_all_correct_outputs_canonical": (
            generation["canonical_correct_outputs"]
            >= GATE_REQUIREMENTS["canonical_correct_outputs"]
        ),
        "value14_correct_identity_generation": (
            generation["correct_strict_exact_rows"]
            >= GATE_REQUIREMENTS["correct_strict_exact_rows"]
        ),
        "value14_donor_identity_generation": (
            generation["donor_identity_strict_exact_rows"]
            >= GATE_REQUIREMENTS["donor_identity_strict_exact_rows"]
        ),
        "value14_correct_strict_micro_f1_not_below_v10": (
            generation["correct_strict_micro_f1"]
            >= GATE_REQUIREMENTS["correct_strict_micro_f1"]
        ),
        "value14_bidirectional_selected_token_switch": (
            identity["bidirectional_identity_switch_rows"]
            >= GATE_REQUIREMENTS["bidirectional_identity_switch_rows"]
        ),
        "value14_correct_state_separation": (
            identity["correct_state_beats_donor_state_on_source_token_rows"]
            >= GATE_REQUIREMENTS[
                "correct_state_beats_donor_state_on_source_token_rows"
            ]
        ),
        "value14_correct_selected_token_identity": (
            identity["correct_state_prefers_source_token_rows"]
            >= GATE_REQUIREMENTS["correct_state_prefers_source_token_rows"]
        ),
        "value14_donor_selected_token_identity": (
            identity["donor_state_prefers_donor_token_rows"]
            >= GATE_REQUIREMENTS["donor_state_prefers_donor_token_rows"]
        ),
        "value14_selected_token_causal_vs_zero": (
            identity["correct_state_beats_zero_on_source_token_rows"]
            >= GATE_REQUIREMENTS[
                "correct_state_beats_zero_on_source_token_rows"
            ]
        ),
        "zero_reset_control_is_row_invariant": zero_invariant,
    }
    passed = all(gates.values())
    current = {
        name: (
            generation[name]
            if name in generation
            else identity[name]
        )
        for name in V10_DIAGNOSTIC_BASELINE
    }
    comparison = {
        "baseline": "completed_v10_checkpoint1_frozen_diagnostic",
        "v10": dict(V10_DIAGNOSTIC_BASELINE),
        "v12": current,
        "delta": {
            name: current[name] - V10_DIAGNOSTIC_BASELINE[name]
            for name in V10_DIAGNOSTIC_BASELINE
        },
    }
    return {
            "contract": GATE_CONTRACT,
            "task": v10.v9.TASK_NAME,
            "criterion": "value14_semantic_boundary_causal_candidate_v1",
            "evaluation_scope": "exact_value14_ordinals_only",
            "checkpoint_step": checkpoint_step,
            "optimizer_unit_pair_presentations": 7,
            "consumed_pair_presentations": checkpoint_step * 7,
            "requirements": dict(GATE_REQUIREMENTS),
            "v10_diagnostic_baseline": dict(V10_DIAGNOSTIC_BASELINE),
            "comparison": comparison,
            "metrics": {
                "value14_generation": generation,
                "value14_selected_token_identity": selected,
            },
            "gates": gates,
            "all_gates_passed": passed,
            "status": "pass" if passed else "fail",
            "candidate_designation": "candidate" if passed else "rejected",
            "candidate_authorized": passed,
            "next_checkpoint_step": None,
            "training_continuation_authorized": False,
            "training_continuation_policy": (
                launch.TRAINING_CONTINUATION_POLICY
            ),
            "hard32_access": HARD32_ACCESS_POLICY,
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
    }


def validate_resume_records(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    donor_by_ordinal: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    _require(condition in CONDITIONS, "V12 resume condition differs")
    _require(
        len(records) <= len(VALUE14_ORDINALS),
        "V12 resume records exceed Value14",
    )
    validated: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        current = dict(record)
        _require(current.get("schema") == GATE_RECORD_SCHEMA, "V12 record schema differs")
        _validate_self_hash(current, field="record_sha256")
        ordinal = VALUE14_ORDINALS[position]
        _require(current.get("fingerprint") == fingerprint, "V12 record fingerprint differs")
        _require(current.get("condition") == condition, "V12 record condition differs")
        _require(current.get("split") == "train", "V12 record split differs")
        _require(
            current.get("train_row_ordinal") == ordinal,
            "V12 records are not a canonical Value14 prefix",
        )
        sample = rows[ordinal]
        expected_donor = donor_by_ordinal[ordinal]
        _require(current.get("source_index") == sample["source_index"], "V12 source differs")
        _require(current.get("row_sha256") == sample["row_sha256"], "V12 row hash differs")
        _require(current.get("gold") == sample["gold"], "V12 record gold differs")
        _require(
            current.get("donor_train_row_ordinal") == expected_donor
            and current.get("donor_source_index")
            == rows[expected_donor]["source_index"],
            "V12 donor identity differs",
        )
        raw_generation = current.get("raw_generation")
        _require(isinstance(raw_generation, str), "V12 raw generation differs")
        _require(
            v10.v9.v8.extract_json(raw_generation) == current.get("parsed_json"),
            "V12 parsed JSON differs",
        )
        v10.v9.v8._strict_score(current)
        _require(
            current.get("score_recovered")
            == v10.v9.v8.recovered_scene_score(
                current.get("parsed_json"),
                sample["gold"],
            ),
            "V12 recovery score differs",
        )
        v10.v9.v8._pair_target_report(current)
        validated[ordinal] = current
    return validated


def evaluator_code_binding() -> dict[str, Any]:
    paths = {
        "v12_gate": Path(__file__).resolve(),
        "v10_gate_metrics": Path(v10.__file__).resolve(),
        "v9_gate_metrics": Path(v10.v9.__file__).resolve(),
        "v8_gate_metrics": Path(v10.v9.v8.__file__).resolve(),
        "train32_metric_runtime": Path(v10.v9.v8.v7.__file__).resolve(),
        "state_runtime": SCRIPT_DIR / "run_scene_state_eval.py",
        "scene_boundary_metric": Path(scene_boundary_metric.__file__).resolve(),
        "v12_launch_contract": Path(launch.__file__).resolve(),
        "v12_warm_start": Path(v12_warm.__file__).resolve(),
        "v10_launch_contract": Path(launch.v10.__file__).resolve(),
        "v10_warm_start": Path(v10_warm.__file__).resolve(),
        "v9_launch_contract": Path(v9_launch.__file__).resolve(),
        "v9_warm_start": Path(v9_warm.__file__).resolve(),
    }
    return {
        name: _artifact_binding(path, description=f"V12 evaluator code {name}")
        for name, path in paths.items()
    }


def build_evaluation_fingerprint_payload(
    *,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_model: Path | str = launch.PINNED_BASE_MODEL,
) -> dict[str, Any]:
    pinned_model = validate_base_model_path(base_model)
    model_identity = launch.validate_base_model_contract(
        base_model=pinned_model,
        baseline=input_contract["v10_diagnostic_baseline"],
    )
    _require(
        checkpoint.get("architecture", {}).get("target_layers") == list(range(42)),
        "V12 fingerprint checkpoint layer identity differs",
    )
    _require(
        isinstance(checkpoint.get("training_provenance"), Mapping),
        "V12 fingerprint training provenance is missing",
    )
    return {
        "schema_version": 1,
        "contract": GATE_CONTRACT,
        "task": v10.v9.TASK_NAME,
        "split": "train",
        "evaluation_scope": "exact_Value14_records_only",
        "training_sources": input_contract["artifacts"],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "checkpoint": dict(checkpoint),
        "training_provenance": dict(checkpoint["training_provenance"]),
        "v10_diagnostic_baseline": dict(
            input_contract["v10_diagnostic_baseline"]
        ),
        "base_model": str(pinned_model),
        "base_model_weights": model_identity["weights"],
        "base_model_prompt_artifacts": model_identity["prompt_artifacts"],
        "expected_memory_layer_count": GATE_EXPECTED_MEMORY_LAYER_COUNT,
        "runtime": {
            "conditions": list(CONDITIONS),
            "semantic_selected_token_ordinals": list(VALUE14_ORDINALS),
            "max_new_tokens": GATE_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache_generation": True,
            "prime_use_cache": False,
            "device": GATE_DEVICE,
            "dtype": GATE_DTYPE,
            "attn_implementation": GATE_ATTN_IMPLEMENTATION,
            "normal_fusion_profile": GATE_NORMAL_FUSION_PROFILE,
            "packages": v10.v9.runtime_package_versions(),
        },
        "objective": dict(V12_OBJECTIVE),
        "requirements": dict(GATE_REQUIREMENTS),
        "code": evaluator_code_binding(),
        "hard32_access": HARD32_ACCESS_POLICY,
        "other_benchmarks_authorized": False,
    }


def validate_existing_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_fingerprint: str,
    expected_fingerprint_payload: Mapping[str, Any],
    require_postload: bool = False,
) -> dict[str, Any]:
    preload_keys = {
        "schema",
        "created_at",
        "fingerprint",
        "fingerprint_payload",
        "hard32_access",
    }
    postload_keys = preload_keys | {"runtime_prefixes", "runtime_fusion_profile"}
    _require(
        set(manifest) in {frozenset(preload_keys), frozenset(postload_keys)},
        "V12 existing manifest fields differ",
    )
    if require_postload:
        _require(set(manifest) == postload_keys, "V12 completed manifest lacks runtime identity")
    _require(manifest.get("schema") == GATE_MANIFEST_SCHEMA, "V12 manifest schema differs")
    _require(
        isinstance(manifest.get("created_at"), str) and bool(manifest["created_at"]),
        "V12 manifest creation time differs",
    )
    expected_payload = dict(expected_fingerprint_payload)
    _require(
        manifest.get("fingerprint_payload") == expected_payload,
        "V12 manifest fingerprint payload differs",
    )
    _require(
        manifest.get("fingerprint") == expected_fingerprint
        and expected_fingerprint
        == v10.v9.fingerprint_payload_sha256(expected_payload),
        "V12 manifest fingerprint differs",
    )
    _require(
        manifest.get("hard32_access") == HARD32_ACCESS_POLICY,
        "V12 manifest Hard32 identity differs",
    )
    if "runtime_prefixes" in manifest:
        _require(
            manifest.get("runtime_prefixes") is not None
            and manifest.get("runtime_fusion_profile") is not None,
            "V12 manifest runtime identity differs",
        )
    return dict(manifest)


def bind_or_validate_manifest_runtime(
    manifest: Mapping[str, Any],
    *,
    runtime_prefixes: Any,
    runtime_fusion_profile: Any,
) -> dict[str, Any]:
    bound = dict(manifest)
    has_prefixes = "runtime_prefixes" in bound
    has_profile = "runtime_fusion_profile" in bound
    _require(has_prefixes == has_profile, "V12 manifest runtime fields are incomplete")
    if has_prefixes:
        _require(bound["runtime_prefixes"] == runtime_prefixes, "V12 runtime prefixes differ")
        _require(
            bound["runtime_fusion_profile"] == runtime_fusion_profile,
            "V12 runtime fusion profile differs",
        )
        return bound
    bound["runtime_prefixes"] = runtime_prefixes
    bound["runtime_fusion_profile"] = runtime_fusion_profile
    return bound


def build_gate_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
    ssd_root: Path,
) -> dict[str, Any]:
    output_dir = _gate_path(
        output_dir,
        description="V12 gate output directory",
        ssd_root=ssd_root,
    )
    outputs = {
        "manifest": _artifact_binding(
            output_dir / "manifest.json",
            description="V12 gate manifest",
        ),
        "summary": _artifact_binding(
            output_dir / "summary.json",
            description="V12 gate summary",
        ),
        "conditions": {
            condition: _artifact_binding(
                output_dir / f"{condition}.jsonl",
                description=f"V12 {condition}",
            )
            for condition in CONDITIONS
        },
    }
    passed = gate.get("status") == "pass" and gate.get("all_gates_passed") is True
    receipt: dict[str, Any] = {
        "schema": GATE_RECEIPT_SCHEMA,
        "created_at": v10.v9.utc_now(),
        "status": "pass" if passed else "fail",
        "contract": GATE_CONTRACT,
        "task": v10.v9.TASK_NAME,
        "evaluation_fingerprint": fingerprint,
        "objective": dict(V12_OBJECTIVE),
        "requirements": dict(GATE_REQUIREMENTS),
        "training_sources": dict(input_contract["artifacts"]),
        "v10_diagnostic_baseline": dict(
            input_contract["v10_diagnostic_baseline"]
        ),
        "checkpoint": dict(checkpoint),
        "training_provenance": dict(checkpoint["training_provenance"]),
        "outputs": outputs,
        "code": evaluator_code_binding(),
        "gate": dict(gate),
        "candidate_designation": {
            "kind": CANDIDATE_DESIGNATION_KIND,
            "designated": passed,
            "checkpoint_binding": dict(checkpoint),
        },
        "training_authorization": {
            "authorized": False,
            "next_checkpoint_step": None,
            "policy": launch.TRAINING_CONTINUATION_POLICY,
            "hard32_authorized": False,
            "full170_authorized": False,
            "test_authorized": False,
            "other_benchmarks_authorized": False,
        },
    }
    receipt["receipt_sha256"] = self_hash_payload(
        receipt,
        hash_field="receipt_sha256",
    )
    return receipt


def _verify_artifact_binding(
    binding: Mapping[str, Any],
    *,
    description: str,
    ssd_root: Path,
) -> Path:
    _require(isinstance(binding, Mapping), f"V12 {description} binding is missing")
    raw_path = binding.get("path")
    _require(
        isinstance(raw_path, str) and bool(raw_path),
        f"V12 {description} path differs",
    )
    expected = _gate_path(
        raw_path,
        description=description,
        ssd_root=ssd_root,
    )
    try:
        actual = v10.v9._verify_artifact_binding(
            binding,
            description=description,
        )
    except Exception as exc:
        raise V12EvaluationContractError(f"V12 {description} differs: {exc}") from exc
    _require(actual == expected, f"V12 {description} resolved path differs")
    return actual


def validate_gate_receipt_for_checkpoint(
    receipt: Path | str | Mapping[str, Any],
    *,
    memory_dir: Path | str,
    launch_receipt: Path | str | None = None,
    completion_receipt: Path | str | None = None,
    input_contract: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
    else:
        receipt_path = _regular_file(
            receipt,
            description="V12 gate receipt",
            ssd_root=ssd_root,
        )
        payload = _load_json(receipt_path, description="V12 gate receipt")
    _require(payload.get("schema") == GATE_RECEIPT_SCHEMA, "V12 receipt schema differs")
    _validate_self_hash(payload, field="receipt_sha256")
    _require(payload.get("contract") == GATE_CONTRACT, "V12 receipt contract differs")
    _require(payload.get("objective") == V12_OBJECTIVE, "V12 receipt objective differs")
    _require(
        payload.get("requirements") == GATE_REQUIREMENTS,
        "V12 receipt requirements differ",
    )
    inputs = (
        validate_v12_train_inputs(ssd_root=ssd_root)
        if input_contract is None
        else dict(input_contract)
    )
    _require(
        launch_receipt is not None and completion_receipt is not None,
        "V12 receipt replay requires exact launch and completion receipts",
    )
    checkpoint = validate_v12_checkpoint(
        memory_dir,
        input_contract=inputs,
        launch_receipt=launch_receipt,
        completion_receipt=completion_receipt,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    _require(payload.get("checkpoint") == checkpoint, "V12 receipt checkpoint differs")
    _require(
        payload.get("training_provenance") == checkpoint["training_provenance"],
        "V12 receipt training provenance differs",
    )
    _require(
        payload.get("training_sources") == inputs["artifacts"],
        "V12 receipt sources differ",
    )
    _require(
        payload.get("v10_diagnostic_baseline")
        == inputs["v10_diagnostic_baseline"],
        "V12 receipt V10 diagnostic differs",
    )
    fingerprint_payload = build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
    )
    fingerprint = v10.v9.fingerprint_payload_sha256(fingerprint_payload)
    _require(
        payload.get("evaluation_fingerprint") == fingerprint,
        "V12 receipt evaluation fingerprint differs from live inputs",
    )
    outputs = payload.get("outputs")
    _require(isinstance(outputs, Mapping), "V12 receipt outputs missing")
    manifest_path = _verify_artifact_binding(
        outputs.get("manifest", {}),
        description="V12 receipt manifest",
        ssd_root=ssd_root,
    )
    summary_path = _verify_artifact_binding(
        outputs.get("summary", {}),
        description="V12 receipt summary",
        ssd_root=ssd_root,
    )
    manifest = validate_existing_manifest(
        _load_json(manifest_path, description="V12 receipt manifest"),
        expected_fingerprint=fingerprint,
        expected_fingerprint_payload=fingerprint_payload,
        require_postload=True,
    )
    summary = _load_json(summary_path, description="V12 receipt summary")
    _require(summary.get("schema") == GATE_SUMMARY_SCHEMA, "V12 summary schema differs")
    _validate_self_hash(summary, field="summary_sha256")
    _require(
        manifest.get("fingerprint") == fingerprint
        and summary.get("fingerprint") == fingerprint,
        "V12 output fingerprints differ",
    )
    bindings = outputs.get("conditions")
    _require(
        isinstance(bindings, Mapping) and set(bindings) == set(CONDITIONS),
        "V12 condition bindings differ",
    )
    records: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = _verify_artifact_binding(
            bindings[condition],
            description=f"V12 {condition}",
            ssd_root=ssd_root,
        )
        indexed = validate_resume_records(
            _read_jsonl(path, description=f"V12 {condition}"),
            condition=condition,
            fingerprint=fingerprint,
            rows=inputs["rows"],
            donor_by_ordinal=inputs["pairing"]["donor_by_ordinal"],
        )
        _require(
            set(indexed) == VALUE14_SET,
            f"V12 {condition} output must contain exactly Value14",
        )
        records[condition] = [indexed[index] for index in VALUE14_ORDINALS]
    recomputed = build_v12_gate(
        records_by_condition=records,
        pairing=inputs["pairing"],
        checkpoint_step=int(checkpoint["global_step"]),
    )
    _require(
        recomputed == payload.get("gate") == summary.get("gate"),
        "V12 gate does not reproduce",
    )
    _require(payload.get("code") == evaluator_code_binding(), "V12 evaluator code differs")
    candidate = payload.get("candidate_designation")
    authorization = payload.get("training_authorization")
    passed = recomputed["status"] == "pass"
    _require(
        isinstance(candidate, Mapping)
        and candidate.get("kind") == CANDIDATE_DESIGNATION_KIND
        and candidate.get("designated") is passed
        and candidate.get("checkpoint_binding") == checkpoint,
        "V12 candidate designation differs",
    )
    _require(
        isinstance(authorization, Mapping)
        and authorization.get("authorized") is False
        and authorization.get("next_checkpoint_step") is None
        and authorization.get("policy") == launch.TRAINING_CONTINUATION_POLICY
        and authorization.get("hard32_authorized") is False
        and authorization.get("full170_authorized") is False
        and authorization.get("test_authorized") is False
        and authorization.get("other_benchmarks_authorized") is False,
        "V12 receipt training authorization differs",
    )
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = v10.v9.sha256_file(receipt_path)
    return result


def validate_continuation_authorization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise V12EvaluationContractError(
        "V12 is a terminal fresh two-cycle run; no gate authorizes continuation"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=GATE_MAX_NEW_TOKENS)
    parser.add_argument("--device", default=GATE_DEVICE)
    parser.add_argument("--dtype", default=GATE_DTYPE, choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default=GATE_ATTN_IMPLEMENTATION)
    parser.add_argument("--normal-fusion-profile", default=GATE_NORMAL_FUSION_PROFILE, choices=("native",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(args.max_new_tokens == GATE_MAX_NEW_TOKENS, "V12 gate requires 128 tokens")
    _require(
        args.expected_memory_layer_count == GATE_EXPECTED_MEMORY_LAYER_COUNT,
        "V12 gate requires all 42 layers",
    )
    _require(args.device == GATE_DEVICE, "V12 gate device differs")
    _require(args.dtype == GATE_DTYPE, "V12 gate dtype differs")
    _require(
        args.attn_implementation == GATE_ATTN_IMPLEMENTATION,
        "V12 gate attention implementation differs",
    )
    _require(
        args.normal_fusion_profile == GATE_NORMAL_FUSION_PROFILE,
        "V12 gate fusion profile differs",
    )
    args.delta_mem_root = str(Path(args.delta_mem_root).expanduser().resolve())
    _require(Path(args.delta_mem_root) == PROJECT_ROOT, "V12 gate requires this checkout")
    base_model = validate_base_model_path(args.base_model)
    args.base_model = str(base_model)
    output_dir = _gate_path(
        args.output_dir,
        description="V12 gate output directory",
        ssd_root=launch.SSD_ROOT,
    )
    args.output_dir = output_dir
    inputs = validate_v12_train_inputs()
    base_model_identity = inputs["v10_diagnostic_baseline"]["base_model_identity"]
    warm = launch.validate_warm_start_contract()
    checkpoint = validate_v12_checkpoint(
        args.memory_dir,
        input_contract=inputs,
        launch_receipt=args.launch_receipt,
        completion_receipt=args.completion_receipt,
        base_model_identity=base_model_identity,
        warm_contract=warm,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "model_loaded": False,
                    "output_created": False,
                    "evaluation_scope": "exact_Value14_records_only",
                    "hard32_access": HARD32_ACCESS_POLICY,
                    "checkpoint": checkpoint,
                    "v10_diagnostic_baseline": inputs["v10_diagnostic_baseline"],
                    "base_model": str(base_model),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    rows = inputs["rows"]
    donors = inputs["pairing"]["donor_by_ordinal"]
    memory_dir = Path(str(checkpoint["memory_dir"]))
    args.memory_dir = memory_dir
    expected_layers = v10.v9.resolved_memory_layer_count(
        memory_dir,
        args.expected_memory_layer_count,
    )
    _require(expected_layers == 42, "V12 resolved memory layer count differs")
    fingerprint_payload = build_evaluation_fingerprint_payload(
        input_contract=inputs,
        checkpoint=checkpoint,
        base_model=base_model,
    )
    fingerprint = v10.v9.fingerprint_payload_sha256(fingerprint_payload)
    manifest = {
        "schema": GATE_MANIFEST_SCHEMA,
        "created_at": v10.v9.utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    paths = {
        condition: output_dir / f"{condition}.jsonl"
        for condition in CONDITIONS
    }
    paths.update(
        {
            "manifest": output_dir / "manifest.json",
            "summary": output_dir / "summary.json",
            "receipt": output_dir / "gate_receipt.json",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    if paths["manifest"].exists():
        manifest = validate_existing_manifest(
            _load_json(paths["manifest"], description="existing V12 manifest"),
            expected_fingerprint=fingerprint,
            expected_fingerprint_payload=fingerprint_payload,
        )
    else:
        atomic_write_json(paths["manifest"], manifest)
    completed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        existing = (
            _read_jsonl(paths[condition], description=f"V12 {condition}")
            if paths[condition].exists()
            else []
        )
        completed[condition] = validate_resume_records(
            existing,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donors,
        )
    if any(
        set(completed[condition]) != VALUE14_SET
        for condition in CONDITIONS
    ):
        model, tokenizer, runtime_profile = v10.v9.load_adapter_model(
            args,
            expected_layers,
        )
        prefixes = v10.v9.v8.v7.validate_runtime_prefixes(tokenizer, rows=rows)
        manifest = bind_or_validate_manifest_runtime(
            manifest,
            runtime_prefixes=prefixes,
            runtime_fusion_profile=runtime_profile,
        )
        atomic_write_json(paths["manifest"], manifest)
        try:
            for condition in CONDITIONS:
                for ordinal in VALUE14_ORDINALS:
                    sample = rows[ordinal]
                    if ordinal in completed[condition]:
                        continue
                    donor_ordinal = donors[ordinal]
                    donor_sample = rows[donor_ordinal]
                    result = v10.v9.evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_sample,
                        condition=condition,
                        max_new_tokens=GATE_MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=ordinal in VALUE14_SET,
                    )
                    record = _record_with_self_hash(
                        {
                            "schema": GATE_RECORD_SCHEMA,
                            "status": "ok",
                            "completed_at": v10.v9.utc_now(),
                            "fingerprint": fingerprint,
                            "condition": condition,
                            "task": v10.v9.TASK_NAME,
                            "task_kind": "scene",
                            "split": "train",
                            "train_row_ordinal": ordinal,
                            "source_index": sample["source_index"],
                            "row_sha256": sample["row_sha256"],
                            "gold": sample["gold"],
                            "donor_train_row_ordinal": donor_ordinal,
                            **result,
                            "donor_source_index": donor_sample["source_index"],
                            "donor_row_sha256": donor_sample["row_sha256"],
                        }
                    )
                    completed[condition][ordinal] = record
                    atomic_write_jsonl(
                        paths[condition],
                        [
                            completed[condition][index]
                            for index in sorted(completed[condition])
                        ],
                    )
        finally:
            del model
            del tokenizer
            v10.v9.clear_model_memory()
    ordered = {
        condition: [completed[condition][index] for index in VALUE14_ORDINALS]
        for condition in CONDITIONS
    }
    gate = build_v12_gate(
        records_by_condition=ordered,
        pairing=inputs["pairing"],
        checkpoint_step=int(checkpoint["global_step"]),
    )
    summaries = {
        condition: v10.v9.summarize_records(records)
        for condition, records in ordered.items()
    }
    summary: dict[str, Any] = {
        "schema": GATE_SUMMARY_SCHEMA,
        "created_at": v10.v9.utc_now(),
        "fingerprint": fingerprint,
        "complete": True,
        "contract": GATE_CONTRACT,
        "task": v10.v9.TASK_NAME,
        "split": "train",
        "evaluation_scope": "exact_Value14_records_only",
        "conditions": summaries,
        "comparisons": v10.v9.build_comparisons(summaries),
        "gate": gate,
        "v10_diagnostic_baseline": inputs["v10_diagnostic_baseline"],
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    summary["summary_sha256"] = self_hash_payload(
        summary,
        hash_field="summary_sha256",
    )
    atomic_write_json(paths["summary"], summary)
    receipt = build_gate_receipt(
        output_dir=output_dir,
        fingerprint=fingerprint,
        input_contract=inputs,
        checkpoint=checkpoint,
        gate=gate,
        ssd_root=launch.SSD_ROOT,
    )
    atomic_write_canonical_json(paths["receipt"], receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
