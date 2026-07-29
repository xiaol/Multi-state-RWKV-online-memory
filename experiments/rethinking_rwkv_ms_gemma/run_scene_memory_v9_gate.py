#!/usr/bin/env python3
"""Run the Train32-only Scene-Memory V9 progression gate.

The evaluator retains the V8 32-row, three-condition generation and selected-
token protocol.  It additionally binds every result to the V9 pair curriculum,
the symmetric V9 training objective, and the complete adapter-only V8-to-V9
checkpoint lineage.  Protected evaluation data is outside this module's input
contract and is never authorized by its receipt.
"""

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

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_scene_memory_v8_gate as v8,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    scene_memory_v9_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v9_data import (  # noqa: E402
    CHECKPOINT_STEPS,
    PAIR_ENTRIES_SHA256,
    PAIR_MANIFEST_FILE_SHA256,
    PAIR_MANIFEST_SHA256,
    SOURCE_MANIFEST_FILE_SHA256,
    SOURCE_MANIFEST_SHA256,
    TRAIN32_ROWS_SHA256,
    TRAIN32_SHA256,
    VALUE14_ORDINALS,
)
from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    TASK_NAME,
    base_model_prompt_identity,
    base_model_weight_identity,
    build_comparisons,
    clear_model_memory,
    evaluate_condition,
    fingerprint_payload_sha256,
    load_adapter_model,
    memory_architecture_contract,
    resolved_memory_layer_count,
    runtime_package_versions,
    sha256_file,
    summarize_records,
    utc_now,
)


GATE_CONTRACT = "scene_memory_v9_train32_progression_gate"
GATE_RECORD_SCHEMA = "rwkv_ms_scene_memory_v9_train32_gate_record.v1"
GATE_SUMMARY_SCHEMA = "rwkv_ms_scene_memory_v9_train32_gate_summary.v1"
GATE_MANIFEST_SCHEMA = "rwkv_ms_scene_memory_v9_train32_gate_manifest.v1"
GATE_RECEIPT_SCHEMA = "rwkv_ms_scene_memory_v9_train32_gate_receipt.v1"
CONTINUATION_AUTHORIZATION_KIND = "scene_memory_v9_train32_progression_receipt"
CONDITIONS = v8.CONDITIONS
VALUE14_SET = frozenset(VALUE14_ORDINALS)
FIRST_GATE_STEP = CHECKPOINT_STEPS[0]
FINAL_GATE_STEP = CHECKPOINT_STEPS[-1]
PAIRING_OBJECTIVE_VERSION = launch.PAIRING_OBJECTIVE_VERSION
HARD32_ACCESS_POLICY = "forbidden_not_resolved_opened_or_hashed"

V9_OBJECTIVE = {
    "training_objective_version": launch.OBJECTIVE_VERSION,
    "evaluation_criterion": (
        "canonical_greedy_generation_plus_first_pair_distinguishing_token_identity"
    ),
    "aggregate_full_answer_ce_authorizes": False,
    "pair_target": "first_pair_distinguishing_semantic_token_v1",
    "causal_control": "same_checkpoint_adapter_active_no_write_state_reset",
    "donor_control": "predeclared_symmetric_train32_donor_state",
    "progression_basis": "strict_generation_and_bidirectional_identity_switch_v1",
}

# Frozen observation from the completed V8 checkpoint-56 Train32 gate.  V9 may
# continue only when generation identity and bidirectional switching materially
# improve, while the already-working selected-token and causal evidence do not
# regress.
V8_CHECKPOINT56_BASELINE = {
    "correct_strict_exact_rows": 3,
    "donor_identity_strict_exact_rows": 3,
    "bidirectional_identity_switch_rows": 6,
    "correct_state_prefers_source_token_rows": 10,
    "donor_state_prefers_donor_token_rows": 10,
    "correct_state_beats_donor_state_on_source_token_rows": 13,
    "correct_state_beats_zero_on_source_token_rows": 11,
}
PROGRESSION_REQUIREMENTS = {
    "canonical_correct_outputs": 14,
    "correct_strict_exact_rows": 4,
    "donor_identity_strict_exact_rows": 4,
    "bidirectional_identity_switch_rows": 7,
    "correct_state_prefers_source_token_rows": 10,
    "donor_state_prefers_donor_token_rows": 10,
    "correct_state_beats_donor_state_on_source_token_rows": 13,
    "correct_state_beats_zero_on_source_token_rows": 11,
}

V9_OBJECTIVE_FORMULA = (
    "symmetric_pair_mean(weighted_full_gold_ce(schema=2,decision=4,termination=1) "
    "+ selected_full_vocab_ce + selected_top_competitor_hinge(0.2) + "
    "selected_correct_vs_detached_zero_nll_hinge(0.2) + 0.5 * "
    "generated_prefix(aligned_gold_ce + safe_wrong_unlikelihood))"
)


class V9EvaluationContractError(ValueError):
    """Raised when a Train32 V9 gate binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V9EvaluationContractError(message)


canonical_sha256 = v8.canonical_sha256
self_hash_payload = v8.self_hash_payload
atomic_write_json = v8.atomic_write_json
atomic_write_canonical_json = v8.atomic_write_canonical_json
atomic_write_jsonl = v8.atomic_write_jsonl
_artifact_binding = v8._artifact_binding
_record_with_self_hash = v8._record_with_self_hash


def _ssd_path(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path,
) -> Path:
    expanded = Path(path).expanduser()
    _require(not expanded.is_symlink(), f"V9 forbids a symlink for {description}")
    try:
        return launch.require_ssd(
            expanded,
            description=description,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V9EvaluationContractError(
            f"V9 {description} must stay on the SSD outside protected paths: {exc}"
        ) from exc


def _regular_file(
    path: Path | str,
    *,
    description: str,
    ssd_root: Path | None = None,
) -> Path:
    expanded = Path(path).expanduser()
    _require(not expanded.is_symlink(), f"V9 forbids a symlink for {description}")
    resolved = (
        expanded.resolve()
        if ssd_root is None
        else _ssd_path(
            expanded,
            description=description,
            ssd_root=ssd_root,
        )
    )
    _require(resolved.is_file(), f"V9 {description} is missing: {resolved}")
    return resolved


def _load_json(path: Path | str, *, description: str) -> dict[str, Any]:
    resolved = _regular_file(path, description=description)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V9EvaluationContractError(
            f"V9 {description} is invalid JSON: {resolved}"
        ) from exc
    _require(isinstance(payload, dict), f"V9 {description} must be an object")
    return payload


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    resolved = _regular_file(path, description=description)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        _require(bool(line.strip()), f"V9 {description} contains a blank row")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V9EvaluationContractError(
                f"V9 {description} row {line_number} is invalid JSON"
            ) from exc
        _require(
            isinstance(payload, dict),
            f"V9 {description} row {line_number} must be an object",
        )
        records.append(payload)
    return records


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> str:
    recorded = payload.get(field)
    _require(isinstance(recorded, str), f"V9 {field} is missing")
    _require(
        recorded == self_hash_payload(payload, hash_field=field),
        f"V9 {field} differs",
    )
    return recorded


def _input_record_path(
    source_manifest: Mapping[str, Any],
    name: str,
) -> Path:
    inputs = source_manifest.get("inputs")
    _require(isinstance(inputs, Mapping), "V9 source inputs are missing")
    record = inputs.get(name)
    _require(isinstance(record, Mapping), f"V9 source input is missing: {name}")
    return Path(str(record.get("path", ""))).expanduser().resolve()


def validate_v9_train_inputs(
    *,
    data_root: Path = launch.DATA_ROOT,
    source_lock_path: Path = launch.SOURCE_LOCK,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Validate only the frozen V9/Train32 inputs."""

    try:
        data = launch.validate_data_contract(
            data_root=data_root,
            source_lock_path=source_lock_path,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 data contract failed: {exc}") from exc
    source_path = Path(str(data["source_manifest"]))
    source = _load_json(source_path, description="source manifest")
    paths = {
        "train32": _input_record_path(source, "train32"),
        "train32_rows": _input_record_path(source, "train32_rows"),
        "train32_pair_manifest": _input_record_path(source, "pair_manifest"),
        "train32_source_manifest": _input_record_path(source, "source_manifest"),
    }
    try:
        v7_input = v8.v7.validate_v7_contract(
            contract="scene_v7_train32_overfit",
            dataset_file=paths["train32"],
            row_manifest_file=paths["train32_rows"],
            pair_manifest_file=paths["train32_pair_manifest"],
            source_manifest_file=paths["train32_source_manifest"],
            expected_dataset_sha256=TRAIN32_SHA256,
            expected_row_manifest_sha256=TRAIN32_ROWS_SHA256,
            expected_pair_manifest_sha256=PAIR_MANIFEST_FILE_SHA256,
            expected_source_manifest_sha256=SOURCE_MANIFEST_FILE_SHA256,
        )
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 Train32 contract failed: {exc}") from exc
    _require(
        v7_input["pair_manifest_sha256"] == PAIR_MANIFEST_SHA256
        and v7_input["pairing"]["entries_sha256"] == PAIR_ENTRIES_SHA256
        and v7_input["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256,
        "V9 Train32 canonical identities differ",
    )
    artifacts = {
        name: _artifact_binding(path, description=name)
        for name, path in paths.items()
    }
    artifacts.update(
        {
            "v9_source_lock": _artifact_binding(
                source_lock_path,
                description="V9 source lock",
            ),
            "v9_bundle_manifest": _artifact_binding(
                data["bundle_manifest"],
                description="V9 bundle manifest",
            ),
            "v9_source_manifest": _artifact_binding(
                data["source_manifest"],
                description="V9 source manifest",
            ),
            "v9_pair_schedule": _artifact_binding(
                data["schedule"],
                description="V9 pair schedule",
            ),
            "v9_pair_schedule_manifest": _artifact_binding(
                data["schedule_manifest"],
                description="V9 pair schedule manifest",
            ),
        }
    )
    _require(
        tuple(data["checkpoint_steps"]) == tuple(CHECKPOINT_STEPS),
        "V9 checkpoint schedule differs",
    )
    return {
        "contract": GATE_CONTRACT,
        "rows": v7_input["rows"],
        "pairing": v7_input["pairing"],
        "artifacts": artifacts,
        "launch_data": data,
        "v9_source_manifest_sha256": data["source_manifest_sha256"],
        "v9_schedule_entries_sha256": data["schedule_entries_sha256"],
        "v9_schedule_manifest_sha256": data["schedule_manifest_sha256"],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "hard32_access": HARD32_ACCESS_POLICY,
    }


def _validate_v9_objective_protocol(
    protocol: Mapping[str, Any],
    *,
    input_contract: Mapping[str, Any],
) -> None:
    artifacts = input_contract["artifacts"]
    expected = {
        "schema_version": launch.OBJECTIVE_SCHEMA_VERSION,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "train_sampler_mode": launch.FIXED_SAMPLER_MODE,
        "scene_generation_objective_formula": V9_OBJECTIVE_FORMULA,
        "scene_generation_backward_mode": (
            "sequential_pair_zero_probe_teacher_then_aligned_replay_v3"
        ),
        "scene_generation_zero_protocol": (
            "shared_exact_selected_causal_prefix_adapter_active_reset_state_detached_v2"
        ),
        "scene_generation_generated_prefix_correction_weight": 0.5,
        "scene_generation_generated_prefix_correction_mode": (
            "levenshtein_raw_generated_prefix_gold_ce_wrong_unlikelihood_v3"
        ),
        "scene_generation_generated_prefix_max_correction_events": 4,
        "scene_generation_pair_unit": (
            "canonical_low_with_reciprocal_full_payload_v1"
        ),
        "scene_generation_pair_physical_batch_size": 1,
        "scene_generation_pair_directional_exposures": 2,
    }
    mismatches = [name for name, value in expected.items() if protocol.get(name) != value]
    _require(
        not mismatches,
        "V9 symmetric objective protocol differs: " + ", ".join(mismatches),
    )
    source = protocol.get("scene_state_source_manifest")
    expected_source = {
        "path": artifacts["v9_source_manifest"]["path"],
        "file_sha256": artifacts["v9_source_manifest"]["sha256"],
        "schema": launch.SOURCE_SCHEMA,
        "train_file": artifacts["train32"]["path"],
        "train_file_sha256": artifacts["train32"]["sha256"],
        "train_rows": 32,
        "train_source_split": "train",
        "episode_contract": v8.v7.EPISODE_CONTRACT,
    }
    _require(source == expected_source, "V9 objective source identity differs")


def _validate_v9_pairing(
    pairing: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    input_contract: Mapping[str, Any],
) -> str:
    manifest_sha256 = _validate_self_hash(pairing, field="manifest_sha256")
    _require(pairing.get("schema_version") == 2, "V9 pairing schema differs")
    _require(
        pairing.get("objective_version") == PAIRING_OBJECTIVE_VERSION,
        "V9 pairing materialization objective differs",
    )
    splits = pairing.get("splits")
    _require(isinstance(splits, Mapping) and set(splits) == {"train"}, "V9 pairing splits differ")
    train = splits["train"]
    _require(isinstance(train, Mapping), "V9 train pairing is missing")
    _validate_self_hash(train, field="manifest_sha256")
    expected = {
        "sample_count": 32,
        "source_pair_manifest_path": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["path"],
        "source_pair_manifest_file_sha256": input_contract["artifacts"][
            "train32_pair_manifest"
        ]["sha256"],
        "source_pair_manifest_sha256": input_contract["pairing"][
            "manifest_sha256"
        ],
        "source_entries_sha256": input_contract["pairing"]["entries_sha256"],
        "target_stratum_row_counts": {
            "presence": 18,
            "same_cardinality_value": 10,
            "cross_cardinality_value": 4,
        },
        "symmetric_full_pair_materialized": True,
    }
    mismatches = [name for name, value in expected.items() if train.get(name) != value]
    _require(not mismatches, "V9 checkpoint pairing differs: " + ", ".join(mismatches))
    protocol_pairing = protocol.get("scene_state_identity_pairing")
    _require(
        isinstance(protocol_pairing, Mapping)
        and protocol_pairing.get("manifest_sha256") == manifest_sha256
        and protocol_pairing.get("target_stratum_row_counts")
        == expected["target_stratum_row_counts"],
        "V9 protocol/checkpoint pairing differs",
    )
    return manifest_sha256


def validate_v9_checkpoint(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Validate a completed V9 endpoint and its recursive adapter-only lineage."""

    checkpoint = Path(memory_dir).expanduser()
    current_warm = (
        launch.validate_warm_start_contract(ssd_root=ssd_root)
        if warm_contract is None
        else dict(warm_contract)
    )
    try:
        lineage = launch.validate_checkpoint_contract(
            checkpoint=checkpoint,
            data=input_contract["launch_data"],
            warm=current_warm,
            ssd_root=ssd_root,
        )
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 checkpoint contract failed: {exc}") from exc
    resolved = Path(str(lineage["checkpoint"]))
    step = int(lineage["checkpoint_step"])
    _require(step in CHECKPOINT_STEPS, "V9 checkpoint step is not locked")
    paths = {
        name.removesuffix(".json").removesuffix(".pt").removesuffix(".pth"): resolved / name
        for name in launch.REQUIRED_CHECKPOINT_ARTIFACTS
    }
    protocol = _load_json(resolved / "training_protocol.json", description="training protocol")
    _validate_v9_objective_protocol(protocol, input_contract=input_contract)
    pairing = _load_json(
        resolved / "scene_state_identity_pairing_manifest.json",
        description="checkpoint pairing manifest",
    )
    pairing_sha256 = _validate_v9_pairing(
        pairing,
        protocol=protocol,
        input_contract=input_contract,
    )
    architecture = memory_architecture_contract(resolved)
    _require(
        architecture.get("target_layers") == list(range(42))
        and architecture.get("delta_heads") == ["q", "o"]
        and architecture.get("rank") == 4
        and architecture.get("rwkv_ms_semantics_version") == 2
        and architecture.get("memory_backend") == "rwkv_ms",
        "V9 checkpoint architecture differs",
    )
    artifacts = {
        name: _artifact_binding(path, description=f"V9 checkpoint {name}")
        for name, path in paths.items()
    }
    rng = [
        _artifact_binding(path, description=f"V9 checkpoint RNG {path.name}")
        for path in sorted(resolved.glob("rng_state*.pth"))
    ]
    lineage_path = resolved / str(lineage["lineage_filename"])
    return {
        "memory_dir": str(resolved),
        "global_step": step,
        "max_steps": step,
        "artifacts": artifacts,
        "rng_state": rng,
        "training_protocol_canonical_sha256": lineage[
            "training_protocol_sha256"
        ],
        "pairing_manifest_sha256": pairing_sha256,
        "lineage": dict(lineage),
        "lineage_artifact": _artifact_binding(
            lineage_path,
            description="V9 checkpoint lineage",
        ),
        "architecture": architecture,
        "objective": dict(V9_OBJECTIVE),
    }


def _previous_checkpoint_from_lineage(
    checkpoint: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    step = int(checkpoint["global_step"])
    if step == FIRST_GATE_STEP:
        return None
    lineage_path = _verify_artifact_binding(
        checkpoint["lineage_artifact"],
        description="current checkpoint lineage",
    )
    lineage = _load_json(lineage_path, description="current checkpoint lineage")
    source = lineage.get("source_checkpoint")
    _require(isinstance(source, str) and source, "V9 continuation source is missing")
    previous = Path(source).expanduser().resolve()
    expected_step = CHECKPOINT_STEPS[CHECKPOINT_STEPS.index(step) - 1]
    _require(
        previous.name == f"checkpoint-{expected_step}",
        "V9 continuation does not identify the previous endpoint",
    )
    return previous, lineage


def validate_previous_gate_receipt(
    previous_receipt: Path | str | None,
    *,
    checkpoint: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    warm_contract: Mapping[str, Any] | None,
    ssd_root: Path,
) -> dict[str, Any] | None:
    """Bind a later V9 endpoint to its immediate passing gate receipt."""

    predecessor = _previous_checkpoint_from_lineage(checkpoint)
    if predecessor is None:
        _require(
            previous_receipt is None,
            "V9 checkpoint-7 forbids a previous gate receipt",
        )
        return None
    previous_checkpoint, continuation_lineage = predecessor
    _require(
        previous_receipt is not None,
        "V9 later checkpoint requires the previous gate receipt",
    )
    validated = validate_gate_receipt_for_checkpoint(
        previous_receipt,
        memory_dir=previous_checkpoint,
        input_contract=input_contract,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    previous_step = CHECKPOINT_STEPS[
        CHECKPOINT_STEPS.index(int(checkpoint["global_step"])) - 1
    ]
    validated_checkpoint = validated.get("checkpoint")
    _require(
        isinstance(validated_checkpoint, Mapping),
        "V9 previous receipt checkpoint binding is missing",
    )
    validated_lineage = validated_checkpoint.get("lineage")
    _require(
        isinstance(validated_lineage, Mapping),
        "V9 previous receipt lineage binding is missing",
    )
    _require(
        validated_checkpoint.get("memory_dir") == str(previous_checkpoint)
        and validated_checkpoint.get("global_step") == previous_step
        and continuation_lineage.get("source_checkpoint")
        == validated_checkpoint.get("memory_dir")
        and continuation_lineage.get("source_global_step") == previous_step
        and continuation_lineage.get("source_training_protocol_sha256")
        == validated_checkpoint.get("training_protocol_canonical_sha256")
        and continuation_lineage.get("source_lineage_filename")
        == validated_lineage.get("lineage_filename")
        and continuation_lineage.get("source_lineage_file_sha256")
        == validated_lineage.get("lineage_file_sha256")
        and continuation_lineage.get("root_warm_start_receipt_sha256")
        == validated_lineage.get("root_warm_start_receipt_sha256")
        and validated["gate"]["next_checkpoint_step"]
        == checkpoint["global_step"],
        "V9 previous receipt is not bound to the exact immediate lineage",
    )
    return validated


def build_v9_gate(
    *,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    pairing: Mapping[str, Any],
    checkpoint_step: int,
    previous_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the generation-based V9 progression decision."""

    _require(checkpoint_step in CHECKPOINT_STEPS, "V9 gate checkpoint step is not locked")
    step_index = CHECKPOINT_STEPS.index(checkpoint_step)
    previous_step = None if step_index == 0 else CHECKPOINT_STEPS[step_index - 1]
    if previous_step is None:
        _require(previous_gate is None, "V9 checkpoint-7 forbids a previous gate")
        comparison = dict(V8_CHECKPOINT56_BASELINE)
        comparison_kind = "v8_checkpoint56_frozen_baseline"
    else:
        _require(isinstance(previous_gate, Mapping), "V9 later checkpoint requires its previous gate")
        _require(
            previous_gate.get("contract") == GATE_CONTRACT
            and previous_gate.get("checkpoint_step") == previous_step
            and previous_gate.get("status") == "pass"
            and previous_gate.get("all_gates_passed") is True
            and previous_gate.get("training_continuation_authorized") is True
            and previous_gate.get("next_checkpoint_step") == checkpoint_step
            and previous_gate.get("hard32_authorized") is False,
            "V9 previous gate is not a passing immediate predecessor",
        )
        previous_generation = previous_gate.get("metrics", {}).get(
            "value14_generation"
        )
        previous_selected = previous_gate.get("metrics", {}).get(
            "value14_selected_token_identity",
            {},
        ).get("overall")
        _require(
            isinstance(previous_generation, Mapping)
            and isinstance(previous_selected, Mapping),
            "V9 previous gate metrics are missing",
        )
        comparison = {
            "correct_strict_exact_rows": previous_generation.get(
                "correct_strict_exact_rows"
            ),
            "donor_identity_strict_exact_rows": previous_generation.get(
                "donor_identity_strict_exact_rows"
            ),
            "bidirectional_identity_switch_rows": previous_selected.get(
                "bidirectional_identity_switch_rows"
            ),
            "correct_state_prefers_source_token_rows": previous_selected.get(
                "correct_state_prefers_source_token_rows"
            ),
            "donor_state_prefers_donor_token_rows": previous_selected.get(
                "donor_state_prefers_donor_token_rows"
            ),
            "correct_state_beats_donor_state_on_source_token_rows": (
                previous_selected.get(
                    "correct_state_beats_donor_state_on_source_token_rows"
                )
            ),
            "correct_state_beats_zero_on_source_token_rows": previous_selected.get(
                "correct_state_beats_zero_on_source_token_rows"
            ),
        }
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in comparison.values()),
            "V9 previous gate comparison metrics are invalid",
        )
        comparison_kind = "immediately_previous_passing_v9_gate"
    try:
        diagnostic = v8.build_v8_gate(
            records_by_condition=records_by_condition,
            pairing=pairing,
        )
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 generation evidence differs: {exc}") from exc
    generation = diagnostic["metrics"]["value14_generation"]
    selected = diagnostic["metrics"]["value14_selected_token_identity"]["overall"]
    requirements = {
        "canonical_correct_outputs": PROGRESSION_REQUIREMENTS[
            "canonical_correct_outputs"
        ],
        "correct_strict_exact_rows": int(
            comparison["correct_strict_exact_rows"]
        )
        + 1,
        "donor_identity_strict_exact_rows": int(
            comparison["donor_identity_strict_exact_rows"]
        )
        + 1,
        "bidirectional_identity_switch_rows": int(
            comparison["bidirectional_identity_switch_rows"]
        )
        + 1,
        "correct_state_prefers_source_token_rows": int(
            comparison["correct_state_prefers_source_token_rows"]
        ),
        "donor_state_prefers_donor_token_rows": int(
            comparison["donor_state_prefers_donor_token_rows"]
        ),
        "correct_state_beats_donor_state_on_source_token_rows": int(
            comparison["correct_state_beats_donor_state_on_source_token_rows"]
        ),
        "correct_state_beats_zero_on_source_token_rows": int(
            comparison["correct_state_beats_zero_on_source_token_rows"]
        ),
    }
    gates = {
        "value14_all_correct_outputs_canonical": (
            generation["canonical_correct_outputs"]
            >= requirements["canonical_correct_outputs"]
        ),
        "value14_correct_generation_strictly_improves": (
            generation["correct_strict_exact_rows"]
            >= requirements["correct_strict_exact_rows"]
        ),
        "value14_donor_generation_strictly_improves": (
            generation["donor_identity_strict_exact_rows"]
            >= requirements["donor_identity_strict_exact_rows"]
        ),
        "value14_bidirectional_switch_strictly_improves": (
            selected["bidirectional_identity_switch_rows"]
            >= requirements["bidirectional_identity_switch_rows"]
        ),
        "value14_source_token_preference_does_not_regress": (
            selected["correct_state_prefers_source_token_rows"]
            >= requirements["correct_state_prefers_source_token_rows"]
        ),
        "value14_donor_token_preference_does_not_regress": (
            selected["donor_state_prefers_donor_token_rows"]
            >= requirements["donor_state_prefers_donor_token_rows"]
        ),
        "value14_source_state_separation_does_not_regress": (
            selected["correct_state_beats_donor_state_on_source_token_rows"]
            >= requirements[
                "correct_state_beats_donor_state_on_source_token_rows"
            ]
        ),
        "value14_causal_zero_control_does_not_regress": (
            selected["correct_state_beats_zero_on_source_token_rows"]
            >= requirements["correct_state_beats_zero_on_source_token_rows"]
        ),
        "zero_reset_control_is_row_invariant": diagnostic["gates"][
            "zero_reset_control_is_row_invariant"
        ],
    }
    passed = all(gates.values())
    next_step = (
        None
        if checkpoint_step == FINAL_GATE_STEP
        else CHECKPOINT_STEPS[step_index + 1]
    )
    final_quality_gates = dict(diagnostic["gates"])
    final_quality_passed = all(final_quality_gates.values())
    return {
        "status": "pass" if passed else "fail",
        "all_gates_passed": passed,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "checkpoint_step": checkpoint_step,
        "criterion": V9_OBJECTIVE["evaluation_criterion"],
        "full_answer_ce_used_for_gate": False,
        "v8_checkpoint56_baseline": dict(V8_CHECKPOINT56_BASELINE),
        "comparison": {
            "kind": comparison_kind,
            "checkpoint_step": previous_step,
            "metrics": comparison,
        },
        "requirements": dict(requirements),
        "metrics": diagnostic["metrics"],
        "gates": gates,
        "final_quality_diagnostic": {
            "requirements": diagnostic["requirements"],
            "gates": final_quality_gates,
            "all_gates_passed": final_quality_passed,
        },
        "training_continuation_authorized": passed and next_step is not None,
        "next_checkpoint_step": next_step,
        "final_checkpoint_reached": checkpoint_step == FINAL_GATE_STEP,
        "final_benchmark_candidate": (
            checkpoint_step == FINAL_GATE_STEP and final_quality_passed
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
    converted: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    for record in records:
        current = dict(record)
        _require(current.get("schema") == GATE_RECORD_SCHEMA, "V9 record schema differs")
        _validate_self_hash(current, field="record_sha256")
        originals.append(current)
        compatible = dict(current)
        compatible["schema"] = v8.GATE_RECORD_SCHEMA
        compatible = v8._record_with_self_hash(compatible)
        converted.append(compatible)
    try:
        validated = v8.validate_resume_records(
            converted,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
        )
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 record contract differs: {exc}") from exc
    return {ordinal: originals[ordinal] for ordinal in validated}


def evaluator_code_binding() -> dict[str, Any]:
    paths = {
        "v9_gate": Path(__file__).resolve(),
        "v8_gate_metrics": Path(v8.__file__).resolve(),
        "v7_train32_runtime": Path(v8.v7.__file__).resolve(),
        "state_runtime": SCRIPT_DIR / "run_scene_state_eval.py",
        "v9_launch_contract": Path(launch.__file__).resolve(),
    }
    return {
        name: _artifact_binding(path, description=f"V9 evaluator code {name}")
        for name, path in paths.items()
    }


def build_gate_receipt(
    *,
    output_dir: Path,
    fingerprint: str,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
    previous_receipt_path: Path | None = None,
    previous_receipt: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    output_dir = _ssd_path(
        output_dir,
        description="V9 gate output directory",
        ssd_root=ssd_root,
    )
    outputs = {
        "manifest": _artifact_binding(
            output_dir / "manifest.json",
            description="V9 gate manifest",
        ),
        "summary": _artifact_binding(
            output_dir / "summary.json",
            description="V9 gate summary",
        ),
        "conditions": {
            condition: _artifact_binding(
                output_dir / f"{condition}.jsonl",
                description=f"V9 gate {condition} output",
            )
            for condition in CONDITIONS
        },
    }
    passed = gate.get("status") == "pass" and gate.get("all_gates_passed") is True
    if checkpoint["global_step"] == FIRST_GATE_STEP:
        _require(
            previous_receipt_path is None and previous_receipt is None,
            "V9 checkpoint-7 receipt forbids a predecessor",
        )
        previous_binding = None
    else:
        _require(
            previous_receipt_path is not None
            and isinstance(previous_receipt, Mapping),
            "V9 later receipt requires its predecessor",
        )
        previous_receipt_path = _regular_file(
            previous_receipt_path,
            description="previous V9 gate receipt",
            ssd_root=ssd_root,
        )
        previous_binding = {
            "artifact": _artifact_binding(
                previous_receipt_path,
                description="previous V9 gate receipt",
            ),
            "receipt_sha256": previous_receipt.get("receipt_sha256"),
            "checkpoint": previous_receipt.get("checkpoint"),
        }
    receipt: dict[str, Any] = {
        "schema": GATE_RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "status": "pass" if passed else "fail",
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "evaluation_fingerprint": fingerprint,
        "objective": dict(V9_OBJECTIVE),
        "training_sources": dict(input_contract["artifacts"]),
        "v9_source_manifest_sha256": input_contract[
            "v9_source_manifest_sha256"
        ],
        "v9_schedule_entries_sha256": input_contract[
            "v9_schedule_entries_sha256"
        ],
        "v9_schedule_manifest_sha256": input_contract[
            "v9_schedule_manifest_sha256"
        ],
        "checkpoint": dict(checkpoint),
        "previous_gate_receipt": previous_binding,
        "outputs": outputs,
        "code": evaluator_code_binding(),
        "gate": dict(gate),
        "training_authorization": {
            "authorization_kind": CONTINUATION_AUTHORIZATION_KIND,
            "authorized": bool(gate.get("training_continuation_authorized")),
            "checkpoint_binding": dict(checkpoint),
            "next_checkpoint_step": gate.get("next_checkpoint_step"),
            "hard32_access": HARD32_ACCESS_POLICY,
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
    ssd_root: Path | None = None,
) -> Path:
    guarded_path = None
    if ssd_root is not None:
        _require(
            isinstance(binding, Mapping),
            f"V9 {description} binding is missing",
        )
        raw_path = binding.get("path")
        _require(
            isinstance(raw_path, str) and raw_path,
            f"V9 {description} path differs",
        )
        guarded_path = _ssd_path(
            raw_path,
            description=description,
            ssd_root=ssd_root,
        )
    try:
        path = v8._verify_artifact_binding(binding, description=description)
    except Exception as exc:
        raise V9EvaluationContractError(f"V9 {description} differs: {exc}") from exc
    if ssd_root is None:
        return path
    _require(path == guarded_path, f"V9 {description} resolved path differs")
    return path


def _validate_receipt_outputs(
    payload: Mapping[str, Any],
    *,
    input_contract: Mapping[str, Any],
    previous_gate: Mapping[str, Any] | None,
    ssd_root: Path,
) -> None:
    outputs = payload.get("outputs")
    _require(isinstance(outputs, Mapping), "V9 receipt outputs are missing")
    manifest_path = _verify_artifact_binding(
        outputs.get("manifest", {}),
        description="receipt manifest",
        ssd_root=ssd_root,
    )
    summary_path = _verify_artifact_binding(
        outputs.get("summary", {}),
        description="receipt summary",
        ssd_root=ssd_root,
    )
    manifest = _load_json(manifest_path, description="receipt manifest")
    summary = _load_json(summary_path, description="receipt summary")
    _require(manifest.get("schema") == GATE_MANIFEST_SCHEMA, "V9 manifest schema differs")
    _require(summary.get("schema") == GATE_SUMMARY_SCHEMA, "V9 summary schema differs")
    _validate_self_hash(summary, field="summary_sha256")
    fingerprint = payload.get("evaluation_fingerprint")
    _require(
        manifest.get("fingerprint") == fingerprint
        and summary.get("fingerprint") == fingerprint,
        "V9 receipt fingerprints differ",
    )
    condition_bindings = outputs.get("conditions")
    _require(
        isinstance(condition_bindings, Mapping)
        and set(condition_bindings) == set(CONDITIONS),
        "V9 receipt condition outputs differ",
    )
    records: dict[str, list[dict[str, Any]]] = {}
    donor_by_ordinal = input_contract["pairing"]["donor_by_ordinal"]
    for condition in CONDITIONS:
        path = _verify_artifact_binding(
            condition_bindings[condition],
            description=f"receipt {condition} output",
            ssd_root=ssd_root,
        )
        rows = _read_jsonl(path, description=f"V9 {condition} output")
        validated = validate_resume_records(
            rows,
            condition=condition,
            fingerprint=str(fingerprint),
            rows=input_contract["rows"],
            donor_by_ordinal=donor_by_ordinal,
        )
        _require(len(validated) == 32, f"V9 receipt {condition} output is incomplete")
        records[condition] = [validated[index] for index in range(32)]
    recomputed = build_v9_gate(
        records_by_condition=records,
        pairing=input_contract["pairing"],
        checkpoint_step=int(payload["checkpoint"]["global_step"]),
        previous_gate=previous_gate,
    )
    _require(recomputed == payload.get("gate"), "V9 receipt gate does not reproduce")
    _require(recomputed == summary.get("gate"), "V9 summary gate does not reproduce")


def validate_gate_receipt_for_checkpoint(
    receipt: Path | str | Mapping[str, Any],
    *,
    memory_dir: Path | str,
    input_contract: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Validate a V9 progression receipt without protected-data access."""

    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
    else:
        receipt_path = _regular_file(
            receipt,
            description="V9 gate receipt",
            ssd_root=ssd_root,
        )
        payload = _load_json(receipt_path, description="V9 gate receipt")
    _require(payload.get("schema") == GATE_RECEIPT_SCHEMA, "V9 receipt schema differs")
    _validate_self_hash(payload, field="receipt_sha256")
    _require(payload.get("contract") == GATE_CONTRACT, "V9 receipt contract differs")
    _require(payload.get("task") == TASK_NAME, "V9 receipt task differs")
    _require(payload.get("objective") == V9_OBJECTIVE, "V9 receipt objective differs")
    _require(payload.get("status") == "pass", "V9 continuation requires a passing receipt")
    current_inputs = (
        validate_v9_train_inputs(ssd_root=ssd_root)
        if input_contract is None
        else dict(input_contract)
    )
    _require(
        payload.get("training_sources") == current_inputs["artifacts"],
        "V9 receipt training sources differ",
    )
    for field in (
        "v9_source_manifest_sha256",
        "v9_schedule_entries_sha256",
        "v9_schedule_manifest_sha256",
    ):
        _require(payload.get(field) == current_inputs[field], f"V9 receipt {field} differs")
    current_checkpoint = validate_v9_checkpoint(
        memory_dir,
        input_contract=current_inputs,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    _require(
        payload.get("checkpoint") == current_checkpoint,
        "V9 receipt checkpoint binding differs",
    )
    previous_binding = payload.get("previous_gate_receipt")
    if current_checkpoint["global_step"] == FIRST_GATE_STEP:
        _require(
            previous_binding is None,
            "V9 checkpoint-7 receipt has an unexpected predecessor",
        )
        validated_previous = None
    else:
        _require(
            isinstance(previous_binding, Mapping),
            "V9 later receipt predecessor binding is missing",
        )
        previous_path = _verify_artifact_binding(
            previous_binding.get("artifact", {}),
            description="previous V9 gate receipt",
            ssd_root=ssd_root,
        )
        validated_previous = validate_previous_gate_receipt(
            previous_path,
            checkpoint=current_checkpoint,
            input_contract=current_inputs,
            warm_contract=warm_contract,
            ssd_root=ssd_root,
        )
        _require(
            previous_binding
            == {
                "artifact": _artifact_binding(
                    previous_path,
                    description="previous V9 gate receipt",
                ),
                "receipt_sha256": validated_previous["receipt_sha256"],
                "checkpoint": validated_previous["checkpoint"],
            },
            "V9 previous receipt binding differs",
        )
    _require(payload.get("code") == evaluator_code_binding(), "V9 evaluator code differs")
    gate = payload.get("gate")
    _require(
        isinstance(gate, Mapping)
        and gate.get("status") == "pass"
        and gate.get("all_gates_passed") is True
        and gate.get("hard32_authorized") is False,
        "V9 receipt gate did not pass safely",
    )
    expected_authorization = {
        "authorization_kind": CONTINUATION_AUTHORIZATION_KIND,
        "authorized": bool(gate.get("training_continuation_authorized")),
        "checkpoint_binding": current_checkpoint,
        "next_checkpoint_step": gate.get("next_checkpoint_step"),
        "hard32_access": HARD32_ACCESS_POLICY,
        "hard32_authorized": False,
        "full170_authorized": False,
        "test_authorized": False,
        "other_benchmarks_authorized": False,
    }
    _require(
        payload.get("training_authorization") == expected_authorization,
        "V9 training authorization differs",
    )
    _validate_receipt_outputs(
        payload,
        input_contract=current_inputs,
        previous_gate=(
            None if validated_previous is None else validated_previous["gate"]
        ),
        ssd_root=ssd_root,
    )
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = sha256_file(receipt_path)
    return result


def validate_continuation_authorization(
    receipt: Path | str,
    *,
    source_checkpoint: Path | str,
    target_step: int,
    input_contract: Mapping[str, Any] | None = None,
    warm_contract: Mapping[str, Any] | None = None,
    ssd_root: Path = launch.SSD_ROOT,
) -> dict[str, Any]:
    """Authorize exactly one immediate V9 continuation from a passing receipt."""

    _require(target_step in CHECKPOINT_STEPS, "V9 continuation target is not locked")
    receipt_path = _regular_file(
        receipt,
        description="V9 continuation gate receipt",
        ssd_root=ssd_root,
    )
    validated = validate_gate_receipt_for_checkpoint(
        receipt_path,
        memory_dir=source_checkpoint,
        input_contract=input_contract,
        warm_contract=warm_contract,
        ssd_root=ssd_root,
    )
    checkpoint = validated["checkpoint"]
    source_step = int(checkpoint["global_step"])
    source_index = CHECKPOINT_STEPS.index(source_step)
    _require(
        source_index + 1 < len(CHECKPOINT_STEPS)
        and CHECKPOINT_STEPS[source_index + 1] == target_step,
        "V9 gate receipt does not bind the immediate target",
    )
    gate = validated["gate"]
    authorization = validated["training_authorization"]
    _require(
        gate.get("training_continuation_authorized") is True
        and gate.get("next_checkpoint_step") == target_step
        and authorization.get("authorization_kind")
        == CONTINUATION_AUTHORIZATION_KIND
        and authorization.get("authorized") is True
        and authorization.get("next_checkpoint_step") == target_step
        and authorization.get("checkpoint_binding") == checkpoint
        and authorization.get("hard32_authorized") is False,
        "V9 gate receipt does not authorize this continuation",
    )
    return {
        "authorization_kind": CONTINUATION_AUTHORIZATION_KIND,
        "gate_receipt": str(receipt_path),
        "gate_receipt_file_sha256": sha256_file(receipt_path),
        "gate_receipt_sha256": validated["receipt_sha256"],
        "source_checkpoint": checkpoint["memory_dir"],
        "source_step": source_step,
        "target_step": target_step,
        "hard32_access": HARD32_ACCESS_POLICY,
        "hard32_authorized": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-gate-receipt", type=Path)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-memory-layer-count", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--normal-fusion-profile", default="native", choices=("native",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _manifest_is_valid(
    manifest: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    _require(manifest.get("schema") == GATE_MANIFEST_SCHEMA, "V9 manifest schema differs")
    payload = manifest.get("fingerprint_payload")
    _require(isinstance(payload, dict), "V9 manifest fingerprint payload is missing")
    _require(
        fingerprint_payload_sha256(payload) == manifest.get("fingerprint"),
        "V9 manifest self-fingerprint differs",
    )
    _require(manifest.get("fingerprint") == expected_fingerprint, "V9 manifest differs")
    return dict(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS, "V9 gate requires 128 tokens")
    _require(args.expected_memory_layer_count == 42, "V9 gate requires all 42 layers")
    args.delta_mem_root = str(Path(args.delta_mem_root).expanduser().resolve())
    _require(Path(args.delta_mem_root) == PROJECT_ROOT, "V9 gate requires this checkout")
    output_dir = _ssd_path(
        args.output_dir,
        description="V9 gate output directory",
        ssd_root=launch.SSD_ROOT,
    )
    args.output_dir = output_dir
    input_contract = validate_v9_train_inputs()
    warm_contract = launch.validate_warm_start_contract()
    checkpoint = validate_v9_checkpoint(
        args.memory_dir,
        input_contract=input_contract,
        warm_contract=warm_contract,
    )
    previous_receipt = validate_previous_gate_receipt(
        args.previous_gate_receipt,
        checkpoint=checkpoint,
        input_contract=input_contract,
        warm_contract=warm_contract,
        ssd_root=launch.SSD_ROOT,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "model_loaded": False,
                    "output_created": False,
                    "hard32_access": HARD32_ACCESS_POLICY,
                    "checkpoint": checkpoint,
                    "previous_gate_receipt": (
                        None
                        if previous_receipt is None
                        else {
                            "receipt_path": previous_receipt.get("receipt_path"),
                            "receipt_sha256": previous_receipt["receipt_sha256"],
                            "checkpoint": previous_receipt["checkpoint"],
                        }
                    ),
                    "training_sources": input_contract["artifacts"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    rows = input_contract["rows"]
    donor_by_ordinal = input_contract["pairing"]["donor_by_ordinal"]
    memory_dir = args.memory_dir.expanduser().resolve()
    args.memory_dir = memory_dir
    base_model = Path(args.base_model).expanduser().resolve()
    args.base_model = str(base_model)
    expected_layers = resolved_memory_layer_count(
        memory_dir,
        args.expected_memory_layer_count,
    )
    fingerprint_payload = {
        "schema_version": 1,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "split": "train",
        "training_sources": input_contract["artifacts"],
        "value14_ordinals": list(VALUE14_ORDINALS),
        "rows": [
            {
                "train_row_ordinal": row["train_row_ordinal"],
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "donor_train_row_ordinal": donor_by_ordinal[
                    row["train_row_ordinal"]
                ],
            }
            for row in rows
        ],
        "checkpoint": checkpoint,
        "base_model": str(base_model),
        "base_model_weights": base_model_weight_identity(base_model),
        "base_model_prompt_artifacts": base_model_prompt_identity(base_model),
        "expected_memory_layer_count": expected_layers,
        "runtime": {
            "conditions": list(CONDITIONS),
            "semantic_selected_token_ordinals": list(VALUE14_ORDINALS),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache_generation": True,
            "prime_use_cache": False,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "normal_fusion_profile": "native",
            "packages": runtime_package_versions(),
        },
        "objective": dict(V9_OBJECTIVE),
        "code": evaluator_code_binding(),
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    fingerprint = fingerprint_payload_sha256(fingerprint_payload)
    manifest = {
        "schema": GATE_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    output_paths: dict[str, Path] = {
        condition: output_dir / f"{condition}.jsonl" for condition in CONDITIONS
    }
    output_paths.update(
        {
            "manifest": output_dir / "manifest.json",
            "summary": output_dir / "summary.json",
            "receipt": output_dir / "gate_receipt.json",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
    if output_paths["manifest"].exists():
        manifest = _manifest_is_valid(
            _load_json(output_paths["manifest"], description="existing V9 manifest"),
            expected_fingerprint=fingerprint,
        )
    else:
        atomic_write_json(output_paths["manifest"], manifest)

    completed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = output_paths[condition]
        existing = _read_jsonl(path, description=f"V9 {condition} output") if path.exists() else []
        completed[condition] = validate_resume_records(
            existing,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
        )

    if any(len(completed[condition]) < 32 for condition in CONDITIONS):
        model, tokenizer, runtime_profile = load_adapter_model(args, expected_layers)
        runtime_prefixes = v8.v7.validate_runtime_prefixes(tokenizer, rows=rows)
        if "runtime_prefixes" in manifest:
            _require(manifest["runtime_prefixes"] == runtime_prefixes, "V9 prefixes differ")
        else:
            manifest["runtime_prefixes"] = runtime_prefixes
        if "runtime_fusion_profile" in manifest:
            _require(
                manifest["runtime_fusion_profile"] == runtime_profile,
                "V9 runtime profile differs",
            )
        else:
            manifest["runtime_fusion_profile"] = runtime_profile
        atomic_write_json(output_paths["manifest"], manifest)
        try:
            for condition in CONDITIONS:
                for ordinal, sample in enumerate(rows):
                    if ordinal in completed[condition]:
                        continue
                    donor_ordinal = donor_by_ordinal[ordinal]
                    donor_sample = rows[donor_ordinal]
                    result = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_sample,
                        condition=condition,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=ordinal in VALUE14_SET,
                    )
                    record = _record_with_self_hash(
                        {
                            "schema": GATE_RECORD_SCHEMA,
                            "status": "ok",
                            "completed_at": utc_now(),
                            "fingerprint": fingerprint,
                            "condition": condition,
                            "task": TASK_NAME,
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
                        output_paths[condition],
                        [completed[condition][index] for index in sorted(completed[condition])],
                    )
        finally:
            del model
            del tokenizer
            clear_model_memory()

    ordered_records = {
        condition: [completed[condition][index] for index in range(32)]
        for condition in CONDITIONS
    }
    gate = build_v9_gate(
        records_by_condition=ordered_records,
        pairing=input_contract["pairing"],
        checkpoint_step=checkpoint["global_step"],
        previous_gate=(
            None if previous_receipt is None else previous_receipt["gate"]
        ),
    )
    summaries = {
        condition: summarize_records(records)
        for condition, records in ordered_records.items()
    }
    summary: dict[str, Any] = {
        "schema": GATE_SUMMARY_SCHEMA,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "complete": True,
        "contract": GATE_CONTRACT,
        "task": TASK_NAME,
        "split": "train",
        "conditions": summaries,
        "comparisons": build_comparisons(summaries),
        "gate": gate,
        "hard32_access": HARD32_ACCESS_POLICY,
    }
    summary["summary_sha256"] = self_hash_payload(
        summary,
        hash_field="summary_sha256",
    )
    atomic_write_json(output_paths["summary"], summary)
    receipt = build_gate_receipt(
        output_dir=output_dir,
        fingerprint=fingerprint,
        input_contract=input_contract,
        checkpoint=checkpoint,
        gate=gate,
        previous_receipt_path=args.previous_gate_receipt,
        previous_receipt=previous_receipt,
        ssd_root=launch.SSD_ROOT,
    )
    atomic_write_canonical_json(output_paths["receipt"], receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
