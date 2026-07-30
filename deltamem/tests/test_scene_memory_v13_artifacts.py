from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from datasets import Dataset
import pytest

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.tests.test_scene_memory_v10_contract import _data as _v10_data
from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v13_gate as gate
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v13_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v13_warm_start as warm


def test_v13_native_training_horizon_and_objective_are_locked() -> None:
    assert warm.WARM_START_MODE == "scene_memory_v13_v8_checkpoint56_adapter_only"
    assert warm.RECEIPT_SCHEMA == (
        "rwkv_ms_scene_memory_v13_adapter_warm_start_receipt.v1"
    )
    assert warm.SOURCE_IMPORT_POLICY == {
        "adapter": True,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": False,
        "rng": False,
        "global_step": False,
    }
    assert warm.TARGET_FRESH_START_POLICY["rng_state"] == "fresh_from_v13_seed"
    assert launch.OBJECTIVE_SCHEMA_VERSION == 16
    assert launch.OBJECTIVE_VERSION == (
        "scene_state_generation_ce_symmetric_dense_boundary_v13"
    )
    assert launch.DENSE_SEMANTIC_MODE == (
        "all_boundary_decision_ce_full_vocab_top1_retention_failed_first_semantic_"
        "actual_greedy_decision_only_repair_v2"
    )
    assert launch.FAILED_DECISION_ALIGNMENT_MODE == (
        "first_boundary_semantic_effect_edit_mapped_to_decision_mask_only_v1"
    )
    assert launch.FIXED_SAMPLER_MODE == (
        "explicit_ordered_v13_four_canonical_seven_pair_cycles_v1"
    )
    assert launch.TOTAL_PAIR_PRESENTATIONS == 28
    assert launch.TOTAL_OPTIMIZER_STEPS == 4
    assert launch.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert launch.PRESENTATION_CHECKPOINTS == (7, 14, 21, 28)
    assert launch.GRADIENT_ACCUMULATION_STEPS == 7
    assert launch.LEARNING_RATE == 1e-4
    assert launch.WARMUP_STEPS == 0
    assert launch.PREFIX_CORRECTION_WEIGHT == 0.0
    assert launch.ROW_OBJECTIVE_AUDIT_FILENAME in (
        launch.REQUIRED_CHECKPOINT_ARTIFACTS
    )


def test_v13_exact_four_cycle_schedule_and_cursors_are_locked() -> None:
    expected_cycles = (
        (
            (3, 24),
            (19, 28),
            (20, 31),
            (10, 23),
            (1, 14),
            (5, 9),
            (22, 26),
        ),
        (
            (19, 28),
            (22, 26),
            (5, 9),
            (3, 24),
            (20, 31),
            (10, 23),
            (1, 14),
        ),
        (
            (1, 14),
            (19, 28),
            (22, 26),
            (20, 31),
            (10, 23),
            (5, 9),
            (3, 24),
        ),
        (
            (22, 26),
            (19, 28),
            (10, 23),
            (3, 24),
            (20, 31),
            (5, 9),
            (1, 14),
        ),
    )
    assert (
        launch.FIRST_CYCLE_PAIRS,
        launch.SECOND_CYCLE_PAIRS,
        launch.THIRD_CYCLE_PAIRS,
        launch.FOURTH_CYCLE_PAIRS,
    ) == expected_cycles
    assert launch.FOUR_CYCLE_PAIRS == tuple(
        pair for cycle in expected_cycles for pair in cycle
    )
    assert launch.canonical_sha256(
        [list(pair) for pair in launch.FOUR_CYCLE_PAIRS]
    ) == launch.FOUR_CYCLE_PAIRS_SHA256
    assert [launch.presentation_cursor(step) for step in range(5)] == [
        0,
        7,
        14,
        21,
        28,
    ]
    with pytest.raises(launch.LaunchContractError):
        launch.presentation_cursor(5)


def _cycle_log(step: int, pairs: tuple[tuple[int, int], ...]) -> dict[str, float]:
    result = {
        "step": float(step),
        "delta/scene_generation_v13_cycle_index": float(step),
        "delta/scene_generation_v13_cycle_pair_presentations": 7.0,
    }
    for index, (low, high) in enumerate(pairs):
        result[
            f"delta/scene_generation_v13_cycle_pair_{index}_low_ordinal"
        ] = float(low)
        result[
            f"delta/scene_generation_v13_cycle_pair_{index}_high_ordinal"
        ] = float(high)
    return result


def _trainer_state(checkpoint_step: int) -> dict[str, list[dict[str, float]]]:
    cycles = (
        launch.FIRST_CYCLE_PAIRS,
        launch.SECOND_CYCLE_PAIRS,
        launch.THIRD_CYCLE_PAIRS,
        launch.FOURTH_CYCLE_PAIRS,
    )
    return {
        "log_history": [
            _cycle_log(step, cycles[step - 1])
            for step in range(1, checkpoint_step + 1)
        ]
    }


@pytest.mark.parametrize("checkpoint_step", launch.CHECKPOINT_STEPS)
def test_v13_cycle_telemetry_binds_every_checkpoint(checkpoint_step: int) -> None:
    result = launch.validate_v13_cycle_pair_telemetry(
        _trainer_state(checkpoint_step),
        checkpoint_step=checkpoint_step,
    )

    assert result["optimizer_step"] == checkpoint_step
    assert result["pair_presentations"] == checkpoint_step * 7
    assert result["ordered_pairs"] == [
        list(pair) for pair in launch.FOUR_CYCLE_PAIRS[: checkpoint_step * 7]
    ]
    assert len(result["cycles"]) == checkpoint_step


@pytest.mark.parametrize("checkpoint_step", launch.CHECKPOINT_STEPS)
def test_v13_cycle_telemetry_rejects_pair_tampering(checkpoint_step: int) -> None:
    trainer_state = _trainer_state(checkpoint_step)
    key = "delta/scene_generation_v13_cycle_pair_0_low_ordinal"
    trainer_state["log_history"][-1][key] += 1.0

    with pytest.raises(launch.LaunchContractError, match="order_differs"):
        launch.validate_v13_cycle_pair_telemetry(
            trainer_state,
            checkpoint_step=checkpoint_step,
        )


_V13_ROW_OBSERVATION_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "pair_role",
        "row_ordinal",
        "paired_row_ordinal",
        "row_sha256",
        "paired_row_sha256",
        "parsed_boundary_exact",
        "raw_token_exact",
        "first_divergence",
        "rollout_token_count",
        "dense_decision_token_count",
        "dense_decision_ce",
        "dense_decision_top1_retention_hinge",
        "dense_decision_gold_vs_top_competitor_margin",
        "dense_decision_top1_fraction",
        "dense_top1_margin",
        "selected_top_competitor_hinge",
        "selected_correct_vs_zero_hinge",
        "selected_zero_minus_correct_nll",
        "selected_top1",
        "failed_semantic_repair_applied",
        "failed_semantic_repair_ce",
        "failed_semantic_repair_competitor_hinge",
        "failed_semantic_repair_loss",
        "failed_semantic_gold_vs_competitor_margin",
        "failed_semantic_decision_ordinal",
        "failed_semantic_label_position",
        "failed_semantic_gold_token_id",
        "failed_semantic_competitor_id",
        "failed_semantic_competitor_is_actual_greedy",
        "failed_semantic_replay_generated_cursor",
        "failed_semantic_alignment_kind_code",
        "failed_semantic_is_termination",
        "dense_teacher_loss",
        "total_side_loss",
    }
)
_V13_PAIR_OBSERVATION_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "source_row_ordinal",
        "donor_row_ordinal",
        "source_row_sha256",
        "donor_row_sha256",
        "pair_mean_dense_decision_ce",
        "pair_mean_dense_decision_top1_retention_hinge",
        "pair_mean_selected_top_competitor_hinge",
        "pair_mean_selected_correct_vs_zero_hinge",
        "pair_mean_failed_semantic_repair_applied_fraction",
        "pair_mean_failed_semantic_ce",
        "pair_mean_failed_semantic_competitor_hinge",
        "pair_mean_failed_semantic_repair_loss",
        "pair_mean_dense_teacher_loss",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    }
)


def _v13_row_hash(ordinal: int) -> str:
    return f"{ordinal:064x}"


def _v13_row_audit_data() -> dict[str, Any]:
    data = _v10_data()
    data["entries"] = [
        {
            "canonical_pair_ordinals": [source, donor],
            "members": [
                {
                    "train_row_ordinal": source,
                    "donor_train_row_ordinal": donor,
                    "row_sha256": _v13_row_hash(source),
                    "donor_row_sha256": _v13_row_hash(donor),
                },
                {
                    "train_row_ordinal": donor,
                    "donor_train_row_ordinal": source,
                    "row_sha256": _v13_row_hash(donor),
                    "donor_row_sha256": _v13_row_hash(source),
                },
            ],
        }
        for source, donor in launch.FOUR_CYCLE_PAIRS
    ]
    return data


def _v13_row_observation(
    *,
    phase_index: int,
    presentation_index: int,
    pair_role: str,
    row_ordinal: int,
    paired_row_ordinal: int,
) -> dict[str, Any]:
    parsed_exact = pair_role == "source"
    repair_ce = 0.0 if parsed_exact else 0.4
    repair_hinge = 0.0 if parsed_exact else 0.5
    repair_loss = repair_ce + repair_hinge
    dense_teacher_loss = 0.7 + 0.6 + 0.2 + 0.1
    return {
        "phase": f"cycle{phase_index + 1}_input",
        "cycle": phase_index + 1,
        "adapter_optimizer_step_before_update": phase_index,
        "presentation": presentation_index + 1,
        "pair_role": pair_role,
        "row_ordinal": row_ordinal,
        "paired_row_ordinal": paired_row_ordinal,
        "row_sha256": _v13_row_hash(row_ordinal),
        "paired_row_sha256": _v13_row_hash(paired_row_ordinal),
        "parsed_boundary_exact": parsed_exact,
        "raw_token_exact": parsed_exact,
        "first_divergence": 8 if parsed_exact else 2,
        "rollout_token_count": 8,
        "dense_decision_token_count": 3,
        "dense_decision_ce": 0.7,
        "dense_decision_top1_retention_hinge": 0.6,
        "dense_decision_gold_vs_top_competitor_margin": 0.4,
        "dense_decision_top1_fraction": 1.0,
        "dense_top1_margin": 1.0,
        "selected_top_competitor_hinge": 0.2,
        "selected_correct_vs_zero_hinge": 0.1,
        "selected_zero_minus_correct_nll": 0.1,
        "selected_top1": False,
        "failed_semantic_repair_applied": not parsed_exact,
        "failed_semantic_repair_ce": repair_ce,
        "failed_semantic_repair_competitor_hinge": repair_hinge,
        "failed_semantic_repair_loss": repair_loss,
        "failed_semantic_gold_vs_competitor_margin": (
            0.0 if parsed_exact else 0.5
        ),
        "failed_semantic_decision_ordinal": -1 if parsed_exact else 1,
        "failed_semantic_label_position": (
            -1 if parsed_exact else 10 + row_ordinal
        ),
        "failed_semantic_gold_token_id": -1 if parsed_exact else 100 + row_ordinal,
        "failed_semantic_competitor_id": (
            -1 if parsed_exact else 200 + row_ordinal
        ),
        "failed_semantic_competitor_is_actual_greedy": not parsed_exact,
        "failed_semantic_replay_generated_cursor": -1 if parsed_exact else 2,
        "failed_semantic_alignment_kind_code": -1 if parsed_exact else 0,
        "failed_semantic_is_termination": False,
        "dense_teacher_loss": dense_teacher_loss,
        "total_side_loss": dense_teacher_loss + repair_loss,
    }


def _v13_pair_observation(
    *,
    phase_index: int,
    presentation_index: int,
    source: int,
    donor: int,
) -> dict[str, Any]:
    dense_teacher_loss = 0.7 + 0.6 + 0.2 + 0.1
    repair_ce = 0.2
    repair_hinge = 0.25
    repair_loss = repair_ce + repair_hinge
    total_loss = dense_teacher_loss + repair_loss
    return {
        "phase": f"cycle{phase_index + 1}_input",
        "cycle": phase_index + 1,
        "adapter_optimizer_step_before_update": phase_index,
        "presentation": presentation_index + 1,
        "source_row_ordinal": source,
        "donor_row_ordinal": donor,
        "source_row_sha256": _v13_row_hash(source),
        "donor_row_sha256": _v13_row_hash(donor),
        "pair_mean_dense_decision_ce": 0.7,
        "pair_mean_dense_decision_top1_retention_hinge": 0.6,
        "pair_mean_selected_top_competitor_hinge": 0.2,
        "pair_mean_selected_correct_vs_zero_hinge": 0.1,
        "pair_mean_failed_semantic_repair_applied_fraction": 0.5,
        "pair_mean_failed_semantic_ce": repair_ce,
        "pair_mean_failed_semantic_competitor_hinge": repair_hinge,
        "pair_mean_failed_semantic_repair_loss": repair_loss,
        "pair_mean_dense_teacher_loss": dense_teacher_loss,
        "pair_mean_total_side_loss": total_loss,
        "reported_objective_total_loss": total_loss,
        "recomputed_objective_total_loss": total_loss,
    }


def _v13_row_audit_payload(checkpoint_step: int) -> dict[str, Any]:
    row_order = [ordinal for pair in launch.FIRST_CYCLE_PAIRS for ordinal in pair]
    rows = {ordinal: {"row_ordinal": ordinal} for ordinal in row_order}
    pair_presentations = []
    presentation_count = checkpoint_step * launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    for presentation_index, (source, donor) in enumerate(
        launch.FOUR_CYCLE_PAIRS[:presentation_count]
    ):
        phase_index = presentation_index // launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
        phase = f"cycle{phase_index + 1}_input"
        rows[source][phase] = _v13_row_observation(
            phase_index=phase_index,
            presentation_index=presentation_index,
            pair_role="source",
            row_ordinal=source,
            paired_row_ordinal=donor,
        )
        rows[donor][phase] = _v13_row_observation(
            phase_index=phase_index,
            presentation_index=presentation_index,
            pair_role="donor",
            row_ordinal=donor,
            paired_row_ordinal=source,
        )
        pair_presentations.append(
            _v13_pair_observation(
                phase_index=phase_index,
                presentation_index=presentation_index,
                source=source,
                donor=donor,
            )
        )
    phases = [f"cycle{cycle}_input" for cycle in range(1, checkpoint_step + 1)]
    return {
        "schema": launch.ROW_OBJECTIVE_AUDIT_SCHEMA,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "checkpoint_optimizer_step": checkpoint_step,
        "completed_pair_presentations": presentation_count,
        "phases": phases,
        "pair_schedule": [
            {"source_row_ordinal": source, "donor_row_ordinal": donor}
            for source, donor in launch.FOUR_CYCLE_PAIRS[:presentation_count]
        ],
        "pair_presentations": pair_presentations,
        "rows": [rows[ordinal] for ordinal in row_order],
    }


@pytest.mark.parametrize("checkpoint_step", launch.CHECKPOINT_STEPS)
def test_v13_native_row_objective_audit_binds_rows_pairs_and_all_cycles(
    checkpoint_step: int,
) -> None:
    payload = _v13_row_audit_payload(checkpoint_step)

    assert set(payload) == {
        "schema",
        "memory_objective_version",
        "checkpoint_optimizer_step",
        "completed_pair_presentations",
        "phases",
        "pair_schedule",
        "pair_presentations",
        "rows",
    }
    assert set(payload["rows"][0]["cycle1_input"]) == _V13_ROW_OBSERVATION_FIELDS
    assert set(payload["pair_presentations"][0]) == _V13_PAIR_OBSERVATION_FIELDS
    assert launch._validate_v13_row_objective_audit(
        payload,
        checkpoint_step=checkpoint_step,
        data=_v13_row_audit_data(),
    ) == payload


@pytest.mark.parametrize("missing_field", sorted(_V13_ROW_OBSERVATION_FIELDS))
def test_v13_row_objective_audit_rejects_each_omitted_row_field(
    missing_field: str,
) -> None:
    payload = _v13_row_audit_payload(1)
    del payload["rows"][0]["cycle1_input"][missing_field]

    with pytest.raises(launch.LaunchContractError, match="v13_row_objective_audit_"):
        launch._validate_v13_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v13_row_audit_data(),
        )


@pytest.mark.parametrize("missing_field", sorted(_V13_PAIR_OBSERVATION_FIELDS))
def test_v13_row_objective_audit_rejects_each_omitted_pair_field(
    missing_field: str,
) -> None:
    payload = _v13_row_audit_payload(1)
    del payload["pair_presentations"][0][missing_field]

    with pytest.raises(launch.LaunchContractError, match="v13_row_objective_audit_"):
        launch._validate_v13_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v13_row_audit_data(),
        )


@pytest.mark.parametrize(
    "field",
    (
        "dense_decision_ce",
        "dense_decision_top1_retention_hinge",
        "selected_top_competitor_hinge",
        "selected_correct_vs_zero_hinge",
        "failed_semantic_repair_ce",
        "failed_semantic_repair_competitor_hinge",
        "failed_semantic_repair_loss",
        "dense_teacher_loss",
        "total_side_loss",
    ),
)
def test_v13_row_objective_audit_rejects_row_component_arithmetic_tampering(
    field: str,
) -> None:
    payload = _v13_row_audit_payload(1)
    failed_row = payload["rows"][1]["cycle1_input"]
    failed_row[field] += 0.125

    with pytest.raises(launch.LaunchContractError, match="arithmetic"):
        launch._validate_v13_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v13_row_audit_data(),
        )


@pytest.mark.parametrize(
    "field",
    (
        "pair_mean_dense_decision_ce",
        "pair_mean_dense_decision_top1_retention_hinge",
        "pair_mean_selected_top_competitor_hinge",
        "pair_mean_selected_correct_vs_zero_hinge",
        "pair_mean_failed_semantic_ce",
        "pair_mean_failed_semantic_competitor_hinge",
        "pair_mean_failed_semantic_repair_loss",
        "pair_mean_dense_teacher_loss",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    ),
)
def test_v13_row_objective_audit_rejects_pair_objective_arithmetic_tampering(
    field: str,
) -> None:
    payload = _v13_row_audit_payload(1)
    payload["pair_presentations"][0][field] += 0.125

    with pytest.raises(launch.LaunchContractError, match="arithmetic"):
        launch._validate_v13_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v13_row_audit_data(),
        )


def test_v13_row_objective_audit_binds_pair_means_to_row_observations() -> None:
    payload = _v13_row_audit_payload(1)
    payload["pair_presentations"][0][
        "pair_mean_failed_semantic_repair_applied_fraction"
    ] = 0.0

    with pytest.raises(launch.LaunchContractError, match="v13_row_objective_audit_"):
        launch._validate_v13_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v13_row_audit_data(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("resume_from_checkpoint", "checkpoint-1", "forbids checkpoint resume"),
        ("initial_global_step", 1, "global step 0"),
        ("initial_global_step", True, "global step 0"),
        ("optimizer_created", True, "preloaded training state"),
        ("scheduler_created", True, "preloaded training state"),
        ("trainer_state_imported", True, "preloaded training state"),
        ("rng_state_imported", True, "preloaded training state"),
        ("optim", "sgd", "fresh AdamW"),
    ),
)
def test_v13_fresh_start_forbids_resume_and_all_imported_state(
    field: str,
    value: Any,
    message: str,
) -> None:
    base = warm.V13FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )
    assert warm.validate_v13_fresh_start_contract(base) == {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v13_seed",
    }

    with pytest.raises(ValueError, match=message):
        warm.validate_v13_fresh_start_contract(replace(base, **{field: value}))


def test_v13_launch_rejects_resume_before_live_artifact_validation() -> None:
    with pytest.raises(launch.LaunchContractError, match="v13_resume_is_forbidden"):
        launch.validate_launch_contract(
            target_step=4,
            resume_checkpoint=Path("checkpoint-1"),
        )
    with pytest.raises(launch.LaunchContractError, match="forbids resume"):
        launch.validate_resume_contract()


@pytest.mark.parametrize(
    "path",
    (
        "/tmp/test.jsonl",
        "/tmp/val/data.jsonl",
        "/tmp/validation/data.jsonl",
        "/tmp/full170.jsonl",
        "/tmp/hard32-copy/train.jsonl",
    ),
)
def test_v13_training_data_guard_rejects_protected_split_before_resolve(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_resolve(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("protected training path was resolved")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(
        launch.LaunchContractError,
        match="protected_split_path_forbidden",
    ):
        launch.guard_v13_training_data_path(path, description="test_data")


def test_v13_training_data_guard_allows_value14_and_exact_pinned_train32() -> None:
    value14 = Path("/tmp/value14/pair_schedule.jsonl")
    assert launch.guard_v13_training_data_path(
        value14,
        description="value14",
    ) == value14
    historical = Path(
        launch.PINNED_HISTORICAL_TRAIN32_ARTIFACTS["train32"]["path"]
    )
    assert launch.guard_v13_training_data_path(
        historical,
        description="historical_train32",
    ) == historical


def _v13_protocol_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "delta-sft",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset-name",
            "synthetic",
        ],
    )
    args = experimental_train.parse_args()
    data = _v10_data()
    scheduled_pairs = list(launch.FOUR_CYCLE_PAIRS)
    data["entries"] = [
        {"canonical_pair_ordinals": list(pair)} for pair in scheduled_pairs
    ]
    binding = {
        "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
        "source_manifest_path": data["source_manifest"],
        "source_manifest_file_sha256": data["source_manifest_file_sha256"],
        "schedule_path": data["schedule"],
        "schedule_file_sha256": data["schedule_file_sha256"],
        "schedule_entries_sha256": data["schedule_entries_sha256"],
        "schedule_manifest_path": data["schedule_manifest"],
        "schedule_manifest_file_sha256": data[
            "schedule_manifest_file_sha256"
        ],
        "schedule_manifest_sha256": data["schedule_manifest_sha256"],
        "ordered_pairs_sha256": data["ordered_pairs_sha256"],
        "canonical_value14_pairs": [
            list(pair) for pair in launch.v10.CANONICAL_VALUE14_PAIRS
        ],
        "total_steps": 28,
        "checkpoint_steps": data["source_presentation_checkpoint_steps"],
        "pair_indices": tuple(scheduled_pairs),
        "indices": tuple(low for low, _ in scheduled_pairs),
    }
    locked_values = {
        "train_file": Path(str(data["train_file"])),
        "memory_loss_mode": "scene_state_generation_ce",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "max_length": 256,
        "max_write_length": 2048,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 7,
        "learning_rate": launch.LEARNING_RATE,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "optim": "adamw_torch_fused",
        "save_steps": 1,
        "logging_steps": 1,
        "eval_steps": 1000,
        "save_total_limit": 4,
        "num_train_epochs": 1.0,
        "max_steps": 4,
        "max_grad_norm": 1.0,
        "validation_split_ratio": 0.0,
        "load_best_model_at_end": False,
        "dataset_num_proc": 1,
        "dataloader_num_workers": 0,
        "frozen_mlp_activation_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "dtype": "bfloat16",
        "bf16": True,
        "tf32": True,
        "train_sampler_seed": None,
        "ignore_data_skip": False,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_contrast_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "memory_causal_weight": 0.0,
        "memory_anchor_weight": 0.0,
        "memory_recover_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "scene_state_generation_objective_version": launch.OBJECTIVE_VERSION,
        "scene_state_generated_unlikelihood_weight": 0.0,
        "scene_state_generated_unlikelihood_max_wrong_tokens": 0,
        "scene_state_generated_prefix_correction_weight": 0.0,
        "scene_state_generated_rollout_extra_tokens": 4,
        "scene_state_generated_rollout_max_tokens": 24,
    }
    for name, value in locked_values.items():
        setattr(args, name, value)
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_source_manifest_identity",
        lambda _args: {"schema": "synthetic"},
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_identity_protocol_pairing_summary",
        lambda _manifest: {},
    )
    protocol = experimental_train.build_training_protocol(
        args,
        Dataset.from_dict({"input_ids": [[1]]}),
        effective_training_mode="episode",
        train_samples=1,
        eval_samples=0,
        warmup_steps=0,
        scene_state_identity_pairing_manifest={},
        train_schedule_binding=binding,
    )
    return json.loads(json.dumps(protocol)), data


def test_persisted_trainer_built_v13_dense_protocol_passes_launch_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, data = _v13_protocol_fixture(monkeypatch, tmp_path)

    launch._validate_checkpoint_protocol(protocol, data=data)
    expected_dense_fields = {
        "scene_generation_dense_semantic_mode": launch.DENSE_SEMANTIC_MODE,
        "scene_generation_dense_decision_scope": launch.DENSE_DECISION_SCOPE,
        "scene_generation_dense_decision_token_overlap_policy": (
            "supervise_whole_token_if_any_character_overlaps_boundary_decision_char_v1"
        ),
        "scene_generation_dense_decision_ce_weight": 1.0,
        "scene_generation_dense_top1_retention_hinge_weight": 1.0,
        "scene_generation_dense_top1_retention_hinge_mode": (
            "dense_gold_vs_detached_top_competitor_hinge_v1"
        ),
        "scene_generation_dense_top1_retention_margin": launch.SEMANTIC_MARGIN,
        "scene_generation_failed_semantic_repair_ce_weight": 1.0,
        "scene_generation_failed_semantic_repair_hinge_weight": 1.0,
        "scene_generation_failed_semantic_repair_margin": launch.SEMANTIC_MARGIN,
        "scene_generation_failed_decision_alignment": (
            launch.FAILED_DECISION_ALIGNMENT_MODE
        ),
        "scene_generation_schema_ce_optimization_scope": (
            "standalone_schema_mask_partition_only_v1"
        ),
    }
    for field, expected in expected_dense_fields.items():
        assert protocol[field] == expected


def test_v13_launcher_is_valid_and_binds_dry_run_and_critical_files() -> None:
    script_path = Path(launch.__file__).with_name("train_scene_memory_v13.sh")
    script = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    for required in (
        "--target-step 4",
        "--gradient-accumulation-steps 7",
        "--learning-rate 1e-4",
        "--lr-scheduler-type constant",
        "--warmup-steps 0",
        "--max-steps 4",
        "--save-total-limit 4",
        'CHECKPOINT1_DIR="${OUTPUT_DIR}/trainer/checkpoint-1"',
        'CHECKPOINT2_DIR="${OUTPUT_DIR}/trainer/checkpoint-2"',
        'CHECKPOINT3_DIR="${OUTPUT_DIR}/trainer/checkpoint-3"',
        'CHECKPOINT4_DIR="${OUTPUT_DIR}/trainer/checkpoint-4"',
        "status --porcelain --untracked-files=no",
        "ls-files --error-unmatch",
        "critical_v13_source_must_be_tracked",
    ):
        assert required in script
    assert "--resume-from-checkpoint" not in script

    dry_run_guard = script.index('if [[ "${DRY_RUN}" == "1" ]]')
    dry_run_exit = script.index("  exit 0", dry_run_guard)
    first_mutating_mkdir = script.index("mkdir -p \\\n", dry_run_exit)
    assert dry_run_guard < dry_run_exit < first_mutating_mkdir

    array_match = re.search(
        r"critical_tracked_files=\(\n(?P<body>.*?)\n\)",
        script,
        flags=re.DOTALL,
    )
    assert array_match is not None
    shell_critical_files = tuple(
        re.findall(r'^\s+"([^"]+)"\s*$', array_match.group("body"), re.MULTILINE)
    )
    assert shell_critical_files == tuple(launch.CRITICAL_TRAINING_FILES)
    for required_path in (
        "deltamem/scene_boundary.py",
        "deltamem/train/delta_sft_experimental.py",
        "deltamem/train/scene_state_generation_alignment.py",
        "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_warm_start.py",
        "experiments/rethinking_rwkv_ms_gemma/scene_memory_v13_launch_contract.py",
        "experiments/rethinking_rwkv_ms_gemma/run_scene_memory_v13_gate.py",
        "experiments/rethinking_rwkv_ms_gemma/train_scene_memory_v13.sh",
    ):
        assert required_path in shell_critical_files

    evaluator_binding = gate.evaluator_code_binding()
    assert evaluator_binding["scene_boundary_metric"]["path"] == str(
        (launch.PROJECT_ROOT / "deltamem/scene_boundary.py").resolve()
    )
