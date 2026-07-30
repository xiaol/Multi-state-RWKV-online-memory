from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.tests.test_scene_memory_v10_contract import _data as _v10_data
from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v12_gate as gate
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v12_launch_contract as launch
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v12_warm_start as warm


def test_v12_native_training_horizon_and_objective_are_locked() -> None:
    assert warm.WARM_START_MODE == "scene_memory_v12_v8_checkpoint56_adapter_only"
    assert warm.RECEIPT_SCHEMA == (
        "rwkv_ms_scene_memory_v12_adapter_warm_start_receipt.v1"
    )
    assert launch.OBJECTIVE_SCHEMA_VERSION == 15
    assert launch.OBJECTIVE_VERSION.endswith("semantic_boundary_margin_v12")
    assert launch.FIXED_SAMPLER_MODE == (
        "explicit_ordered_v12_two_canonical_seven_pair_cycles_v1"
    )
    assert launch.TOTAL_PAIR_PRESENTATIONS == 14
    assert launch.TOTAL_OPTIMIZER_STEPS == 2
    assert launch.CHECKPOINT_STEPS == (1, 2)
    assert launch.PRESENTATION_CHECKPOINTS == (7, 14)
    assert launch.GRADIENT_ACCUMULATION_STEPS == 7
    assert launch.LEARNING_RATE == 2e-5
    assert launch.WARMUP_STEPS == 0
    assert launch.PREFIX_CORRECTION_WEIGHT == 0.0
    assert launch.ROW_OBJECTIVE_AUDIT_FILENAME in (
        launch.REQUIRED_CHECKPOINT_ARTIFACTS
    )


def test_v12_exact_two_cycle_schedule_is_not_a_repeated_v11_cycle() -> None:
    assert launch.TWO_CYCLE_PAIRS == (
        (3, 24),
        (19, 28),
        (20, 31),
        (10, 23),
        (1, 14),
        (5, 9),
        (22, 26),
        (19, 28),
        (22, 26),
        (5, 9),
        (3, 24),
        (20, 31),
        (10, 23),
        (1, 14),
    )
    assert launch.canonical_sha256(
        [list(pair) for pair in launch.TWO_CYCLE_PAIRS]
    ) == launch.TWO_CYCLE_PAIRS_SHA256
    assert launch.presentation_cursor(0) == 0
    assert launch.presentation_cursor(1) == 7
    assert launch.presentation_cursor(2) == 14
    with pytest.raises(launch.LaunchContractError):
        launch.presentation_cursor(3)


def test_v12_fresh_start_is_version_native_and_forbids_resume() -> None:
    receipt = warm.validate_v12_fresh_start_contract(
        warm.V12FreshStartContract(
            resume_from_checkpoint=None,
            initial_global_step=0,
            optimizer_created=False,
            scheduler_created=False,
            trainer_state_imported=False,
            rng_state_imported=False,
            optim="adamw_torch_fused",
        )
    )
    assert receipt["rng_state"] == "fresh_from_v12_seed"
    with pytest.raises(ValueError, match="forbids checkpoint resume"):
        warm.validate_v12_fresh_start_contract(
            warm.V12FreshStartContract(
                resume_from_checkpoint="checkpoint-1",
                initial_global_step=0,
                optimizer_created=False,
                scheduler_created=False,
                trainer_state_imported=False,
                rng_state_imported=False,
                optim="adamw_torch_fused",
            )
        )


@pytest.mark.parametrize(
    "path",
    (
        "/tmp/test.jsonl",
        "/tmp/val.jsonl",
        "/tmp/validation.jsonl",
        "/tmp/full170.jsonl",
        "/tmp/hard32-copy/train.jsonl",
    ),
)
def test_v12_training_data_guard_rejects_protected_splits(path: str) -> None:
    with pytest.raises(
        launch.LaunchContractError,
        match="protected_split_path_forbidden",
    ):
        launch.guard_v12_training_data_path(path, description="test_data")


def test_v12_training_data_guard_allows_value14_and_pinned_train32() -> None:
    value14 = Path("/tmp/value14/pair_schedule.jsonl")
    assert launch.guard_v12_training_data_path(
        value14,
        description="value14",
    ) == value14
    historical = Path(
        launch.PINNED_HISTORICAL_TRAIN32_ARTIFACTS["train32"]["path"]
    )
    assert launch.guard_v12_training_data_path(
        historical,
        description="historical_train32",
    ) == historical


def _cycle_log(step: int, pairs: tuple[tuple[int, int], ...]) -> dict[str, float]:
    result = {
        "step": float(step),
        "delta/scene_generation_v12_cycle_index": float(step),
        "delta/scene_generation_v12_cycle_pair_presentations": 7.0,
    }
    for index, (low, high) in enumerate(pairs):
        result[
            f"delta/scene_generation_v12_cycle_pair_{index}_low_ordinal"
        ] = float(low)
        result[
            f"delta/scene_generation_v12_cycle_pair_{index}_high_ordinal"
        ] = float(high)
    return result


def test_v12_cycle_telemetry_binds_both_optimizer_checkpoints() -> None:
    trainer_state = {
        "log_history": [
            _cycle_log(1, launch.FIRST_CYCLE_PAIRS),
            _cycle_log(2, launch.SECOND_CYCLE_PAIRS),
        ]
    }
    checkpoint1 = launch.validate_v12_cycle_pair_telemetry(
        {"log_history": trainer_state["log_history"][:1]},
        checkpoint_step=1,
    )
    checkpoint2 = launch.validate_v12_cycle_pair_telemetry(
        trainer_state,
        checkpoint_step=2,
    )
    assert checkpoint1["pair_presentations"] == 7
    assert checkpoint2["pair_presentations"] == 14
    assert checkpoint2["ordered_pairs"] == [
        list(pair) for pair in launch.TWO_CYCLE_PAIRS
    ]


def _v12_row_hash(ordinal: int) -> str:
    return f"{ordinal:064x}"


def _v12_row_audit_data() -> dict[str, Any]:
    data = _v10_data()
    data["entries"] = [
        {
            "canonical_pair_ordinals": [source, donor],
            "members": [
                {
                    "train_row_ordinal": source,
                    "donor_train_row_ordinal": donor,
                    "row_sha256": _v12_row_hash(source),
                    "donor_row_sha256": _v12_row_hash(donor),
                },
                {
                    "train_row_ordinal": donor,
                    "donor_train_row_ordinal": source,
                    "row_sha256": _v12_row_hash(donor),
                    "donor_row_sha256": _v12_row_hash(source),
                },
            ],
        }
        for source, donor in launch.TWO_CYCLE_PAIRS
    ]
    return data


def _v12_row_audit_payload(checkpoint_step: int) -> dict[str, Any]:
    row_order = [
        ordinal for pair in launch.FIRST_CYCLE_PAIRS for ordinal in pair
    ]
    rows = {ordinal: {"row_ordinal": ordinal} for ordinal in row_order}
    presentation_count = checkpoint_step * launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
    for presentation_index, (source, donor) in enumerate(
        launch.TWO_CYCLE_PAIRS[:presentation_count]
    ):
        phase_index = presentation_index // launch.PAIR_PRESENTATIONS_PER_OPTIMIZER_STEP
        phase = "cycle1_input" if phase_index == 0 else "cycle2_input"
        for pair_role, row_ordinal, paired_row_ordinal in (
            ("source", source, donor),
            ("donor", donor, source),
        ):
            parsed_exact = pair_role == "source"
            raw_exact = parsed_exact and phase_index == 0
            selected_ce = 0.7
            selected_hinge = 0.6
            rows[row_ordinal][phase] = {
                "phase": phase,
                "cycle": phase_index + 1,
                "adapter_optimizer_step_before_update": phase_index,
                "presentation": presentation_index + 1,
                "pair_role": pair_role,
                "row_ordinal": row_ordinal,
                "paired_row_ordinal": paired_row_ordinal,
                "row_sha256": _v12_row_hash(row_ordinal),
                "paired_row_sha256": _v12_row_hash(paired_row_ordinal),
                "parsed_boundary_exact": parsed_exact,
                "raw_token_exact": raw_exact,
                "first_divergence": 8 if raw_exact else 2,
                "first_relevant_decision_ordinal": 0 if parsed_exact else 1,
                "selected_decision_ordinal": 1,
                "selected_label_position": 10 + row_ordinal,
                "gold_token_id": 100 + row_ordinal,
                "competitor_token_id": 200 + row_ordinal,
                "competitor_is_actual_greedy": not parsed_exact,
                "selected_is_termination": (
                    not parsed_exact and row_ordinal % 2 == 0
                ),
                "gold_vs_competitor_margin": 0.4,
                "selected_ce": selected_ce,
                "selected_hinge": selected_hinge,
                "selected_objective_loss": (
                    selected_hinge
                    if parsed_exact
                    else selected_ce + selected_hinge
                ),
                "relevant_decision_count": 3 if parsed_exact else 2,
                "rollout_token_count": 8,
                "failed_replay_generated_cursor": -1 if parsed_exact else 2,
                "failed_alignment_kind_code": (
                    -1 if parsed_exact else row_ordinal % 3
                ),
            }
    return {
        "schema": launch.ROW_OBJECTIVE_AUDIT_SCHEMA,
        "memory_objective_version": launch.OBJECTIVE_VERSION,
        "checkpoint_optimizer_step": checkpoint_step,
        "completed_pair_presentations": presentation_count,
        "phases": ["cycle1_input"]
        + (["cycle2_input"] if checkpoint_step == 2 else []),
        "pair_schedule": [
            {"source_row_ordinal": source, "donor_row_ordinal": donor}
            for source, donor in launch.TWO_CYCLE_PAIRS[:presentation_count]
        ],
        "rows": [rows[ordinal] for ordinal in row_order],
    }


@pytest.mark.parametrize("checkpoint_step", (1, 2))
def test_v12_row_objective_audit_binds_full_emitted_schema(
    checkpoint_step: int,
) -> None:
    payload = _v12_row_audit_payload(checkpoint_step)

    assert launch._validate_v12_row_objective_audit(
        payload,
        checkpoint_step=checkpoint_step,
        data=_v12_row_audit_data(),
    ) == payload


@pytest.mark.parametrize(
    "missing_field",
    sorted(launch._V12_ROW_AUDIT_OBSERVATION_FIELDS),
)
def test_v12_row_objective_audit_rejects_every_omitted_observation_field(
    missing_field: str,
) -> None:
    payload = _v12_row_audit_payload(1)
    del payload["rows"][0]["cycle1_input"][missing_field]

    with pytest.raises(
        launch.LaunchContractError,
        match="v12_row_objective_audit_observation_fields_differ",
    ):
        launch._validate_v12_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v12_row_audit_data(),
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("phase", "wrong_phase"),
        ("cycle", 0),
        ("adapter_optimizer_step_before_update", 1),
        ("presentation", 0),
        ("pair_role", "donor"),
        ("row_ordinal", 24),
        ("paired_row_ordinal", 3),
        ("row_sha256", "f" * 64),
        ("paired_row_sha256", "e" * 64),
        ("parsed_boundary_exact", False),
        ("raw_token_exact", 1),
        ("first_divergence", 9),
        ("first_relevant_decision_ordinal", 1),
        ("selected_decision_ordinal", 3),
        ("selected_label_position", 0),
        ("gold_token_id", -1),
        ("competitor_token_id", 103),
        ("competitor_is_actual_greedy", True),
        ("selected_is_termination", True),
        ("gold_vs_competitor_margin", float("nan")),
        ("selected_ce", -0.1),
        ("selected_hinge", -0.1),
        ("selected_objective_loss", -0.1),
        ("relevant_decision_count", 0),
        ("rollout_token_count", 25),
        ("failed_replay_generated_cursor", 0),
        ("failed_alignment_kind_code", 0),
    ),
)
def test_v12_row_objective_audit_rejects_mutated_observation_field(
    field: str,
    invalid_value: Any,
) -> None:
    payload = _v12_row_audit_payload(1)
    payload["rows"][0]["cycle1_input"][field] = invalid_value

    with pytest.raises(launch.LaunchContractError, match="v12_row_objective_audit_"):
        launch._validate_v12_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v12_row_audit_data(),
        )


@pytest.mark.parametrize("container", ("top_level", "row", "observation"))
def test_v12_row_objective_audit_rejects_extra_schema_fields(container: str) -> None:
    payload = _v12_row_audit_payload(1)
    target = {
        "top_level": payload,
        "row": payload["rows"][0],
        "observation": payload["rows"][0]["cycle1_input"],
    }[container]
    target["unexpected"] = True

    with pytest.raises(launch.LaunchContractError, match="v12_row_objective_audit_"):
        launch._validate_v12_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_v12_row_audit_data(),
        )


def _gate_records() -> dict[str, list[dict[str, Any]]]:
    return {
        condition: [
            {
                "condition": condition,
                "train_row_ordinal": ordinal,
                "gold": {"boundaries": [ordinal]},
                "parsed_json": {"boundaries": [ordinal]},
                "raw_generation": "same-zero" if condition == "state_only_no_write" else str(ordinal),
            }
            for ordinal in gate.VALUE14_ORDINALS
        ]
        for condition in gate.CONDITIONS
    }


def _pairing() -> dict[str, Any]:
    donor = {
        low: high
        for low, high in launch.FIRST_CYCLE_PAIRS
    }
    donor.update({high: low for low, high in launch.FIRST_CYCLE_PAIRS})
    return {
        "directed_pairs": [
            {
                "train_row_ordinal": ordinal,
                "donor_train_row_ordinal": donor[ordinal],
                "target_stratum": (
                    "cross_cardinality_value"
                    if ordinal in {1, 14, 22, 26}
                    else "same_cardinality_value"
                ),
            }
            for ordinal in gate.VALUE14_ORDINALS
        ]
    }


def test_v12_gate_accepts_both_checkpoints_but_only_exact_value14(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = gate.v10.v9.v8
    monkeypatch.setattr(metrics, "_strict_score", lambda _record: {"tp": 1, "fp": 0, "fn": 0})
    monkeypatch.setattr(metrics, "_strict_exact", lambda _score: 1)
    monkeypatch.setattr(metrics, "_micro_f1", lambda _scores: 1.0)
    monkeypatch.setattr(metrics, "_strictly_better", lambda *_args: True)
    monkeypatch.setattr(metrics, "is_canonical_scene_prediction", lambda _value: True)
    monkeypatch.setattr(metrics, "score_prediction", lambda *_args: {"tp": 1, "fp": 0, "fn": 0})
    identity = {
        "bidirectional_identity_switch_rows": 14,
        "correct_state_beats_donor_state_on_source_token_rows": 14,
        "correct_state_prefers_source_token_rows": 14,
        "donor_state_prefers_donor_token_rows": 14,
        "correct_state_beats_zero_on_source_token_rows": 14,
    }
    monkeypatch.setattr(
        metrics,
        "build_value14_selected_token_evidence",
        lambda **_kwargs: {"overall": identity, "rows": []},
    )

    records = _gate_records()
    for checkpoint_step in launch.CHECKPOINT_STEPS:
        result = gate.build_v12_gate(
            records_by_condition=records,
            pairing=_pairing(),
            checkpoint_step=checkpoint_step,
        )
        assert result["checkpoint_step"] == checkpoint_step
        assert result["consumed_pair_presentations"] == checkpoint_step * 7
        assert result["evaluation_scope"] == "exact_value14_ordinals_only"

    records[gate.CONDITIONS[0]].append(
        {
            "condition": gate.CONDITIONS[0],
            "train_row_ordinal": 0,
        }
    )
    with pytest.raises(gate.V12EvaluationContractError, match="exactly 14"):
        gate.build_v12_gate(
            records_by_condition=records,
            pairing=_pairing(),
            checkpoint_step=1,
        )


def test_v12_launcher_pins_two_step_settings_and_both_checkpoint_receipts() -> None:
    script = (
        Path(launch.__file__).with_name("train_scene_memory_v12.sh").read_text(
            encoding="utf-8"
        )
    )
    for required in (
        "--target-step 2",
        "--gradient-accumulation-steps 7",
        "--learning-rate 2e-5",
        "--lr-scheduler-type constant",
        "--warmup-steps 0",
        "--max-steps 2",
        "--save-steps",
        "--save-total-limit 2",
        'CHECKPOINT1_DIR="${OUTPUT_DIR}/trainer/checkpoint-1"',
        'CHECKPOINT2_DIR="${OUTPUT_DIR}/trainer/checkpoint-2"',
    ):
        assert required in script
    assert "scene_state_generation_ce_symmetric_cycle_suffix_repair_v5" not in script


def test_v12_launcher_argv_is_accepted_with_locked_zero_correction_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        experimental_train,
        "_validate_scene_v12_warm_start_args",
        lambda _args: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_source_manifest_identity",
        lambda _args: {"schema": "synthetic"},
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_generation_pairing_binding",
        lambda _args: {"schema": "synthetic"},
    )
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
            "--warm-start-from-checkpoint",
            str(tmp_path / "checkpoint-56"),
            "--warm-start-mode",
            warm.WARM_START_MODE,
            "--memory-loss-mode",
            "scene_state_generation_ce",
            "--episode-recent-messages",
            "0",
            "--assistant-loss-mode",
            "final_assistant_only",
            "--memory-kl-weight",
            "0",
            "--gradient-accumulation-steps",
            "7",
            "--scene-state-generation-objective-version",
            launch.OBJECTIVE_VERSION,
            "--scene-state-generated-prefix-correction-weight",
            "0",
            "--scene-state-generated-unlikelihood-max-wrong-tokens",
            "0",
        ],
    )

    args = experimental_train.parse_args()

    assert args.scene_state_generation_objective_version == launch.OBJECTIVE_VERSION
    assert args.scene_state_generated_prefix_correction_weight == 0.0
    assert args.scene_state_generated_unlikelihood_max_wrong_tokens == 0


def test_v12_shared_parser_is_bound_for_training_and_evaluation() -> None:
    parser_path = "deltamem/scene_boundary.py"
    script = Path(launch.__file__).with_name("train_scene_memory_v12.sh").read_text(
        encoding="utf-8"
    )

    assert parser_path in launch.CRITICAL_TRAINING_FILES
    assert f'  "{parser_path}"' in script
    evaluator_binding = gate.evaluator_code_binding()
    assert evaluator_binding["scene_boundary_metric"]["path"] == str(
        (launch.PROJECT_ROOT / parser_path).resolve()
    )


def test_persisted_built_v12_protocol_passes_launch_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    scheduled_pairs = [
        *launch.TWO_CYCLE_PAIRS,
        *launch.TWO_CYCLE_PAIRS,
    ]
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
        "save_total_limit": 2,
        "num_train_epochs": 1.0,
        "max_steps": 2,
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
    persisted_protocol = json.loads(json.dumps(protocol))

    launch._validate_checkpoint_protocol(persisted_protocol, data=data)
    assert persisted_protocol["scene_generation_exact_value_mask_mode"] == (
        launch.EXACT_VALUE_MASK_MODE
    )
    assert persisted_protocol["scene_generation_exact_retention_scope"] == (
        launch.EXACT_RETENTION_SCOPE
    )
