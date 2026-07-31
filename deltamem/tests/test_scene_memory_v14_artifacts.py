from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import subprocess
from typing import Any

import pytest

from experiments.rethinking_rwkv_ms_gemma import run_scene_memory_v14_gate as gate
from experiments.rethinking_rwkv_ms_gemma import (
    scene_memory_v14_launch_contract as launch,
)
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v14_warm_start as warm


def test_v14_cached_objective_and_native_horizon_are_locked() -> None:
    assert warm.WARM_START_MODE == (
        "scene_memory_v14_v13_checkpoint4_adapter_only"
    )
    assert warm.SOURCE_IMPORT_POLICY == {
        "adapter": True,
        "optimizer": False,
        "scheduler": False,
        "trainer_state": False,
        "rng": False,
        "global_step": False,
    }
    assert warm.TARGET_FRESH_START_POLICY["rng_state"] == "fresh_from_v14_seed"
    assert launch.OBJECTIVE_SCHEMA_VERSION == 17
    assert launch.OBJECTIVE_VERSION == (
        "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14"
    )
    assert launch.CACHED_PREFIX_MODE == (
        "cached_actual_greedy_prefix_failed_repair_cached_gold_prefix_all_"
        "decision_retention_v1"
    )
    assert launch.FAILED_REPLAY_MODE == (
        "use_cache_true_logits_to_keep_1_actual_greedy_prefix_v2"
    )
    assert launch.EXACT_REPLAY_MODE == (
        "use_cache_true_logits_to_keep_1_gold_prefix_v1"
    )
    assert launch.REPLAY_LOGITS_TO_KEEP == 1
    assert "raw_token_exact=telemetry_only" in launch.OBJECTIVE_FORMULA
    assert "full_answer_schema_footer_and_chat_termination_ce=0" in (
        launch.OBJECTIVE_FORMULA
    )
    assert launch.TOTAL_PAIR_PRESENTATIONS == 28
    assert launch.TOTAL_OPTIMIZER_STEPS == 4
    assert launch.CHECKPOINT_STEPS == (1, 2, 3, 4)
    assert launch.PRESENTATION_CHECKPOINTS == (7, 14, 21, 28)
    assert launch.GRADIENT_ACCUMULATION_STEPS == 7
    assert launch.LEARNING_RATE == 1e-4
    assert launch.PREFIX_CORRECTION_WEIGHT == 0.0
    assert launch.ROW_OBJECTIVE_AUDIT_FILENAME in (
        launch.REQUIRED_CHECKPOINT_ARTIFACTS
    )


def test_v14_exact_four_cycle_schedule_and_cursors_are_locked() -> None:
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
        "delta/scene_generation_v14_cycle_index": float(step),
        "delta/scene_generation_v14_cycle_pair_presentations": 7.0,
    }
    for index, (low, high) in enumerate(pairs):
        result[f"delta/scene_generation_v14_cycle_pair_{index}_low_ordinal"] = (
            float(low)
        )
        result[f"delta/scene_generation_v14_cycle_pair_{index}_high_ordinal"] = (
            float(high)
        )
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
def test_v14_cycle_telemetry_binds_every_checkpoint(checkpoint_step: int) -> None:
    result = launch.validate_v14_cycle_pair_telemetry(
        _trainer_state(checkpoint_step),
        checkpoint_step=checkpoint_step,
    )

    assert result["optimizer_step"] == checkpoint_step
    assert result["pair_presentations"] == checkpoint_step * 7
    assert result["ordered_pairs"] == [
        list(pair) for pair in launch.FOUR_CYCLE_PAIRS[: checkpoint_step * 7]
    ]
    assert result["ordered_pairs_sha256"] == (
        launch.PAIR_PREFIX_SHA256_BY_CHECKPOINT[checkpoint_step]
    )
    assert len(result["cycles"]) == checkpoint_step


@pytest.mark.parametrize("checkpoint_step", launch.CHECKPOINT_STEPS)
def test_v14_cycle_telemetry_rejects_pair_tampering(
    checkpoint_step: int,
) -> None:
    trainer_state = _trainer_state(checkpoint_step)
    key = "delta/scene_generation_v14_cycle_pair_0_low_ordinal"
    trainer_state["log_history"][-1][key] += 1.0

    with pytest.raises(launch.LaunchContractError, match="order_differs"):
        launch.validate_v14_cycle_pair_telemetry(
            trainer_state,
            checkpoint_step=checkpoint_step,
        )


_ROW_FIELDS = frozenset(
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
        "cached_branch_kind",
        "cached_branch_kind_code",
        "cached_replay_use_cache",
        "cached_replay_logits_to_keep",
        "cached_replay_token_count",
        "cached_replay_selected_cursor",
        "cached_decision_token_count",
        "cached_selected_decision_ordinal",
        "cached_selected_label_position",
        "cached_selected_gold_token_id",
        "cached_selected_competitor_id",
        "cached_competitor_is_actual_greedy",
        "cached_replay_top1_matches_actual",
        "cached_replay_top1_match_count",
        "cached_ce",
        "cached_failed_competitor_hinge",
        "cached_exact_retention_hinge",
        "cached_selected_gold_vs_competitor_margin",
        "cached_gold_top1_fraction",
        "cached_alignment_kind_code",
        "cached_selected_is_termination",
        "cached_branch_loss",
        "auxiliary_optimization_loss",
        "auxiliary_telemetry_loss",
        "selected_top_competitor_hinge_telemetry",
        "selected_correct_vs_zero_hinge_telemetry",
        "total_side_loss",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "phase",
        "cycle",
        "adapter_optimizer_step_before_update",
        "presentation",
        "source_row_ordinal",
        "donor_row_ordinal",
        "source_row_sha256",
        "donor_row_sha256",
        "pair_mean_cached_branch_loss",
        "pair_mean_cached_exact_retention_hinge",
        "pair_mean_cached_failed_ce",
        "pair_mean_cached_failed_competitor_hinge",
        "pair_mean_auxiliary_optimization_loss",
        "pair_mean_selected_top_competitor_hinge_telemetry",
        "pair_mean_selected_correct_vs_zero_hinge_telemetry",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    }
)


def _row_hash(ordinal: int) -> str:
    return f"{ordinal:064x}"


def _audit_data() -> dict[str, Any]:
    return {
        "entries": [
            {
                "canonical_pair_ordinals": [source, donor],
                "members": [
                    {
                        "train_row_ordinal": source,
                        "donor_train_row_ordinal": donor,
                        "row_sha256": _row_hash(source),
                        "donor_row_sha256": _row_hash(donor),
                    },
                    {
                        "train_row_ordinal": donor,
                        "donor_train_row_ordinal": source,
                        "row_sha256": _row_hash(donor),
                        "donor_row_sha256": _row_hash(source),
                    },
                ],
            }
            for source, donor in launch.FOUR_CYCLE_PAIRS
        ]
    }


def _row_observation(
    *,
    cycle: int,
    presentation: int,
    pair_role: str,
    row_ordinal: int,
    paired_row_ordinal: int,
) -> dict[str, Any]:
    parsed_exact = pair_role == "source"
    cached_ce = 0.0 if parsed_exact else 0.4
    failed_hinge = 0.0 if parsed_exact else 0.5
    exact_hinge = 0.3 if parsed_exact else 0.0
    branch_loss = cached_ce + failed_hinge + exact_hinge
    return {
        "phase": f"cycle{cycle}_input",
        "cycle": cycle,
        "adapter_optimizer_step_before_update": cycle - 1,
        "presentation": presentation,
        "pair_role": pair_role,
        "row_ordinal": row_ordinal,
        "paired_row_ordinal": paired_row_ordinal,
        "row_sha256": _row_hash(row_ordinal),
        "paired_row_sha256": _row_hash(paired_row_ordinal),
        "parsed_boundary_exact": parsed_exact,
        "raw_token_exact": False,
        "first_divergence": 8 if parsed_exact else 2,
        "rollout_token_count": 8,
        "cached_branch_kind": (
            "cached_gold_prefix"
            if parsed_exact
            else "cached_actual_greedy_prefix"
        ),
        "cached_branch_kind_code": 0 if parsed_exact else 1,
        "cached_replay_use_cache": True,
        "cached_replay_logits_to_keep": 1,
        "cached_replay_token_count": 3,
        "cached_replay_selected_cursor": 2,
        "cached_decision_token_count": 3,
        "cached_selected_decision_ordinal": 1,
        "cached_selected_label_position": 10 + row_ordinal,
        "cached_selected_gold_token_id": 100 + row_ordinal,
        "cached_selected_competitor_id": 200 + row_ordinal,
        "cached_competitor_is_actual_greedy": not parsed_exact,
        "cached_replay_top1_matches_actual": not parsed_exact,
        "cached_replay_top1_match_count": 0 if parsed_exact else 3,
        "cached_ce": cached_ce,
        "cached_failed_competitor_hinge": failed_hinge,
        "cached_exact_retention_hinge": exact_hinge,
        "cached_selected_gold_vs_competitor_margin": 0.5,
        "cached_gold_top1_fraction": 0.75 if parsed_exact else 0.25,
        "cached_alignment_kind_code": -1 if parsed_exact else 0,
        "cached_selected_is_termination": False,
        "cached_branch_loss": branch_loss,
        "auxiliary_optimization_loss": 0.0,
        "auxiliary_telemetry_loss": 0.3,
        "selected_top_competitor_hinge_telemetry": (
            0.2 if parsed_exact else 0.4
        ),
        "selected_correct_vs_zero_hinge_telemetry": (
            0.1 if parsed_exact else 0.3
        ),
        "total_side_loss": branch_loss,
    }


def _pair_observation(
    *,
    cycle: int,
    presentation: int,
    source: int,
    donor: int,
) -> dict[str, Any]:
    exact_hinge = 0.15
    failed_ce = 0.2
    failed_hinge = 0.25
    total = exact_hinge + failed_ce + failed_hinge
    return {
        "phase": f"cycle{cycle}_input",
        "cycle": cycle,
        "adapter_optimizer_step_before_update": cycle - 1,
        "presentation": presentation,
        "source_row_ordinal": source,
        "donor_row_ordinal": donor,
        "source_row_sha256": _row_hash(source),
        "donor_row_sha256": _row_hash(donor),
        "pair_mean_cached_branch_loss": total,
        "pair_mean_cached_exact_retention_hinge": exact_hinge,
        "pair_mean_cached_failed_ce": failed_ce,
        "pair_mean_cached_failed_competitor_hinge": failed_hinge,
        "pair_mean_auxiliary_optimization_loss": 0.0,
        "pair_mean_selected_top_competitor_hinge_telemetry": 0.3,
        "pair_mean_selected_correct_vs_zero_hinge_telemetry": 0.2,
        "pair_mean_total_side_loss": total,
        "reported_objective_total_loss": total,
        "recomputed_objective_total_loss": total,
    }


def _row_audit(checkpoint_step: int) -> dict[str, Any]:
    row_order = [ordinal for pair in launch.FIRST_CYCLE_PAIRS for ordinal in pair]
    rows = {ordinal: {"row_ordinal": ordinal} for ordinal in row_order}
    pair_presentations = []
    presentation_count = checkpoint_step * 7
    for presentation, (source, donor) in enumerate(
        launch.FOUR_CYCLE_PAIRS[:presentation_count],
        1,
    ):
        cycle = (presentation - 1) // 7 + 1
        phase = f"cycle{cycle}_input"
        rows[source][phase] = _row_observation(
            cycle=cycle,
            presentation=presentation,
            pair_role="source",
            row_ordinal=source,
            paired_row_ordinal=donor,
        )
        rows[donor][phase] = _row_observation(
            cycle=cycle,
            presentation=presentation,
            pair_role="donor",
            row_ordinal=donor,
            paired_row_ordinal=source,
        )
        pair_presentations.append(
            _pair_observation(
                cycle=cycle,
                presentation=presentation,
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
def test_v14_row_objective_audit_binds_cached_branches_and_horizon(
    checkpoint_step: int,
) -> None:
    payload = _row_audit(checkpoint_step)

    assert set(payload["rows"][0]["cycle1_input"]) == _ROW_FIELDS
    assert set(payload["pair_presentations"][0]) == _PAIR_FIELDS
    assert launch._validate_v14_row_objective_audit(
        payload,
        checkpoint_step=checkpoint_step,
        data=_audit_data(),
    ) == {
        "schema": launch.ROW_OBJECTIVE_AUDIT_SCHEMA,
        "checkpoint_optimizer_step": checkpoint_step,
        "completed_pair_presentations": checkpoint_step * 7,
        "pair_schedule_sha256": launch.canonical_sha256(payload["pair_schedule"]),
        "rows": 14,
        "pair_presentations": checkpoint_step * 7,
    }


@pytest.mark.parametrize("missing_field", sorted(_ROW_FIELDS))
def test_v14_row_audit_rejects_every_omitted_cached_row_field(
    missing_field: str,
) -> None:
    payload = _row_audit(1)
    del payload["rows"][0]["cycle1_input"][missing_field]

    with pytest.raises(launch.LaunchContractError, match="row_fields_differ"):
        launch._validate_v14_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_audit_data(),
        )


@pytest.mark.parametrize("missing_field", sorted(_PAIR_FIELDS))
def test_v14_row_audit_rejects_every_omitted_pair_field(
    missing_field: str,
) -> None:
    payload = _row_audit(1)
    del payload["pair_presentations"][0][missing_field]

    with pytest.raises(launch.LaunchContractError, match="pair_fields_differ"):
        launch._validate_v14_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_audit_data(),
        )


@pytest.mark.parametrize(
    ("row_index", "field"),
    (
        (0, "cached_exact_retention_hinge"),
        (0, "cached_branch_loss"),
        (0, "total_side_loss"),
        (1, "cached_ce"),
        (1, "cached_failed_competitor_hinge"),
        (1, "cached_branch_loss"),
        (1, "total_side_loss"),
    ),
)
def test_v14_row_audit_rejects_cached_branch_arithmetic_tampering(
    row_index: int,
    field: str,
) -> None:
    payload = _row_audit(1)
    payload["rows"][row_index]["cycle1_input"][field] += 0.125

    with pytest.raises(launch.LaunchContractError, match="arithmetic"):
        launch._validate_v14_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_audit_data(),
        )


@pytest.mark.parametrize(
    "field",
    (
        "pair_mean_cached_branch_loss",
        "pair_mean_cached_exact_retention_hinge",
        "pair_mean_cached_failed_ce",
        "pair_mean_cached_failed_competitor_hinge",
        "pair_mean_auxiliary_optimization_loss",
        "pair_mean_total_side_loss",
        "reported_objective_total_loss",
        "recomputed_objective_total_loss",
    ),
)
def test_v14_row_audit_rejects_pair_arithmetic_tampering(field: str) -> None:
    payload = _row_audit(1)
    payload["pair_presentations"][0][field] += 0.125

    with pytest.raises(launch.LaunchContractError):
        launch._validate_v14_row_objective_audit(
            payload,
            checkpoint_step=1,
            data=_audit_data(),
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
def test_v14_fresh_start_forbids_resume_and_imported_training_state(
    field: str,
    value: Any,
    message: str,
) -> None:
    base = warm.V14FreshStartContract(
        resume_from_checkpoint=None,
        initial_global_step=0,
        optimizer_created=False,
        scheduler_created=False,
        trainer_state_imported=False,
        rng_state_imported=False,
        optim="adamw_torch_fused",
    )
    assert warm.validate_v14_fresh_start_contract(base) == {
        "initial_global_step": 0,
        "optimizer_implementation": "adamw_torch_fused",
        "optimizer_created_after_adapter_load": True,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "trainer_state": "fresh",
        "rng_state": "fresh_from_v14_seed",
    }

    with pytest.raises(ValueError, match=message):
        warm.validate_v14_fresh_start_contract(
            replace(base, **{field: value})
        )


def test_v14_launch_guard_rejects_resume_gate_and_wrong_horizon_first() -> None:
    with pytest.raises(launch.LaunchContractError, match="target_step_must_be_four"):
        launch.validate_launch_contract(target_step=3)
    with pytest.raises(launch.LaunchContractError, match="resume_is_forbidden"):
        launch.validate_launch_contract(
            target_step=4,
            resume_checkpoint=Path("checkpoint-1"),
        )
    with pytest.raises(
        launch.LaunchContractError,
        match="gate_receipt_cannot_authorize_training",
    ):
        launch.validate_launch_contract(
            target_step=4,
            gate_receipt=Path("gate.json"),
        )
    with pytest.raises(
        launch.LaunchContractError,
        match="smoke_target_step_must_be_one",
    ):
        launch.validate_launch_contract(target_step=4, smoke=True)
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
def test_v14_training_guard_rejects_protected_splits_before_resolve(
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
        launch.guard_v14_training_data_path(path, description="test_data")


def test_v14_launcher_dry_run_and_critical_file_bindings() -> None:
    script_path = Path(launch.__file__).with_name("train_scene_memory_v14.sh")
    script = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    for required in (
        'TARGET_STEP=4',
        '[[ -z "${RESUME_FROM_CHECKPOINT}" ]] || fail "v14_resume_is_forbidden"',
        "--scene-state-generation-objective-version "
        "scene_state_generation_ce_symmetric_cached_prefix_boundary_v14",
        "--gradient-accumulation-steps \"${GRADIENT_ACCUMULATION_STEPS}\"",
        "--learning-rate 1e-4",
        "--lr-scheduler-type constant",
        "--warmup-steps 0",
        "--max-steps \"${MAX_STEPS}\"",
        "--save-total-limit \"${SAVE_TOTAL_LIMIT}\"",
        '--scene-state-v14-one-pair-smoke',
        'CHECKPOINT1_DIR="${OUTPUT_DIR}/trainer/checkpoint-1"',
        'CHECKPOINT4_DIR="${OUTPUT_DIR}/trainer/checkpoint-4"',
        "status --porcelain --untracked-files=no",
        "ls-files --error-unmatch",
        "critical_v14_source_must_be_tracked",
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
    shell_files = tuple(
        re.findall(r'^\s+"([^"]+)"\s*$', array_match.group("body"), re.MULTILINE)
    )
    assert len(shell_files) == len(set(shell_files))
    assert set(shell_files) == set(launch.CRITICAL_TRAINING_FILES)

    bindings = launch.critical_training_code_bindings()
    assert tuple(bindings) == tuple(launch.CRITICAL_TRAINING_FILES)
    assert all(binding["sha256"] for binding in bindings.values())
    evaluator_binding = gate.evaluator_code_binding()
    assert evaluator_binding["scene_boundary_metric"]["path"] == str(
        (launch.PROJECT_ROOT / "deltamem/scene_boundary.py").resolve()
    )
    assert evaluator_binding["v14_launch_contract"]["sha256"] == (
        bindings[
            "experiments/rethinking_rwkv_ms_gemma/scene_memory_v14_launch_contract.py"
        ]["sha256"]
    )
