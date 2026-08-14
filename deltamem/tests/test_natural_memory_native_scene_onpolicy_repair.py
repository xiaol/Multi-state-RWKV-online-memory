from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_onpolicy_repair as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_onpolicy_repair as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_onpolicy_repair as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_onpolicy_repair_eval as evaluator,
)


def make_row(index: int) -> contrast.SceneContrastRow:
    example = evolution.NativeFullRowExample(
        row_id=f"row-{index}",
        task="scene",
        source_ordinal=index,
        row_sha256=f"{index:064x}",
        write_input_ids=(1, 2),
        write_attention_mask=(1, 1),
        read_input_ids=(1, 2, 3),
        read_attention_mask=(1, 1, 1),
        labels=(-100, -100, 3),
        assistant_target_tokens=1,
    )
    return contrast.SceneContrastRow(example=example, assistant_identity=str(index))


def test_onpolicy_repair_protocol_is_fully_locked() -> None:
    protocol = runner.validate_protocol()

    assert runner.PROTOCOL_PAYLOAD_SHA256 == (
        "a01497344e733e53caf3f49f4db12e2190076bda34807a4a137c5dc3b001f4d6"
    )
    assert protocol["training_data"]["remaining_untouched_rows"] == 103
    assert protocol["training_data"]["selected_rows"] == 96
    assert protocol["training_data"]["unused_eligible_rows_after_selection"] == 7
    assert protocol["greedy_mining"]["mining_updates_model"] is False
    assert protocol["intervention"]["full_sequence_gold_ce_weight"] == 0.0
    assert protocol["intervention"]["pairwise_margin"] == 0.5
    assert protocol["intervention"]["learning_rate"] == 5e-6
    assert protocol["intervention"]["updates"] == 6
    assert protocol["frozen_constraints"]["world_size"] == 4
    assert protocol["protected_splits_opened_by_this_study"] == []


def test_schedule_excludes_precision_rows_and_uses_last_untouched_rows(
    monkeypatch,
) -> None:
    rows = [make_row(index) for index in range(runner.EXPECTED_ELIGIBLE_ROWS)]
    mapping = {index: (index + 1) % len(rows) for index in range(len(rows))}
    deltas = {index: 0 for index in range(len(rows))}
    monkeypatch.setattr(
        runner,
        "_ACTIVE_REPAIR_SOURCES",
        {
            index: runner.RepairSource(messages=(), gold_boundaries=())
            for index in range(len(rows))
        },
    )
    monkeypatch.setattr(
        runner.precision,
        "prior_excluded_rows",
        lambda *args, **kwargs: set(range(runner.EXPECTED_PRIOR_EXCLUDED_ROWS)),
    )
    monkeypatch.setattr(runner, "canonical_sha256", lambda value: "locked")
    monkeypatch.setattr(runner, "SELECTED_ROWS_PAYLOAD_SHA256", "locked")
    monkeypatch.setattr(runner, "SCHEDULE_PAYLOAD_SHA256", "locked")

    eligible = set(range(len(rows)))
    prior = set(range(runner.EXPECTED_PRIOR_EXCLUDED_ROWS))
    precision_rows = runner._precision_selected_rows(rows, eligible, prior)
    schedules, payloads = runner.build_schedules(rows, mapping, deltas)
    selected = {
        source_ordinal
        for step in schedules[runner.SEED]
        for source_ordinal in step.source_ordinals
    }

    assert len(precision_rows) == 256
    assert len(selected) == 96
    assert selected.isdisjoint(prior)
    assert selected.isdisjoint(precision_rows)
    assert len(schedules[runner.SEED]) == 6
    assert all(len(step.source_ordinals) == 16 for step in schedules[runner.SEED])
    assert all(
        row["condition"] == "correct_state"
        and row["repair"] == "current_greedy_first_divergence"
        for step in payloads[runner.SEED]
        for row in step["rows"]
    )


def test_repair_plan_targets_only_first_divergence_with_false_positive() -> None:
    plan = runner.repair_plan(
        gold_token_ids=(10, 20, 30, 40),
        generated_token_ids=(10, 21, 30, 40),
        gold_boundaries=(3,),
        prediction=(1, 3),
    )

    assert plan.status == "actionable"
    assert plan.false_positive_boundaries == (1,)
    assert plan.divergence_index == 1
    assert plan.correct_token_id == 20
    assert plan.wrong_token_id == 21

    no_fp = runner.repair_plan(
        gold_token_ids=(10, 20),
        generated_token_ids=(10, 21),
        gold_boundaries=(3,),
        prediction=(3,),
    )
    assert no_fp.status == "no_false_positive"
    assert no_fp.divergence_index is None


def test_pairwise_margin_gradient_only_swaps_correct_and_wrong_tokens() -> None:
    logits = torch.tensor([0.0, 2.0, -1.0, 1.0], requires_grad=True)

    loss = runner.pairwise_margin_loss(
        logits,
        correct_token_id=3,
        wrong_token_id=1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[3] < 0
    assert logits.grad[1] > 0
    assert logits.grad[0] == 0
    assert logits.grad[2] == 0


def test_training_engine_uses_locked_repair_hyperparameters(monkeypatch) -> None:
    monkeypatch.setattr(runner.robust, "PROTOCOL_PAYLOAD_SHA256", "old")
    monkeypatch.setattr(runner.robust, "PATCH_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "STEP_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "LEARNING_RATE", 1.0)
    monkeypatch.setattr(runner.robust, "MAX_GRAD_NORM", 1.0)
    monkeypatch.setattr(runner.robust, "POST_STEP_DELTA_RETENTION", 0.0)
    monkeypatch.setattr(runner.robust, "train", SimpleNamespace())

    runner.configure_training_engine()

    assert runner.robust.PROTOCOL_PAYLOAD_SHA256 == runner.PROTOCOL_PAYLOAD_SHA256
    assert runner.robust.PATCH_SCHEMA == runner.PATCH_SCHEMA
    assert runner.robust.STEP_SCHEMA == runner.STEP_SCHEMA
    assert runner.robust.LEARNING_RATE == 5e-6
    assert runner.robust.MAX_GRAD_NORM == 0.05
    assert runner.robust.POST_STEP_DELTA_RETENTION == 0.99
    assert runner.robust.train is runner.train


def test_materialization_evaluation_and_analysis_bind_endpoint(monkeypatch) -> None:
    protocol = runner.validate_protocol()
    monkeypatch.setattr(
        evaluator,
        "_SHARED_INPUT_BINDING",
        lambda **kwargs: {"runner_sha256": "shared"},
    )

    assert materializer.CANDIDATE_ID == "onpolicy_repair_endpoint"
    assert evaluator.SCHEMA.endswith("onpolicy_repair_eval_shard.v1")
    assert evaluator.input_binding()["runner_sha256"] == evaluator.sha256_file(
        evaluator.Path(evaluator.__file__)
    )
    assert evaluator.ROW_PAYLOAD_SHA256 == protocol["evaluation"][
        "row_payload_sha256"
    ]
    assert protocol["evaluation"]["gates"][
        "candidate_minus_checkpoint_16_micro_f1_minimum"
    ] == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
