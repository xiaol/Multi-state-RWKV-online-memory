from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_precision_unlikelihood as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_precision_unlikelihood as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_precision_unlikelihood as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_precision_unlikelihood_eval as evaluator,
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


def make_negative(index: int) -> runner.PrecisionNegative:
    return runner.PrecisionNegative(
        example=make_row(index).example,
        gold_boundaries=(),
        wrong_boundary=1,
        wrong_target_positions=(0,),
        wrong_token_ids=(1,),
        negative_content='{"boundaries": [1]}',
        payload_sha256=f"{index + 2000:064x}",
    )


def test_precision_protocol_is_fully_locked() -> None:
    protocol = runner.validate_protocol()

    assert runner.PROTOCOL_PAYLOAD_SHA256 == (
        "ad397d198a841b9734745ff1ff8faa6f559428a41e216fbcf540a36cf98da51e"
    )
    assert protocol["training_data"]["remaining_untouched_rows"] == 359
    assert protocol["training_data"]["selected_rows"] == 256
    assert protocol["training_data"]["conditions"] == {
        "correct_state": 256,
        "no_state": 0,
        "wrong_state": 0,
    }
    assert protocol["negative_construction"]["wrong_boundary_digit_tokens"] == 270
    assert protocol["intervention"]["semantic_unlikelihood_weight"] == 0.5
    assert protocol["intervention"]["learning_rate"] == 1.5e-5
    assert protocol["intervention"]["updates"] == 16
    assert protocol["frozen_constraints"]["world_size"] == 4
    assert protocol["protected_splits_opened_by_this_study"] == []


def test_wrong_boundary_prefers_one_then_hash_ranked_absent() -> None:
    assert runner.choose_wrong_boundary(
        gold_boundaries=(3,),
        maximum_boundary=5,
        row_sha256="0" * 64,
    ) == 1

    selected = runner.choose_wrong_boundary(
        gold_boundaries=(1, 3),
        maximum_boundary=5,
        row_sha256="0" * 64,
    )
    assert selected in {2, 4, 5}
    assert selected == runner.choose_wrong_boundary(
        gold_boundaries=(1, 3),
        maximum_boundary=5,
        row_sha256="0" * 64,
    )


def test_schedule_uses_untouched_rows_and_correct_state_only(monkeypatch) -> None:
    rows = [make_row(index) for index in range(runner.EXPECTED_ELIGIBLE_ROWS)]
    mapping = {index: (index + 1) % len(rows) for index in range(len(rows))}
    deltas = {index: 0 for index in range(len(rows))}
    monkeypatch.setattr(
        runner,
        "_ACTIVE_NEGATIVES",
        {index: make_negative(index) for index in range(len(rows))},
    )
    monkeypatch.setattr(
        runner,
        "prior_excluded_rows",
        lambda *args, **kwargs: set(range(runner.EXPECTED_EXCLUDED_ROWS)),
    )
    monkeypatch.setattr(runner, "canonical_sha256", lambda value: "locked")
    monkeypatch.setattr(runner, "SELECTED_ROWS_PAYLOAD_SHA256", "locked")
    monkeypatch.setattr(runner, "NEGATIVE_PAYLOAD_SHA256", "locked")
    monkeypatch.setattr(runner, "SCHEDULE_PAYLOAD_SHA256", "locked")

    schedules, payloads = runner.build_schedules(rows, mapping, deltas)
    schedule = schedules[runner.SEED]
    selected = {
        source_ordinal
        for step in schedule
        for source_ordinal in step.source_ordinals
    }

    assert len(schedule) == 16
    assert len(selected) == 256
    assert min(selected) >= runner.EXPECTED_EXCLUDED_ROWS
    assert all(len(step.source_ordinals) == 16 for step in schedule)
    assert all(step.no_state_ordinals == frozenset() for step in schedule)
    assert all(
        row["condition"] == "correct_state"
        for step in payloads[runner.SEED]
        for row in step["rows"]
    )


def test_unlikelihood_gradient_suppresses_only_selected_wrong_tokens() -> None:
    logits = torch.tensor(
        [[2.0, 0.5, -1.0, 1.0], [0.0, 3.0, 1.0, -2.0]],
        requires_grad=True,
    )
    wrong = torch.tensor([0, 1])

    loss = runner.unlikelihood_from_logits(logits, wrong)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[0, 0] > 0
    assert logits.grad[1, 1] > 0
    assert logits.grad[0, 1:].max() < 0
    assert torch.cat((logits.grad[1, :1], logits.grad[1, 2:])).max() < 0


def test_training_engine_uses_locked_precision_hyperparameters(monkeypatch) -> None:
    monkeypatch.setattr(runner.robust, "PROTOCOL_PAYLOAD_SHA256", "old")
    monkeypatch.setattr(runner.robust, "PATCH_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "STEP_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "LEARNING_RATE", 1.0)
    monkeypatch.setattr(runner.robust, "POST_STEP_DELTA_RETENTION", 0.0)
    monkeypatch.setattr(runner.robust, "train", SimpleNamespace())

    runner.configure_training_engine()

    assert runner.robust.PROTOCOL_PAYLOAD_SHA256 == runner.PROTOCOL_PAYLOAD_SHA256
    assert runner.robust.PATCH_SCHEMA == runner.PATCH_SCHEMA
    assert runner.robust.STEP_SCHEMA == runner.STEP_SCHEMA
    assert runner.robust.LEARNING_RATE == 1.5e-5
    assert runner.robust.POST_STEP_DELTA_RETENTION == 0.995
    assert runner.robust.train is runner.train


def test_materialization_evaluation_and_analysis_bind_endpoint(monkeypatch) -> None:
    protocol = runner.validate_protocol()
    monkeypatch.setattr(
        evaluator,
        "_SHARED_INPUT_BINDING",
        lambda **kwargs: {"runner_sha256": "shared"},
    )

    assert materializer.CANDIDATE_ID == "precision_unlikelihood_endpoint"
    assert evaluator.SCHEMA.endswith("precision_unlikelihood_eval_shard.v1")
    assert evaluator.input_binding()["runner_sha256"] == evaluator.sha256_file(
        evaluator.Path(evaluator.__file__)
    )
    assert evaluator.ROW_PAYLOAD_SHA256 == protocol["evaluation"][
        "row_payload_sha256"
    ]
    assert protocol["evaluation"]["gates"][
        "candidate_minus_checkpoint_16_micro_f1_minimum"
    ] == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
