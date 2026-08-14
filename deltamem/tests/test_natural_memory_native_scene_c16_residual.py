from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_c16_residual as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_c16_residual as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_c16_residual as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_c16_residual_eval as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
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


def test_c16_residual_protocol_and_bindings_are_locked() -> None:
    protocol = runner.validate_protocol()

    assert runner.PROTOCOL_PAYLOAD_SHA256 == (
        "e0b797bda233266b8806e67202b4c30067d95d7a604f6a33607b6340383d55e3"
    )
    assert protocol["frozen_constraints"]["starting_checkpoint"].endswith(
        "checkpoint-16"
    )
    assert [item["seed"] for item in protocol["training_data"]["seeds"]] == [
        71,
        89,
        107,
    ]
    assert protocol["training_data"]["excluded_unique_rows"] == 692
    assert protocol["training_data"]["remaining_unseen_rows"] == 359
    assert protocol["intervention"]["learning_rate"] == 2.5e-5
    assert protocol["intervention"]["global_batch_size"] == 16
    assert protocol["intervention"]["updates_per_seed"] == 8
    for item in protocol["training_data"]["seeds"]:
        assert runner.SEED_BINDINGS[item["seed"]] == {
            "selected_rows_payload_sha256": item[
                "selected_rows_payload_sha256"
            ],
            "schedule_payload_sha256": item["schedule_payload_sha256"],
        }


def test_c16_residual_schedules_use_fresh_disjoint_rows(monkeypatch) -> None:
    rows = [make_row(index) for index in range(1435)]
    mapping = {index: (index + 1) % len(rows) for index in range(len(rows))}
    deltas = {index: 0 for index in range(len(rows))}
    original_steps = [
        SimpleNamespace(source_ordinals=tuple(range(offset, offset + 8)))
        for offset in range(0, 128, 8)
    ]
    prior_indices = {
        17: tuple(range(128, 384)),
        29: tuple(range(384, 640)),
        43: tuple(range(640, 692)) + tuple(range(128, 332)),
    }

    def prior_schedule(*args, seed: int, **kwargs):
        indices = prior_indices[seed]
        steps = [
            SimpleNamespace(source_ordinals=indices[offset : offset + 16])
            for offset in range(0, len(indices), 16)
        ]
        return tuple(steps), []

    monkeypatch.setattr(
        runner.contrast,
        "build_schedule",
        lambda *args, **kwargs: (tuple(original_steps), []),
    )
    monkeypatch.setattr(runner.robust, "build_schedule", prior_schedule)
    monkeypatch.setattr(runner, "canonical_sha256", lambda value: "locked")
    monkeypatch.setattr(
        runner,
        "SEED_BINDINGS",
        {
            seed: {
                "selected_rows_payload_sha256": "locked",
                "schedule_payload_sha256": "locked",
            }
            for seed in runner.SEEDS
        },
    )

    schedules, _ = runner.build_schedules(rows, mapping, deltas)
    selected = {
        seed: {
            source_ordinal
            for step in schedules[seed]
            for source_ordinal in step.source_ordinals
        }
        for seed in runner.SEEDS
    }

    assert all(len(indices) == 128 for indices in selected.values())
    assert all(min(indices) >= 692 for indices in selected.values())
    assert selected[71].isdisjoint(selected[89])
    assert selected[71].isdisjoint(selected[107])
    assert selected[89].isdisjoint(selected[107])
    assert all(
        len(step.source_ordinals) == 16 and len(step.no_state_ordinals) == 4
        for schedule in schedules.values()
        for step in schedule
    )


def test_training_engine_uses_locked_hyperparameters(monkeypatch) -> None:
    monkeypatch.setattr(runner.robust, "PROTOCOL_PAYLOAD_SHA256", "old")
    monkeypatch.setattr(runner.robust, "PATCH_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "STEP_SCHEMA", "old")
    monkeypatch.setattr(runner.robust, "LEARNING_RATE", 1.0)
    monkeypatch.setattr(runner.robust, "POST_STEP_DELTA_RETENTION", 0.0)

    runner.configure_training_engine()

    assert runner.robust.PROTOCOL_PAYLOAD_SHA256 == runner.PROTOCOL_PAYLOAD_SHA256
    assert runner.robust.PATCH_SCHEMA == runner.PATCH_SCHEMA
    assert runner.robust.STEP_SCHEMA == runner.STEP_SCHEMA
    assert runner.robust.LEARNING_RATE == 2.5e-5
    assert runner.robust.POST_STEP_DELTA_RETENTION == 0.995


def test_materialization_averages_checkpoint_relative_residuals() -> None:
    anchor = {"gate": torch.tensor([10.0, 20.0])}
    states = {
        71: {"gate": torch.tensor([11.0, 18.0])},
        89: {"gate": torch.tensor([13.0, 20.0])},
        107: {"gate": torch.tensor([9.0, 25.0])},
    }

    mixed = materializer.mean_residual(anchor, states)

    assert torch.allclose(mixed["gate"], torch.tensor([11.0, 21.0]))


def test_evaluation_and_analysis_bind_the_new_candidate(monkeypatch) -> None:
    protocol = runner.validate_protocol()
    monkeypatch.setattr(
        evaluator,
        "_SHARED_INPUT_BINDING",
        lambda **kwargs: {"runner_sha256": "shared"},
    )

    assert evaluator.SCHEMA.endswith("c16_residual_eval_shard.v1")
    assert evaluator.input_binding()["runner_sha256"] == evaluator.sha256_file(
        evaluator.Path(evaluator.__file__)
    )
    assert evaluator.ROW_PAYLOAD_SHA256 == protocol["evaluation"][
        "row_payload_sha256"
    ]
    assert materializer.CANDIDATE_ID == "c16_residual_mean"
    assert protocol["evaluation"]["gates"][
        "candidate_minus_checkpoint_16_micro_f1_minimum"
    ] == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
