from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_dualpath_repair as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_dualpath_repair as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_dualpath_repair as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_dualpath_repair_eval as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_contrast_dropout as contrast,
)


def make_row() -> contrast.SceneContrastRow:
    return contrast.SceneContrastRow(
        example=evolution.NativeFullRowExample(
            row_id="row-7",
            task="scene",
            source_ordinal=7,
            row_sha256="7" * 64,
            write_input_ids=(1, 2),
            write_attention_mask=(1, 1),
            read_input_ids=(3,),
            read_attention_mask=(1,),
            labels=(3,),
            assistant_target_tokens=1,
        ),
        assistant_identity="assistant-7",
    )


def test_dualpath_repair_protocol_is_adaptive_and_locked() -> None:
    protocol = runner.validate_protocol()

    assert runner.PROTOCOL_PAYLOAD_SHA256 == (
        "74373a496b63e0b4a6c4ca134ae40b25118a4b311aa9734d4001eecc210781b7"
    )
    assert protocol["study_scope"]["independent_evidence"] is False
    assert protocol["study_scope"]["protected_split_authorization_from_this_study"] is False
    assert protocol["training_data"]["selected_rows"] == 96
    assert protocol["intervention"]["corrective_path_supervised_tokens"] == 1
    assert protocol["intervention"]["fallback_continuation_tokens_maximum"] == 3
    assert protocol["intervention"]["fallback_continuation_ce_weight"] == 0.25
    assert protocol["intervention"]["forward_backward_paths_per_actionable_row"] == 2
    assert protocol["intervention"]["full_sequence_gold_ce_weight"] == 0.0
    assert protocol["intervention"]["learning_rate"] == 5e-6
    assert protocol["intervention"]["max_gradient_norm"] == 0.05
    assert protocol["intervention"]["post_step_delta_retention"] == 0.99
    assert protocol["protected_splits_opened_by_this_study"] == []


def test_dualpath_losses_separate_correction_and_wrong_branch_continuation() -> None:
    corrective_logits = torch.zeros(8, requires_grad=True)
    fallback_logits = torch.zeros((3, 8), requires_grad=True)
    fallback_labels = torch.tensor([4, 5, 6])

    pairwise = runner.pairwise_margin_loss(
        corrective_logits,
        correct_token_id=3,
        wrong_token_id=2,
    )
    fallback = runner.fallback_continuation_loss(
        fallback_logits,
        fallback_labels,
    )
    objective = pairwise + runner.FALLBACK_CONTINUATION_CE_WEIGHT * fallback
    objective.backward()

    assert torch.isfinite(objective)
    assert torch.isfinite(pairwise)
    assert torch.isfinite(fallback)
    assert corrective_logits.grad is not None
    assert corrective_logits.grad[3] < 0
    assert corrective_logits.grad[2] > 0
    assert corrective_logits.grad[0] == 0
    assert fallback_logits.grad is not None
    assert fallback_logits.grad[0, 4] < 0
    assert fallback_logits.grad[1, 5] < 0
    assert fallback_logits.grad[2, 6] < 0


def test_dualpath_fallback_conditions_on_wrong_token() -> None:
    plan = runner.RepairPlan(
        status="actionable",
        prediction=(1,),
        gold_boundaries=(),
        false_positive_boundaries=(1,),
        divergence_index=1,
        correct_token_id=99,
        wrong_token_id=21,
    )

    corrective, fallback, fallback_ids = runner.build_dualpath_examples(
        make_row(),
        prompt_ids=(10, 11),
        generated_ids=(20, 21, 22, 23, 24),
        plan=plan,
    )

    assert corrective is not None
    assert corrective.read_input_ids == (10, 11, 20, 99)
    assert corrective.labels == (-100, -100, -100, 99)
    assert fallback is not None
    assert fallback_ids == (22, 23, 24)
    assert fallback.read_input_ids == (10, 11, 20, 21, 22, 23, 24)
    assert fallback.labels == (-100, -100, -100, -100, 22, 23, 24)


def test_dualpath_repair_configures_shared_engine(monkeypatch) -> None:
    for name in (
        "SCHEMA",
        "STEP_SCHEMA",
        "PATCH_SCHEMA",
        "PROTOCOL",
        "PROTOCOL_PAYLOAD_SHA256",
        "LEARNING_RATE",
        "MAX_GRAD_NORM",
        "POST_STEP_DELTA_RETENTION",
        "validate_protocol",
        "mine_repair",
        "backward_repair",
        "train",
    ):
        monkeypatch.setattr(runner.shared, name, getattr(runner.shared, name))

    runner.configure_engine()

    assert runner.shared.PROTOCOL_PAYLOAD_SHA256 == runner.PROTOCOL_PAYLOAD_SHA256
    assert runner.shared.LEARNING_RATE == 5e-6
    assert runner.shared.MAX_GRAD_NORM == 0.05
    assert runner.shared.POST_STEP_DELTA_RETENTION == 0.99
    assert runner.shared.mine_repair is runner.mine_repair
    assert runner.shared.backward_repair is runner.backward_repair
    assert runner.shared.train is runner.train


def test_dualpath_repair_endpoint_bindings(monkeypatch) -> None:
    protocol = runner.validate_protocol()
    monkeypatch.setattr(
        evaluator,
        "_SHARED_INPUT_BINDING",
        lambda **kwargs: {"runner_sha256": "shared"},
    )

    assert materializer.CANDIDATE_ID == "dualpath_repair_endpoint"
    assert materializer.TRAINING_BINDING.STARTING_GATE_STATE_SHA256 == (
        runner.shared.STARTING_GATE_STATE_SHA256
    )
    assert evaluator.SCHEMA.endswith("dualpath_repair_eval_shard.v1")
    assert evaluator.TRAINING_BINDING.gate is runner.shared.gate
    assert evaluator.input_binding()["runner_sha256"] == evaluator.sha256_file(
        evaluator.Path(evaluator.__file__)
    )
    assert evaluator.ROW_PAYLOAD_SHA256 == protocol["evaluation"]["row_payload_sha256"]
    assert protocol["evaluation"]["development_gates"][
        "candidate_minus_checkpoint_16_micro_f1_minimum"
    ] == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
