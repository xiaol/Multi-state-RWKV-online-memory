from __future__ import annotations

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    analyze_natural_memory_native_scene_suffix_repair as analyzer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    materialize_natural_memory_native_scene_suffix_repair as materializer,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_suffix_repair as runner,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_scene_suffix_repair_eval as evaluator,
)


def test_suffix_repair_protocol_is_adaptive_and_locked() -> None:
    protocol = runner.validate_protocol()

    assert runner.PROTOCOL_PAYLOAD_SHA256 == (
        "2a65f6105b1c6d31acc6b683b894914b743e7c645dc4ba98d667d32b2880af0c"
    )
    assert protocol["study_scope"]["independent_evidence"] is False
    assert protocol["study_scope"]["protected_split_authorization_from_this_study"] is False
    assert protocol["training_data"]["selected_rows"] == 96
    assert protocol["intervention"]["supervised_gold_tokens_from_divergence_maximum"] == 4
    assert protocol["intervention"]["local_gold_suffix_ce_weight"] == 0.25
    assert protocol["intervention"]["full_sequence_gold_ce_weight"] == 0.0
    assert protocol["intervention"]["learning_rate"] == 2.5e-6
    assert protocol["intervention"]["max_gradient_norm"] == 0.025
    assert protocol["intervention"]["post_step_delta_retention"] == 0.995
    assert protocol["protected_splits_opened_by_this_study"] == []


def test_suffix_repair_loss_corrects_divergence_and_closure() -> None:
    logits = torch.zeros((3, 8), requires_grad=True)
    labels = torch.tensor([3, 4, 5])

    objective, pairwise, closure = runner.suffix_repair_loss(
        logits,
        labels,
        wrong_token_id=2,
    )
    objective.backward()

    assert torch.isfinite(objective)
    assert torch.isfinite(pairwise)
    assert torch.isfinite(closure)
    assert logits.grad is not None
    assert logits.grad[0, 3] < 0
    assert logits.grad[0, 2] > 0
    assert logits.grad[1, 4] < 0
    assert logits.grad[2, 5] < 0
    assert logits.grad[0, 0] == 0


def test_suffix_repair_configures_shared_engine(monkeypatch) -> None:
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
    assert runner.shared.LEARNING_RATE == 2.5e-6
    assert runner.shared.MAX_GRAD_NORM == 0.025
    assert runner.shared.POST_STEP_DELTA_RETENTION == 0.995
    assert runner.shared.mine_repair is runner.mine_repair
    assert runner.shared.backward_repair is runner.backward_repair
    assert runner.shared.train is runner.train


def test_suffix_repair_endpoint_bindings(monkeypatch) -> None:
    protocol = runner.validate_protocol()
    monkeypatch.setattr(
        evaluator,
        "_SHARED_INPUT_BINDING",
        lambda **kwargs: {"runner_sha256": "shared"},
    )

    assert materializer.CANDIDATE_ID == "suffix_repair_endpoint"
    assert materializer.TRAINING_BINDING.STARTING_GATE_STATE_SHA256 == (
        runner.shared.STARTING_GATE_STATE_SHA256
    )
    assert evaluator.SCHEMA.endswith("suffix_repair_eval_shard.v1")
    assert evaluator.input_binding()["runner_sha256"] == evaluator.sha256_file(
        evaluator.Path(evaluator.__file__)
    )
    assert evaluator.ROW_PAYLOAD_SHA256 == protocol["evaluation"]["row_payload_sha256"]
    assert protocol["evaluation"]["development_gates"][
        "candidate_minus_checkpoint_16_micro_f1_minimum"
    ] == analyzer.GATE_THRESHOLDS["candidate_minus_checkpoint_16_micro_f1"]
