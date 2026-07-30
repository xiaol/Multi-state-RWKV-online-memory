from __future__ import annotations

from argparse import Namespace
import math
from types import SimpleNamespace

import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v10_launch_contract
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v9_warm_start
from experiments.rethinking_rwkv_ms_gemma import scene_memory_v10_warm_start
from deltamem.tests.test_scene_memory_v9_training import (
    _PairModel,
    _bind_pair_model,
    _masks,
    _trainer,
)


def _pair_inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    source = {
        "input_ids": torch.tensor([[7, 8, 2, 0, 0, 0]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 0, 0, 0]]),
    }
    donor = {
        "input_ids": torch.tensor([[7, 8, 2, 1, 1, 1]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 1, 1, 1]]),
    }
    return source, donor


def test_v9_v10_warm_start_branches_preserve_target_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config = SimpleNamespace(to_dict=lambda: {"rank": 4})
    monkeypatch.setattr(
        experimental_train,
        "_validate_adapter_warm_start_args",
        lambda args: None,
    )
    monkeypatch.setattr(
        experimental_train.HFDeltaMemConfig,
        "from_pretrained",
        staticmethod(lambda checkpoint: source_config),
    )

    specifications = (
        (
            scene_memory_v9_warm_start,
            "prepare_v9_v8_checkpoint56_warm_start",
            "apply_v9_v8_checkpoint56_adapter_only_warm_start",
            "scene_v9_context",
            "scene_v9_fresh_start",
        ),
        (
            scene_memory_v10_warm_start,
            "prepare_v10_v8_checkpoint56_warm_start",
            "apply_v10_v8_checkpoint56_adapter_only_warm_start",
            "scene_v10_context",
            "scene_v10_fresh_start",
        ),
    )
    for module, prepare_name, apply_name, context_field, fresh_start_field in specifications:
        checkpoint = (tmp_path / module.WARM_START_MODE / "checkpoint-56").resolve()
        pinned_context_type = (
            scene_memory_v9_warm_start.V9WarmStartContext
            if module is scene_memory_v9_warm_start
            else scene_memory_v10_warm_start.V10WarmStartContext
        )
        fresh_start_type = (
            scene_memory_v9_warm_start.V9FreshStartContract
            if module is scene_memory_v9_warm_start
            else scene_memory_v10_warm_start.V10FreshStartContract
        )
        pinned_context = pinned_context_type(
            checkpoint=checkpoint,
            lock_path=tmp_path / "source-lock.json",
            lock={},
            source_config={},
            source_trainer_state={},
            source_training_protocol={"schema_version": 11},
            source_pairing_manifest={},
            continuation_lineage=(),
        )
        monkeypatch.setattr(
            experimental_train,
            prepare_name,
            lambda *args, **kwargs: pinned_context,
        )
        received = {}

        def apply_warm_start(model, context, *, fresh_start):
            received.update(
                model=model,
                context=context,
                fresh_start=fresh_start,
            )
            return {
                "schema": module.RECEIPT_SCHEMA,
                "schema_version": 1,
                "mode": module.WARM_START_MODE,
            }

        monkeypatch.setattr(experimental_train, apply_name, apply_warm_start)
        args = Namespace(
            output_dir=str(tmp_path / f"output-{module.WARM_START_MODE}"),
            warm_start_mode=module.WARM_START_MODE,
            optim="adamw_torch",
        )

        prepared = experimental_train.prepare_adapter_warm_start(args, str(checkpoint))

        assert prepared is not None
        assert prepared.mode == module.WARM_START_MODE
        assert prepared.manifest["mode"] == module.WARM_START_MODE
        assert getattr(prepared, context_field) is pinned_context
        assert isinstance(getattr(prepared, fresh_start_field), fresh_start_type)
        receipt = experimental_train.apply_adapter_warm_start(
            object(),
            prepared,
            source_config,
            [],
        )
        assert received["context"] is pinned_context
        assert isinstance(received["fresh_start"], fresh_start_type)
        assert receipt["schema"] == module.RECEIPT_SCHEMA
        assert receipt["mode"] == module.WARM_START_MODE


def _run_reciprocal_objective(
    objective_version: str,
    *,
    gradient_scale: float,
    prefix_probe: bool = False,
) -> tuple[
    experimental_train.DeltaMemTrainer,
    _PairModel,
    torch.Tensor,
    dict[str, float],
]:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = objective_version
    trainer.current_gradient_accumulation_steps = (
        experimental_train._SCENE_STATE_CYCLE_RETENTION_GRADIENT_ACCUMULATION_STEPS
        if objective_version
        == experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
        else 1
    )
    model = _PairModel()
    if prefix_probe:
        trainer.scene_state_generated_prefix_correction_weight = 0.5

        def prefix_correction(active_model, *args, **kwargs):
            del args, kwargs
            return active_model.parameters_by_side.sum(), {
                "scene_generation_prefix_positive_ce": 0.3,
                "scene_generation_prefix_negative_unlikelihood": 0.2,
                "scene_generation_prefix_correction_applied": 1.0,
                "scene_generation_prefix_correction_event_count": 1.0,
                "scene_generation_prefix_positive_event_count": 1.0,
                "scene_generation_prefix_negative_event_count": 1.0,
                "scene_generation_prefix_substitution_count": 1.0,
                "scene_generation_prefix_insertion_count": 0.0,
                "scene_generation_prefix_deletion_count": 0.0,
                "scene_generation_generated_rollout_token_count": 1.0,
                "scene_generation_generated_first_divergence": 0.0,
                "scene_generation_generated_exact_fraction": 0.0,
            }

        trainer._scene_state_generated_prefix_correction_branch = prefix_correction
    monkeypatch = pytest.MonkeyPatch()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs, donor_inputs = _pair_inputs()
    try:
        loss, stats = trainer._scene_state_generation_symmetric_sequential_backward(
            model,
            source_inputs,
            donor_inputs,
            loss_kwargs={},
            source_write_input_ids=torch.tensor([[10]]),
            source_write_attention_mask=torch.ones(1, 1, dtype=torch.long),
            source_write_message_ids=None,
            source_write_sentence_ids=None,
            donor_write_input_ids=torch.tensor([[20]]),
            donor_write_attention_mask=torch.ones(1, 1, dtype=torch.long),
            donor_write_message_ids=None,
            donor_write_sentence_ids=None,
            source_target_mask=masks["target"],
            source_content_mask=masks["content"],
            source_schema_mask=masks["schema"],
            source_decision_mask=masks["decision"],
            source_termination_mask=masks["termination"],
            source_pair_target_mask=masks["pair"],
            donor_target_mask=masks["target"],
            donor_content_mask=masks["content"],
            donor_schema_mask=masks["schema"],
            donor_decision_mask=masks["decision"],
            donor_termination_mask=masks["termination"],
            donor_pair_target_mask=masks["pair"],
            gradient_scale=gradient_scale,
        )
    finally:
        monkeypatch.undo()
    return trainer, model, loss, stats


def test_exact_top1_trajectory_retains_margin_gradient() -> None:
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = torch.tensor([1.1, 1.0, 0.0])
        logits[0, 1] = torch.tensor([1.0, 1.1, 0.0])
    labels = torch.tensor([[-100, 0, 1]])
    target_mask = torch.tensor([[False, True, True]])

    metrics = experimental_train.DeltaMemTrainer._scene_state_all_target_top1_retention_metrics(
        logits,
        labels,
        target_mask,
    )

    assert metrics["top1_fraction_row"].item() == 1.0
    assert metrics["retention_hinge_row"].item() == pytest.approx(0.1)
    metrics["retention_hinge_row"].mean().backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0
    assert logits.grad[0, 0, 0].item() < 0.0
    assert logits.grad[0, 0, 1].item() == 0.0
    assert logits.grad[0, 1, 1].item() < 0.0
    assert logits.grad[0, 1, 0].item() == 0.0

    high_margin_logits = logits.detach().clone()
    high_margin_logits[0, 0] = torch.tensor([1.3, 1.0, 0.0])
    high_margin_logits[0, 1] = torch.tensor([1.0, 1.3, 0.0])
    high_margin = experimental_train.DeltaMemTrainer._scene_state_all_target_top1_retention_metrics(
        high_margin_logits,
        labels,
        target_mask,
    )
    assert high_margin["retention_hinge_row"].item() == 0.0


def test_v10_total_includes_first_error_but_excludes_selected_ce() -> None:
    _, _, loss, stats = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0,
    )
    expected_teacher = sum(
        stats[name]
        for name in (
            "scene_generation_v10_pair_mean_weighted_suffix_ce",
            "scene_generation_v10_pair_mean_first_error_top1_hinge",
            "scene_generation_v10_pair_mean_all_target_top1_retention_hinge",
            "scene_generation_v10_pair_mean_selected_top_competitor_hinge",
            "scene_generation_v10_pair_mean_selected_correct_vs_zero_hinge",
        )
    )
    selected_ce = stats[
        "scene_generation_v10_pair_mean_selected_full_vocab_ce_telemetry_only"
    ]

    assert stats["scene_generation_v10_pair_mean_first_error_top1_hinge"] > 0.0
    assert selected_ce > 0.0
    assert stats["scene_generation_v10_objective_teacher_loss"] == pytest.approx(
        expected_teacher
    )
    assert stats["scene_generation_v10_objective_total_loss"] == pytest.approx(
        expected_teacher
    )
    assert loss.item() == pytest.approx(expected_teacher)
    assert stats["scene_generation_v10_objective_total_loss"] != pytest.approx(
        expected_teacher + selected_ce
    )
    assert set(experimental_train._SCENE_STATE_CYCLE_RETENTION_LOG_METRICS).issubset(
        stats
    )
    assert all(
        math.isfinite(stats[name])
        for name in experimental_train._SCENE_STATE_CYCLE_RETENTION_LOG_METRICS
    )


def test_v10_prefix_mixed_edits_use_one_mean_over_event_slots() -> None:
    event_logits = torch.tensor(
        [
            [0.2, 0.8, -0.1, 0.0],
            [0.5, -0.2, 0.7, 0.1],
            [-0.1, 0.1, 0.2, 0.9],
        ],
        requires_grad=True,
    )
    positive_indices = torch.tensor([0, 1, 2])
    positive_ids = torch.tensor([0, 1, 2])
    negative_indices = torch.tensor([0, 1])
    negative_ids = torch.tensor([1, 2])

    loss, positive_mean, negative_mean = (
        experimental_train.DeltaMemTrainer._scene_state_cycle_retention_prefix_event_loss(
            event_logits,
            positive_indices=positive_indices,
            positive_token_ids=positive_ids,
            negative_indices=negative_indices,
            negative_token_ids=negative_ids,
        )
    )
    positive_values = torch.nn.functional.cross_entropy(
        event_logits,
        positive_ids,
        reduction="none",
    )
    negative_values = (
        experimental_train.DeltaMemTrainer._scene_state_generated_unlikelihood_values_from_logits(
            event_logits.index_select(0, negative_indices),
            negative_ids,
        )
    )
    expected = (
        positive_values.sum() + negative_values.sum()
    ) / event_logits.size(0)

    assert loss.item() == pytest.approx(expected.item())
    assert positive_mean.item() == pytest.approx(
        positive_values.sum().item() / event_logits.size(0)
    )
    assert negative_mean.item() == pytest.approx(
        negative_values.sum().item() / event_logits.size(0)
    )
    assert loss.item() == pytest.approx(
        positive_mean.item() + negative_mean.item()
    )
    assert loss.item() != pytest.approx(
        positive_values.mean().item() + negative_values.mean().item()
    )
    loss.backward()
    assert event_logits.grad is not None
    assert bool(event_logits.grad.abs().sum(dim=1).gt(0.0).all())


def test_v10_every_backward_and_returned_loss_scale_by_one_seventh() -> None:
    _, full_model, full_loss, _ = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0,
    )
    _, scaled_model, scaled_loss, _ = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0
        / experimental_train._SCENE_STATE_CYCLE_RETENTION_GRADIENT_ACCUMULATION_STEPS,
    )

    assert scaled_loss.item() == pytest.approx(full_loss.item() / 7.0)
    assert full_model.parameters_by_side.grad is not None
    assert scaled_model.parameters_by_side.grad is not None
    assert torch.allclose(
        scaled_model.parameters_by_side.grad,
        full_model.parameters_by_side.grad / 7.0,
    )

    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 7
    trainer.model_accepts_loss_kwargs = True
    trainer.compute_loss_func = lambda *args, **kwargs: None
    assert trainer._scene_state_generation_sequential_gradient_scale(
        num_items_in_batch=torch.tensor(14)
    ) == pytest.approx(1.0 / 7.0)


def test_v10_cycle_scaling_matches_trainer_managed_accumulation(tmp_path) -> None:
    training_args = experimental_train.TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        gradient_accumulation_steps=7,
        report_to=[],
    )
    trainer = experimental_train.Trainer(
        model=torch.nn.Linear(1, 1),
        args=training_args,
    )

    assert trainer.args.gradient_accumulation_steps == 7
    assert trainer.accelerator.gradient_accumulation_steps == 1


def test_v10_prefix_backward_contributions_also_scale_by_one_seventh() -> None:
    full_trainer, full_model, full_loss, _ = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0,
        prefix_probe=True,
    )
    scaled_trainer, scaled_model, scaled_loss, _ = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0 / 7.0,
        prefix_probe=True,
    )

    assert full_trainer.accelerator.backward_calls == 4
    assert scaled_trainer.accelerator.backward_calls == 4
    assert scaled_loss.item() == pytest.approx(full_loss.item() / 7.0)
    assert full_model.parameters_by_side.grad is not None
    assert scaled_model.parameters_by_side.grad is not None
    assert torch.allclose(
        scaled_model.parameters_by_side.grad,
        full_model.parameters_by_side.grad / 7.0,
    )


def test_v10_runtime_requires_exact_cycle_accumulation_and_data_skip() -> None:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 6
    with pytest.raises(ValueError, match="gradient_accumulation_steps=7"):
        trainer._validate_scene_state_generation_sequential_runtime()

    trainer.current_gradient_accumulation_steps = 7
    trainer.args = Namespace(optim="adamw_torch", ignore_data_skip=True)
    with pytest.raises(ValueError, match="ignore_data_skip=False"):
        trainer._validate_scene_state_generation_sequential_runtime()

    trainer.args = Namespace(optim="adamw_torch", ignore_data_skip=False)
    trainer.accelerator.gradient_accumulation_steps = 7
    with pytest.raises(ValueError, match="accelerator accumulation factor"):
        trainer._validate_scene_state_generation_sequential_runtime()

    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SYMMETRIC_OBJECTIVE_VERSION
    )
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        trainer._validate_scene_state_generation_sequential_runtime()


def test_v9_objective_remains_numerically_identical() -> None:
    trainer, model, loss, stats = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_SYMMETRIC_OBJECTIVE_VERSION,
        gradient_scale=1.0,
    )

    assert trainer.accelerator.backward_calls == 2
    assert loss.item() == pytest.approx(1.390519380569458)
    assert model.parameters_by_side.grad is not None
    assert model.parameters_by_side.grad.tolist() == pytest.approx(
        [-1.4332103729248047, -1.174180030822754]
    )
    assert stats["scene_generation_v9_objective_total_loss"] == pytest.approx(
        1.390519380569458
    )
    assert stats["scene_generation_v9_pair_mean_selected_full_vocab_ce"] == pytest.approx(
        0.667932391166687
    )
    assert not any(name.startswith("scene_generation_v10_") for name in stats)


def _cycle_binding() -> dict[str, object]:
    canonical_pairs = [
        [1, 14],
        [22, 26],
        [3, 24],
        [5, 9],
        [10, 23],
        [19, 28],
        [20, 31],
    ]
    cycles = [
        canonical_pairs,
        canonical_pairs[2:] + canonical_pairs[:2],
        list(reversed(canonical_pairs)),
        canonical_pairs[1:] + canonical_pairs[:1],
    ]
    return {
        "canonical_value14_pairs": canonical_pairs,
        "pair_indices": tuple(tuple(pair) for cycle in cycles for pair in cycle),
        "checkpoint_steps": [7, 14, 21, 28],
    }


def test_v10_cycle_schedule_requires_every_pair_once_per_cycle() -> None:
    binding = _cycle_binding()
    experimental_train._validate_scene_state_v10_cycle_schedule(binding)

    invalid = dict(binding)
    pairs = list(invalid["pair_indices"])
    pairs[6] = pairs[0]
    invalid["pair_indices"] = tuple(pairs)
    with pytest.raises(ValueError, match="each canonical pair exactly once"):
        experimental_train._validate_scene_state_v10_cycle_schedule(invalid)


def test_v10_protocol_uses_optimizer_cycle_endpoints() -> None:
    protocol = {
        "memory_objective_version": (
            experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
        ),
        "gradient_accumulation_steps": 7,
        "train_sampler_mode": experimental_train._V10_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE,
        "ignore_data_skip": False,
        "logging_steps": 1,
        "train_schedule": {
            "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
            "checkpoint_steps": [7, 14, 21, 28],
            "optimizer_checkpoint_steps": [1, 2, 3, 4],
            "microbatch_cycle_size": 7,
            "resume_schedule_cursor_formula": "global_step_times_7_v1",
        },
    }

    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(
        protocol
    ) == (1, 2, 3, 4)
    assert experimental_train._scene_memory_v9_protocol_checkpoint_steps(protocol) is None

    protocol["ignore_data_skip"] = True
    with pytest.raises(ValueError, match="cycle protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)

    protocol["ignore_data_skip"] = False
    protocol["logging_steps"] = 2
    with pytest.raises(ValueError, match="cycle protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)


@pytest.mark.parametrize("schedule", [None, {"schema": "wrong"}])
def test_v10_protocol_rejects_missing_or_wrong_schedule(schedule: object) -> None:
    protocol = {
        "memory_objective_version": (
            experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
        ),
        "train_schedule": schedule,
    }

    with pytest.raises(ValueError, match="cycle protocol schedule differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)


def test_v10_trainer_protocol_constants_match_launch_contract() -> None:
    assert experimental_train._V10_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE == (
        scene_memory_v10_launch_contract.FIXED_SAMPLER_MODE
    )
    assert experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_FORMULA == (
        scene_memory_v10_launch_contract.OBJECTIVE_FORMULA
    )
    assert experimental_train._SCENE_STATE_CYCLE_RETENTION_BACKWARD_MODE == (
        scene_memory_v10_launch_contract.BACKWARD_MODE
    )
    assert experimental_train._SCENE_STATE_CYCLE_RETENTION_GENERATED_MODE == (
        scene_memory_v10_launch_contract.GENERATED_PREFIX_MODE
    )
    assert experimental_train._SCENE_STATE_CYCLE_RETENTION_MODE == (
        scene_memory_v10_launch_contract.CYCLE_RETENTION_MODE
    )


def test_v10_cycle_telemetry_is_seven_pair_mean_at_log_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LogTrainer(experimental_train.DeltaMemTrainer):
        def __getattr__(self, name: str):
            if name.startswith("_last_"):
                return 0.0
            raise AttributeError(name)

    trainer = object.__new__(LogTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
    )
    trainer.memory_loss_mode = "scene_state_generation_ce"
    trainer.scene_boundary_payload_ce_weight = 0.0
    trainer.model = None
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0
    averaged = None
    for presentation in range(1, 8):
        stats = {
            name: float(presentation)
            for name in experimental_train._SCENE_STATE_CYCLE_RETENTION_LOG_METRICS
        }
        averaged = trainer._scene_state_cycle_retention_aggregate_memory_stats(stats)
    assert averaged is not None
    assert averaged["scene_generation_v10_objective_total_loss"] == 4.0
    assert averaged["scene_generation_v10_cycle_pair_presentations"] == 7.0
    assert trainer._scene_state_cycle_retention_metric_presentations == 0

    trainer._last_scene_generation_objective_logs = (
        trainer._scene_state_generation_log_metrics(averaged)
    )
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        experimental_train.Trainer,
        "log",
        lambda self, logs, start_time=None: captured.update(logs),
    )
    trainer.log({"loss": averaged["scene_generation_v10_objective_total_loss"]})

    assert captured["delta/scene_generation_v10_objective_total_loss"] == 4.0
    assert captured["delta/scene_generation_v10_cycle_pair_presentations"] == 7.0


def test_v10_cycle_telemetry_rejects_nonfinite_values() -> None:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
    )
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer._scene_state_cycle_retention_aggregate_memory_stats(
            {"scene_generation_v10_objective_total_loss": float("nan")}
        )


def test_v10_warm_start_contract_rejects_noncycle_arguments() -> None:
    args = SimpleNamespace(
        warm_start_mode=experimental_train._SCENE_V10_WARM_START_MODE,
        resume_from_checkpoint=None,
        resume_mode="exact",
        memory_loss_mode="scene_state_generation_ce",
        scene_state_generation_objective_version=(
            experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION
        ),
        scene_state_generated_prefix_correction_weight=0.5,
        scene_state_generated_unlikelihood_weight=0.0,
        target_layers=",".join(str(index) for index in range(42)),
        delta_heads="q,o",
        rank=4,
        alpha=8.0,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=4,
        rwkv_ms_chunk_size=128,
        rwkv_ms_semantics_version=2,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        online_gain=0.2,
        memory_fusion_mode="add",
        memory_fusion_placement="attention_output",
        memory_fusion_residual_scale=1.0,
        memory_fusion_residual_scale_max=1.0,
        trainable_delta_scale=True,
        delta_scale_init=0.1,
        delta_scale_max=0.5,
        delta_scale_granularity="head",
        delta_scale_parameterization="alpha_over_rank",
        memory_readout_mode="delta",
        memory_write_source="learned_hidden",
        memory_write_granularity="token",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=7,
        ignore_data_skip=False,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        warmup_steps=0,
        optim="adamw_torch_fused",
        num_train_epochs=1.0,
        max_steps=1,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        eval_steps=1000,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        frozen_mlp_activation_checkpointing=True,
        max_length=256,
        max_write_length=2048,
        per_device_eval_batch_size=1,
        weight_decay=0.0,
        dtype="bfloat16",
        bf16=True,
        tf32=True,
        scene_state_generated_unlikelihood_max_wrong_tokens=4,
        scene_state_generated_rollout_extra_tokens=4,
        scene_state_generated_rollout_max_tokens=24,
        scene_boundary_payload_ce_weight=0.0,
        memory_dropout_no_memory_prob=0.0,
        memory_dropout_state_only_prob=0.0,
        memory_base_kl_weight=0.0,
        memory_contrast_weight=0.0,
        memory_representation_weight=0.0,
        memory_kl_weight=0.0,
        memory_causal_weight=0.0,
        memory_anchor_weight=0.0,
        memory_recover_weight=0.0,
        write_sparsity_weight=0.0,
        memory_partition_alignment_weight=0.0,
        memory_partition_entropy_weight=0.0,
        memory_partition_balance_weight=0.0,
        train_sampler_seed=None,
        validation_split_ratio=0.0,
        load_best_model_at_end=False,
        seed=42,
        data_seed=42,
    )
    experimental_train._validate_scene_v10_warm_start_args(args)

    args.gradient_accumulation_steps = 1
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        experimental_train._validate_scene_v10_warm_start_args(args)
