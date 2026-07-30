from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.tests.test_scene_memory_v10_training import (
    _pair_inputs,
    _run_reciprocal_objective,
)
from deltamem.tests.test_scene_memory_v9_training import (
    _PairModel,
    _bind_pair_model,
    _masks,
    _trainer,
)


class _RepairModel(torch.nn.Module):
    def __init__(self, *, sequence_capacity: int = 16, vocabulary_size: int = 8) -> None:
        super().__init__()
        self.position_logits = torch.nn.Parameter(
            torch.linspace(
                -0.2,
                0.2,
                steps=sequence_capacity * vocabulary_size,
            ).reshape(sequence_capacity, vocabulary_size)
        )
        self.writer = torch.nn.Parameter(torch.tensor(0.2))
        self.reader = torch.nn.Parameter(torch.tensor(0.3))
        self.online_state: torch.Tensor | None = None
        self.forward_calls = 0
        self.last_input_ids: torch.Tensor | None = None
        self.last_use_cache: bool | None = None

    def forward(self, input_ids, attention_mask, *, use_cache=False, **kwargs):
        del attention_mask, kwargs
        self.forward_calls += 1
        self.last_input_ids = input_ids.detach().clone()
        self.last_use_cache = bool(use_cache)
        if self.online_state is None:
            raise RuntimeError("Suffix-repair replay was not differentiably primed")
        sequence_length = input_ids.size(1)
        vocabulary_size = self.position_logits.size(1)
        basis = torch.arange(
            vocabulary_size,
            device=input_ids.device,
            dtype=self.position_logits.dtype,
        )
        state_signal = self.online_state * self.reader
        logits = self.position_logits[:sequence_length].unsqueeze(0)
        return {"logits": logits + state_signal * basis.view(1, 1, -1)}


def _first_raw_divergence(generated: list[int], gold: list[int]) -> int:
    common_length = min(len(generated), len(gold))
    for index in range(common_length):
        if generated[index] != gold[index]:
            return index
    return common_length


def _repair_masks(gold_count: int) -> dict[str, torch.Tensor]:
    if gold_count < 3:
        raise ValueError("Synthetic repair gold requires schema, decision, termination")
    prefix = [False, False]
    return {
        "target": torch.tensor([prefix + [True] * gold_count]),
        "schema": torch.tensor(
            [prefix + [True] + [False] * (gold_count - 1)]
        ),
        "decision": torch.tensor(
            [prefix + [False] + [True] * (gold_count - 2) + [False]]
        ),
        "termination": torch.tensor(
            [prefix + [False] * (gold_count - 1) + [True]]
        ),
    }


def _run_suffix_repair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generated: list[int],
    gold: list[int],
) -> tuple[
    experimental_train.DeltaMemTrainer,
    _RepairModel,
    torch.Tensor | None,
    dict[str, float],
]:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
    )
    model = _RepairModel()
    prompt = torch.tensor([[6, 7]])
    gold_tensor = torch.tensor(gold)
    generated_tensor = torch.tensor(generated)
    divergence = _first_raw_divergence(generated, gold)
    trainer._scene_state_generated_greedy_rollout = lambda *args, **kwargs: {
        "generated_token_ids": generated_tensor,
        "gold_token_ids": gold_tensor,
        "prompt_input_ids": prompt,
        "prompt_attention_mask": torch.ones_like(prompt),
        "generation_start": prompt.size(1),
        "first_divergence": divergence,
        "exact_through_termination": generated == gold,
    }
    trainer._reset_online_state = lambda active: setattr(
        active,
        "online_state",
        None,
    )
    trainer._prime_episode_state = lambda active, **kwargs: setattr(
        active,
        "online_state",
        active.writer * kwargs["write_input_ids"][0, 0].float(),
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda active, enabled: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda active, mask: None,
    )
    masks = _repair_masks(len(gold))
    model_inputs = {
        "input_ids": torch.cat((prompt, gold_tensor.unsqueeze(0)), dim=1),
        "attention_mask": torch.ones(1, prompt.size(1) + len(gold), dtype=torch.long),
        "labels": torch.tensor([[-100, -100] + gold]),
    }
    loss, stats = trainer._scene_state_generated_suffix_repair_branch(
        model,
        model_inputs,
        online_state_snapshot={"mock.delta_state": torch.ones(1)},
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        write_message_ids=None,
        write_sentence_ids=None,
        target_mask=masks["target"],
        schema_mask=masks["schema"],
        decision_mask=masks["decision"],
        termination_mask=masks["termination"],
    )
    return trainer, model, loss, stats


def _run_v11_reciprocal(
    *,
    gradient_scale: float,
) -> tuple[
    experimental_train.DeltaMemTrainer,
    _PairModel,
    torch.Tensor,
    dict[str, float],
]:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 7
    trainer.scene_state_generated_prefix_correction_weight = 0.5
    model = _PairModel()
    binding = pytest.MonkeyPatch()
    _bind_pair_model(trainer, binding)
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
        binding.undo()
    return trainer, model, loss, stats


def test_v11_multi_token_deletion_repairs_every_gold_suffix_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, loss, stats = _run_suffix_repair(
        monkeypatch,
        generated=[1],
        gold=[1, 2, 3, 4, 5],
    )

    assert loss is not None
    assert model.last_input_ids is not None
    assert model.last_input_ids.tolist() == [[6, 7, 1, 2, 3, 4, 5]]
    assert model.last_use_cache is False
    assert stats["scene_generation_suffix_repair_first_divergence"] == 1.0
    assert stats["scene_generation_suffix_repair_repaired_tail_token_count"] == 4.0
    assert stats["scene_generation_suffix_repair_weighted_ce"] > 0.0
    assert stats[
        "scene_generation_suffix_repair_first_wrong_unlikelihood"
    ] == 0.0
    loss.backward()
    assert model.position_logits.grad is not None
    repaired_predictors = model.position_logits.grad[2:6].abs().sum(dim=1)
    assert bool(repaired_predictors.gt(0.0).all())
    assert repaired_predictors.numel() == 4


def test_v11_extra_token_replays_common_prefix_and_suppresses_only_first_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, loss, stats = _run_suffix_repair(
        monkeypatch,
        generated=[1, 6, 2, 7],
        gold=[1, 2, 7],
    )

    assert loss is not None
    assert model.last_input_ids is not None
    assert model.last_input_ids.tolist() == [[6, 7, 1, 2, 7]]
    assert stats["scene_generation_suffix_repair_first_divergence"] == 1.0
    assert stats["scene_generation_suffix_repair_repaired_tail_token_count"] == 2.0
    assert stats[
        "scene_generation_suffix_repair_first_wrong_unlikelihood"
    ] > 0.0


def test_v11_substitution_trains_suffix_and_writer_read_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, loss, stats = _run_suffix_repair(
        monkeypatch,
        generated=[1, 6, 7],
        gold=[1, 2, 7],
    )

    assert loss is not None
    assert stats["scene_generation_suffix_repair_repaired_tail_token_count"] == 2.0
    loss.backward()
    assert model.writer.grad is not None and model.writer.grad.abs().item() > 0.0
    assert model.reader.grad is not None and model.reader.grad.abs().item() > 0.0


def test_v11_exact_rollout_has_zero_repair_and_no_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model, loss, stats = _run_suffix_repair(
        monkeypatch,
        generated=[1, 2, 7],
        gold=[1, 2, 7],
    )

    assert loss is None
    assert model.forward_calls == 0
    assert stats["scene_generation_suffix_repair_weighted_ce"] == 0.0
    assert stats[
        "scene_generation_suffix_repair_first_wrong_unlikelihood"
    ] == 0.0
    assert stats["scene_generation_suffix_repair_repaired_tail_token_count"] == 0.0
    assert stats["scene_generation_suffix_repair_exact_fraction"] == 1.0


def test_v11_every_backward_and_returned_loss_scale_by_one_seventh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def repair_probe(self, active_model, *args, **kwargs):
        del self, args, kwargs
        return active_model.parameters_by_side.sum(), {
            "scene_generation_suffix_repair_weighted_ce": 0.3,
            "scene_generation_suffix_repair_first_wrong_unlikelihood": 0.2,
            "scene_generation_suffix_repair_first_divergence": 1.0,
            "scene_generation_suffix_repair_repaired_tail_token_count": 2.0,
            "scene_generation_suffix_repair_exact_fraction": 0.0,
            "scene_generation_suffix_repair_rollout_token_count": 3.0,
        }

    monkeypatch.setattr(
        experimental_train.DeltaMemTrainer,
        "_scene_state_generated_suffix_repair_branch",
        repair_probe,
    )
    full_trainer, full_model, full_loss, full_stats = _run_v11_reciprocal(
        gradient_scale=1.0,
    )
    scaled_trainer, scaled_model, scaled_loss, _ = _run_v11_reciprocal(
        gradient_scale=1.0 / 7.0,
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
    assert set(experimental_train._SCENE_STATE_SUFFIX_REPAIR_LOG_METRICS).issubset(
        full_stats
    )
    assert not any(name.startswith("scene_generation_v10_") for name in full_stats)


def test_v10_cycle_objective_numerics_remain_unchanged() -> None:
    trainer, model, loss, stats = _run_reciprocal_objective(
        experimental_train._SCENE_STATE_CYCLE_RETENTION_OBJECTIVE_VERSION,
        gradient_scale=1.0,
    )

    assert trainer.accelerator.backward_calls == 2
    assert loss.item() == pytest.approx(1.597586989402771)
    assert model.parameters_by_side.grad is not None
    assert model.parameters_by_side.grad.tolist() == pytest.approx(
        [0.3516947031021118, 0.49852555990219116]
    )
    assert stats["scene_generation_v10_objective_total_loss"] == pytest.approx(
        1.597586989402771
    )
    assert not any(name.startswith("scene_generation_v11_") for name in stats)


def test_v11_protocol_constants_are_truthful() -> None:
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION == (
        "scene_state_generation_ce_symmetric_cycle_suffix_repair_v5"
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_TRAINING_PROTOCOL_SCHEMA_VERSION == 14
    assert experimental_train._V11_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE == (
        "explicit_ordered_v11_canonical_seven_pair_cycle_v1"
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_CHECKPOINT_STEPS == (1,)
    assert (
        experimental_train._SCENE_STATE_SUFFIX_REPAIR_PRESENTATION_CHECKPOINT_STEPS
        == (7,)
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_CONTINUATION_POLICY == (
        "forbidden"
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_GENERATED_MODE == (
        "first_raw_token_divergence_common_prefix_weighted_gold_suffix_ce_first_"
        "generated_wrong_unlikelihood_v5"
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_FORMULA == (
        "symmetric_pair_mean(weighted_full_gold_ce(schema=2,decision=4,termination=1) "
        "+ first_error_top1_hinge(0.2) + all_target_top1_retention_hinge(0.2) + "
        "selected_top_competitor_hinge(0.2) + "
        "selected_correct_vs_detached_zero_nll_hinge(0.2) + 0.5 * "
        "first_divergence_suffix_repair(weighted_gold_suffix_ce(schema=2,decision=4,"
        "termination=1) + first_generated_wrong_unlikelihood)); "
        "selected_full_vocab_ce=telemetry_only"
    )
    assert experimental_train._SCENE_STATE_SUFFIX_REPAIR_BACKWARD_MODE == (
        "sequential_pair_zero_probe_full_gold_first_error_all_target_retention_"
        "then_first_divergence_gold_suffix_replay_v6"
    )


def test_v11_built_protocol_describes_live_suffix_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
    args.memory_loss_mode = "scene_state_generation_ce"
    args.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
    )
    args.scene_state_generated_prefix_correction_weight = 0.5
    args.scene_state_generated_unlikelihood_weight = 0.0
    args.scene_state_generated_unlikelihood_max_wrong_tokens = 1
    args.gradient_accumulation_steps = 7
    args.learning_rate = 2e-4
    args.lr_scheduler_type = "constant"
    args.warmup_ratio = 0.0
    args.warmup_steps = 0
    args.max_steps = 1
    args.max_grad_norm = 1.0
    args.ignore_data_skip = False
    args.logging_steps = 1
    args.save_total_limit = 1
    args.load_best_model_at_end = False
    args.memory_contrast_weight = 0.0
    args.memory_causal_weight = 0.0
    args.memory_anchor_weight = 0.0
    args.memory_recover_weight = 0.0
    args.episode_read_write_enabled = False
    binding = {
        "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
        "indices": tuple(range(28)),
        "total_steps": 28,
        "checkpoint_steps": [7, 14, 21, 28],
        "canonical_value14_pairs": [[1, 14]],
        "pair_indices": tuple((1, 14) for _ in range(28)),
    }
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_source_manifest_identity",
        lambda args: {"schema": "synthetic"},
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_identity_protocol_pairing_summary",
        lambda manifest: {},
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

    expected = {
        "schema_version": 14,
        "memory_objective_version": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
        ),
        "train_sampler_mode": (
            experimental_train._V11_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE
        ),
        "scene_generation_objective_formula": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_FORMULA
        ),
        "scene_generation_backward_mode": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_BACKWARD_MODE
        ),
        "scene_generation_generated_prefix_correction_weight": 0.5,
        "scene_generation_generated_prefix_correction_mode": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_MODE
        ),
        "scene_generation_generated_prefix_max_correction_events": 1,
        "scene_generation_suffix_repair_mode": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_MODE
        ),
        "scene_generation_suffix_repair_weight": 0.5,
        "scene_generation_suffix_repair_divergence": (
            "first_raw_token_divergence_including_length_mismatch_v1"
        ),
        "scene_generation_suffix_repair_gold_weighting": (
            "schema_2_decision_4_termination_1_v1"
        ),
        "scene_generation_suffix_repair_first_wrong_unlikelihood": True,
        "scene_generation_suffix_repair_premature_termination_suppression": True,
        "scene_generation_suffix_repair_exact_rollout_loss": 0.0,
        "scene_generation_generated_replay_state_gradient": True,
        "scene_generation_generated_replay_read_path_gradient": True,
        "max_grad_norm": 1.0,
    }
    assert {key: protocol[key] for key in expected} == expected
    schedule = protocol["train_schedule"]
    assert schedule["checkpoint_steps"] == [7]
    assert schedule["optimizer_checkpoint_steps"] == [1]
    assert schedule["microbatch_cycle_size"] == 7
    assert schedule["continuation_policy"] == "forbidden"
    assert "resume_schedule_cursor_formula" not in schedule


def test_v11_cycle_protocol_accepts_only_v11_sampler() -> None:
    protocol = {
        "memory_objective_version": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
        ),
        "max_steps": 1,
        "gradient_accumulation_steps": 7,
        "train_sampler_mode": experimental_train._V11_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE,
        "ignore_data_skip": False,
        "logging_steps": 1,
        "train_schedule": {
            "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
            "checkpoint_steps": [7],
            "optimizer_checkpoint_steps": [1],
            "microbatch_cycle_size": 7,
            "continuation_policy": "forbidden",
        },
    }

    assert experimental_train._scene_memory_v10_protocol_checkpoint_steps(
        protocol
    ) == (1,)
    protocol["train_sampler_mode"] = (
        experimental_train._V10_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE
    )
    with pytest.raises(ValueError, match="cycle protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)

    protocol["train_sampler_mode"] = (
        experimental_train._V11_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE
    )
    schedule = protocol["train_schedule"]
    schedule["optimizer_checkpoint_steps"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="cycle protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)

    schedule["optimizer_checkpoint_steps"] = [1]
    schedule["resume_schedule_cursor_formula"] = "global_step_times_7_v1"
    with pytest.raises(ValueError, match="cycle protocol differs"):
        experimental_train._scene_memory_v10_protocol_checkpoint_steps(protocol)


def _valid_v11_contract_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.args = SimpleNamespace(
        max_steps=1,
        gradient_accumulation_steps=7,
        max_grad_norm=1.0,
    )
    trainer.resume_mode = "exact"
    trainer.training_protocol = {
        "schema_version": 14,
        "memory_objective_version": (
            experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
        ),
        "max_steps": 1,
        "gradient_accumulation_steps": 7,
        "max_grad_norm": 1.0,
        "train_sampler_mode": (
            experimental_train._V11_PAIR_TRAIN_SCHEDULE_SAMPLER_MODE
        ),
        "ignore_data_skip": False,
        "train_schedule": {
            "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
            "checkpoint_steps": [7],
            "optimizer_checkpoint_steps": [1],
            "microbatch_cycle_size": 7,
            "continuation_policy": "forbidden",
        },
    }
    trainer.continuation_manifest = {
        "schema": experimental_train.SCENE_V11_WARM_START_RECEIPT_SCHEMA,
        "mode": experimental_train._SCENE_V11_WARM_START_MODE,
        "source_global_step": 56,
        "trainer_resume_from_checkpoint": None,
        "target_initial_global_step": 0,
        "target_fresh_start": {
            "initial_global_step": 0,
            "optimizer_state": "fresh",
            "scheduler_state": "fresh",
            "trainer_state": "fresh",
            "rng_state": "fresh_from_v11_seed",
        },
    }
    four_cycles = experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS * 4
    trainer.train_schedule_binding = {
        "schema": experimental_train._SCENE_MEMORY_V9_CURRICULUM_SCHEMA,
        "total_steps": 28,
        "pair_indices": four_cycles,
    }
    trainer.train_schedule_indices = tuple(low for low, _ in four_cycles)
    return trainer


def test_v11_trainer_contract_requires_fresh_warm_start_and_one_cycle() -> None:
    trainer = _valid_v11_contract_trainer()
    trainer._validate_scene_state_v11_trainer_contract()

    trainer.continuation_manifest = None
    with pytest.raises(ValueError, match="fresh_v11_warm_start"):
        trainer._validate_scene_state_v11_trainer_contract()

    trainer = _valid_v11_contract_trainer()
    trainer.args.max_steps = 2
    with pytest.raises(ValueError, match="training_horizon"):
        trainer._validate_scene_state_v11_trainer_contract()

    trainer = _valid_v11_contract_trainer()
    trainer.resume_mode = "extend"
    with pytest.raises(ValueError, match="resume_mode"):
        trainer._validate_scene_state_v11_trainer_contract()

    trainer = _valid_v11_contract_trainer()
    schedule = trainer.training_protocol["train_schedule"]
    schedule["optimizer_checkpoint_steps"] = [
        1,
        2,
        3,
        4,
    ]
    with pytest.raises(ValueError, match="training_protocol"):
        trainer._validate_scene_state_v11_trainer_contract()

    trainer = _valid_v11_contract_trainer()
    schedule = trainer.training_protocol["train_schedule"]
    schedule["resume_schedule_cursor_formula"] = "global_step_times_7_v1"
    with pytest.raises(ValueError, match="training_protocol"):
        trainer._validate_scene_state_v11_trainer_contract()

    trainer = _valid_v11_contract_trainer()
    pairs = list(trainer.train_schedule_binding["pair_indices"])
    pairs[1] = pairs[0]
    trainer.train_schedule_binding["pair_indices"] = tuple(pairs)
    with pytest.raises(ValueError, match="one_cycle_schedule"):
        trainer._validate_scene_state_v11_trainer_contract()


def _v11_telemetry_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SUFFIX_REPAIR_OBJECTIVE_VERSION
    )
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0
    trainer._scene_state_v11_cycle_pairs = []
    return trainer


def test_v11_cycle_telemetry_emits_exact_ordered_pair_ordinals() -> None:
    trainer = _v11_telemetry_trainer()
    averaged = None
    for position, pair in enumerate(
        experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS
    ):
        trainer._scene_state_v11_record_pair_presentation(
            torch.tensor([pair[0]]),
            torch.tensor([pair[1]]),
        )
        averaged = trainer._scene_state_cycle_retention_aggregate_memory_stats(
            {"scene_generation_v11_objective_total_loss": float(position + 1)}
        )

    assert averaged is not None
    assert averaged["scene_generation_v11_cycle_pair_presentations"] == 7.0
    for position, (low_ordinal, high_ordinal) in enumerate(
        experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS
    ):
        low_key = f"scene_generation_v11_cycle_pair_{position}_low_ordinal"
        high_key = f"scene_generation_v11_cycle_pair_{position}_high_ordinal"
        assert averaged[low_key] == float(low_ordinal)
        assert averaged[high_key] == float(high_ordinal)
        assert torch.isfinite(torch.tensor(averaged[low_key]))
        assert torch.isfinite(torch.tensor(averaged[high_key]))
    assert trainer._scene_state_cycle_retention_metric_presentations == 0
    assert trainer._scene_state_v11_cycle_pairs == []


def test_v11_cycle_telemetry_rejects_duplicate_and_wrong_order() -> None:
    trainer = _v11_telemetry_trainer()
    first = experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS[0]
    trainer._scene_state_v11_record_pair_presentation(
        torch.tensor([first[0]]),
        torch.tensor([first[1]]),
    )
    with pytest.raises(ValueError, match="duplicate"):
        trainer._scene_state_v11_record_pair_presentation(
            torch.tensor([first[0]]),
            torch.tensor([first[1]]),
        )

    trainer = _v11_telemetry_trainer()
    second = experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS[1]
    with pytest.raises(ValueError, match="order differs"):
        trainer._scene_state_v11_record_pair_presentation(
            torch.tensor([second[0]]),
            torch.tensor([second[1]]),
        )


def test_v11_cycle_telemetry_rejects_missing_pair_and_partial_cycle() -> None:
    trainer = _v11_telemetry_trainer()
    with pytest.raises(RuntimeError, match="missing its ordered pair"):
        trainer._scene_state_cycle_retention_aggregate_memory_stats(
            {"scene_generation_v11_objective_total_loss": 1.0}
        )

    trainer = _v11_telemetry_trainer()
    for pair in experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS[:-1]:
        trainer._scene_state_v11_record_pair_presentation(
            torch.tensor([pair[0]]),
            torch.tensor([pair[1]]),
        )
        trainer._scene_state_cycle_retention_aggregate_memory_stats(
            {"scene_generation_v11_objective_total_loss": 1.0}
        )
    trainer._last_scene_generation_objective_logs = {}
    with pytest.raises(RuntimeError, match="complete seven-pair"):
        trainer.log({"loss": 1.0})
