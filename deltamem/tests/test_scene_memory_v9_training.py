from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.train.scene_state_generation_alignment import (
    generated_correction_events,
)


def _trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.episode_read_write_enabled = False
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.memory_representation_weight = 0.0
    trainer.write_sparsity_weight = 0.0
    trainer.memory_partition_alignment_weight = 0.0
    trainer.memory_partition_entropy_weight = 0.0
    trainer.memory_partition_balance_weight = 0.0
    trainer.scene_boundary_payload_ce_weight = 0.0
    trainer.current_gradient_accumulation_steps = 1
    trainer.args = Namespace(optim="adamw_torch")
    trainer.compute_loss_context_manager = nullcontext
    trainer.scene_state_generated_unlikelihood_max_wrong_tokens = 4
    trainer.scene_state_generated_prefix_correction_weight = 0.0

    class Accelerator:
        distributed_type = SimpleNamespace(name="NO")
        gradient_accumulation_steps = 1

        def __init__(self) -> None:
            self.backward_calls = 0

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            assert kwargs == {}
            self.backward_calls += 1
            loss.backward()

    trainer.accelerator = Accelerator()
    return trainer


def _masks() -> dict[str, torch.Tensor]:
    return {
        "target": torch.tensor([[False, False, True, True, True, True]]),
        "content": torch.tensor([[False, False, True, True, True, False]]),
        "schema": torch.tensor([[False, False, True, False, False, False]]),
        "decision": torch.tensor([[False, False, False, True, True, False]]),
        "termination": torch.tensor([[False, False, False, False, False, True]]),
        "pair": torch.tensor([[False, False, False, True, False, False]]),
    }


def test_generated_correction_events_handle_all_edit_kinds_safely() -> None:
    substitution = generated_correction_events([1, 9, 3], [1, 2, 3], max_events=4)
    assert substitution == (
        experimental_train.generated_correction_events(
            [1, 9, 3], [1, 2, 3], max_events=4
        )[0],
    )
    assert (
        substitution[0].kind,
        substitution[0].generated_cursor,
        substitution[0].positive_token_id,
        substitution[0].negative_token_id,
    ) == ("substitution", 1, 2, 9)

    insertion = generated_correction_events([1, 8, 2], [1, 2], max_events=4)
    assert (
        insertion[0].kind,
        insertion[0].generated_cursor,
        insertion[0].positive_token_id,
        insertion[0].negative_token_id,
    ) == ("generated_insertion", 1, 2, 8)

    deletion = generated_correction_events([1, 4], [1, 2, 3, 4], max_events=4)
    assert len(deletion) == 1
    assert (
        deletion[0].kind,
        deletion[0].generated_cursor,
        deletion[0].positive_token_id,
        deletion[0].negative_token_id,
    ) == ("gold_deletion", 1, 2, None)

    trailing = generated_correction_events([1, 2, 8], [1, 2], max_events=4)
    assert trailing[0].positive_token_id is None
    assert trailing[0].negative_token_id == 8
    assert generated_correction_events([1, 2], [1, 2], max_events=4) == ()


def test_selected_loss_uses_full_vocab_top_competitor() -> None:
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = torch.tensor([1.0, 0.0, 2.0])
    labels = torch.tensor([[-100, 0]])
    pair_mask = torch.tensor([[False, True]])

    metrics = experimental_train.DeltaMemTrainer._scene_state_symmetric_selected_metrics(
        logits,
        labels,
        pair_mask,
    )
    assert metrics["selected_gold_margin_row"].item() == pytest.approx(-1.0)
    assert metrics["selected_top1_row"].item() == 0.0
    loss = metrics["selected_ce_row"].mean() + metrics[
        "selected_top_hinge_row"
    ].mean()
    loss.backward()
    assert logits.grad[0, 0, 0].item() < 0.0
    assert logits.grad[0, 0, 2].item() > 0.0


def test_selected_zero_hinge_has_correct_sign_and_detaches_zero() -> None:
    selected = torch.tensor([1.0], requires_grad=True)
    zero = torch.tensor([0.4], requires_grad=True)
    loss = experimental_train.DeltaMemTrainer._scene_state_symmetric_zero_hinge(
        selected,
        zero,
    ).mean()
    assert loss.item() == pytest.approx(0.8)
    loss.backward()
    assert selected.grad.item() > 0.0
    assert zero.grad is None

    assert experimental_train.DeltaMemTrainer._scene_state_symmetric_zero_hinge(
        torch.tensor([0.4]),
        torch.tensor([1.0]),
    ).item() == 0.0


class _PairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameters_by_side = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
        self.active_write_token: int | None = None

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        logits = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            3,
            dtype=self.parameters_by_side.dtype,
            device=input_ids.device,
        )
        positions = torch.arange(
            1,
            input_ids.size(1) + 1,
            dtype=logits.dtype,
            device=logits.device,
        )
        if self.active_write_token == 10:
            logits[..., 0] = self.parameters_by_side[0] * positions
        elif self.active_write_token == 20:
            logits[..., 1] = self.parameters_by_side[1] * positions
        return {"logits": logits}


def _bind_pair_model(
    trainer: experimental_train.DeltaMemTrainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer._reset_online_state = lambda model: setattr(
        model, "active_write_token", None
    )
    trainer._prime_episode_state = lambda model, **kwargs: setattr(
        model,
        "active_write_token",
        int(kwargs["write_input_ids"][0, 0].item()),
    )
    trainer._capture_live_online_state = lambda model: {
        "mock.delta_state": model.parameters_by_side.detach().clone()
    }
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda model, enabled: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda model, mask: None,
    )


def test_symmetric_sequential_backward_trains_both_full_gold_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    model = _PairModel()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs = {
        "input_ids": torch.tensor([[7, 8, 2, 0, 0, 0]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 0, 0, 0]]),
    }
    donor_inputs = {
        "input_ids": torch.tensor([[7, 8, 2, 1, 1, 1]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 1, 1, 1]]),
    }
    _, stats = trainer._scene_state_generation_symmetric_sequential_backward(
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
        gradient_scale=1.0,
    )
    assert trainer.accelerator.backward_calls == 2
    assert model.parameters_by_side.grad is not None
    assert bool(model.parameters_by_side.grad.ne(0).all())
    assert stats["scene_generation_source_full_gold_ce"] > 0.0
    assert stats["scene_generation_donor_full_gold_ce"] > 0.0
    assert set(experimental_train._SCENE_STATE_SYMMETRIC_LOG_METRICS).issubset(
        stats
    )
    assert all(
        torch.isfinite(torch.tensor(stats[name])).item()
        for name in experimental_train._SCENE_STATE_SYMMETRIC_LOG_METRICS
    )


def test_generated_prefix_correction_has_positive_gold_ce_and_writer_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()

    class WriterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.writer = torch.nn.Parameter(torch.tensor(0.2))
            self.reader = torch.nn.Parameter(torch.tensor(0.3))
            self.online_state = None

        def forward(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            logits = torch.zeros(
                input_ids.size(0), input_ids.size(1), 3, device=input_ids.device
            )
            logits[..., 2] = self.online_state * self.reader
            logits[..., 1] = -self.online_state * self.reader
            return {"logits": logits}

    model = WriterModel()
    trainer._reset_online_state = lambda active: setattr(active, "online_state", None)
    trainer._prime_episode_state = lambda active, **kwargs: setattr(
        active,
        "online_state",
        active.writer * kwargs["write_input_ids"][0, 0].float(),
    )
    trainer._scene_state_generated_greedy_rollout = lambda *args, **kwargs: {
        "generated_token_ids": torch.tensor([1]),
        "gold_token_ids": torch.tensor([2]),
        "prompt_input_ids": torch.tensor([[7, 8]]),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "generation_start": 2,
        "first_divergence": 0,
        "exact_through_termination": False,
    }
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
    loss, stats = trainer._scene_state_generated_prefix_correction_branch(
        model,
        {
            "input_ids": torch.tensor([[7, 8, 2]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 2]]),
        },
        online_state_snapshot={"mock.delta_state": torch.ones(1)},
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        write_message_ids=None,
        write_sentence_ids=None,
        target_mask=torch.tensor([[False, False, True]]),
        termination_mask=torch.tensor([[False, False, True]]),
    )
    assert loss is not None
    assert stats["scene_generation_prefix_positive_ce"] > 0.0
    assert stats["scene_generation_prefix_negative_unlikelihood"] > 0.0
    loss.backward()
    assert model.writer.grad is not None and model.writer.grad.abs().item() > 0.0
    assert model.reader.grad is not None and model.reader.grad.abs().item() > 0.0


def _symmetric_feature(*, canonical_low: bool = True) -> dict[str, object]:
    masks = {
        name: tensor[0].tolist() for name, tensor in _masks().items()
    }
    source_ids = [7, 8, 2, 0, 0, 0]
    donor_ids = [7, 8, 2, 1, 1, 1]
    source_labels = [-100, -100, 2, 0, 0, 0]
    donor_labels = [-100, -100, 2, 1, 1, 1]
    feature = {
        "input_ids": source_ids,
        "attention_mask": [1] * 6,
        "labels": source_labels,
        "write_input_ids": [10],
        "write_attention_mask": [1],
        "write_message_ids": [0],
        "write_sentence_ids": [-1],
        "teacher_input_ids": source_ids,
        "teacher_attention_mask": [1] * 6,
        "teacher_labels": source_labels,
        "state_only_write_input_ids": [],
        "state_only_write_attention_mask": [],
        "state_only_write_message_ids": [],
        "state_only_write_sentence_ids": [],
        "state_only_input_ids": [],
        "state_only_attention_mask": [],
        "state_only_labels": [],
        "scene_state_semantic_mask": masks["decision"],
        "scene_state_generation_target_mask": masks["target"],
        "scene_state_generation_content_mask": masks["content"],
        "scene_state_generation_schema_mask": masks["schema"],
        "scene_state_generation_decision_mask": masks["decision"],
        "scene_state_generation_termination_mask": masks["termination"],
        "scene_state_identity_target_mask": masks["pair"],
        "scene_state_identity_target_mask_sha256": experimental_train._canonical_json_sha256(
            masks["pair"]
        ),
        "scene_state_donor_write_input_ids": [20],
        "scene_state_donor_write_attention_mask": [1],
        "scene_state_donor_write_message_ids": [0],
        "scene_state_donor_write_sentence_ids": [-1],
        "scene_state_boundary_count": 1,
        "scene_state_donor_boundary_count": 1,
        "scene_state_identity_target_stratum": "same_cardinality_value",
        "scene_state_identity_donor_target_token_id": 1,
        "scene_state_source_index": 0,
        "scene_state_donor_index": 1,
        "scene_state_source_row_sha256": "1" * 64,
        "scene_state_donor_row_sha256": "2" * 64,
        "scene_state_source_label_sha256": "3" * 64,
        "scene_state_donor_label_sha256": "4" * 64,
        "scene_state_source_write_sha256": experimental_train._canonical_json_sha256(
            [10]
        ),
        "scene_state_donor_write_sha256": experimental_train._canonical_json_sha256(
            [20]
        ),
        "scene_state_donor_input_ids": donor_ids,
        "scene_state_donor_attention_mask": [1] * 6,
        "scene_state_donor_labels": donor_labels,
        "scene_state_donor_generation_target_mask": masks["target"],
        "scene_state_donor_generation_content_mask": masks["content"],
        "scene_state_donor_generation_schema_mask": masks["schema"],
        "scene_state_donor_generation_decision_mask": masks["decision"],
        "scene_state_donor_generation_termination_mask": masks["termination"],
        "scene_state_donor_identity_target_mask": masks["pair"],
        "scene_state_donor_identity_target_mask_sha256": experimental_train._canonical_json_sha256(
            masks["pair"]
        ),
        "scene_state_generation_symmetric_full_pair": True,
        "scene_state_pair_canonical_low": canonical_low,
    }
    return feature


def test_symmetric_collator_emits_one_canonical_pair_payload() -> None:
    collator = experimental_train.EpisodeCausalLMCollator(
        SimpleNamespace(pad_token_id=99)
    )
    batch = collator([_symmetric_feature()])
    assert batch["input_ids"].shape[0] == 1
    assert batch["scene_state_donor_input_ids"].tolist() == [
        [7, 8, 2, 1, 1, 1]
    ]
    assert batch["scene_state_source_index"].tolist() == [0]
    assert batch["scene_state_donor_index"].tolist() == [1]

    with pytest.raises(ValueError, match="canonical-low"):
        collator([_symmetric_feature(canonical_low=False)])


def test_v9_component_metrics_are_exposed_without_legacy_zero_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_stats = {
        "keep_loss": 0.75,
        "scene_generation_total_loss": 2.5,
        "scene_generation_weighted_ce": 0.75,
        "scene_generation_symmetric_selected_vocab_ce": 1.25,
        "scene_generation_symmetric_selected_top_hinge": 0.4,
        "scene_generation_symmetric_zero_hinge": 0.2,
        "scene_generation_symmetric_prefix_positive_ce": 0.6,
        "scene_generation_symmetric_prefix_negative_unlikelihood": 0.1,
    }

    metrics = (
        experimental_train.DeltaMemTrainer._scene_state_generation_log_metrics(
            memory_stats
        )
    )

    assert metrics == {
        "delta/scene_generation_total_loss": 2.5,
        "delta/scene_generation_weighted_ce": 0.75,
        "delta/scene_generation_symmetric_selected_vocab_ce": 1.25,
        "delta/scene_generation_symmetric_selected_top_hinge": 0.4,
        "delta/scene_generation_symmetric_zero_hinge": 0.2,
        "delta/scene_generation_symmetric_prefix_positive_ce": 0.6,
        "delta/scene_generation_symmetric_prefix_negative_unlikelihood": 0.1,
    }

    class LogTrainer(experimental_train.DeltaMemTrainer):
        def __getattr__(self, name: str):
            if name.startswith("_last_"):
                return 0.0
            raise AttributeError(name)

    trainer = object.__new__(LogTrainer)
    trainer.memory_loss_mode = "scene_state_generation_ce"
    trainer.scene_boundary_payload_ce_weight = 0.0
    trainer.model = None
    trainer._last_scene_generation_objective_logs = metrics
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        experimental_train.Trainer,
        "log",
        lambda self, logs, start_time=None: captured.update(logs),
    )

    trainer.log({"loss": 2.5})

    assert captured["delta/scene_generation_weighted_ce"] == 0.75
    assert captured["delta/scene_generation_symmetric_selected_vocab_ce"] == 1.25
