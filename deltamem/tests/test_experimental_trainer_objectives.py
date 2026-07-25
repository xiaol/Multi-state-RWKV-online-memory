from __future__ import annotations

from argparse import Namespace
import sys

import pytest
import torch
from datasets import Dataset

import deltamem.train.delta_sft_experimental as experimental_train


class _FakeDeltaModule:
    def __init__(self) -> None:
        self.active_delta_heads = frozenset({"q", "o"})
        self.delta_state = None
        self.last_delta_o_ratio = None


class _DeltaAwareModel(torch.nn.Module):
    def __init__(self, delta_module: _FakeDeltaModule) -> None:
        super().__init__()
        self.delta_module = delta_module
        self.delta_parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[frozenset[str]] = []
        self.input_id_calls: list[torch.Tensor] = []
        self.grad_enabled_calls: list[bool] = []

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        del attention_mask, labels, kwargs
        active_heads = self.delta_module.active_delta_heads
        self.calls.append(active_heads)
        self.input_id_calls.append(input_ids.detach().clone())
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        base_logits = torch.zeros(input_ids.size(0), input_ids.size(1), 3)
        if active_heads:
            self.delta_module.delta_state = torch.tensor([3.0])
            self.delta_module.last_delta_o_ratio = torch.tensor(2.0)
            delta_pattern = torch.zeros_like(base_logits)
            delta_pattern[:, 1, 0] = 2.0
            delta_pattern[:, 2, 1] = 1.0
            logits = base_logits + self.delta_parameter * delta_pattern
            loss = self.delta_parameter * 0.0 + 2.0
        else:
            self.delta_module.delta_state = torch.tensor([0.0])
            self.delta_module.last_delta_o_ratio = None
            logits = base_logits
            loss = torch.tensor(1.0)
        return {"loss": loss, "logits": logits}


class _ReducedLogitDeltaAwareModel(_DeltaAwareModel):
    def __init__(self, delta_module: _FakeDeltaModule) -> None:
        super().__init__(delta_module)
        self.logits_to_keep_calls: list[int | torch.Tensor] = []

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        self.logits_to_keep_calls.append(logits_to_keep)
        outputs = super().forward(
            input_ids,
            attention_mask,
            labels=labels,
            **kwargs,
        )
        if isinstance(logits_to_keep, torch.Tensor):
            outputs = dict(outputs)
            outputs["logits"] = outputs["logits"].index_select(1, logits_to_keep)
        return outputs


class _ContentContrastReadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.active_write_token: int | None = None
        self.read_calls: list[tuple[torch.Tensor, bool, int | None]] = []

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        del attention_mask, labels, kwargs
        self.read_calls.append(
            (
                input_ids.detach().clone(),
                torch.is_grad_enabled(),
                self.active_write_token,
            )
        )
        if self.active_write_token == 10:
            loss = 1.0 + self.parameter
        elif self.active_write_token == 20:
            loss = 2.0 + 2.0 * self.parameter
        else:
            raise AssertionError("read executed without a primed write")
        logits = self.parameter * torch.ones(input_ids.size(0), input_ids.size(1), 3)
        return {"loss": loss, "logits": logits}


def _build_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.episode_read_write_enabled = False

    def reset_online_state(model) -> None:
        model.delta_module.delta_state = None
        model.delta_module.last_delta_o_ratio = None

    trainer._reset_online_state = reset_online_state
    trainer._prime_episode_state = lambda model, **kwargs: None
    return trainer


def _episode_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 2, 1]]),
    }


def _compute_context_dropout(
    trainer: experimental_train.DeltaMemTrainer,
    model: torch.nn.Module,
    **episode_overrides,
):
    model_inputs = episode_overrides.pop("model_inputs", _episode_inputs())
    episode_kwargs = {
        "loss_kwargs": {},
        "write_input_ids": torch.tensor([[1]]),
        "write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "write_message_ids": None,
        "write_sentence_ids": None,
        "state_only_input_ids": None,
        "state_only_attention_mask": None,
        "state_only_labels": None,
        "state_only_write_input_ids": None,
        "state_only_write_attention_mask": None,
        "state_only_write_message_ids": None,
        "state_only_write_sentence_ids": None,
        "teacher_input_ids": torch.tensor([[9, 1, 2, 3, 4]]),
        "teacher_attention_mask": torch.ones(1, 5, dtype=torch.long),
        "teacher_labels": torch.tensor([[-100, -100, -100, 2, 1]]),
    }
    episode_kwargs.update(episode_overrides)
    return trainer._compute_context_dropout_ce(model, model_inputs, **episode_kwargs)


@pytest.fixture
def delta_controls(monkeypatch: pytest.MonkeyPatch):
    delta_module = _FakeDeltaModule()
    monkeypatch.setattr(
        experimental_train,
        "iter_delta_mem_modules",
        lambda model: iter((("delta", delta_module),)),
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda model, mask: None,
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda model, enabled: None,
    )
    return delta_module


def test_temporarily_disable_delta_heads_restores_after_failure(
    delta_controls: _FakeDeltaModule,
) -> None:
    original_heads = delta_controls.active_delta_heads

    with pytest.raises(RuntimeError, match="teacher failed"):
        with experimental_train._temporarily_disable_delta_heads(object()):
            assert delta_controls.active_delta_heads == frozenset()
            raise RuntimeError("teacher failed")

    assert delta_controls.active_delta_heads == original_heads


def test_preserve_delta_runtime_restores_after_failure(
    delta_controls: _FakeDeltaModule,
) -> None:
    original_state = torch.tensor([3.0], requires_grad=True)
    original_ratio = torch.tensor(2.0)
    delta_controls.delta_state = original_state
    delta_controls.last_delta_o_ratio = original_ratio
    delta_controls.write_enabled = True

    with pytest.raises(RuntimeError, match="teacher failed"):
        with experimental_train._preserve_delta_runtime(object()):
            delta_controls.delta_state = torch.tensor([0.0])
            delta_controls.last_delta_o_ratio = None
            delta_controls.write_enabled = False
            raise RuntimeError("teacher failed")

    assert delta_controls.delta_state is original_state
    assert delta_controls.last_delta_o_ratio is original_ratio
    assert delta_controls.write_enabled is True


def test_no_memory_dropout_uses_exact_base_and_remains_backward_safe(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 1.0
    trainer.memory_dropout_state_only_prob = 0.0
    trainer.memory_base_kl_weight = 0.0
    model = _DeltaAwareModel(delta_controls)

    loss, outputs, stats = _compute_context_dropout(trainer, model)

    assert model.calls == [frozenset()]
    assert delta_controls.active_delta_heads == frozenset({"q", "o"})
    assert stats["wmem"] == 0.0
    assert torch.equal(outputs["logits"], torch.zeros_like(outputs["logits"]))
    assert loss.requires_grad

    loss.backward()
    assert model.delta_parameter.grad is not None
    assert model.delta_parameter.grad.item() == 0.0


def test_context_dropout_is_disabled_during_evaluation(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 1.0
    trainer.memory_dropout_state_only_prob = 0.0
    trainer.memory_base_kl_weight = 0.0
    model = _DeltaAwareModel(delta_controls).eval()

    _, outputs, stats = _compute_context_dropout(trainer, model)

    assert model.calls == [frozenset({"q", "o"})]
    assert torch.count_nonzero(outputs["logits"]) > 0
    assert stats["wmem"] == 1.0


def test_episode_read_configures_writes_before_context_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer()
    trainer.episode_read_write_enabled = True
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda model, enabled: events.append(("write", enabled)),
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda model, mask: events.append(("mask", mask)),
    )

    trainer._configure_episode_read(object(), _episode_inputs())

    assert events[0] == ("write", True)
    assert events[1][0] == "mask"
    assert torch.equal(
        events[1][1],
        torch.tensor([[True, True, False, False]]),
    )


def test_context_dropout_base_kl_uses_no_delta_teacher(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 0.0
    trainer.memory_dropout_state_only_prob = 0.0
    trainer.memory_base_kl_weight = 0.5
    model = _DeltaAwareModel(delta_controls)

    loss, outputs, stats = _compute_context_dropout(trainer, model)
    student_mask, teacher_mask, targets = trainer._validate_supervised_next_token_alignment(
        _episode_inputs()["labels"],
        _episode_inputs()["attention_mask"],
        torch.tensor([[-100, -100, -100, 2, 1]]),
        torch.ones(1, 5, dtype=torch.long),
    )
    expected_kl = trainer._selected_teacher_kl_loss(
        trainer._select_supervised_next_token_logits(outputs["logits"], student_mask),
        trainer._select_supervised_next_token_logits(
            torch.zeros(1, 5, 3),
            teacher_mask,
        ),
    )

    assert model.calls == [frozenset(), frozenset({"q", "o"})]
    assert model.grad_enabled_calls == [False, True]
    assert model.input_id_calls[0].tolist() == [[9, 1, 2, 3, 4]]
    assert model.input_id_calls[1].tolist() == _episode_inputs()["input_ids"].tolist()
    assert delta_controls.active_delta_heads == frozenset({"q", "o"})
    assert stats["wmem"] == 1.0
    assert stats["teacher_loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert stats["kl_loss"] == pytest.approx(expected_kl.item())
    assert loss.item() == pytest.approx(2.0 + 0.5 * expected_kl.item())
    assert torch.equal(delta_controls.delta_state, torch.tensor([3.0]))
    assert delta_controls.last_delta_o_ratio.item() == 2.0

    loss.backward()
    assert model.delta_parameter.grad is not None
    assert model.delta_parameter.grad.abs().item() > 0.0


def test_context_dropout_base_kl_requires_canonical_teacher_tensors(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 0.0
    trainer.memory_dropout_state_only_prob = 0.0
    trainer.memory_base_kl_weight = 0.5
    model = _DeltaAwareModel(delta_controls)

    with pytest.raises(ValueError, match="requires canonical teacher tensors"):
        _compute_context_dropout(
            trainer,
            model,
            teacher_input_ids=None,
            teacher_attention_mask=None,
            teacher_labels=None,
        )

    assert model.calls == []


def test_projected_teacher_logits_preserve_per_row_position_mapping() -> None:
    trainer = _build_trainer()
    teacher_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, False, False],
        ]
    )
    predictor_positions = torch.tensor([1, 2, 3, 4])
    full_logits = torch.arange(12, dtype=torch.float32).view(2, 6, 1)
    projected_logits = full_logits.index_select(1, predictor_positions)

    selected = trainer._select_projected_supervised_next_token_logits(
        projected_logits,
        teacher_mask,
        predictor_positions,
    )

    assert selected.squeeze(-1).tolist() == [3.0, 4.0, 7.0, 8.0]


def test_context_dropout_reduces_teacher_logits_for_padded_batch(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 0.0
    trainer.memory_dropout_state_only_prob = 0.0
    trainer.memory_base_kl_weight = 0.5
    model = _ReducedLogitDeltaAwareModel(delta_controls)
    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 2, 1], [-100, 2, 1, -100]]),
    }
    teacher_input_ids = torch.tensor(
        [[9, 9, 1, 2, 3, 4], [8, 5, 6, 7, 0, 0]]
    )
    teacher_attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    )
    teacher_labels = torch.tensor(
        [[-100, -100, -100, -100, 2, 1], [-100, -100, 2, 1, -100, -100]]
    )

    loss, outputs, stats = _compute_context_dropout(
        trainer,
        model,
        model_inputs=model_inputs,
        write_input_ids=torch.tensor([[1], [1]]),
        write_attention_mask=torch.ones(2, 1, dtype=torch.long),
        teacher_input_ids=teacher_input_ids,
        teacher_attention_mask=teacher_attention_mask,
        teacher_labels=teacher_labels,
    )

    assert isinstance(model.logits_to_keep_calls[0], torch.Tensor)
    assert model.logits_to_keep_calls[0].tolist() == [1, 2, 3, 4]
    assert model.logits_to_keep_calls[1] == 0
    assert model.input_id_calls[0].shape == (2, 6)
    assert model.input_id_calls[1].shape == (2, 4)
    assert outputs["logits"].shape == (2, 4, 3)
    assert stats["teacher_loss"] == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert torch.isfinite(loss)

    loss.backward()
    assert model.delta_parameter.grad is not None


@pytest.mark.parametrize(
    ("teacher_labels", "message"),
    [
        (
            torch.tensor([[-100, -100, 2, -100]]),
            "target counts",
        ),
        (
            torch.tensor([[-100, -100, 0, 1]]),
            "target token IDs",
        ),
    ],
)
def test_supervised_next_token_alignment_rejects_mismatched_teacher_targets(
    teacher_labels: torch.Tensor,
    message: str,
) -> None:
    trainer = _build_trainer()

    with pytest.raises(ValueError, match=message):
        trainer._validate_supervised_next_token_alignment(
            _episode_inputs()["labels"],
            _episode_inputs()["attention_mask"],
            teacher_labels,
            torch.ones_like(teacher_labels),
        )


def test_state_only_dropout_uses_state_only_read_and_write_tensors(
    delta_controls: _FakeDeltaModule,
) -> None:
    trainer = _build_trainer()
    trainer.memory_dropout_no_memory_prob = 0.0
    trainer.memory_dropout_state_only_prob = 1.0
    trainer.memory_base_kl_weight = 0.5
    primed: list[dict[str, object]] = []
    trainer._prime_episode_state = lambda model, **kwargs: primed.append(kwargs)
    model = _DeltaAwareModel(delta_controls)
    state_only_input_ids = torch.tensor([[9, 8, 7]])
    state_only_attention_mask = torch.ones_like(state_only_input_ids)
    state_only_labels = torch.tensor([[-100, 8, 7]])
    state_only_write_input_ids = torch.tensor([[6, 5]])
    state_only_write_attention_mask = torch.ones_like(state_only_write_input_ids)

    _, _, stats = _compute_context_dropout(
        trainer,
        model,
        state_only_input_ids=state_only_input_ids,
        state_only_attention_mask=state_only_attention_mask,
        state_only_labels=state_only_labels,
        state_only_write_input_ids=state_only_write_input_ids,
        state_only_write_attention_mask=state_only_write_attention_mask,
    )

    assert len(model.input_id_calls) == 1
    assert torch.equal(model.input_id_calls[0], state_only_input_ids)
    assert len(primed) == 1
    assert primed[0]["write_input_ids"] is state_only_write_input_ids
    assert primed[0]["write_attention_mask"] is state_only_write_attention_mask
    assert model.calls == [frozenset({"q", "o"})]
    assert stats["wmem"] == 1.0
    assert stats["kl_loss"] == 0.0


def test_base_kl_uses_supervised_next_token_positions() -> None:
    trainer = _build_trainer()
    teacher_logits = torch.zeros(1, 4, 2)
    student_logits = teacher_logits.clone()
    student_logits[:, 0, 0] = 8.0
    student_logits[:, 3, 0] = 8.0
    labels = torch.tensor([[-100, -100, -100, 1]])
    attention_mask = torch.ones_like(labels)

    ignored_kl = trainer._masked_next_token_kl_loss(
        student_logits,
        teacher_logits,
        labels,
        attention_mask,
    )
    student_logits[:, 2, 0] = 8.0
    supervised_kl = trainer._masked_next_token_kl_loss(
        student_logits,
        teacher_logits,
        labels,
        attention_mask,
    )

    assert ignored_kl.item() == 0.0
    assert supervised_kl.item() > 0.0


def test_content_contrast_objective_exact_arithmetic_and_gradient_signs() -> None:
    trainer = _build_trainer()
    trainer.memory_contrast_weight = 0.25
    trainer.memory_margin = 0.5
    correct_loss = torch.tensor(1.0, requires_grad=True)
    wrong_loss = torch.tensor(2.0, requires_grad=True)

    total, contrast, gap = trainer._content_contrast_objective(correct_loss, wrong_loss)

    expected_contrast = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert gap.item() == pytest.approx(1.0)
    assert contrast.item() == pytest.approx(expected_contrast.item())
    assert total.item() == pytest.approx(1.0 + 0.25 * expected_contrast.item())

    total.backward()
    sigmoid = torch.sigmoid(torch.tensor(-1.0)).item()
    assert correct_loss.grad.item() == pytest.approx(1.0 + 0.25 * sigmoid / 0.5)
    assert wrong_loss.grad.item() == pytest.approx(-0.25 * sigmoid / 0.5)
    assert correct_loss.grad.item() > 0.0
    assert wrong_loss.grad.item() < 0.0


def test_content_contrast_resets_between_branches_and_never_runs_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _build_trainer()
    trainer.memory_contrast_weight = 0.25
    trainer.memory_margin = 0.5
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.episode_read_write_enabled = False
    model = _ContentContrastReadModel()
    events: list[tuple[str, object]] = []

    def reset_online_state(active_model) -> None:
        events.append(("reset", active_model.active_write_token))
        active_model.active_write_token = None

    def prime_episode_state(active_model, **kwargs) -> None:
        token = int(kwargs["write_input_ids"][0, 0].item())
        events.append(("prime", (token, torch.is_grad_enabled())))
        active_model.active_write_token = token

    trainer._reset_online_state = reset_online_state
    trainer._prime_episode_state = prime_episode_state
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_write_enabled",
        lambda active_model, enabled: events.append(("write", enabled)),
    )
    monkeypatch.setattr(
        experimental_train,
        "set_delta_mem_read_context_mask",
        lambda active_model, mask: events.append(("mask", mask.detach().clone())),
    )

    loss, outputs, stats = trainer._compute_content_contrast_ce(
        model,
        _episode_inputs(),
        loss_kwargs={},
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        write_message_ids=torch.zeros(1, 1, dtype=torch.long),
        write_sentence_ids=torch.zeros(1, 1, dtype=torch.long),
        negative_write_input_ids=torch.tensor([[20]]),
        negative_write_attention_mask=torch.ones(1, 1, dtype=torch.long),
        negative_write_message_ids=torch.zeros(1, 1, dtype=torch.long),
        negative_write_sentence_ids=torch.zeros(1, 1, dtype=torch.long),
    )

    assert [event for event in events if event[0] == "reset"] == [
        ("reset", None),
        ("reset", 10),
    ]
    assert [event for event in events if event[0] == "prime"] == [
        ("prime", (10, True)),
        ("prime", (20, False)),
    ]
    assert [event for event in events if event[0] == "write"] == [
        ("write", False),
        ("write", False),
    ]
    assert len(model.read_calls) == 2
    assert all(call[1] for call in model.read_calls)
    assert [call[2] for call in model.read_calls] == [10, 20]
    assert torch.equal(model.read_calls[0][0], model.read_calls[1][0])
    assert stats["keep_loss"] == pytest.approx(1.0)
    assert stats["corrupt_loss"] == pytest.approx(2.0)
    assert stats["margin_gap"] == pytest.approx(1.0)
    assert stats["causal_loss"] == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(-1.0)).item()
    )
    assert stats["teacher_loss"] == 0.0
    assert stats["kl_loss"] == 0.0
    assert outputs["loss"] is loss

    loss.backward()
    assert model.parameter.grad is not None
    assert torch.isfinite(model.parameter.grad)


def test_masked_lm_loss_never_supervises_the_first_sequence_token() -> None:
    trainer = _build_trainer()
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels = torch.tensor([[1, -100, -100]])
    first_token_only = labels.ne(-100)

    loss = trainer._masked_lm_loss(logits, labels, first_token_only)

    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_parse_args_exposes_memory_objective_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path),
            "--memory-loss-mode",
            "context_ablation_ce",
            "--context-ablation-mode",
            "state_only",
            "--context-ablation-no-state-prob",
            "0.1",
            "--context-ablation-state-only-prob",
            "0.3",
            "--memory-dropout-no-memory-prob",
            "0.2",
            "--episode-read-write-enabled",
            "--frozen-mlp-activation-checkpointing",
        ],
    )

    args = experimental_train.parse_args()

    assert args.memory_loss_mode == "context_ablation_ce"
    assert args.context_ablation_mode == "state_only"
    assert args.context_ablation_no_state_prob == 0.1
    assert args.context_ablation_state_only_prob == 0.3
    assert args.memory_dropout_no_memory_prob == 0.2
    assert args.episode_read_write_enabled is True
    assert args.frozen_mlp_activation_checkpointing is True


def test_parse_args_preserves_legacy_objective_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path),
        ],
    )

    args = experimental_train.parse_args()

    assert args.memory_loss_mode == "context_dropout_ce"
    assert args.memory_dropout_no_memory_prob == 0.0
    assert args.memory_dropout_state_only_prob == 0.0
    assert args.memory_base_kl_weight == 0.0
    assert args.episode_read_write_enabled is False
    assert args.frozen_mlp_activation_checkpointing is False


def test_parse_args_accepts_kl_free_content_contrast_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path),
            "--memory-loss-mode",
            "content_contrast_ce",
            "--memory-kl-weight",
            "0",
            "--memory-contrast-weight",
            "0.25",
            "--memory-margin",
            "0.5",
        ],
    )

    args = experimental_train.parse_args()

    assert args.memory_loss_mode == "content_contrast_ce"
    assert args.memory_kl_weight == 0.0
    assert args.memory_base_kl_weight == 0.0
    assert args.memory_contrast_weight == 0.25
    assert args.memory_margin == 0.5
    assert args.episode_read_write_enabled is False


def test_parse_args_rejects_invalid_combined_dropout_probability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path),
            "--memory-dropout-no-memory-prob",
            "0.8",
            "--memory-dropout-state-only-prob",
            "0.3",
        ],
    )

    with pytest.raises(ValueError, match="memory dropout probabilities"):
        experimental_train.parse_args()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--memory-dropout-no-memory-prob", "-0.1"], "memory dropout probabilities"),
        (["--memory-dropout-state-only-prob", "-0.1"], "memory dropout probabilities"),
        (
            [
                "--context-ablation-no-state-prob",
                "0.8",
                "--context-ablation-state-only-prob",
                "0.3",
            ],
            "context ablation probabilities",
        ),
        (["--context-ablation-state-only-prob", "-0.1"], "context ablation probabilities"),
        (["--memory-base-kl-weight", "-0.1"], "memory-base-kl-weight"),
        (
            [
                "--memory-loss-mode",
                "keep_only",
                "--memory-base-kl-weight",
                "0.1",
            ],
            "requires memory-loss-mode=context_dropout_ce",
        ),
        (
            [
                "--memory-backend",
                "rwkv_ms",
                "--output-init",
                "zero",
            ],
            "gradient-dead",
        ),
        (
            ["--memory-loss-mode", "content_contrast_ce"],
            "all KL weights to be zero",
        ),
        (
            [
                "--memory-loss-mode",
                "content_contrast_ce",
                "--memory-kl-weight",
                "0",
                "--episode-read-write-enabled",
            ],
            "read writes to be disabled",
        ),
        (
            [
                "--memory-loss-mode",
                "content_contrast_ce",
                "--memory-kl-weight",
                "0",
                "--memory-contrast-weight",
                "-0.1",
            ],
            "non-negative contrast weight",
        ),
        (
            [
                "--memory-loss-mode",
                "content_contrast_ce",
                "--memory-kl-weight",
                "0",
                "--write-sparsity-weight",
                "0.1",
            ],
            "write sparsity loss to be disabled",
        ),
    ],
)
def test_parse_args_rejects_invalid_objective_protocols(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    extra_args: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft_experimental.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path),
            *extra_args,
        ],
    )

    with pytest.raises(ValueError, match=message):
        experimental_train.parse_args()


class _CacheTokenizer:
    name_or_path = "cache-tokenizer"
    pad_token_id = 0


def _pairing_row(write_token: int) -> dict[str, list[int]]:
    return {
        "write_input_ids": [write_token, write_token + 1],
        "write_attention_mask": [1, 1],
        "write_message_ids": [0, 0],
        "write_sentence_ids": [0, 0],
    }


def _collator_episode_row(write_token: int) -> dict[str, list[int]]:
    row = _pairing_row(write_token)
    row.update(
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
            "teacher_input_ids": [write_token, 1, 2, 3],
            "teacher_attention_mask": [1, 1, 1, 1],
            "teacher_labels": [-100, -100, 2, 3],
            "state_only_write_input_ids": [write_token],
            "state_only_write_attention_mask": [1],
            "state_only_write_message_ids": [0],
            "state_only_write_sentence_ids": [0],
            "state_only_input_ids": [1, 2, 3],
            "state_only_attention_mask": [1, 1, 1],
            "state_only_labels": [-100, 2, 3],
        }
    )
    return row


def test_content_contrast_pairing_is_post_split_half_rotation_and_auditable() -> None:
    train = Dataset.from_list([_pairing_row(token) for token in (10, 20, 30, 40)])
    paired, split_manifest = experimental_train.materialize_content_contrast_pairs(
        train,
        split_name="train",
    )

    assert paired["content_contrast_source_index"] == [0, 1, 2, 3]
    assert paired["content_contrast_partner_index"] == [2, 3, 0, 1]
    assert paired["content_contrast_source_id"] == [
        "train:0",
        "train:1",
        "train:2",
        "train:3",
    ]
    assert paired["content_contrast_partner_id"] == [
        "train:2",
        "train:3",
        "train:0",
        "train:1",
    ]
    for source_index, partner_index in enumerate((2, 3, 0, 1)):
        for field in ("input_ids", "attention_mask", "message_ids", "sentence_ids"):
            assert paired[source_index][f"negative_write_{field}"] == (
                train[partner_index][f"write_{field}"]
            )
        assert paired[source_index]["content_contrast_source_write_sha256"] != (
            paired[source_index]["content_contrast_partner_write_sha256"]
        )
        assert paired[source_index]["content_contrast_negative_write_sha256"] == (
            paired[source_index]["content_contrast_partner_write_sha256"]
        )
        assert split_manifest["pairs"][source_index]["negative_write_sha256"] == (
            paired[source_index]["content_contrast_negative_write_sha256"]
        )
    assert split_manifest["rotation"] == 2
    assert split_manifest["sample_count"] == 4
    assert split_manifest["pairing_version"] == (
        experimental_train._CONTENT_CONTRAST_PAIRING_VERSION
    )
    assert len(split_manifest["pairs_sha256"]) == 64
    assert len(split_manifest["manifest_sha256"]) == 64


def test_content_contrast_pairing_stays_within_each_post_split_partition() -> None:
    train, train_manifest = experimental_train.materialize_content_contrast_pairs(
        Dataset.from_list([_pairing_row(token) for token in (10, 20)]),
        split_name="train",
    )
    evaluation, eval_manifest = experimental_train.materialize_content_contrast_pairs(
        Dataset.from_list([_pairing_row(token) for token in (110, 120)]),
        split_name="eval",
    )
    manifest = experimental_train.build_content_contrast_pairing_manifest(
        tokenized_fingerprint="base-fingerprint",
        data_seed=17,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
    )

    assert all(identifier.startswith("train:") for identifier in train["content_contrast_partner_id"])
    assert all(identifier.startswith("eval:") for identifier in evaluation["content_contrast_partner_id"])
    assert max(value for row in train["negative_write_input_ids"] for value in row) < 100
    assert min(value for row in evaluation["negative_write_input_ids"] for value in row) > 100
    assert manifest["pairing_scope"] == "within_post_split_partition"
    assert set(manifest["splits"]) == {"train", "eval"}
    assert len(manifest["manifest_sha256"]) == 64


def test_split_then_pair_does_not_mutate_persisted_objective_neutral_base(tmp_path) -> None:
    base = Dataset.from_list(
        [
            {"sample_id": index, **_pairing_row(10 + index * 10)}
            for index in range(8)
        ]
    )
    cache_dir = tmp_path / "tokenized-base"
    base.save_to_disk(str(cache_dir))
    persisted = experimental_train.load_from_disk(str(cache_dir))
    train, evaluation = experimental_train.split_tokenized_dataset(
        persisted,
        validation_split_ratio=0.25,
        data_seed=17,
    )
    assert evaluation is not None

    paired_train, _ = experimental_train.materialize_content_contrast_pairs(
        train,
        split_name="train",
    )
    paired_evaluation, _ = experimental_train.materialize_content_contrast_pairs(
        evaluation,
        split_name="eval",
    )

    train_tokens = {row[0] for row in train["write_input_ids"]}
    eval_tokens = {row[0] for row in evaluation["write_input_ids"]}
    assert {
        row[0] for row in paired_train["negative_write_input_ids"]
    }.issubset(train_tokens)
    assert {
        row[0] for row in paired_evaluation["negative_write_input_ids"]
    }.issubset(eval_tokens)
    assert train_tokens.isdisjoint(eval_tokens)
    assert not any(column.startswith("negative_write_") for column in persisted.column_names)
    reloaded = experimental_train.load_from_disk(str(cache_dir))
    assert not any(column.startswith("negative_write_") for column in reloaded.column_names)


@pytest.mark.parametrize("sample_count", [0, 1, 3])
def test_content_contrast_pairing_rejects_non_even_split_sizes(sample_count: int) -> None:
    dataset = Dataset.from_list([_pairing_row(10 + index * 10) for index in range(sample_count)])

    with pytest.raises(ValueError, match="even sample count >= 2"):
        experimental_train.materialize_content_contrast_pairs(dataset, split_name="train")


def test_content_contrast_pairing_rejects_equal_partner_writes() -> None:
    dataset = Dataset.from_list([_pairing_row(10), _pairing_row(10)])

    with pytest.raises(ValueError, match="equal writes"):
        experimental_train.materialize_content_contrast_pairs(dataset, split_name="train")


def test_episode_collator_emits_materialized_negative_write_for_batch_size_one() -> None:
    source = _collator_episode_row(10)
    donor = _collator_episode_row(20)
    paired, _ = experimental_train.materialize_content_contrast_pairs(
        Dataset.from_list([source, donor]),
        split_name="train",
    )

    batch = experimental_train.EpisodeCausalLMCollator(_CacheTokenizer())([paired[0]])

    assert batch["negative_write_input_ids"].shape == (1, 2)
    assert batch["negative_write_input_ids"].tolist() == [[20, 21]]
    assert batch["negative_write_attention_mask"].tolist() == [[1, 1]]
    assert batch["negative_write_message_ids"].tolist() == [[0, 0]]
    assert batch["negative_write_sentence_ids"].tolist() == [[0, 0]]


def _cache_args(**overrides) -> Namespace:
    values = {
        "dataset_name": "dataset",
        "dataset_split": "train",
        "train_file": None,
        "max_length": 128,
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 1,
        "max_write_length": 64,
        "memory_write_granularity": "token",
        "group_by_length": False,
        "memory_loss_mode": "context_dropout_ce",
        "memory_contrast_weight": 0.1,
        "memory_kl_weight": 0.1,
        "memory_margin": 0.1,
        "write_sparsity_weight": 0.0,
        "memory_partition_alignment_weight": 0.0,
        "memory_partition_entropy_weight": 0.0,
        "memory_partition_balance_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "episode_read_write_enabled": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_tokenized_cache_key_tracks_data_shape_but_not_objective_weights() -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    )
    tokenizer = _CacheTokenizer()
    base_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        tokenizer,
    )

    for override in (
        {"max_length": 256},
        {"training_mode": "dialogue"},
        {"assistant_loss_mode": "all_assistant_turns"},
        {"episode_recent_messages": 2},
        {"max_write_length": 96},
        {"memory_write_granularity": "sentence_mean"},
        {"group_by_length": True},
    ):
        assert experimental_train._tokenized_dataset_cache_key(
            _cache_args(**override),
            dataset,
            tokenizer,
        ) != base_key

    assert experimental_train._tokenized_dataset_cache_key(
        _cache_args(
            memory_loss_mode="context_ablation_ce",
            memory_dropout_no_memory_prob=0.2,
            memory_dropout_state_only_prob=0.3,
            episode_read_write_enabled=True,
        ),
        dataset,
        tokenizer,
    ) == base_key
    assert experimental_train._tokenized_dataset_cache_key(
        _cache_args(memory_base_kl_weight=0.4),
        dataset,
        tokenizer,
    ) == base_key
    assert experimental_train._tokenized_dataset_cache_key(
        _cache_args(
            memory_loss_mode="content_contrast_ce",
            memory_contrast_weight=0.25,
            memory_kl_weight=0.0,
            memory_margin=0.5,
        ),
        dataset,
        tokenizer,
    ) == base_key


def test_tokenized_cache_key_tracks_chat_template_and_local_tokenizer_artifacts(
    tmp_path,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    )
    tokenizer = _CacheTokenizer()
    tokenizer.chat_template = "template-a"
    first_template_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        tokenizer,
    )
    tokenizer.chat_template = "template-b"
    second_template_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        tokenizer,
    )
    assert second_template_key != first_template_key

    tokenizer.name_or_path = str(tmp_path)
    tokenizer_artifact = tmp_path / "tokenizer.json"
    tokenizer_artifact.write_text('{"version": 1}')
    first_artifact_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        tokenizer,
    )
    tokenizer_artifact.write_text('{"version": 2}')
    second_artifact_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        tokenizer,
    )
    assert second_artifact_key != first_artifact_key


def test_tokenized_cache_key_tracks_resolved_fallback_chat_template(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ]
    )
    fallback_config = tmp_path / "tokenizer_config.json"
    fallback_config.write_text('{"chat_template": "fallback-a"}')
    monkeypatch.setattr(
        experimental_train.project_chat_templates,
        "DEFAULT_LOCAL_MODEL_PATH",
        str(tmp_path),
    )
    template_loader = (
        experimental_train.project_chat_templates._load_chat_template_from_tokenizer_config
    )
    template_loader.cache_clear()
    first_tokenizer = _CacheTokenizer()
    first_tokenizer.name_or_path = "qwen3-cache-tokenizer"
    first_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        first_tokenizer,
    )
    assert first_tokenizer.chat_template == "fallback-a"

    fallback_config.write_text('{"chat_template": "fallback-b"}')
    template_loader.cache_clear()
    second_tokenizer = _CacheTokenizer()
    second_tokenizer.name_or_path = "qwen3-cache-tokenizer"
    second_key = experimental_train._tokenized_dataset_cache_key(
        _cache_args(),
        dataset,
        second_tokenizer,
    )
    assert second_tokenizer.chat_template == "fallback-b"
    assert second_key != first_key
    template_loader.cache_clear()


def test_training_protocol_records_canonical_teacher_objective_version() -> None:
    args = _cache_args()
    args.tokenized_dataset_dir = None
    args.memory_write_source = "learned_hidden"
    args.context_ablation_mode = "mixed"
    args.context_ablation_no_state_prob = 0.2
    args.context_ablation_state_only_prob = 0.2
    args.validation_split_ratio = 0.1
    args.seed = 7
    args.data_seed = 11
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = None
    args.gradient_accumulation_steps = 2
    args.learning_rate = 1e-4
    args.lr_scheduler_type = "cosine"
    args.warmup_ratio = 0.05
    args.weight_decay = 0.0
    args.optim = "adamw_torch"
    args.num_train_epochs = 1.0
    args.max_steps = -1
    args.eval_steps = 10
    args.save_steps = 10
    args.dtype = "bfloat16"
    args.bf16 = True
    args.tf32 = True
    args.memory_fusion_placement = "post_attention_norm"
    dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]]})

    protocol = experimental_train.build_training_protocol(
        args,
        dataset,
        effective_training_mode="episode",
        train_samples=1,
        eval_samples=0,
        warmup_steps=1,
    )

    assert protocol["schema_version"] == experimental_train._TRAINING_PROTOCOL_SCHEMA_VERSION
    assert protocol["memory_objective_version"] == experimental_train._MEMORY_OBJECTIVE_VERSION
    assert protocol["teacher_max_length"] == args.max_write_length + args.max_length
    assert protocol["frozen_mlp_activation_checkpointing"] is False
    assert protocol["memory_fusion_placement"] == "post_attention_norm"

    args.frozen_mlp_activation_checkpointing = True
    checkpointed_protocol = experimental_train.build_training_protocol(
        args,
        dataset,
        effective_training_mode="episode",
        train_samples=1,
        eval_samples=0,
        warmup_steps=1,
    )
    assert checkpointed_protocol["frozen_mlp_activation_checkpointing"] is True


def test_training_protocol_records_exact_content_contrast_pairing() -> None:
    tokenized = Dataset.from_list([_pairing_row(token) for token in (10, 20)])
    _, train_pairing = experimental_train.materialize_content_contrast_pairs(
        tokenized,
        split_name="train",
    )
    pairing_manifest = experimental_train.build_content_contrast_pairing_manifest(
        tokenized_fingerprint=getattr(tokenized, "_fingerprint", None),
        data_seed=11,
        train_manifest=train_pairing,
        eval_manifest=None,
    )
    args = _cache_args(
        memory_loss_mode="content_contrast_ce",
        memory_contrast_weight=0.25,
        memory_kl_weight=0.0,
        memory_margin=0.5,
    )
    protocol_values = {
        "tokenized_dataset_dir": None,
        "memory_write_source": "learned_hidden",
        "context_ablation_mode": "mixed",
        "context_ablation_no_state_prob": 0.2,
        "context_ablation_state_only_prob": 0.2,
        "validation_split_ratio": 0.0,
        "seed": 7,
        "data_seed": 11,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": None,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.0,
        "optim": "adamw_torch",
        "num_train_epochs": 1.0,
        "max_steps": -1,
        "eval_steps": 10,
        "save_steps": 10,
        "dtype": "bfloat16",
        "bf16": True,
        "tf32": True,
    }
    for name, value in protocol_values.items():
        setattr(args, name, value)

    protocol = experimental_train.build_training_protocol(
        args,
        tokenized,
        effective_training_mode="episode",
        train_samples=2,
        eval_samples=0,
        warmup_steps=1,
        content_contrast_pairing_manifest=pairing_manifest,
    )

    assert protocol["schema_version"] == (
        experimental_train._CONTENT_CONTRAST_TRAINING_PROTOCOL_SCHEMA_VERSION
    )
    assert protocol["memory_objective_version"] == (
        experimental_train._CONTENT_CONTRAST_OBJECTIVE_VERSION
    )
    assert protocol["memory_contrast_weight"] == 0.25
    assert protocol["memory_margin"] == 0.5
    assert protocol["memory_kl_weight"] == 0.0
    assert protocol["write_sparsity_weight"] == 0.0
    assert protocol["memory_partition_alignment_weight"] == 0.0
    assert protocol["memory_partition_entropy_weight"] == 0.0
    assert protocol["memory_partition_balance_weight"] == 0.0
    assert protocol["content_contrast_negative_priming_grad"] is False
    assert protocol["content_contrast_pairing"]["manifest_sha256"] == (
        pairing_manifest["manifest_sha256"]
    )
    assert protocol["content_contrast_pairing"]["splits"]["train"]["pairs_sha256"] == (
        train_pairing["pairs_sha256"]
    )


def test_legacy_tokenized_episode_dataset_fails_canonical_teacher_validation() -> None:
    legacy_dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "labels": [[-100, 2]],
        }
    )

    with pytest.raises(ValueError, match="missing canonical teacher columns; rebuild"):
        experimental_train.validate_canonical_teacher_columns(legacy_dataset)

    current_dataset = legacy_dataset.add_column("teacher_input_ids", [[1, 2]])
    current_dataset = current_dataset.add_column("teacher_attention_mask", [[1, 1]])
    current_dataset = current_dataset.add_column("teacher_labels", [[-100, 2]])
    experimental_train.validate_canonical_teacher_columns(current_dataset)


def test_validate_wrapped_target_layers_requires_exact_match() -> None:
    experimental_train.validate_wrapped_target_layers((4, 5), (4, 5))
    experimental_train.validate_wrapped_target_layers((), (4, 5))

    with pytest.raises(ValueError, match=r"missing=\(41,\)"):
        experimental_train.validate_wrapped_target_layers((4, 41), (4,))
    with pytest.raises(ValueError, match="unexpected"):
        experimental_train.validate_wrapped_target_layers((4,), (4, 5))
    with pytest.raises(ValueError, match="duplicates"):
        experimental_train.validate_wrapped_target_layers((4, 4), (4,))
