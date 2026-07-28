from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace
import weakref

from datasets import Dataset
import pytest
import torch
import torch.nn.functional as F

import deltamem.train.delta_sft_experimental as experimental_train


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

    class Accelerator:
        distributed_type = SimpleNamespace(name="NO")

        @staticmethod
        def backward(loss: torch.Tensor, **kwargs) -> None:
            assert kwargs == {}
            loss.backward()

    trainer.accelerator = Accelerator()
    return trainer


def _generation_masks() -> dict[str, torch.Tensor]:
    return {
        "target_mask": torch.tensor([[False, False, True, True, True, True]]),
        "content_mask": torch.tensor([[False, False, True, True, True, False]]),
        "schema_mask": torch.tensor([[False, False, True, False, False, False]]),
        "decision_mask": torch.tensor([[False, False, False, True, True, False]]),
        "termination_mask": torch.tensor([[False, False, False, False, False, True]]),
        "pair_target_mask": torch.tensor(
            [[False, False, False, True, False, False]]
        ),
    }


def _model_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[7, 8, 0, 0, 0, 0]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 0, 0, 0, 0]]),
    }


def _legacy_generation_ce_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    schema_mask: torch.Tensor,
    decision_mask: torch.Tensor,
    termination_mask: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    shift_labels = labels[:, 1:]

    def masked_row_ce(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        shift_logits = logits[:, :-1].float()
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        shift_mask = mask[:, 1:]
        counts = shift_mask.sum(dim=1)
        return (
            (token_ce * shift_mask).sum(dim=1) / counts,
            int(counts.sum().item()),
        )

    schema_row_ce, schema_count = masked_row_ce(schema_mask)
    decision_row_ce, decision_count = masked_row_ce(decision_mask)
    termination_row_ce, termination_count = masked_row_ce(termination_mask)
    shift_logits = logits[:, :-1].float()
    token_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    token_weights = (
        schema_mask[:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_SCHEMA_WEIGHT
        + decision_mask[:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_DECISION_WEIGHT
        + termination_mask[:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_TERMINATION_WEIGHT
    )
    weighted_generation_row_ce = (token_ce * token_weights).sum(dim=1) / (
        token_weights.sum(dim=1)
    )
    return {
        "token_ce": token_ce,
        "weighted_generation_row_ce": weighted_generation_row_ce,
        "schema_row_ce": schema_row_ce,
        "decision_row_ce": decision_row_ce,
        "termination_row_ce": termination_row_ce,
        "schema_token_count": schema_count,
        "decision_token_count": decision_count,
        "termination_token_count": termination_count,
    }


def _production_generation_ce_case(
    shifted_token_count: int,
    target_positions: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    sequence_length = shifted_token_count + 1
    labels = torch.full(
        (1, sequence_length),
        -100,
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(labels)
    masks = {
        name: torch.zeros(
            (1, sequence_length),
            device=device,
            dtype=torch.bool,
        )
        for name in ("target", "schema", "decision", "termination")
    }
    position_tensor = torch.tensor(target_positions, device=device)
    labels[0, position_tensor] = (
        torch.arange(len(target_positions), device=device) * 7919 + 17
    ) % 262144
    masks["target"][0, position_tensor] = True
    masks["schema"][0, position_tensor[:2]] = True
    masks["decision"][0, position_tensor[2:-1]] = True
    masks["termination"][0, position_tensor[-1:]] = True
    return {
        "labels": labels,
        "attention_mask": attention_mask,
        "target_mask": masks["target"],
        "schema_mask": masks["schema"],
        "decision_mask": masks["decision"],
        "termination_mask": masks["termination"],
    }


def test_generation_metrics_shared_token_ce_matches_legacy_reference() -> None:
    model_inputs = _model_inputs()
    masks = _generation_masks()
    generator = torch.Generator().manual_seed(20260729)
    base_logits = torch.randn(1, 6, 4, generator=generator, dtype=torch.float64)
    legacy_logits = base_logits.clone().requires_grad_()
    shared_logits = base_logits.clone().requires_grad_()
    legacy = _legacy_generation_ce_metrics(
        legacy_logits,
        model_inputs["labels"],
        schema_mask=masks["schema_mask"],
        decision_mask=masks["decision_mask"],
        termination_mask=masks["termination_mask"],
    )
    shared = _trainer()._scene_state_generation_metrics(
        shared_logits,
        model_inputs["labels"],
        model_inputs["attention_mask"],
        **masks,
        donor_target_token_ids=torch.tensor([1]),
    )

    for key in (
        "weighted_generation_row_ce",
        "schema_row_ce",
        "decision_row_ce",
        "termination_row_ce",
    ):
        torch.testing.assert_close(shared[key], legacy[key], rtol=0.0, atol=0.0)
    for key in (
        "schema_token_count",
        "decision_token_count",
        "termination_token_count",
    ):
        assert shared[key] == legacy[key]

    shared_loss = shared["weighted_generation_row_ce"].mean()
    legacy_loss = legacy["weighted_generation_row_ce"].mean()
    torch.testing.assert_close(shared_loss, legacy_loss, rtol=0.0, atol=0.0)
    shared_loss.backward()
    legacy_loss.backward()
    torch.testing.assert_close(
        shared_logits.grad,
        legacy_logits.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_generation_metrics_computes_selected_token_ce_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_inputs = _model_inputs()
    token_ce_inputs: list[tuple[torch.Size, torch.dtype]] = []
    original_cross_entropy = experimental_train.F.cross_entropy

    def tracked_cross_entropy(*args, **kwargs):
        token_ce_inputs.append((args[0].shape, args[0].dtype))
        return original_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(experimental_train.F, "cross_entropy", tracked_cross_entropy)
    _trainer()._scene_state_generation_metrics(
        torch.randn(1, 6, 4),
        model_inputs["labels"],
        model_inputs["attention_mask"],
        **_generation_masks(),
        donor_target_token_ids=torch.tensor([1]),
    )

    assert token_ce_inputs == [(torch.Size([4, 4]), torch.float32)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("shifted_token_count", "target_positions"),
    [
        (65, (1, 4, 9, 17, 28, 41, 53, 65)),
        (69, (2, 5, 8, 13, 19, 26, 34, 43, 51, 58, 64, 69)),
    ],
)
def test_generation_selected_ce_matches_legacy_at_production_shape(
    monkeypatch: pytest.MonkeyPatch,
    shifted_token_count: int,
    target_positions: tuple[int, ...],
) -> None:
    case = _production_generation_ce_case(
        shifted_token_count,
        target_positions,
    )
    generator = torch.Generator(device="cuda").manual_seed(
        20260729 + shifted_token_count
    )
    base_logits = torch.randn(
        1,
        shifted_token_count + 1,
        262144,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    legacy_logits = base_logits.clone().requires_grad_()
    selected_logits = base_logits.clone().requires_grad_()
    del base_logits
    legacy = _legacy_generation_ce_metrics(
        legacy_logits,
        case["labels"],
        schema_mask=case["schema_mask"],
        decision_mask=case["decision_mask"],
        termination_mask=case["termination_mask"],
    )
    token_ce_inputs: list[tuple[torch.Size, torch.dtype]] = []
    original_cross_entropy = experimental_train.F.cross_entropy

    def tracked_cross_entropy(*args, **kwargs):
        token_ce_inputs.append((args[0].shape, args[0].dtype))
        return original_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(experimental_train.F, "cross_entropy", tracked_cross_entropy)
    trainer = _trainer()
    schema_row_ce, schema_count, token_ce = (
        trainer._scene_state_generation_token_ce(
            selected_logits,
            case["labels"],
            case["attention_mask"],
            case["schema_mask"],
            description="schema",
            ce_target_mask=case["target_mask"],
        )
    )
    decision_row_ce, decision_count, token_ce = (
        trainer._scene_state_generation_token_ce(
            selected_logits,
            case["labels"],
            case["attention_mask"],
            case["decision_mask"],
            description="decision",
            token_ce=token_ce,
        )
    )
    termination_row_ce, termination_count, token_ce = (
        trainer._scene_state_generation_token_ce(
            selected_logits,
            case["labels"],
            case["attention_mask"],
            case["termination_mask"],
            description="termination",
            token_ce=token_ce,
        )
    )
    token_weights = (
        case["schema_mask"][:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_SCHEMA_WEIGHT
        + case["decision_mask"][:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_DECISION_WEIGHT
        + case["termination_mask"][:, 1:].float()
        * experimental_train._SCENE_STATE_GENERATION_TERMINATION_WEIGHT
    )
    weighted_generation_row_ce = (token_ce * token_weights).sum(dim=1) / (
        token_weights.sum(dim=1)
    )

    torch.testing.assert_close(token_ce, legacy["token_ce"], rtol=0.0, atol=0.0)
    for actual, key in (
        (weighted_generation_row_ce, "weighted_generation_row_ce"),
        (schema_row_ce, "schema_row_ce"),
        (decision_row_ce, "decision_row_ce"),
        (termination_row_ce, "termination_row_ce"),
    ):
        torch.testing.assert_close(actual, legacy[key], rtol=0.0, atol=0.0)
    assert schema_count == legacy["schema_token_count"]
    assert decision_count == legacy["decision_token_count"]
    assert termination_count == legacy["termination_token_count"]
    assert token_ce_inputs == [
        (torch.Size([len(target_positions), 262144]), torch.float32)
    ]

    weighted_generation_row_ce.mean().backward()
    legacy["weighted_generation_row_ce"].mean().backward()
    torch.testing.assert_close(
        selected_logits.grad,
        legacy_logits.grad,
        rtol=0.0,
        atol=0.0,
    )


def test_generation_token_masks_use_exact_generation_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = '{"boundaries":[1]}'
    content_start = 3
    rendered = "abc" + content + "zz"
    content_end = content_start + len(content)
    array_start = content.index("[")
    digit_start = content.index("1")
    close_start = content.index("]")
    offsets = [
        (0, 0),
        (0, 1),
        (1, 3),
        (content_start, content_start + array_start),
        (content_start + array_start, content_start + array_start + 1),
        (content_start + digit_start, content_start + digit_start + 1),
        (content_start + close_start, content_end),
        (content_end, len(rendered)),
    ]
    input_ids = list(range(len(offsets)))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": content},
    ]
    monkeypatch.setattr(
        experimental_train,
        "_rendered_message_content_span",
        lambda tokenizer, active_messages, message_index: (
            rendered,
            content_start,
            content_end,
        ),
    )
    monkeypatch.setattr(
        experimental_train,
        "_tokenizer_ids_and_offsets",
        lambda tokenizer, active_rendered: (input_ids, offsets),
    )
    monkeypatch.setattr(
        experimental_train,
        "_tokenize_chat_generation_prompt",
        lambda tokenizer, active_messages: input_ids[:3],
    )

    masks = experimental_train._scene_state_generation_token_masks(
        object(),
        messages,
        1,
        input_ids,
    )

    assert masks["scene_state_generation_target_mask"] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert masks["scene_state_generation_content_mask"] == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert masks["scene_state_generation_schema_mask"] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert masks["scene_state_generation_decision_mask"] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert masks["scene_state_generation_termination_mask"] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    monkeypatch.setattr(
        experimental_train,
        "_tokenize_chat_generation_prompt",
        lambda tokenizer, active_messages: [99],
    )
    with pytest.raises(ValueError, match="exact system-only generation prompt"):
        experimental_train._scene_state_generation_token_masks(
            object(),
            messages,
            1,
            input_ids,
        )


def test_generation_labels_exclude_supplied_assistant_role_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": '{"boundaries":[]}'},
    ]
    tokenizations = {
        1: [10, 11, 12],
        2: [10, 11, 12, 13, 14, 15, 16],
    }
    monkeypatch.setattr(
        experimental_train,
        "_tokenize_chat_messages",
        lambda tokenizer, active_messages: tokenizations[len(active_messages)],
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_boundary_semantic_token_mask",
        lambda tokenizer, active_messages, message_index, expected_input_ids: [
            False,
            False,
            False,
            False,
            False,
            True,
            False,
        ],
    )
    monkeypatch.setattr(
        experimental_train,
        "_scene_boundary_payload_metadata",
        lambda content: ((0, len(content)), 0),
    )
    generation_masks = {
        "scene_state_generation_target_mask": [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ],
        "scene_state_generation_content_mask": [
            False,
            False,
            False,
            False,
            True,
            True,
            False,
        ],
        "scene_state_generation_schema_mask": [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ],
        "scene_state_generation_decision_mask": [
            False,
            False,
            False,
            False,
            False,
            True,
            False,
        ],
        "scene_state_generation_termination_mask": [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    }
    monkeypatch.setattr(
        experimental_train,
        "_scene_state_generation_token_masks",
        lambda tokenizer, active_messages, message_index, expected_input_ids: (
            generation_masks
        ),
    )

    features = experimental_train.tokenize_messages_for_sft(
        object(),
        messages,
        32,
        assistant_loss_mode="final_assistant_only",
        require_scene_state_semantic_mask=True,
        require_scene_state_generation_masks=True,
    )

    assert features["labels"] == [-100, -100, -100, -100, 14, 15, 16]
    assert features["scene_state_generation_target_mask"] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_generation_objective_values_gradients_and_zero_detach() -> None:
    generation_ce = torch.tensor([1.0], requires_grad=True)
    first_error = torch.tensor([0.3], requires_grad=True)
    correct_pair_logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    donor_pair_logits = torch.tensor([[0.0, 2.0]], requires_grad=True)
    correct_margin = torch.tensor([0.1], requires_grad=True)
    donor_margin = torch.tensor([-0.2], requires_grad=True)
    zero_margin = torch.tensor([0.0], requires_grad=True)
    correct = {
        "weighted_generation_row_ce": generation_ce,
        "first_error_row_loss": first_error,
        "pair_logits": correct_pair_logits,
        "decision_margin_row": correct_margin,
    }
    donor = {
        "pair_logits": donor_pair_logits,
        "decision_margin_row": donor_margin,
    }
    zero = {"decision_margin_row": zero_margin}

    objective = experimental_train.DeltaMemTrainer._scene_state_generation_objective(
        correct,
        donor,
        zero,
    )
    expected_pair = F.cross_entropy(correct_pair_logits, torch.tensor([0]))
    expected_pair = expected_pair + F.cross_entropy(
        donor_pair_logits,
        torch.tensor([1]),
    )
    assert objective["total_loss"].item() == pytest.approx(
        1.0 + 0.3 + expected_pair.item() + 0.1
    )

    objective["total_loss"].backward()

    assert generation_ce.grad.item() > 0.0
    assert first_error.grad.item() > 0.0
    assert correct_pair_logits.grad[0, 0].item() < 0.0
    assert correct_pair_logits.grad[0, 1].item() > 0.0
    assert donor_pair_logits.grad[0, 0].item() > 0.0
    assert donor_pair_logits.grad[0, 1].item() < 0.0
    assert correct_margin.grad.item() < 0.0
    assert zero_margin.grad is None


@pytest.mark.parametrize(
    "distributed_name",
    ["DEEPSPEED", "MEGATRON_LM"],
)
def test_generation_sequential_rejects_unsupported_distributed_modes(
    distributed_name: str,
) -> None:
    trainer = _trainer()
    trainer.accelerator.distributed_type = SimpleNamespace(name=distributed_name)

    with pytest.raises(ValueError, match=f"distributed mode {distributed_name}"):
        trainer._validate_scene_state_generation_sequential_runtime()


def test_generation_sequential_rejects_sagemaker_model_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    monkeypatch.setattr(
        experimental_train,
        "is_sagemaker_mp_enabled",
        lambda: True,
    )

    with pytest.raises(ValueError, match="SageMaker model parallelism"):
        trainer._validate_scene_state_generation_sequential_runtime()


class _SceneGenerationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.2))
        self.active_write_token: int | None = None
        self.read_calls: list[tuple[bool, int | None]] = []

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        del attention_mask, labels, kwargs
        self.read_calls.append((torch.is_grad_enabled(), self.active_write_token))
        coefficient = {10: 1.0, 20: -1.0, None: 0.0}[
            self.active_write_token
        ]
        position_scale = torch.arange(
            1,
            input_ids.size(1) + 1,
            device=input_ids.device,
            dtype=self.parameter.dtype,
        )
        logits = input_ids.new_zeros(
            (input_ids.size(0), input_ids.size(1), 4),
            dtype=self.parameter.dtype,
        )
        logits[..., 0] = self.parameter * coefficient * position_scale
        logits[..., 1] = -self.parameter * coefficient * position_scale
        return {"loss": logits.sum() * 0.0, "logits": logits}


def _bind_state_controls(
    trainer: experimental_train.DeltaMemTrainer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reset_online_state(model: _SceneGenerationModel) -> None:
        model.active_write_token = None

    def prime_episode_state(model: _SceneGenerationModel, **kwargs) -> None:
        model.active_write_token = int(kwargs["write_input_ids"][0, 0].item())

    trainer._reset_online_state = reset_online_state
    trainer._prime_episode_state = prime_episode_state
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


def _branch_kwargs() -> dict[str, torch.Tensor | None]:
    return {
        "write_input_ids": torch.tensor([[10]]),
        "write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "write_message_ids": None,
        "write_sentence_ids": None,
        "donor_write_input_ids": torch.tensor([[20]]),
        "donor_write_attention_mask": torch.ones(1, 1, dtype=torch.long),
        "donor_write_message_ids": None,
        "donor_write_sentence_ids": None,
        **_generation_masks(),
        "donor_target_token_ids": torch.tensor([1]),
        "target_stratum_codes": torch.tensor(
            [
                experimental_train._SCENE_STATE_IDENTITY_TARGET_STRATUM_CODES[
                    "same_cardinality_value"
                ]
            ]
        ),
    }


def test_generation_sequential_replay_matches_joint_and_zero_is_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_trainer = _trainer()
    joint_model = _SceneGenerationModel()
    _bind_state_controls(joint_trainer, monkeypatch)
    joint_loss, _, joint_stats = joint_trainer._compute_scene_state_generation_ce(
        joint_model,
        _model_inputs(),
        loss_kwargs={},
        **_branch_kwargs(),
    )
    joint_loss.backward()
    expected_gradient = joint_model.parameter.grad.detach().clone()
    assert joint_model.read_calls == [(True, 10), (True, 20), (False, None)]

    replay_trainer = _trainer()
    replay_model = _SceneGenerationModel()
    _bind_state_controls(replay_trainer, monkeypatch)
    replay_loss, replay_stats = (
        replay_trainer._scene_state_generation_sequential_backward(
            replay_model,
            _model_inputs(),
            loss_kwargs={},
            gradient_scale=1.0,
            **_branch_kwargs(),
        )
    )

    assert replay_model.read_calls == [
        (False, 20),
        (False, None),
        (True, 10),
        (True, 20),
    ]
    assert replay_loss.item() == pytest.approx(joint_loss.item(), abs=1e-6)
    assert replay_model.parameter.grad.item() == pytest.approx(
        expected_gradient.item(),
        abs=1e-6,
    )
    for key in (
        "scene_generation_weighted_ce",
        "scene_generation_first_error_loss",
        "scene_generation_pair_correct_ce",
        "scene_generation_pair_donor_ce",
        "scene_generation_zero_margin_loss",
        "scene_generation_gold_top1_accuracy",
        "scene_generation_correct_decision_margin",
        "scene_generation_donor_decision_margin",
        "scene_generation_zero_decision_margin",
    ):
        assert replay_stats[key] == pytest.approx(joint_stats[key], abs=1e-6)


def test_generation_sequential_releases_graphful_branches_before_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint_trainer = _trainer()
    joint_model = _SceneGenerationModel()
    _bind_state_controls(joint_trainer, monkeypatch)
    joint_loss, _, _ = joint_trainer._compute_scene_state_generation_ce(
        joint_model,
        _model_inputs(),
        loss_kwargs={},
        **_branch_kwargs(),
    )
    joint_loss.backward()
    expected_gradient = joint_model.parameter.grad.detach().clone()

    replay_trainer = _trainer()
    replay_model = _SceneGenerationModel()
    _bind_state_controls(replay_trainer, monkeypatch)

    class WeakRefDict(dict):
        pass

    graphful_branch_refs: list[
        tuple[
            weakref.ReferenceType[WeakRefDict],
            weakref.ReferenceType[WeakRefDict],
            weakref.ReferenceType[torch.Tensor],
            weakref.ReferenceType[torch.Tensor],
        ]
    ] = []
    original_branch = replay_trainer._scene_state_generation_branch

    def tracked_branch(*args, **kwargs):
        outputs, metrics = original_branch(*args, **kwargs)
        if torch.is_grad_enabled():
            outputs = WeakRefDict(outputs)
            metrics = WeakRefDict(metrics)
            graphful_branch_refs.append(
                (
                    weakref.ref(outputs),
                    weakref.ref(metrics),
                    weakref.ref(outputs["loss"]),
                    weakref.ref(metrics["schema_row_ce"]),
                )
            )
        return outputs, metrics

    replay_trainer._scene_state_generation_branch = tracked_branch

    class LifecycleCheckingAccelerator:
        distributed_type = SimpleNamespace(name="NO")

        def __init__(self) -> None:
            self.backward_calls = 0

        def backward(self, loss: torch.Tensor, **kwargs) -> None:
            assert kwargs == {}
            branch_refs = graphful_branch_refs[self.backward_calls]
            assert all(branch_ref() is None for branch_ref in branch_refs)
            self.backward_calls += 1
            loss.backward()

    accelerator = LifecycleCheckingAccelerator()
    replay_trainer.accelerator = accelerator
    gradient_scale = 0.375
    replay_trainer._scene_state_generation_sequential_backward(
        replay_model,
        _model_inputs(),
        loss_kwargs={},
        gradient_scale=gradient_scale,
        **_branch_kwargs(),
    )

    assert accelerator.backward_calls == 2
    assert replay_model.parameter.grad.item() == pytest.approx(
        (expected_gradient * gradient_scale).item(),
        abs=1e-6,
    )


def _neutral_generation_row(source_token: int, write_token: int) -> dict[str, object]:
    input_ids = [7, 8, 9, source_token, 0, 0]
    labels = [-100, -100, 9, source_token, 0, 0]
    write_ids = [write_token, write_token]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "write_input_ids": write_ids,
        "write_attention_mask": [1, 1],
        "write_message_ids": [0, 0],
        "write_sentence_ids": [-1, -1],
        "teacher_input_ids": input_ids,
        "teacher_attention_mask": [1] * len(input_ids),
        "teacher_labels": labels,
        "state_only_write_input_ids": write_ids,
        "state_only_write_attention_mask": [1, 1],
        "state_only_write_message_ids": [0, 0],
        "state_only_write_sentence_ids": [-1, -1],
        "state_only_input_ids": input_ids,
        "state_only_attention_mask": [1] * len(input_ids),
        "state_only_labels": labels,
        "episode_target_message_index": 2,
        "write_message_count": 2,
        "visible_message_count": 1,
        "scene_state_semantic_mask": [False, False, False, True, True, False],
        "scene_state_boundary_count": 1,
        "scene_state_generation_target_mask": [
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        "scene_state_generation_content_mask": [
            False,
            False,
            True,
            True,
            True,
            False,
        ],
        "scene_state_generation_schema_mask": [
            False,
            False,
            True,
            False,
            False,
            False,
        ],
        "scene_state_generation_decision_mask": [
            False,
            False,
            False,
            True,
            True,
            False,
        ],
        "scene_state_generation_termination_mask": [
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    }


def _v7_entries(
    rows: list[dict[str, object]],
    *,
    raw_hashes: list[str] | None = None,
    label_hashes: list[str] | None = None,
) -> list[dict[str, object]]:
    entries = []
    if raw_hashes is None:
        raw_hashes = [
            hashlib.sha256(f"row-{index}".encode()).hexdigest()
            for index in range(2)
        ]
    base_record_hashes = [
        hashlib.sha256(f"base-{index}".encode()).hexdigest()
        for index in range(2)
    ]
    strict_score_hashes = [
        hashlib.sha256(f"score-{index}".encode()).hexdigest()
        for index in range(2)
    ]
    strict_strata = ["invalid_schema", "wrong_boundaries"]
    if label_hashes is None:
        label_hashes = [
            str(
                experimental_train._scene_state_label_identity(
                    row,
                    row_name=f"label:{index}",
                )["label_sha256"]
            )
            for index, row in enumerate(rows)
        ]
    for source_index, donor_index in ((0, 1), (1, 0)):
        source = rows[source_index]
        donor = rows[donor_index]
        source_write = experimental_train._content_contrast_write_payload(source)
        donor_write = experimental_train._content_contrast_write_payload(donor)
        _, metadata = (
            experimental_train._select_scene_state_identity_target_with_metadata(
                source,
                donor,
            )
        )
        source_generation_start = source[
            "scene_state_generation_target_mask"
        ].index(True)
        donor_generation_start = donor[
            "scene_state_generation_target_mask"
        ].index(True)
        entry = {
            "train_row_ordinal": source_index,
            "donor_train_row_ordinal": donor_index,
            "official_source_index": 100 + source_index,
            "donor_official_source_index": 100 + donor_index,
            "source_row_sha256": raw_hashes[source_index],
            "donor_row_sha256": raw_hashes[donor_index],
            "source_label_sha256": label_hashes[source_index],
            "donor_label_sha256": label_hashes[donor_index],
            "source_base_record_sha256": base_record_hashes[source_index],
            "donor_base_record_sha256": base_record_hashes[donor_index],
            "source_strict_failure_stratum": strict_strata[source_index],
            "donor_strict_failure_stratum": strict_strata[donor_index],
            "source_strict_score_sha256": strict_score_hashes[source_index],
            "donor_strict_score_sha256": strict_score_hashes[donor_index],
            "source_boundary_count": 1,
            "donor_boundary_count": 1,
            "target_stratum": "same_cardinality_value",
            "source_generation_prefix_sha256": (
                experimental_train._canonical_json_sha256(
                    source["input_ids"][:source_generation_start]
                )
            ),
            "donor_generation_prefix_sha256": (
                experimental_train._canonical_json_sha256(
                    donor["input_ids"][:donor_generation_start]
                )
            ),
            "source_write_sha256": experimental_train._canonical_json_sha256(
                source_write["input_ids"]
            ),
            "donor_write_sha256": experimental_train._canonical_json_sha256(
                donor_write["input_ids"]
            ),
            "source_write_token_count": len(source_write["input_ids"]),
            "donor_write_token_count": len(donor_write["input_ids"]),
            "write_token_count_delta": 0,
            "first_differing_semantic_ordinal": metadata[
                "first_differing_semantic_ordinal"
            ],
            "selected_target_positions": metadata["target_label_positions"],
            "selected_target_predictor_positions": metadata[
                "target_predictor_positions"
            ],
            "selected_target_token_ids": metadata["target_token_ids"],
            "donor_target_token_ids": metadata["donor_token_ids"],
            "causal_prefix_sha256": metadata["causal_prefix_sha256"],
        }
        entry["entry_sha256"] = experimental_train._canonical_json_sha256(entry)
        entries.append(entry)
    return entries


class _Tokenizer:
    pad_token_id = 0


def test_pairing_binding_accepts_generated_v7_entry_schema(tmp_path) -> None:
    rows = [_neutral_generation_row(1, 10), _neutral_generation_row(2, 20)]
    raw_rows = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"row-{index}"},
                {
                    "role": "assistant",
                    "content": json.dumps({"boundaries": [index + 1]}),
                },
            ]
        }
        for index in range(2)
    ]
    raw_lines = [
        json.dumps(row, sort_keys=True) for row in raw_rows
    ]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("\n".join(raw_lines) + "\n")
    raw_hashes = [
        hashlib.sha256(line.encode("utf-8")).hexdigest() for line in raw_lines
    ]
    label_hashes = [
        experimental_train._canonical_json_sha256([index + 1])
        for index in range(2)
    ]
    entries = _v7_entries(
        rows,
        raw_hashes=raw_hashes,
        label_hashes=label_hashes,
    )
    quotas = {
        "presence": 0,
        "same_cardinality_value": 2,
        "cross_cardinality_value": 0,
    }
    pair_manifest = {
        "schema": "rwkv_ms_scene_memory_v7_pairing.v1",
        "dataset": {
            "path": str(train_path.resolve()),
            "sha256": experimental_train._sha256_file(train_path),
            "rows": len(raw_lines),
            "ordered_row_sha256": experimental_train._canonical_json_sha256(
                raw_hashes
            ),
        },
        "quotas": quotas,
        "optimization": {"global_minimum_after_exact_quotas": True},
        "directed_pairs": entries,
        "entries_sha256": experimental_train._canonical_json_sha256(entries),
    }
    pair_manifest["manifest_sha256"] = (
        experimental_train._canonical_json_sha256(pair_manifest)
    )
    pair_path = tmp_path / "pairs.json"
    pair_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True))
    source_manifest = {
        "schema": "rwkv_ms_scene_memory_v7_source.v1",
        "contract": {
            "episode_contract": {
                "episode_recent_messages": 0,
                "write_phase": "system + user",
                "read_supervision": "system + assistant",
            }
        },
        "partitions": {
            "train": {
                "data": {
                    "path": str(train_path.resolve()),
                    "sha256": experimental_train._sha256_file(train_path),
                },
                "rows": len(raw_lines),
                "source_split": "train",
            }
        },
        "v7_pairing": {
            "schema": "rwkv_ms_scene_memory_v7_pairing_binding.v1",
            "dataset_sha256": experimental_train._sha256_file(train_path),
            "directed_entry_count": len(entries),
            "entries_sha256": experimental_train._canonical_json_sha256(entries),
            "quotas": quotas,
            "pair_manifest": {
                "path": str(pair_path.resolve()),
                "sha256": experimental_train._sha256_file(pair_path),
                "manifest_sha256": pair_manifest["manifest_sha256"],
            },
        },
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True))

    binding = experimental_train._scene_state_generation_pairing_binding(
        Namespace(
            scene_state_source_manifest=source_path,
            expected_scene_state_source_manifest_sha256=(
                experimental_train._sha256_file(source_path)
            ),
            train_file=train_path,
        )
    )

    assert binding["entries"] == entries
    assert binding["quotas"] == quotas


def test_predeclared_pair_materialization_and_collator_integrity() -> None:
    rows = [_neutral_generation_row(1, 10), _neutral_generation_row(2, 20)]
    entries = _v7_entries(rows)
    paired, manifest = experimental_train.materialize_scene_state_generation_pairs(
        Dataset.from_list(rows),
        split_name="train",
        pairing_binding={
            "entries": entries,
            "quotas": {
                "presence": 0,
                "same_cardinality_value": 2,
                "cross_cardinality_value": 0,
            },
            "pair_path": "/tmp/pairs.json",
            "pair_file_sha256": "1" * 64,
            "pair_manifest_sha256": "2" * 64,
            "entries_sha256": "3" * 64,
        },
    )

    assert paired["scene_state_donor_index"] == [1, 0]
    assert paired["scene_state_identity_donor_target_token_id"] == [2, 1]
    assert manifest["target_stratum_row_counts"] == {
        "presence": 0,
        "same_cardinality_value": 2,
        "cross_cardinality_value": 0,
    }
    batch = experimental_train.EpisodeCausalLMCollator(_Tokenizer())(
        [paired[0], paired[1]]
    )
    assert batch["scene_state_identity_donor_target_token_id"].tolist() == [2, 1]
    assert torch.equal(
        batch["scene_state_generation_target_mask"],
        batch["labels"].ne(-100),
    )
    assert not bool(
        (
            batch["scene_state_identity_target_mask"]
            & ~batch["scene_state_generation_decision_mask"]
        ).any()
    )

    corrupted_entries = _v7_entries(rows)
    corrupted_entries[0]["source_write_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="tokenized pairing differs"):
        experimental_train.materialize_scene_state_generation_pairs(
            Dataset.from_list(rows),
            split_name="train",
            pairing_binding={
                "entries": corrupted_entries,
                "quotas": {
                    "presence": 0,
                    "same_cardinality_value": 2,
                    "cross_cardinality_value": 0,
                },
                "pair_path": "/tmp/pairs.json",
                "pair_file_sha256": "1" * 64,
                "pair_manifest_sha256": "2" * 64,
                "entries_sha256": "3" * 64,
            },
        )
