from __future__ import annotations

import pytest
import torch

import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.tests.test_scene_memory_v10_training import _pair_inputs
from deltamem.tests.test_scene_memory_v11_training import _RepairModel
from deltamem.tests.test_scene_memory_v9_training import (
    _PairModel,
    _bind_pair_model,
    _masks,
    _trainer,
)


def _v12_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SEMANTIC_MARGIN_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 7
    trainer.scene_state_generated_prefix_correction_weight = 0.0
    trainer.scene_state_generated_unlikelihood_max_wrong_tokens = 0
    trainer.scene_state_generated_rollout_extra_tokens = 0
    trainer.scene_state_generated_rollout_max_tokens = 16
    return trainer


def _failed_alignment(
    generated: list[int],
    gold: list[int],
) -> dict[str, torch.Tensor | int]:
    generation_start = 2
    labels = torch.tensor([[-100, -100, *gold]])
    decision_mask = torch.zeros_like(labels, dtype=torch.bool)
    termination_mask = torch.zeros_like(labels, dtype=torch.bool)
    decision_mask[0, generation_start + 1 : generation_start + len(gold) - 1] = True
    termination_mask[0, generation_start + len(gold) - 1] = True
    return experimental_train.DeltaMemTrainer._scene_state_v12_failed_decision_alignment(
        labels,
        decision_mask,
        termination_mask,
        generation_start=generation_start,
        generated_token_ids=torch.tensor(generated),
        gold_token_ids=torch.tensor(gold),
    )


@pytest.mark.parametrize(
    (
        "generated",
        "expected_kind_code",
        "expected_cursor",
        "expected_competitor",
    ),
    [
        ([3, 9, 5, 6], 0, 1, 9),
        ([3, 9, 4, 5, 6], 1, 1, 9),
        ([3, 5, 6], 2, 1, 5),
    ],
    ids=("substitution", "generated-insertion", "gold-deletion"),
)
def test_v12_failed_semantic_alignment_selects_actual_greedy_competitor(
    generated: list[int],
    expected_kind_code: int,
    expected_cursor: int,
    expected_competitor: int,
) -> None:
    alignment = _failed_alignment(generated, [3, 4, 5, 6])

    assert int(alignment["selected_position"]) == 3
    assert alignment["selected_decision_ordinal"] == 0
    assert alignment["generated_cursor"] == expected_cursor
    assert int(alignment["competitor_id"]) == expected_competitor
    assert alignment["alignment_kind_code"] == expected_kind_code
    assert not bool(alignment["selected_is_termination"])


class _DecodeByTokenSequence:
    def __init__(self, values: dict[tuple[int, ...], str]) -> None:
        self.values = values

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.values[tuple(token_ids)]


class _PieceTokenizer:
    def __init__(self, pieces: dict[int, str]) -> None:
        self.pieces = pieces

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


def test_v12_semantic_alignment_skips_harmless_whitespace_before_boundary() -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {
            1: '{"boundaries":[',
            2: "10,",
            3: "20,",
            4: "30",
            5: "]}",
            8: "31",
            9: " ",
        }
    )
    gold = torch.tensor([1, 2, 3, 4, 5])
    # Token 9 changes only JSON whitespace; token 8 changes boundary 30 to 31.
    generated = torch.tensor([1, 2, 9, 3, 8, 5])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.tensor(
        [[False, False, False, False, True, True, False]]
    )
    termination_mask = torch.tensor(
        [[False, False, False, False, False, False, True]]
    )
    rollout = {
        "generated_token_ids": generated,
        "gold_token_ids": gold,
        "generation_start": 2,
    }

    parsed_exact, alignment = trainer._scene_state_v12_rollout_semantics(
        {"labels": labels},
        rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert parsed_exact is False
    assert alignment is not None
    assert int(alignment["selected_position"]) == 5
    assert alignment["selected_decision_ordinal"] == 1
    assert alignment["generated_cursor"] == 4
    assert int(alignment["competitor_id"]) == 8
    assert alignment["alignment_kind_code"] == 0


def test_v12_semantic_alignment_skips_formatting_before_malformed_json_error() -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {
            1: '{"boundaries":[',
            2: "10,",
            3: "20,",
            4: "30",
            5: "]}",
            8: "broken",
            9: " ",
        }
    )
    gold = torch.tensor([1, 2, 3, 4, 5])
    # Removing token 9 leaves malformed JSON. Replacing token 8 restores the
    # first trustworthy boundary payload, so token 8 is the semantic candidate.
    generated = torch.tensor([1, 2, 9, 3, 8, 5])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.tensor(
        [[False, False, False, False, True, True, False]]
    )
    termination_mask = torch.tensor(
        [[False, False, False, False, False, False, True]]
    )

    parsed_exact, alignment = trainer._scene_state_v12_rollout_semantics(
        {"labels": labels},
        {
            "generated_token_ids": generated,
            "gold_token_ids": gold,
            "generation_start": 2,
        },
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert parsed_exact is False
    assert alignment is not None
    assert int(alignment["selected_position"]) == 5
    assert alignment["selected_decision_ordinal"] == 1
    assert alignment["generated_cursor"] == 4
    assert int(alignment["competitor_id"]) == 8
    assert alignment["alignment_kind_code"] == 0


def test_v12_semantic_alignment_isolates_first_edit_in_multiply_malformed_json() -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {
            1: '{"boundaries":[',
            2: "10,",
            3: "20,",
            4: "30",
            5: "]}",
            6: "bad-footer",
            8: "broken-value",
            9: " ",
        }
    )
    gold = torch.tensor([1, 2, 3, 4, 5])
    # No one correction repairs this rollout. Classify each edit as the only
    # error in the valid gold sequence so whitespace cannot hide token 8.
    generated = torch.tensor([1, 2, 9, 3, 8, 6])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.tensor(
        [[False, False, False, False, True, True, False]]
    )
    termination_mask = torch.tensor(
        [[False, False, False, False, False, False, True]]
    )

    parsed_exact, alignment = trainer._scene_state_v12_rollout_semantics(
        {"labels": labels},
        {
            "generated_token_ids": generated,
            "gold_token_ids": gold,
            "generation_start": 2,
        },
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert parsed_exact is False
    assert alignment is not None
    assert int(alignment["selected_position"]) == 5
    assert alignment["selected_decision_ordinal"] == 1
    assert alignment["generated_cursor"] == 4
    assert int(alignment["competitor_id"]) == 8
    assert alignment["alignment_kind_code"] == 0


def test_v12_semantic_alignment_ignores_text_outside_extracted_json() -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {
            1: '{"boundaries":[1',
            2: "]",
            3: "}",
            8: '{"boundaries":[2',
            9: "Preface:\n",
            10: "\nFooter",
        }
    )
    gold = torch.tensor([1, 2, 3])
    # Prefix/footer are ignored by extract_json; token 8 is the actual boundary error.
    generated = torch.tensor([9, 8, 2, 3, 10])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.tensor([[False, False, True, False, False]])
    termination_mask = torch.tensor([[False, False, False, False, True]])
    rollout = {
        "generated_token_ids": generated,
        "gold_token_ids": gold,
        "generation_start": 2,
    }

    parsed_exact, alignment = trainer._scene_state_v12_rollout_semantics(
        {"labels": labels},
        rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert parsed_exact is False
    assert alignment is not None
    assert int(alignment["selected_position"]) == 2
    assert alignment["selected_decision_ordinal"] == 0
    assert alignment["generated_cursor"] == 1
    assert int(alignment["competitor_id"]) == 8
    assert alignment["alignment_kind_code"] == 0


def test_v12_consecutive_deletions_reach_later_eligible_gold_boundary() -> None:
    generation_start = 2
    gold = torch.tensor([1, 2, 3, 4])
    generated = torch.tensor([1, 4])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    # Gold index 1 is schema-only; gold index 2 is the eligible boundary.
    decision_mask = torch.tensor(
        [[False, False, False, False, True, False]]
    )
    termination_mask = torch.tensor(
        [[False, False, False, False, False, True]]
    )

    alignment = (
        experimental_train.DeltaMemTrainer._scene_state_v12_failed_decision_alignment(
            labels,
            decision_mask,
            termination_mask,
            generation_start=generation_start,
            generated_token_ids=generated,
            gold_token_ids=gold,
        )
    )

    assert int(alignment["selected_position"]) == generation_start + 2
    assert alignment["selected_decision_ordinal"] == 0
    assert alignment["generated_cursor"] == 1
    assert int(alignment["competitor_id"]) == 4
    assert alignment["alignment_kind_code"] == 2


def test_v12_premature_termination_without_competitor_fails_closed() -> None:
    gold = torch.tensor([1, 2, 3])
    generated = torch.tensor([1, 2])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.zeros_like(labels, dtype=torch.bool)
    termination_mask = torch.tensor([[False, False, False, False, True]])

    with pytest.raises(
        ValueError,
        match="ended without an actual competitor",
    ):
        experimental_train.DeltaMemTrainer._scene_state_v12_failed_decision_alignment(
            labels,
            decision_mask,
            termination_mask,
            generation_start=2,
            generated_token_ids=generated,
            gold_token_ids=gold,
        )


def test_v12_repeated_token_alignment_is_deterministic() -> None:
    results = [_failed_alignment([3, 9, 3, 6], [3, 3, 6]) for _ in range(2)]

    for alignment in results:
        assert int(alignment["selected_position"]) == 3
        assert alignment["selected_decision_ordinal"] == 0
        assert alignment["generated_cursor"] == 1
        assert int(alignment["competitor_id"]) == 9
        assert alignment["alignment_kind_code"] == 1


def test_v12_parsed_equal_rollout_retains_global_weakest_eligible_decision() -> None:
    trainer = _v12_trainer()
    generated = torch.tensor([11, 10])
    gold = torch.tensor([5, 6, 7, 8])
    trainer.scene_state_generation_tokenizer = _DecodeByTokenSequence(
        {
            tuple(generated.tolist()): (
                'prefix\n{ "boundaries": [3, 1, 3] }\nignored footer'
            ),
            tuple(gold.tolist()): '{"boundaries":[1,3]}',
            (6,): "1",
            (7,): "3",
        }
    )
    labels = torch.tensor([[-100, -100, 5, 6, 7, 8, 9]])
    decision_mask = torch.tensor(
        [[False, False, False, True, True, False, False]]
    )
    # The second termination token represents a rendered footer and is ineligible.
    termination_mask = torch.tensor(
        [[False, False, False, False, False, True, True]]
    )
    logits = torch.full((1, labels.size(1), 12), -10.0, requires_grad=True)
    with torch.no_grad():
        # Only the two digit-token margins, 2.0 and 0.7, are exact-retention
        # candidates. Structural termination and footer tokens are excluded.
        for predictor, target, gold_logit, competitor_logit in (
            (2, 6, 2.0, 0.0),
            (3, 7, 0.7, 0.0),
            (4, 8, 0.2, 0.0),
            (5, 9, -5.0, 5.0),
        ):
            logits[0, predictor, target] = gold_logit
            logits[0, predictor, 10] = competitor_logit
    rollout = {
        "generated_token_ids": generated,
        "gold_token_ids": gold,
        "generation_start": 2,
        "first_divergence": 0,
        "exact_through_termination": False,
    }

    loss, stats = trainer._scene_state_v12_semantic_margin_branch(
        {"labels": labels},
        logits,
        rollout=rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert loss.item() == pytest.approx(0.3)
    assert stats["scene_generation_v12_parsed_boundary_exact"] == 1.0
    assert stats["scene_generation_v12_raw_token_exact"] == 0.0
    assert stats["scene_generation_v12_selected_decision_ordinal"] == 1.0
    assert stats["scene_generation_v12_selected_label_position"] == 4.0
    assert stats["scene_generation_v12_selected_is_termination"] == 0.0
    assert stats["scene_generation_v12_relevant_decision_count"] == 2.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 3, 7].item() < 0.0
    assert logits.grad[0, 3, 10].item() > 0.0
    assert logits.grad[0, 4].abs().sum().item() == 0.0


def test_v12_exact_retention_excludes_generic_structure_and_termination() -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {
            5: '{"boundaries":',
            6: "[",
            7: "3",
            8: "]}",
            9: "<eos>",
            10: "1",
        }
    )
    gold = torch.tensor([5, 6, 7, 8])
    labels = torch.tensor([[-100, -100, 5, 6, 7, 8, 9]])
    decision_mask = torch.tensor(
        [[False, False, False, True, True, False, False]]
    )
    termination_mask = torch.tensor(
        [[False, False, False, False, False, False, True]]
    )
    logits = torch.full((1, labels.size(1), 12), -10.0, requires_grad=True)
    with torch.no_grad():
        # Generic '[' and EOS are deliberately weaker than the boundary value.
        for predictor, target, gold_logit, competitor_logit in (
            (2, 6, -2.0, 2.0),
            (3, 7, 0.4, 0.0),
            (5, 9, -3.0, 3.0),
        ):
            logits[0, predictor, target] = gold_logit
            logits[0, predictor, 10] = competitor_logit
    rollout = {
        "generated_token_ids": gold,
        "gold_token_ids": gold,
        "generation_start": 2,
        "first_divergence": int(gold.numel()),
        "exact_through_termination": True,
    }

    loss, stats = trainer._scene_state_v12_semantic_margin_branch(
        {"labels": labels},
        logits,
        rollout=rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
    )

    assert loss.item() == pytest.approx(0.6)
    assert stats["scene_generation_v12_selected_label_position"] == 4.0
    assert stats["scene_generation_v12_gold_token_id"] == 7.0
    assert stats["scene_generation_v12_selected_is_termination"] == 0.0
    assert stats["scene_generation_v12_relevant_decision_count"] == 1.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 3, 7].item() < 0.0
    assert logits.grad[0, 2].abs().sum().item() == 0.0
    assert logits.grad[0, 5].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        (" 7 ", 7),
        ('prefix {"boundaries":[1]} footer', {"boundaries": [1]}),
        ("malformed {", None),
    ),
)
def test_shared_scene_json_parser_preserves_legacy_extraction(text, expected) -> None:
    assert experimental_train.extract_json(text) == expected


def test_v12_failed_replay_uses_actual_prefix_and_trains_writer_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _v12_trainer()
    model = _RepairModel(vocabulary_size=12)
    trainer._reset_online_state = lambda active: setattr(active, "online_state", None)
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
    gold = torch.tensor([1, 4, 7])
    generated = torch.tensor([1, 9, 7])
    labels = torch.tensor([[-100, -100, *gold.tolist()]])
    decision_mask = torch.tensor([[False, False, False, True, False]])
    termination_mask = torch.tensor([[False, False, False, False, True]])
    rollout = {
        "prompt_input_ids": torch.tensor([[6, 7]]),
        "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "generated_token_ids": generated,
        "gold_token_ids": gold,
        "generation_start": 2,
        "first_divergence": 1,
        "exact_through_termination": False,
    }
    alignment = trainer._scene_state_v12_failed_decision_alignment(
        labels,
        decision_mask,
        termination_mask,
        generation_start=2,
        generated_token_ids=generated,
        gold_token_ids=gold,
    )

    replay_logits = trainer._scene_state_v12_failed_replay_logits(
        model,
        rollout,
        generated_cursor=int(alignment["generated_cursor"]),
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
    )
    loss, stats = trainer._scene_state_v12_semantic_margin_branch(
        {"labels": labels},
        None,
        rollout=rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
        parsed_boundary_exact=False,
        failed_alignment=alignment,
        failed_replay_logits=replay_logits,
    )

    assert model.last_input_ids is not None
    assert model.last_input_ids.tolist() == [[6, 7, 1]]
    assert stats["scene_generation_v12_top_competitor_id"] == 9.0
    assert stats["scene_generation_v12_competitor_is_actual_greedy"] == 1.0
    assert stats["scene_generation_v12_failed_replay_generated_cursor"] == 1.0
    loss.backward()
    assert model.writer.grad is not None and model.writer.grad.abs().item() > 0.0
    assert model.reader.grad is not None and model.reader.grad.abs().item() > 0.0


class _GreedyModel(torch.nn.Module):
    def __init__(self, generated_suffix: list[int]) -> None:
        super().__init__()
        self.generated_suffix = torch.tensor([generated_suffix])

    def generate(self, *, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.cat((input_ids, self.generated_suffix.to(input_ids.device)), dim=1)


@pytest.mark.parametrize(
    "objective_version",
    (
        experimental_train._SCENE_STATE_SEMANTIC_MARGIN_OBJECTIVE_VERSION,
        experimental_train._SCENE_STATE_HARD_FAILURE_OBJECTIVE_VERSION,
    ),
)
def test_semantic_rollout_uses_positive_internal_alignment_cap_when_legacy_cap_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    objective_version: str,
) -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_objective_version = objective_version
    model = _GreedyModel([1, 9, 3])
    trainer._reset_online_state = lambda active: None
    monkeypatch.setattr(
        experimental_train,
        "load_delta_mem_online_state",
        lambda active, state: None,
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
    observed_caps: list[int] = []
    original = (
        experimental_train.DeltaMemTrainer._scene_state_generated_wrong_positions
    )

    def tracked_alignment(generated, gold, *, max_wrong_tokens):
        observed_caps.append(max_wrong_tokens)
        return original(
            generated,
            gold,
            max_wrong_tokens=max_wrong_tokens,
        )

    monkeypatch.setattr(
        experimental_train.DeltaMemTrainer,
        "_scene_state_generated_wrong_positions",
        staticmethod(tracked_alignment),
    )
    model_inputs = {
        "input_ids": torch.tensor([[6, 7, 1, 2, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 1, 2, 3]]),
    }
    target_mask = torch.tensor([[False, False, True, True, True]])
    termination_mask = torch.tensor([[False, False, False, False, True]])

    rollout = trainer._scene_state_generated_greedy_rollout(
        model,
        model_inputs,
        online_state_snapshot={"mock.delta_state": torch.ones(1)},
        target_mask=target_mask,
        termination_mask=termination_mask,
    )

    assert trainer.scene_state_generated_unlikelihood_max_wrong_tokens == 0
    assert observed_caps == [6]
    assert rollout["first_divergence"] == 1
    assert rollout["wrong_positions"].tolist() == [1]
    assert model.training is True


def test_v12_resolves_each_rollout_before_its_teacher_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _v12_trainer()
    trainer.scene_state_generation_tokenizer = _PieceTokenizer(
        {0: "0", 1: "1", 2: "2"}
    )
    model = _PairModel()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs, donor_inputs = _pair_inputs()
    events: list[tuple[str, bool]] = []
    original_branch = trainer._scene_state_generation_branch

    def tracked_branch(*args, **kwargs):
        events.append(("teacher", torch.is_grad_enabled()))
        return original_branch(*args, **kwargs)

    def rollout_probe(active_model, model_inputs, **kwargs):
        del active_model, kwargs
        events.append(("rollout", torch.is_grad_enabled()))
        suffix = model_inputs["input_ids"][0, 2:].detach()
        return {
            "generated_token_ids": suffix,
            "gold_token_ids": suffix,
            "generation_start": 2,
            "first_divergence": int(suffix.numel()),
            "exact_through_termination": True,
        }

    trainer._scene_state_generation_branch = tracked_branch
    trainer._scene_state_generated_greedy_rollout = rollout_probe
    trainer._scene_state_v12_rollout_semantics = lambda *args, **kwargs: (True, None)
    trainer._scene_state_v12_record_pair_presentation = lambda *args, **kwargs: None

    trainer._scene_state_generation_symmetric_sequential_backward(
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
        gradient_scale=1.0 / 7.0,
        source_indices=torch.tensor([3]),
        donor_indices=torch.tensor([24]),
        source_row_sha256=torch.zeros(1, 32, dtype=torch.uint8),
        donor_row_sha256=torch.ones(1, 32, dtype=torch.uint8),
    )

    assert events == [
        ("teacher", False),  # The detached zero-state diagnostic.
        ("rollout", False),
        ("teacher", True),
        ("rollout", False),
        ("teacher", True),
    ]
