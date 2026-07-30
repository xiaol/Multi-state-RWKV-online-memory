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


def _v13_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_DENSE_SEMANTIC_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 7
    trainer.scene_state_generated_prefix_correction_weight = 0.0
    trainer.scene_state_generated_unlikelihood_max_wrong_tokens = 0
    trainer.scene_state_generated_rollout_extra_tokens = 0
    trainer.scene_state_generated_rollout_max_tokens = 16
    return trainer


def _row_hash(row_ordinal: int) -> torch.Tensor:
    return torch.full((1, 32), row_ordinal, dtype=torch.uint8)


def _run_v13_pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gradient_scale: float,
    parsed_exact: bool,
    events: list[tuple[str, bool]] | None = None,
) -> tuple[
    experimental_train.DeltaMemTrainer,
    _PairModel,
    torch.Tensor,
    dict[str, float],
]:
    trainer = _v13_trainer()
    model = _PairModel()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs, donor_inputs = _pair_inputs()
    original_branch = trainer._scene_state_generation_branch

    def tracked_branch(*args, **kwargs):
        if events is not None:
            events.append(("teacher", torch.is_grad_enabled()))
        return original_branch(*args, **kwargs)

    def rollout_probe(active_model, model_inputs, **kwargs):
        del active_model, kwargs
        if events is not None:
            events.append(("rollout", torch.is_grad_enabled()))
        suffix = model_inputs["input_ids"][0, 2:].detach()
        generated = suffix if parsed_exact else suffix.clone()
        if not parsed_exact:
            generated[1] = 2
        return {
            "prompt_input_ids": model_inputs["input_ids"][:, :2].detach(),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.long),
            "generated_token_ids": generated,
            "gold_token_ids": suffix,
            "generation_start": 2,
            "first_divergence": int(suffix.numel()) if parsed_exact else 1,
            "exact_through_termination": parsed_exact,
        }

    trainer._scene_state_generation_branch = tracked_branch
    trainer._scene_state_generated_greedy_rollout = rollout_probe
    trainer._scene_state_v13_rollout_semantics = lambda *args, **kwargs: (
        (True, None)
        if parsed_exact
        else (
            False,
            {
                "generated_cursor": 1,
                "selected_position": 3,
                "selected_decision_ordinal": 0,
                "competitor_id": torch.tensor(2),
                "alignment_kind_code": 0,
                "selected_is_termination": False,
            },
        )
    )
    trainer._scene_state_v13_record_pair_presentation = lambda *args, **kwargs: None
    if not parsed_exact:
        trainer._scene_state_v12_failed_replay_logits = (
            lambda active_model, *args, **kwargs: active_model(
                input_ids=torch.tensor([[7, 8, 2]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )["logits"]
        )

        def failed_repair(active_inputs, **kwargs):
            del active_inputs, kwargs
            repair_loss = model.parameters_by_side.sum()
            repair_value = float(repair_loss.detach().item())
            return repair_loss, {
                "scene_generation_v13_failed_semantic_repair_applied": 1.0,
                "scene_generation_v13_failed_semantic_repair_loss": repair_value,
                "scene_generation_v13_failed_semantic_ce": repair_value,
                "scene_generation_v13_failed_semantic_competitor_hinge": 0.0,
                "scene_generation_v13_failed_semantic_gold_vs_competitor_margin": 0.0,
                "scene_generation_v13_failed_semantic_decision_ordinal": 0.0,
                "scene_generation_v13_failed_semantic_label_position": 3.0,
                "scene_generation_v13_failed_semantic_gold_token_id": 0.0,
                "scene_generation_v13_failed_semantic_competitor_id": 2.0,
                "scene_generation_v13_failed_semantic_competitor_is_actual_greedy": 1.0,
                "scene_generation_v13_failed_semantic_replay_generated_cursor": 1.0,
                "scene_generation_v13_failed_semantic_alignment_kind_code": 0.0,
                "scene_generation_v13_failed_semantic_is_termination": 0.0,
            }

        trainer._scene_state_v13_failed_semantic_repair_branch = failed_repair

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
        source_indices=torch.tensor([3]),
        donor_indices=torch.tensor([24]),
        source_row_sha256=_row_hash(3),
        donor_row_sha256=_row_hash(24),
    )
    return trainer, model, loss, stats


def test_v13_decision_mask_selects_punctuation_digits_and_whole_fused_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = '{"boundaries":[12,3]}'
    content_start = 3
    rendered = "abc" + content + "zz"
    content_end = content_start + len(content)
    array_start = content.index("[")
    comma = content.index(",")
    close = content.index("]")
    offsets = [
        (0, 0),
        (0, 1),
        (1, content_start),
        (content_start, content_start + array_start - 1),
        # This token contains a schema colon plus the decision characters "[12".
        (content_start + array_start - 1, content_start + comma),
        (content_start + comma, content_start + comma + 1),
        (content_start + comma + 1, content_start + close + 1),
        (content_start + close + 1, content_end),
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
        lambda *args: (rendered, content_start, content_end),
    )
    monkeypatch.setattr(
        experimental_train,
        "_tokenizer_ids_and_offsets",
        lambda *args: (input_ids, offsets),
    )
    monkeypatch.setattr(
        experimental_train,
        "_tokenize_chat_generation_prompt",
        lambda *args: input_ids[:3],
    )

    masks = experimental_train._scene_state_generation_token_masks(
        object(),
        messages,
        1,
        input_ids,
    )

    assert masks["scene_state_generation_decision_mask"] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
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
        True,
        False,
    ]
    assert masks["scene_state_generation_termination_mask"][-1] is True


def test_v13_dense_loss_only_updates_selected_causal_predictors() -> None:
    logits = torch.zeros(1, 5, 4, requires_grad=True)
    labels = torch.tensor([[-100, 2, 1, 3, 0]])
    decision_mask = torch.tensor([[False, False, True, False, True]])

    metrics = experimental_train.DeltaMemTrainer._scene_state_v13_dense_decision_metrics(
        logits,
        labels,
        decision_mask,
    )
    loss = metrics["decision_ce_row"].mean() + metrics[
        "retention_hinge_row"
    ].mean()
    loss.backward()

    assert metrics["decision_token_count"] == 2
    assert logits.grad is not None
    assert logits.grad[0, 1].abs().sum().item() > 0.0
    assert logits.grad[0, 3].abs().sum().item() > 0.0
    assert logits.grad[0, [0, 2, 4]].abs().sum().item() == 0.0


def test_v13_dense_retention_hinge_detaches_top_competitor() -> None:
    logits = torch.full((1, 3, 4), -2.0, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1]])
    decision_mask = torch.tensor([[False, False, True]])
    with torch.no_grad():
        logits[0, 1, 0] = 0.25
        logits[0, 1, 1] = 0.0

    metrics = experimental_train.DeltaMemTrainer._scene_state_v13_dense_decision_metrics(
        logits,
        labels,
        decision_mask,
    )
    hinge = metrics["retention_hinge_row"].mean()
    assert hinge.item() == pytest.approx(1.25)
    hinge.backward()

    assert logits.grad is not None
    assert logits.grad[0, 1, 1].item() == pytest.approx(-1.0)
    assert logits.grad[0, 1, 0].item() == 0.0
    assert logits.grad[0, [0, 2]].abs().sum().item() == 0.0


def test_v13_exact_pair_resolves_rollouts_before_teacher_graph_and_skips_v12_weakest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        experimental_train.DeltaMemTrainer,
        "_scene_state_v12_weakest_decision_metrics",
        lambda *args, **kwargs: pytest.fail(
            "V13 exact dense training entered the V12 weakest-token objective"
        ),
    )

    trainer, _, _, stats = _run_v13_pair(
        monkeypatch,
        gradient_scale=1.0 / 7.0,
        parsed_exact=True,
        events=events,
    )

    assert events == [
        ("teacher", False),
        ("rollout", False),
        ("teacher", True),
        ("rollout", False),
        ("teacher", True),
    ]
    assert trainer.accelerator.backward_calls == 2
    assert stats["scene_generation_v13_pair_mean_failed_semantic_repair_applied_fraction"] == 0.0


def test_v13_failed_pair_has_teacher_and_repair_backward_per_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, model, _, stats = _run_v13_pair(
        monkeypatch,
        gradient_scale=1.0 / 7.0,
        parsed_exact=False,
    )

    assert trainer.accelerator.backward_calls == 4
    assert model.parameters_by_side.grad is not None
    assert stats["scene_generation_v13_pair_mean_failed_semantic_repair_applied_fraction"] == 1.0
    assert stats["scene_generation_v13_pair_mean_failed_semantic_repair_loss"] > 0.0


def test_v13_exact_pair_scales_returned_loss_and_gradients_by_one_seventh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_trainer, full_model, full_loss, _ = _run_v13_pair(
        monkeypatch,
        gradient_scale=1.0,
        parsed_exact=True,
    )
    scaled_trainer, scaled_model, scaled_loss, _ = _run_v13_pair(
        monkeypatch,
        gradient_scale=1.0 / 7.0,
        parsed_exact=True,
    )

    assert full_trainer.accelerator.backward_calls == 2
    assert scaled_trainer.accelerator.backward_calls == 2
    assert scaled_loss.item() == pytest.approx(full_loss.item() / 7.0)
    assert full_model.parameters_by_side.grad is not None
    assert scaled_model.parameters_by_side.grad is not None
    torch.testing.assert_close(
        scaled_model.parameters_by_side.grad,
        full_model.parameters_by_side.grad / 7.0,
    )


def test_v13_failed_repair_trains_writer_and_reader_without_termination_ce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _v13_trainer()
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
        include_first_termination=False,
    )
    replay_logits = trainer._scene_state_v12_failed_replay_logits(
        model,
        rollout,
        generated_cursor=int(alignment["generated_cursor"]),
        write_input_ids=torch.tensor([[10]]),
        write_attention_mask=torch.ones(1, 1, dtype=torch.long),
    )

    loss, stats = trainer._scene_state_v13_failed_semantic_repair_branch(
        {"labels": labels},
        rollout=rollout,
        decision_mask=decision_mask,
        termination_mask=termination_mask,
        failed_alignment=alignment,
        failed_replay_logits=replay_logits,
    )
    loss.backward()

    assert stats["scene_generation_v13_failed_semantic_is_termination"] == 0.0
    assert stats["scene_generation_v13_failed_semantic_label_position"] == 3.0
    assert model.writer.grad is not None and model.writer.grad.abs().item() > 0.0
    assert model.reader.grad is not None and model.reader.grad.abs().item() > 0.0


def _exact_audit_stats() -> dict[str, float]:
    stats: dict[str, float] = {}
    for role in ("source", "donor"):
        common = f"scene_generation_{role}_"
        v13 = f"scene_generation_v13_"
        stats.update(
            {
                f"{common}selected_top_hinge": 0.3,
                f"{common}zero_hinge": 0.4,
                f"{common}zero_minus_correct_selected_nll": 0.5,
                f"{common}selected_top1": 1.0,
                f"{v13}parsed_boundary_exact_{role}": 1.0,
                f"{v13}raw_token_exact_{role}": 1.0,
                f"{v13}first_divergence_{role}": 3.0,
                f"{v13}rollout_token_count_{role}": 3.0,
                f"{v13}dense_decision_token_count_{role}": 2.0,
                f"{v13}dense_decision_ce_{role}": 1.0,
                f"{v13}dense_decision_top1_retention_hinge_{role}": 0.2,
                f"{v13}dense_decision_gold_vs_top_competitor_margin_{role}": -0.1,
                f"{v13}dense_decision_top1_fraction_{role}": 0.5,
                f"{v13}dense_top1_margin_{role}": 1.0,
                f"{v13}failed_semantic_repair_applied_{role}": 0.0,
                f"{v13}failed_semantic_repair_loss_{role}": 0.0,
                f"{v13}failed_semantic_ce_{role}": 0.0,
                f"{v13}failed_semantic_competitor_hinge_{role}": 0.0,
                f"{v13}failed_semantic_gold_vs_competitor_margin_{role}": 0.0,
                f"{v13}failed_semantic_decision_ordinal_{role}": -1.0,
                f"{v13}failed_semantic_label_position_{role}": -1.0,
                f"{v13}failed_semantic_gold_token_id_{role}": -1.0,
                f"{v13}failed_semantic_competitor_id_{role}": -1.0,
                f"{v13}failed_semantic_competitor_is_actual_greedy_{role}": 0.0,
                f"{v13}failed_semantic_replay_generated_cursor_{role}": -1.0,
                f"{v13}failed_semantic_alignment_kind_code_{role}": -1.0,
                f"{v13}failed_semantic_is_termination_{role}": 0.0,
                f"{v13}dense_teacher_loss_{role}": 1.9,
                f"{v13}total_side_loss_{role}": 1.9,
            }
        )
    stats.update(
        {
            "scene_generation_v13_pair_mean_dense_decision_ce": 1.0,
            "scene_generation_v13_pair_mean_dense_decision_top1_retention_hinge": 0.2,
            "scene_generation_v13_pair_mean_selected_top_competitor_hinge": 0.3,
            "scene_generation_v13_pair_mean_selected_correct_vs_zero_hinge": 0.4,
            "scene_generation_v13_pair_mean_failed_semantic_repair_applied_fraction": 0.0,
            "scene_generation_v13_pair_mean_failed_semantic_ce": 0.0,
            "scene_generation_v13_pair_mean_failed_semantic_competitor_hinge": 0.0,
            "scene_generation_v13_pair_mean_failed_semantic_repair_loss": 0.0,
            "scene_generation_v13_pair_mean_dense_teacher_loss": 1.9,
            "scene_generation_v13_pair_mean_total_side_loss": 1.9,
            "scene_generation_v13_objective_total_loss": 1.9,
            "scene_generation_v13_recomputed_objective_total_loss": 1.9,
        }
    )
    return stats


def _v13_audit_trainer() -> experimental_train.DeltaMemTrainer:
    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_DENSE_SEMANTIC_OBJECTIVE_VERSION
    )
    trainer._scene_state_cycle_retention_metric_sums = {}
    trainer._scene_state_cycle_retention_metric_presentations = 0
    trainer._scene_state_v13_cycle_pairs = []
    trainer._scene_state_v13_completed_cycles = 0
    trainer._scene_state_v13_row_observations = []
    trainer._scene_state_v13_pair_observations = []
    manifest_pairs: list[dict[str, object]] = [{} for _ in range(32)]
    for source, donor in experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS:
        manifest_pairs[source] = {
            "source_index": source,
            "donor_index": donor,
            "source_row_sha256": bytes([source] * 32).hex(),
            "donor_row_sha256": bytes([donor] * 32).hex(),
        }
    trainer.scene_state_identity_pairing_manifest = {
        "splits": {"train": {"pairs": manifest_pairs}}
    }
    return trainer


def test_v13_four_cycle_order_and_row_audit_arithmetic() -> None:
    pairs = experimental_train._SCENE_STATE_V13_FOUR_CYCLE_PAIRS
    binding = {
        "canonical_value14_pairs": list(
            experimental_train._SCENE_STATE_V11_FIRST_CYCLE_PAIRS
        ),
        "pair_indices": pairs,
        "indices": tuple(source for source, _ in pairs),
        "checkpoint_steps": [7, 14, 21, 28],
    }
    experimental_train._validate_scene_state_v13_four_cycle_schedule(binding)
    wrong_binding = dict(binding)
    wrong_pairs = list(pairs)
    wrong_pairs[0], wrong_pairs[1] = wrong_pairs[1], wrong_pairs[0]
    wrong_binding["pair_indices"] = tuple(wrong_pairs)
    wrong_binding["indices"] = tuple(source for source, _ in wrong_pairs)
    with pytest.raises(ValueError, match="V13 four-cycle order differs"):
        experimental_train._validate_scene_state_v13_four_cycle_schedule(
            wrong_binding
        )

    trainer = _v13_audit_trainer()
    stats = _exact_audit_stats()
    final_cycle_stats: dict[str, float] | None = None
    for source, donor in pairs:
        trainer._scene_state_v13_record_pair_presentation(
            torch.tensor([source]),
            torch.tensor([donor]),
            _row_hash(source),
            _row_hash(donor),
            stats,
        )
        final_cycle_stats = (
            trainer._scene_state_cycle_retention_aggregate_memory_stats(stats)
        )

    assert final_cycle_stats is not None
    assert final_cycle_stats["scene_generation_v13_cycle_index"] == 4.0
    assert trainer._scene_state_v13_completed_cycles == 4
    payload = trainer._scene_state_v13_row_audit_payload()
    assert payload["checkpoint_optimizer_step"] == 4
    assert payload["completed_pair_presentations"] == 28
    assert payload["phases"] == [
        "cycle1_input",
        "cycle2_input",
        "cycle3_input",
        "cycle4_input",
    ]
    assert len(payload["pair_presentations"]) == 28
    assert len(payload["rows"]) == 14
    first_pair = payload["pair_presentations"][0]
    assert first_pair["pair_mean_dense_teacher_loss"] == pytest.approx(
        first_pair["pair_mean_dense_decision_ce"]
        + first_pair["pair_mean_dense_decision_top1_retention_hinge"]
        + first_pair["pair_mean_selected_top_competitor_hinge"]
        + first_pair["pair_mean_selected_correct_vs_zero_hinge"]
    )
    assert first_pair["reported_objective_total_loss"] == pytest.approx(
        first_pair["pair_mean_dense_teacher_loss"]
        + first_pair["pair_mean_failed_semantic_repair_loss"]
    )


class _DigitTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        del token_ids
        assert skip_special_tokens is True
        return "1"


def test_v12_objective_never_dispatches_to_v13_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    trainer.scene_state_generation_objective_version = (
        experimental_train._SCENE_STATE_SEMANTIC_MARGIN_OBJECTIVE_VERSION
    )
    trainer.current_gradient_accumulation_steps = 7
    trainer.scene_state_generated_prefix_correction_weight = 0.0
    trainer.scene_state_generated_unlikelihood_max_wrong_tokens = 0
    trainer.scene_state_generation_tokenizer = _DigitTokenizer()
    model = _PairModel()
    _bind_pair_model(trainer, monkeypatch)
    masks = _masks()
    source_inputs, donor_inputs = _pair_inputs()
    trainer._scene_state_generated_greedy_rollout = lambda active, inputs, **kwargs: {
        "generated_token_ids": inputs["input_ids"][0, 2:].detach(),
        "gold_token_ids": inputs["input_ids"][0, 2:].detach(),
        "generation_start": 2,
        "first_divergence": 4,
        "exact_through_termination": True,
    }
    trainer._scene_state_v12_rollout_semantics = lambda *args, **kwargs: (True, None)
    trainer._scene_state_v12_record_pair_presentation = lambda *args, **kwargs: None

    def forbidden(*args, **kwargs):
        del args, kwargs
        pytest.fail("V12 dispatched into a V13-only helper")

    for name in (
        "_scene_state_v13_rollout_semantics",
        "_scene_state_v13_dense_decision_metrics",
        "_scene_state_v13_teacher_loss",
        "_scene_state_v13_failed_semantic_repair_branch",
        "_scene_state_v13_record_pair_presentation",
    ):
        setattr(trainer, name, forbidden)

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
        source_row_sha256=_row_hash(3),
        donor_row_sha256=_row_hash(24),
    )
