from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as runner


class CharacterTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.tokenized_texts: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        self.chat_calls.append(
            {
                "messages": deepcopy(messages),
                "add_generation_prompt": add_generation_prompt,
            }
        )
        content = messages[0]["content"]
        suffix = "<assistant>" if add_generation_prompt else ""
        return f"<user>{content}</user>{suffix}"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_attention_mask: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        assert return_attention_mask is True
        assert return_offsets_mapping is True
        self.tokenized_texts.append(text)
        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids if token_id)


def _canonical(value: Any) -> str:
    return runner.source.canonical_json(value)


def _payload_hash(records: list[dict[str, Any]]) -> str:
    return runner.source.sha256_text(_canonical(records))


def _production_training_dataset_audit(
    schedule_mode: str | None = None,
) -> dict[str, Any]:
    conditions = list(runner.DEFAULT_TRAINING_CONDITIONS)
    tasks = list(runner.PRODUCTION_TASKS)
    rows_per_condition_task = {
        condition: {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK for task in tasks
        }
        for condition in conditions
    }
    audit: dict[str, Any] = {
        "schema": runner.TRAINING_DATASET_AUDIT_SCHEMA,
        "training_conditions": conditions,
        "tasks": tasks,
        "rows": runner.PRODUCTION_TRAINING_ROWS,
        "unique_row_ids": True,
        "row_id_policy": runner.TRAINING_ROW_ID_POLICY,
        "row_id_policy_passed": True,
        "sampling_policy": runner.TRAINING_SAMPLING_POLICY,
        "payload_digest_policy": runner.TRAINING_PAYLOAD_DIGEST_POLICY,
        "family_invariant_policy": runner.TRAINING_FAMILY_INVARIANT_POLICY,
        "condition_set_exact": True,
        "condition_task_strata_exact": True,
        "condition_task_strata_balanced": True,
        "rows_per_condition_task": rows_per_condition_task,
        "answer_tokens_per_condition_task": rows_per_condition_task,
        "rows_by_condition": {
            condition: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
            for condition in conditions
        },
        "rows_by_task": {
            task: runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(conditions)
            for task in tasks
        },
        "source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "complete_source_query_condition_families": (
            runner.PRODUCTION_ROWS_PER_CONDITION_TASK * len(tasks)
        ),
        "paired_condition_coverage": True,
        "family_invariants_passed": True,
        "family_invariant_failure_count": 0,
        "training_row_id_set_sha256": "a" * 64,
        "ordered_training_examples_sha256": "b" * 64,
        "passed": True,
    }
    if schedule_mode is None:
        return audit
    return runner.bind_production_training_contract(
        audit,
        epochs=runner.PRODUCTION_EPOCHS,
        global_batch_size=runner.distributed.REQUIRED_GLOBAL_BATCH_SIZE,
        requested_max_steps=(
            runner.DISTRIBUTED_PREFLIGHT_STEPS
            if schedule_mode == "preflight"
            else runner.PRODUCTION_UPDATES
        ),
        schedule_mode=schedule_mode,
    )


def _record(
    slot: int,
    value: str,
    *,
    physical_slot: int | None = None,
) -> dict[str, Any]:
    value_json = _canonical(value)
    key = f"natural-key-{slot}"
    return {
        "record_id": f"natural-record-{slot}",
        "slot_id": slot,
        "physical_index": slot if physical_slot is None else physical_slot,
        "key_text": key,
        "value": value,
        "value_json": value_json,
        "write_text": f"memory_key: {key}\nmemory_value: {value_json}",
    }


def _raw_episode(*, split: str = "train", identity: str = "episode-a") -> dict[str, Any]:
    correct = [_record(slot, f"correct-{slot}") for slot in range(4)]
    donor = [_record(slot, f"donor-{slot}") for slot in range(4)]
    swap_sources = (1, 2, 3, 0)
    value_swap = [
        _record(slot, f"correct-{swap_sources[slot]}") for slot in range(4)
    ]
    shuffled_order = (2, 0, 3, 1)
    shuffled = [
        _record(semantic_slot, f"correct-{semantic_slot}", physical_slot=physical_slot)
        for physical_slot, semantic_slot in enumerate(shuffled_order)
    ]
    state_variants = {
        "correct_state": {
            "records": correct,
            "record_payload_sha256": _payload_hash(correct),
        },
        "donor_state": {
            "records": donor,
            "record_payload_sha256": _payload_hash(donor),
        },
        "value_swap": {
            "records": value_swap,
            "record_payload_sha256": _payload_hash(value_swap),
            "source_slot_by_destination_slot": list(swap_sources),
        },
        "shuffled_slots": {
            "records": shuffled,
            "record_payload_sha256": _payload_hash(shuffled),
            "physical_order_to_semantic_slot": list(shuffled_order),
        },
        "no_state": {
            "records": [],
            "record_payload_sha256": _payload_hash([]),
        },
    }
    queries: list[dict[str, Any]] = []
    counterfactuals: dict[str, dict[str, Any]] = {}
    for slot in range(4):
        rewrite_value = f"donor-{slot}" if slot == 0 else f"rewrite-{slot}"
        replacement = _record(slot, rewrite_value)
        rewrite_records = deepcopy(correct)
        rewrite_records[slot] = replacement
        rewrite_hash = _payload_hash(rewrite_records)
        counterfactuals[str(slot)] = {
            "base_state": "correct_state",
            "target_slot_rewrite": {
                "replace_slot": slot,
                "replacement_record": replacement,
                "result_record_payload_sha256": rewrite_hash,
            },
        }
        expected = {
            "correct_state": f"correct-{slot}",
            "donor_state": f"donor-{slot}",
            "value_swap": f"correct-{swap_sources[slot]}",
            "target_slot_rewrite": rewrite_value,
            "shuffled_slots": f"correct-{slot}",
            "no_state": f"correct-{slot}",
            "pristine_frozen_base": f"correct-{slot}",
        }
        queries.append(
            {
                "query_id": f"{identity}:q{slot}",
                "query_family": "four_slot_target",
                "shared_correct_runtime_state_group": f"{identity}:correct_state",
                "target_slot": slot,
                "target_record_id": correct[slot]["record_id"],
                "address_text": correct[slot]["key_text"],
                "read_prompt": (
                    f"Recall {correct[slot]['key_text']} from external memory. "
                    "Return the canonical JSON value only."
                ),
                "answer_absent_from_read_prompt": True,
                "gold": f"correct-{slot}",
                "gold_json": _canonical(f"correct-{slot}"),
                "expected_by_state": expected,
                "record_payload_sha256_by_condition": {
                    **{
                        name: variant["record_payload_sha256"]
                        for name, variant in state_variants.items()
                    },
                    "target_slot_rewrite": rewrite_hash,
                },
                "binding_sha256_by_condition": {
                    name: runner.source.sha256_text(f"{identity}:{slot}:{name}")
                    for name in (
                        "correct_state",
                        "donor_state",
                        "value_swap",
                        "target_slot_rewrite",
                    )
                },
                "binding_absent_from_training": {
                    "correct_state": True,
                    "donor_state": True,
                    "value_swap": True,
                    "target_slot_rewrite": True,
                },
            }
        )
    return {
        "schema": runner.source.SCHEMA,
        "episode_id": identity,
        "split": split,
        "task": "attribution",
        "passage_components": [f"{identity}:component-{slot}" for slot in range(4)],
        "records": correct,
        "state_variants": state_variants,
        "query_counterfactual_records": counterfactuals,
        "queries": queries,
        "donor_source_component_ids": [
            f"{identity}:donor-component-{slot}" for slot in range(4)
        ],
        "value_swap_source_slot_by_destination_slot": list(swap_sources),
    }


@pytest.fixture
def tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer()


@pytest.fixture
def episode() -> runner.NaturalEpisode:
    return runner.adapt_episode(_raw_episode())


def test_adapt_episode_enforces_v2_four_slot_schema_and_sparse_rewrites() -> None:
    raw = _raw_episode()
    episode = runner.adapt_episode(raw)

    assert runner.source.SCHEMA == "novel_natural_causal_memory_gate.v2"
    assert len(episode.records_by_condition["correct_state"]) == 4
    assert len(episode.queries) == 4
    assert [query.target_slot for query in episode.queries] == list(range(4))
    assert episode.queries[0].rewrite_records[0].value_json == _canonical("donor-0")
    assert episode.records_by_condition["donor_state"][0].value_json == _canonical(
        "donor-0"
    )
    for query in episode.queries:
        changed_slots = [
            record.semantic_slot
            for correct, record in zip(
                episode.records_by_condition["correct_state"],
                query.rewrite_records,
                strict=True,
            )
            if correct.value_json != record.value_json
        ]
        assert changed_slots == [query.target_slot]

    missing_query = deepcopy(raw)
    missing_query["queries"].pop()
    with pytest.raises(ValueError, match="exactly four queries"):
        runner.adapt_episode(missing_query)

    wrong_schema = deepcopy(raw)
    wrong_schema["schema"] = "novel_natural_causal_memory_gate.v1"
    with pytest.raises(ValueError, match="schema must be"):
        runner.adapt_episode(wrong_schema)


def test_all_seven_conditions_materialize_expected_states_and_physical_routes(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    examples = {
        condition: runner.build_condition_examples([episode], tokenizer, condition)
        for condition in runner.CONDITIONS
    }

    assert tuple(examples) == runner.CONDITIONS
    assert all(len(rows) == 4 for rows in examples.values())
    assert len({row.memory_state_id for row in examples["correct_state"]}) == 1
    assert [row.write_value_jsons[0] for row in examples["donor_state"]] == [
        _canonical("donor-0")
    ] * 4
    assert examples["value_swap"][2].expected_value == _canonical("correct-3")

    shuffled_physical_slot_by_semantic_slot = {2: 0, 0: 1, 3: 2, 1: 3}
    assert [row.target_slot for row in examples["shuffled_slots"]] == [
        shuffled_physical_slot_by_semantic_slot[slot] for slot in range(4)
    ]
    for correct, rewrite in zip(
        examples["correct_state"],
        examples["target_slot_rewrite"],
        strict=True,
    ):
        changed = [
            index
            for index, (left, right) in enumerate(
                zip(correct.write_value_jsons, rewrite.write_value_jsons, strict=True)
            )
            if left != right
        ]
        assert changed == [correct.semantic_target_slot]
        assert rewrite.target_slot_rewrite_selection == {
            "semantic_target_slot": correct.semantic_target_slot
        }
    for condition in ("no_state", "pristine_frozen_base"):
        assert all(not row.write_records for row in examples[condition])
        assert all(not row.write_slots for row in examples[condition])
        assert all(row.target_slot is None for row in examples[condition])


def test_query_encoding_is_answer_free_before_labels_and_masks_are_disjoint(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    query = episode.queries[1]
    encoded = runner.encode_query_read(
        tokenizer,
        query,
        query.expected_json_by_condition["correct_state"],
    )
    prefix_ids = encoded["input_ids"][: encoded["query_prefix_length"]]
    answer_ids = encoded["expected_answer_token_ids"]
    prefix = tokenizer.decode(list(prefix_ids), skip_special_tokens=True)
    selected_address = tokenizer.decode(
        [
            token
            for token, selected in zip(
                encoded["input_ids"], encoded["query_mask"], strict=True
            )
            if selected
        ],
        skip_special_tokens=True,
    )

    assert query.expected_json_by_condition["correct_state"] not in prefix
    assert selected_address == query.address_text
    assert tokenizer.decode(list(answer_ids), skip_special_tokens=True) == query.gold_json
    assert not any(
        query_selected and answer_selected
        for query_selected, answer_selected in zip(
            encoded["query_mask"], encoded["answer_mask"], strict=True
        )
    )
    assert all(
        label == (token if answer_selected else -100)
        for token, answer_selected, label in zip(
            encoded["input_ids"],
            encoded["answer_mask"],
            encoded["labels"],
            strict=True,
        )
    )
    assert tokenizer.chat_calls[-1]["messages"] == [
        {"role": "user", "content": query.read_prompt}
    ]
    expected_prefix = f"<user>{query.read_prompt}</user><assistant>"
    assert tokenizer.tokenized_texts == [expected_prefix + query.gold_json]

    leaked = replace(query, read_prompt=f"Prompt leaks {query.gold_json}")
    with pytest.raises(ValueError, match="leaks its expected answer"):
        runner.encode_query_read(tokenizer, leaked, query.gold_json)


def test_query_encoding_rejects_a_token_crossing_the_answer_boundary(
    episode: runner.NaturalEpisode,
) -> None:
    class CrossingTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            self.tokenized_texts.append(text)
            return {
                "input_ids": [1],
                "attention_mask": [1],
                "offset_mapping": [(0, len(text))],
            }

    query = episode.queries[0]
    with pytest.raises(ValueError, match="crossing the prefix/answer boundary"):
        runner.encode_query_read(CrossingTokenizer(), query, query.gold_json)


def test_training_curriculum_defaults_to_balanced_paired_positive_states(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    default_rows = runner.build_training_examples([episode], tokenizer)
    assert len(default_rows) == 20
    assert len({row.row_id for row in default_rows}) == 20
    assert {row.condition for row in default_rows} == set(runner.POSITIVE_CONDITIONS)
    audit = runner.audit_training_dataset(default_rows)
    assert audit["passed"] is True
    assert audit["source_query_condition_families"] == 4
    assert audit["complete_source_query_condition_families"] == 4
    assert audit["rows_per_condition_task"] == {
        condition: {episode.task: 4} for condition in runner.POSITIVE_CONDITIONS
    }

    baseline = runner.build_training_examples(
        [episode],
        CharacterTokenizer(),
        runner.BASELINE_TRAINING_CONDITIONS,
    )
    assert len(baseline) == 4
    assert {row.condition for row in baseline} == {"correct_state"}

    with pytest.raises(ValueError, match="positive memory conditions"):
        runner._parse_training_conditions("correct_state,no_state")


def test_collator_preserves_write_slots_and_control_batches(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    correct = runner.build_condition_examples([episode], tokenizer, "correct_state")
    batch = runner.collate_examples(
        correct,
        pad_token_id=tokenizer.pad_token_id,
        device=torch.device("cpu"),
    )

    assert len(batch.write_records) == 4
    assert [record["slots"].tolist() for record in batch.write_records] == [
        [slot] * 4 for slot in range(4)
    ]
    assert batch.target_slots.tolist() == [0, 1, 2, 3]
    assert batch.query_mask.dtype == torch.bool
    assert batch.answer_mask.dtype == torch.bool
    assert not bool((batch.query_mask & batch.answer_mask).any().item())
    assert bool(batch.labels.ne(-100).any().item())

    for condition in ("no_state", "pristine_frozen_base"):
        controls = runner.build_condition_examples([episode], tokenizer, condition)
        control_batch = runner.collate_examples(
            controls,
            pad_token_id=tokenizer.pad_token_id,
            device=torch.device("cpu"),
        )
        assert control_batch.write_records == []
        assert control_batch.target_slots.tolist() == [-1, -1, -1, -1]


def test_shared_runtime_write_read_and_greedy_calls_disable_cache(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = runner.build_condition_examples([episode], tokenizer, "correct_state")[:2]
    batch = runner.collate_examples(
        examples,
        pad_token_id=tokenizer.pad_token_id,
        device=torch.device("cpu"),
    )
    memory = SimpleNamespace(
        last_write_routes=None,
        projected_kv_occupied=None,
        active_delta_heads=frozenset({"q", "o"}),
    )

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []
            self.current_slots: torch.Tensor | None = None

        def forward(self, **kwargs: Any) -> SimpleNamespace:
            self.calls.append(dict(kwargs))
            input_ids = kwargs["input_ids"]
            if self.current_slots is not None:
                routes = torch.full(
                    (input_ids.size(0), 1, 4),
                    -100.0,
                    dtype=torch.float32,
                )
                routes.scatter_(2, self.current_slots.view(-1, 1, 1), 100.0)
                memory.last_write_routes = routes
                memory.projected_kv_occupied = torch.ones(
                    input_ids.size(0), 4, dtype=torch.bool
                )
            return SimpleNamespace(
                logits=torch.zeros(
                    input_ids.size(0), input_ids.size(1), 256, dtype=torch.float32
                )
            )

    model = RecordingModel()
    shared = runner.runtime
    monkeypatch.setattr(
        shared, "iter_delta_mem_modules", lambda _model: iter((("memory", memory),))
    )
    monkeypatch.setattr(shared, "reset_delta_mem_states", lambda _model: None)
    monkeypatch.setattr(shared, "set_delta_mem_write_enabled", lambda *_args: None)
    monkeypatch.setattr(
        shared,
        "set_delta_mem_projected_kv_write_spans",
        lambda target, _keys, _values, slots: setattr(target, "current_slots", slots),
    )
    monkeypatch.setattr(
        shared, "set_delta_mem_projected_kv_read_query_mask", lambda *_args: None
    )
    monkeypatch.setattr(
        shared,
        "collect_delta_mem_projected_kv_read_logits",
        lambda _model: {
            "memory": torch.zeros(
                batch.read_input_ids.size(0), batch.read_input_ids.size(1), 4
            )
        },
    )
    monkeypatch.setattr(
        shared, "_temporarily_disable_delta_heads", lambda _model: nullcontext()
    )

    write_audit = runner._write_episode_batch(model, batch, dtype=torch.float32)
    assert write_audit["full_occupancy_count"] == len(examples)
    assert write_audit["forced_write_route_match_count"] == 4 * len(examples)
    assert len(model.calls) == 4
    assert all(call["use_cache"] is False for call in model.calls)

    model.calls.clear()
    model.current_slots = None
    runner._read_episode_batch(model, batch, dtype=torch.float32)
    assert len(model.calls) == 1
    assert model.calls[0]["use_cache"] is False
    assert torch.equal(
        model.calls[0]["logits_to_keep"],
        shared._answer_predictor_indices(batch.labels),
    )

    model.calls.clear()
    runner._greedy_answer_predictions(
        model,
        batch,
        pad_token_id=tokenizer.pad_token_id,
        dtype=torch.float32,
    )
    assert model.calls
    assert all(call["use_cache"] is False for call in model.calls)
    assert all(call["logits_to_keep"] == 1 for call in model.calls)


def test_state_identity_and_rewrite_output_change_audits(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    correct = runner.build_condition_examples([episode], tokenizer, "correct_state")
    rewrite = runner.build_condition_examples(
        [episode], tokenizer, "target_slot_rewrite"
    )
    identity_evaluation = {
        "state_digest_by_row": {row.row_id: "d" * 64 for row in correct},
        "route_predictions_by_row": {
            row.row_id: {"layer-0": row.target_slot} for row in correct
        },
        "route_by_layer": {"layer-0": {}},
    }
    identity = runner.audit_correct_state_identity(correct, identity_evaluation)
    assert identity["runtime_byte_identical_state_fraction"] == 1.0
    assert identity["family_layer_all_four_correct_fraction"] == 1.0

    changed_identity = deepcopy(identity_evaluation)
    changed_identity["state_digest_by_row"][correct[-1].row_id] = "e" * 64
    assert (
        runner.audit_correct_state_identity(correct, changed_identity)[
            "runtime_byte_identical_state_fraction"
        ]
        == 0.0
    )

    examples_by_condition = {
        condition: runner.build_condition_examples([episode], tokenizer, condition)
        for condition in runner.POSITIVE_CONDITIONS
    }
    condition_digest = {
        "correct_state": "a" * 64,
        "donor_state": "b" * 64,
        "value_swap": "c" * 64,
        "shuffled_slots": "d" * 64,
    }
    state_evaluations = {
        condition: {
            "state_digest_by_row": {
                row.row_id: (
                    condition_digest[condition]
                    if condition != "target_slot_rewrite"
                    else f"{index + 1:064x}"
                )
                for index, row in enumerate(rows)
            }
        }
        for condition, rows in examples_by_condition.items()
    }
    state_causality = runner.audit_runtime_state_causality(
        examples_by_condition, state_evaluations
    )
    assert state_causality["runtime_byte_identical_state_fraction"] == 1.0
    assert state_causality["write_payload_difference_fraction"] == 1.0
    assert state_causality["runtime_tensor_state_difference_fraction"] == 1.0
    assert (
        state_causality["counterfactual_pair_contract_passed_fraction"] == 1.0
    )

    query_dependent = deepcopy(state_evaluations)
    donor_rows = examples_by_condition["donor_state"]
    query_dependent["donor_state"]["state_digest_by_row"][
        donor_rows[-1].row_id
    ] = "e" * 64
    query_dependent_audit = runner.audit_runtime_state_causality(
        examples_by_condition, query_dependent
    )
    assert query_dependent_audit["runtime_byte_identical_state_fraction"] == 0.75

    missing_digest = deepcopy(state_evaluations)
    missing_digest["value_swap"]["state_digest_by_row"].pop(
        examples_by_condition["value_swap"][0].row_id
    )
    with pytest.raises(ValueError, match="digest rows differ"):
        runner.audit_runtime_state_causality(
            examples_by_condition, missing_digest
        )

    semantically_mispaired = dict(examples_by_condition)
    semantically_mispaired["donor_state"] = [
        replace(row, episode_id="wrong-episode") if index == 0 else row
        for index, row in enumerate(donor_rows)
    ]
    mispaired_audit = runner.audit_runtime_state_causality(
        semantically_mispaired, state_evaluations
    )
    assert (
        mispaired_audit["counterfactual_pair_contract_passed_fraction"]
        == 15 / 16
    )

    collided = deepcopy(state_evaluations)
    collided["donor_state"]["state_digest_by_row"] = {
        row.row_id: "a" * 64 for row in examples_by_condition["donor_state"]
    }
    collided_audit = runner.audit_runtime_state_causality(
        examples_by_condition, collided
    )
    assert collided_audit["runtime_byte_identical_state_fraction"] == 1.0
    assert collided_audit["runtime_tensor_state_difference_fraction"] == 0.75

    def predictions(rows: list[runner.NaturalMemoryExample]) -> dict[str, Any]:
        return {
            "greedy_answer_evaluated": True,
            "answer_predictions_by_row": {
                row.row_id: {
                    "teacher_forced_prediction_token_ids": list(
                        row.expected_answer_token_ids
                    ),
                    "teacher_forced_exact": True,
                    "greedy_generated_token_ids": list(row.expected_answer_token_ids),
                    "greedy_exact": True,
                }
                for row in rows
            },
        }

    correct_evaluation = predictions(correct)
    rewrite_evaluation = predictions(rewrite)
    audit = runner.audit_rewrite_output_change(
        correct,
        rewrite,
        correct_evaluation,
        rewrite_evaluation,
    )
    assert audit["pair_contract_passed_fraction"] == 1.0
    assert audit["expected_answers_differ_fraction"] == 1.0
    assert audit["teacher_forced_output_change_fraction"] == 1.0
    assert audit["greedy_output_change_fraction"] == 1.0

    unchanged_predictions = deepcopy(rewrite_evaluation)
    for row in correct:
        prediction = unchanged_predictions["answer_predictions_by_row"][row.row_id]
        prediction["teacher_forced_prediction_token_ids"] = list(
            row.expected_answer_token_ids
        )
        prediction["teacher_forced_exact"] = False
        prediction["greedy_generated_token_ids"] = list(row.expected_answer_token_ids)
        prediction["greedy_exact"] = False
    unchanged = runner.audit_rewrite_output_change(
        correct,
        rewrite,
        correct_evaluation,
        unchanged_predictions,
    )
    assert unchanged["teacher_forced_output_change_fraction"] == 0.0
    assert unchanged["greedy_output_change_fraction"] == 0.0


def test_no_state_and_pristine_outputs_are_audited_for_equivalence() -> None:
    rows = {
        f"row-{index}": {
            "teacher_forced_prediction_token_ids": [index, index + 1],
            "greedy_generated_token_ids": [index + 2],
        }
        for index in range(4)
    }
    no_state = {
        "greedy_answer_evaluated": True,
        "answer_predictions_by_row": rows,
    }
    pristine = {
        "greedy_answer_evaluated": True,
        "answer_predictions_by_row": deepcopy(rows),
    }
    audit = runner.audit_control_equivalence(no_state, pristine)
    assert audit["teacher_forced_output_equivalence_fraction"] == 1.0
    assert audit["greedy_output_equivalence_fraction"] == 1.0

    pristine["answer_predictions_by_row"]["row-3"][
        "teacher_forced_prediction_token_ids"
    ] = [999]
    changed = runner.audit_control_equivalence(no_state, pristine)
    assert changed["teacher_forced_output_equivalence_fraction"] == 0.75


def test_control_state_absence_evidence_records_both_runtime_phases() -> None:
    evidence = runner._control_state_absence_evidence(
        ["row-0", "row-1"],
        before_read_state_names=[],
        after_read_state_names=[],
    )
    assert set(evidence) == {"row-0", "row-1"}
    assert all(
        row["projected_kv_state_absent_before_read"]
        and row["projected_kv_state_absent_after_read"]
        for row in evidence.values()
    )

    stale = runner._control_state_absence_evidence(
        ["row-0"],
        before_read_state_names=["layer-0.__projected_kv_keys"],
        after_read_state_names=["layer-0.__projected_kv_keys"],
    )
    assert stale["row-0"]["projected_kv_state_absent_before_read"] is False
    assert stale["row-0"]["projected_kv_state_absent_after_read"] is False


def test_projected_kv_state_filter_ignores_inert_recurrent_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        delta_state=torch.zeros(1),
        rwkv_ms_positions=torch.zeros(1),
        rwkv_ms_previous_source=torch.zeros(1),
        projected_kv_keys=None,
        projected_kv_values=None,
        projected_kv_occupied=None,
        projected_kv_surprise=None,
    )
    monkeypatch.setattr(
        runner,
        "iter_delta_mem_modules",
        lambda _model: iter((("layer-0", module),)),
    )
    assert runner._projected_kv_state_names(torch.nn.Linear(1, 1)) == ()

    module.projected_kv_occupied = torch.zeros(1, 4, dtype=torch.bool)
    assert runner._projected_kv_state_names(torch.nn.Linear(1, 1)) == (
        "layer-0.projected_kv_occupied",
    )


def test_shared_write_evaluation_batches_never_split_a_family(
    episode: runner.NaturalEpisode,
    tokenizer: CharacterTokenizer,
) -> None:
    rows = runner.build_condition_examples([episode], tokenizer, "donor_state")
    batches = list(
        runner._evaluation_batches(rows, condition="donor_state", batch_size=6)
    )
    assert [len(batch) for batch in batches] == [4]
    assert {row.memory_state_id for row in batches[0]} == {rows[0].memory_state_id}
    with pytest.raises(ValueError, match="complete family"):
        list(
            runner._evaluation_batches(
                rows, condition="donor_state", batch_size=3
            )
        )


def _passing_evaluations() -> dict[str, dict[str, Any]]:
    evaluations: dict[str, dict[str, Any]] = {}
    for condition in runner.POSITIVE_CONDITIONS:
        evaluations[condition] = {
            "teacher_forced_structured_json_exact_accuracy": 1.0,
            "semantic_route_accuracy": 1.0,
            "full_occupancy_fraction": 1.0,
            "forced_write_route_accuracy": 1.0,
            "greedy_answer_evaluated": True,
            "greedy_structured_json_exact_accuracy": 1.0,
        }
    for condition in runner.CONTROL_CONDITIONS:
        rows = {
            f"row-{index}": {
                "projected_kv_state_names_before_read": [],
                "projected_kv_state_names_after_read": [],
                "projected_kv_state_absent_before_read": True,
                "projected_kv_state_absent_after_read": True,
            }
            for index in range(4)
        }
        evaluations[condition] = {
            "rows": 4,
            "full_occupancy_total": 0,
            "forced_write_route_total": 0,
            "semantic_route_total": 0,
            "route_absent_fraction": 1.0,
            "state_digest_by_row": {},
            "runtime_state_absence_rows": 4,
            "runtime_state_absence_fraction": 1.0,
            "runtime_state_absence_by_row": rows,
            "greedy_answer_evaluated": True,
            "delta_heads_disabled": condition == "pristine_frozen_base",
            "adapter_attached": condition != "pristine_frozen_base",
            "attached_delta_mem_module_count": (
                0 if condition == "pristine_frozen_base" else 42
            ),
            "trainable_parameter_count": (
                0 if condition == "pristine_frozen_base" else 1
            ),
            "pristine_base_adapter_excluded": condition == "pristine_frozen_base",
        }
    return evaluations


def test_gate_is_conjunctive_across_conditions_audits_and_artifact_checks() -> None:
    arguments = {
        "state_identity": {
            "runtime_byte_identical_state_fraction": 1.0,
            "family_layer_all_four_correct_fraction": 1.0,
        },
        "state_causality": {
            "runtime_byte_identical_state_fraction": 1.0,
            "write_payload_difference_fraction": 1.0,
            "counterfactual_pair_contract_passed_fraction": 1.0,
            "runtime_tensor_state_difference_fraction": 1.0,
        },
        "rewrite_audit": {
            "expected_answers_differ_fraction": 1.0,
            "pair_contract_passed_fraction": 1.0,
            "teacher_forced_output_change_fraction": 1.0,
            "teacher_forced_joint_exact_output_flip_fraction": 1.0,
            "greedy_answer_evaluated": True,
            "greedy_output_change_fraction": 1.0,
            "greedy_joint_exact_output_flip_fraction": 1.0,
        },
        "control_equivalence": {
            "teacher_forced_output_equivalence_fraction": 1.0,
            "greedy_answer_evaluated": True,
            "greedy_output_equivalence_fraction": 1.0,
        },
        "profile_eligibility": {"profile": "development", "passed": True},
        "trainable_audit": {
            "only_delta_mem_parameters_trainable": True,
            "passed": True,
        },
        "immutability_passed": True,
        "training": {
            "optimizer_skipped": False,
            "adapter_changed": True,
            "router_gradient_audit": {"all_modules_finite_nonzero": True},
            "training_dataset_audit": _production_training_dataset_audit(
                "complete"
            ),
        },
    }
    evaluations = _passing_evaluations()
    gate = runner.build_gate(evaluations, **arguments)
    assert gate["passed"] is True
    assert gate["failed_checks"] == []

    spoofed = deepcopy(arguments)
    spoofed_audit = spoofed["training"]["training_dataset_audit"]
    spoofed_audit["rows"] = 4
    spoofed_audit["passed"] = True
    spoofed_audit["production_contract_passed"] = True
    rejected = runner.build_gate(evaluations, **spoofed)
    assert rejected["passed"] is False
    assert "training.dataset_audit" in rejected["failed_checks"]
    assert "training.compositional_production_dataset" in rejected["failed_checks"]

    malformed = deepcopy(arguments)
    malformed["training"]["training_dataset_audit"] = {
        "training_conditions": [[]],
        "rows": "not-an-integer",
    }
    rejected = runner.build_gate(evaluations, **malformed)
    assert rejected["passed"] is False
    assert "training.dataset_audit" in rejected["failed_checks"]
    assert "training.compositional_production_dataset" in rejected["failed_checks"]

    failing = deepcopy(evaluations)
    failing["value_swap"]["semantic_route_accuracy"] = 0.0
    rejected = runner.build_gate(failing, **arguments)
    assert rejected["passed"] is False
    assert "value_swap.semantic_route_min" in rejected["failed_checks"]

    collided_state = deepcopy(arguments)
    collided_state["state_causality"][
        "runtime_tensor_state_difference_fraction"
    ] = 0.75
    rejected = runner.build_gate(evaluations, **collided_state)
    assert rejected["passed"] is False
    assert (
        "counterfactual_states.runtime_tensors_differ"
        in rejected["failed_checks"]
    )

    query_dependent_shared_state = deepcopy(arguments)
    query_dependent_shared_state["state_causality"][
        "runtime_byte_identical_state_fraction"
    ] = 0.75
    rejected = runner.build_gate(evaluations, **query_dependent_shared_state)
    assert rejected["passed"] is False
    assert "positive_states.shared_write_identity" in rejected["failed_checks"]

    stale_control = deepcopy(evaluations)
    stale_control["no_state"]["runtime_state_absence_fraction"] = 0.0
    rejected = runner.build_gate(stale_control, **arguments)
    assert rejected["passed"] is False
    assert "no_state.runtime_state_absent" in rejected["failed_checks"]

    teacher_forced_only = deepcopy(evaluations)
    teacher_forced_only["correct_state"]["greedy_answer_evaluated"] = False
    rejected = runner.build_gate(teacher_forced_only, **arguments)
    assert rejected["passed"] is False
    assert "formal.greedy_answer_evaluation" in rejected["failed_checks"]


def test_hf_mirror_is_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert runner.configure_hf_mirror() == "https://hf-mirror.com"
    assert runner.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    with pytest.raises(ValueError, match="HF_ENDPOINT must be"):
        runner.configure_hf_mirror()
    with pytest.raises(ValueError, match="HF_ENDPOINT must be"):
        runner.configure_hf_mirror("https://huggingface.co")


def test_pristine_evaluation_rejects_an_attached_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        runner,
        "iter_delta_mem_modules",
        lambda _model: iter((("memory", object()),)),
    )
    with pytest.raises(ValueError, match="examples do not match"):
        runner.evaluate_condition(
            model,
            CharacterTokenizer(),
            [],
            condition="pristine_frozen_base",
            batch_size=1,
            pad_token_id=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            greedy=False,
        )

    example = SimpleNamespace(condition="pristine_frozen_base")
    with pytest.raises(RuntimeError, match="without Delta-Mem attached"):
        runner.evaluate_condition(
            model,
            CharacterTokenizer(),
            [example],
            condition="pristine_frozen_base",
            batch_size=1,
            pad_token_id=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            greedy=False,
        )


@pytest.mark.parametrize("profile", ["development", "sealed_validation"])
def test_formal_profiles_reject_partial_split_limits(
    profile: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)
    kwargs: dict[str, Any] = {}
    if profile == "sealed_validation":
        kwargs = {
            "adapter_path": tmp_path / "adapter",
            "development_run_dir": tmp_path / "development",
        }
    with pytest.raises(ValueError, match="require complete splits"):
        runner.run_experiment(
            source_manifest=tmp_path / "manifest.json",
            output_dir=tmp_path / "output",
            profile=profile,
            eval_limit=1,
            **kwargs,
        )


def test_formal_profile_rejects_disabled_greedy_before_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)
    with pytest.raises(ValueError, match="require greedy evaluation"):
        runner.run_experiment(
            source_manifest=tmp_path / "missing-manifest.json",
            output_dir=tmp_path / "output",
            profile="development",
            greedy=False,
        )


def _formal_distributed_context(
    process_rank: int = 0,
) -> runner.distributed.DistributedTrainingContext:
    return runner.distributed.DistributedTrainingContext(
        process_rank=process_rank,
        local_rank=process_rank,
        world_size=4,
        device=torch.device("cpu"),
        backend="nccl",
        control_backend="gloo",
        control_group=object(),
        rank_devices=(),
    )


def test_formal_development_locks_profiled_rank_before_opening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)

    with pytest.raises(ValueError, match="adapter_rank"):
        runner.run_experiment(
            source_manifest=tmp_path / "missing-manifest.json",
            output_dir=tmp_path / "output",
            profile="development",
            rank=4,
            distributed_context=_formal_distributed_context(),
        )


def test_distributed_preflight_requires_exact_three_step_production_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)

    with pytest.raises(ValueError, match="exactly 3 updates"):
        runner.run_experiment(
            source_manifest=tmp_path / "missing-manifest.json",
            output_dir=tmp_path / "output",
            profile="development",
            distributed_context=_formal_distributed_context(),
            distributed_preflight=True,
        )


@pytest.mark.parametrize("process_rank", [0, 2])
def test_distributed_preflight_primary_and_worker_complete_lifecycle(
    process_rank: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    model_root = tmp_path / "model"
    model_root.mkdir()
    model_artifact = model_root / "model.safetensors"
    model_artifact.write_bytes(b"frozen-model")
    output_dir = tmp_path / f"preflight-rank-{process_rank}"
    context = _formal_distributed_context(process_rank)
    model = torch.nn.Linear(1, 1, bias=False)
    tokenizer = SimpleNamespace(pad_token_id=0)
    phases: list[str] = []
    destroyed: list[runner.distributed.DistributedTrainingContext] = []
    bundle = runner.ProfileBundle(
        profile="development",
        train_episodes=(object(),),
        evaluation_episodes=(object(),),
        evaluation_split="development",
        development_manifest={
            "manifest_receipt": {"payload_sha256": "a" * 64},
        },
        sealed_manifest=None,
        source_paths=(source_manifest,),
        model_binding={"binding_sha256": "b" * 64},
        eligibility={"opened_splits": ["train", "development"]},
    )

    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)
    monkeypatch.setattr(runner, "load_profile_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(
        runner,
        "resolve_model_artifacts",
        lambda *_args, **_kwargs: (model_root, (model_artifact,)),
    )
    monkeypatch.setattr(
        runner,
        "_load_model_and_tokenizer",
        lambda *_args, **_kwargs: (
            model,
            tokenizer,
            (0,),
            ("weight",),
            (),
        ),
    )
    monkeypatch.setattr(
        runner,
        "audit_trainable_parameters",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        runner,
        "_named_adapter_parameters",
        lambda current_model: (("weight", current_model.weight),),
    )
    monkeypatch.setattr(
        runner,
        "snapshot_delta_mem_weights",
        lambda current_model: {"weight": current_model.weight.detach().clone()},
    )
    monkeypatch.setattr(
        runner.distributed,
        "broadcast_named_parameters",
        lambda *_args, **_kwargs: {
            "parameter_tensors": 1,
            "parameter_names_sha256": "c" * 64,
            "bucket_plan_sha256": "d" * 64,
            "collective_buckets": 1,
            "broadcast_bytes": 4,
        },
    )
    monkeypatch.setattr(
        runner,
        "select_complete_episodes",
        lambda episodes, _limit: list(episodes),
    )
    monkeypatch.setattr(
        runner,
        "build_training_examples",
        lambda *_args, **_kwargs: [SimpleNamespace(row_id=f"row-{index}") for index in range(4)],
    )
    monkeypatch.setattr(
        runner,
        "audit_training_dataset",
        lambda *_args, **_kwargs: _production_training_dataset_audit(),
    )

    def fake_train_model_distributed(
        current_model: torch.nn.Linear,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with torch.no_grad():
            current_model.weight.add_(1.0)
        return {
            "steps": runner.DISTRIBUTED_PREFLIGHT_STEPS,
            "max_steps": runner.DISTRIBUTED_PREFLIGHT_STEPS,
            "progress_sha256": "e" * 64,
            "router_gradient_audit": {
                "all_ranks_all_modules_finite_nonzero": True,
            },
            "distributed": {},
        }

    monkeypatch.setattr(runner, "train_model_distributed", fake_train_model_distributed)

    def gather_objects(
        current: runner.distributed.DistributedTrainingContext,
        value: Any,
    ) -> tuple[Any, ...]:
        assert current is context
        if isinstance(value, dict) and "source_snapshot_sha256" in value:
            return tuple(dict(value, rank=rank) for rank in range(current.world_size))
        return tuple(value for _ in range(current.world_size))

    monkeypatch.setattr(runner.distributed, "gather_objects", gather_objects)
    monkeypatch.setattr(
        runner.distributed,
        "require_consensus",
        lambda current, value, **_kwargs: tuple(value for _ in range(current.world_size)),
    )

    def phase_consensus(
        current: runner.distributed.DistributedTrainingContext,
        *,
        phase: str,
        error: BaseException | None,
    ) -> None:
        assert current is context
        assert error is None
        phases.append(phase)

    monkeypatch.setattr(runner.distributed, "phase_consensus", phase_consensus)
    monkeypatch.setattr(
        runner.distributed,
        "destroy_distributed_training",
        lambda current: destroyed.append(current),
    )
    monkeypatch.setattr(
        runner,
        "build_distributed_preflight_gate",
        lambda _training: {"passed": True, "failed_checks": []},
    )
    monkeypatch.setattr(runner, "_preflight_code_bindings", lambda: {})
    monkeypatch.setattr(
        runner,
        "build_condition_examples",
        lambda *_args, **_kwargs: pytest.fail("preflight entered evaluation"),
    )

    result = runner.run_experiment(
        source_manifest=source_manifest,
        output_dir=output_dir,
        profile="development",
        max_steps=runner.DISTRIBUTED_PREFLIGHT_STEPS,
        distributed_context=context,
        distributed_preflight=True,
    )

    assert phases[-1] == "rank-zero-preflight-receipt"
    assert destroyed == [context]
    if process_rank == 0:
        receipt_path = output_dir / "preflight_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema"] == runner.DISTRIBUTED_PREFLIGHT_SCHEMA
        assert receipt["status"] == "passed"
        assert receipt["gate_passed"] is True
        assert len(receipt["preflight_receipt_sha256"]) == 64
        assert _canonical(result["preflight_receipt"]) == _canonical(receipt)
        assert result["gate"] == receipt["gate"]
    else:
        assert not output_dir.exists()
        assert result == {
            "output_dir": str(output_dir.resolve()),
            "distributed_worker_rank": process_rank,
            "training_complete": True,
        }


def test_natural_cli_defaults_to_profiled_route64_contract() -> None:
    args = runner.parse_args(
        [
            "--source-manifest",
            "manifest.json",
            "--output-dir",
            "output",
        ]
    )

    assert args.rank == runner.PRODUCTION_ADAPTER_RANK == 32
    assert args.key_dim == runner.PRODUCTION_KEY_DIM == 64
    assert args.max_steps == runner.PRODUCTION_UPDATES == 3840
    assert runner._parse_training_conditions(args.training_conditions) == (
        runner.SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS
    )


def test_natural_training_progress_rewrites_shared_runtime_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_path = tmp_path / "training_progress.jsonl"

    def fake_train_model(*_args: Any, progress_path: Path, **_kwargs: Any) -> dict[str, Any]:
        progress_path.write_text(
            json.dumps(
                {
                    "schema": "rwkv_ms_synthetic_compositional_train_step.v3",
                    "step": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"steps": 1}

    monkeypatch.setattr(runner.runtime, "train_model", fake_train_model)
    result = runner.train_model(
        torch.nn.Linear(1, 1),
        [],
        seed=42,
        epochs=1,
        max_steps=1,
        batch_size=1,
        learning_rate=2e-4,
        answer_weight=1.0,
        route_weight=1.0,
        max_grad_norm=1.0,
        pad_token_id=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        progress_path=progress_path,
        training_conditions=("correct_state",),
    )

    record = json.loads(progress_path.read_text(encoding="utf-8"))
    assert record["schema"] == runner.TRAIN_STEP_SCHEMA
    assert record["training_conditions"] == ["correct_state"]
    assert result["progress_schema"] == runner.TRAIN_STEP_SCHEMA
    assert not (tmp_path / ".training_progress.jsonl.synthetic-runtime.tmp").exists()


@pytest.mark.parametrize(
    ("profile", "expected_splits"),
    [
        ("train", ["train"]),
        ("development", ["train", "development"]),
        ("sealed_validation", ["sealed_validation"]),
    ],
)
def test_profile_loader_opens_only_authorized_splits(
    profile: str,
    expected_splits: list[str],
    episode: runner.NaturalEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    episodes = {
        split: replace(
            episode,
            episode_id=f"{split}-episode",
            split=split,
            passage_components=tuple(
                f"{split}:component-{slot}" for slot in range(4)
            ),
        )
        for split in runner.PROFILES
    }
    opened: list[str] = []
    manifest = {"model_binding": {"binding_sha256": "b" * 64}}

    monkeypatch.setenv("HF_ENDPOINT", runner.HF_MIRROR_ENDPOINT)
    monkeypatch.setattr(runner, "_read_json_file", lambda *_args: manifest)
    monkeypatch.setattr(
        runner,
        "_validate_profile_manifest",
        lambda _manifest, *, manifest_path, profile: (
            "sealed_validation" if profile == "sealed_validation" else profile
        ),
    )

    def fake_load_split(
        _manifest: dict[str, Any],
        *,
        manifest_path: Path,
        split: str,
    ) -> tuple[tuple[runner.NaturalEpisode, ...], Path]:
        opened.append(split)
        return (episodes[split],), manifest_path.parent / f"{split}.jsonl"

    monkeypatch.setattr(runner, "_load_split", fake_load_split)
    bundle = runner.load_profile_bundle(manifest_path, profile=profile)

    assert opened == expected_splits
    assert bundle.eligibility["opened_splits"] == expected_splits
    assert bundle.evaluation_split == (
        "sealed_validation" if profile == "sealed_validation" else profile
    )
    assert bundle.train_episodes == (() if profile == "sealed_validation" else (episodes["train"],))


def test_source_and_model_artifact_immutability_helpers_reject_changes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.jsonl"
    source_path.write_text('{"row":1}\n', encoding="utf-8")
    before = runner.snapshot_files([source_path])
    assert runner.assert_snapshot_unchanged(before, description="source") == before
    source_path.write_text('{"row":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source artifacts changed"):
        runner.assert_snapshot_unchanged(before, description="source")

    model_root = tmp_path / "model"
    model_root.mkdir()
    artifact = model_root / "model.safetensors"
    artifact.write_bytes(b"frozen-model-v1")
    fingerprint = runner.snapshot_files([artifact])[str(artifact.resolve())]
    binding = {
        "local_model_path": str(model_root.resolve()),
        "local_artifacts": {artifact.name: fingerprint},
    }
    resolved, paths = runner.resolve_model_artifacts(binding)
    assert resolved == model_root.resolve()
    assert paths == (artifact.resolve(),)
    artifact.write_bytes(b"changed-model-v2")
    with pytest.raises(ValueError, match="Model artifact hash differs"):
        runner.resolve_model_artifacts(binding)


def test_trainable_parameter_audit_rejects_frozen_base_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MemoryModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta = torch.nn.Parameter(torch.ones(()))

        @staticmethod
        def is_trainable_parameter(name: str) -> bool:
            return name == "delta"

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(1, 1)
            self.memory = MemoryModule()
            self.base.requires_grad_(False)

    model = Model()
    monkeypatch.setattr(
        runner,
        "iter_delta_mem_modules",
        lambda target: iter((("memory", target.memory),)),
    )
    audit = runner.audit_trainable_parameters(
        model,
        expected_trainable_names=("memory.delta",),
    )
    assert audit["passed"] is True
    assert audit["only_delta_mem_parameters_trainable"] is True

    model.base.weight.requires_grad_(True)
    leaked = runner.audit_trainable_parameters(model)
    assert leaked["passed"] is False
    assert leaked["only_delta_mem_parameters_trainable"] is False
