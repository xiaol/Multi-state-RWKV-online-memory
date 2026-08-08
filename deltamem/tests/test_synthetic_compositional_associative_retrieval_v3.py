from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_compositional_associative_retrieval_canary_v3 as canary,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runner,
)


@pytest.fixture(scope="module")
def tokenizer():
    os.environ["HF_ENDPOINT"] = canary.HF_MIRROR_ENDPOINT
    return canary.load_local_tokenizer(canary.DEFAULT_MODEL_PATH)


@pytest.fixture(scope="module")
def train_rows(tokenizer):
    return canary.build_partition_rows(tokenizer, "train")


def test_delta_config_locks_explicit_four_slot_topology() -> None:
    config = runner.build_delta_config(
        target_layers=(0, 1), rank=16, key_dim=16, temperature=8.0
    )

    assert config.rank == 16
    assert config.alpha == 32
    assert config.alpha / config.rank == 2.0
    assert config.memory_backend == "rwkv_ms"
    assert config.memory_readout_mode == "projected_kv_slots"
    assert config.memory_write_granularity == "token"
    assert config.rwkv_ms_num_states == 4
    assert config.projected_kv_key_dim == 16
    assert config.projected_kv_temperature == 8.0
    assert config.projected_kv_update_cosine_threshold == 1.0
    assert config.target_layers == (0, 1)


def test_delta_config_rejects_nonpositive_value_rank() -> None:
    with pytest.raises(ValueError, match="value rank must be positive"):
        runner.build_delta_config(rank=0)


def test_examples_bind_correct_donor_swap_rewrite_shuffle_and_no_write(
    tokenizer,
    train_rows,
) -> None:
    row = train_rows[0]
    correct = runner.correct_example(row)
    donor = runner.donor_example(row, train_rows)
    swapped = runner.value_swap_example(row, tokenizer)
    rewritten = runner.target_slot_rewrite_example(row, tokenizer)
    shuffled = runner.shuffled_slot_example(row)
    no_write = runner.no_write_example(row)

    assert correct.condition == "correct"
    assert len(correct.write_records) == len(correct.write_slots) == 4
    assert correct.write_slots == (0, 1, 2, 3)
    assert correct.target_slot == row["query_route_target_slot"]
    assert correct.expected_value == row["query"]["target_value"]
    assert correct.expected_answer_token_ids == tuple(
        token for token in correct.labels if token != -100
    )

    donor_row = train_rows[row["donor"]["row_ordinal"]]
    assert donor.condition == "donor"
    assert donor.expected_value == row["donor"]["expected_target_value"]
    assert [record["value"] for record in donor.write_records] == [
        record["value"] for record in donor_row["record_local_writes"]
    ]
    assert donor.expected_value != correct.expected_value

    target_slot = int(row["query_route_target_slot"])
    source_slot = row["value_swap"]["source_slot_by_destination_slot"][target_slot]
    assert swapped.condition == "value_swap"
    assert swapped.write_records[target_slot]["value"] == row["record_local_writes"][
        source_slot
    ]["value"]
    assert swapped.expected_value == row["value_swap"]["expected_target_value"]
    assert swapped.target_slot == target_slot

    assert rewritten.condition == "target_slot_rewrite"
    assert rewritten.target_slot == target_slot
    assert rewritten.expected_value == rewritten.write_records[target_slot]["value"]
    assert rewritten.expected_value not in {
        record["value"] for record in row["record_local_writes"]
    }

    assert shuffled.condition == "shuffled_slots"
    assert shuffled.write_slots == (2, 0, 3, 1)
    assert shuffled.target_slot == shuffled.write_slots[target_slot]
    assert shuffled.expected_value == correct.expected_value

    assert no_write.condition == "no_write"
    assert no_write.write_records == no_write.write_slots == ()
    assert no_write.target_slot is None


def test_condition_builder_uses_full_partition_for_donor_lookup(
    tokenizer,
    train_rows,
) -> None:
    selected = runner.select_complete_memory_states(train_rows, 4)
    examples = runner.build_condition_examples(
        selected,
        tokenizer,
        "donor",
        all_rows=train_rows,
    )

    assert len(examples) == 4
    for source, example in zip(selected, examples, strict=True):
        donor = train_rows[source["donor"]["row_ordinal"]]
        assert example.expected_value == donor["query"]["target_value"]


def test_target_slot_rewrite_is_deterministic_and_changes_only_target_record(
    tokenizer,
    train_rows,
) -> None:
    row = train_rows[0]
    correct = runner.correct_example(row)
    first = runner.target_slot_rewrite_example(row, tokenizer)
    second = runner.target_slot_rewrite_example(copy.deepcopy(row), tokenizer)
    selection = runner._target_slot_rewrite_selection(row)
    target_slot = int(row["query_route_target_slot"])

    assert first == second
    assert selection["source_split"] == "train"
    assert selection["source_mapping_offset"] == row["mapping_offset"]
    assert selection["alternate_mapping_offset"] in canary.TRAIN_OFFSETS
    assert selection["alternate_mapping_offset"] != row["mapping_offset"]
    assert selection["value"] == first.expected_value
    assert first.target_slot_rewrite_selection == selection
    assert selection["value_index"] == canary._mapped_value_index(
        selection["key_index"], selection["alternate_mapping_offset"]
    )
    assert first.write_slots == correct.write_slots
    assert first.target_slot == correct.target_slot == target_slot
    assert [record["key"] for record in first.write_records] == [
        record["key"] for record in correct.write_records
    ]
    changed_records = [
        index
        for index, (original, rewritten) in enumerate(
            zip(correct.write_records, first.write_records, strict=True)
        )
        if original != rewritten
    ]
    assert changed_records == [target_slot]
    assert all(
        first.write_records[index] == correct.write_records[index]
        for index in range(len(correct.write_records))
        if index != target_slot
    )
    first_answer = first.answer_mask.index(True)
    correct_answer = correct.answer_mask.index(True)
    assert first.read_input_ids[:first_answer] == correct.read_input_ids[:correct_answer]
    assert tuple(
        token
        for token, selected in zip(first.read_input_ids, first.query_mask, strict=True)
        if selected
    ) == tuple(
        token
        for token, selected in zip(
            correct.read_input_ids, correct.query_mask, strict=True
        )
        if selected
    )


def _minimal_rewrite_selector_row(
    source_split: str,
    mapping_offset: int,
    key_index: int,
):
    key_indices = [
        (key_index + relative_index) % canary.NONCE_COUNT
        for relative_index in range(canary.RECORDS_PER_EPISODE)
    ]
    records = [
        {
            "key_index": record_key_index,
            "key": canary.KEY_LABELS[record_key_index],
            "value": canary.VALUE_LABELS[
                canary._mapped_value_index(record_key_index, mapping_offset)
            ],
        }
        for record_key_index in key_indices
    ]
    return {
        "source_split": source_split,
        "row_id": (
            f"selector-{source_split}-offset-{mapping_offset:02d}-key-{key_index:02d}"
        ),
        "mapping_offset": mapping_offset,
        "record_local_writes": records,
        "query_route_target_slot": 0,
        "query": {
            "key_index": key_index,
            "key": canary.KEY_LABELS[key_index],
        },
    }


def test_target_slot_rewrite_uses_alternate_train_offset_for_every_train_row(
    train_rows,
) -> None:
    for row in train_rows:
        selection = runner._target_slot_rewrite_selection(row)
        expected_value_index = canary._mapped_value_index(
            int(row["query"]["key_index"]),
            selection["alternate_mapping_offset"],
        )

        assert selection["source_split"] == "train"
        assert selection["source_mapping_offset"] == row["mapping_offset"]
        assert selection["alternate_mapping_offset"] in canary.TRAIN_OFFSETS
        assert selection["alternate_mapping_offset"] != row["mapping_offset"]
        assert selection["value_index"] == expected_value_index
        assert selection["value"] == canary.VALUE_LABELS[expected_value_index]
        assert selection["value"] not in {
            record["value"] for record in row["record_local_writes"]
        }


def test_heldout_rewrite_bindings_are_split_mapped_and_train_novel() -> None:
    training_bindings = {
        (
            canary.KEY_LABELS[key_index],
            canary.VALUE_LABELS[
                canary._mapped_value_index(key_index, mapping_offset)
            ],
        )
        for key_index in range(canary.NONCE_COUNT)
        for mapping_offset in canary.TRAIN_OFFSETS
    }
    checked = 0
    for mapping_offset in canary.HELDOUT_OFFSETS:
        for key_index in range(canary.NONCE_COUNT):
            row = _minimal_rewrite_selector_row(
                "heldout",
                mapping_offset,
                key_index,
            )
            selection = runner._target_slot_rewrite_selection(row)
            expected_value_index = canary._mapped_value_index(
                key_index,
                selection["alternate_mapping_offset"],
            )
            binding = (row["query"]["key"], selection["value"])

            assert selection == runner._target_slot_rewrite_selection(
                copy.deepcopy(row)
            )
            assert selection["source_split"] == "heldout"
            assert selection["source_mapping_offset"] == mapping_offset
            assert selection["alternate_mapping_offset"] in canary.HELDOUT_OFFSETS
            assert selection["alternate_mapping_offset"] != mapping_offset
            assert selection["value_index"] == expected_value_index
            assert selection["value"] == canary.VALUE_LABELS[expected_value_index]
            assert selection["value"] not in {
                record["value"] for record in row["record_local_writes"]
            }
            assert binding not in training_bindings
            checked += 1

    assert checked == len(canary.HELDOUT_OFFSETS) * canary.NONCE_COUNT


def test_condition_builder_constructs_target_slot_rewrites(
    tokenizer,
    train_rows,
) -> None:
    selected = runner.select_complete_memory_states(train_rows, 4)
    examples = runner.build_condition_examples(
        selected,
        tokenizer,
        "target_slot_rewrite",
    )

    assert len(examples) == 4
    assert all(example.condition == "target_slot_rewrite" for example in examples)
    assert [example.target_slot for example in examples] == [0, 1, 2, 3]


def test_select_complete_memory_states_never_splits_query_family(train_rows) -> None:
    selected = runner.select_complete_memory_states(train_rows, 7)

    assert len(selected) == 4
    assert len({row["memory_state_id"] for row in selected}) == 1
    assert sorted(row["query_route_target_slot"] for row in selected) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="smaller than one complete"):
        runner.select_complete_memory_states(train_rows, 3)


def test_collator_preserves_record_major_masks_and_answer_labels(train_rows) -> None:
    examples = [runner.correct_example(row) for row in train_rows[:2]]
    batch = runner.collate_examples(
        examples,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert len(batch.write_records) == 4
    assert batch.read_input_ids.shape == batch.labels.shape
    assert batch.query_mask.shape == batch.labels.shape
    assert batch.answer_mask.shape == batch.labels.shape
    assert torch.equal(batch.labels.ne(-100), batch.answer_mask)
    assert torch.equal(batch.target_slots, torch.tensor([0, 1]))
    for record_index, record in enumerate(batch.write_records):
        assert torch.equal(record["slots"], torch.tensor([record_index, record_index]))
        assert not bool((record["key_mask"] & record["value_mask"]).any().item())
        assert bool(record["key_mask"].any(dim=1).all().item())
        assert bool(record["value_mask"].any(dim=1).all().item())


def test_no_write_batch_has_no_records_and_absent_targets(train_rows) -> None:
    batch = runner.collate_examples(
        [runner.no_write_example(train_rows[0])],
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert batch.write_records == []
    assert torch.equal(batch.target_slots, torch.tensor([-1]))


def test_answer_loss_and_exact_predictions_use_causal_predictor_positions() -> None:
    labels = torch.tensor([[-100, -100, 2, 3]])
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0

    loss = runner.causal_answer_loss(logits, labels)
    exact, token_correct, token_total = runner._answer_exact_predictions(logits, labels)

    expected = F.cross_entropy(
        torch.stack((logits[0, 1], logits[0, 2])), torch.tensor([2, 3])
    )
    assert torch.allclose(loss, expected)
    assert exact == [True]
    assert (token_correct, token_total) == (2, 2)
    predicted_ids, expected_ids = runner._answer_prediction_token_ids(logits, labels)
    assert predicted_ids == expected_ids == [(2, 3)]


def test_compact_answer_logits_are_exactly_equivalent_to_full_logits() -> None:
    labels = torch.tensor(
        [
            [-100, -100, 2, 3, -100, -100],
            [-100, -100, -100, 4, 5, 6],
        ]
    )
    logits = torch.randn(2, 6, 7)
    predictor_indices = runner._answer_predictor_indices(labels)
    compact_logits = logits.index_select(1, predictor_indices)

    reference = F.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    full_loss = runner.causal_answer_loss(logits, labels)
    compact_loss = runner.causal_answer_loss(compact_logits, labels)

    assert predictor_indices.tolist() == [1, 2, 3, 4]
    assert torch.allclose(full_loss, reference)
    assert torch.allclose(compact_loss, reference)
    assert runner._answer_prediction_token_ids(
        compact_logits, labels
    ) == runner._answer_prediction_token_ids(logits, labels)
    assert runner._answer_exact_predictions(
        compact_logits, labels
    ) == runner._answer_exact_predictions(logits, labels)


def test_route_loss_pools_query_tokens_and_keeps_graph_connection() -> None:
    first = torch.tensor(
        [[[0.0, 0.0], [1.0, 3.0], [1.0, 3.0]]], requires_grad=True
    )
    second = torch.tensor(
        [[[0.0, 0.0], [2.0, 4.0], [2.0, 4.0]]], requires_grad=True
    )
    query_mask = torch.tensor([[False, True, True]])
    targets = torch.tensor([1])

    loss, predictions = runner.route_loss_and_predictions(
        {"layer0": first, "layer1": second}, query_mask, targets
    )
    loss.backward()

    assert predictions == {"layer0": torch.tensor([1]), "layer1": torch.tensor([1])}
    assert first.grad is not None and torch.count_nonzero(first.grad).item() > 0
    assert second.grad is not None and torch.count_nonzero(second.grad).item() > 0


def test_route_loss_adds_only_violated_hard_negative_margins() -> None:
    logits = torch.tensor(
        [
            [[1.0, 2.0, -1.0, -2.0]],
            [[4.0, 3.0, 0.0, -1.0]],
        ],
        requires_grad=True,
    )
    query_mask = torch.ones(2, 1, dtype=torch.bool)
    targets = torch.tensor([0, 0])

    base, base_predictions = runner.route_loss_and_predictions(
        {"layer": logits}, query_mask, targets
    )
    with_margin, predictions = runner.route_loss_and_predictions(
        {"layer": logits},
        query_mask,
        targets,
        hard_negative_margin=0.5,
        hard_negative_weight=0.1,
    )
    with_margin.backward()

    assert predictions.keys() == base_predictions.keys() == {"layer"}
    assert torch.equal(predictions["layer"], torch.tensor([1, 0]))
    assert torch.equal(base_predictions["layer"], predictions["layer"])
    assert float((with_margin - base).item()) == pytest.approx(0.075)
    assert logits.grad is not None
    assert logits.grad[0, 0, 0] < 0.0
    assert logits.grad[0, 0, 1] > 0.0


def test_query_counterfactual_audit_requires_all_four_routes_and_identical_state(
    train_rows,
) -> None:
    examples = [runner.correct_example(row) for row in train_rows[:4]]
    route_by_row = {
        example.row_id: {"layer0": int(example.target_slot)} for example in examples
    }
    result = {
        "route_predictions_by_row": route_by_row,
        "state_digest_by_row": {example.row_id: "same" for example in examples},
        "route_by_layer": {"layer0": {"correct": 4, "total": 4, "accuracy": 1.0}},
    }

    audit = runner.query_counterfactual_audit(examples, result)

    assert audit["runtime_byte_identical_state_fraction"] == 1.0
    assert audit["query_counterfactual_route_accuracy"] == 1.0
    assert audit["family_layer_all_four_correct_fraction"] == 1.0

    corrupted = copy.deepcopy(result)
    corrupted["route_predictions_by_row"][examples[3].row_id]["layer0"] = 0
    corrupted["state_digest_by_row"][examples[3].row_id] = "different"
    failed = runner.query_counterfactual_audit(examples, corrupted)
    assert failed["runtime_byte_identical_state_fraction"] == 0.0
    assert failed["query_counterfactual_route_accuracy"] == 0.75
    assert failed["family_layer_all_four_correct_fraction"] == 0.0


def _answer_prediction_result(examples, *, greedy: bool = True):
    return {
        "greedy_answer_evaluated": greedy,
        "answer_predictions_by_row": {
            example.row_id: {
                "expected_answer_token_ids": list(
                    example.expected_answer_token_ids
                ),
                "teacher_forced_prediction_token_ids": list(
                    example.expected_answer_token_ids
                ),
                "teacher_forced_exact": True,
                "greedy_generated_token_ids": (
                    list(example.expected_answer_token_ids) if greedy else None
                ),
                "greedy_exact": True if greedy else None,
            }
            for example in examples
        },
    }


def _complete_condition_evidence(
    examples,
    *,
    condition: str,
    module_names: tuple[str, ...],
    greedy: bool,
):
    rows = len(examples)
    answer_predictions = {
        example.row_id: {
            "expected_answer_token_ids": list(example.expected_answer_token_ids),
            "teacher_forced_prediction_token_ids": list(
                example.expected_answer_token_ids
            ),
            "teacher_forced_exact": True,
            "greedy_generated_token_ids": (
                list(example.expected_answer_token_ids) if greedy else None
            ),
            "greedy_exact": True if greedy else None,
        }
        for example in examples
    }
    token_total = sum(len(example.expected_answer_token_ids) for example in examples)
    positive = condition != "no_write"
    if positive:
        routes = {
            example.row_id: {
                module_name: int(example.target_slot) for module_name in module_names
            }
            for example in examples
        }
        layer_metrics = {
            module_name: {"correct": rows, "total": rows, "accuracy": 1.0}
            for module_name in module_names
        }
        state_digests = {
            example.row_id: canary.canonical_sha256(example.memory_state_id)
            for example in examples
        }
        route_total = rows * len(module_names)
        occupancy_total = route_total
        forced_total = route_total * canary.RECORDS_PER_EPISODE
    else:
        routes = {}
        layer_metrics = {
            module_name: {"correct": 0, "total": 0, "accuracy": None}
            for module_name in module_names
        }
        state_digests = {}
        route_total = occupancy_total = forced_total = 0
    return {
        "condition": condition,
        "rows": rows,
        "teacher_forced_answer_exact_count": rows,
        "teacher_forced_answer_exact_accuracy": 1.0,
        "teacher_forced_answer_token_correct": token_total,
        "teacher_forced_answer_token_total": token_total,
        "teacher_forced_answer_token_accuracy": 1.0,
        "greedy_answer_evaluated": greedy,
        "greedy_answer_exact_count": rows if greedy else None,
        "greedy_answer_exact_accuracy": 1.0 if greedy else None,
        "answer_predictions_by_row": answer_predictions,
        "semantic_route_correct": route_total,
        "semantic_route_total": route_total,
        "semantic_route_accuracy": 1.0 if positive else None,
        "route_by_layer": layer_metrics,
        "route_predictions_by_row": routes,
        "full_occupancy_count": occupancy_total,
        "full_occupancy_total": occupancy_total,
        "full_occupancy_fraction": 1.0 if positive else None,
        "forced_write_route_correct": forced_total,
        "forced_write_route_total": forced_total,
        "forced_write_route_accuracy": 1.0 if positive else None,
        "route_absent_module_rows": 0 if positive else rows * len(module_names),
        "route_possible_module_rows": rows * len(module_names),
        "route_absent_fraction": 0.0 if positive else 1.0,
        "state_digest_by_row": state_digests,
    }


def test_evaluation_evidence_is_recomputed_from_rows(
    tokenizer,
    train_rows,
) -> None:
    selected = runner.select_complete_memory_states(train_rows, 4)
    module_names = ("model.language_model.layers.0.self_attn",)
    examples_by_condition = {
        condition: runner.build_condition_examples(
            selected,
            tokenizer,
            condition,
            all_rows=train_rows,
        )
        for condition in runner.CONDITIONS
    }
    conditions = {
        condition: _complete_condition_evidence(
            examples,
            condition=condition,
            module_names=module_names,
            greedy=True,
        )
        for condition, examples in examples_by_condition.items()
    }
    evaluation = {
        "eval_rows": 4,
        "conditions": conditions,
        "query_counterfactual_audit": runner.query_counterfactual_audit(
            examples_by_condition["correct"], conditions["correct"]
        ),
        "target_slot_rewrite_audit": runner.target_slot_rewrite_audit(
            examples_by_condition["correct"],
            examples_by_condition["target_slot_rewrite"],
            conditions["correct"],
            conditions["target_slot_rewrite"],
        ),
    }
    protocol = {
        "eval_split": "train",
        "eval_limit": 4,
        "target_layers": [0],
        "greedy_answer_evaluation": True,
    }
    receipt = {
        "model_attachment": {"replaced_modules": list(module_names)},
        "training": {
            "router_gradient_audit": {
                "modules": 1,
                "finite_nonzero_modules": 1,
                "all_modules_finite_nonzero": True,
                "records": [
                    {
                        "module": module_names[0],
                        "layer": 0,
                        "projected_kv_key_route_grad_norm": 0.5,
                        "finite_nonzero": True,
                    }
                ],
            }
        },
    }
    source = {
        "model": {"path": str(canary.DEFAULT_MODEL_PATH)},
        "partitions": {"train": train_rows, "heldout": []},
    }

    runner._validate_evaluation_evidence(
        evaluation,
        source=source,
        protocol=protocol,
        receipt=receipt,
    )

    corrupted = copy.deepcopy(evaluation)
    corrupted["conditions"]["correct"][
        "teacher_forced_answer_exact_count"
    ] = 3
    with pytest.raises(ValueError, match="answer aggregate binding differs"):
        runner._validate_evaluation_evidence(
            corrupted,
            source=source,
            protocol=protocol,
            receipt=receipt,
        )

    corrupted = copy.deepcopy(evaluation)
    first_row = selected[0]["row_id"]
    corrupted["conditions"]["correct"]["route_predictions_by_row"][first_row][
        module_names[0]
    ] = 3
    with pytest.raises(ValueError, match="route aggregate binding differs"):
        runner._validate_evaluation_evidence(
            corrupted,
            source=source,
            protocol=protocol,
            receipt=receipt,
        )


def test_target_slot_rewrite_audit_binds_same_row_output_flips(
    tokenizer,
    train_rows,
) -> None:
    rows = runner.select_complete_memory_states(train_rows, 4)
    correct = [runner.correct_example(row) for row in rows]
    rewritten = [runner.target_slot_rewrite_example(row, tokenizer) for row in rows]
    correct_result = _answer_prediction_result(correct)
    rewrite_result = _answer_prediction_result(rewritten)

    audit = runner.target_slot_rewrite_audit(
        correct,
        rewritten,
        correct_result,
        rewrite_result,
    )

    assert audit["rows"] == 4
    assert audit["expected_answers_differ_fraction"] == 1.0
    assert audit["query_prefix_unchanged_fraction"] == 1.0
    assert audit["query_key_unchanged_fraction"] == 1.0
    assert audit["split_mapping_selection_valid_fraction"] == 1.0
    assert audit["heldout_rows"] == 0
    assert audit["heldout_rewrite_binding_absent_from_training_fraction"] is None
    assert audit["target_slot_unchanged_fraction"] == 1.0
    assert audit["write_slots_unchanged_fraction"] == 1.0
    assert audit["target_write_record_only_changed_fraction"] == 1.0
    assert audit["rewrite_target_value_matches_expected_fraction"] == 1.0
    assert (
        audit["replacement_value_absent_from_original_episode_fraction"] == 1.0
    )
    assert audit["pair_contract_passed_fraction"] == 1.0
    assert audit["teacher_forced_joint_exact_output_flip_fraction"] == 1.0
    assert audit["greedy_joint_exact_output_flip_fraction"] == 1.0
    assert all(
        pair["changed_write_record_indices"] == [correct[index].target_slot]
        and pair["greedy_output_flip"] is True
        for index, pair in enumerate(audit["pairs_by_row"].values())
    )
    assert all(
        pair["replacement_value_absent_from_original_episode"] is True
        for pair in audit["pairs_by_row"].values()
    )

    first_row_id = correct[0].row_id
    corrupted_selection_examples = copy.deepcopy(rewritten)
    corrupted_selection_examples[0].target_slot_rewrite_selection[
        "alternate_mapping_offset"
    ] = corrupted_selection_examples[0].source_mapping_offset
    failed_selection_contract = runner.target_slot_rewrite_audit(
        correct,
        corrupted_selection_examples,
        correct_result,
        rewrite_result,
    )
    assert failed_selection_contract["pair_contract_passed_fraction"] == 0.75

    corrupted_examples = copy.deepcopy(rewritten)
    non_target_slot = (int(corrupted_examples[0].target_slot) + 1) % 4
    corrupted_examples[0].write_records[non_target_slot]["value"] = "audit-corruption"
    failed_write_contract = runner.target_slot_rewrite_audit(
        correct,
        corrupted_examples,
        correct_result,
        rewrite_result,
    )
    assert failed_write_contract["pair_contract_passed_fraction"] == 0.75

    corrupted = copy.deepcopy(rewrite_result)
    corrupted_prediction = corrupted["answer_predictions_by_row"][first_row_id]
    corrupted_prediction["greedy_generated_token_ids"] = list(
        correct[0].expected_answer_token_ids
    )
    corrupted_prediction["greedy_exact"] = False
    failed = runner.target_slot_rewrite_audit(
        correct,
        rewritten,
        correct_result,
        corrupted,
    )
    assert failed["greedy_joint_exact_output_flip_fraction"] == 0.75


def _condition_result(*, answer: float, route: float | None, no_write: bool = False):
    return {
        "greedy_answer_exact_accuracy": answer,
        "teacher_forced_answer_exact_accuracy": answer,
        "semantic_route_accuracy": route,
        "route_absent_fraction": 1.0 if no_write else 0.0,
        "full_occupancy_fraction": None if no_write else 1.0,
        "forced_write_route_accuracy": None if no_write else 1.0,
    }


def test_acceptance_gate_is_conjunctive() -> None:
    acceptance = canary.canary_spec()["acceptance_gate"]
    evaluation = {
        "seed": 42,
        "acceptance_contract": acceptance,
        "query_counterfactual_audit": {
            "query_counterfactual_route_accuracy": 1.0,
            "runtime_byte_identical_state_fraction": 1.0,
        },
        "target_slot_rewrite_audit": {
            "pair_contract_passed_fraction": 1.0,
            "teacher_forced_joint_exact_output_flip_fraction": 1.0,
            "greedy_joint_exact_output_flip_fraction": 1.0,
        },
        "conditions": {
            "correct": _condition_result(answer=1.0, route=1.0),
            "donor": _condition_result(answer=1.0, route=1.0),
            "value_swap": _condition_result(answer=1.0, route=1.0),
            "target_slot_rewrite": _condition_result(answer=1.0, route=1.0),
            "shuffled_slots": _condition_result(answer=1.0, route=1.0),
            "no_write": _condition_result(answer=0.0, route=None, no_write=True),
        },
    }
    training = {"router_gradient_audit": {"all_modules_finite_nonzero": True}}

    gate = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert gate["passed"] is True
    assert all(gate["criteria"].values())

    evaluation["conditions"]["value_swap"]["greedy_answer_exact_accuracy"] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert failed["criteria"]["heldout_value_swap_expected_answer_accuracy"] is False

    evaluation["conditions"]["value_swap"]["greedy_answer_exact_accuracy"] = 1.0
    evaluation["conditions"]["value_swap"]["semantic_route_accuracy"] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert failed["criteria"]["heldout_value_swap_semantic_route_accuracy"] is False

    evaluation["conditions"]["value_swap"]["semantic_route_accuracy"] = 1.0
    evaluation["conditions"]["target_slot_rewrite"][
        "greedy_answer_exact_accuracy"
    ] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert (
        failed["criteria"][
            "heldout_target_slot_rewrite_expected_answer_accuracy"
        ]
        is False
    )

    evaluation["conditions"]["target_slot_rewrite"][
        "greedy_answer_exact_accuracy"
    ] = 1.0
    evaluation["conditions"]["target_slot_rewrite"][
        "semantic_route_accuracy"
    ] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert (
        failed["criteria"][
            "heldout_target_slot_rewrite_semantic_route_accuracy"
        ]
        is False
    )

    evaluation["conditions"]["target_slot_rewrite"][
        "semantic_route_accuracy"
    ] = 1.0
    evaluation["target_slot_rewrite_audit"][
        "greedy_joint_exact_output_flip_fraction"
    ] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert (
        failed["criteria"][
            "heldout_target_slot_rewrite_joint_exact_output_flip"
        ]
        is False
    )

    evaluation["target_slot_rewrite_audit"][
        "greedy_joint_exact_output_flip_fraction"
    ] = 1.0
    evaluation["conditions"]["shuffled_slots"][
        "greedy_answer_exact_accuracy"
    ] = 0.94
    failed = runner.build_gate(
        evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=True,
    )
    assert failed["passed"] is False
    assert (
        failed["criteria"]["heldout_shuffled_slot_expected_answer_accuracy"]
        is False
    )


def test_condition_contract_binds_new_artifacts_and_allows_legacy_absence() -> None:
    contract = runner._condition_contract()

    assert contract["conditions"] == list(runner.CONDITIONS)
    assert "shuffled_slots" in contract["positive_answer_conditions"]
    assert contract["target_slot_rewrite"]["changed_write_records"] == 1
    assert contract["target_slot_rewrite"]["train_rewrite_offsets"] == list(
        canary.TRAIN_OFFSETS
    )
    assert contract["target_slot_rewrite"]["heldout_rewrite_offsets"] == list(
        canary.HELDOUT_OFFSETS
    )
    runner._validate_condition_contract_binding(None, None, None)
    runner._validate_condition_contract_binding(
        copy.deepcopy(contract),
        copy.deepcopy(contract),
        copy.deepcopy(contract),
    )

    mutated = copy.deepcopy(contract)
    mutated["target_slot_rewrite"]["target_slot_unchanged"] = False
    with pytest.raises(ValueError, match="condition contract differs"):
        runner._validate_condition_contract_binding(contract, mutated, contract)
    with pytest.raises(ValueError, match="condition contract differs"):
        runner._validate_condition_contract_binding(contract, None, contract)

    legacy = runner._legacy_revision_4_condition_contract()
    assert "shuffled_slots" not in legacy["positive_answer_conditions"]
    runner._validate_condition_contract_binding(
        copy.deepcopy(legacy),
        copy.deepcopy(legacy),
        copy.deepcopy(legacy),
        expected=legacy,
    )


def test_acceptance_contract_is_bound_to_the_canary_spec() -> None:
    contract = runner._acceptance_contract()

    runner._validate_acceptance_contract_binding(
        copy.deepcopy(contract),
        copy.deepcopy(contract),
        copy.deepcopy(contract),
    )
    relaxed = copy.deepcopy(contract)
    relaxed["heldout_answer_accuracy_min"] = 0.0
    with pytest.raises(ValueError, match="acceptance contract differs"):
        runner._validate_acceptance_contract_binding(contract, relaxed, contract)


def test_proof_provenance_binds_transitive_outer_memory_sources() -> None:
    required = {
        "deltamem/__init__.py",
        "deltamem/core/__init__.py",
        "deltamem/core/backbone_compat.py",
        "deltamem/core/delta.py",
        "deltamem/core/delta_impl.py",
        "deltamem/core/hrm_rwkv7.py",
        "deltamem/kernels/__init__.py",
        "deltamem/kernels/affine_scan.py",
    }
    bound = set(runner.PROOF_SOURCE_RELATIVE_PATHS)
    repository = Path(runner.__file__).resolve().parents[2]

    assert required <= bound
    assert "deltamem/train/delta_sft_experimental.py" not in bound
    assert all((repository / relative_path).is_file() for relative_path in bound)


def test_heldout_and_train_screen_must_share_code_provenance() -> None:
    provenance = {
        "git_commit": "a" * 40,
        "source_sha256_by_path": {"runner.py": "b" * 64},
    }

    runner._validate_linked_train_screen_provenance(
        {"code_provenance": copy.deepcopy(provenance)},
        provenance,
    )
    changed = copy.deepcopy(provenance)
    changed["git_commit"] = "c" * 40
    with pytest.raises(ValueError, match="code provenance differs"):
        runner._validate_linked_train_screen_provenance(
            {"code_provenance": changed},
            provenance,
        )


def test_recorded_model_must_match_before_after_and_source() -> None:
    model = {"path": "/model", "identity_sha256": "a" * 64}

    runner._validate_recorded_model_binding(
        {"model_before": copy.deepcopy(model), "model_after": copy.deepcopy(model)},
        model,
    )
    changed = copy.deepcopy(model)
    changed["identity_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="recorded model binding differs"):
        runner._validate_recorded_model_binding(
            {"model_before": model, "model_after": changed},
            model,
        )


def test_model_attachment_is_recomputed_from_adapter_state() -> None:
    module_name = "model.language_model.layers.0.self_attn"
    protocol = {
        "target_layers": [0],
        "delta_config": runner.build_delta_config(target_layers=(0,)).to_dict(),
    }
    adapter_state = {
        f"{module_name}.delta_scale_raw": torch.zeros(3),
        f"{module_name}.memory_q_proj": torch.zeros(5),
        f"{module_name}.projected_kv_key_proj": torch.zeros(7),
        f"{module_name}.delta_q_proj": torch.zeros(11),
        f"{module_name}.delta_k_proj": torch.zeros(13),
        f"{module_name}.memory_v_proj": torch.zeros(17),
        f"{module_name}.hrm_rwkv7_core.ln_x.bias": torch.zeros(19),
    }
    trainable_names = [
        f"{module_name}.delta_scale_raw",
        f"{module_name}.projected_kv_key_proj",
        f"{module_name}.delta_q_proj",
        f"{module_name}.memory_v_proj",
    ]
    attachment = {
        "replaced_modules": [module_name],
        "trainable_parameter_names": trainable_names,
        "checkpointed_frozen_mlps": ["model.language_model.layers.0.mlp"],
        "trainable_parameter_count": 3 + 7 + 11 + 17,
    }

    runner._validate_model_attachment_binding(
        attachment,
        protocol=protocol,
        adapter_state=adapter_state,
    )
    corrupted = copy.deepcopy(attachment)
    corrupted["trainable_parameter_names"].append(
        f"{module_name}.memory_q_proj"
    )
    with pytest.raises(
        ValueError, match="frozen-base model attachment binding differs"
    ):
        runner._validate_model_attachment_binding(
            corrupted,
            protocol=protocol,
            adapter_state=adapter_state,
        )


def _selected_protocol_fixture(*, screen: bool):
    source = {"train_rows": 384, "heldout_rows": 192}
    contract = runner._selected_protocol_contract()
    common = contract["common_configuration"]
    mode = contract["train_screen"] if screen else contract["heldout_proof"]
    seed = mode.get("seed", mode.get("seeds", [42])[0])
    train_screen_binding = None
    if not screen:
        train_screen_binding = {
            "receipt_path": "/tmp/selected-train-screen/run_receipt.json",
            "receipt_file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
            "evaluation_sha256": "3" * 64,
            "seed": runner.SELECTED_TRAIN_SCREEN_SEED,
            "current_protocol_valid": True,
            "train_screen_passed": True,
        }
    protocol = {
        "protocol_revision": runner.CURRENT_PROTOCOL_REVISION,
        "seed": seed,
        "profile": mode["profile"],
        "source": source,
        "eval_split": mode["eval_split"],
        "epochs": common["epochs"],
        "max_steps": common["max_steps"],
        "batch_size": common["batch_size"],
        "eval_batch_size": common["eval_batch_size"],
        "learning_rate": common["learning_rate"],
        "answer_weight": common["answer_weight"],
        "route_weight": common["route_weight"],
        "max_grad_norm": common["max_grad_norm"],
        "device": "cuda:3",
        "dtype": common["dtype"],
        "attn_implementation": common["attn_implementation"],
        "target_layers": common["target_layers"],
        "projected_kv_value_rank": common["projected_kv_value_rank"],
        "projected_kv_key_dim": common["projected_kv_key_dim"],
        "projected_kv_temperature": common["projected_kv_temperature"],
        "train_limit": common["train_limit"],
        "eval_limit": common["eval_limit"],
        "greedy_answer_evaluation": mode["greedy_answer_evaluation"],
        "train_screen_binding": train_screen_binding,
        "acceptance_contract": runner._acceptance_contract(),
        "condition_contract": runner._condition_contract(),
        "selected_protocol_contract": contract,
    }
    evaluation = {
        "protocol_revision": runner.CURRENT_PROTOCOL_REVISION,
        "seed": seed,
        "profile": mode["profile"],
        "eval_split": mode["eval_split"],
        "eval_rows": source[f"{mode['eval_split']}_rows"],
        "source": source,
        "acceptance_contract": canary.canary_spec()["acceptance_gate"],
        "train_screen_binding": train_screen_binding,
        "condition_contract": runner._condition_contract(),
        "selected_protocol_contract": contract,
    }
    training = {"steps": common["actual_training_steps"]}
    return protocol, evaluation, training


def test_selected_protocol_eligibility_locks_every_common_field() -> None:
    protocol, evaluation, training = _selected_protocol_fixture(screen=False)

    eligible = runner.build_protocol_eligibility(protocol, evaluation, training)

    assert eligible["acceptance_eligible"] is True
    assert eligible["train_screen_eligible"] is False
    assert runner._selected_heldout_request_matches(protocol) is True

    protocol_fields = {
        "epochs": 7,
        "max_steps": 767,
        "batch_size": 8,
        "eval_batch_size": 4,
        "learning_rate": 1e-4,
        "answer_weight": 0.5,
        "route_weight": 0.5,
        "max_grad_norm": 0.5,
        "device": "cpu",
        "dtype": "float16",
        "attn_implementation": "eager",
        "target_layers": [0],
        "projected_kv_value_rank": 16,
        "projected_kv_key_dim": 16,
        "projected_kv_temperature": 8.0,
        "train_limit": 4,
        "eval_limit": 4,
    }
    for field, invalid in protocol_fields.items():
        mutated = copy.deepcopy(protocol)
        mutated[field] = invalid
        result = runner.build_protocol_eligibility(mutated, evaluation, training)
        assert result["acceptance_eligible"] is False, field
        assert result["selected_common_configuration_matches"] is False, field
        assert runner._selected_heldout_request_matches(mutated) is False, field

    short_training = {"steps": runner.SELECTED_PROOF_MAX_STEPS - 1}
    result = runner.build_protocol_eligibility(
        protocol,
        evaluation,
        short_training,
    )
    assert result["acceptance_eligible"] is False
    assert result["selected_common_configuration_matches"] is False

    missing_screen = copy.deepcopy(protocol)
    missing_screen["train_screen_binding"] = None
    result = runner.build_protocol_eligibility(
        missing_screen,
        evaluation,
        training,
    )
    assert result["acceptance_eligible"] is False
    assert runner._selected_heldout_request_matches(missing_screen) is False

    relaxed_acceptance = copy.deepcopy(protocol)
    relaxed_acceptance["acceptance_contract"]["heldout_answer_accuracy_min"] = 0.0
    result = runner.build_protocol_eligibility(
        relaxed_acceptance,
        evaluation,
        training,
    )
    assert result["acceptance_eligible"] is False
    assert result["current_contracts_present"] is False
    assert runner._selected_heldout_request_matches(relaxed_acceptance) is False


def test_selected_train_screen_has_a_first_class_teacher_forced_gate() -> None:
    protocol, evaluation, training = _selected_protocol_fixture(screen=True)
    metric_evaluation = {
        **evaluation,
        "query_counterfactual_audit": {
            "query_counterfactual_route_accuracy": 1.0,
            "runtime_byte_identical_state_fraction": 1.0,
        },
        "target_slot_rewrite_audit": {
            "pair_contract_passed_fraction": 1.0,
            "teacher_forced_joint_exact_output_flip_fraction": 1.0,
            "greedy_joint_exact_output_flip_fraction": None,
        },
        "conditions": {
            "correct": _condition_result(answer=1.0, route=1.0),
            "donor": _condition_result(answer=1.0, route=1.0),
            "value_swap": _condition_result(answer=1.0, route=1.0),
            "target_slot_rewrite": _condition_result(answer=1.0, route=1.0),
            "shuffled_slots": _condition_result(answer=1.0, route=1.0),
            "no_write": _condition_result(answer=0.0, route=None, no_write=True),
        },
    }
    training["router_gradient_audit"] = {"all_modules_finite_nonzero": True}

    metric_gate = runner.build_gate(
        metric_evaluation,
        training=training,
        split_audit_passed=True,
        input_immutability_passed=True,
        require_greedy=False,
    )
    eligibility = runner.build_protocol_eligibility(
        protocol,
        metric_evaluation,
        training,
    )
    gate = runner.finalize_gate(metric_gate, eligibility)

    assert gate["answer_metric"] == "teacher_forced_whole_answer_exact"
    assert gate["metric_gate_passed"] is True
    assert gate["train_screen_eligible"] is True
    assert gate["train_screen_passed"] is True
    assert gate["acceptance_eligible"] is False
    assert gate["passed"] is False


def test_selected_protocol_contract_binding_requires_all_current_copies() -> None:
    contract = runner._selected_protocol_contract()

    assert (
        runner._validate_selected_protocol_contract_binding(None, None, None)
        is False
    )
    assert (
        runner._validate_selected_protocol_contract_binding(
            copy.deepcopy(contract),
            copy.deepcopy(contract),
            copy.deepcopy(contract),
        )
        is True
    )
    mutated = copy.deepcopy(contract)
    mutated["common_configuration"]["max_steps"] = 384
    with pytest.raises(ValueError, match="selected proof protocol contract differs"):
        runner._validate_selected_protocol_contract_binding(
            contract,
            mutated,
            contract,
        )


def test_training_progress_binding_checks_hash_count_and_step_sequence(
    tmp_path,
) -> None:
    progress_path = tmp_path / "training_progress.jsonl"
    records = [
        {
            "schema": "rwkv_ms_synthetic_compositional_train_step.v3",
            "step": step,
        }
        for step in range(1, 4)
    ]
    progress_path.write_text(
        "".join(
            f"{canary.canonical_json_bytes(record).decode('ascii')}\n"
            for record in records
        ),
        encoding="ascii",
    )
    receipt = {
        "training_progress_path": str(progress_path),
        "training_progress_file_sha256": canary.sha256_file(progress_path),
    }

    runner._validate_training_progress_binding(receipt, {"steps": 3})
    with pytest.raises(ValueError, match="step binding differs"):
        runner._validate_training_progress_binding(receipt, {"steps": 2})


def test_signed_payload_rejects_mutation() -> None:
    signed = runner._signed_payload({"schema": "fixture", "value": 3}, "sha256")

    assert runner._validate_signed_payload(
        signed, hash_field="sha256", description="fixture"
    ) == signed["sha256"]
    signed["value"] = 4
    with pytest.raises(ValueError, match="canonical SHA-256 differs"):
        runner._validate_signed_payload(
            signed, hash_field="sha256", description="fixture"
        )


def test_proof_set_requires_all_three_common_bound_seed_receipts(
    tmp_path,
    monkeypatch,
) -> None:
    code_provenance = {
        "git_commit": "a" * 40,
        "source_sha256_by_path": {},
    }
    train_screen_binding = {
        "receipt_path": "/proof/train/run_receipt.json",
        "receipt_file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "evaluation_sha256": "3" * 64,
        "seed": runner.SELECTED_TRAIN_SCREEN_SEED,
        "current_protocol_valid": True,
        "train_screen_passed": True,
    }
    receipt_paths = []
    for seed in (42, 43, 44):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "protocol_revision": runner.CURRENT_PROTOCOL_REVISION,
                    "source_before": {"manifest_sha256": "4" * 64},
                    "model_before": {"identity_sha256": "5" * 64},
                    "train_screen_binding": train_screen_binding,
                    "code_provenance": code_provenance,
                }
            ),
            encoding="utf-8",
        )
        receipt_paths.append(path)

    def fake_validate(path, **_kwargs):
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = receipt["seed"]
        return {
            "valid": True,
            "receipt_path": str(Path(path).resolve()),
            "receipt_file_sha256": str(seed) * 32,
            "receipt_sha256": str(seed + 1) * 32,
            "evaluation_sha256": str(seed + 2) * 32,
            "seed": seed,
            "current_protocol_valid": True,
            "gate": {
                "passed": True,
                "acceptance_eligible": True,
                "required_seed_passes": 3,
                "answer_metric": "greedy_whole_answer_exact",
            },
        }

    monkeypatch.setattr(runner, "validate_receipt", fake_validate)
    monkeypatch.setattr(runner, "_validate_code_provenance", lambda _binding: None)
    monkeypatch.setattr(
        runner,
        "_capture_code_provenance",
        lambda: copy.deepcopy(code_provenance),
    )

    output = tmp_path / "proof-set.json"
    created = runner.create_proof_set(
        receipt_paths,
        output_path=output,
        source_manifest=tmp_path / "source.json",
        model_path=tmp_path / "model",
    )
    validated = runner.validate_proof_set(
        output,
        source_manifest=tmp_path / "source.json",
        model_path=tmp_path / "model",
        verify_model_hashes=True,
    )

    assert created["aggregate_passed"] is True
    assert validated["aggregate_passed"] is True
    assert validated["required_seeds"] == [42, 43, 44]

    receipt_paths[-1].write_text(
        receipt_paths[-1].read_text(encoding="utf-8").replace('"seed": 44', '"seed": 45'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact required seeds"):
        runner._proof_set_payload(
            receipt_paths,
            source_manifest=tmp_path / "source.json",
            model_path=tmp_path / "model",
            verify_model_hashes=True,
            require_current_checkout=False,
        )
