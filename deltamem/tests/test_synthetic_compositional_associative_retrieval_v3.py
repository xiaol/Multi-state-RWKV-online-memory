from __future__ import annotations

import copy
import os

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
def heldout_rows(tokenizer):
    return canary.build_partition_rows(tokenizer, "heldout")


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


def test_examples_bind_correct_donor_swap_shuffle_and_no_write(
    tokenizer,
    heldout_rows,
) -> None:
    row = heldout_rows[0]
    correct = runner.correct_example(row)
    donor = runner.donor_example(row, heldout_rows)
    swapped = runner.value_swap_example(row, tokenizer)
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

    donor_row = heldout_rows[row["donor"]["row_ordinal"]]
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

    assert shuffled.condition == "shuffled_slots"
    assert shuffled.write_slots == (2, 0, 3, 1)
    assert shuffled.target_slot == shuffled.write_slots[target_slot]
    assert shuffled.expected_value == correct.expected_value

    assert no_write.condition == "no_write"
    assert no_write.write_records == no_write.write_slots == ()
    assert no_write.target_slot is None


def test_condition_builder_uses_full_partition_for_donor_lookup(
    tokenizer,
    heldout_rows,
) -> None:
    selected = runner.select_complete_memory_states(heldout_rows, 4)
    examples = runner.build_condition_examples(
        selected,
        tokenizer,
        "donor",
        all_rows=heldout_rows,
    )

    assert len(examples) == 4
    for source, example in zip(selected, examples, strict=True):
        donor = heldout_rows[source["donor"]["row_ordinal"]]
        assert example.expected_value == donor["query"]["target_value"]


def test_select_complete_memory_states_never_splits_query_family(heldout_rows) -> None:
    selected = runner.select_complete_memory_states(heldout_rows, 7)

    assert len(selected) == 4
    assert len({row["memory_state_id"] for row in selected}) == 1
    assert sorted(row["query_route_target_slot"] for row in selected) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="smaller than one complete"):
        runner.select_complete_memory_states(heldout_rows, 3)


def test_collator_preserves_record_major_masks_and_answer_labels(heldout_rows) -> None:
    examples = [runner.correct_example(row) for row in heldout_rows[:2]]
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


def test_no_write_batch_has_no_records_and_absent_targets(heldout_rows) -> None:
    batch = runner.collate_examples(
        [runner.no_write_example(heldout_rows[0])],
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


def test_query_counterfactual_audit_requires_all_four_routes_and_identical_state(
    heldout_rows,
) -> None:
    examples = [runner.correct_example(row) for row in heldout_rows[:4]]
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
        "conditions": {
            "correct": _condition_result(answer=1.0, route=1.0),
            "donor": _condition_result(answer=1.0, route=1.0),
            "value_swap": _condition_result(answer=1.0, route=1.0),
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
