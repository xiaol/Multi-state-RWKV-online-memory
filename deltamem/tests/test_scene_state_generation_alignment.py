from __future__ import annotations

import pytest
import torch

from deltamem.train.scene_state_generation_alignment import (
    align_generated_token_ids,
    clone_detached_online_state,
    generated_unlikelihood_positions,
)


def test_exact_alignment_selects_no_generated_tokens() -> None:
    alignment = align_generated_token_ids([1, 2, 3], [1, 2, 3])

    assert alignment.first_divergence == 3
    assert alignment.edit_distance == 0
    assert alignment.wrong_generated_positions == ()
    assert [edit.kind for edit in alignment.edits] == ["match", "match", "match"]


def test_substitution_selects_only_the_replaced_generated_token() -> None:
    alignment = align_generated_token_ids([1, 9, 3], [1, 2, 3])

    assert alignment.first_divergence == 1
    assert alignment.edit_distance == 1
    assert alignment.wrong_generated_positions == (1,)
    assert [edit.kind for edit in alignment.edits] == [
        "match",
        "substitution",
        "match",
    ]


def test_generated_insertion_resynchronizes_without_penalizing_shifted_matches() -> None:
    alignment = align_generated_token_ids([1, 9, 2, 3, 4], [1, 2, 3, 4])

    assert alignment.first_divergence == 1
    assert alignment.edit_distance == 1
    assert alignment.wrong_generated_positions == (1,)
    assert [edit.kind for edit in alignment.edits] == [
        "match",
        "generated_insertion",
        "match",
        "match",
        "match",
    ]


def test_gold_deletion_selects_the_next_generated_token_as_negative_anchor() -> None:
    alignment = align_generated_token_ids([1, 3, 4], [1, 2, 3, 4])

    assert alignment.first_divergence == 1
    assert alignment.edit_distance == 1
    assert alignment.wrong_generated_positions == (1,)
    assert [edit.kind for edit in alignment.edits] == [
        "match",
        "gold_deletion",
        "match",
        "match",
    ]


def test_premature_termination_is_the_negative_anchor_for_missing_gold() -> None:
    end_token_id = 99
    alignment = align_generated_token_ids(
        [1, end_token_id],
        [1, 2, end_token_id],
    )

    assert alignment.edit_distance == 1
    assert alignment.wrong_generated_positions == (1,)
    assert alignment.edits[1].kind == "gold_deletion"


def test_terminal_gold_deletion_without_generated_anchor_selects_nothing() -> None:
    alignment = align_generated_token_ids([1, 2], [1, 2, 3])

    assert alignment.first_divergence == 2
    assert alignment.edit_distance == 1
    assert alignment.wrong_generated_positions == ()
    assert alignment.edits[-1].kind == "gold_deletion"
    assert alignment.edits[-1].generated_index is None


def test_alignment_is_deterministic_for_repeated_tokens() -> None:
    first = align_generated_token_ids([7, 7], [7])
    second = align_generated_token_ids([7, 7], [7])

    assert first == second
    assert first.wrong_generated_positions == (1,)
    assert [edit.kind for edit in first.edits] == ["match", "generated_insertion"]


def test_equal_distance_alignment_maximizes_resynchronized_matches() -> None:
    alignment = align_generated_token_ids([1, 3, 2, 4], [1, 2, 3, 4])

    assert alignment.edit_distance == 2
    assert alignment.wrong_generated_positions == (1, 3)
    assert [edit.kind for edit in alignment.edits] == [
        "match",
        "generated_insertion",
        "match",
        "gold_deletion",
        "match",
    ]


def test_torch_wrapper_preserves_device_and_caps_edit_aligned_positions() -> None:
    generated = torch.tensor([1, 9, 2, 8, 3, 7, 4], dtype=torch.long)
    gold = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    first_divergence, positions = generated_unlikelihood_positions(
        generated,
        gold,
        max_wrong_tokens=2,
    )

    assert first_divergence == 1
    assert positions.tolist() == [1, 3]
    assert positions.device == generated.device
    assert positions.dtype == torch.long


@pytest.mark.parametrize(
    ("generated", "gold", "message"),
    [
        (torch.ones(1, 2, dtype=torch.long), torch.ones(2, dtype=torch.long), "one-dimensional"),
        (torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.long), "integer dtype"),
    ],
)
def test_alignment_rejects_invalid_token_tensors(
    generated: torch.Tensor,
    gold: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        align_generated_token_ids(generated, gold)


def test_generated_position_wrapper_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        generated_unlikelihood_positions(
            torch.tensor([1]),
            torch.tensor([2]),
            max_wrong_tokens=0,
        )


def test_detached_online_state_snapshot_is_graph_free_and_storage_independent() -> None:
    base = torch.tensor([1.0, 2.0], requires_grad=True)
    live_state = {
        "layer.delta_state": base.square(),
        "layer.__rwkv_ms_positions": torch.tensor([7], dtype=torch.long),
        "layer.__rwkv_ms_previous_source": base * 3.0,
    }

    snapshot = clone_detached_online_state(live_state)

    assert list(snapshot) == list(live_state)
    for name, tensor in snapshot.items():
        assert torch.equal(tensor, live_state[name])
        assert tensor.device == live_state[name].device
        assert tensor.dtype == live_state[name].dtype
        assert tensor.requires_grad is False
        assert tensor.grad_fn is None
        assert tensor.data_ptr() != live_state[name].data_ptr()

    live_state["layer.delta_state"].add_(10.0)
    assert torch.equal(snapshot["layer.delta_state"], torch.tensor([1.0, 4.0]))


def test_detached_online_state_snapshot_rejects_empty_or_nontensor_state() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        clone_detached_online_state({})
    with pytest.raises(ValueError, match="not a tensor"):
        clone_detached_online_state({"state": object()})
