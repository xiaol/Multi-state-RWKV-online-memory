from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch


TokenEditKind = Literal[
    "match",
    "substitution",
    "generated_insertion",
    "gold_deletion",
]


@dataclass(frozen=True)
class TokenEdit:
    kind: TokenEditKind
    generated_index: int | None
    gold_index: int | None


@dataclass(frozen=True)
class GeneratedTokenAlignment:
    first_divergence: int
    edit_distance: int
    edits: tuple[TokenEdit, ...]
    wrong_generated_positions: tuple[int, ...]


def _token_list(values: Sequence[int] | torch.Tensor, *, description: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{description} token IDs must be one-dimensional")
        if values.dtype == torch.bool or torch.is_floating_point(values) or values.is_complex():
            raise ValueError(f"{description} token IDs must use an integer dtype")
        return [int(value) for value in values.detach().cpu().tolist()]
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{description} token IDs must be integers")
        result.append(value)
    return result


def _alignment_tables(
    generated_token_ids: Sequence[int],
    gold_token_ids: Sequence[int],
) -> tuple[list[list[tuple[int, int]]], list[list[TokenEditKind | None]]]:
    generated_count = len(generated_token_ids)
    gold_count = len(gold_token_ids)
    scores = [
        [(0, 0) for _ in range(gold_count + 1)]
        for _ in range(generated_count + 1)
    ]
    operations: list[list[TokenEditKind | None]] = [
        [None for _ in range(gold_count + 1)]
        for _ in range(generated_count + 1)
    ]
    for generated_index in range(generated_count - 1, -1, -1):
        scores[generated_index][gold_count] = (
            generated_count - generated_index,
            0,
        )
        operations[generated_index][gold_count] = "generated_insertion"
    for gold_index in range(gold_count - 1, -1, -1):
        scores[generated_count][gold_index] = (gold_count - gold_index, 0)
        operations[generated_count][gold_index] = "gold_deletion"
    operation_priority: dict[TokenEditKind, int] = {
        "match": 0,
        "substitution": 1,
        "generated_insertion": 2,
        "gold_deletion": 3,
    }
    for generated_index in range(generated_count - 1, -1, -1):
        for gold_index in range(gold_count - 1, -1, -1):
            candidates: list[tuple[TokenEditKind, tuple[int, int]]] = []
            if generated_token_ids[generated_index] == gold_token_ids[gold_index]:
                next_edits, next_matches = scores[generated_index + 1][
                    gold_index + 1
                ]
                candidates.append(("match", (next_edits, next_matches + 1)))
            else:
                next_edits, next_matches = scores[generated_index + 1][
                    gold_index + 1
                ]
                candidates.append(
                    ("substitution", (next_edits + 1, next_matches))
                )
            insertion_edits, insertion_matches = scores[generated_index + 1][
                gold_index
            ]
            candidates.append(
                (
                    "generated_insertion",
                    (insertion_edits + 1, insertion_matches),
                )
            )
            deletion_edits, deletion_matches = scores[generated_index][gold_index + 1]
            candidates.append(
                ("gold_deletion", (deletion_edits + 1, deletion_matches))
            )
            selected_kind, selected_score = min(
                candidates,
                key=lambda candidate: (
                    candidate[1][0],
                    -candidate[1][1],
                    operation_priority[candidate[0]],
                ),
            )
            scores[generated_index][gold_index] = selected_score
            operations[generated_index][gold_index] = selected_kind
    return scores, operations


def align_generated_token_ids(
    generated_token_ids: Sequence[int] | torch.Tensor,
    gold_token_ids: Sequence[int] | torch.Tensor,
) -> GeneratedTokenAlignment:
    generated = _token_list(generated_token_ids, description="Generated")
    gold = _token_list(gold_token_ids, description="Gold")
    scores, operations = _alignment_tables(generated, gold)
    generated_count = len(generated)
    gold_count = len(gold)
    common_count = min(generated_count, gold_count)
    first_divergence = common_count
    for position in range(common_count):
        if generated[position] != gold[position]:
            first_divergence = position
            break

    edits: list[TokenEdit] = []
    generated_index = 0
    gold_index = 0
    while generated_index < generated_count or gold_index < gold_count:
        selected_kind = operations[generated_index][gold_index]
        if selected_kind == "gold_deletion":
            edits.append(
                TokenEdit(
                    kind="gold_deletion",
                    generated_index=(
                        generated_index
                        if generated_index < generated_count
                        else None
                    ),
                    gold_index=gold_index,
                )
            )
            gold_index += 1
            continue
        if selected_kind == "generated_insertion":
            edits.append(
                TokenEdit(
                    kind="generated_insertion",
                    generated_index=generated_index,
                    gold_index=None,
                )
            )
            generated_index += 1
            continue
        if selected_kind == "match":
            edits.append(
                TokenEdit(
                    kind="match",
                    generated_index=generated_index,
                    gold_index=gold_index,
                )
            )
            generated_index += 1
            gold_index += 1
            continue
        if selected_kind == "substitution":
            edits.append(
                TokenEdit(
                    kind=selected_kind,
                    generated_index=generated_index,
                    gold_index=gold_index,
                )
            )
            generated_index += 1
            gold_index += 1
        else:
            raise RuntimeError("Token edit alignment did not select an operation")

    wrong_positions: list[int] = []
    seen_positions: set[int] = set()
    for edit in edits:
        if edit.kind == "match" or edit.generated_index is None:
            continue
        if edit.generated_index in seen_positions:
            continue
        seen_positions.add(edit.generated_index)
        wrong_positions.append(edit.generated_index)
    return GeneratedTokenAlignment(
        first_divergence=first_divergence,
        edit_distance=scores[0][0][0],
        edits=tuple(edits),
        wrong_generated_positions=tuple(wrong_positions),
    )


def generated_unlikelihood_positions(
    generated_token_ids: torch.Tensor,
    gold_token_ids: torch.Tensor,
    *,
    max_wrong_tokens: int,
) -> tuple[int, torch.Tensor]:
    if max_wrong_tokens <= 0:
        raise ValueError("Generated-prefix wrong-token limit must be positive")
    alignment = align_generated_token_ids(generated_token_ids, gold_token_ids)
    selected_positions = alignment.wrong_generated_positions[:max_wrong_tokens]
    return (
        alignment.first_divergence,
        torch.tensor(
            selected_positions,
            device=generated_token_ids.device,
            dtype=torch.long,
        ),
    )


def clone_detached_online_state(
    online_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not online_state:
        raise ValueError("Online-state snapshot must not be empty")
    snapshot: dict[str, torch.Tensor] = {}
    for name, tensor in online_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Online-state names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Online-state entry is not a tensor: {name}")
        snapshot[name] = tensor.detach().clone()
    return snapshot


__all__ = [
    "GeneratedTokenAlignment",
    "TokenEdit",
    "align_generated_token_ids",
    "clone_detached_online_state",
    "generated_unlikelihood_positions",
]
