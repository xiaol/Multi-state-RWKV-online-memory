from __future__ import annotations

import copy

import pytest

from experiments.rethinking_rwkv_ms_gemma.make_content_control_probe import (
    build_content_control_rows,
)


def _row(index: int) -> dict:
    return {
        "source": index,
        "messages": [
            {"role": "system", "content": f"style {index}"},
            {"role": "user", "content": f"story {index}"},
            {"role": "assistant", "content": f"history {index}"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": f"target {index}"},
        ],
    }


def test_content_control_probe_moves_visible_identity_into_write_history() -> None:
    source = [_row(0), _row(1)]
    original = copy.deepcopy(source)

    transformed = build_content_control_rows(
        source,
        source_sha256="abc123",
        system_prompt="same system",
        user_prompt="same user",
    )

    assert source == original
    assert {row["messages"][0]["content"] for row in transformed} == {"same system"}
    assert {row["messages"][-2]["content"] for row in transformed} == {"same user"}
    assert transformed[0]["messages"][1]["content"] == "style 0\n\nstory 0"
    assert transformed[1]["messages"][1]["content"] == "style 1\n\nstory 1"
    assert transformed[0]["messages"][-1]["content"] == "target 0"
    assert transformed[0]["content_control_probe"]["source_row_index"] == 0
    assert transformed[1]["content_control_probe"]["source_row_index"] == 1


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(messages=row["messages"][:4]), "at least five messages"),
        (lambda row: row["messages"][0].update(role="user"), "First message"),
        (lambda row: row["messages"][-1].update(content=""), "target is empty"),
    ],
)
def test_content_control_probe_rejects_invalid_episode_rows(mutator, message: str) -> None:
    rows = [_row(0), _row(1)]
    mutator(rows[0])

    with pytest.raises(ValueError, match=message):
        build_content_control_rows(
            rows,
            source_sha256="abc123",
            system_prompt="same system",
            user_prompt="same user",
        )
