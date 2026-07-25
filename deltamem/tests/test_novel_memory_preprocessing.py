from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deltamem.tools.prepare_novel_memory_dataset import (
    DEFAULT_BREAK_QUERY,
    SHORT_SKIP_REASON,
    preprocess_dataset,
)


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict:
        assert add_special_tokens is False
        input_ids = [ord(character) for character in text]
        if text.startswith("!") and not return_offsets_mapping:
            input_ids.insert(0, 999_999)
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return [ord(character) for character in rendered]


def make_row(
    *,
    system: str = "system",
    user: str = "prompt",
    assistant: str = "a" * 70,
    roles: tuple[str, str, str] = ("system", "user", "assistant"),
) -> dict:
    return {
        "messages": [
            {"role": roles[0], "content": system},
            {"role": roles[1], "content": user},
            {"role": roles[2], "content": assistant},
        ]
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_preprocess(input_paths: list[Path], output_path: Path) -> dict:
    return preprocess_dataset(
        input_paths=input_paths,
        tokenizer=FakeTokenizer(),
        model_path="fake-tokenizer",
        output_path=output_path,
    )


def test_dedupe_preserves_first_source_and_output_is_deterministic(tmp_path: Path) -> None:
    first_input = tmp_path / "first.jsonl"
    second_input = tmp_path / "second.jsonl"
    write_jsonl(
        first_input,
        [
            make_row(
                system="first-system",
                user="first header\nBody with spaces",
                assistant="x" * 70,
            )
        ],
    )
    write_jsonl(
        second_input,
        [
            make_row(
                system="duplicate-system",
                user="different header\nBodywithspaces",
                assistant=" ".join("x" * 70),
            ),
            make_row(system="unique-system", user="unique", assistant="y" * 70),
        ],
    )

    first_output = tmp_path / "first-output.jsonl"
    second_output = tmp_path / "second-output.jsonl"
    first_summary = run_preprocess([first_input, second_input], first_output)
    second_summary = run_preprocess([first_input, second_input], second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_summary["output"]["sha256"] == second_summary["output"]["sha256"]
    assert first_summary["counts"] == {
        "input_rows": 3,
        "dedupe_winners": 2,
        "duplicates": 1,
        "emitted": 2,
        "skipped": {},
    }
    assert [source["dedupe_winners"] for source in first_summary["inputs"]] == [1, 1]
    assert [source["duplicates"] for source in first_summary["inputs"]] == [0, 1]
    output_rows = load_jsonl(first_output)
    assert output_rows[0]["messages"][0]["content"] == "first-system"
    assert output_rows[1]["messages"][0]["content"] == "unique-system"
    assert hashlib.sha256(first_output.read_bytes()).hexdigest() == first_summary["output"][
        "sha256"
    ]


def test_split_retokenizes_suffix_until_it_meets_target_cap(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    assistant = "a" * 33 + "!" + "b" * 111
    write_jsonl(input_path, [make_row(assistant=assistant)])
    output_path = tmp_path / "output.jsonl"

    summary = run_preprocess([input_path], output_path)

    output_row = load_jsonl(output_path)[0]
    messages = output_row["messages"]
    assert messages[2]["content"] == "a" * 33 + "!"
    assert messages[4]["content"] == "b" * 111
    assert output_row["memory_preprocessing"]["cut_adjustment_tokens"] == 1
    assert output_row["memory_preprocessing"]["target_tokens"] == 111
    assert summary["token_stats"]["target_suffix"]["max"] == 111


def test_short_assistant_is_skipped_with_counted_reason(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, [make_row(assistant="s" * 63)])
    output_path = tmp_path / "output.jsonl"

    summary = run_preprocess([input_path], output_path)

    assert output_path.read_text(encoding="utf-8") == ""
    assert summary["counts"]["emitted"] == 0
    assert summary["counts"]["skipped"] == {SHORT_SKIP_REASON: 1}
    assert summary["inputs"][0]["skipped"] == {SHORT_SKIP_REASON: 1}


def test_invalid_roles_fail_without_replacing_existing_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(
        input_path,
        [make_row(roles=("system", "assistant", "user"))],
    )
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected exact roles"):
        run_preprocess([input_path], output_path)

    assert output_path.read_text(encoding="utf-8") == "existing\n"
    assert not Path(str(output_path) + ".summary.json").exists()


def test_output_roles_and_break_query_form_one_memory_episode(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    original_system = "You write fiction."
    original_user = "Continue this chapter."
    write_jsonl(
        input_path,
        [make_row(system=original_system, user=original_user, assistant="z" * 80)],
    )
    output_path = tmp_path / "output.jsonl"

    summary = run_preprocess([input_path], output_path)

    messages = load_jsonl(output_path)[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == original_system
    assert messages[1]["content"] == original_user
    assert messages[3]["content"] == DEFAULT_BREAK_QUERY
    assert messages[2]["content"] + messages[4]["content"] == "z" * 80
    assert summary["token_stats"]["full_write"]["count"] == 1
    assert summary["token_stats"]["full_read"]["count"] == 1
