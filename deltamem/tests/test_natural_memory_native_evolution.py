from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_gate as gate,
)
from experiments.rethinking_rwkv_ms_gemma import (
    run_natural_memory_native_evolution as evolution,
)


class PrefixStableTokenizer:
    pad_token_id = 0

    @staticmethod
    def _render(
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        **_kwargs,
    ):
        rendered = self._render(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return rendered
        token_ids = [ord(character) for character in rendered]
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        return token_ids

    def __call__(self, rendered: str, **_kwargs):
        return {"input_ids": [ord(character) for character in rendered]}


def native_raw_line() -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Classify this passage."},
                {"role": "assistant", "content": '{"label":"dream"}'},
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_native_encoding_preserves_exact_full_row_and_target() -> None:
    tokenizer = PrefixStableTokenizer()
    raw_line = native_raw_line()

    example = evolution.encode_native_full_row(
        tokenizer,
        task="narrative",
        source_ordinal=7,
        raw_line=raw_line,
    )

    messages = json.loads(raw_line)["messages"]
    expected_full = tuple(
        ord(character)
        for character in tokenizer._render(
            messages,
            add_generation_prompt=False,
        )
    )
    assert example.read_input_ids == expected_full
    assert example.read_attention_mask == (1,) * len(expected_full)
    assert example.labels[: len(example.write_input_ids)] == (
        -100,
    ) * len(example.write_input_ids)
    assert example.labels[len(example.write_input_ids) :] == expected_full[
        len(example.write_input_ids) :
    ]
    assert example.assistant_target_tokens == (
        len(expected_full) - len(example.write_input_ids)
    )


def test_native_write_prefix_excludes_assistant_generation_marker() -> None:
    tokenizer = PrefixStableTokenizer()
    raw_line = native_raw_line()
    messages = json.loads(raw_line)["messages"]

    example = evolution.encode_native_full_row(
        tokenizer,
        task="narrative",
        source_ordinal=0,
        raw_line=raw_line,
    )

    expected_write = tokenizer._render(
        messages[:-1],
        add_generation_prompt=False,
    )
    generation_prompt = tokenizer._render(
        messages[:-1],
        add_generation_prompt=True,
    )
    assert example.write_input_ids == tuple(map(ord, expected_write))
    assert generation_prompt == expected_write + "<assistant>"
    assert tuple(map(ord, generation_prompt)) != example.write_input_ids


def _synthetic_examples(families: int = 384) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            row_id=f"synthetic:{family}:{member}",
            condition="correct_state",
            episode_id=f"episode:{family}",
            semantic_target_slot=member,
        )
        for family in range(families)
        for member in range(4)
    ]


def _native_examples() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(row_id=f"native:{task}:{index}", task=task)
        for task in sorted(evolution.TASK_FILES)
        for index in range(evolution.GLOBAL_BATCH_SIZE)
    ]


def test_stage1_schedule_is_strictly_alternating_and_balanced() -> None:
    schedule, audit = evolution.build_mixed_schedule(
        _synthetic_examples(),
        _native_examples(),
        total_updates=evolution.STAGE1_UPDATES,
    )

    assert len(schedule) == 192
    assert audit["synthetic_updates"] == 96
    assert audit["native_updates"] == 96
    assert audit["alternation"] == "odd_synthetic_even_native"
    assert audit["native_task_updates"] == {
        "attribution": 32,
        "narrative": 32,
        "scene": 32,
    }
    assert all(
        step.update_kind == ("synthetic" if step.step % 2 else "native")
        for step in schedule
    )


def test_native_update_has_zero_route_denominator() -> None:
    batches = [
        SimpleNamespace(
            labels=torch.tensor(
                [[-100, -100, 11], [-100, 12, 13]],
                dtype=torch.long,
            )
        ),
        SimpleNamespace(
            labels=torch.tensor(
                [[-100, 14], [-100, 15]],
                dtype=torch.long,
            )
        ),
    ]

    answer_tokens, route_rows = evolution.local_objective_denominators(
        "native",
        batches,
    )

    assert answer_tokens == 5
    assert route_rows == 0


def test_native_execution_serializes_each_logical_microbatch() -> None:
    assert evolution.LOCAL_MICROBATCH_SIZE == 2
    assert evolution.execution_subbatch_size("synthetic") == 2
    assert evolution.execution_subbatch_size("native") == 1
    assert (
        evolution.LOCAL_BATCH_SIZE
        // evolution.execution_subbatch_size("native")
        == 4
    )


def test_r12_warm_start_adapter_aggregate_hash_is_bound() -> None:
    adapter_files = gate.snapshot_directory_files(evolution.R12_ADAPTER)

    assert gate._sha256_json(adapter_files) == evolution.R12_ADAPTER_FILES_SHA256
