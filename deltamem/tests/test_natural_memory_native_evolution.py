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


def test_checkpointed_native_ce_matches_full_ce_and_gradient() -> None:
    labels = torch.tensor(
        [[-100, 2, 3, 4, -100, 6, 7]],
        dtype=torch.long,
    )
    full_logits = torch.randn(1, 5, 17, dtype=torch.float32, requires_grad=True)
    chunked_logits = full_logits.detach().clone().requires_grad_(True)

    full_loss, full_count = evolution.distributed.answer_loss_sum_and_count(
        full_logits,
        labels,
    )
    chunked_loss, chunked_count, chunks = (
        evolution.checkpointed_native_answer_loss_sum_and_count(
            chunked_logits,
            labels,
            chunk_tokens=2,
        )
    )
    full_loss.backward()
    chunked_loss.backward()

    assert chunks == 3
    assert chunked_count == full_count == 5
    torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)
    assert torch.equal(chunked_logits.grad, full_logits.grad)


def test_r12_warm_start_adapter_aggregate_hash_is_bound() -> None:
    adapter_files = gate.snapshot_directory_files(evolution.R12_ADAPTER)

    assert gate._sha256_json(adapter_files) == evolution.R12_ADAPTER_FILES_SHA256


def test_residual_hybrid_topology_is_bounded_and_signed() -> None:
    source_config = evolution.build_evolution_delta_config("attention_output")
    hybrid_config = evolution.build_evolution_delta_config(
        "post_attention_residual_hybrid"
    )
    protocol = evolution.load_evolution_protocol(
        "post_attention_residual_hybrid"
    )

    assert source_config.memory_fusion_placement == "attention_output"
    assert hybrid_config.memory_fusion_placement == (
        "post_attention_residual_hybrid"
    )
    assert hybrid_config.memory_fusion_residual_scale == 0.01
    assert hybrid_config.memory_fusion_residual_scale_max == 0.02
    assert protocol["topology_change"]["new_parameter_initial_value"] == 0.01
    assert protocol["topology_change"]["new_parameter_hard_max"] == 0.02


def test_content_gate_topology_is_initialized_and_signed() -> None:
    config = evolution.build_evolution_delta_config(
        "content_gated_attention_output"
    )
    protocol = evolution.load_evolution_protocol(
        "content_gated_attention_output"
    )

    assert config.memory_fusion_placement == "attention_output"
    assert config.memory_fusion_mode == "content_gated_add"
    assert config.memory_fusion_gate_init == 0.1
    assert protocol["topology_change"]["gate_initial_value"] == 0.1
    assert protocol["topology_change"]["q_head_preserved"] is True
    assert protocol["topology_change"]["o_head_preserved"] is True
    assert protocol["execution_change"][
        "content_gate_activation_checkpointing"
    ] == "torch_non_reentrant_recompute_during_backward"
    assert protocol["execution_change"]["objective_change"] is False
    assert protocol["execution_change"]["serialized_row_graph_release"] == (
        "clear_graph_references_immediately_after_backward_and_metrics"
    )
    assert protocol["execution_change"]["cuda_allocator"] == (
        "expandable_segments:True"
    )


def test_content_gate_gradient_audit_requires_every_family() -> None:
    named_trainable = []
    for family in evolution.CONTENT_GATE_PARAMETER_FAMILIES:
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        parameter.grad = torch.full_like(parameter, 0.25)
        named_trainable.append((f"layer.0.{family}", parameter))

    passing = evolution.audit_content_gate_gradients(named_trainable)
    assert passing["passed"] is True
    assert passing["parameter_tensors"] == 3
    assert passing["minimum_family_l2_norm"] > 0.0

    named_trainable[-1][1].grad.zero_()
    failing = evolution.audit_content_gate_gradients(named_trainable)
    assert failing["passed"] is False
    assert failing["families"]["memory_fusion_bias"]["passed"] is False


def test_shared_qo_gate_topology_is_initialized_and_signed() -> None:
    config = evolution.build_evolution_delta_config(
        "shared_qo_content_gated_attention_output"
    )
    protocol = evolution.load_evolution_protocol(
        "shared_qo_content_gated_attention_output"
    )

    assert config.memory_fusion_placement == "attention_output"
    assert config.memory_fusion_mode == "content_gated_qo_add"
    assert config.memory_fusion_gate_init == 0.1
    assert protocol["topology_change"]["gate_initial_value"] == 0.1
    assert protocol["topology_change"]["q_correction"] == (
        "multiply_by_shared_content_gate_before_attention"
    )
    assert protocol["topology_change"]["o_correction"] == (
        "multiply_by_same_shared_content_gate_after_attention"
    )
    assert protocol["execution_change"]["allocator_cache_release"] == (
        "gc_collect_and_cuda_empty_cache_before_native_write_and_read"
    )
    assert protocol["execution_change"]["gradient_equivalence_required"] is True
    assert protocol["execution_change"]["batch_change"] is False
    assert protocol["execution_change"]["selective_offload_min_bytes"] == (
        8 * 1024 * 1024
    )
    assert evolution.NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES == 8 * 1024 * 1024


def test_native_selective_offload_excludes_cpu_leaf_and_small_tensors() -> None:
    large_cpu = torch.ones(6 * 1024 * 1024, dtype=torch.float32).sin()
    small_cpu = torch.ones(2, dtype=torch.float32).sin()
    leaf_cpu = torch.ones(6 * 1024 * 1024, dtype=torch.float32)

    assert large_cpu.is_leaf is True
    assert evolution.should_selectively_offload_native_activation(
        large_cpu,
    ) is False
    assert evolution.should_selectively_offload_native_activation(
        small_cpu,
    ) is False
    assert evolution.should_selectively_offload_native_activation(
        leaf_cpu,
    ) is False


def test_native_row_allocator_cache_release_is_cuda_only(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(evolution.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        evolution.torch.cuda,
        "empty_cache",
        lambda: calls.append("cuda"),
    )

    evolution.release_native_row_allocator_cache(torch.device("cpu"))
    assert calls == ["gc"]

    evolution.release_native_row_allocator_cache(torch.device("cuda", 0))
    assert calls == ["gc", "gc", "cuda"]
