from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from deltamem.chat_templates import (
    apply_chat_template as apply_project_chat_template,
    qwen3_enable_thinking_override,
    smollm3_enable_thinking_override,
)
from deltamem.core.delta import (
    DeltaMemAttention,
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_output_ratio_stats,
    collect_delta_mem_read_representations,
    freeze_non_delta_mem_params,
    get_delta_mem_online_state,
    get_delta_mem_write_regularization,
    load_delta_mem_adapter,
    load_delta_mem_online_state,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_read_context_mask,
    set_delta_mem_read_representation_capture_mask,
    set_delta_mem_write_enabled,
    set_delta_mem_write_message_ids,
    set_delta_mem_write_sentence_ids,
)
import deltamem.runtime.session as pipeline
import deltamem.train.delta_sft_experimental as experimental_train
from deltamem.runtime.session import DeltaMemChatSession, DeltaMemSessionSnapshot
from deltamem.train.delta_sft import (
    DeltaMemTrainer,
    EpisodeCausalLMCollator,
    build_episode_training_examples,
    parse_layer_indices,
    parse_args,
    prepare_tokenized_dataset,
    tokenize_messages_for_sft,
)
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention

try:
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextModel
except Exception:  # pragma: no cover - optional Transformers version support
    Qwen3_5TextConfig = None
    Qwen3_5Attention = None
    Qwen3_5TextModel = None

try:
    from transformers.models.gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForCausalLM,
        Gemma4TextAttention,
        Gemma4TextModel,
    )
except Exception:  # pragma: no cover - optional Transformers version support
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None
    Gemma4TextAttention = None
    Gemma4TextModel = None


class FakeTokenizer:
    def __init__(
        self,
        response_text: str = "",
        eos_token_id: int | None = None,
        *,
        name_or_path: str = "fake-tokenizer",
    ) -> None:
        self.response_text = response_text
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.name_or_path = name_or_path
        self.last_apply_kwargs: dict[str, object] | None = None

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.response_text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if enable_thinking is False:
            rendered += "<think></think>"
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered] or [0]
        return torch.tensor([token_ids], dtype=torch.long)


class AlternationStrictTokenizer(FakeTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chat_template = (
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
            "{% endif %}"
            "<{{ message['role'] }}>{{ message['content'] }}"
            "{% endfor %}"
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        roles = [message["role"] for message in messages if message["role"] != "system"]
        if (
            "Conversation roles must alternate user/assistant/user/assistant/..." in self.chat_template
            and any(left == right for left, right in zip(roles, roles[1:]))
        ):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        token_ids = [ord(ch) for ch in rendered] or [0]
        return torch.tensor([token_ids], dtype=torch.long)


class SentenceSpanUnstableTokenizer(FakeTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._piece_to_id: dict[str, int] = {}
        self._next_piece_id = 1000

    def _piece_id(self, piece: str) -> int:
        token_id = self._piece_to_id.get(piece)
        if token_id is None:
            token_id = self._next_piece_id
            self._piece_to_id[piece] = token_id
            self._next_piece_id += 1
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces: list[int] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                start = index
                while index < len(text) and text[index].isspace():
                    index += 1
                if index < len(text) and text[index].isalnum():
                    word_start = index
                    while index < len(text) and text[index].isalnum():
                        index += 1
                    pieces.append(self._piece_id(text[start:index]))
                else:
                    pieces.append(self._piece_id(text[start:index]))
                continue
            if char.isalnum():
                start = index
                while index < len(text) and text[index].isalnum():
                    index += 1
                pieces.append(self._piece_id(text[start:index]))
                continue
            pieces.append(self._piece_id(char))
            index += 1
        return pieces

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
        enable_thinking: bool | None = None,
    ):
        self.last_apply_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "return_tensors": return_tensors,
            "enable_thinking": enable_thinking,
        }
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if enable_thinking is False:
            rendered += "<think></think>"
        if not tokenize:
            return rendered
        token_ids: list[int] = []
        for message in messages:
            token_ids.extend([
                self._piece_id(f"<role:{message['role']}>"),
                self._piece_id("\\n"),
            ])
            token_ids.extend(self.encode(message["content"], add_special_tokens=False))
            token_ids.extend([
                self._piece_id("<end>"),
                self._piece_id("\\n"),
            ])
        if add_generation_prompt:
            token_ids.append(self._piece_id("<assistant>"))
        return torch.tensor([token_ids or [0]], dtype=torch.long)


def test_qwen3_chat_template_override_respects_model_type() -> None:
    unified = FakeTokenizer(name_or_path="Qwen/Qwen3-8B")
    instruct = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Instruct-2507")
    thinking = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Thinking-2507")

    assert qwen3_enable_thinking_override(unified) is False
    assert qwen3_enable_thinking_override(instruct) is None
    assert qwen3_enable_thinking_override(thinking) is None


def test_qwen3_chat_template_does_not_break_instruct_prompt() -> None:
    tokenizer = FakeTokenizer(name_or_path="Qwen/Qwen3-4B-Instruct-2507")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is None
    assert "<think></think>" not in rendered


def test_smollm3_chat_template_disables_thinking_by_default() -> None:
    tokenizer = FakeTokenizer(name_or_path="HuggingFaceTB/SmolLM3-3B")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert smollm3_enable_thinking_override(tokenizer) is False
    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is False
    assert "<think></think>" in rendered


def test_qwen3_chat_template_injects_no_think_marker_for_unified_model() -> None:
    tokenizer = FakeTokenizer(name_or_path="Qwen/Qwen3-8B")
    rendered = apply_project_chat_template(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert tokenizer.last_apply_kwargs is not None
    assert tokenizer.last_apply_kwargs["enable_thinking"] is False
    assert "<think></think>" in rendered


def test_chat_template_falls_back_for_non_alternating_roles() -> None:
    tokenizer = AlternationStrictTokenizer(name_or_path="example/non-qwen-chat-template")
    rendered = apply_project_chat_template(
        tokenizer,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert rendered == "<system>sys<user>u1<user>u2<assistant>a1"
    assert "Conversation roles must alternate" in tokenizer.chat_template


def test_chat_template_keeps_default_path_for_alternating_roles() -> None:
    tokenizer = AlternationStrictTokenizer(name_or_path="example/non-qwen-chat-template")
    rendered = apply_project_chat_template(
        tokenizer,
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == "<user>u1<assistant>a1<assistant>"
    assert "Conversation roles must alternate" in tokenizer.chat_template


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        return_dict: bool = True,
    ):
        self.calls += 1
        vocab_size = 8
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        return type(
            "FakeOutputs",
            (),
            {
                "logits": logits,
                "past_key_values": ("pkv", self.calls),
            },
        )()


def test_disable_training_cache_updates_composite_and_text_configs() -> None:
    class TextConfig:
        use_cache = True

    class CompositeConfig:
        def __init__(self) -> None:
            self.text_config = TextConfig()

        def get_text_config(self, decoder: bool | None = None):
            assert decoder is True
            return self.text_config

    model = type("Model", (), {"config": CompositeConfig()})()
    assert not hasattr(model.config, "use_cache")

    experimental_train._disable_training_cache(model)

    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False


def test_promote_trainable_parameters_to_fp32_preserves_frozen_dtype() -> None:
    model = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
    model.weight.requires_grad = False

    experimental_train._promote_trainable_parameters_to_fp32(model)

    assert model.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.float32


def test_logits_to_keep_kwargs_supports_wrapped_current_and_legacy_models() -> None:
    class CurrentModel(torch.nn.Module):
        def forward(self, logits_to_keep=0):
            raise AssertionError("signature inspection must not run the model")

    class LegacyModel(torch.nn.Module):
        def forward(self, num_logits_to_keep=0):
            raise AssertionError("signature inspection must not run the model")

    class ModelWithoutLogitLimit(torch.nn.Module):
        def forward(self, input_ids):
            raise AssertionError("signature inspection must not run the model")

    class ModuleWrapper(torch.nn.Module):
        def __init__(self, module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class CompiledWrapper(torch.nn.Module):
        def __init__(self, module) -> None:
            super().__init__()
            self._orig_mod = module

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    trainer = object.__new__(experimental_train.DeltaMemTrainer)

    assert trainer._logits_to_keep_kwargs(ModuleWrapper(CurrentModel()), 1) == {
        "logits_to_keep": 1
    }
    assert trainer._logits_to_keep_kwargs(CompiledWrapper(LegacyModel()), 1) == {
        "num_logits_to_keep": 1
    }
    assert trainer._logits_to_keep_kwargs(ModelWithoutLogitLimit(), 1) == {}


def test_gemma4_prime_logit_limit_preserves_rwkv_ms_state() -> None:
    if Gemma4TextConfig is None or Gemma4ForCausalLM is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config).eval()
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            target_layers=(0,),
            target_modules=("self_attn",),
        ),
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 7))
    attention_mask = torch.ones_like(input_ids)
    lm_head_output_shapes: list[tuple[int, ...]] = []
    hook = model.lm_head.register_forward_hook(
        lambda _module, _inputs, output: lm_head_output_shapes.append(tuple(output.shape))
    )

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        full_logits_state = get_delta_mem_online_state(model)
        reset_delta_mem_states(model)
        trainer = object.__new__(experimental_train.DeltaMemTrainer)
        trainer._prime_episode_state(
            model,
            write_input_ids=input_ids,
            write_attention_mask=attention_mask,
            batch_size=1,
        )
        limited_logits_state = get_delta_mem_online_state(model)
    hook.remove()

    assert lm_head_output_shapes == [(1, 7, 64), (1, 1, 64)]
    assert full_logits_state.keys() == limited_logits_state.keys()
    assert all(
        torch.equal(full_logits_state[name], limited_logits_state[name])
        for name in full_logits_state
    )


class StubSession(DeltaMemChatSession):
    def _ingest_full_ids(self, full_ids: torch.Tensor) -> torch.Tensor:
        self.processed_input_ids = full_ids.detach().cpu()
        self.last_ingest_stats = {
            "prev_tokens": 0,
            "full_tokens": int(full_ids.size(1)),
            "prefix_tokens": 0,
            "suffix_tokens": int(full_ids.size(1)),
            "rebuilt": False,
            "elapsed_ms": 0.0,
        }
        return torch.zeros(1, 4)

    def _greedy_generate(self, next_token_logits: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        generated_ids = torch.tensor([[11, 12]], dtype=torch.long)
        generated_cpu = generated_ids.detach().cpu()
        self.processed_input_ids = torch.cat([self.processed_input_ids, generated_cpu], dim=1)
        self.last_decode_stats = {
            "generated_tokens": int(generated_ids.size(1)),
            "elapsed_ms": 0.0,
        }
        return generated_ids

    def state_stats(self) -> dict[str, float]:
        return {"num_modules": 0}


def make_qwen3_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 4,
) -> Qwen3Attention:
    config = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3Attention(config, layer_idx)


def make_smollm3_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 4,
) -> SmolLM3Attention:
    config = SmolLM3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
        attention_bias=False,
        use_sliding_window=False,
    )
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.no_rope_layers = [1] * config.num_hidden_layers
    config._attn_implementation = "eager"
    return SmolLM3Attention(config, layer_idx)


def make_qwen3_5_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 8,
) -> Qwen3_5Attention:
    if Qwen3_5TextConfig is None or Qwen3_5Attention is None:
        pytest.skip("Qwen3.5 is not available in this Transformers version")
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention"] * max(layer_idx + 1, 1),
        partial_rotary_factor=0.5,
    )
    config._attn_implementation = "eager"
    return Qwen3_5Attention(config, layer_idx)


def make_gemma4_attention(
    layer_idx: int = 0,
    *,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    head_dim: int = 8,
    global_head_dim: int = 8,
) -> Gemma4TextAttention:
    if Gemma4TextConfig is None or Gemma4TextAttention is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    config = Gemma4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=max(layer_idx + 1, 1),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention"] * max(layer_idx + 1, 1),
        sliding_window=4,
    )
    config._attn_implementation = "eager"
    return Gemma4TextAttention(config, layer_idx)


def make_gemma4_shared_kv_config() -> Gemma4TextConfig:
    if Gemma4TextConfig is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def make_position_embeddings(
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.ones(batch_size, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.zeros_like(cos)
    return cos, sin


def make_causal_attention_mask(attention_mask_2d: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask_2d.shape
    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    for batch_idx in range(batch_size):
        valid_prefix = int(attention_mask_2d[batch_idx].sum().item())
        for query_idx in range(seq_len):
            if query_idx < valid_prefix:
                mask[batch_idx, 0, query_idx, : query_idx + 1] = 0.0
            else:
                mask[batch_idx, 0, query_idx, :valid_prefix] = 0.0
    return mask


def make_causal_attention_mask_with_past(
    *,
    batch_size: int,
    query_len: int,
    past_len: int,
) -> torch.Tensor:
    total_k_len = past_len + query_len
    mask = torch.full(
        (batch_size, 1, query_len, total_k_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    for query_idx in range(query_len):
        mask[:, :, query_idx, : past_len + query_idx + 1] = 0.0
    return mask


def make_delta_module(
    *,
    output_init: str = "base_slice",
    rank: int = 2,
    num_state_heads: int = 1,
    memory_backend: str = "delta_rule",
    rwkv_ms_num_states: int = 4,
    rwkv_ms_chunk_size: int = 2,
    rwkv_ms_output_init_scale: float = 0.02,
    rwkv_ms_semantics_version: int = 2,
    rankwise_gates: bool = True,
    slot_read_top_k: int = 0,
    memory_readout_mode: str = "delta",
    synthetic_memory_slots: int = 1,
    delta_heads: tuple[str, ...] | str = ("q", "k", "v", "o"),
    delta_o_rmsnorm: bool = False,
    memory_fusion_mode: str = "add",
    memory_fusion_gate_init: float = 0.1,
) -> DeltaMemAttention:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    return DeltaMemAttention(
        base,
        HFDeltaMemConfig(
            rank=rank,
            num_state_heads=num_state_heads,
            memory_backend=memory_backend,
            rwkv_ms_num_states=rwkv_ms_num_states,
            rwkv_ms_chunk_size=rwkv_ms_chunk_size,
            rwkv_ms_output_init_scale=rwkv_ms_output_init_scale,
            rwkv_ms_semantics_version=rwkv_ms_semantics_version,
            output_init=output_init,
            rankwise_gates=rankwise_gates,
            slot_read_top_k=slot_read_top_k,
            memory_readout_mode=memory_readout_mode,
            synthetic_memory_slots=synthetic_memory_slots,
            delta_heads=delta_heads,
            delta_o_rmsnorm=delta_o_rmsnorm,
            memory_fusion_mode=memory_fusion_mode,
            memory_fusion_gate_init=memory_fusion_gate_init,
        ),
    )


def naive_delta_forward(
    module: DeltaMemAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state = torch.zeros(
        hidden_states.size(0),
        module.rank,
        module.rank,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    reads = []
    for token_idx in range(hidden_states.size(1)):
        token = hidden_states[:, token_idx, :]
        q = torch.nn.functional.linear(token, module.memory_q_proj)
        k = torch.nn.functional.linear(token, module.memory_k_proj)
        v = torch.nn.functional.linear(token, module.memory_v_proj)
        if module.normalize_qk:
            q = torch.tanh(q)
            q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            k = torch.tanh(k)
            k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        v_old = torch.bmm(state, k.unsqueeze(-1)).squeeze(-1)
        beta = torch.sigmoid(torch.nn.functional.linear(token, module.beta_proj) + module.beta_bias).view(
            token.size(0), module.gate_dim, 1
        )
        if module.couple_lambda:
            lam = 1.0 - beta
        else:
            lam = torch.sigmoid(
                torch.nn.functional.linear(token, module.lambda_proj) + module.lambda_bias
            ).view(token.size(0), module.gate_dim, 1)
        state = lam * state + beta * torch.einsum("bi,bj->bij", v - v_old, k)
        reads.append(read)

    reads = torch.stack(reads, dim=1)
    packed_delta = torch.nn.functional.linear(
        reads,
        torch.cat(
            [
                module.delta_q_proj,
                module.delta_k_proj,
                module.delta_v_proj,
                module.delta_o_proj,
            ],
            dim=0,
        ),
    ) * module.delta_scaling
    delta_q, delta_k, delta_v, delta_o = torch.split(
        packed_delta,
        [
            module.base.q_proj.out_features,
            module.base.k_proj.out_features,
            module.base.v_proj.out_features,
            module.base.o_proj.out_features,
        ],
        dim=-1,
    )
    delta_o = module._apply_delta_o_rmsnorm(delta_o)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.base.head_dim)
    query_states = module.base.q_proj(hidden_states) + delta_q
    key_states = module.base.k_proj(hidden_states) + delta_k
    value_states = module.base.v_proj(hidden_states) + delta_v
    query_states = module.base.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.base.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        position_embeddings[0],
        position_embeddings[1],
    )
    attn_output, attn_weights = eager_attention_forward(
        module.base,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=module.base.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = module.base.o_proj(attn_output) + delta_o
    return attn_output, attn_weights


class ToyAttentionModel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.self_attn = make_qwen3_attention() if module is None else module
        self.residual = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.self_attn, DeltaMemAttention):
            attn_output, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        else:
            attn_output, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        return self.residual(attn_output)


def test_two_training_steps_succeed_with_state_reset() -> None:
    module = make_delta_module(output_init="base_slice")
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)

    for _ in range(2):
        module.reset_state()
        x = torch.randn(1, 3, 8)
        position_embeddings = make_position_embeddings(
            batch_size=x.size(0),
            seq_len=x.size(1),
            head_dim=module.base.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        loss = module(x, position_embeddings, None)[0].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_content_contrast_real_rwkv_ms_state_keeps_both_writer_graphs() -> None:
    class TinyDeltaMemoryLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.delta = make_delta_module(
                output_init="base_slice_fixed",
                rank=2,
                memory_backend="rwkv_ms",
                rwkv_ms_num_states=2,
                rwkv_ms_chunk_size=2,
            )
            self.lm_head = torch.nn.Linear(8, 32, bias=False)

        def forward(self, input_ids, attention_mask, labels=None, **kwargs):
            del attention_mask, kwargs
            hidden_states = self.embedding(input_ids)
            position_embeddings = make_position_embeddings(
                batch_size=hidden_states.size(0),
                seq_len=hidden_states.size(1),
                head_dim=self.delta.base.head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            hidden_states, _ = self.delta(
                hidden_states,
                position_embeddings,
                None,
            )
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}

    trainer = object.__new__(experimental_train.DeltaMemTrainer)
    trainer.episode_read_write_enabled = False
    trainer.memory_kl_weight = 0.0
    trainer.memory_base_kl_weight = 0.0
    trainer.memory_margin = 0.5
    trainer.memory_contrast_weight = 0.25
    model = TinyDeltaMemoryLM()
    prime_state_requires_grad: list[bool] = []
    original_prime = trainer._prime_episode_state

    def tracked_prime(active_model, **kwargs) -> None:
        original_prime(active_model, **kwargs)
        assert model.delta.delta_state is not None
        prime_state_requires_grad.append(model.delta.delta_state.requires_grad)

    trainer._prime_episode_state = tracked_prime
    model_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
    }

    loss, _, stats = trainer._compute_content_contrast_ce(
        model,
        model_inputs,
        loss_kwargs={},
        write_input_ids=torch.tensor([[10, 11, 12]]),
        write_attention_mask=torch.ones(1, 3, dtype=torch.long),
        write_message_ids=torch.zeros(1, 3, dtype=torch.long),
        write_sentence_ids=torch.zeros(1, 3, dtype=torch.long),
        negative_write_input_ids=torch.tensor([[20, 21, 22]]),
        negative_write_attention_mask=torch.ones(1, 3, dtype=torch.long),
        negative_write_message_ids=torch.zeros(1, 3, dtype=torch.long),
        negative_write_sentence_ids=torch.zeros(1, 3, dtype=torch.long),
    )

    assert prime_state_requires_grad == [True, True]
    assert stats["teacher_loss"] == 0.0
    assert stats["kl_loss"] == 0.0
    assert torch.isfinite(loss)

    loss.backward()
    readout_grad = model.delta.hrm_rwkv7_core.output.weight.grad
    assert readout_grad is not None
    assert torch.isfinite(readout_grad).all()
    assert torch.count_nonzero(readout_grad) > 0
    embedding_grad = model.embedding.weight.grad
    assert embedding_grad is not None
    donor_writer_grad = embedding_grad[20:23]
    assert torch.isfinite(donor_writer_grad).all()
    assert torch.count_nonzero(donor_writer_grad) > 0


@pytest.mark.parametrize("rankwise_gates", [False, True])
def test_sequence_scan_matches_naive_reference(rankwise_gates: bool) -> None:
    module = make_delta_module(output_init="random", rank=2, rankwise_gates=rankwise_gates)
    x = torch.randn(2, 5, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    with torch.inference_mode():
        optimized, optimized_weights = module(x, position_embeddings, None)
    module.reset_state()
    reference, reference_weights = naive_delta_forward(module, x, position_embeddings)
    assert torch.allclose(optimized, reference, atol=1e-5, rtol=1e-5)
    assert torch.allclose(optimized_weights, reference_weights, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton scan test")
@pytest.mark.parametrize("rankwise_gates", [False, True])
def test_triton_sequence_scan_matches_torch_forward_and_backward(rankwise_gates: bool) -> None:
    module = make_delta_module(output_init="random", rank=2, rankwise_gates=rankwise_gates).cuda().to(torch.float32)
    batch_size = 2
    seq_len = 5
    rank = module.rank
    token_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        device="cuda",
        dtype=torch.bool,
    )

    inputs = {
        "state": torch.randn(batch_size, rank, rank, device="cuda", dtype=torch.float32),
        "q": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "k": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "v": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
        "lam": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
    }

    ref_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    keep_ref, erase_ref, write_ref = module._memory_update_coefficients(
        ref_inputs["beta"],
        ref_inputs["lam"],
    )
    state_ref, reads_ref = module._memory_affine_scan_torch(
        ref_inputs["state"],
        ref_inputs["q"],
        ref_inputs["k"],
        ref_inputs["v"],
        keep_ref,
        erase_ref,
        write_ref,
        token_mask=token_mask,
    )
    ref_loss = state_ref.sum() + reads_ref.sum()
    ref_grads = torch.autograd.grad(
        ref_loss,
        tuple(ref_inputs.values()),
        retain_graph=False,
    )

    module.scan_impl = "triton"
    opt_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    state_opt, reads_opt = module._memory_affine_scan(
        opt_inputs["state"],
        opt_inputs["q"],
        opt_inputs["k"],
        opt_inputs["v"],
        opt_inputs["beta"],
        opt_inputs["lam"],
        token_mask=token_mask,
    )
    opt_loss = state_opt.sum() + reads_opt.sum()
    opt_grads = torch.autograd.grad(
        opt_loss,
        tuple(opt_inputs.values()),
        retain_graph=False,
    )

    assert torch.allclose(state_opt, state_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reads_opt, reads_ref, atol=1e-5, rtol=1e-5)
    for grad_ref, grad_opt in zip(ref_grads, opt_grads):
        assert torch.allclose(grad_opt, grad_ref, atol=1e-5, rtol=1e-5)


def test_zero_init_attention_wrapper_matches_base_attention() -> None:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    wrapped_base = make_qwen3_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(rank=2, output_init="zero", rankwise_gates=True),
    )
    x = torch.randn(2, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    base_out, base_weights = base(x, position_embeddings, None)
    wrapped_out, wrapped_weights = wrapped(x, position_embeddings, None)

    assert torch.allclose(base_out, wrapped_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_weights, wrapped_weights, atol=1e-6, rtol=1e-6)


def test_zero_init_rwkv_ms_attention_wrapper_matches_base_attention() -> None:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    wrapped_base = make_qwen3_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            rankwise_gates=True,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=3,
            rwkv_ms_chunk_size=2,
        ),
    )
    x = torch.randn(2, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    base_out, base_weights = base(x, position_embeddings, None)
    wrapped_out, wrapped_weights = wrapped(x, position_embeddings, None)

    assert torch.allclose(base_out, wrapped_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_weights, wrapped_weights, atol=1e-6, rtol=1e-6)
    assert wrapped.delta_state is not None
    assert wrapped.delta_state.shape == (2, 1, 3, 2, 2)
    assert wrapped.hrm_rwkv7_core is not None
    assert torch.count_nonzero(wrapped.hrm_rwkv7_core.output.weight) == 0


def test_zero_init_rwkv_ms_qwen3_5_attention_wrapper_matches_output_gate() -> None:
    torch.manual_seed(0)
    base = make_qwen3_5_attention()
    wrapped_base = make_qwen3_5_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            rankwise_gates=True,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=3,
            rwkv_ms_chunk_size=2,
        ),
    )
    x = torch.randn(2, 4, 16)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=int(base.head_dim * base.config.partial_rotary_factor),
        device=x.device,
        dtype=x.dtype,
    )

    base_out, base_weights = base(x, position_embeddings, None)
    wrapped_out, wrapped_weights = wrapped(x, position_embeddings, None)

    assert torch.allclose(base_out, wrapped_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_weights, wrapped_weights, atol=1e-6, rtol=1e-6)
    assert wrapped.query_out_features == base.config.num_attention_heads * base.head_dim
    assert wrapped.delta_q_proj.shape == (wrapped.query_out_features, wrapped.state_read_dim)
    assert wrapped.delta_state is not None
    assert wrapped.delta_state.shape == (2, 1, 3, 2, 2)


def test_zero_init_rwkv_ms_gemma4_attention_wrapper_matches_base_attention() -> None:
    torch.manual_seed(0)
    base = make_gemma4_attention()
    wrapped_base = make_gemma4_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(
            rank=2,
            num_state_heads=2,
            output_init="zero",
            rankwise_gates=True,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=3,
            rwkv_ms_chunk_size=2,
        ),
    )
    x = torch.randn(1, 5, 16)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 5, dtype=torch.long))

    base_out, base_weights = base(x, position_embeddings, attention_mask, shared_kv_states={})
    wrapped_out, wrapped_weights = wrapped(x, position_embeddings, attention_mask, shared_kv_states={})

    assert torch.allclose(base_out, wrapped_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_weights, wrapped_weights, atol=1e-6, rtol=1e-6)
    assert wrapped.delta_state is not None
    assert wrapped.delta_state.shape == (1, 2, 3, 2, 2)


def test_high_rank_scan_supports_large_rank() -> None:
    module = make_delta_module(rank=64)
    module.scan_impl = "auto"
    batch_size = 1
    seq_len = 4
    state = torch.randn(batch_size, 64, 64)
    q = torch.randn(batch_size, seq_len, 64)
    k = torch.randn(batch_size, seq_len, 64)
    v = torch.randn(batch_size, seq_len, 64)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, module.gate_dim, 1))
    lam = torch.sigmoid(torch.randn(batch_size, seq_len, module.gate_dim, 1))

    final_state, reads = module._memory_affine_scan(
        state,
        q,
        k,
        v,
        beta,
        lam,
    )

    assert final_state.shape == (batch_size, 64, 64)
    assert reads.shape == (batch_size, seq_len, 64)
    assert torch.isfinite(final_state).all()
    assert torch.isfinite(reads).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton high-rank scan test")
@pytest.mark.parametrize("rank", [64, 128])
def test_triton_scan_supports_high_rank(rank: int) -> None:
    module = make_delta_module(rank=rank).cuda().to(torch.float32)
    module.scan_impl = "triton"
    batch_size = 1
    seq_len = 4
    token_mask = torch.tensor([[1, 1, 1, 1]], device="cuda", dtype=torch.bool)

    inputs = {
        "state": torch.randn(batch_size, rank, rank, device="cuda", dtype=torch.float32),
        "q": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "k": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "v": torch.randn(batch_size, seq_len, rank, device="cuda", dtype=torch.float32),
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
        "lam": torch.sigmoid(
            torch.randn(batch_size, seq_len, module.gate_dim, 1, device="cuda", dtype=torch.float32)
        ),
    }

    ref_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    keep_ref, erase_ref, write_ref = module._memory_update_coefficients(
        ref_inputs["beta"],
        ref_inputs["lam"],
    )
    state_ref, reads_ref = module._memory_affine_scan_torch(
        ref_inputs["state"],
        ref_inputs["q"],
        ref_inputs["k"],
        ref_inputs["v"],
        keep_ref,
        erase_ref,
        write_ref,
        token_mask=token_mask,
    )
    ref_loss = state_ref.sum() + reads_ref.sum()
    ref_grads = torch.autograd.grad(ref_loss, tuple(ref_inputs.values()), retain_graph=False)

    triton_inputs = {
        name: tensor.clone().detach().requires_grad_(True) for name, tensor in inputs.items()
    }
    state_opt, reads_opt = module._memory_affine_scan(
        triton_inputs["state"],
        triton_inputs["q"],
        triton_inputs["k"],
        triton_inputs["v"],
        triton_inputs["beta"],
        triton_inputs["lam"],
        token_mask=token_mask,
    )
    opt_loss = state_opt.sum() + reads_opt.sum()
    opt_grads = torch.autograd.grad(opt_loss, tuple(triton_inputs.values()), retain_graph=False)

    assert torch.allclose(state_opt, state_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(reads_opt, reads_ref, atol=1e-4, rtol=1e-4)
    for grad_ref, grad_opt in zip(ref_grads, opt_grads):
        assert torch.allclose(grad_opt, grad_ref, atol=2e-2, rtol=5e-4)


def test_attach_delta_mem_wraps_attention_modules_not_projection_modules() -> None:
    model = ToyAttentionModel()
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(rank=2, output_init="zero", target_modules=("self_attn",)),
    )

    assert replaced == ["self_attn"]
    assert isinstance(model.self_attn, DeltaMemAttention)
    assert isinstance(model.self_attn.base.q_proj, torch.nn.Linear)


def test_attach_delta_mem_wraps_smollm3_attention_modules() -> None:
    model = ToyAttentionModel(module=make_smollm3_attention())
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(rank=2, output_init="zero", target_modules=("self_attn",)),
    )

    assert replaced == ["self_attn"]
    assert isinstance(model.self_attn, DeltaMemAttention)
    assert isinstance(model.self_attn.base, SmolLM3Attention)


def test_attach_delta_mem_wraps_only_qwen3_5_full_attention_layers() -> None:
    if Qwen3_5TextConfig is None or Qwen3_5TextModel is None:
        pytest.skip("Qwen3.5 is not available in this Transformers version")
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        partial_rotary_factor=0.5,
    )
    config._attn_implementation = "eager"
    model = Qwen3_5TextModel(config)

    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            target_modules=("self_attn",),
        ),
    )

    assert replaced == ["layers.3.self_attn"]
    assert isinstance(model.layers[3].self_attn, DeltaMemAttention)
    assert all(not hasattr(model.layers[layer_idx], "self_attn") for layer_idx in range(3))


def test_qwen3_5_hybrid_topology_wraps_all_16_physical_attention_layers() -> None:
    if Qwen3_5TextConfig is None or Qwen3_5TextModel is None:
        pytest.skip("Qwen3.5 is not available in this Transformers version")
    full_attention_layers = tuple(range(3, 64, 4))
    layer_types = [
        "full_attention" if layer_idx in full_attention_layers else "linear_attention"
        for layer_idx in range(64)
    ]
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=layer_types,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        partial_rotary_factor=0.25,
    )
    config._attn_implementation = "eager"
    model = Qwen3_5TextModel(config)

    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            target_layers=full_attention_layers,
            target_modules=("self_attn",),
        ),
    )

    expected = [f"layers.{layer_idx}.self_attn" for layer_idx in full_attention_layers]
    assert replaced == expected
    assert len(replaced) == 16
    for layer_idx in full_attention_layers:
        wrapped = model.layers[layer_idx].self_attn
        assert isinstance(wrapped, DeltaMemAttention)
        assert wrapped.hrm_rwkv7_core is not None
        assert wrapped.hrm_rwkv7_core.layer_id == layer_idx
        assert wrapped.hrm_rwkv7_core.n_layer == 64


def test_qwen3_5_text_model_forward_runs_after_rwkv_ms_attach() -> None:
    if Qwen3_5TextConfig is None or Qwen3_5TextModel is None:
        pytest.skip("Qwen3.5 is not available in this Transformers version")
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention", "full_attention"],
        partial_rotary_factor=0.5,
    )
    config._attn_implementation = "eager"
    model = Qwen3_5TextModel(config)
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            target_modules=("self_attn",),
        ),
    )

    output = model(input_ids=torch.randint(0, 64, (1, 5)), use_cache=False)

    assert replaced == ["layers.0.self_attn", "layers.1.self_attn"]
    assert output.last_hidden_state.shape == (1, 5, 16)
    assert torch.isfinite(output.last_hidden_state).all()


def test_qwen3_5_hybrid_cached_decode_matches_zero_init_rwkv_ms_wrapper() -> None:
    if Qwen3_5TextConfig is None or Qwen3_5TextModel is None:
        pytest.skip("Qwen3.5 is not available in this Transformers version")
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        partial_rotary_factor=0.5,
    )
    config._attn_implementation = "eager"
    base_model = Qwen3_5TextModel(config).eval()
    wrapped_model = copy.deepcopy(base_model).eval()
    replaced = attach_delta_mem(
        wrapped_model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            target_layers=(3,),
            target_modules=("self_attn",),
        ),
    )

    prefill_ids = torch.tensor([[1, 2, 3]])
    decode_ids = torch.tensor([[4]])
    with torch.inference_mode():
        base_prefill = base_model(input_ids=prefill_ids, use_cache=True)
        wrapped_prefill = wrapped_model(input_ids=prefill_ids, use_cache=True)
        base_prefill_cache_length = base_prefill.past_key_values.get_seq_length()
        wrapped_prefill_cache_length = wrapped_prefill.past_key_values.get_seq_length()
        base_decode = base_model(
            input_ids=decode_ids,
            past_key_values=base_prefill.past_key_values,
            use_cache=True,
        )
        wrapped_decode = wrapped_model(
            input_ids=decode_ids,
            past_key_values=wrapped_prefill.past_key_values,
            use_cache=True,
        )

    assert replaced == ["layers.3.self_attn"]
    torch.testing.assert_close(
        wrapped_prefill.last_hidden_state,
        base_prefill.last_hidden_state,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        wrapped_decode.last_hidden_state,
        base_decode.last_hidden_state,
        atol=1e-6,
        rtol=1e-6,
    )
    assert base_prefill_cache_length == wrapped_prefill_cache_length == 3
    assert base_decode.past_key_values.get_seq_length() == 4
    assert wrapped_decode.past_key_values.get_seq_length() == 4


def test_attach_delta_mem_wraps_all_gemma4_shared_kv_layers_for_o_only() -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    config = make_gemma4_shared_kv_config()
    model = Gemma4TextModel(config)

    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )

    assert replaced == [f"layers.{layer_idx}.self_attn" for layer_idx in range(4)]
    for layer_idx, layer in enumerate(model.layers):
        wrapped = layer.self_attn
        assert isinstance(wrapped, DeltaMemAttention)
        assert wrapped.active_delta_heads == frozenset({"o"})
        assert wrapped.is_kv_shared_layer is (layer_idx >= 2)
        assert hasattr(wrapped, "memory_fusion_hidden_weight")
        if layer_idx >= 2:
            assert not hasattr(wrapped.base, "k_proj")
            assert not hasattr(wrapped.base, "v_proj")
    output = model(input_ids=torch.randint(0, 64, (1, 5)), use_cache=False)
    assert output.last_hidden_state.shape == (1, 5, 16)
    assert torch.isfinite(output.last_hidden_state).all()


def test_gemma4_post_attention_norm_runs_once_and_reports_layer_groups() -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    config = make_gemma4_shared_kv_config()
    model = Gemma4TextModel(config).eval()
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_placement="post_attention_norm",
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    base_calls = [0, 0, 0, 0]
    handles = []
    for layer_idx, layer in enumerate(model.layers):
        wrapped = layer.self_attn
        assert isinstance(wrapped, DeltaMemAttention)

        def count_base(_module, _inputs, _output, *, index=layer_idx) -> None:
            base_calls[index] += 1

        handles.append(wrapped.base.register_forward_hook(count_base))

    with torch.no_grad():
        output = model(input_ids=torch.randint(0, config.vocab_size, (1, 5)), use_cache=False)
    for handle in handles:
        handle.remove()

    assert replaced == [f"layers.{layer_idx}.self_attn" for layer_idx in range(4)]
    assert output.last_hidden_state.shape == (1, 5, config.hidden_size)
    assert base_calls == [1, 1, 1, 1]
    for layer in model.layers:
        wrapped = layer.self_attn
        assert wrapped.rwkv_ms_positions.tolist() == [5]
        assert wrapped._pending_post_attention_delta is None
        assert wrapped._post_attention_norm_hook_handle is not None

    stats = collect_delta_mem_output_ratio_stats(model)
    assert stats["attention_output_fusion_modules"] == 0
    assert stats["post_attention_norm_fusion_modules"] == 4
    assert stats["normalized_residual_correction_fusion_modules"] == 0
    assert stats["nonshared_local_modules"] == 1
    assert stats["nonshared_full_modules"] == 1
    assert stats["shared_local_modules"] == 1
    assert stats["shared_full_modules"] == 1
    assert 0.0 <= stats["mean_delta_o_gate_lt_001_fraction"] <= 1.0
    assert 0.0 <= stats["mean_delta_o_gate_gt_099_fraction"] <= 1.0


def test_gemma4_post_attention_norm_matches_manual_decoder_equation() -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(7)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Gemma4TextModel(config).eval()
    layer = model.layers[0]
    legacy_norm = copy.deepcopy(layer.post_attention_layernorm).eval()
    captures: dict[str, torch.Tensor] = {}

    def capture_raw_norm(_module, _inputs, output) -> None:
        captures["raw_norm"] = output.detach().clone()

    def capture_residual(_module, inputs) -> None:
        captures["residual"] = inputs[0].detach().clone()

    def capture_base_o(_module, _inputs, output) -> None:
        captures["base_o"] = output[0].detach().clone()

    def capture_postnorm_plus_delta(_module, _inputs, output) -> None:
        captures["postnorm_plus_delta"] = output.detach().clone()

    def capture_attention_residual(_module, inputs) -> None:
        captures["attention_residual"] = inputs[0].detach().clone()

    raw_norm_hook = layer.post_attention_layernorm.register_forward_hook(capture_raw_norm)
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            alpha=4.0,
            output_init="random",
            online_gain=0.2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
            memory_fusion_placement="post_attention_norm",
            target_layers=(0,),
            target_modules=("self_attn",),
        ),
    )
    wrapped = layer.self_attn
    with torch.no_grad():
        wrapped.delta_o_proj.mul_(1000.0)
    handles = [
        layer.input_layernorm.register_forward_pre_hook(capture_residual),
        wrapped.base.register_forward_hook(capture_base_o),
        layer.post_attention_layernorm.register_forward_hook(capture_postnorm_plus_delta),
        layer.pre_feedforward_layernorm.register_forward_pre_hook(capture_attention_residual),
    ]

    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
    raw_norm_hook.remove()
    for handle in handles:
        handle.remove()

    fused_delta = captures["postnorm_plus_delta"] - captures["raw_norm"]
    expected = captures["residual"] + captures["raw_norm"] + fused_delta
    legacy_equation = captures["residual"] + legacy_norm(
        captures["base_o"] + fused_delta
    )
    torch.testing.assert_close(captures["attention_residual"], expected)
    assert fused_delta.float().norm().item() > 0.0
    assert not torch.allclose(
        captures["attention_residual"],
        legacy_equation,
        atol=1e-6,
        rtol=1e-6,
    )
    assert wrapped._pending_post_attention_delta is None
    assert wrapped.last_base_o_norm.item() == pytest.approx(
        captures["raw_norm"].float().norm(dim=-1).mean().item(),
        rel=1e-6,
    )


@pytest.mark.parametrize("memory_fusion_residual_scale", [0.0, 0.375, 1.0])
def test_gemma4_normalized_residual_correction_matches_manual_decoder_equation(
    memory_fusion_residual_scale: float,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(7)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = Gemma4TextModel(config).eval()
    layer = model.layers[0]
    reference_norm = copy.deepcopy(layer.post_attention_layernorm).eval()
    captures: dict[str, torch.Tensor] = {}

    def capture_frozen_norm(_module, inputs, output) -> None:
        captures["raw_attention"] = inputs[0].detach().clone()
        captures["frozen_norm"] = output.detach().clone()
        payload = wrapped._pending_post_attention_delta
        assert payload is not None
        fused_delta = payload[1]
        assert fused_delta is not None
        captures["fused_delta"] = fused_delta.detach().clone()

    def capture_residual(_module, inputs) -> None:
        captures["residual"] = inputs[0].detach().clone()

    def capture_corrected_norm(_module, _inputs, output) -> None:
        captures["corrected_norm"] = output.detach().clone()

    def capture_attention_residual(_module, inputs) -> None:
        captures["attention_residual"] = inputs[0].detach().clone()

    frozen_norm_hook = layer.post_attention_layernorm.register_forward_hook(
        capture_frozen_norm
    )
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            alpha=4.0,
            output_init="random",
            online_gain=0.2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
            memory_fusion_placement="normalized_residual_correction",
            memory_fusion_residual_scale=memory_fusion_residual_scale,
            target_layers=(0,),
            target_modules=("self_attn",),
        ),
    )
    wrapped = layer.self_attn
    assert isinstance(wrapped, DeltaMemAttention)
    with torch.no_grad():
        wrapped.delta_o_proj.mul_(1000.0)
    handles = [
        layer.input_layernorm.register_forward_pre_hook(capture_residual),
        layer.post_attention_layernorm.register_forward_hook(capture_corrected_norm),
        layer.pre_feedforward_layernorm.register_forward_pre_hook(capture_attention_residual),
    ]

    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
    frozen_norm_hook.remove()
    for handle in handles:
        handle.remove()

    memory_norm = reference_norm(
        captures["raw_attention"] + captures["fused_delta"]
    )
    correction = memory_norm - captures["frozen_norm"]
    expected_norm = (
        captures["frozen_norm"] + memory_fusion_residual_scale * correction
    )
    assert captures["fused_delta"].float().norm().item() > 0.0
    assert correction.float().norm().item() > 0.0
    if memory_fusion_residual_scale == 0.0:
        assert torch.equal(captures["corrected_norm"], captures["frozen_norm"])
    elif memory_fusion_residual_scale == 1.0:
        assert torch.equal(captures["corrected_norm"], memory_norm)
    else:
        torch.testing.assert_close(captures["corrected_norm"], expected_norm)
    torch.testing.assert_close(
        captures["attention_residual"],
        captures["residual"] + expected_norm,
    )
    assert wrapped._pending_post_attention_delta is None

    stats = collect_delta_mem_output_ratio_stats(model)
    assert stats["attention_output_fusion_modules"] == 0
    assert stats["post_attention_norm_fusion_modules"] == 0
    assert stats["normalized_residual_correction_fusion_modules"] == 1
    assert torch.isfinite(torch.tensor(stats["mean_memory_residual_norm"]))
    assert torch.isfinite(torch.tensor(stats["mean_memory_residual_ratio"]))


@pytest.mark.parametrize(
    ("memory_fusion_placement", "memory_fusion_residual_scale"),
    [
        ("post_attention_norm", 1.0),
        ("normalized_residual_correction", 0.5),
    ],
)
def test_gemma4_post_attention_norm_gradients_reach_memory_and_gate(
    memory_fusion_placement: str,
    memory_fusion_residual_scale: float,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    model = Gemma4TextModel(make_gemma4_shared_kv_config()).train()
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
            memory_fusion_placement=memory_fusion_placement,
            memory_fusion_residual_scale=memory_fusion_residual_scale,
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    freeze_non_delta_mem_params(model)
    output = model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
    (output.last_hidden_state * torch.randn_like(output.last_hidden_state)).sum().backward()

    wrapped = model.layers[3].self_attn
    required = {
        "memory_v_proj": wrapped.memory_v_proj,
        "beta_proj": wrapped.beta_proj,
        "delta_o_proj": wrapped.delta_o_proj,
        "memory_fusion_hidden_weight": wrapped.memory_fusion_hidden_weight,
        "memory_fusion_read_weight": wrapped.memory_fusion_read_weight,
        "memory_fusion_bias": wrapped.memory_fusion_bias,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.float().norm().item() > 0.0, name
    assert any(
        parameter.grad is not None and parameter.grad.float().norm().item() > 0.0
        for parameter in wrapped.hrm_rwkv7_core.parameters()
    )
    assert all(parameter.grad is None for parameter in wrapped.base.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.layers[3].post_attention_layernorm.parameters()
    )


def test_post_attention_norm_rejects_non_gemma_and_non_o_targets_atomically() -> None:
    qwen_model = ToyAttentionModel()
    original_qwen = qwen_model.self_attn
    with pytest.raises(ValueError, match="Gemma4-only"):
        attach_delta_mem(
            qwen_model,
            HFDeltaMemConfig(
                rank=2,
                delta_heads=("o",),
                memory_fusion_placement="post_attention_norm",
                target_layers=(0,),
            ),
        )
    assert qwen_model.self_attn is original_qwen

    gemma_model = Gemma4TextModel(make_gemma4_shared_kv_config())
    original_gemma = [layer.self_attn for layer in gemma_model.layers]
    with pytest.raises(ValueError, match="O-only"):
        attach_delta_mem(
            gemma_model,
            HFDeltaMemConfig(
                rank=2,
                delta_heads=("q", "o"),
                memory_fusion_placement="post_attention_norm",
                target_layers=(0, 1, 2, 3),
            ),
        )
    assert [layer.self_attn for layer in gemma_model.layers] == original_gemma


def test_post_attention_norm_attach_rolls_back_on_hook_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Gemma4TextModel(make_gemma4_shared_kv_config())
    original = [layer.self_attn for layer in model.layers]

    def fail_hook_registration(*_args, **_kwargs):
        raise RuntimeError("injected hook failure")

    monkeypatch.setattr(
        model.layers[1].post_attention_layernorm,
        "register_forward_hook",
        fail_hook_registration,
    )
    with pytest.raises(RuntimeError, match="injected hook failure"):
        attach_delta_mem(
            model,
            HFDeltaMemConfig(
                rank=2,
                memory_backend="rwkv_ms",
                delta_heads=("o",),
                memory_fusion_placement="post_attention_norm",
                target_layers=(0, 1, 2, 3),
            ),
        )

    assert [layer.self_attn for layer in model.layers] == original
    assert not model.layers[0].post_attention_layernorm._forward_hooks


def test_gemma4_empty_target_layers_preserve_legacy_non_shared_scope() -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    model = Gemma4TextModel(make_gemma4_shared_kv_config())

    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            target_modules=("self_attn",),
        ),
    )

    assert replaced == ["layers.0.self_attn", "layers.1.self_attn"]
    assert all(
        not isinstance(model.layers[layer_idx].self_attn, DeltaMemAttention)
        for layer_idx in (2, 3)
    )


@pytest.mark.parametrize("unsupported_head", ["q", "k", "v"])
def test_attach_delta_mem_rejects_non_o_heads_for_gemma4_shared_kv_atomically(
    unsupported_head: str,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    model = Gemma4TextModel(make_gemma4_shared_kv_config())
    original_attention = [layer.self_attn for layer in model.layers]

    with pytest.raises(
        ValueError,
        match=rf"only O-residual.*unsupported delta heads.*'{unsupported_head}'",
    ):
        attach_delta_mem(
            model,
            HFDeltaMemConfig(
                rank=2,
                output_init="zero",
                memory_backend="rwkv_ms",
                delta_heads=("o", unsupported_head),
                target_layers=(0, 1, 2, 3),
                target_modules=("self_attn",),
            ),
        )

    assert [layer.self_attn for layer in model.layers] == original_attention
    assert all(not isinstance(layer.self_attn, DeltaMemAttention) for layer in model.layers)


@pytest.mark.parametrize(
    "memory_fusion_placement",
    [
        "attention_output",
        "post_attention_norm",
        "normalized_residual_correction",
    ],
)
def test_gemma4_shared_kv_o_only_zero_memory_matches_frozen_base(
    memory_fusion_placement: str,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    config = make_gemma4_shared_kv_config()
    base_model = Gemma4TextModel(config).eval()
    wrapped_model = copy.deepcopy(base_model).eval()
    attach_delta_mem(
        wrapped_model,
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_placement=memory_fusion_placement,
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    freeze_non_delta_mem_params(wrapped_model)
    set_delta_mem_write_enabled(wrapped_model, False)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.inference_mode():
        expected = base_model(input_ids=input_ids, use_cache=False).last_hidden_state
        actual = wrapped_model(input_ids=input_ids, use_cache=False).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for layer in wrapped_model.layers:
        wrapped = layer.self_attn
        assert isinstance(wrapped, DeltaMemAttention)
        assert all(not parameter.requires_grad for parameter in wrapped.base.parameters())
        assert wrapped.delta_state is not None
        assert torch.count_nonzero(wrapped.delta_state) == 0
        assert wrapped.last_delta_o_norm is not None
        assert wrapped.last_delta_o_norm.item() == 0.0
        assert wrapped._pending_post_attention_delta is None


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "memory_fusion_placement",
    [
        "attention_output",
        "post_attention_norm",
        "normalized_residual_correction",
    ],
)
def test_gemma4_shared_kv_o_only_cached_decode_matches_base(
    attn_implementation: str,
    memory_fusion_placement: str,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    config = make_gemma4_shared_kv_config()
    config._attn_implementation = attn_implementation
    base_model = Gemma4TextModel(config).eval()
    wrapped_model = copy.deepcopy(base_model).eval()
    attach_delta_mem(
        wrapped_model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_placement=memory_fusion_placement,
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    prefill_ids = torch.tensor([[1, 2, 3]])
    decode_ids = torch.tensor([[4]])

    with torch.inference_mode():
        base_prefill = base_model(input_ids=prefill_ids, use_cache=True)
        wrapped_prefill = wrapped_model(input_ids=prefill_ids, use_cache=True)
        base_decode = base_model(
            input_ids=decode_ids,
            past_key_values=base_prefill.past_key_values,
            use_cache=True,
        )
        wrapped_decode = wrapped_model(
            input_ids=decode_ids,
            past_key_values=wrapped_prefill.past_key_values,
            use_cache=True,
        )

    torch.testing.assert_close(
        wrapped_prefill.last_hidden_state,
        base_prefill.last_hidden_state,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        wrapped_decode.last_hidden_state,
        base_decode.last_hidden_state,
        atol=1e-6,
        rtol=1e-6,
    )
    assert base_decode.past_key_values.get_seq_length() == 4
    assert wrapped_decode.past_key_values.get_seq_length() == 4
    assert all(
        layer.self_attn._pending_post_attention_delta is None
        for layer in wrapped_model.layers
    )


def test_gemma4_shared_kv_o_only_memory_and_gate_receive_gradients() -> None:
    if Gemma4TextAttention is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    config = make_gemma4_shared_kv_config()
    wrapped = DeltaMemAttention(
        Gemma4TextAttention(config, layer_idx=2),
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
        ),
    )
    freeze_non_delta_mem_params(ToyAttentionModel(module=wrapped))
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_embeddings = make_position_embeddings(
        batch_size=hidden_states.size(0),
        seq_len=hidden_states.size(1),
        head_dim=wrapped.head_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    shared_kv_states = {
        "sliding_attention": (
            torch.randn(2, config.num_key_value_heads, 5, wrapped.head_dim),
            torch.randn(2, config.num_key_value_heads, 5, wrapped.head_dim),
        )
    }

    output, _ = wrapped(
        hidden_states,
        position_embeddings,
        make_causal_attention_mask(torch.ones(2, 5, dtype=torch.long)),
        shared_kv_states=shared_kv_states,
    )
    (output * torch.randn_like(output)).sum().backward()

    required = {
        "memory_v_proj": wrapped.memory_v_proj,
        "delta_o_proj": wrapped.delta_o_proj,
        "memory_fusion_hidden_weight": wrapped.memory_fusion_hidden_weight,
        "memory_fusion_read_weight": wrapped.memory_fusion_read_weight,
        "memory_fusion_bias": wrapped.memory_fusion_bias,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.float().norm().item() > 0.0, name
    assert all(not parameter.requires_grad for parameter in wrapped.base.parameters())
    assert all(parameter.grad is None for parameter in wrapped.base.parameters())


@pytest.mark.parametrize(
    "memory_fusion_placement",
    [
        "attention_output",
        "post_attention_norm",
        "normalized_residual_correction",
    ],
)
def test_gemma4_shared_kv_o_only_adapter_round_trip(
    tmp_path: Path,
    memory_fusion_placement: str,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(0)
    config = make_gemma4_shared_kv_config()
    source = Gemma4TextModel(config)
    target = copy.deepcopy(source)
    delta_config = HFDeltaMemConfig(
        rank=2,
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=2,
        rwkv_ms_chunk_size=2,
        delta_heads=("o",),
        memory_fusion_mode="content_gated_add",
        memory_fusion_placement=memory_fusion_placement,
        target_layers=(0, 1, 2, 3),
        target_modules=("self_attn",),
    )
    attach_delta_mem(source, delta_config)
    attach_delta_mem(target, delta_config)
    with torch.no_grad():
        for _, module in source.named_modules():
            if isinstance(module, DeltaMemAttention):
                module.memory_fusion_hidden_weight.normal_()
                module.memory_fusion_read_weight.normal_()

    save_delta_mem_adapter(source, tmp_path, delta_config)
    adapter_path = tmp_path / "delta_mem_adapter.pt"
    adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    torch.save(dict(reversed(list(adapter_state.items()))), adapter_path)
    loaded_config = load_delta_mem_adapter(target, tmp_path)

    assert loaded_config.target_layers == (0, 1, 2, 3)
    assert loaded_config.memory_fusion_placement == memory_fusion_placement
    source_modules = dict(source.named_modules())
    target_modules = dict(target.named_modules())
    for module_name, source_module in source_modules.items():
        if not isinstance(source_module, DeltaMemAttention):
            continue
        target_module = target_modules[module_name]
        assert isinstance(target_module, DeltaMemAttention)
        for parameter_name, source_parameter in source_module.named_parameters():
            if parameter_name.startswith("base."):
                continue
            target_parameter = dict(target_module.named_parameters())[parameter_name]
            torch.testing.assert_close(target_parameter, source_parameter)
        assert (
            source_module._post_attention_norm_hook_handle is not None
        ) is (
            memory_fusion_placement
            in {"post_attention_norm", "normalized_residual_correction"}
        )
        assert (
            target_module._post_attention_norm_hook_handle is not None
        ) is (
            memory_fusion_placement
            in {"post_attention_norm", "normalized_residual_correction"}
        )


@pytest.mark.parametrize(
    "memory_fusion_placement",
    [
        "attention_output",
        "post_attention_norm",
        "normalized_residual_correction",
    ],
)
def test_frozen_mlp_activation_checkpointing_recomputes_only_mlps(
    memory_fusion_placement: str,
) -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")

    class CountingMLP(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module
            self.forward_calls = 0

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.module(hidden_states)

    torch.manual_seed(0)
    model = Gemma4TextModel(make_gemma4_shared_kv_config()).train()
    for layer in model.layers:
        layer.mlp = CountingMLP(layer.mlp)
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_placement=memory_fusion_placement,
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    freeze_non_delta_mem_params(model)

    checkpointed = experimental_train.checkpoint_frozen_mlp_activations(model)
    output = model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
    positions_after_forward = {
        name: module.rwkv_ms_positions.clone()
        for name, module in model.named_modules()
        if isinstance(module, DeltaMemAttention)
    }
    states_after_forward = {
        name: module.delta_state.clone()
        for name, module in model.named_modules()
        if isinstance(module, DeltaMemAttention)
    }
    (output.last_hidden_state * torch.randn_like(output.last_hidden_state)).sum().backward()

    assert checkpointed == [f"layers.{layer_idx}.mlp" for layer_idx in range(4)]
    for layer in model.layers:
        assert isinstance(
            layer.mlp,
            experimental_train.FrozenMLPActivationCheckpointWrapper,
        )
        assert isinstance(layer.mlp.module, CountingMLP)
        assert layer.mlp.module.forward_calls == 2
    assert len(positions_after_forward) == 4
    for name, module in model.named_modules():
        if not isinstance(module, DeltaMemAttention):
            continue
        assert torch.equal(module.rwkv_ms_positions, positions_after_forward[name])
        assert torch.equal(module.delta_state, states_after_forward[name])
        assert module.rwkv_ms_positions.tolist() == [5]
    tail = model.layers[3].self_attn
    assert isinstance(tail, DeltaMemAttention)
    for name, parameter in {
        "memory_v_proj": tail.memory_v_proj,
        "beta_proj": tail.beta_proj,
        "delta_o_proj": tail.delta_o_proj,
    }.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.float().norm().item() > 0.0, name


def test_frozen_mlp_activation_checkpointing_eval_and_no_grad_are_direct() -> None:
    class CountingMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.forward_calls = 0

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return torch.tanh(self.proj(hidden_states))

    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = make_delta_module(memory_backend="rwkv_ms", delta_heads=("o",))
            self.mlp = CountingMLP()

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([TinyLayer()])

    model = TinyModel()
    freeze_non_delta_mem_params(model)
    original_mlp = model.layers[0].mlp
    hidden_states = torch.randn(2, 3, 8, requires_grad=True)
    expected = original_mlp(hidden_states)
    original_mlp.forward_calls = 0

    checkpointed = experimental_train.checkpoint_frozen_mlp_activations(model)
    model.eval()
    actual_eval = model.layers[0].mlp(hidden_states)

    assert checkpointed == ["layers.0.mlp"]
    torch.testing.assert_close(actual_eval, expected)
    assert original_mlp.forward_calls == 1

    original_mlp.forward_calls = 0
    model.train()
    with torch.no_grad():
        actual_no_grad = model.layers[0].mlp(hidden_states)
    torch.testing.assert_close(actual_no_grad, expected)
    assert original_mlp.forward_calls == 1


def test_frozen_mlp_activation_checkpointing_rejects_trainable_mlp_atomically() -> None:
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = make_delta_module(memory_backend="rwkv_ms", delta_heads=("o",))
            self.mlp = torch.nn.Linear(8, 8)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
    for parameter in model.layers[0].mlp.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="trainable parameters: layers.1.mlp"):
        experimental_train.checkpoint_frozen_mlp_activations(model)

    assert all(isinstance(layer.mlp, torch.nn.Linear) for layer in model.layers)


def test_frozen_mlp_activation_checkpointing_rejects_missing_decoder_mlp() -> None:
    class TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = make_delta_module(memory_backend="rwkv_ms", delta_heads=("o",))

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([TinyLayer()])

    with pytest.raises(ValueError, match="missing MLP"):
        experimental_train.checkpoint_frozen_mlp_activations(model)


def test_gemma4_sdpa_sliding_attention_writes_beyond_window() -> None:
    if Gemma4TextConfig is None or Gemma4TextModel is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    sequence_length = 513
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        sliding_window=512,
    )
    config._attn_implementation = "sdpa"
    model = Gemma4TextModel(config).eval()
    replaced = attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=128,
            target_layers=(0,),
            target_modules=("self_attn",),
        ),
    )

    input_ids = torch.randint(0, config.vocab_size, (1, sequence_length))
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
        )

    wrapped = model.layers[0].self_attn
    assert replaced == ["layers.0.self_attn"]
    assert isinstance(wrapped, DeltaMemAttention)
    assert output.last_hidden_state.shape == (1, sequence_length, config.hidden_size)
    assert wrapped.rwkv_ms_positions is not None
    assert wrapped.rwkv_ms_positions.tolist() == [sequence_length]
    assert wrapped.delta_state is not None
    assert torch.count_nonzero(wrapped.delta_state) > 0


def test_reset_state_restores_independent_sample_behavior() -> None:
    module = make_delta_module(output_init="random")
    fresh = make_delta_module(output_init="random")
    fresh.load_state_dict(copy.deepcopy(module.state_dict()))

    first_sample = torch.randn(1, 3, 8)
    second_sample = torch.randn(1, 3, 8)
    first_position_embeddings = make_position_embeddings(
        batch_size=first_sample.size(0),
        seq_len=first_sample.size(1),
        head_dim=module.base.head_dim,
        device=first_sample.device,
        dtype=first_sample.dtype,
    )
    second_position_embeddings = make_position_embeddings(
        batch_size=second_sample.size(0),
        seq_len=second_sample.size(1),
        head_dim=module.base.head_dim,
        device=second_sample.device,
        dtype=second_sample.dtype,
    )

    _ = module(first_sample, first_position_embeddings, None)
    module.reset_state()
    output_after_reset = module(second_sample, second_position_embeddings, None)[0]
    fresh_output = fresh(second_sample, second_position_embeddings, None)[0]

    assert torch.allclose(output_after_reset, fresh_output)


def test_padding_mask_prevents_padded_tokens_from_updating_state() -> None:
    module = make_delta_module(output_init="random")
    standalone = make_delta_module(output_init="random")
    standalone.load_state_dict(copy.deepcopy(module.state_dict()))

    short = torch.randn(1, 3, 8)
    long = torch.randn(1, 5, 8)
    padded_short = torch.cat([short, torch.randn(1, 2, 8)], dim=1)
    batch = torch.cat([padded_short, long], dim=0)
    attention_mask = make_causal_attention_mask(torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]))
    standalone_attention_mask = make_causal_attention_mask(torch.tensor([[1, 1, 1]]))
    batch_position_embeddings = make_position_embeddings(
        batch_size=batch.size(0),
        seq_len=batch.size(1),
        head_dim=module.base.head_dim,
        device=batch.device,
        dtype=batch.dtype,
    )
    standalone_position_embeddings = make_position_embeddings(
        batch_size=short.size(0),
        seq_len=short.size(1),
        head_dim=standalone.base.head_dim,
        device=short.device,
        dtype=short.dtype,
    )

    batch_output = module(batch, batch_position_embeddings, attention_mask)[0]
    standalone_output = standalone(
        short,
        standalone_position_embeddings,
        standalone_attention_mask,
    )[0]

    assert torch.allclose(batch_output[0, : short.size(1)], standalone_output[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(module.delta_state[0], standalone.delta_state[0], atol=1e-5, rtol=1e-5)


def test_disable_writes_preserves_delta_state_during_read_phase() -> None:
    module = make_delta_module(output_init="random")
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = module(x, position_embeddings, None)
    state_before = module.delta_state.clone()

    set_delta_mem_write_enabled(module, False)
    _ = module(x, position_embeddings, None)
    set_delta_mem_write_enabled(module, True)

    assert torch.allclose(module.delta_state, state_before)


def test_disable_writes_preserves_rwkv_ms_state_during_read_phase() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        rwkv_ms_chunk_size=2,
    )
    x = torch.randn(1, 5, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = module(x, position_embeddings, None)
    state_before = module.delta_state.clone()
    positions_before = module.rwkv_ms_positions.clone()
    previous_source_before = module.rwkv_ms_previous_source.clone()

    set_delta_mem_write_enabled(module, False)
    _ = module(x, position_embeddings, None)
    set_delta_mem_write_enabled(module, True)

    assert torch.allclose(module.delta_state, state_before)
    assert torch.equal(module.rwkv_ms_positions, positions_before)
    assert torch.equal(module.rwkv_ms_previous_source, previous_source_before)


def test_rwkv_ms_read_context_mask_gates_memory_without_changing_write_validity() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        delta_heads=("o",),
    ).eval()
    write_x = torch.randn(1, 5, 8)
    write_position_embeddings = make_position_embeddings(
        batch_size=write_x.size(0),
        seq_len=write_x.size(1),
        head_dim=module.base.head_dim,
        device=write_x.device,
        dtype=write_x.dtype,
    )
    with torch.no_grad():
        _ = module(write_x, write_position_embeddings, None)
    baseline = copy.deepcopy(module)
    assert baseline.delta_state is not None
    baseline.delta_state.zero_()

    read_x = torch.randn(1, 4, 8)
    read_position_embeddings = make_position_embeddings(
        batch_size=read_x.size(0),
        seq_len=read_x.size(1),
        head_dim=module.base.head_dim,
        device=read_x.device,
        dtype=read_x.dtype,
    )
    context_mask = torch.tensor([[True, False, True, True]])
    attention_mask = make_causal_attention_mask(torch.tensor([[1, 1, 1, 0]]))
    for active in (module, baseline):
        set_delta_mem_write_enabled(active, False)
        set_delta_mem_read_context_mask(active, context_mask)

    with torch.no_grad():
        output, _ = module(read_x, read_position_embeddings, attention_mask)
        baseline_output, _ = baseline(read_x, read_position_embeddings, attention_mask)
    memory_correction = output - baseline_output

    assert torch.count_nonzero(memory_correction[:, [0, 2]]).item() > 0
    assert torch.equal(
        memory_correction[:, [1, 3]],
        torch.zeros_like(memory_correction[:, [1, 3]]),
    )
    assert module.last_read_routes is not None
    assert torch.equal(
        module.last_read_routes[:, [1, 3]],
        torch.zeros_like(module.last_read_routes[:, [1, 3]]),
    )


def test_read_representation_capture_collects_live_gated_scaled_delta_o() -> None:
    modules = [
        make_delta_module(
            output_init="random",
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
            memory_fusion_gate_init=0.25,
        )
        for _ in range(2)
    ]
    model = torch.nn.Module()
    model.add_module("first", modules[0])
    model.add_module("second", modules[1])
    batch_size, seq_len, hidden_size = 2, 4, 8
    capture_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    set_delta_mem_read_representation_capture_mask(model, capture_mask)

    read_features = []
    expected = {}
    for name, module in zip(("first", "second"), modules):
        features = torch.randn(
            batch_size,
            seq_len,
            module.state_read_dim,
            requires_grad=True,
        )
        read_features.append(features)
        scaled_delta_o = module._project_delta_head(
            features,
            module.delta_o_proj,
            "o",
        )
        assert scaled_delta_o is not None
        expected[name] = torch.stack(
            [scaled_delta_o[0, 1], scaled_delta_o[1, 2]],
        ) * 0.25
        module._fuse_delta_o_output(
            torch.randn(batch_size, seq_len, hidden_size),
            scaled_delta_o,
            torch.randn(batch_size, seq_len, hidden_size),
            features,
            valid_mask,
        )
        module.delta_state = torch.randn(batch_size, module.rank, module.rank)

    representations = collect_delta_mem_read_representations(model)

    assert list(representations) == ["first", "second"]
    for name, representation in representations.items():
        assert representation.shape == (batch_size, hidden_size)
        assert representation.requires_grad
        assert representation.grad_fn is not None
        assert torch.allclose(representation, expected[name])

    sum(value.square().sum() for value in representations.values()).backward()
    for module, features in zip(modules, read_features):
        assert features.grad is not None
        assert torch.count_nonzero(features.grad[capture_mask]) > 0
        assert torch.count_nonzero(features.grad[~capture_mask]) == 0
        assert module.delta_o_proj.grad is not None
        assert torch.count_nonzero(module.delta_o_proj.grad) > 0
        for gate_parameter in (
            module.memory_fusion_bias,
            module.memory_fusion_hidden_weight,
            module.memory_fusion_read_weight,
        ):
            assert gate_parameter.grad is not None
            assert torch.isfinite(gate_parameter.grad).all()
            assert torch.count_nonzero(gate_parameter.grad) > 0

    assert set(get_delta_mem_online_state(model)) == {"first", "second"}

    set_delta_mem_read_representation_capture_mask(model, None)
    assert collect_delta_mem_read_representations(model) == {}
    assert all(module.last_read_representation is None for module in modules)


def test_read_representation_capture_validates_selection_shape_and_validity() -> None:
    module = make_delta_module(output_init="random", delta_heads=("o",))
    model = ToyAttentionModel(module)

    with pytest.raises(ValueError, match=r"shape \[batch, sequence\]"):
        set_delta_mem_read_representation_capture_mask(
            model,
            torch.tensor([False, True, False]),
        )
    with pytest.raises(ValueError, match="exactly one token per batch row"):
        set_delta_mem_read_representation_capture_mask(
            model,
            torch.tensor([[False, False, False], [True, True, False]]),
        )

    hidden_states = torch.randn(1, 4, 8)
    reads = torch.randn(1, 4, module.state_read_dim)
    delta_o = module._project_delta_head(reads, module.delta_o_proj, "o")
    assert delta_o is not None

    set_delta_mem_read_representation_capture_mask(
        model,
        torch.tensor([[False, True, False]]),
    )
    with pytest.raises(ValueError, match="must match the model token shape"):
        module._fuse_delta_o_output(
            torch.randn_like(hidden_states),
            delta_o,
            hidden_states,
            reads,
            torch.ones(1, 4, dtype=torch.bool),
        )

    set_delta_mem_read_representation_capture_mask(
        model,
        torch.tensor([[False, False, True, False]]),
    )
    with pytest.raises(ValueError, match="outside the valid read mask"):
        module._fuse_delta_o_output(
            torch.randn_like(hidden_states),
            delta_o,
            hidden_states,
            reads,
            torch.tensor([[True, True, False, False]]),
        )
    assert module.last_read_representation is None

    module.reset_state()
    assert module.read_representation_capture_mask is None
    assert module.last_read_representation is None
    assert collect_delta_mem_read_representations(model) == {}


def test_read_representation_capture_is_inert_without_a_mask() -> None:
    module = make_delta_module(output_init="random", delta_heads=("o",))
    model = ToyAttentionModel(module)
    hidden_states = torch.randn(1, 3, 8)
    reads = torch.randn(1, 3, module.state_read_dim)
    delta_o = module._project_delta_head(reads, module.delta_o_proj, "o")
    assert delta_o is not None

    module._fuse_delta_o_output(
        torch.randn_like(hidden_states),
        delta_o,
        hidden_states,
        reads,
        torch.ones(1, 3, dtype=torch.bool),
    )

    assert module.last_read_representation is None
    assert collect_delta_mem_read_representations(model) == {}


def test_rwkv_ms_read_time_mix_keeps_live_gradient_to_final_write_source() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        delta_heads=("o",),
    ).eval()

    for _ in range(2):
        module.reset_state()
        set_delta_mem_write_enabled(module, True)
        module.zero_grad(set_to_none=True)
        write_x = torch.randn(1, 5, 8, requires_grad=True)
        write_position_embeddings = make_position_embeddings(
            batch_size=write_x.size(0),
            seq_len=write_x.size(1),
            head_dim=module.base.head_dim,
            device=write_x.device,
            dtype=write_x.dtype,
        )
        _ = module(write_x, write_position_embeddings, None)
        assert module.delta_state is not None
        assert module.rwkv_ms_previous_source is not None
        assert module.rwkv_ms_previous_source.requires_grad

        # Remove the ordinary memory-state edge so this checks only the
        # write-final-source -> read time-mix path.
        module.delta_state = module.delta_state.detach()
        set_delta_mem_write_enabled(module, False)
        read_x = torch.randn(1, 4, 8)
        read_position_embeddings = make_position_embeddings(
            batch_size=read_x.size(0),
            seq_len=read_x.size(1),
            head_dim=module.base.head_dim,
            device=read_x.device,
            dtype=read_x.dtype,
        )
        output, _ = module(read_x, read_position_embeddings, None)
        output.square().sum().backward()

        assert write_x.grad is not None
        assert torch.isfinite(write_x.grad).all()
        assert torch.count_nonzero(write_x.grad[:, -1]).item() > 0


def test_rwkv_ms_read_only_reads_are_chunk_invariant() -> None:
    module = make_delta_module(
        output_init="random",
        rank=2,
        num_state_heads=2,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
    )
    assert module.hrm_rwkv7_core is not None
    torch.nn.init.normal_(module.hrm_rwkv7_core.output.weight)
    state = torch.randn(2, 2, 3, 2, 2)
    memory_source = torch.randn(2, 7, module.state_read_dim)
    token_mask = torch.tensor(
        [[True, True, False, True, True, True, False], [True, False, True, True, False, True, True]]
    )
    module.rwkv_ms_previous_source = torch.randn(2, module.state_read_dim)
    previous_source = module.rwkv_ms_previous_source.clone()

    full_reads = module._rwkv_ms_token_state_reads(state, memory_source, token_mask)
    chunked_reads = []
    for start, end in ((0, 2), (2, 3), (3, 6), (6, 7)):
        chunked_reads.append(
            module._rwkv_ms_token_state_reads(
                state,
                memory_source[:, start:end],
                token_mask[:, start:end],
            )
        )

    assert torch.allclose(torch.cat(chunked_reads, dim=1), full_reads, atol=1e-6, rtol=1e-5)
    assert torch.equal(module.rwkv_ms_previous_source, previous_source)


def test_rwkv_ms_batch_transition_resets_all_online_state_tensors() -> None:
    module = make_delta_module(memory_backend="rwkv_ms")
    write_x = torch.randn(2, 3, 8)
    write_position_embeddings = make_position_embeddings(
        batch_size=write_x.size(0),
        seq_len=write_x.size(1),
        head_dim=module.base.head_dim,
        device=write_x.device,
        dtype=write_x.dtype,
    )
    _ = module(write_x, write_position_embeddings, None)
    assert module.rwkv_ms_positions.shape == (2,)
    assert module.rwkv_ms_previous_source.shape == (2, module.state_read_dim)

    set_delta_mem_write_enabled(module, False)
    read_x = torch.randn(1, 2, 8)
    read_position_embeddings = make_position_embeddings(
        batch_size=read_x.size(0),
        seq_len=read_x.size(1),
        head_dim=module.base.head_dim,
        device=read_x.device,
        dtype=read_x.dtype,
    )
    _ = module(read_x, read_position_embeddings, None)

    assert module.delta_state.shape[0] == 1
    assert module.rwkv_ms_positions.shape == (1,)
    assert module.rwkv_ms_previous_source.shape == (1, module.state_read_dim)
    assert torch.count_nonzero(module.delta_state) == 0
    assert torch.count_nonzero(module.rwkv_ms_positions) == 0
    assert torch.count_nonzero(module.rwkv_ms_previous_source) == 0


def test_rwkv_ms_chunked_updates_match_full_sequence() -> None:
    full_sequence = make_delta_module(
        output_init="random",
        rank=2,
        num_state_heads=2,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        rwkv_ms_chunk_size=2,
    )
    chunked = copy.deepcopy(full_sequence)
    token_by_token = copy.deepcopy(full_sequence)
    x = torch.randn(2, 7, 8)

    full_position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=full_sequence.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = full_sequence(x, full_position_embeddings, None)

    for start, end in ((0, 3), (3, 4), (4, 6), (6, 7)):
        chunk = x[:, start:end]
        chunk_position_embeddings = make_position_embeddings(
            batch_size=chunk.size(0),
            seq_len=chunk.size(1),
            head_dim=chunked.base.head_dim,
            device=chunk.device,
            dtype=chunk.dtype,
        )
        _ = chunked(chunk, chunk_position_embeddings, None)

    for token_idx in range(x.size(1)):
        token = x[:, token_idx : token_idx + 1]
        token_position_embeddings = make_position_embeddings(
            batch_size=token.size(0),
            seq_len=token.size(1),
            head_dim=token_by_token.base.head_dim,
            device=token.device,
            dtype=token.dtype,
        )
        _ = token_by_token(token, token_position_embeddings, None)

    assert torch.allclose(chunked.delta_state, full_sequence.delta_state, atol=1e-6, rtol=1e-5)
    assert torch.equal(chunked.rwkv_ms_positions, full_sequence.rwkv_ms_positions)
    assert torch.allclose(
        chunked.rwkv_ms_previous_source,
        full_sequence.rwkv_ms_previous_source,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        token_by_token.delta_state,
        full_sequence.delta_state,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(token_by_token.rwkv_ms_positions, full_sequence.rwkv_ms_positions)
    assert torch.allclose(
        token_by_token.rwkv_ms_previous_source,
        full_sequence.rwkv_ms_previous_source,
        atol=1e-6,
        rtol=1e-5,
    )


def test_rwkv_ms_reset_clears_streaming_predecessor() -> None:
    module = make_delta_module(memory_backend="rwkv_ms")
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = module(x, position_embeddings, None)
    assert module.rwkv_ms_previous_source is not None

    module.reset_state()

    assert module.delta_state is None
    assert module.rwkv_ms_positions is None
    assert module.rwkv_ms_previous_source is None


def test_rwkv_ms_online_state_round_trips_streaming_predecessor() -> None:
    source = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
    )
    source_model = torch.nn.Module()
    source_model.add_module("attn", source)
    x = torch.randn(2, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=source.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = source(x, position_embeddings, None)
    snapshot = get_delta_mem_online_state(source_model)

    target = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
    )
    target_model = torch.nn.Module()
    target_model.add_module("attn", target)
    load_delta_mem_online_state(target_model, snapshot)

    assert set(snapshot) == {
        "attn",
        "attn.__rwkv_ms_positions",
        "attn.__rwkv_ms_previous_source",
    }
    assert torch.equal(target.delta_state, source.delta_state)
    assert torch.equal(target.rwkv_ms_positions, source.rwkv_ms_positions)
    assert torch.equal(target.rwkv_ms_previous_source, source.rwkv_ms_previous_source)
    assert not target.rwkv_ms_previous_source.requires_grad
    assert target.rwkv_ms_previous_source.grad_fn is None
    for tensor in snapshot.values():
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_rwkv_ms_trainer_capture_and_scatter_include_streaming_predecessor() -> None:
    module = make_delta_module(memory_backend="rwkv_ms")
    model = torch.nn.Module()
    model.add_module("attn", module)
    x = torch.randn(2, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = module(x, position_embeddings, None)
    active_state = module.delta_state.clone()
    active_positions = module.rwkv_ms_positions.clone()
    active_previous_source = module.rwkv_ms_previous_source.clone()
    trainer = object.__new__(experimental_train.DeltaMemTrainer)

    captured = trainer._capture_live_online_state(model)
    trainer._scatter_episode_state(
        model,
        active_rows=torch.tensor([True, False, True, False]),
        batch_size=4,
    )

    assert set(captured) == {
        "attn",
        "attn.__rwkv_ms_positions",
        "attn.__rwkv_ms_previous_source",
    }
    assert torch.equal(captured["attn.__rwkv_ms_previous_source"], active_previous_source)
    assert torch.equal(module.delta_state[[0, 2]], active_state)
    assert torch.equal(module.rwkv_ms_positions[[0, 2]], active_positions)
    assert torch.equal(module.rwkv_ms_previous_source[[0, 2]], active_previous_source)
    assert torch.count_nonzero(module.delta_state[[1, 3]]) == 0
    assert torch.count_nonzero(module.rwkv_ms_positions[[1, 3]]) == 0
    assert torch.count_nonzero(module.rwkv_ms_previous_source[[1, 3]]) == 0


def test_rwkv_ms_core_uses_physical_backbone_depth_for_initialization() -> None:
    base = make_qwen3_attention(layer_idx=3)
    base.config.num_hidden_layers = 64
    module = DeltaMemAttention(
        base,
        HFDeltaMemConfig(rank=2, memory_backend="rwkv_ms"),
    )

    core = module.hrm_rwkv7_core
    assert core is not None
    assert core.layer_id == 3
    assert core.n_layer == 64
    ddd = torch.arange(core.dim, dtype=torch.float32) / core.dim
    expected_x_w = 1.0 - torch.pow(ddd, 0.9 * (1.0 - 3.0 / 64.0))
    assert torch.allclose(core.x_w, expected_x_w)


def test_rwkv_ms_semantics_version_migrates_legacy_configs() -> None:
    fresh = HFDeltaMemConfig(memory_backend="rwkv_ms")
    legacy_payload = fresh.to_dict()
    legacy_payload.pop("rwkv_ms_semantics_version")

    assert fresh.rwkv_ms_semantics_version == 2
    assert HFDeltaMemConfig.from_dict(legacy_payload).rwkv_ms_semantics_version == 1
    assert HFDeltaMemConfig.from_dict(fresh.to_dict()).rwkv_ms_semantics_version == 2
    assert HFDeltaMemConfig.from_dict({}).rwkv_ms_semantics_version == 2
    with pytest.raises(ValueError, match="rwkv_ms_semantics_version"):
        HFDeltaMemConfig(memory_backend="rwkv_ms", rwkv_ms_semantics_version=3)


def test_memory_fusion_config_preserves_legacy_add_and_round_trips(
    tmp_path: Path,
) -> None:
    gated = HFDeltaMemConfig(
        memory_backend="rwkv_ms",
        delta_heads=("o",),
        memory_fusion_mode="content_gated_add",
        memory_fusion_gate_init=0.25,
        memory_fusion_placement="post_attention_norm",
    )
    gated.save_pretrained(tmp_path)
    legacy_payload = gated.to_dict()
    legacy_payload.pop("memory_fusion_mode")
    legacy_payload.pop("memory_fusion_gate_init")
    legacy_payload.pop("memory_fusion_placement")

    assert HFDeltaMemConfig.from_pretrained(tmp_path) == gated
    assert HFDeltaMemConfig.from_dict(legacy_payload).memory_fusion_mode == "add"
    assert HFDeltaMemConfig.from_dict(legacy_payload).memory_fusion_gate_init == 0.1
    assert (
        HFDeltaMemConfig.from_dict(legacy_payload).memory_fusion_placement
        == "attention_output"
    )
    with pytest.raises(ValueError, match="memory fusion mode"):
        HFDeltaMemConfig(memory_fusion_mode="unknown")
    with pytest.raises(ValueError, match="memory_fusion_gate_init"):
        HFDeltaMemConfig(memory_fusion_gate_init=0.0)
    with pytest.raises(ValueError, match="fusion placement"):
        HFDeltaMemConfig(memory_fusion_placement="inside_mlp")


def test_normalized_residual_scale_defaults_and_round_trips(tmp_path: Path) -> None:
    assert HFDeltaMemConfig().memory_fusion_residual_scale == 1.0
    config = HFDeltaMemConfig(
        memory_backend="rwkv_ms",
        delta_heads=("o",),
        memory_fusion_placement="normalized-residual-correction",
        memory_fusion_residual_scale="0.875",
    )
    config.save_pretrained(tmp_path)

    assert config.memory_fusion_placement == "normalized_residual_correction"
    assert config.memory_fusion_residual_scale == 0.875
    assert HFDeltaMemConfig.from_pretrained(tmp_path) == config
    legacy_payload = config.to_dict()
    legacy_payload.pop("memory_fusion_residual_scale")
    assert HFDeltaMemConfig.from_dict(legacy_payload).memory_fusion_residual_scale == 1.0


@pytest.mark.parametrize(
    "invalid_scale",
    [-0.001, 1.001, float("nan"), float("inf"), float("-inf")],
)
def test_normalized_residual_scale_rejects_invalid_bounds(invalid_scale: float) -> None:
    with pytest.raises(ValueError, match="memory_fusion_residual_scale"):
        HFDeltaMemConfig(memory_fusion_residual_scale=invalid_scale)


def test_content_gated_delta_o_matches_manual_interpolation_and_reports_stats() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        delta_heads=("o",),
        memory_fusion_mode="content_gated_add",
        memory_fusion_gate_init=0.25,
    ).eval()
    x = torch.randn(2, 4, module.hidden_size)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(2, 4, dtype=torch.long))

    module.memory_fusion_mode = "add"
    additive_output, _ = module(x, position_embeddings, attention_mask)
    module.reset_state()
    module.active_delta_heads = frozenset()
    base_output, _ = module(x, position_embeddings, attention_mask)
    module.reset_state()
    module.active_delta_heads = frozenset({"o"})
    module.memory_fusion_mode = "content_gated_add"
    gated_output, _ = module(x, position_embeddings, attention_mask)

    expected = base_output + 0.25 * (additive_output - base_output)
    assert torch.allclose(gated_output, expected, atol=1e-6, rtol=1e-6)
    assert module.last_delta_o_gate_mean.item() == pytest.approx(0.25, abs=1e-6)
    assert module.last_delta_o_gate_min.item() == pytest.approx(0.25, abs=1e-6)
    assert module.last_delta_o_gate_max.item() == pytest.approx(0.25, abs=1e-6)
    assert module.last_delta_o_ratio.item() > 0.0
    assert module.last_fused_delta_o_ratio.item() == pytest.approx(
        module.last_delta_o_ratio.item() * 0.25,
        rel=1e-5,
        abs=1e-7,
    )
    stats = collect_delta_mem_output_ratio_stats(ToyAttentionModel(module))
    assert stats["content_gated_modules"] == 1
    assert stats["mean_delta_o_gate"] == pytest.approx(0.25, abs=1e-6)
    assert stats["mean_fused_delta_o_ratio"] == pytest.approx(
        module.last_fused_delta_o_ratio.item()
    )


def test_content_gated_delta_o_zero_state_matches_base_for_full_sequence() -> None:
    torch.manual_seed(0)
    base = make_qwen3_attention()
    wrapped_base = make_qwen3_attention()
    wrapped_base.load_state_dict(copy.deepcopy(base.state_dict()))
    module = DeltaMemAttention(
        wrapped_base,
        HFDeltaMemConfig(
            rank=2,
            memory_backend="rwkv_ms",
            output_init="random",
            delta_heads=("o",),
            memory_fusion_mode="content_gated_add",
        ),
    ).eval()
    with torch.no_grad():
        module.memory_fusion_hidden_weight.normal_()
        module.memory_fusion_read_weight.normal_()
        module.memory_fusion_bias.fill_(3.0)
    module.set_write_enabled(False)
    x = torch.randn(2, 5, module.hidden_size)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(2, 5, dtype=torch.long))

    base_output, _ = base(x, position_embeddings, attention_mask)
    wrapped_output, _ = module(x, position_embeddings, attention_mask)

    assert torch.equal(module.delta_state, torch.zeros_like(module.delta_state))
    assert torch.allclose(base_output, wrapped_output, atol=1e-6, rtol=1e-6)
    assert module.last_delta_o_norm.item() == 0.0
    assert module.last_fused_delta_o_norm.item() == 0.0


def test_content_gated_delta_o_first_step_gradients_reach_memory_and_gate() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        delta_heads=("o",),
        memory_fusion_mode="content_gated_add",
        memory_fusion_gate_init=0.1,
    )
    x = torch.randn(2, 5, module.hidden_size)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(2, 5, dtype=torch.long))
    output, _ = module(x, position_embeddings, attention_mask)
    loss = (output * torch.randn_like(output)).sum()
    loss.backward()

    required = {
        "memory_v_proj": module.memory_v_proj,
        "delta_o_proj": module.delta_o_proj,
        "beta_proj": module.beta_proj,
        "memory_fusion_hidden_weight": module.memory_fusion_hidden_weight,
        "memory_fusion_read_weight": module.memory_fusion_read_weight,
        "memory_fusion_bias": module.memory_fusion_bias,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.float().norm().item() > 0.0, name
    assert module.hrm_rwkv7_core is not None
    assert any(
        parameter.grad is not None and parameter.grad.float().norm().item() > 0.0
        for parameter in module.hrm_rwkv7_core.parameters()
        if parameter.requires_grad
    )


def test_rwkv_ms_zero_state_readout_ignores_stored_group_norm_bias() -> None:
    module = make_delta_module(
        rank=2,
        num_state_heads=2,
        memory_backend="rwkv_ms",
    )
    core = module.hrm_rwkv7_core
    assert core is not None
    torch.nn.init.normal_(core.output.weight)
    stored_state = core.state_dict()
    stored_state["ln_x.bias"].fill_(7.0)
    core.load_state_dict(stored_state, strict=True)
    reads = torch.zeros(2, 3, module.state_read_dim, requires_grad=True)
    gate = torch.randn_like(reads)

    readout = core.readout(reads, gate)
    readout.sum().backward()

    assert torch.count_nonzero(core.ln_x.bias) > 0
    assert core.ln_x.bias.requires_grad is False
    assert core.ln_x.bias.grad is None
    assert torch.count_nonzero(readout) == 0


def test_rwkv_ms_skips_and_freezes_unused_memory_qk_projections() -> None:
    module = make_delta_module(memory_backend="rwkv_ms")
    with torch.no_grad():
        module.memory_q_proj.fill_(torch.nan)
        module.memory_k_proj.fill_(torch.nan)

    memory_q, memory_k, memory_v, _, _ = module._memory_sequence_projections(
        torch.randn(2, 3, module.hidden_size)
    )

    assert module.memory_q_proj.requires_grad is False
    assert module.memory_k_proj.requires_grad is False
    assert module.is_trainable_parameter("memory_q_proj") is False
    assert module.is_trainable_parameter("memory_k_proj") is False
    assert memory_q.data_ptr() == memory_v.data_ptr()
    assert memory_k.data_ptr() == memory_v.data_ptr()
    assert torch.isfinite(memory_v).all()

    model = ToyAttentionModel(module)
    trainable_names = freeze_non_delta_mem_params(model)
    assert all(not name.endswith(("memory_q_proj", "memory_k_proj")) for name in trainable_names)
    assert module.hrm_rwkv7_core is not None
    assert module.hrm_rwkv7_core.ln_x.bias.requires_grad is False


def test_rwkv_ms_output_init_scale_round_trips_and_is_exposed_by_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_dir = tmp_path / "adapter"
    config = HFDeltaMemConfig(
        memory_backend="rwkv_ms",
        output_init="base_slice_fixed",
        rwkv_ms_output_init_scale=0.013,
    )
    config.save_pretrained(adapter_dir)

    assert HFDeltaMemConfig.from_pretrained(adapter_dir) == config
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft.py",
            "--model-path",
            "model",
            "--output-dir",
            str(tmp_path / "output"),
            "--memory-backend",
            "rwkv_ms",
            "--output-init",
            "base_slice_fixed",
            "--rwkv-ms-output-init-scale",
            "0.013",
            "--memory-fusion-mode",
            "content_gated_add",
            "--memory-fusion-gate-init",
            "0.25",
            "--memory-fusion-placement",
            "post_attention_norm",
        ],
    )

    parsed = parse_args()
    assert parsed.rwkv_ms_output_init_scale == 0.013
    assert parsed.memory_fusion_mode == "content_gated_add"
    assert parsed.memory_fusion_gate_init == 0.25
    assert parsed.memory_fusion_placement == "post_attention_norm"


def test_rwkv_ms_v2_normalizes_the_live_source_but_v1_remains_raw() -> None:
    legacy = make_delta_module(memory_backend="rwkv_ms", rwkv_ms_semantics_version=1)
    fixed = make_delta_module(memory_backend="rwkv_ms", rwkv_ms_semantics_version=2)
    hidden_states = torch.randn(2, 3, fixed.hidden_size) * 1000.0
    raw = torch.nn.functional.linear(hidden_states, fixed.memory_v_proj)

    _, _, legacy_source, _, _ = legacy._memory_sequence_projections(hidden_states)
    _, _, fixed_source, _, _ = fixed._memory_sequence_projections(hidden_states)
    expected_fixed = fixed._normalize_memory_projection(raw, force=True)

    assert torch.allclose(legacy_source, raw)
    assert torch.allclose(fixed_source, expected_fixed)
    assert torch.isfinite(fixed_source).all()
    assert torch.all(fixed_source.norm(dim=-1) <= 1.0 + 1e-6)


def test_rwkv_ms_v2_beta_gates_writes_without_scaling_carry() -> None:
    fixed = make_delta_module(memory_backend="rwkv_ms", rwkv_ms_semantics_version=2)
    legacy = make_delta_module(memory_backend="rwkv_ms", rwkv_ms_semantics_version=1)
    beta = torch.full((1, 3, fixed.rank, 1), 0.25)
    low_lambda = torch.full_like(beta, 0.1)
    high_lambda = torch.full_like(beta, 0.9)

    fixed_low = fixed._rwkv_ms_update_coefficients(beta, low_lambda)
    fixed_high = fixed._rwkv_ms_update_coefficients(beta, high_lambda)
    legacy_low = legacy._rwkv_ms_update_coefficients(beta, low_lambda)
    legacy_high = legacy._rwkv_ms_update_coefficients(beta, high_lambda)

    assert torch.count_nonzero(fixed_low[0] - 1.0) == 0
    for low, high in zip(fixed_low, fixed_high):
        assert torch.equal(low, high)
    assert not torch.equal(legacy_low[0], legacy_high[0])


def test_rwkv_ms_v2_cosine_routes_are_scale_invariant_and_finite() -> None:
    module = make_delta_module(
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        rwkv_ms_semantics_version=2,
    )
    slot_reads = torch.randn(2, 1, 3, module.rank)
    slot_reads[0, :, 0] = 0.0
    query = torch.randn(2, 1, module.rank)
    scales = torch.tensor([0.5, 3.0, 11.0]).view(1, 1, 3, 1)

    routes = module._rwkv_ms_read_routes(slot_reads, query, None)
    scaled_routes = module._rwkv_ms_read_routes(slot_reads * scales, query * 7.0, None)

    assert torch.isfinite(routes).all()
    assert torch.allclose(routes, scaled_routes, atol=1e-6, rtol=1e-6)


def test_rwkv_ms_v2_recurrence_matches_direct_fp32_reference() -> None:
    module = make_delta_module(
        output_init="random",
        rank=2,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=1,
        rwkv_ms_semantics_version=2,
    )
    core = module.hrm_rwkv7_core
    assert core is not None
    source = torch.randn(1, 4, module.state_read_dim)
    initial_state = torch.randn(1, 1, 1, module.rank, module.rank)
    beta = torch.ones(1, source.size(1), module.rank, 1)
    lam = torch.rand_like(beta)
    features = core.project(source, previous_x=torch.zeros(1, module.state_read_dim))
    k_seq = module._rwkv_ms_project_heads(features.k).float()
    v_seq = module._rwkv_ms_project_heads(features.v).float()
    w_seq = module._rwkv_ms_project_heads(features.w).float()
    a_seq = module._rwkv_ms_project_heads(features.a).float()
    b_seq = module._rwkv_ms_project_heads(features.b).float()
    expected = initial_state.float()
    for token_idx in range(source.size(1)):
        decay = torch.exp(-torch.exp(w_seq[:, token_idx]))
        correction_read = torch.einsum(
            "bhsij,bhj->bhsi",
            expected,
            a_seq[:, token_idx],
        )
        write_outer = (
            v_seq[:, token_idx].unsqueeze(2).unsqueeze(-1)
            * k_seq[:, token_idx].unsqueeze(2).unsqueeze(-2)
        )
        correction_outer = (
            correction_read.unsqueeze(-1)
            * b_seq[:, token_idx].unsqueeze(2).unsqueeze(-2)
        )
        expected = (
            decay.unsqueeze(2).unsqueeze(-2) * expected
            + write_outer
            + module.rwkv_ms_erase_gate * correction_outer
        )

    actual, _ = module._rwkv_ms_scan(initial_state, source, beta, lam)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_rwkv_ms_first_ce_backward_reaches_writer_and_controller() -> None:
    module = make_delta_module(
        output_init="base_slice_fixed",
        memory_backend="rwkv_ms",
        rwkv_ms_output_init_scale=0.02,
        delta_heads=("o",),
    )
    core = module.hrm_rwkv7_core
    assert core is not None
    assert torch.count_nonzero(core.output.weight) > 0
    freeze_non_delta_mem_params(ToyAttentionModel(module))

    hidden_states = torch.randn(2, 5, module.hidden_size)
    position_embeddings = make_position_embeddings(
        batch_size=hidden_states.size(0),
        seq_len=hidden_states.size(1),
        head_dim=module.base.head_dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output, _ = module(hidden_states, position_embeddings, None)
    vocab_weight = torch.randn(11, module.hidden_size)
    labels = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(output, vocab_weight).reshape(-1, 11),
        labels.reshape(-1),
    )

    loss.backward()

    assert module.memory_q_proj.grad is None
    assert module.memory_k_proj.grad is None
    assert module.memory_v_proj.grad is not None
    assert module.memory_v_proj.grad.norm().item() > 0
    assert module.beta_proj.grad is not None
    assert module.beta_proj.grad.norm().item() > 0
    assert core.value.weight.grad is not None
    assert core.value.weight.grad.norm().item() > 0


def test_rwkv_ms_state_dtype_conversion_preserves_memory() -> None:
    module = make_delta_module(
        output_init="random",
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        rwkv_ms_chunk_size=2,
    )
    initial = torch.randn(1, 1, 3, module.rank, module.rank, dtype=torch.float32)
    module.delta_state = initial.clone()

    converted = module._ensure_state(
        batch_size=1,
        device=initial.device,
        dtype=torch.bfloat16,
    )

    assert converted.dtype == torch.float32
    assert torch.equal(converted, initial)


def test_base_slice_initialization_gives_controller_gradients() -> None:
    module = make_delta_module(output_init="base_slice")
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    loss = module(x, position_embeddings, None)[0].sum()
    loss.backward()

    assert module.memory_q_proj.grad is not None and module.memory_q_proj.grad.norm().item() > 0
    assert module.memory_k_proj.grad is not None and module.memory_k_proj.grad.norm().item() > 0
    assert module.memory_v_proj.grad is not None and module.memory_v_proj.grad.norm().item() > 0
    assert module.beta_proj.grad is not None and module.beta_proj.grad.norm().item() > 0


def test_base_slice_zero_state_matches_base_even_at_high_rank() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank8 = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank64 = make_qwen3_attention(**common_kwargs)
    wrapped_base_rank8.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped_base_rank64.load_state_dict(copy.deepcopy(base.state_dict()))
    wrapped_rank8 = DeltaMemAttention(
        wrapped_base_rank8,
        HFDeltaMemConfig(rank=8, output_init="base_slice", rankwise_gates=True),
    )
    wrapped_rank64 = DeltaMemAttention(
        wrapped_base_rank64,
        HFDeltaMemConfig(rank=64, output_init="base_slice", rankwise_gates=True),
    )
    x = torch.randn(2, 1, common_kwargs["hidden_size"])
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )

    base_out, _ = base(x, position_embeddings, None)
    rank8_out, _ = wrapped_rank8(x, position_embeddings, None)
    rank64_out, _ = wrapped_rank64(x, position_embeddings, None)

    assert torch.allclose(base_out, rank8_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_out, rank64_out, atol=1e-6, rtol=1e-6)



def test_base_slice_initialization_strength_grows_with_rank() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base_rank8 = make_qwen3_attention(**common_kwargs)
    base_rank64 = make_qwen3_attention(**common_kwargs)
    base_rank64.load_state_dict(copy.deepcopy(base_rank8.state_dict()))
    rank8_module = DeltaMemAttention(
        base_rank8,
        HFDeltaMemConfig(rank=8, output_init="base_slice", rankwise_gates=True),
    )
    rank64_module = DeltaMemAttention(
        base_rank64,
        HFDeltaMemConfig(rank=64, output_init="base_slice", rankwise_gates=True),
    )

    rank8_q_nonzero = int((rank8_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_q_nonzero = int((rank64_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank8_o_nonzero = int((rank8_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_o_nonzero = int((rank64_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())

    assert rank8_q_nonzero == 8
    assert rank64_q_nonzero == 64
    assert rank8_o_nonzero == 8
    assert rank64_o_nonzero == 64
    assert rank64_module.delta_q_proj.float().norm().item() > rank8_module.delta_q_proj.float().norm().item()
    assert rank64_module.delta_o_proj.float().norm().item() > rank8_module.delta_o_proj.float().norm().item()


def test_base_slice_fixed_keeps_initialization_width_constant_across_ranks() -> None:
    torch.manual_seed(0)
    common_kwargs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
    }
    base_rank8 = make_qwen3_attention(**common_kwargs)
    base_rank64 = make_qwen3_attention(**common_kwargs)
    base_rank64.load_state_dict(copy.deepcopy(base_rank8.state_dict()))
    rank8_module = DeltaMemAttention(
        base_rank8,
        HFDeltaMemConfig(
            rank=8,
            output_init="base_slice_fixed",
            base_slice_ref_width=8,
            rankwise_gates=True,
        ),
    )
    rank64_module = DeltaMemAttention(
        base_rank64,
        HFDeltaMemConfig(
            rank=64,
            output_init="base_slice_fixed",
            base_slice_ref_width=8,
            rankwise_gates=True,
        ),
    )

    rank8_q_nonzero = int((rank8_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_q_nonzero = int((rank64_module.delta_q_proj.abs().sum(dim=0) > 0).sum().item())
    rank8_o_nonzero = int((rank8_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())
    rank64_o_nonzero = int((rank64_module.delta_o_proj.abs().sum(dim=0) > 0).sum().item())

    assert rank8_q_nonzero == 8
    assert rank64_q_nonzero == 8
    assert rank8_o_nonzero == 8
    assert rank64_o_nonzero == 8
    assert torch.allclose(
        rank8_module.delta_q_proj[:, :8],
        rank64_module.delta_q_proj[:, :8],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(rank64_module.delta_q_proj[:, 8:], torch.zeros_like(rank64_module.delta_q_proj[:, 8:]))
    assert torch.allclose(rank64_module.delta_o_proj[:, 8:], torch.zeros_like(rank64_module.delta_o_proj[:, 8:]))


def test_delta_o_rmsnorm_adds_trainable_weight() -> None:
    module = make_delta_module(output_init="zero", delta_heads=("o",), delta_o_rmsnorm=True)

    assert module.delta_o_rmsnorm is True
    assert hasattr(module, "delta_o_rmsnorm_weight")
    assert module.delta_o_rmsnorm_weight.shape == (module.base.o_proj.out_features,)
    assert module.is_trainable_parameter("delta_o_rmsnorm_weight") is True


def test_delta_o_rmsnorm_matches_manual_rms_norm() -> None:
    module = make_delta_module(output_init="zero", delta_heads=("o",), delta_o_rmsnorm=True)
    delta_o = torch.randn(2, 3, module.base.o_proj.out_features)

    actual = module._apply_delta_o_rmsnorm(delta_o)
    expected = torch.nn.functional.rms_norm(
        delta_o.float(),
        (delta_o.shape[-1],),
        weight=module.delta_o_rmsnorm_weight.float(),
        eps=module.delta_o_rmsnorm_eps,
    ).to(dtype=delta_o.dtype)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_memory_reader_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_reader_layers"):
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            memory_reader_layers=(1,),
            memory_reader_hidden_size=4,
        )


def test_synthetic_kv_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=8,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="synthetic_kv",
            synthetic_memory_slots=4,
        )


def test_latent_context_readout_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=8,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="latent_context",
            latent_memory_layers=(0,),
            latent_memory_slots=4,
            latent_gate_init=0.01,
        )


def test_training_cli_rejects_latent_context_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["delta_sft.py", "--memory-readout-mode", "latent_context"])
    with pytest.raises(SystemExit):
        parse_args()


def test_training_cli_rejects_latent_prefix_margin_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["delta_sft.py", "--memory-readout-mode", "latent_context"],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_collect_latent_stats_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_readout_mode='delta'"):
        HFDeltaMemConfig(
            rank=2,
            output_init="zero",
            delta_heads=(),
            memory_readout_mode="latent_context",
            latent_memory_layers=(0,),
            latent_memory_slots=2,
        )


def test_training_cli_rejects_synthetic_kv_readout_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "delta_sft.py",
            "--memory-readout-mode",
            "synthetic_kv",
            "--synthetic-memory-slots",
            "4",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_layer_indices_supports_off_and_csv() -> None:
    assert parse_layer_indices("off") == ()
    assert parse_layer_indices(" 24, 25,26 ") == (24, 25, 26)


def test_rankwise_gates_expand_gate_parameters() -> None:
    module = make_delta_module(output_init="base_slice", rank=3, rankwise_gates=True)
    assert module.beta_proj.shape == (3, 8)
    assert module.beta_bias.shape == (3,)


def test_write_sparsity_regularization_is_positive_after_forward() -> None:
    model = ToyAttentionModel(
        DeltaMemAttention(
            make_qwen3_attention(),
            HFDeltaMemConfig(rank=2, output_init="base_slice", rankwise_gates=True),
        )
    )
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=model.self_attn.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = model(x, position_embeddings)
    penalty = get_delta_mem_write_regularization(model, target=0.0)
    assert penalty.item() > 0


def test_tokenize_messages_supervises_all_assistant_turns() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    tokenized = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
    )
    final_only = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="final_assistant_only",
    )

    prefix_user = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_assistant_1 = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_user_2 = tokenizer.apply_chat_template(
        messages[:3],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    prefix_assistant_2 = tokenizer.apply_chat_template(
        messages[:4],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    assistant_1_span = len(prefix_assistant_1) - len(prefix_user)
    assistant_2_span = len(prefix_assistant_2) - len(prefix_user_2)

    supervised_all = sum(label != -100 for label in tokenized["labels"])
    supervised_final = sum(label != -100 for label in final_only["labels"])

    assert supervised_all == assistant_1_span + assistant_2_span
    assert supervised_final == assistant_2_span


def test_tokenize_messages_keeps_tail_supervision_when_truncated() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "u" * 40},
        {"role": "assistant", "content": "a" * 10},
        {"role": "user", "content": "v" * 40},
        {"role": "assistant", "content": "b" * 12},
    ]

    full = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=4096,
        assistant_loss_mode="all_assistant_turns",
    )
    truncated = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=80,
        assistant_loss_mode="all_assistant_turns",
    )

    assert truncated["input_ids"] == full["input_ids"][-80:]
    assert truncated["labels"] == full["labels"][-80:]
    assert any(label != -100 for label in truncated["labels"])

    last_assistant_tokens = len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0).tolist()
    ) - len(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).squeeze(0).tolist()
    )
    assert sum(label != -100 for label in truncated["labels"][-last_assistant_tokens:]) > 0


def test_build_episode_training_examples_splits_write_and_visible_context() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    episodes = build_episode_training_examples(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=1,
        max_write_length=1024,
        include_sentence_ids=True,
    )

    assert len(episodes) == 2
    assert episodes[0]["write_input_ids"] == []

    expected_write = tokenizer.apply_chat_template(
        messages[:3],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    expected_read = tokenize_messages_for_sft(
        tokenizer,
        [messages[0], messages[3], messages[4]],
        max_length=1024,
        assistant_loss_mode="final_assistant_only",
    )
    expected_teacher = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=2048,
        assistant_loss_mode="final_assistant_only",
    )

    assert episodes[1]["write_input_ids"] == expected_write
    assert episodes[1]["input_ids"] == expected_read["input_ids"]
    assert episodes[1]["labels"] == expected_read["labels"]
    assert episodes[1]["teacher_input_ids"] == expected_teacher["input_ids"]
    assert episodes[1]["teacher_attention_mask"] == expected_teacher["attention_mask"]
    assert episodes[1]["teacher_labels"] == expected_teacher["labels"]
    assert episodes[1]["episode_target_message_index"] == 4
    assert len(episodes[1]["write_message_ids"]) == len(episodes[1]["write_input_ids"])
    assert len(episodes[1]["write_sentence_ids"]) == len(episodes[1]["write_input_ids"])
    assert set(episodes[1]["write_message_ids"]) == {-1, 0, 1}
    assert set(episodes[1]["write_sentence_ids"]) == {-1, 0, 1}
    assert episodes[1]["state_only_write_input_ids"] == expected_write
    assert episodes[1]["state_only_input_ids"] == expected_read["input_ids"]
    assert len(episodes[1]["state_only_write_message_ids"]) == len(episodes[1]["state_only_write_input_ids"])
    assert len(episodes[1]["state_only_write_sentence_ids"]) == len(episodes[1]["state_only_write_input_ids"])
    assert set(episodes[1]["state_only_write_message_ids"]) == {-1, 0, 1}
    assert set(episodes[1]["state_only_write_sentence_ids"]) == {-1, 0, 1}


def test_episode_teacher_masks_extra_long_target_prefix() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "x" * 160},
    ]

    episode = build_episode_training_examples(
        tokenizer,
        messages,
        max_length=40,
        assistant_loss_mode="final_assistant_only",
        episode_recent_messages=1,
        max_write_length=60,
        include_sentence_ids=False,
    )[0]
    unmasked_teacher = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=100,
        assistant_loss_mode="final_assistant_only",
    )
    student_targets = [label for label in episode["labels"][1:] if label != -100]
    teacher_targets = [label for label in episode["teacher_labels"][1:] if label != -100]

    assert teacher_targets == student_targets
    assert sum(label != -100 for label in unmasked_teacher["labels"][1:]) > len(teacher_targets)


def test_episode_teacher_starts_at_retained_write_boundary() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u" * 80},
        {"role": "assistant", "content": "a" * 80},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "target" * 4},
    ]
    max_write_length = 30
    max_read_length = 40

    episode = build_episode_training_examples(
        tokenizer,
        messages,
        max_length=max_read_length,
        assistant_loss_mode="final_assistant_only",
        episode_recent_messages=1,
        max_write_length=max_write_length,
        include_sentence_ids=False,
    )[0]
    full_write_input_ids = tokenizer.apply_chat_template(
        messages[:3],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    canonical_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    ).squeeze(0).tolist()
    retained_boundary = len(full_write_input_ids) - max_write_length
    expected_start = max(
        retained_boundary,
        len(canonical_input_ids) - max_write_length - max_read_length,
    )

    assert episode["write_input_ids"] == full_write_input_ids[-max_write_length:]
    assert episode["teacher_input_ids"] == canonical_input_ids[expected_start:]


def test_experimental_sentence_span_builder_handles_non_prefix_stable_sentence_chunks() -> None:
    tokenizer = SentenceSpanUnstableTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Sentence one. Sentence two."},
        {"role": "assistant", "content": "ok"},
    ]

    episodes = experimental_train.build_episode_training_examples(
        tokenizer,
        messages,
        max_length=1024,
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=0,
        max_write_length=1024,
        include_sentence_ids=True,
    )

    assert len(episodes) == 1
    assert len(episodes[0]["write_sentence_ids"]) == len(episodes[0]["write_input_ids"])
    active_sentence_ids = [value for value in episodes[0]["write_sentence_ids"] if value >= 0]
    assert active_sentence_ids
    assert set(active_sentence_ids) == {0, 1}


def test_runtime_sentence_span_builder_handles_non_prefix_stable_sentence_chunks() -> None:
    tokenizer = SentenceSpanUnstableTokenizer()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Sentence one. Sentence two."},
    ]

    input_ids, message_ids, sentence_ids = pipeline._tokenize_chat_messages_with_write_span_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        include_sentence_ids=True,
    )

    assert len(input_ids) == len(message_ids) == len(sentence_ids)
    active_sentence_ids = [value for value in sentence_ids if value >= 0]
    assert active_sentence_ids
    assert set(active_sentence_ids) == {0, 1}


def test_prepare_tokenized_dataset_reuses_saved_cache(tmp_path: Path) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ]
    )
    args = Namespace(
        tokenized_cache=True,
        tokenized_dataset_root=tmp_path / "tokenized",
        dataset_name="fake",
        dataset_split="train",
        train_file=None,
        training_mode="episode",
        assistant_loss_mode="all_assistant_turns",
        episode_recent_messages=1,
        max_length=128,
        max_write_length=64,
        group_by_length=True,
        dataset_num_proc=1,
        memory_write_granularity="token",
    )
    tokenizer = FakeTokenizer()

    tokenized_1, cache_hit_1, cache_dir_1 = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=False,
        local_rank=-1,
    )
    tokenized_2, cache_hit_2, cache_dir_2 = prepare_tokenized_dataset(
        args,
        dataset,
        tokenizer,
        distributed=False,
        local_rank=-1,
    )

    assert cache_hit_1 is False
    assert cache_hit_2 is True
    assert cache_dir_1 == cache_dir_2
    assert tokenized_1._fingerprint == tokenized_2._fingerprint
    assert tokenized_1[0]["input_ids"] == tokenized_2[0]["input_ids"]
    assert "write_input_ids" in tokenized_1.column_names


def test_episode_collator_builds_teacher_inputs() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = 0
    collator = EpisodeCausalLMCollator(tokenizer)
    features = [
        {
            "write_input_ids": [1, 2],
            "write_attention_mask": [1, 1],
            "write_message_ids": [-1, 0],
            "write_sentence_ids": [-1, 0],
            "input_ids": [3, 4, 5],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 4, 5],
            "teacher_input_ids": [9, 10, 3, 4, 5],
            "teacher_attention_mask": [1, 1, 1, 1, 1],
            "teacher_labels": [-100, -100, -100, 4, 5],
            "state_only_write_input_ids": [1, 2],
            "state_only_write_attention_mask": [1, 1],
            "state_only_write_message_ids": [-1, 0],
            "state_only_write_sentence_ids": [-1, 0],
            "state_only_input_ids": [3, 4],
            "state_only_attention_mask": [1, 1],
            "state_only_labels": [-100, 4],
        },
        {
            "write_input_ids": [6],
            "write_attention_mask": [1],
            "write_message_ids": [0],
            "write_sentence_ids": [0],
            "input_ids": [7, 8],
            "attention_mask": [1, 1],
            "labels": [-100, 8],
            "teacher_input_ids": [11, 7, 8],
            "teacher_attention_mask": [1, 1, 1],
            "teacher_labels": [-100, -100, 8],
            "state_only_write_input_ids": [6],
            "state_only_write_attention_mask": [1],
            "state_only_write_message_ids": [0],
            "state_only_write_sentence_ids": [0],
            "state_only_input_ids": [7],
            "state_only_attention_mask": [1],
            "state_only_labels": [-100],
        },
    ]

    batch = collator(features)

    assert torch.equal(batch["write_lengths"], torch.tensor([2, 1], dtype=torch.long))
    assert torch.equal(batch["read_lengths"], torch.tensor([3, 2], dtype=torch.long))
    assert batch["write_message_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["write_sentence_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["state_only_write_message_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["state_only_write_sentence_ids"].tolist() == [[-1, 0], [0, -1]]
    assert batch["teacher_input_ids"].tolist() == [[9, 10, 3, 4, 5], [11, 7, 8, 0, 0]]
    assert batch["teacher_attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["teacher_labels"].tolist() == [
        [-100, -100, -100, 4, 5],
        [-100, -100, 8, -100, -100],
    ]
    assert batch["full_input_ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]]
    assert batch["full_attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["full_labels"].tolist() == [[-100, -100, -100, 4, 5], [-100, -100, 8, -100, -100]]


def test_ssw_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="message_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 2, 2)


def test_sentence_mean_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="sentence_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    set_delta_mem_write_sentence_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 2, 2)


def test_multi_head_state_forward_runs() -> None:
    module = make_delta_module(
        rank=2,
        num_state_heads=4,
        delta_heads=("q", "o"),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 4, 2, 2)
    assert module.delta_q_proj.shape[-1] == 8
    assert module.delta_o_proj.shape[-1] == 8


def test_rwkv_ms_backend_forward_runs_with_routes() -> None:
    module = make_delta_module(
        rank=2,
        num_state_heads=2,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=3,
        rwkv_ms_chunk_size=2,
        delta_heads=("q", "o"),
    )
    x = torch.randn(1, 5, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=5,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 5, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 2, 3, 2, 2)
    assert module.rwkv_ms_positions is not None
    assert module.rwkv_ms_positions.tolist() == [5]
    assert module.last_write_routes is not None
    assert module.last_write_routes.shape == (1, 5, 3)
    expected_write_routes = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    assert torch.allclose(module.last_write_routes.cpu(), expected_write_routes)
    assert module.last_read_routes is not None
    assert module.last_read_routes.shape == (1, 5, 3)
    assert module.hrm_rwkv7_core is not None
    assert any(name.startswith("hrm_rwkv7_core.") for name, _ in module.named_parameters())


def test_msw_state_with_ssw_writer_forward_runs() -> None:
    module = DeltaMemAttention(
        make_qwen3_attention(),
        HFDeltaMemConfig(
            rank=2,
            num_state_heads=4,
            output_init="base_slice",
            delta_heads=("q", "o"),
            memory_write_granularity="message_mean",
        ),
    )
    set_delta_mem_write_enabled(module, True)
    set_delta_mem_write_message_ids(module, torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    x = torch.randn(1, 4, 8)
    position_embeddings = make_position_embeddings(
        batch_size=1,
        seq_len=4,
        head_dim=module.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    attention_mask = make_causal_attention_mask(torch.ones(1, 4, dtype=torch.long))

    output, _ = module(x, position_embeddings, attention_mask)

    assert output.shape == x.shape
    assert module.delta_state is not None
    assert module.delta_state.shape == (1, 4, 2, 2)
    assert module.delta_q_proj.shape[-1] == 8
    assert module.delta_o_proj.shape[-1] == 8


def test_runtime_ingest_sets_write_message_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    write_message_ids_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_message_ids",
        lambda model, message_ids: write_message_ids_calls.append(
            None if message_ids is None else message_ids.detach().cpu().clone()
        ),
    )
    session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    session.messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    full_ids = session._tokenize_messages(session.messages, add_generation_prompt=False)
    session._ingest_full_ids(full_ids)

    non_none_calls = [call for call in write_message_ids_calls if call is not None]
    assert len(non_none_calls) == 1
    assert set(non_none_calls[0].squeeze(0).tolist()) == {-1, 0, 1}
    assert write_message_ids_calls[-1] is None


def test_runtime_write_message_ids_reuses_snapshot_cache_for_appended_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper_calls = 0
    original_helper = pipeline._tokenize_chat_messages_with_write_span_ids

    def counted_helper(*args, **kwargs):
        nonlocal helper_calls
        helper_calls += 1
        return original_helper(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_tokenize_chat_messages_with_write_span_ids", counted_helper)

    history_session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    history_session.messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    history_ids = history_session._tokenize_messages(history_session.messages, add_generation_prompt=False)
    history_session._ingest_full_ids(history_ids)
    assert helper_calls == 1

    snapshot = DeltaMemSessionSnapshot(
        messages=[dict(message) for message in history_session.messages],
        processed_input_ids=history_ids.squeeze(0).tolist(),
        delta_state={},
        past_key_values=None,
        write_message_ids=list(history_session.cached_write_message_ids or []),
        write_sentence_ids=list(history_session.cached_write_sentence_ids or []),
    )
    followup_session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    followup_session.load_snapshot(snapshot)
    followup_session.messages.append({"role": "user", "content": "u2"})

    prompt_ids = followup_session._tokenize_messages(
        followup_session.messages,
        add_generation_prompt=True,
    )
    prompt_message_ids = followup_session._runtime_write_message_ids(prompt_ids)

    assert prompt_message_ids is not None
    assert helper_calls == 1
    assert prompt_message_ids.shape[1] == prompt_ids.shape[1]
    assert set(prompt_message_ids.squeeze(0).tolist()) == {-1, 0, 1, 2}


def test_rwkv_ms_session_snapshot_rejects_missing_or_mismatched_semantics() -> None:
    model = ToyAttentionModel(
        make_delta_module(memory_backend="rwkv_ms", rwkv_ms_semantics_version=2)
    )
    session = DeltaMemChatSession(model=model, tokenizer=FakeTokenizer(), device="cpu")
    legacy_snapshot = DeltaMemSessionSnapshot(
        messages=[],
        processed_input_ids=[],
        delta_state={},
    )
    wrong_snapshot = DeltaMemSessionSnapshot(
        messages=[],
        processed_input_ids=[],
        delta_state={},
        rwkv_ms_semantics_versions=[1],
    )
    matching_snapshot = DeltaMemSessionSnapshot(
        messages=[],
        processed_input_ids=[],
        delta_state={},
        rwkv_ms_semantics_versions=[2],
    )

    with pytest.raises(ValueError, match="predates semantics metadata"):
        session.load_snapshot(legacy_snapshot)
    with pytest.raises(ValueError, match="semantics mismatch"):
        session.load_snapshot(wrong_snapshot)
    session.load_snapshot(matching_snapshot)



def test_runtime_generate_sets_assistant_message_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    write_message_ids_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_message_ids",
        lambda model, message_ids: write_message_ids_calls.append(
            None if message_ids is None else message_ids.detach().cpu().clone()
        ),
    )
    session = DeltaMemChatSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )
    session.messages = [{"role": "user", "content": "u1"}]

    session._decode_generate(
        torch.zeros(1, 8),
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    assistant_calls = [call for call in write_message_ids_calls if call is not None]
    assert len(assistant_calls) == 2
    assert all(torch.equal(call, torch.ones((1, 1), dtype=torch.long)) for call in assistant_calls)
    assert write_message_ids_calls[-1] is None


def test_generate_reply_preserves_raw_assistant_text() -> None:
    session = StubSession(
        model=FakeModel(),
        tokenizer=FakeTokenizer(response_text=" hello\n"),
        device="cpu",
    )

    result = session.generate_reply("remember this", max_new_tokens=2)

    assert result["assistant"] == " hello\n"
    assert result["assistant_display"] == "hello"
    assert session.messages[-1]["content"] == " hello\n"


def test_generate_reply_restores_write_enabled_after_read_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_enabled_calls = []
    monkeypatch.setattr(
        pipeline,
        "set_delta_mem_write_enabled",
        lambda model, enabled: write_enabled_calls.append(enabled),
    )
    model = FakeModel()
    session = StubSession(
        model=model,
        tokenizer=FakeTokenizer(response_text="ok"),
        device="cpu",
    )

    session.generate_reply("probe", max_new_tokens=2, write_enabled=False)

    assert write_enabled_calls == [False, True]


def test_decode_generate_does_not_commit_eos_to_history() -> None:
    model = FakeModel()
    session = DeltaMemChatSession(
        model=model,
        tokenizer=FakeTokenizer(eos_token_id=5),
        device="cpu",
    )
    session.processed_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    next_token_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    generated_ids = session._decode_generate(
        next_token_logits,
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    assert generated_ids.shape == (1, 0)
    assert torch.equal(session.processed_input_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert model.calls == 0


def test_reset_delta_mem_states_clears_wrapped_model_state() -> None:
    wrapped = ToyAttentionModel(make_delta_module(output_init="random"))
    x = torch.randn(1, 3, 8)
    position_embeddings = make_position_embeddings(
        batch_size=x.size(0),
        seq_len=x.size(1),
        head_dim=wrapped.self_attn.base.head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    _ = wrapped(x, position_embeddings)
    reset_delta_mem_states(wrapped)
    assert wrapped.self_attn.delta_state is None
