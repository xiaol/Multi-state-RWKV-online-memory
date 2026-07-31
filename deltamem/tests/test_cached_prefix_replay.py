from __future__ import annotations

import deltamem.train.cached_prefix_replay as cached_prefix_replay_module
import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    freeze_non_delta_mem_params,
    iter_delta_mem_modules,
    reset_delta_mem_states,
    set_delta_mem_read_context_mask,
    set_delta_mem_write_enabled,
)
from deltamem.train.cached_prefix_replay import cached_prefix_replay_logits
from deltamem.train.delta_sft_experimental import (
    checkpoint_frozen_mlp_activations,
)

try:
    from transformers.models.gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
except ImportError:  # pragma: no cover - depends on the Transformers version.
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None


class _CacheSensitiveMemoryModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, *, prompt_length: int) -> None:
        super().__init__(config)
        self.prompt_length = prompt_length
        self.memory_logits = torch.nn.Parameter(
            torch.linspace(-0.2, 0.2, config.vocab_size)
        )
        self.forward_trace: list[tuple[int, int]] = []
        self.logits_to_keep_trace: list[int | torch.Tensor] = []

    @staticmethod
    def _cache_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].size(-2))

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        cache_length = self._cache_length(past_key_values)
        self.forward_trace.append((cache_length, int(input_ids.size(1))))
        self.logits_to_keep_trace.append(logits_to_keep)
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        path_bias = outputs.logits.new_zeros(self.config.vocab_size)
        if cache_length > 0:
            path_bias[(cache_length + 5) % self.config.vocab_size] = 3.0
        elif input_ids.size(1) > self.prompt_length:
            # A full-prefix replay is intentionally observably different.
            path_bias[(input_ids.size(1) + 11) % self.config.vocab_size] = 3.0
        if logits_to_keep != 1:
            path_bias[(input_ids.size(1) + 3) % self.config.vocab_size] += 2.0
        outputs.logits = outputs.logits + self.memory_logits + path_bias
        return outputs


class _DifferentiableMemoryCache:
    def __init__(self, *, length: int, memory_signal: torch.Tensor) -> None:
        self.length = length
        self.memory_signal = memory_signal


class _DifferentiableCachedMemoryModel(torch.nn.Module):
    vocabulary_size = 11

    def __init__(self) -> None:
        super().__init__()
        self.memory_gain = torch.nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        past_key_values=None,
        use_cache=False,
        return_dict=False,
    ):
        assert use_cache and return_dict
        previous_length = 0 if past_key_values is None else past_key_values.length
        assert attention_mask.size(1) == previous_length + input_ids.size(1)
        if past_key_values is None:
            memory_signal = self.memory_gain * input_ids.float().sum(dim=1)
        else:
            # Cached decode can reach memory_gain only through the live prefill cache.
            memory_signal = past_key_values.memory_signal
        vocabulary_basis = torch.arange(
            self.vocabulary_size,
            dtype=memory_signal.dtype,
            device=memory_signal.device,
        )
        logits = memory_signal[:, None, None] * vocabulary_basis[None, None, :]
        logits = logits.expand(-1, input_ids.size(1), -1)
        next_memory_signal = memory_signal * (
            1.0 + input_ids.float().sum(dim=1) / 100.0
        )
        return {
            "logits": logits,
            "past_key_values": _DifferentiableMemoryCache(
                length=previous_length + input_ids.size(1),
                memory_signal=next_memory_signal,
            ),
        }


class _LegacyLogitLimitModel(_DifferentiableCachedMemoryModel):
    def __init__(self) -> None:
        super().__init__()
        self.logit_limits: list[int] = []

    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        past_key_values=None,
        use_cache=False,
        return_dict=False,
        num_logits_to_keep=0,
    ):
        self.logit_limits.append(num_logits_to_keep)
        outputs = super().forward(
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        if num_logits_to_keep == 1:
            outputs["logits"] = outputs["logits"][:, -1:]
        return outputs


class _IgnoredLogitLimitModel(_DifferentiableCachedMemoryModel):
    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        past_key_values=None,
        use_cache=False,
        return_dict=False,
        logits_to_keep=0,
    ):
        del logits_to_keep
        return super().forward(
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class _ModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _model_and_prompt() -> tuple[
    _CacheSensitiveMemoryModel,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(4)
    prompt_input_ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    model = _CacheSensitiveMemoryModel(
        GPT2Config(
            vocab_size=19,
            n_positions=16,
            n_ctx=16,
            n_embd=12,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        ),
        prompt_length=prompt_input_ids.size(1),
    ).eval()
    return model, prompt_input_ids, torch.ones_like(prompt_input_ids)


def test_cached_prefix_replay_matches_greedy_cached_generation_logits() -> None:
    model, prompt_input_ids, prompt_attention_mask = _model_and_prompt()
    generated = model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        do_sample=False,
        max_new_tokens=4,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
    )
    replay_token_ids = generated.sequences[:, prompt_input_ids.size(1) :]
    greedy_selection_logits = torch.stack(generated.logits, dim=1)
    greedy_selection_scores = torch.stack(generated.scores, dim=1)

    model.forward_trace.clear()
    model.logits_to_keep_trace.clear()
    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        replay_token_ids=replay_token_ids,
    )

    assert torch.equal(replay_logits.detach(), greedy_selection_logits)
    assert torch.equal(replay_logits.detach(), greedy_selection_scores)
    torch.testing.assert_close(replay_logits.argmax(dim=-1), replay_token_ids)
    assert model.forward_trace == [(0, 3), (3, 1), (4, 1), (5, 1)]
    assert model.logits_to_keep_trace == [1, 1, 1, 1]

    uncapped_prefill = model(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        use_cache=True,
        return_dict=True,
        logits_to_keep=0,
    )
    assert not torch.equal(
        uncapped_prefill.logits[:, -1],
        replay_logits[:, 0],
    )

    full_prefix_outputs = model(
        input_ids=generated.sequences[:, :4],
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        use_cache=False,
        return_dict=True,
        logits_to_keep=0,
    )
    assert not torch.allclose(
        full_prefix_outputs.logits[:, -1],
        replay_logits[:, 1],
    )


def test_cached_prefix_replay_keeps_memory_parameter_gradients() -> None:
    model = _DifferentiableCachedMemoryModel()
    prompt_input_ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    prompt_attention_mask = torch.ones_like(prompt_input_ids)
    replay_token_ids = torch.tensor([[2, 7, 3]], dtype=torch.long)

    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        replay_token_ids=replay_token_ids,
        require_single_logit_projection=False,
    )
    loss = F.cross_entropy(replay_logits[:, -1], replay_token_ids[:, -1])
    loss.backward()

    assert replay_logits.requires_grad
    assert model.memory_gain.grad is not None
    assert model.memory_gain.grad.abs().item() > 0.0


def test_gemma4_delta_mem_cached_replay_backpropagates_through_full_path() -> None:
    if Gemma4TextConfig is None or Gemma4ForCausalLM is None:
        pytest.skip("Gemma4 is not available in this Transformers version")
    torch.manual_seed(7)
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
        sliding_window=8,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(config).train()
    attach_delta_mem(
        model,
        HFDeltaMemConfig(
            rank=2,
            output_init="random",
            online_gain=0.2,
            memory_backend="rwkv_ms",
            rwkv_ms_num_states=2,
            rwkv_ms_chunk_size=2,
            delta_heads=("q", "o"),
            memory_fusion_mode="add",
            memory_fusion_placement="attention_output",
            target_layers=(0, 1, 2, 3),
            target_modules=("self_attn",),
        ),
    )
    freeze_non_delta_mem_params(model)
    assert checkpoint_frozen_mlp_activations(model) == [
        "model.layers.0.mlp",
        "model.layers.1.mlp",
        "model.layers.2.mlp",
        "model.layers.3.mlp",
    ]

    write_input_ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    prompt_input_ids = torch.tensor([[1, 7, 8]], dtype=torch.long)
    replay_token_ids = torch.tensor([[9, 10, 11]], dtype=torch.long)
    captured_layer_outputs: list[torch.Tensor] = []

    def retain_layer_output(_module, _inputs, outputs) -> None:
        hidden_states = outputs[0]
        hidden_states.retain_grad()
        captured_layer_outputs.append(hidden_states)

    hook = model.model.layers[0].self_attn.register_forward_hook(
        retain_layer_output
    )
    reset_delta_mem_states(model)
    model(
        input_ids=write_input_ids,
        attention_mask=torch.ones_like(write_input_ids),
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    write_states = {}
    for name, module in iter_delta_mem_modules(model):
        assert module.delta_state is not None
        module.delta_state.retain_grad()
        write_states[name] = module.delta_state

    set_delta_mem_write_enabled(model, False)
    set_delta_mem_read_context_mask(model, None)
    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=torch.ones_like(prompt_input_ids),
        replay_token_ids=replay_token_ids,
    )
    loss = F.cross_entropy(
        replay_logits[:, -1].float(),
        torch.tensor([12], dtype=torch.long),
    )
    loss.backward()
    hook.remove()

    assert replay_logits.shape == (1, 3, config.vocab_size)
    # One write, one prompt prefill, and two one-token cached decode forwards.
    assert [tuple(output.shape) for output in captured_layer_outputs] == [
        (1, 4, config.hidden_size),
        (1, 3, config.hidden_size),
        (1, 1, config.hidden_size),
        (1, 1, config.hidden_size),
    ]
    for output in captured_layer_outputs:
        assert output.grad is not None
        assert torch.isfinite(output.grad).all()
        assert output.grad.float().norm().item() > 0.0
    for state in write_states.values():
        assert state.grad is not None
        assert torch.isfinite(state.grad).all()
        assert state.grad.float().norm().item() > 0.0
    for layer in model.model.layers:
        assert layer.self_attn.memory_v_proj.grad is not None
        assert layer.self_attn.memory_v_proj.grad.float().norm().item() > 0.0


def test_cached_prefix_replay_requires_explicit_logit_projection_support() -> None:
    model = _DifferentiableCachedMemoryModel()

    with pytest.raises(
        RuntimeError,
        match="requires explicit logits_to_keep or num_logits_to_keep support",
    ):
        cached_prefix_replay_logits(
            model,
            prompt_input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
            prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
            replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
        )


def test_cached_prefix_replay_can_explicitly_allow_uncapped_logits() -> None:
    model = _DifferentiableCachedMemoryModel()

    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
        require_single_logit_projection=False,
    )

    assert replay_logits.shape == (1, 2, model.vocabulary_size)


def test_cached_prefix_replay_fails_closed_when_signature_is_uninspectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, prompt_input_ids, prompt_attention_mask = _model_and_prompt()

    def _raise_signature_error(_callable):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(
        cached_prefix_replay_module.inspect,
        "signature",
        _raise_signature_error,
    )
    with pytest.raises(
        RuntimeError,
        match="could not verify single-logit projection support",
    ):
        cached_prefix_replay_logits(
            model,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
        )


def test_cached_prefix_replay_rejects_ignored_logit_projection_keyword() -> None:
    model = _IgnoredLogitLimitModel()

    with pytest.raises(
        RuntimeError,
        match="did not honor single-logit projection",
    ):
        cached_prefix_replay_logits(
            model,
            prompt_input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
            prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
            replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
        )


def test_cached_prefix_replay_supports_legacy_logit_limit_keyword() -> None:
    model = _LegacyLogitLimitModel()
    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
    )

    assert replay_logits.shape == (1, 2, model.vocabulary_size)
    assert model.logit_limits == [1, 1]


def test_cached_prefix_replay_discovers_projection_keyword_through_wrapper() -> None:
    base_model = _LegacyLogitLimitModel()
    model = _ModuleWrapper(base_model)

    replay_logits = cached_prefix_replay_logits(
        model,
        prompt_input_ids=torch.tensor([[1, 4, 5]], dtype=torch.long),
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        replay_token_ids=torch.tensor([[2, 7]], dtype=torch.long),
    )

    assert replay_logits.shape == (1, 2, base_model.vocabulary_size)
    assert base_model.logit_limits == [1, 1]


@pytest.mark.parametrize(
    ("input_overrides", "error_type", "message"),
    [
        (
            {"prompt_input_ids": torch.tensor([1, 4, 5])},
            ValueError,
            "prompt_input_ids must have shape",
        ),
        (
            {"prompt_attention_mask": torch.tensor([[1, 0, 1]])},
            ValueError,
            "requires an unpadded prompt",
        ),
        (
            {"replay_token_ids": torch.empty((1, 0), dtype=torch.long)},
            ValueError,
            "requires at least one replay token",
        ),
        (
            {"replay_token_ids": torch.tensor([[2], [3]])},
            ValueError,
            "batches must have the same size",
        ),
        (
            {"replay_token_ids": torch.tensor([[19]])},
            ValueError,
            "exceeds the model vocabulary",
        ),
    ],
)
def test_cached_prefix_replay_rejects_invalid_shapes_and_prefixes(
    input_overrides: dict[str, torch.Tensor],
    error_type: type[Exception],
    message: str,
) -> None:
    model, prompt_input_ids, prompt_attention_mask = _model_and_prompt()
    inputs = {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "replay_token_ids": torch.tensor([[2, 7]], dtype=torch.long),
    }
    inputs.update(input_overrides)

    with pytest.raises(error_type, match=message):
        cached_prefix_replay_logits(model, **inputs)
