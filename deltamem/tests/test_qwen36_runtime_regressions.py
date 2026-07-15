from __future__ import annotations

from types import SimpleNamespace

import torch

import deltamem.runtime.session as runtime_session
from deltamem.model_loading import resolve_attn_implementation
from deltamem.runtime.session import (
    DeltaMemChatSession,
    _generation_eos_token_ids,
)


class PrefixSensitiveQwenTokenizer:
    name_or_path = "Qwen/Qwen3.6-27B"
    eos_token_id = 5
    pad_token_id = 0

    def __init__(self) -> None:
        self.preserve_thinking_calls: list[bool | None] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors=None,
        enable_thinking=None,
        preserve_thinking=None,
    ):
        self.preserve_thinking_calls.append(preserve_thinking)
        rendered = ""
        for index, message in enumerate(messages):
            content = message["content"]
            if (
                message["role"] == "assistant"
                and not preserve_thinking
                and any(item["role"] == "user" for item in messages[index + 1 :])
            ):
                content = content.split("</think>")[-1].lstrip()
            rendered += f"<{message['role']}>{content}</{message['role']}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        token_ids = [ord(char) for char in rendered] or [0]
        return torch.tensor([token_ids], dtype=torch.long)


class GenerationConfigModel(torch.nn.Module):
    def __init__(self, eos_token_id) -> None:
        super().__init__()
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            logits=torch.zeros(input_ids.size(0), input_ids.size(1), 8),
            past_key_values=None,
        )


def test_resolve_attn_implementation_leaves_auto_to_transformers() -> None:
    model_path = "/unused/local/model"

    assert resolve_attn_implementation(model_path, None) is None
    assert resolve_attn_implementation(model_path, "auto") is None
    assert resolve_attn_implementation(model_path, "AUTO") is None
    assert resolve_attn_implementation(model_path, "none") is None


def test_resolve_attn_implementation_preserves_explicit_request() -> None:
    model_path = "/unused/local/model"

    assert resolve_attn_implementation(model_path, "eager") == "eager"
    assert (
        resolve_attn_implementation(model_path, "flash_attention_2")
        == "flash_attention_2"
    )


def test_delta_chat_loader_resolves_auto_before_transformers(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tokenizer = object()

    class LoadedModel:
        def eval(self):
            return self

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return LoadedModel()

    monkeypatch.setattr(
        runtime_session.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        runtime_session.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        runtime_session.HFDeltaMemConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runtime_session, "attach_delta_mem", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_session, "load_delta_mem_adapter", lambda *args, **kwargs: None)

    model, loaded_tokenizer = runtime_session.load_delta_mem_chat_model(
        model_path="/models/Qwen3.6-27B",
        device="cpu",
        dtype="float32",
        attn_implementation="auto",
        adapter_dir="/unused/adapter",
    )

    assert isinstance(model, LoadedModel)
    assert loaded_tokenizer is tokenizer
    assert captured["attn_implementation"] is None


def test_qwen36_session_preserves_historical_thinking_prefix() -> None:
    tokenizer = PrefixSensitiveQwenTokenizer()
    session = DeltaMemChatSession(
        model=GenerationConfigModel(eos_token_id=None),
        tokenizer=tokenizer,
        device="cpu",
    )
    first_turn = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "<think>reasoning</think>answer one"},
    ]
    continued = [*first_turn, {"role": "user", "content": "question two"}]

    first_ids = session._tokenize_messages(first_turn, add_generation_prompt=False)
    continued_ids = session._tokenize_messages(continued, add_generation_prompt=True)

    assert tokenizer.preserve_thinking_calls == [True, True]
    assert torch.equal(first_ids[0], continued_ids[0, : first_ids.size(1)])


def test_manual_decode_honors_all_generation_and_tokenizer_eos_ids() -> None:
    tokenizer = PrefixSensitiveQwenTokenizer()
    model = GenerationConfigModel(eos_token_id=[6, 7])
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cpu")
    session.processed_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert _generation_eos_token_ids(model, tokenizer) == {5, 6, 7}

    next_token_logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    )
    generated_ids = session._decode_generate(
        next_token_logits,
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
    )

    assert generated_ids.shape == (1, 0)
    assert torch.equal(
        session.processed_input_ids,
        torch.tensor([[1, 2, 3]], dtype=torch.long),
    )
    assert model.calls == 0
