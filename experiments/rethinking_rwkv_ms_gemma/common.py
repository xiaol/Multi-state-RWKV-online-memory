#!/usr/bin/env python3
"""Shared helpers for Gemma + RWKV-MS mechanism diagnostics."""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_MEMORY_REPO = "xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1"


@dataclass(frozen=True)
class NiahSample:
    sample_id: str
    prompt: str
    answer: str
    candidates: tuple[str, ...]
    needle_marker: str

    @classmethod
    def from_json(cls, row: dict) -> "NiahSample":
        return cls(
            sample_id=str(row["id"]),
            prompt=str(row["prompt"]),
            answer=str(row["answer"]),
            candidates=tuple(str(item) for item in row["candidates"]),
            needle_marker=str(row["needle_marker"]),
        )


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_samples(path: str | Path, limit: int | None = None) -> list[NiahSample]:
    return [NiahSample.from_json(row) for row in read_jsonl(path, limit=limit)]


def add_delta_mem_root(delta_mem_root: str | Path | None) -> None:
    if delta_mem_root is None:
        return
    root = Path(delta_mem_root).expanduser().resolve()
    if not (root / "deltamem").is_dir():
        raise FileNotFoundError(f"{root} does not look like a delta-Mem checkout")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def resolve_memory_dir(memory_dir: str | Path | None, memory_repo: str | None) -> str | None:
    if memory_dir:
        path = Path(memory_dir).expanduser().resolve()
        if not (path / "delta_mem_config.json").is_file():
            raise FileNotFoundError(f"Missing delta_mem_config.json in {path}")
        if not (path / "delta_mem_adapter.pt").is_file():
            raise FileNotFoundError(f"Missing delta_mem_adapter.pt in {path}")
        return str(path)
    if memory_repo:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=memory_repo)
    return None


def load_model_and_tokenizer(
    *,
    base_model: str,
    device: str,
    dtype: str,
    attn_implementation: str | None,
    delta_mem_root: str | Path | None = None,
    memory_dir: str | Path | None = None,
    memory_repo: str | None = None,
):
    add_delta_mem_root(delta_mem_root)
    adapter_dir = resolve_memory_dir(memory_dir, memory_repo)
    if adapter_dir is not None:
        from deltamem.runtime.session import load_delta_mem_chat_model

        return load_delta_mem_chat_model(
            model_path=base_model,
            adapter_dir=adapter_dir,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map={"": device},
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()
    return model, tokenizer


def iter_delta_modules(model) -> Iterator[tuple[str, object]]:
    try:
        from deltamem.core.delta import iter_delta_mem_modules
    except Exception:
        return iter(())
    return iter_delta_mem_modules(model)


def reset_delta_state(model) -> None:
    try:
        from deltamem.core.delta import reset_delta_mem_states
    except Exception:
        return
    reset_delta_mem_states(model)


def set_delta_write_enabled(model, enabled: bool) -> None:
    try:
        from deltamem.core.delta import set_delta_mem_write_enabled
    except Exception:
        return
    set_delta_mem_write_enabled(model, enabled)


@contextlib.contextmanager
def memory_condition(model, condition: str):
    """Apply a temporary RWKV-MS ablation condition."""

    modules = list(iter_delta_modules(model))
    saved_heads = [(module, getattr(module, "active_delta_heads", None)) for _, module in modules]
    saved_write = [(module, getattr(module, "write_enabled", None)) for _, module in modules]
    if condition == "normal" or condition.startswith("reset_"):
        pass
    elif condition == "no_write":
        set_delta_write_enabled(model, False)
    elif condition == "no_delta":
        for _, module in modules:
            module.active_delta_heads = frozenset()
    else:
        raise ValueError(f"Unsupported memory condition: {condition}")
    try:
        yield
    finally:
        for module, heads in saved_heads:
            if heads is not None:
                module.active_delta_heads = heads
        for module, enabled in saved_write:
            if enabled is not None:
                module.set_write_enabled(bool(enabled))


def parse_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    if not conditions:
        raise ValueError("At least one condition is required")
    for condition in conditions:
        if condition in {"base", "normal", "no_write", "no_delta"}:
            continue
        if condition.startswith("reset_") and condition.split("_", 1)[1].isdigit():
            continue
        raise ValueError(f"Unsupported condition: {condition}")
    return conditions


def reset_interval_for_condition(condition: str) -> int:
    if not condition.startswith("reset_"):
        return 0
    return int(condition.split("_", 1)[1])


def tokenize_prompt(tokenizer, prompt: str, device: str):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {key: value.to(device) for key, value in encoded.items()}


def first_candidate_token_ids(tokenizer, candidates: Iterable[str]) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    used: dict[int, str] = {}
    for candidate in candidates:
        ids = tokenizer(" " + candidate, add_special_tokens=False).input_ids
        if not ids:
            ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"Candidate {candidate!r} did not tokenize")
        token_id = int(ids[0])
        if token_id in used:
            raise ValueError(
                f"Candidates {used[token_id]!r} and {candidate!r} share first token id {token_id}; "
                "use labels that map to unique first tokens."
            )
        token_ids[candidate] = token_id
        used[token_id] = candidate
    return token_ids


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    last = len(sequence) - len(subsequence)
    for start in range(last + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


def find_marker_token_index(tokenizer, input_ids, marker: str) -> int:
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    ids = input_ids.detach().cpu().view(-1).tolist()
    index = find_subsequence(ids, marker_ids)
    if index >= 0:
        return index
    spaced_ids = tokenizer(" " + marker, add_special_tokens=False).input_ids
    return find_subsequence(ids, spaced_ids)


def logits_to_keep_kwargs(model, value: int) -> dict[str, int]:
    """Limit causal-LM logits when the local Transformers model supports it."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: value}
    return {}


def forward_logits(
    model,
    *,
    input_ids,
    attention_mask,
    reset_interval: int = 0,
):
    import torch

    reset_delta_state(model)
    with torch.inference_mode():
        if reset_interval <= 0:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **logits_to_keep_kwargs(model, 1),
            )
            return outputs.logits[:, -1, :]

        past_key_values = None
        logits = None
        seq_len = input_ids.size(1)
        for start in range(0, seq_len, reset_interval):
            end = min(seq_len, start + reset_interval)
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, :end]
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **logits_to_keep_kwargs(model, 1),
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            if end < seq_len:
                reset_delta_state(model)
        if logits is None:
            raise RuntimeError("No logits were produced")
        return logits


def score_candidates_from_logits(logits, candidate_token_ids: dict[str, int]) -> dict[str, float]:
    import torch

    selected = {
        candidate: float(logits[0, token_id].detach().float().item())
        for candidate, token_id in candidate_token_ids.items()
    }
    normalizer = torch.logsumexp(
        torch.tensor(list(selected.values()), dtype=torch.float32),
        dim=0,
    ).item()
    return {candidate: score - normalizer for candidate, score in selected.items()}


def entropy(probs) -> float:
    import torch

    values = probs.detach().float().clamp_min(1e-8)
    return float(-(values * values.log()).sum().item())


def collect_rwkv_trace(model, *, needle_token_index: int | None = None) -> list[dict]:
    metrics: list[dict] = []
    for name, module in iter_delta_modules(model):
        if getattr(module, "memory_backend", None) != "rwkv_ms":
            continue
        routes = getattr(module, "last_read_routes", None)
        if routes is None or routes.numel() == 0:
            continue
        query_routes = routes.detach().float()[0, -1]
        num_states = int(getattr(module, "rwkv_ms_num_states", query_routes.numel()))
        chunk_size = int(getattr(module, "rwkv_ms_chunk_size", 0))
        needle_slot = None
        needle_slot_mass = None
        if needle_token_index is not None and needle_token_index >= 0 and chunk_size > 0:
            needle_slot = (int(needle_token_index) // chunk_size) % num_states
            needle_slot_mass = float(query_routes[needle_slot].item())
        state = getattr(module, "delta_state", None)
        state_norm = None if state is None else float(state.detach().float().norm().item())
        delta_o_ratio = getattr(module, "last_delta_o_ratio", None)
        metrics.append(
            {
                "module": name,
                "layer_idx": int(getattr(module, "layer_idx", -1)),
                "num_states": num_states,
                "chunk_size": chunk_size,
                "needle_slot": needle_slot,
                "needle_slot_mass": needle_slot_mass,
                "read_entropy": entropy(query_routes),
                "read_max": float(query_routes.max().item()),
                "read_argmax": int(query_routes.argmax().item()),
                "state_norm": state_norm,
                "delta_o_ratio": None
                if delta_o_ratio is None
                else float(delta_o_ratio.detach().float().item()),
            }
        )
    return metrics


def summarize_numeric(rows: list[dict], keys: Iterable[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
    return summary
