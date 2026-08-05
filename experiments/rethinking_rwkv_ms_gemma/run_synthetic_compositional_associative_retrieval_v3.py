#!/usr/bin/env python3
"""Train and evaluate the V3 compositional projected-KV memory canary."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    collect_delta_mem_projected_kv_read_logits,
    diff_delta_mem_snapshots,
    freeze_non_delta_mem_params,
    get_delta_mem_online_state,
    iter_delta_mem_modules,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_projected_kv_read_query_mask,
    set_delta_mem_projected_kv_write_spans,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from deltamem.train.delta_sft_experimental import (
    _disable_training_cache,
    _promote_trainable_parameters_to_fp32,
    _temporarily_disable_delta_heads,
    checkpoint_frozen_mlp_activations,
)
from experiments.rethinking_rwkv_ms_gemma import (
    prepare_synthetic_compositional_associative_retrieval_canary_v3 as canary,
)


RUN_SCHEMA = "rwkv_ms_synthetic_compositional_associative_run.v3"
EVALUATION_SCHEMA = "rwkv_ms_synthetic_compositional_associative_eval.v3"
PROTOCOL_SCHEMA = "rwkv_ms_synthetic_compositional_associative_protocol.v3"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
TARGET_LAYERS = tuple(range(42))
CONDITIONS = (
    "correct",
    "donor",
    "value_swap",
    "shuffled_slots",
    "no_write",
)
POSITIVE_ANSWER_CONDITIONS = ("correct", "donor", "value_swap")


@dataclass(frozen=True)
class EpisodeExample:
    row_id: str
    memory_state_id: str
    condition: str
    write_records: tuple[dict[str, Any], ...]
    write_slots: tuple[int, ...]
    read_input_ids: tuple[int, ...]
    read_attention_mask: tuple[int, ...]
    query_mask: tuple[bool, ...]
    answer_mask: tuple[bool, ...]
    labels: tuple[int, ...]
    target_slot: int | None
    expected_answer_token_ids: tuple[int, ...]
    expected_value: str | None


@dataclass
class EpisodeBatch:
    examples: list[EpisodeExample]
    write_records: list[dict[str, torch.Tensor]]
    read_input_ids: torch.Tensor
    read_attention_mask: torch.Tensor
    query_mask: torch.Tensor
    answer_mask: torch.Tensor
    labels: torch.Tensor
    target_slots: torch.Tensor


def configure_hf_mirror() -> str:
    current = os.environ.get("HF_ENDPOINT")
    if current is not None and current.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {current!r}"
        )
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    return HF_MIRROR_ENDPOINT


def build_delta_config(
    *,
    target_layers: Sequence[int] = TARGET_LAYERS,
    rank: int = 4,
    key_dim: int = 32,
    temperature: float = 16.0,
) -> HFDeltaMemConfig:
    if rank <= 0:
        raise ValueError("Projected-KV value rank must be positive")
    return HFDeltaMemConfig(
        rank=rank,
        alpha=2 * rank,
        memory_backend="rwkv_ms",
        rwkv_ms_num_states=canary.RWKV_MS_NUM_STATES,
        rwkv_ms_chunk_size=128,
        rwkv_ms_boundary_mode="fixed_chunk",
        rwkv_ms_write_mode="recurrent",
        rwkv_ms_erase_gate=1.0,
        rwkv_ms_read_top_k=0,
        rwkv_ms_output_init_scale=0.02,
        rwkv_ms_semantics_version=2,
        num_state_heads=1,
        beta_bias_init=0.0,
        couple_lambda=True,
        state_update_mode="standard",
        rankwise_gates=True,
        output_init="base_slice_fixed",
        base_slice_ref_width=8,
        delta_heads=("q", "o"),
        delta_o_rmsnorm=False,
        memory_fusion_mode="add",
        memory_fusion_placement="attention_output",
        memory_fusion_residual_scale=1.0,
        memory_fusion_residual_scale_max=1.0,
        trainable_delta_scale=True,
        delta_scale_init=0.1,
        delta_scale_max=0.5,
        delta_scale_granularity="head",
        delta_scale_parameterization="alpha_over_rank",
        online_gain=0.2,
        target_layers=tuple(int(layer) for layer in target_layers),
        memory_readout_mode="projected_kv_slots",
        projected_kv_key_dim=key_dim,
        projected_kv_temperature=temperature,
        projected_kv_update_cosine_threshold=1.0,
        memory_write_source="learned_hidden",
        memory_write_granularity="token",
    )


def _trim_read_features(row: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    answer_positions = row["read_route"]["answer_token_positions"]
    if not answer_positions:
        raise ValueError(f"Row {row['row_id']} has no answer positions")
    expected_positions = list(
        range(int(answer_positions[0]), int(answer_positions[-1]) + 1)
    )
    if answer_positions != expected_positions:
        raise ValueError(f"Row {row['row_id']} answer tokens are not contiguous")
    end = int(answer_positions[-1]) + 1
    result = {
        "input_ids": tuple(int(value) for value in row["read_route_input_ids"][:end]),
        "attention_mask": tuple(
            int(value) for value in row["read_route_attention_mask"][:end]
        ),
        "query_mask": tuple(bool(value) for value in row["read_route_target_mask"][:end]),
        "answer_mask": tuple(
            bool(value) for value in row["read_answer_target_mask"][:end]
        ),
        "labels": tuple(int(value) for value in row["read_answer_labels"][:end]),
    }
    lengths = {len(value) for value in result.values()}
    if lengths != {end} or not all(result["attention_mask"]):
        raise ValueError(f"Row {row['row_id']} read features are misaligned")
    if not any(result["query_mask"]) or not any(result["answer_mask"]):
        raise ValueError(f"Row {row['row_id']} omits query or answer supervision")
    return result


def _record_from_encoding(
    *,
    key: str,
    value: str,
    encoding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "input_ids": tuple(int(token) for token in encoding["input_ids"]),
        "attention_mask": tuple(
            int(token) for token in encoding["attention_mask"]
        ),
        "key_mask": tuple(bool(selected) for selected in encoding["key_token_mask"]),
        "value_mask": tuple(
            bool(selected) for selected in encoding["value_token_mask"]
        ),
    }


def _source_records(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        _record_from_encoding(
            key=str(record["key"]),
            value=str(record["value"]),
            encoding=record["tokenization"],
        )
        for record in row["record_local_writes"]
    )


def _encode_record(tokenizer: Any, key: str, value: str) -> dict[str, Any]:
    content = canary.RECORD_TEMPLATE.format(key=key, value=value)
    encoding = canary._encode_spans(
        tokenizer,
        content,
        {"key": canary._span(content, key), "value": canary._span(content, value)},
    )
    return _record_from_encoding(key=key, value=value, encoding=encoding)


def _encode_read(tokenizer: Any, key: str, value: str) -> dict[str, Any]:
    query_text = canary.QUERY_TEMPLATE.format(key=key)
    answer_text = canary.RESPONSE_TEMPLATE.format(value=value)
    messages = [
        {"role": "system", "content": canary.SYSTEM_PROMPT},
        {"role": "user", "content": query_text},
        {"role": "assistant", "content": answer_text},
    ]
    route = canary._encode_read_route(tokenizer, messages, key, answer_text)
    labels = [
        token if selected else -100
        for token, selected in zip(
            route["input_ids"], route["answer_token_mask"], strict=True
        )
    ]
    row = {
        "row_id": f"intervention-{key}-{value}",
        "read_route_input_ids": route["input_ids"],
        "read_route_attention_mask": route["attention_mask"],
        "read_route_target_mask": route["query_key_token_mask"],
        "read_answer_target_mask": route["answer_token_mask"],
        "read_answer_labels": labels,
        "read_route": route,
    }
    return _trim_read_features(row)


def _episode_example(
    *,
    row: Mapping[str, Any],
    condition: str,
    records: Sequence[dict[str, Any]],
    write_slots: Sequence[int],
    read: Mapping[str, Sequence[Any]],
    target_slot: int | None,
    expected_value: str | None,
) -> EpisodeExample:
    answer_ids = tuple(
        int(label) for label in read["labels"] if int(label) != -100
    )
    if expected_value is not None and not answer_ids:
        raise ValueError("Positive condition has no expected answer tokens")
    return EpisodeExample(
        row_id=str(row["row_id"]),
        memory_state_id=str(row["memory_state_id"]),
        condition=condition,
        write_records=tuple(records),
        write_slots=tuple(int(slot) for slot in write_slots),
        read_input_ids=tuple(int(value) for value in read["input_ids"]),
        read_attention_mask=tuple(int(value) for value in read["attention_mask"]),
        query_mask=tuple(bool(value) for value in read["query_mask"]),
        answer_mask=tuple(bool(value) for value in read["answer_mask"]),
        labels=tuple(int(value) for value in read["labels"]),
        target_slot=target_slot,
        expected_answer_token_ids=answer_ids,
        expected_value=expected_value,
    )


def correct_example(row: Mapping[str, Any]) -> EpisodeExample:
    return _episode_example(
        row=row,
        condition="correct",
        records=_source_records(row),
        write_slots=row["write_record_slot_indices"],
        read=_trim_read_features(row),
        target_slot=int(row["query_route_target_slot"]),
        expected_value=str(row["query"]["target_value"]),
    )


def donor_example(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> EpisodeExample:
    donor = rows[int(row["donor"]["row_ordinal"])]
    if donor["query"]["key"] != row["query"]["key"]:
        raise ValueError("Donor query key differs")
    return _episode_example(
        row=row,
        condition="donor",
        records=_source_records(donor),
        write_slots=donor["write_record_slot_indices"],
        read=_trim_read_features(donor),
        target_slot=int(donor["query_route_target_slot"]),
        expected_value=str(row["donor"]["expected_target_value"]),
    )


def value_swap_example(row: Mapping[str, Any], tokenizer: Any) -> EpisodeExample:
    records = row["record_local_writes"]
    source_slots = row["value_swap"]["source_slot_by_destination_slot"]
    swapped = tuple(
        _encode_record(
            tokenizer,
            str(destination["key"]),
            str(records[int(source_slots[index])]["value"]),
        )
        for index, destination in enumerate(records)
    )
    expected_value = str(row["value_swap"]["expected_target_value"])
    read = _encode_read(tokenizer, str(row["query"]["key"]), expected_value)
    return _episode_example(
        row=row,
        condition="value_swap",
        records=swapped,
        write_slots=row["write_record_slot_indices"],
        read=read,
        target_slot=int(row["query_route_target_slot"]),
        expected_value=expected_value,
    )


def shuffled_slot_example(
    row: Mapping[str, Any],
    slot_permutation: Sequence[int] = (2, 0, 3, 1),
) -> EpisodeExample:
    if sorted(slot_permutation) != list(range(canary.RWKV_MS_NUM_STATES)):
        raise ValueError("Shuffled-slot intervention must be a slot permutation")
    target_record = int(row["query_route_target_slot"])
    return _episode_example(
        row=row,
        condition="shuffled_slots",
        records=_source_records(row),
        write_slots=slot_permutation,
        read=_trim_read_features(row),
        target_slot=int(slot_permutation[target_record]),
        expected_value=str(row["query"]["target_value"]),
    )


def no_write_example(row: Mapping[str, Any]) -> EpisodeExample:
    return _episode_example(
        row=row,
        condition="no_write",
        records=(),
        write_slots=(),
        read=_trim_read_features(row),
        target_slot=None,
        expected_value=str(row["query"]["target_value"]),
    )


def build_condition_examples(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    condition: str,
    *,
    all_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[EpisodeExample]:
    if condition == "correct":
        return [correct_example(row) for row in rows]
    if condition == "donor":
        donor_rows = rows if all_rows is None else all_rows
        return [donor_example(row, donor_rows) for row in rows]
    if condition == "value_swap":
        return [value_swap_example(row, tokenizer) for row in rows]
    if condition == "shuffled_slots":
        return [shuffled_slot_example(row) for row in rows]
    if condition == "no_write":
        return [no_write_example(row) for row in rows]
    raise ValueError(f"Unsupported condition: {condition}")


def select_complete_memory_states(
    rows: Sequence[Mapping[str, Any]],
    limit: int | None,
) -> list[Mapping[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("Row limit must be positive")
    if limit < canary.RECORDS_PER_EPISODE:
        raise ValueError("Row limit is smaller than one complete memory-state family")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        state_id = str(row["memory_state_id"])
        if state_id not in grouped:
            order.append(state_id)
        grouped[state_id].append(row)
    result: list[Mapping[str, Any]] = []
    for state_id in order:
        family = sorted(
            grouped[state_id], key=lambda item: int(item["query_route_target_slot"])
        )
        if len(family) != canary.RECORDS_PER_EPISODE:
            raise ValueError(f"Memory state {state_id} is not a four-query family")
        if result and len(result) + len(family) > limit:
            break
        result.extend(family)
        if len(result) >= limit:
            break
    if not result:
        raise ValueError("Row limit is smaller than one complete memory-state family")
    return result


def _pad_1d(
    values: Sequence[Sequence[Any]],
    *,
    padding_value: int | bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    width = max(len(value) for value in values)
    result = torch.full(
        (len(values), width), padding_value, dtype=dtype, device=device
    )
    for index, value in enumerate(values):
        result[index, : len(value)] = torch.tensor(value, dtype=dtype, device=device)
    return result


def _left_pad_1d(
    values: Sequence[Sequence[Any]],
    *,
    padding_value: int | bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    width = max(len(value) for value in values)
    result = torch.full(
        (len(values), width), padding_value, dtype=dtype, device=device
    )
    for index, value in enumerate(values):
        result[index, width - len(value) :] = torch.tensor(
            value, dtype=dtype, device=device
        )
    return result


def collate_examples(
    examples: Sequence[EpisodeExample],
    *,
    pad_token_id: int,
    device: torch.device,
) -> EpisodeBatch:
    if not examples:
        raise ValueError("Cannot collate an empty episode batch")
    records_per_example = {len(example.write_records) for example in examples}
    if len(records_per_example) != 1:
        raise ValueError("A batch cannot mix write and no-write examples")
    write_records: list[dict[str, torch.Tensor]] = []
    record_count = next(iter(records_per_example))
    for record_index in range(record_count):
        records = [example.write_records[record_index] for example in examples]
        write_records.append(
            {
                "input_ids": _pad_1d(
                    [record["input_ids"] for record in records],
                    padding_value=pad_token_id,
                    dtype=torch.long,
                    device=device,
                ),
                "attention_mask": _pad_1d(
                    [record["attention_mask"] for record in records],
                    padding_value=0,
                    dtype=torch.long,
                    device=device,
                ),
                "key_mask": _pad_1d(
                    [record["key_mask"] for record in records],
                    padding_value=False,
                    dtype=torch.bool,
                    device=device,
                ),
                "value_mask": _pad_1d(
                    [record["value_mask"] for record in records],
                    padding_value=False,
                    dtype=torch.bool,
                    device=device,
                ),
                "slots": torch.tensor(
                    [example.write_slots[record_index] for example in examples],
                    dtype=torch.long,
                    device=device,
                ),
            }
        )
    target_slots = torch.tensor(
        [-1 if example.target_slot is None else example.target_slot for example in examples],
        dtype=torch.long,
        device=device,
    )
    return EpisodeBatch(
        examples=list(examples),
        write_records=write_records,
        read_input_ids=_pad_1d(
            [example.read_input_ids for example in examples],
            padding_value=pad_token_id,
            dtype=torch.long,
            device=device,
        ),
        read_attention_mask=_pad_1d(
            [example.read_attention_mask for example in examples],
            padding_value=0,
            dtype=torch.long,
            device=device,
        ),
        query_mask=_pad_1d(
            [example.query_mask for example in examples],
            padding_value=False,
            dtype=torch.bool,
            device=device,
        ),
        answer_mask=_pad_1d(
            [example.answer_mask for example in examples],
            padding_value=False,
            dtype=torch.bool,
            device=device,
        ),
        labels=_pad_1d(
            [example.labels for example in examples],
            padding_value=-100,
            dtype=torch.long,
            device=device,
        ),
        target_slots=target_slots,
    )


def causal_answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("Answer logits and labels are misaligned")
    shifted_labels = labels[:, 1:].contiguous()
    if not bool(shifted_labels.ne(-100).any().item()):
        raise ValueError("Answer labels contain no supervised predictor targets")
    return F.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def selected_route_logits(
    logits: torch.Tensor,
    query_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != query_mask.shape:
        raise ValueError("Route logits and query mask are misaligned")
    counts = query_mask.sum(dim=1, keepdim=True)
    if bool(counts.eq(0).any().item()):
        raise ValueError("Every row must select at least one query token")
    return torch.einsum(
        "btc,bt->bc", logits.float(), query_mask.to(dtype=torch.float32)
    ) / counts.to(dtype=torch.float32)


def route_loss_and_predictions(
    logits_by_module: Mapping[str, torch.Tensor],
    query_mask: torch.Tensor,
    target_slots: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not logits_by_module:
        raise RuntimeError("No graph-connected projected-KV route logits were exposed")
    if bool(target_slots.lt(0).any().item()):
        raise ValueError("Route loss requires a target slot for every row")
    losses: list[torch.Tensor] = []
    predictions: dict[str, torch.Tensor] = {}
    for name, logits in logits_by_module.items():
        selected = selected_route_logits(logits, query_mask)
        losses.append(F.cross_entropy(selected, target_slots))
        predictions[name] = selected.argmax(dim=-1)
    return torch.stack(losses).mean(), predictions


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _write_episode_batch(
    model: torch.nn.Module,
    batch: EpisodeBatch,
    *,
    dtype: torch.dtype,
) -> dict[str, Any]:
    reset_delta_mem_states(model)
    route_matches = 0
    route_total = 0
    module_count = len(list(iter_delta_mem_modules(model)))
    if not batch.write_records:
        set_delta_mem_write_enabled(model, False)
        return {
            "module_count": module_count,
            "full_occupancy_count": 0,
            "full_occupancy_total": 0,
            "forced_write_route_match_count": 0,
            "forced_write_route_total": 0,
        }

    with _temporarily_disable_delta_heads(model):
        for record in batch.write_records:
            set_delta_mem_write_enabled(model, True)
            set_delta_mem_projected_kv_write_spans(
                model,
                record["key_mask"],
                record["value_mask"],
                record["slots"],
            )
            with _autocast_context(record["input_ids"].device, dtype):
                model(
                    input_ids=record["input_ids"],
                    attention_mask=record["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            for name, module in iter_delta_mem_modules(model):
                routes = module.last_write_routes
                if routes is None or tuple(routes.shape[:2]) != (
                    len(batch.examples),
                    1,
                ):
                    raise RuntimeError(f"Forced write route is absent at {name}")
                predicted = routes[:, 0].argmax(dim=-1)
                route_matches += int(predicted.eq(record["slots"]).sum().item())
                route_total += len(batch.examples)

    full_occupancy = 0
    occupancy_total = 0
    for name, module in iter_delta_mem_modules(model):
        occupied = module.projected_kv_occupied
        if occupied is None or tuple(occupied.shape) != (
            len(batch.examples),
            canary.RWKV_MS_NUM_STATES,
        ):
            raise RuntimeError(f"Projected-KV occupancy is absent at {name}")
        counts = occupied.sum(dim=-1)
        full_occupancy += int(counts.eq(canary.RWKV_MS_NUM_STATES).sum().item())
        occupancy_total += len(batch.examples)
    set_delta_mem_write_enabled(model, False)
    return {
        "module_count": module_count,
        "full_occupancy_count": full_occupancy,
        "full_occupancy_total": occupancy_total,
        "forced_write_route_match_count": route_matches,
        "forced_write_route_total": route_total,
    }


def _read_episode_batch(
    model: torch.nn.Module,
    batch: EpisodeBatch,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_projected_kv_read_query_mask(model, batch.query_mask)
    with _autocast_context(batch.read_input_ids.device, dtype):
        outputs = model(
            input_ids=batch.read_input_ids,
            attention_mask=batch.read_attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits, collect_delta_mem_projected_kv_read_logits(model)


def _answer_exact_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[bool], int, int]:
    shifted_predictions = logits[:, :-1].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    exact: list[bool] = []
    token_correct = 0
    token_total = 0
    for row_index in range(labels.size(0)):
        selected = shifted_labels[row_index].ne(-100)
        expected = shifted_labels[row_index, selected]
        predicted = shifted_predictions[row_index, selected]
        if expected.numel() == 0:
            raise ValueError("Evaluation row has no answer targets")
        matches = predicted.eq(expected)
        exact.append(bool(matches.all().item()))
        token_correct += int(matches.sum().item())
        token_total += int(matches.numel())
    return exact, token_correct, token_total


def _state_digests(model: torch.nn.Module, batch_size: int) -> list[str]:
    state = get_delta_mem_online_state(model)
    digests = [hashlib.sha256() for _ in range(batch_size)]
    for name in sorted(state):
        tensor = state[name]
        if tensor.ndim == 0 or tensor.size(0) != batch_size:
            raise RuntimeError(f"Online state {name} lacks the episode batch axis")
        encoded_name = name.encode("utf-8")
        for row_index, digest in enumerate(digests):
            row = tensor[row_index].detach().cpu().contiguous()
            raw = row.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(encoded_name).to_bytes(4, "big"))
            digest.update(encoded_name)
            digest.update(str(row.dtype).encode("ascii"))
            digest.update(canonical_shape(row.shape))
            digest.update(raw)
    return [digest.hexdigest() for digest in digests]


def canonical_shape(shape: Iterable[int]) -> bytes:
    return canary.canonical_json_bytes([int(value) for value in shape])


def _greedy_answer_predictions(
    model: torch.nn.Module,
    batch: EpisodeBatch,
    *,
    pad_token_id: int,
    dtype: torch.dtype,
) -> list[tuple[int, ...]]:
    prefixes: list[tuple[int, ...]] = []
    expected_lengths: list[int] = []
    query_masks: list[tuple[bool, ...]] = []
    for example in batch.examples:
        answer_positions = [
            index for index, selected in enumerate(example.answer_mask) if selected
        ]
        if answer_positions != list(
            range(answer_positions[0], answer_positions[-1] + 1)
        ):
            raise ValueError("Greedy evaluation requires contiguous answer targets")
        prefix_length = answer_positions[0]
        prefixes.append(example.read_input_ids[:prefix_length])
        query_masks.append(example.query_mask[:prefix_length])
        expected_lengths.append(len(example.expected_answer_token_ids))

    input_ids = _left_pad_1d(
        prefixes,
        padding_value=pad_token_id,
        dtype=torch.long,
        device=batch.read_input_ids.device,
    )
    attention_mask = _left_pad_1d(
        [tuple(1 for _ in prefix) for prefix in prefixes],
        padding_value=0,
        dtype=torch.long,
        device=batch.read_input_ids.device,
    )
    query_mask = _left_pad_1d(
        query_masks,
        padding_value=False,
        dtype=torch.bool,
        device=batch.read_input_ids.device,
    )
    generated: list[list[int]] = [[] for _ in batch.examples]
    last_positions = torch.full(
        (input_ids.size(0),),
        input_ids.size(1) - 1,
        dtype=torch.long,
        device=input_ids.device,
    )
    for step in range(max(expected_lengths)):
        set_delta_mem_write_enabled(model, False)
        set_delta_mem_projected_kv_read_query_mask(model, query_mask)
        with _autocast_context(input_ids.device, dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        row_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        next_tokens = outputs.logits[row_indices, last_positions].argmax(dim=-1)
        for row_index, token in enumerate(next_tokens.detach().cpu().tolist()):
            if step < expected_lengths[row_index]:
                generated[row_index].append(int(token))
        input_ids = torch.cat((input_ids, next_tokens.unsqueeze(1)), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    input_ids.size(0),
                    1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        query_mask = torch.cat(
            (
                query_mask,
                torch.zeros(
                    input_ids.size(0),
                    1,
                    dtype=torch.bool,
                    device=query_mask.device,
                ),
            ),
            dim=1,
        )
        last_positions = torch.full_like(last_positions, input_ids.size(1) - 1)
    return [tuple(tokens) for tokens in generated]


def _batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def evaluate_condition(
    model: torch.nn.Module,
    examples: Sequence[EpisodeExample],
    *,
    condition: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    greedy: bool,
) -> dict[str, Any]:
    if not examples or any(example.condition != condition for example in examples):
        raise ValueError(f"Evaluation examples do not match condition {condition}")
    module_names = [name for name, _ in iter_delta_mem_modules(model)]
    answer_exact: list[bool] = []
    greedy_exact: list[bool] = []
    token_correct = 0
    token_total = 0
    route_correct = 0
    route_total = 0
    route_by_layer = {
        name: {"correct": 0, "total": 0} for name in module_names
    }
    route_by_row: dict[str, dict[str, int]] = {}
    state_digest_by_row: dict[str, str] = {}
    occupancy_correct = 0
    occupancy_total = 0
    write_route_correct = 0
    write_route_total = 0
    absent_modules = 0
    possible_modules = 0

    model.eval()
    with torch.no_grad():
        for raw_batch in _batches(list(examples), batch_size):
            batch = collate_examples(
                raw_batch, pad_token_id=pad_token_id, device=device
            )
            write_audit = _write_episode_batch(model, batch, dtype=dtype)
            occupancy_correct += write_audit["full_occupancy_count"]
            occupancy_total += write_audit["full_occupancy_total"]
            write_route_correct += write_audit["forced_write_route_match_count"]
            write_route_total += write_audit["forced_write_route_total"]
            if batch.write_records:
                digests = _state_digests(model, len(batch.examples))
                for example, digest in zip(batch.examples, digests, strict=True):
                    state_digest_by_row[example.row_id] = digest

            logits, route_logits = _read_episode_batch(model, batch, dtype=dtype)
            exact, batch_token_correct, batch_token_total = _answer_exact_predictions(
                logits, batch.labels
            )
            answer_exact.extend(exact)
            token_correct += batch_token_correct
            token_total += batch_token_total

            possible_modules += len(module_names) * len(batch.examples)
            absent_modules += (
                len(module_names) - len(route_logits)
            ) * len(batch.examples)
            if condition == "no_write":
                if route_logits:
                    raise RuntimeError("No-write condition unexpectedly exposed routes")
            else:
                if set(route_logits) != set(module_names):
                    raise RuntimeError("Positive condition omitted projected-KV routes")
                row_predictions = {example.row_id: {} for example in batch.examples}
                for name, layer_logits in route_logits.items():
                    selected = selected_route_logits(layer_logits, batch.query_mask)
                    predictions = selected.argmax(dim=-1)
                    matches = predictions.eq(batch.target_slots)
                    layer_matches = int(matches.sum().item())
                    route_correct += layer_matches
                    route_total += len(batch.examples)
                    route_by_layer[name]["correct"] += layer_matches
                    route_by_layer[name]["total"] += len(batch.examples)
                    for example, predicted in zip(
                        batch.examples,
                        predictions.detach().cpu().tolist(),
                        strict=True,
                    ):
                        row_predictions[example.row_id][name] = int(predicted)
                route_by_row.update(row_predictions)

            if greedy:
                generated = _greedy_answer_predictions(
                    model,
                    batch,
                    pad_token_id=pad_token_id,
                    dtype=dtype,
                )
                greedy_exact.extend(
                    generated_row == example.expected_answer_token_ids
                    for generated_row, example in zip(
                        generated, batch.examples, strict=True
                    )
                )
            reset_delta_mem_states(model)

    layer_metrics = {
        name: {
            **counts,
            "accuracy": counts["correct"] / counts["total"]
            if counts["total"]
            else None,
        }
        for name, counts in route_by_layer.items()
    }
    result: dict[str, Any] = {
        "condition": condition,
        "rows": len(examples),
        "teacher_forced_answer_exact_count": sum(answer_exact),
        "teacher_forced_answer_exact_accuracy": sum(answer_exact) / len(answer_exact),
        "teacher_forced_answer_token_correct": token_correct,
        "teacher_forced_answer_token_total": token_total,
        "teacher_forced_answer_token_accuracy": token_correct / token_total,
        "greedy_answer_evaluated": greedy,
        "greedy_answer_exact_count": sum(greedy_exact) if greedy else None,
        "greedy_answer_exact_accuracy": (
            sum(greedy_exact) / len(greedy_exact) if greedy else None
        ),
        "semantic_route_correct": route_correct,
        "semantic_route_total": route_total,
        "semantic_route_accuracy": route_correct / route_total if route_total else None,
        "route_by_layer": layer_metrics,
        "route_predictions_by_row": route_by_row,
        "full_occupancy_count": occupancy_correct,
        "full_occupancy_total": occupancy_total,
        "full_occupancy_fraction": (
            occupancy_correct / occupancy_total if occupancy_total else None
        ),
        "forced_write_route_correct": write_route_correct,
        "forced_write_route_total": write_route_total,
        "forced_write_route_accuracy": (
            write_route_correct / write_route_total if write_route_total else None
        ),
        "route_absent_module_rows": absent_modules,
        "route_possible_module_rows": possible_modules,
        "route_absent_fraction": absent_modules / possible_modules,
        "state_digest_by_row": state_digest_by_row,
    }
    return result


def query_counterfactual_audit(
    examples: Sequence[EpisodeExample],
    correct_result: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeExample]] = defaultdict(list)
    for example in examples:
        grouped[example.memory_state_id].append(example)
    route_predictions = correct_result["route_predictions_by_row"]
    state_digests = correct_result["state_digest_by_row"]
    module_names = list(correct_result["route_by_layer"])
    family_layer_total = 0
    family_layer_all_four_correct = 0
    query_correct = 0
    query_total = 0
    identical_state_families = 0
    family_count = 0
    for state_id, family in grouped.items():
        family = sorted(family, key=lambda item: int(item.target_slot))
        expected_slots = list(range(canary.RECORDS_PER_EPISODE))
        if [example.target_slot for example in family] != expected_slots:
            raise ValueError(f"Query family {state_id} does not cover all four slots")
        family_count += 1
        digests = {state_digests[example.row_id] for example in family}
        if len(digests) == 1:
            identical_state_families += 1
        for module_name in module_names:
            predicted = [
                int(route_predictions[example.row_id][module_name])
                for example in family
            ]
            matches = [
                prediction == target
                for prediction, target in zip(predicted, expected_slots, strict=True)
            ]
            query_correct += sum(matches)
            query_total += len(matches)
            family_layer_total += 1
            family_layer_all_four_correct += int(all(matches))
    return {
        "memory_state_families": family_count,
        "runtime_byte_identical_state_families": identical_state_families,
        "runtime_byte_identical_state_fraction": (
            identical_state_families / family_count
        ),
        "query_counterfactual_route_correct": query_correct,
        "query_counterfactual_route_total": query_total,
        "query_counterfactual_route_accuracy": query_correct / query_total,
        "family_layer_all_four_correct": family_layer_all_four_correct,
        "family_layer_total": family_layer_total,
        "family_layer_all_four_correct_fraction": (
            family_layer_all_four_correct / family_layer_total
        ),
    }


def _router_gradient_audit(
    model: torch.nn.Module,
    route_loss: torch.Tensor,
) -> dict[str, Any]:
    modules = list(iter_delta_mem_modules(model))
    parameters = [module.projected_kv_key_proj for _, module in modules]
    gradients = torch.autograd.grad(
        route_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    records: list[dict[str, Any]] = []
    finite_nonzero = 0
    for (name, module), gradient in zip(modules, gradients, strict=True):
        norm = None if gradient is None else float(gradient.detach().float().norm().item())
        passed = norm is not None and math.isfinite(norm) and norm > 0.0
        finite_nonzero += int(passed)
        records.append(
            {
                "module": name,
                "layer": int(module.layer_idx),
                "projected_kv_key_route_grad_norm": norm,
                "finite_nonzero": passed,
            }
        )
    return {
        "modules": len(records),
        "finite_nonzero_modules": finite_nonzero,
        "all_modules_finite_nonzero": finite_nonzero == len(records),
        "records": records,
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()


def train_model(
    model: torch.nn.Module,
    examples: Sequence[EpisodeExample],
    *,
    seed: int,
    epochs: int,
    max_steps: int | None,
    batch_size: int,
    learning_rate: float,
    answer_weight: float,
    route_weight: float,
    max_grad_norm: float,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    progress_path: Path,
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("Training epochs and learning rate must be positive")
    if answer_weight <= 0.0 or route_weight <= 0.0:
        raise ValueError("Both answer and route loss weights must be positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Frozen-Gemma runner found no trainable memory parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    rng = random.Random(seed)
    global_step = 0
    router_gradient: dict[str, Any] | None = None
    totals = {
        "answer_loss": 0.0,
        "route_loss": 0.0,
        "total_loss": 0.0,
        "answer_exact_correct": 0,
        "answer_rows": 0,
        "route_correct": 0,
        "route_total": 0,
        "full_occupancy_count": 0,
        "full_occupancy_total": 0,
        "forced_write_route_correct": 0,
        "forced_write_route_total": 0,
    }
    started = time.time()
    model.train()
    stop = False
    for epoch in range(epochs):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        for index_batch in _batches(indices, batch_size):
            selected = [examples[index] for index in index_batch]
            batch = collate_examples(
                selected, pad_token_id=pad_token_id, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            write_audit = _write_episode_batch(model, batch, dtype=dtype)
            logits, route_logits = _read_episode_batch(model, batch, dtype=dtype)
            answer_loss = causal_answer_loss(logits, batch.labels)
            route_loss, route_predictions = route_loss_and_predictions(
                route_logits, batch.query_mask, batch.target_slots
            )
            total_loss = answer_weight * answer_loss + route_weight * route_loss
            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError(f"Non-finite training loss at step {global_step + 1}")
            if router_gradient is None:
                router_gradient = _router_gradient_audit(model, route_loss)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError(f"Non-finite gradient norm at step {global_step + 1}")
            reset_delta_mem_states(model)
            optimizer.step()
            global_step += 1

            exact, _, _ = _answer_exact_predictions(logits.detach(), batch.labels)
            route_matches = sum(
                int(prediction.eq(batch.target_slots).sum().item())
                for prediction in route_predictions.values()
            )
            route_count = len(route_predictions) * len(selected)
            step_record = {
                "schema": "rwkv_ms_synthetic_compositional_train_step.v3",
                "step": global_step,
                "epoch": epoch,
                "rows": len(selected),
                "answer_loss": float(answer_loss.detach().float().item()),
                "route_loss": float(route_loss.detach().float().item()),
                "total_loss": float(total_loss.detach().float().item()),
                "gradient_norm": float(grad_norm.detach().float().item()),
                "teacher_forced_answer_exact_accuracy": sum(exact) / len(exact),
                "semantic_route_accuracy": route_matches / route_count,
                "full_occupancy_fraction": (
                    write_audit["full_occupancy_count"]
                    / write_audit["full_occupancy_total"]
                ),
            }
            _append_jsonl(progress_path, step_record)
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "answer_loss": round(step_record["answer_loss"], 6),
                        "route_loss": round(step_record["route_loss"], 6),
                        "answer_exact": round(
                            step_record["teacher_forced_answer_exact_accuracy"], 4
                        ),
                        "route_accuracy": round(
                            step_record["semantic_route_accuracy"], 4
                        ),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )
            totals["answer_loss"] += step_record["answer_loss"]
            totals["route_loss"] += step_record["route_loss"]
            totals["total_loss"] += step_record["total_loss"]
            totals["answer_exact_correct"] += sum(exact)
            totals["answer_rows"] += len(exact)
            totals["route_correct"] += route_matches
            totals["route_total"] += route_count
            totals["full_occupancy_count"] += write_audit["full_occupancy_count"]
            totals["full_occupancy_total"] += write_audit["full_occupancy_total"]
            totals["forced_write_route_correct"] += write_audit[
                "forced_write_route_match_count"
            ]
            totals["forced_write_route_total"] += write_audit[
                "forced_write_route_total"
            ]
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        if stop:
            break
    if global_step == 0 or router_gradient is None:
        raise RuntimeError("Training executed no optimization steps")
    return {
        "steps": global_step,
        "epochs_requested": epochs,
        "max_steps": max_steps,
        "elapsed_seconds": time.time() - started,
        "mean_answer_loss": totals["answer_loss"] / global_step,
        "mean_route_loss": totals["route_loss"] / global_step,
        "mean_total_loss": totals["total_loss"] / global_step,
        "teacher_forced_answer_exact_accuracy": (
            totals["answer_exact_correct"] / totals["answer_rows"]
        ),
        "semantic_route_accuracy": totals["route_correct"] / totals["route_total"],
        "full_occupancy_fraction": (
            totals["full_occupancy_count"] / totals["full_occupancy_total"]
        ),
        "forced_write_route_accuracy": (
            totals["forced_write_route_correct"]
            / totals["forced_write_route_total"]
        ),
        "router_gradient_audit": router_gradient,
    }


def build_gate(
    evaluation: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    split_audit_passed: bool,
    input_immutability_passed: bool,
    require_greedy: bool,
) -> dict[str, Any]:
    conditions = evaluation["conditions"]
    query_audit = evaluation["query_counterfactual_audit"]
    acceptance = evaluation["acceptance_contract"]

    def answer_accuracy(condition: str) -> float:
        result = conditions[condition]
        if require_greedy:
            value = result["greedy_answer_exact_accuracy"]
            if value is None:
                return -1.0
            return float(value)
        return float(result["teacher_forced_answer_exact_accuracy"])

    criteria = {
        "heldout_answer_accuracy": (
            answer_accuracy("correct")
            >= acceptance["heldout_answer_accuracy_min"]
        ),
        "heldout_semantic_route_accuracy": (
            conditions["correct"]["semantic_route_accuracy"]
            >= acceptance["heldout_semantic_route_accuracy_min"]
        ),
        "heldout_query_counterfactual_route_accuracy": (
            query_audit["query_counterfactual_route_accuracy"]
            >= acceptance["heldout_query_counterfactual_route_accuracy_min"]
        ),
        "heldout_donor_expected_answer_accuracy": (
            answer_accuracy("donor")
            >= acceptance["heldout_donor_expected_answer_accuracy_min"]
        ),
        "heldout_value_swap_expected_answer_accuracy": (
            answer_accuracy("value_swap")
            >= acceptance["heldout_value_swap_expected_answer_accuracy_min"]
        ),
        "heldout_no_write_answer_near_chance": (
            answer_accuracy("no_write")
            <= acceptance["heldout_no_write_answer_accuracy_max"]
        ),
        "heldout_no_write_route_absent": (
            conditions["no_write"]["route_absent_fraction"]
            >= acceptance["heldout_no_write_route_absent_fraction_min"]
        ),
        "shuffled_slot_semantic_route_accuracy": (
            conditions["shuffled_slots"]["semantic_route_accuracy"]
            >= acceptance["heldout_semantic_route_accuracy_min"]
        ),
        "runtime_query_states_are_byte_identical": (
            query_audit["runtime_byte_identical_state_fraction"] == 1.0
        ),
        "full_four_slot_occupancy": all(
            conditions[condition]["full_occupancy_fraction"] == 1.0
            for condition in CONDITIONS
            if condition != "no_write"
        ),
        "forced_write_routes_exact": all(
            conditions[condition]["forced_write_route_accuracy"] == 1.0
            for condition in CONDITIONS
            if condition != "no_write"
        ),
        "router_gradients_finite_nonzero": training["router_gradient_audit"][
            "all_modules_finite_nonzero"
        ],
        "split_leakage_audit_passed": split_audit_passed,
        "input_and_model_immutability_passed": input_immutability_passed,
        "greedy_answer_evaluation_required": require_greedy,
    }
    return {
        "passed": all(criteria.values())
        and evaluation["seed"] in acceptance["training_seeds"],
        "criteria": criteria,
        "required_seed_passes": acceptance["required_seed_passes"],
        "seed_is_one_of_locked_acceptance_seeds": evaluation["seed"]
        in acceptance["training_seeds"],
    }


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _parse_target_layers(value: str) -> tuple[int, ...]:
    if value == "all":
        return TARGET_LAYERS
    layers = tuple(int(item) for item in value.split(",") if item)
    if not layers or len(layers) != len(set(layers)) or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("Target layers must be 'all' or unique CSV integers")
    return layers


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_shape(tensor.shape))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _signed_payload(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(hash_field, None)
    result[hash_field] = canary.canonical_sha256(result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    canary.atomic_write(
        path,
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def _validate_signed_payload(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    description: str,
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(hash_field, None)
    actual = canary.canonical_sha256(unsigned)
    if declared != actual:
        raise ValueError(f"{description} canonical SHA-256 differs")
    return actual


def _source_receipt(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_path": str(source["manifest_path"]),
        "manifest_file_sha256": source["manifest_file_sha256"],
        "manifest_sha256": source["manifest_sha256"],
        "split_manifest_path": str(source["split_manifest_path"]),
        "split_manifest_sha256": source["split_manifest"]["manifest_sha256"],
        "partitions_sha256": canary.canonical_sha256(source["partitions"]),
        "train_rows": len(source["partitions"]["train"]),
        "heldout_rows": len(source["partitions"]["heldout"]),
    }


def _load_model_and_tokenizer(
    source: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    delta_config: HFDeltaMemConfig,
) -> tuple[torch.nn.Module, Any, list[str], list[str], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source["model"]["path"],
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    _disable_training_cache(model)
    replaced = attach_delta_mem(model, delta_config)
    trainable_names = freeze_non_delta_mem_params(model)
    _promote_trainable_parameters_to_fp32(model)
    checkpointed_mlps = checkpoint_frozen_mlp_activations(model)
    if len(replaced) != len(delta_config.target_layers):
        raise RuntimeError(
            f"Attached {len(replaced)} layers, expected {len(delta_config.target_layers)}"
        )
    return model, tokenizer, replaced, trainable_names, checkpointed_mlps


def run_experiment(
    *,
    source_manifest: Path,
    model_path: Path,
    output_dir: Path,
    seed: int,
    profile: str,
    train_limit: int | None,
    eval_split: str,
    eval_limit: int | None,
    epochs: int,
    max_steps: int | None,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    answer_weight: float,
    route_weight: float,
    max_grad_norm: float,
    device_name: str,
    dtype_name: str,
    attn_implementation: str,
    target_layers: Sequence[int],
    rank: int,
    key_dim: int,
    temperature: float,
    greedy: bool,
) -> dict[str, Any]:
    configure_hf_mirror()
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() or resolved_output.is_symlink():
        raise ValueError(f"V3 run output must be fresh: {resolved_output}")
    resolved_output.mkdir(parents=True)
    progress_path = resolved_output / "training_progress.jsonl"
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=True,
    )
    source_before = _source_receipt(source)
    model_before = canary.bind_model_artifacts(model_path)
    if source["model"] != model_before:
        raise ValueError("Bound source model differs before the run")
    split_audit_passed = source["split_manifest"]["audit"]["passed"] is True
    if not split_audit_passed:
        raise ValueError("V3 split leakage audit did not pass")

    device = torch.device(device_name)
    dtype = _dtype(dtype_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(seed)
    delta_config = build_delta_config(
        target_layers=target_layers,
        rank=rank,
        key_dim=key_dim,
        temperature=temperature,
    )
    protocol = _signed_payload(
        {
            "schema": PROTOCOL_SCHEMA,
            "profile": profile,
            "seed": seed,
            "source": source_before,
            "model": model_before,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "train_limit": train_limit,
            "eval_split": eval_split,
            "eval_limit": eval_limit,
            "epochs": epochs,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "answer_weight": answer_weight,
            "route_weight": route_weight,
            "max_grad_norm": max_grad_norm,
            "device": str(device),
            "dtype": dtype_name,
            "attn_implementation": attn_implementation,
            "target_layers": list(target_layers),
            "projected_kv_value_rank": rank,
            "projected_kv_key_dim": key_dim,
            "projected_kv_temperature": temperature,
            "greedy_answer_evaluation": greedy,
            "delta_config": delta_config.to_dict(),
            "delta_config_sha256": canary.canonical_sha256(
                delta_config.to_dict()
            ),
        },
        "protocol_sha256",
    )
    _write_json(resolved_output / "protocol.json", protocol)

    model, tokenizer, replaced, trainable_names, checkpointed_mlps = (
        _load_model_and_tokenizer(
            source,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            delta_config=delta_config,
        )
    )
    initial_snapshot = snapshot_delta_mem_weights(model)
    initial_adapter_sha256 = _state_dict_sha256(initial_snapshot)
    train_rows = select_complete_memory_states(
        source["partitions"]["train"], train_limit
    )
    train_examples = [correct_example(row) for row in train_rows]
    training = train_model(
        model,
        train_examples,
        seed=seed,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        answer_weight=answer_weight,
        route_weight=route_weight,
        max_grad_norm=max_grad_norm,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        dtype=dtype,
        progress_path=progress_path,
    )
    final_snapshot = snapshot_delta_mem_weights(model)
    final_adapter_sha256 = _state_dict_sha256(final_snapshot)
    training["adapter_weight_diff"] = diff_delta_mem_snapshots(
        initial_snapshot, final_snapshot
    )
    training["initial_adapter_sha256"] = initial_adapter_sha256
    training["final_adapter_sha256"] = final_adapter_sha256

    eval_partition = source["partitions"][eval_split]
    eval_rows = select_complete_memory_states(eval_partition, eval_limit)
    condition_results: dict[str, Any] = {}
    examples_by_condition: dict[str, list[EpisodeExample]] = {}
    for condition in CONDITIONS:
        print(json.dumps({"evaluating": condition, "rows": len(eval_rows)}), flush=True)
        condition_examples = build_condition_examples(
            eval_rows,
            tokenizer,
            condition,
            all_rows=eval_partition,
        )
        examples_by_condition[condition] = condition_examples
        condition_results[condition] = evaluate_condition(
            model,
            condition_examples,
            condition=condition,
            batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            dtype=dtype,
            greedy=greedy,
        )
    query_audit = query_counterfactual_audit(
        examples_by_condition["correct"], condition_results["correct"]
    )

    adapter_dir = resolved_output / "adapter"
    save_delta_mem_adapter(model, adapter_dir, delta_config)
    adapter_binding = {
        "config_path": str(adapter_dir / "delta_mem_config.json"),
        "config_sha256": canary.sha256_file(adapter_dir / "delta_mem_config.json"),
        "weights_path": str(adapter_dir / "delta_mem_adapter.pt"),
        "weights_sha256": canary.sha256_file(adapter_dir / "delta_mem_adapter.pt"),
    }
    source_after_bundle = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=True,
    )
    source_after = _source_receipt(source_after_bundle)
    model_after = canary.bind_model_artifacts(model_path)
    input_immutability_passed = (
        source_after == source_before and model_after == model_before
    )
    evaluation: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "seed": seed,
        "profile": profile,
        "eval_split": eval_split,
        "eval_rows": len(eval_rows),
        "source": source_before,
        "acceptance_contract": source["manifest"]["spec"]["acceptance_gate"],
        "conditions": condition_results,
        "query_counterfactual_audit": query_audit,
    }
    acceptance_eligible = (
        profile == "proof"
        and eval_split == "heldout"
        and len(eval_rows) == len(eval_partition)
        and len(train_rows) == len(source["partitions"]["train"])
        and greedy
        and tuple(target_layers) == TARGET_LAYERS
        and key_dim == 32
        and temperature == 16.0
    )
    gate = build_gate(
        evaluation,
        training=training,
        split_audit_passed=split_audit_passed,
        input_immutability_passed=input_immutability_passed,
        require_greedy=greedy,
    )
    gate["acceptance_eligible"] = acceptance_eligible
    gate["passed"] = gate["passed"] and acceptance_eligible
    evaluation["gate"] = gate
    evaluation = _signed_payload(evaluation, "evaluation_sha256")
    evaluation_path = resolved_output / "evaluation.json"
    _write_json(evaluation_path, evaluation)

    receipt = _signed_payload(
        {
            "schema": RUN_SCHEMA,
            "seed": seed,
            "profile": profile,
            "source_before": source_before,
            "source_after": source_after,
            "model_before": model_before,
            "model_after": model_after,
            "input_immutability_passed": input_immutability_passed,
            "protocol_path": str(resolved_output / "protocol.json"),
            "protocol_file_sha256": canary.sha256_file(
                resolved_output / "protocol.json"
            ),
            "protocol_sha256": protocol["protocol_sha256"],
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": str(device),
                "dtype": dtype_name,
                "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            },
            "model_attachment": {
                "replaced_modules": replaced,
                "trainable_parameter_names": trainable_names,
                "checkpointed_frozen_mlps": checkpointed_mlps,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
            "training": training,
            "adapter": adapter_binding,
            "evaluation_path": str(evaluation_path),
            "evaluation_file_sha256": canary.sha256_file(evaluation_path),
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "gate": gate,
        },
        "receipt_sha256",
    )
    receipt_path = resolved_output / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "output_dir": str(resolved_output),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": canary.sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "gate": gate,
    }


def validate_receipt(
    receipt_path: Path,
    *,
    source_manifest: Path,
    model_path: Path,
    verify_model_hashes: bool,
) -> dict[str, Any]:
    configure_hf_mirror()
    path = receipt_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"V3 run receipt is invalid: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt_sha256 = _validate_signed_payload(
        receipt,
        hash_field="receipt_sha256",
        description="V3 run receipt",
    )
    if receipt.get("schema") != RUN_SCHEMA:
        raise ValueError("V3 run receipt schema differs")
    source = canary.load_source_bundle(
        source_manifest,
        model_path=model_path,
        verify_model_hashes=verify_model_hashes,
    )
    expected_source = _source_receipt(source)
    if (
        receipt.get("source_before") != expected_source
        or receipt.get("source_after") != expected_source
        or receipt.get("input_immutability_passed") is not True
    ):
        raise ValueError("V3 run source binding differs")
    if verify_model_hashes:
        current_model = canary.bind_model_artifacts(model_path)
        if (
            receipt.get("model_before") != current_model
            or receipt.get("model_after") != current_model
        ):
            raise ValueError("V3 run model binding differs")

    protocol_path = Path(str(receipt.get("protocol_path", ""))).expanduser().resolve()
    evaluation_path = Path(str(receipt.get("evaluation_path", ""))).expanduser().resolve()
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or not evaluation_path.is_file()
        or evaluation_path.is_symlink()
    ):
        raise ValueError("V3 run protocol or evaluation artifact is absent")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    protocol_sha256 = _validate_signed_payload(
        protocol,
        hash_field="protocol_sha256",
        description="V3 run protocol",
    )
    evaluation_sha256 = _validate_signed_payload(
        evaluation,
        hash_field="evaluation_sha256",
        description="V3 run evaluation",
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or evaluation.get("schema") != EVALUATION_SCHEMA
        or receipt.get("protocol_sha256") != protocol_sha256
        or receipt.get("evaluation_sha256") != evaluation_sha256
        or receipt.get("protocol_file_sha256") != canary.sha256_file(protocol_path)
        or receipt.get("evaluation_file_sha256") != canary.sha256_file(evaluation_path)
        or receipt.get("gate") != evaluation.get("gate")
    ):
        raise ValueError("V3 run protocol/evaluation binding differs")
    adapter = receipt.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("V3 run adapter binding is absent")
    for prefix in ("config", "weights"):
        artifact_path = Path(str(adapter.get(f"{prefix}_path", ""))).expanduser().resolve()
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or adapter.get(f"{prefix}_sha256") != canary.sha256_file(artifact_path)
        ):
            raise ValueError(f"V3 run adapter {prefix} binding differs")
    return {
        "valid": True,
        "receipt_path": str(path),
        "receipt_file_sha256": canary.sha256_file(path),
        "receipt_sha256": receipt_sha256,
        "evaluation_sha256": evaluation_sha256,
        "seed": receipt["seed"],
        "gate": receipt["gate"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(canary.DEFAULT_OUTPUT_DIR / "source_manifest.json"),
    )
    parser.add_argument("--model-path", type=Path, default=canary.DEFAULT_MODEL_PATH)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--profile", choices=("microfit", "proof"), default="proof")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-split", choices=canary.PARTITION_ORDER, default="heldout")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--target-layers", type=_parse_target_layers, default=TARGET_LAYERS)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=16.0)
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verify-model-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_receipt is not None:
        result = validate_receipt(
            args.validate_receipt,
            source_manifest=args.source_manifest,
            model_path=args.model_path,
            verify_model_hashes=args.verify_model_hashes,
        )
    else:
        max_steps = None if args.max_steps == 0 else args.max_steps
        result = run_experiment(
            source_manifest=args.source_manifest,
            model_path=args.model_path,
            output_dir=args.output_dir,
            seed=args.seed,
            profile=args.profile,
            train_limit=args.train_limit,
            eval_split=args.eval_split,
            eval_limit=args.eval_limit,
            epochs=args.epochs,
            max_steps=max_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            answer_weight=args.answer_weight,
            route_weight=args.route_weight,
            max_grad_norm=args.max_grad_norm,
            device_name=args.device,
            dtype_name=args.dtype,
            attn_implementation=args.attn_implementation,
            target_layers=args.target_layers,
            rank=args.rank,
            key_dim=args.key_dim,
            temperature=args.temperature,
            greedy=args.greedy,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
