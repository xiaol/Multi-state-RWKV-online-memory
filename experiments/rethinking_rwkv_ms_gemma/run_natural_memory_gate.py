#!/usr/bin/env python3
"""Train and audit Delta-Mem on the natural four-slot causal memory gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from deltamem.core.delta import (
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


RUN_SCHEMA = "rwkv_ms_natural_memory_gate_run.v1"
EVALUATION_SCHEMA = "rwkv_ms_natural_memory_gate_evaluation.v1"
PROTOCOL_SCHEMA = "rwkv_ms_natural_memory_gate_protocol.v1"
TRAIN_STEP_SCHEMA = "rwkv_ms_natural_memory_gate_train_step.v1"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
RECORDS_PER_EPISODE = 4
CONDITIONS = (
    "correct_state",
    "donor_state",
    "value_swap",
    "target_slot_rewrite",
    "shuffled_slots",
    "no_state",
    "pristine_frozen_base",
)
POSITIVE_CONDITIONS = CONDITIONS[:5]
SHARED_WRITE_CONDITIONS = (
    "correct_state",
    "donor_state",
    "value_swap",
    "shuffled_slots",
)
COUNTERFACTUAL_STATE_CONDITIONS = (
    "donor_state",
    "value_swap",
    "target_slot_rewrite",
    "shuffled_slots",
)
DEFAULT_TRAINING_CONDITIONS = ("correct_state",)
SUPERVISED_COMPOSITIONAL_TRAINING_CONDITIONS = POSITIVE_CONDITIONS
CONTROL_CONDITIONS = CONDITIONS[5:]
PROFILES = ("train", "development", "sealed_validation")
FORMAL_PROFILES = ("development", "sealed_validation")
DEFAULT_TARGET_LAYERS = tuple(range(42))
SHARED_STATE_BATCHING_POLICY = (
    "complete four-query shared-write families are kept in one evaluation batch"
)
ANSWER_LOGIT_POLICY = (
    "full-sequence hidden states with vocabulary logits projected only at the union "
    "of supervised causal answer-predictor positions; ignored labels and token-mean "
    "cross-entropy are unchanged"
)


# These functions operate on a small duck-typed example/batch interface and are
# shared with the synthetic causal proof runner.
build_delta_config = runtime.build_delta_config
collate_examples = runtime.collate_examples
causal_answer_loss = runtime.causal_answer_loss
route_loss_and_predictions = runtime.route_loss_and_predictions
selected_route_logits = runtime.selected_route_logits
_write_episode_batch = runtime._write_episode_batch
_read_episode_batch = runtime._read_episode_batch
_answer_prediction_token_ids = runtime._answer_prediction_token_ids
_answer_exact_predictions = runtime._answer_exact_predictions
_greedy_answer_predictions = runtime._greedy_answer_predictions
_state_digests = runtime._state_digests
_dtype = runtime._dtype
_signed_payload = runtime._signed_payload
_state_dict_sha256 = runtime._state_dict_sha256
_load_model_and_tokenizer = runtime._load_model_and_tokenizer


def train_model(
    model: torch.nn.Module,
    examples: Sequence[Any],
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
    training_conditions: str | Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
) -> dict[str, Any]:
    """Reuse the proven optimizer loop while emitting natural-run evidence."""

    runtime_progress = progress_path.with_name(
        f".{progress_path.name}.synthetic-runtime.tmp"
    )
    if progress_path.exists() or runtime_progress.exists():
        raise ValueError("Natural training progress paths must be fresh")
    try:
        result = dict(
            runtime.train_model(
                model,
                examples,
                seed=seed,
                epochs=epochs,
                max_steps=max_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                answer_weight=answer_weight,
                route_weight=route_weight,
                max_grad_norm=max_grad_norm,
                pad_token_id=pad_token_id,
                device=device,
                dtype=dtype,
                progress_path=runtime_progress,
            )
        )
        records: list[dict[str, Any]] = []
        with runtime_progress.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = _require_mapping(
                    json.loads(line), f"runtime training step {line_number}"
                )
                if record.get("schema") != "rwkv_ms_synthetic_compositional_train_step.v3":
                    raise ValueError("Shared optimizer emitted an unexpected progress schema")
                natural_record = dict(record)
                natural_record["schema"] = TRAIN_STEP_SCHEMA
                natural_record["training_conditions"] = list(
                    _parse_training_conditions(training_conditions)
                )
                records.append(natural_record)
        if not records or len(records) != result.get("steps"):
            raise ValueError("Natural training progress does not bind every optimizer step")
        progress_path.write_text(
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        result["progress_schema"] = TRAIN_STEP_SCHEMA
        result["progress_sha256"] = source.sha256_file(progress_path)
        return result
    finally:
        runtime_progress.unlink(missing_ok=True)


def load_pristine_base_model(
    model_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
) -> torch.nn.Module:
    """Load frozen Gemma without ever attaching a Delta-Mem wrapper."""

    model = runtime.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    runtime._disable_training_cache(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if list(iter_delta_mem_modules(model)):
        raise RuntimeError("Pristine frozen base unexpectedly contains Delta-Mem modules")
    return model


@dataclass(frozen=True)
class NaturalRecord:
    record_id: str
    semantic_slot: int
    physical_slot: int
    key_text: str
    value_json: str
    write_text: str


@dataclass(frozen=True)
class NaturalQuery:
    query_id: str
    target_slot: int
    target_record_id: str
    address_text: str
    read_prompt: str
    gold_json: str
    expected_json_by_condition: Mapping[str, str]
    rewrite_records: tuple[NaturalRecord, ...]
    record_payload_sha256_by_condition: Mapping[str, str]
    binding_absent_from_training: Mapping[str, bool]
    shared_correct_runtime_state_group: str


@dataclass(frozen=True)
class NaturalEpisode:
    episode_id: str
    split: str
    task: str
    passage_components: tuple[str, ...]
    records_by_condition: Mapping[str, tuple[NaturalRecord, ...]]
    queries: tuple[NaturalQuery, ...]


@dataclass(frozen=True)
class NaturalMemoryExample:
    row_id: str
    memory_state_id: str
    source_split: str
    source_mapping_offset: int
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
    expected_value: str
    target_slot_rewrite_selection: dict[str, Any] | None
    episode_id: str
    task: str
    semantic_target_slot: int
    write_record_ids: tuple[str, ...]
    write_semantic_slots: tuple[int, ...]
    write_value_jsons: tuple[str, ...]
    record_payload_sha256: str
    binding_absent_from_training: bool | None
    query_prefix_length: int


@dataclass(frozen=True)
class ProfileBundle:
    profile: str
    train_episodes: tuple[NaturalEpisode, ...]
    evaluation_episodes: tuple[NaturalEpisode, ...]
    evaluation_split: str
    development_manifest: Mapping[str, Any]
    sealed_manifest: Mapping[str, Any] | None
    source_paths: tuple[Path, ...]
    model_binding: Mapping[str, Any]
    eligibility: Mapping[str, Any]


@dataclass(frozen=True)
class GateThresholds:
    answer_exact_min: float = 0.80
    route_accuracy_min: float = 0.95
    rewrite_output_change_min: float = 0.80


def configure_hf_mirror(endpoint: str | None = None) -> str:
    requested = endpoint or os.environ.get("HF_ENDPOINT") or HF_MIRROR_ENDPOINT
    if requested.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {requested!r}"
        )
    current = os.environ.get("HF_ENDPOINT")
    if current is not None and current.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {current!r}"
        )
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    return HF_MIRROR_ENDPOINT


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _require_sequence(value: Any, description: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{description} must be an array")
    return value


def _canonical_value(value: Any) -> str:
    return source.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return source.sha256_text(source.canonical_json(value))


def _adapt_record(value: Any, *, location: str) -> NaturalRecord:
    record = _require_mapping(value, location)
    semantic_slot = record.get("slot_id")
    physical_slot = record.get("physical_index")
    if type(semantic_slot) is not int or semantic_slot not in range(RECORDS_PER_EPISODE):
        raise ValueError(f"{location} has an invalid semantic slot")
    if type(physical_slot) is not int or physical_slot not in range(RECORDS_PER_EPISODE):
        raise ValueError(f"{location} has an invalid physical slot")
    value_json = str(record.get("value_json", ""))
    if value_json != _canonical_value(record.get("value")):
        raise ValueError(f"{location} value_json is not canonical")
    result = NaturalRecord(
        record_id=str(record.get("record_id", "")),
        semantic_slot=semantic_slot,
        physical_slot=physical_slot,
        key_text=str(record.get("key_text", "")),
        value_json=value_json,
        write_text=str(record.get("write_text", "")),
    )
    if not all((result.record_id, result.key_text, result.value_json, result.write_text)):
        raise ValueError(f"{location} omits a required record field")
    if result.key_text not in result.write_text or result.value_json not in result.write_text:
        raise ValueError(f"{location} write text does not bind its key and value")
    return result


def _validate_record_set(
    records: Sequence[NaturalRecord],
    *,
    location: str,
    allow_empty: bool = False,
) -> tuple[NaturalRecord, ...]:
    result = tuple(records)
    if allow_empty and not result:
        return result
    if len(result) != RECORDS_PER_EPISODE:
        raise ValueError(f"{location} must contain exactly four records")
    if {record.semantic_slot for record in result} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError(f"{location} does not cover all semantic slots")
    if {record.physical_slot for record in result} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError(f"{location} does not cover all physical slots")
    if len({record.record_id for record in result}) != RECORDS_PER_EPISODE:
        raise ValueError(f"{location} has duplicate record IDs")
    return result


def _record_at_semantic_slot(
    records: Sequence[NaturalRecord], slot: int
) -> NaturalRecord:
    matches = [record for record in records if record.semantic_slot == slot]
    if len(matches) != 1:
        raise ValueError(f"State has {len(matches)} records for semantic slot {slot}")
    return matches[0]


def adapt_episode(raw_value: Mapping[str, Any]) -> NaturalEpisode:
    """Validate the source schema and isolate all generator-specific field access."""

    raw = _require_mapping(raw_value, "episode")
    if raw.get("schema") != source.SCHEMA:
        raise ValueError(
            f"Natural episode schema must be {source.SCHEMA}, not {raw.get('schema')!r}"
        )
    episode_id = str(raw.get("episode_id", ""))
    split = str(raw.get("split", ""))
    task = str(raw.get("task", ""))
    if not episode_id or split not in source.SPLITS or not task:
        raise ValueError("Natural episode identity is invalid")

    state_variants = _require_mapping(raw.get("state_variants"), "state_variants")
    required_states = {
        "correct_state",
        "donor_state",
        "value_swap",
        "shuffled_slots",
        "no_state",
    }
    if set(state_variants) != required_states:
        raise ValueError("Natural episode state variants differ from the v2 contract")

    records_by_condition: dict[str, tuple[NaturalRecord, ...]] = {}
    raw_records_by_condition: dict[str, Sequence[Any]] = {}
    payload_by_condition: dict[str, str] = {}
    for condition in required_states:
        variant = _require_mapping(state_variants[condition], f"state {condition}")
        raw_records = _require_sequence(
            variant.get("records"), f"state {condition} records"
        )
        raw_records_by_condition[condition] = raw_records
        expected_payload = str(variant.get("record_payload_sha256", ""))
        actual_payload = _sha256_json(raw_records)
        if expected_payload != actual_payload:
            raise ValueError(f"State {condition} record payload hash differs")
        payload_by_condition[condition] = actual_payload
        adapted = tuple(
            _adapt_record(record, location=f"{episode_id}:{condition}:{index}")
            for index, record in enumerate(raw_records)
        )
        records_by_condition[condition] = _validate_record_set(
            adapted,
            location=f"{episode_id}:{condition}",
            allow_empty=condition == "no_state",
        )

    correct_records = records_by_condition["correct_state"]
    raw_canonical_records = _require_sequence(raw.get("records"), "episode records")
    if source.canonical_json(raw_canonical_records) != source.canonical_json(
        raw_records_by_condition["correct_state"]
    ):
        raise ValueError("Episode records differ from correct_state")
    correct_identity = {
        record.semantic_slot: (record.record_id, record.key_text)
        for record in correct_records
    }
    for condition in ("donor_state", "value_swap", "shuffled_slots"):
        identity = {
            record.semantic_slot: (record.record_id, record.key_text)
            for record in records_by_condition[condition]
        }
        if identity != correct_identity:
            raise ValueError(f"State {condition} changed a semantic key")
    for condition in ("donor_state", "value_swap"):
        unchanged = [
            slot
            for slot in range(RECORDS_PER_EPISODE)
            if _record_at_semantic_slot(
                records_by_condition[condition], slot
            ).value_json
            == _record_at_semantic_slot(correct_records, slot).value_json
        ]
        if unchanged:
            raise ValueError(f"State {condition} left values unchanged at {unchanged}")
    shuffled = records_by_condition["shuffled_slots"]
    if {
        (record.semantic_slot, record.value_json) for record in shuffled
    } != {
        (record.semantic_slot, record.value_json) for record in correct_records
    }:
        raise ValueError("shuffled_slots changed a semantic value")
    donor_components = tuple(
        str(value)
        for value in _require_sequence(
            raw.get("donor_source_component_ids"),
            "donor_source_component_ids",
        )
    )
    if (
        len(donor_components) != RECORDS_PER_EPISODE
        or len(set(donor_components)) != RECORDS_PER_EPISODE
    ):
        raise ValueError(
            "Natural donor state is not backed by four distinct external components"
        )

    query_deltas = _require_mapping(
        raw.get("query_counterfactual_records"),
        "query_counterfactual_records",
    )
    raw_queries = _require_sequence(raw.get("queries"), "queries")
    if len(raw_queries) != RECORDS_PER_EPISODE:
        raise ValueError("Natural episode must contain exactly four queries")
    queries: list[NaturalQuery] = []
    for query_index, raw_query_value in enumerate(raw_queries):
        raw_query = _require_mapping(raw_query_value, f"query {query_index}")
        target_slot = raw_query.get("target_slot")
        if type(target_slot) is not int or target_slot not in range(RECORDS_PER_EPISODE):
            raise ValueError(f"Query {query_index} has an invalid target slot")
        correct_target = _record_at_semantic_slot(correct_records, target_slot)
        if raw_query.get("target_record_id") != correct_target.record_id:
            raise ValueError(f"Query {query_index} target record differs")
        if raw_query.get("address_text") != correct_target.key_text:
            raise ValueError(f"Query {query_index} address differs from its key")
        if raw_query.get("answer_absent_from_read_prompt") is not True:
            raise ValueError(f"Query {query_index} does not assert answer absence")
        gold_json = str(raw_query.get("gold_json", ""))
        if gold_json != correct_target.value_json or gold_json != _canonical_value(
            raw_query.get("gold")
        ):
            raise ValueError(f"Query {query_index} gold value differs")
        read_prompt = str(raw_query.get("read_prompt", ""))
        if (
            not read_prompt
            or correct_target.key_text not in read_prompt
            or gold_json in read_prompt
            or "memory_value:" in read_prompt
        ):
            raise ValueError(f"Query {query_index} read prompt leaks an answer")

        expected_raw = _require_mapping(
            raw_query.get("expected_by_state"),
            f"query {query_index} expected_by_state",
        )
        if set(expected_raw) != set(CONDITIONS):
            raise ValueError(f"Query {query_index} expected conditions differ")
        expected = {
            condition: _canonical_value(expected_raw[condition])
            for condition in CONDITIONS
        }
        for condition in ("correct_state", "donor_state", "value_swap", "shuffled_slots"):
            condition_target = _record_at_semantic_slot(
                records_by_condition[condition], target_slot
            )
            if expected[condition] != condition_target.value_json:
                raise ValueError(
                    f"Query {query_index} expectation differs for {condition}"
                )
        if expected["no_state"] != gold_json or expected["pristine_frozen_base"] != gold_json:
            raise ValueError(f"Query {query_index} control expectation differs")

        delta_group = _require_mapping(
            query_deltas.get(str(target_slot)),
            f"query {query_index} counterfactual group",
        )
        if delta_group.get("base_state") != "correct_state":
            raise ValueError(f"Query {query_index} rewrite base differs")
        rewrite = _require_mapping(
            delta_group.get("target_slot_rewrite"),
            f"query {query_index} target_slot_rewrite",
        )
        if rewrite.get("replace_slot") != target_slot:
            raise ValueError(f"Query {query_index} rewrite slot differs")
        raw_replacement = _require_mapping(
            rewrite.get("replacement_record"),
            f"query {query_index} replacement record",
        )
        replacement = _adapt_record(
            raw_replacement,
            location=f"{episode_id}:query-{target_slot}:replacement",
        )
        if (
            replacement.semantic_slot != target_slot
            or replacement.record_id != correct_target.record_id
            or replacement.key_text != correct_target.key_text
            or replacement.physical_slot != correct_target.physical_slot
        ):
            raise ValueError(f"Query {query_index} rewrite changed target identity")
        if replacement.value_json == correct_target.value_json:
            raise ValueError(f"Query {query_index} rewrite did not change the value")
        if expected["target_slot_rewrite"] != replacement.value_json:
            raise ValueError(f"Query {query_index} rewrite expectation differs")
        rewrite_records = list(correct_records)
        replacement_index = next(
            index
            for index, record in enumerate(rewrite_records)
            if record.semantic_slot == target_slot
        )
        rewrite_records[replacement_index] = replacement
        rewrite_records_tuple = _validate_record_set(
            rewrite_records,
            location=f"{episode_id}:query-{target_slot}:rewrite-state",
        )
        rewrite_payload = _sha256_json(
            [
                raw_replacement
                if int(record.get("slot_id", -1)) == target_slot
                else record
                for record in raw_records_by_condition["correct_state"]
            ]
        )
        if rewrite.get("result_record_payload_sha256") != rewrite_payload:
            raise ValueError(f"Query {query_index} rewrite payload hash differs")

        payloads_raw = _require_mapping(
            raw_query.get("record_payload_sha256_by_condition"),
            f"query {query_index} record payload hashes",
        )
        expected_payloads = {
            **payload_by_condition,
            "target_slot_rewrite": rewrite_payload,
        }
        if set(payloads_raw) != set(expected_payloads) or any(
            payloads_raw[name] != digest
            for name, digest in expected_payloads.items()
        ):
            raise ValueError(f"Query {query_index} condition payload hashes differ")

        binding_absence_raw = _require_mapping(
            raw_query.get("binding_absent_from_training"),
            f"query {query_index} training-binding audit",
        )
        binding_absence = {
            str(name): value is True for name, value in binding_absence_raw.items()
        }
        if split != "train" and not all(binding_absence.values()):
            raise ValueError(f"Query {query_index} overlaps a training binding")
        shared_group = str(raw_query.get("shared_correct_runtime_state_group", ""))
        if not shared_group:
            raise ValueError(f"Query {query_index} omits its correct-state group")
        queries.append(
            NaturalQuery(
                query_id=str(raw_query.get("query_id", "")),
                target_slot=target_slot,
                target_record_id=correct_target.record_id,
                address_text=correct_target.key_text,
                read_prompt=read_prompt,
                gold_json=gold_json,
                expected_json_by_condition=expected,
                rewrite_records=rewrite_records_tuple,
                record_payload_sha256_by_condition=expected_payloads,
                binding_absent_from_training=binding_absence,
                shared_correct_runtime_state_group=shared_group,
            )
        )
        if not queries[-1].query_id:
            raise ValueError(f"Query {query_index} omits query_id")
    if {query.target_slot for query in queries} != set(range(RECORDS_PER_EPISODE)):
        raise ValueError("Natural episode queries do not cover all four target slots")
    if len({query.query_id for query in queries}) != RECORDS_PER_EPISODE:
        raise ValueError("Natural episode has duplicate query IDs")
    if len({query.shared_correct_runtime_state_group for query in queries}) != 1:
        raise ValueError("Natural episode correct queries do not share one state group")

    components = tuple(str(value) for value in _require_sequence(
        raw.get("passage_components"), "passage_components"
    ))
    if len(components) != RECORDS_PER_EPISODE or len(set(components)) != len(components):
        raise ValueError("Natural episode passage components are invalid")
    return NaturalEpisode(
        episode_id=episode_id,
        split=split,
        task=task,
        passage_components=components,
        records_by_condition=records_by_condition,
        queries=tuple(sorted(queries, key=lambda query: query.target_slot)),
    )


def _flat_ints(value: Any, description: str) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{description} unexpectedly contains a batch")
        value = value[0]
    result = tuple(int(item) for item in value)
    if not result:
        raise ValueError(f"{description} is empty")
    return result


def _flat_offsets(value: Any, description: str) -> tuple[tuple[int, int], ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        if len(value) != 1:
            raise ValueError(f"{description} unexpectedly contains a batch")
        value = value[0]
    result = tuple((int(start), int(end)) for start, end in value)
    if not result:
        raise ValueError(f"{description} is empty")
    return result


def _render_user_chat(
    tokenizer: Any,
    content: str,
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str) or content not in rendered:
        raise ValueError("Tokenizer chat template did not preserve the user content")
    return rendered


def _tokenize_offsets(tokenizer: Any, text: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    input_ids = _flat_ints(encoded["input_ids"], "input_ids")
    attention_mask = _flat_ints(encoded.get("attention_mask", [1] * len(input_ids)), "attention_mask")
    offsets = _flat_offsets(encoded["offset_mapping"], "offset_mapping")
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(offsets):
        raise ValueError("Tokenizer IDs, attention mask, and offsets are misaligned")
    if not all(attention_mask):
        raise ValueError("Unpadded example tokenization contains masked tokens")
    return input_ids, attention_mask, offsets


def _span_mask(
    offsets: Sequence[tuple[int, int]],
    *,
    start: int,
    end: int,
    description: str,
) -> tuple[bool, ...]:
    if start < 0 or end <= start:
        raise ValueError(f"{description} character span is invalid")
    selected = tuple(
        token_end > token_start and token_end > start and token_start < end
        for token_start, token_end in offsets
    )
    if not any(selected):
        raise ValueError(f"{description} selected no tokens")
    return selected


def encode_record(tokenizer: Any, record: NaturalRecord) -> dict[str, Any]:
    rendered = _render_user_chat(
        tokenizer,
        record.write_text,
        add_generation_prompt=False,
    )
    content_start = rendered.find(record.write_text)
    key_local = record.write_text.find(record.key_text)
    value_local = record.write_text.find(record.value_json)
    if key_local < 0 or value_local < 0:
        raise ValueError(f"Record {record.record_id} write spans are absent")
    input_ids, attention_mask, offsets = _tokenize_offsets(tokenizer, rendered)
    key_mask = _span_mask(
        offsets,
        start=content_start + key_local,
        end=content_start + key_local + len(record.key_text),
        description=f"record {record.record_id} key",
    )
    value_mask = _span_mask(
        offsets,
        start=content_start + value_local,
        end=content_start + value_local + len(record.value_json),
        description=f"record {record.record_id} value",
    )
    if any(left and right for left, right in zip(key_mask, value_mask, strict=True)):
        raise ValueError(f"Record {record.record_id} key and value masks overlap")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "key_mask": key_mask,
        "value_mask": value_mask,
        "record_id": record.record_id,
        "semantic_slot": record.semantic_slot,
        "physical_slot": record.physical_slot,
        "value_json": record.value_json,
    }


def encode_query_read(
    tokenizer: Any,
    query: NaturalQuery,
    expected_value_json: str,
) -> dict[str, Any]:
    if expected_value_json in query.read_prompt:
        raise ValueError(f"Query {query.query_id} leaks its expected answer")
    prefix = _render_user_chat(
        tokenizer,
        query.read_prompt,
        add_generation_prompt=True,
    )
    if expected_value_json in prefix:
        raise ValueError(f"Query {query.query_id} rendered prefix leaks its answer")
    rendered = prefix + expected_value_json
    input_ids, attention_mask, offsets = _tokenize_offsets(tokenizer, rendered)
    crossing = [
        (start, end)
        for start, end in offsets
        if start < len(prefix) < end
    ]
    if crossing:
        raise ValueError(
            f"Query {query.query_id} has a tokenizer token crossing the prefix/answer boundary"
        )
    address_local = query.read_prompt.find(query.address_text)
    content_start = prefix.find(query.read_prompt)
    query_mask = _span_mask(
        offsets,
        start=content_start + address_local,
        end=content_start + address_local + len(query.address_text),
        description=f"query {query.query_id} address",
    )
    answer_mask = _span_mask(
        offsets,
        start=len(prefix),
        end=len(rendered),
        description=f"query {query.query_id} answer",
    )
    answer_positions = [index for index, selected in enumerate(answer_mask) if selected]
    if answer_positions != list(range(answer_positions[0], answer_positions[-1] + 1)):
        raise ValueError(f"Query {query.query_id} answer tokens are not contiguous")
    if any(answer_mask[: answer_positions[0]]) or any(
        query_mask[answer_positions[0] :]
    ):
        raise ValueError(f"Query {query.query_id} masks cross the answer boundary")
    answer_ids = tuple(input_ids[index] for index in answer_positions)
    if tokenizer.decode(list(answer_ids), skip_special_tokens=True).strip() == "":
        raise ValueError(
            f"Query {query.query_id} canonical JSON answer decodes to empty text"
        )
    if any(left and right for left, right in zip(query_mask, answer_mask, strict=True)):
        raise ValueError(f"Query {query.query_id} route and answer masks overlap")
    labels = tuple(
        token if selected else -100
        for token, selected in zip(input_ids, answer_mask, strict=True)
    )
    prefix_length = answer_positions[0]
    if any(answer_mask[:prefix_length]) or not any(query_mask[:prefix_length]):
        raise ValueError(f"Query {query.query_id} prefix masks are invalid")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_mask": query_mask,
        "answer_mask": answer_mask,
        "labels": labels,
        "expected_answer_token_ids": tuple(
            token for token, selected in zip(input_ids, answer_mask, strict=True) if selected
        ),
        "query_prefix_length": prefix_length,
    }


def _records_for_query(
    episode: NaturalEpisode,
    query: NaturalQuery,
    condition: str,
) -> tuple[NaturalRecord, ...]:
    if condition == "target_slot_rewrite":
        return query.rewrite_records
    if condition in CONTROL_CONDITIONS:
        return ()
    return episode.records_by_condition[condition]


def build_condition_examples(
    episodes: Sequence[NaturalEpisode],
    tokenizer: Any,
    condition: str,
) -> list[NaturalMemoryExample]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown natural condition {condition!r}")
    examples: list[NaturalMemoryExample] = []
    for episode in episodes:
        for query in episode.queries:
            records = _records_for_query(episode, query, condition)
            encoded_records = tuple(encode_record(tokenizer, record) for record in records)
            read = encode_query_read(
                tokenizer,
                query,
                query.expected_json_by_condition[condition],
            )
            if records:
                target_record = _record_at_semantic_slot(records, query.target_slot)
                target_slot: int | None = target_record.physical_slot
            else:
                target_slot = None
            if condition == "correct_state":
                memory_state_id = query.shared_correct_runtime_state_group
            elif condition == "target_slot_rewrite":
                memory_state_id = f"{episode.episode_id}:{condition}:q{query.target_slot}"
            else:
                memory_state_id = f"{episode.episode_id}:{condition}"
            absent = query.binding_absent_from_training.get(condition)
            examples.append(
                NaturalMemoryExample(
                    row_id=query.query_id,
                    memory_state_id=memory_state_id,
                    source_split=episode.split,
                    source_mapping_offset=0,
                    condition=condition,
                    write_records=encoded_records,
                    write_slots=tuple(record.physical_slot for record in records),
                    read_input_ids=read["input_ids"],
                    read_attention_mask=read["attention_mask"],
                    query_mask=read["query_mask"],
                    answer_mask=read["answer_mask"],
                    labels=read["labels"],
                    target_slot=target_slot,
                    expected_answer_token_ids=read["expected_answer_token_ids"],
                    expected_value=query.expected_json_by_condition[condition],
                    target_slot_rewrite_selection=(
                        {"semantic_target_slot": query.target_slot}
                        if condition == "target_slot_rewrite"
                        else None
                    ),
                    episode_id=episode.episode_id,
                    task=episode.task,
                    semantic_target_slot=query.target_slot,
                    write_record_ids=tuple(record.record_id for record in records),
                    write_semantic_slots=tuple(record.semantic_slot for record in records),
                    write_value_jsons=tuple(record.value_json for record in records),
                    record_payload_sha256=query.record_payload_sha256_by_condition.get(
                        condition,
                        _sha256_json([]),
                    ),
                    binding_absent_from_training=absent,
                    query_prefix_length=read["query_prefix_length"],
                )
            )
    return examples


def build_training_examples(
    episodes: Sequence[NaturalEpisode],
    tokenizer: Any,
    training_conditions: Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
) -> list[NaturalMemoryExample]:
    examples: list[NaturalMemoryExample] = []
    for condition in _parse_training_conditions(training_conditions):
        examples.extend(build_condition_examples(episodes, tokenizer, condition))
    return examples


def _read_json_file(path: Path, description: str) -> Mapping[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{description} is not a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {description}: {resolved}") from error
    return _require_mapping(value, description)


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        requested = raw_path.expanduser()
        if requested.is_symlink():
            raise ValueError(f"Bound artifact must not be a symbolic link: {requested}")
        path = requested.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Bound artifact is not a regular file: {path}")
        key = str(path)
        if key in snapshot:
            continue
        snapshot[key] = {
            "bytes": path.stat().st_size,
            "sha256": source.sha256_file(path),
        }
    return dict(sorted(snapshot.items()))


def assert_snapshot_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    *,
    description: str,
) -> dict[str, dict[str, Any]]:
    paths = [Path(path) for path in before]
    try:
        after = snapshot_files(paths)
    except (OSError, ValueError) as error:
        raise ValueError(f"{description} artifacts changed or disappeared") from error
    normalized_before = {
        str(path): dict(fingerprint) for path, fingerprint in before.items()
    }
    if after != normalized_before:
        changed = sorted(
            path
            for path in set(after) | set(normalized_before)
            if after.get(path) != normalized_before.get(path)
        )
        raise ValueError(f"{description} artifacts changed: {changed}")
    return after


def _validate_manifest_common(
    manifest: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    if manifest.get("schema") != source.SCHEMA:
        raise ValueError(f"Natural manifest schema differs at {path}")
    if manifest.get("hf_endpoint") != HF_MIRROR_ENDPOINT:
        raise ValueError(f"Natural manifest does not bind the HF mirror at {path}")
    if not source.verify_manifest_receipt(manifest):
        raise ValueError(f"Natural manifest self-receipt is invalid at {path}")
    split_audit = _require_mapping(manifest.get("split_audit"), "split_audit")
    signature_audit = _require_mapping(
        manifest.get("signature_audit"), "signature_audit"
    )
    if (
        split_audit.get("passage_disjoint") is not True
        or split_audit.get("normalized_units_passage_disjoint") is not True
        or split_audit.get("normalized_signature_cross_split_overlap_count") != 0
        or signature_audit.get("signature_components_atomic") is not True
    ):
        raise ValueError(f"Natural manifest split isolation failed at {path}")
    binding = _require_mapping(manifest.get("model_binding"), "model_binding")
    if (
        binding.get("weights_bound") is not True
        or binding.get("hf_endpoint") != HF_MIRROR_ENDPOINT
        or not isinstance(binding.get("binding_sha256"), str)
    ):
        raise ValueError(f"Natural manifest lacks a formal local model binding at {path}")


def _validate_profile_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    profile: str,
) -> str:
    _validate_manifest_common(manifest, path=manifest_path)
    materialized = tuple(manifest.get("materialized_splits", ()))
    build_profile = manifest.get("build_profile")
    if profile in {"train", "development"}:
        if build_profile != "development" or materialized != (
            "train",
            "development",
        ):
            raise ValueError("Train/development run requires an exact development package")
        if (manifest_path.parent / "sealed_validation.jsonl").exists():
            raise ValueError("Development package unexpectedly exposes sealed validation")
        return "train" if profile == "train" else "development"
    if profile == "sealed_validation":
        if build_profile != "sealed_validation" or materialized != (
            "sealed_validation",
        ):
            raise ValueError("Sealed run requires an exact sealed-validation package")
        for forbidden in ("train.jsonl", "development.jsonl"):
            if (manifest_path.parent / forbidden).exists():
                raise ValueError(f"Sealed package unexpectedly exposes {forbidden}")
        sealed_lock = _require_mapping(manifest.get("sealed_lock"), "sealed_lock")
        lock_receipt = _require_mapping(sealed_lock.get("receipt"), "sealed lock receipt")
        if (
            lock_receipt.get("schema") != source.SEALED_LOCK_SCHEMA
            or lock_receipt.get("configuration_frozen") is not True
            or lock_receipt.get("benchmark_contract_sha256")
            != manifest.get("benchmark_contract_sha256")
        ):
            raise ValueError("Sealed package does not contain a valid frozen lock")
        return "sealed_validation"
    raise ValueError(f"Unknown natural profile {profile!r}")


def _load_split(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    split: str,
) -> tuple[tuple[NaturalEpisode, ...], Path]:
    output_hashes = _require_mapping(manifest.get("output_sha256"), "output_sha256")
    expected_hash = output_hashes.get(split)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"Manifest does not bind selected split {split}")
    requested_path = manifest_path.parent / f"{split}.jsonl"
    if requested_path.is_symlink():
        raise ValueError(f"Selected split must not be a symbolic link: {requested_path}")
    path = requested_path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Selected split is not a regular file: {path}")
    before_hash = source.sha256_file(path)
    if before_hash != expected_hash:
        raise ValueError(f"Selected split hash differs before read: {split}")
    episodes: list[NaturalEpisode] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            episode = adapt_episode(raw)
            if episode.split != split:
                raise ValueError(f"Episode split differs at {path}:{line_number}")
            episodes.append(episode)
    if source.sha256_file(path) != before_hash:
        raise ValueError(f"Selected split changed while being read: {split}")
    if not episodes:
        raise ValueError(f"Selected split is empty: {split}")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError(f"Selected split has duplicate episode IDs: {split}")
    expected_count = _require_mapping(
        manifest.get("episode_audit"), "episode_audit"
    ).get("episodes_by_split", {}).get(split)
    if expected_count != len(episodes):
        raise ValueError(f"Selected split count differs: {split}")
    return tuple(episodes), path


def load_profile_bundle(
    source_manifest: Path,
    *,
    profile: str,
) -> ProfileBundle:
    """Open only the JSONL files authorized by the selected run profile."""

    configure_hf_mirror()
    if profile not in PROFILES:
        raise ValueError(f"Unknown natural profile {profile!r}")
    requested_manifest = source_manifest.expanduser()
    if requested_manifest.is_symlink():
        raise ValueError(
            f"Natural source manifest must not be a symbolic link: {requested_manifest}"
        )
    manifest_path = requested_manifest.resolve(strict=True)
    manifest = _read_json_file(manifest_path, "natural source manifest")
    evaluation_split = _validate_profile_manifest(
        manifest,
        manifest_path=manifest_path,
        profile=profile,
    )
    selected_paths: list[Path] = [manifest_path]
    if profile == "sealed_validation":
        train_episodes: tuple[NaturalEpisode, ...] = ()
        evaluation_episodes, evaluation_path = _load_split(
            manifest,
            manifest_path=manifest_path,
            split="sealed_validation",
        )
        selected_paths.append(evaluation_path)
        development_manifest: Mapping[str, Any] = {}
        sealed_manifest: Mapping[str, Any] | None = manifest
    else:
        train_episodes, train_path = _load_split(
            manifest,
            manifest_path=manifest_path,
            split="train",
        )
        selected_paths.append(train_path)
        if profile == "train":
            evaluation_episodes = train_episodes
        else:
            evaluation_episodes, evaluation_path = _load_split(
                manifest,
                manifest_path=manifest_path,
                split="development",
            )
            selected_paths.append(evaluation_path)
        development_manifest = manifest
        sealed_manifest = None

    train_components = {
        component
        for episode in train_episodes
        for component in episode.passage_components
    }
    evaluation_components = {
        component
        for episode in evaluation_episodes
        for component in episode.passage_components
    }
    component_overlap = (
        train_components & evaluation_components if profile != "train" else set()
    )
    if component_overlap:
        raise ValueError("Loaded train and evaluation passage components overlap")
    heldout_queries = [
        query
        for episode in evaluation_episodes
        if episode.split != "train"
        for query in episode.queries
    ]
    heldout_novel = all(
        query.binding_absent_from_training
        and all(query.binding_absent_from_training.values())
        for query in heldout_queries
    )
    if heldout_queries and not heldout_novel:
        raise ValueError("Evaluation bindings overlap training")
    tasks = sorted({episode.task for episode in evaluation_episodes})
    eligibility = {
        "profile": profile,
        "evaluation_split": evaluation_split,
        "optimizer_authorized": profile != "sealed_validation",
        "opened_splits": (
            ["train"]
            if profile == "train"
            else ["train", "development"]
            if profile == "development"
            else ["sealed_validation"]
        ),
        "train_episodes": len(train_episodes),
        "evaluation_episodes": len(evaluation_episodes),
        "evaluation_tasks": tasks,
        "passage_component_overlap_count": len(component_overlap),
        "heldout_binding_novel": heldout_novel,
        "manifest_receipt_valid": True,
        "passed": not component_overlap and (not heldout_queries or heldout_novel),
    }
    return ProfileBundle(
        profile=profile,
        train_episodes=train_episodes,
        evaluation_episodes=evaluation_episodes,
        evaluation_split=evaluation_split,
        development_manifest=development_manifest,
        sealed_manifest=sealed_manifest,
        source_paths=tuple(selected_paths),
        model_binding=_require_mapping(manifest["model_binding"], "model_binding"),
        eligibility=eligibility,
    )


def resolve_model_artifacts(
    model_binding: Mapping[str, Any],
    *,
    model_path: Path | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    declared_path = model_binding.get("local_model_path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("Manifest does not declare a local model path")
    declared_requested = Path(declared_path).expanduser()
    if declared_requested.is_symlink():
        raise ValueError("Manifest local model path must not be a symbolic link")
    declared = declared_requested.resolve(strict=True)
    runtime_requested = declared_requested if model_path is None else model_path.expanduser()
    if runtime_requested.is_symlink():
        raise ValueError("Runtime model path must not be a symbolic link")
    resolved = runtime_requested.resolve(strict=True)
    if resolved != declared or not resolved.is_dir():
        raise ValueError("Runtime model path differs from the manifest binding")
    artifacts = _require_mapping(
        model_binding.get("local_artifacts"), "local model artifacts"
    )
    if not artifacts:
        raise ValueError("Manifest model binding contains no local artifacts")
    paths: list[Path] = []
    for name, raw_fingerprint in sorted(artifacts.items()):
        if Path(name).name != name:
            raise ValueError(f"Model artifact name is not local: {name!r}")
        fingerprint = _require_mapping(raw_fingerprint, f"model artifact {name}")
        requested_artifact = resolved / name
        if requested_artifact.is_symlink():
            raise ValueError(f"Model artifact must not be a symbolic link: {name}")
        path = requested_artifact.resolve(strict=True)
        if path.parent != resolved or not path.is_file():
            raise ValueError(f"Model artifact is invalid: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": source.sha256_file(path)}
        if actual != dict(fingerprint):
            raise ValueError(f"Model artifact hash differs: {name}")
        paths.append(path)
    return resolved, tuple(paths)


def select_complete_episodes(
    episodes: Sequence[NaturalEpisode], limit: int | None
) -> tuple[NaturalEpisode, ...]:
    if limit is None:
        return tuple(episodes)
    if limit <= 0:
        raise ValueError("Episode limit must be positive")
    return tuple(episodes[:limit])


def _batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _evaluation_batches(
    examples: Sequence[NaturalMemoryExample],
    *,
    condition: str,
    batch_size: int,
) -> Iterable[Sequence[NaturalMemoryExample]]:
    if condition not in SHARED_WRITE_CONDITIONS:
        yield from _batches(examples, batch_size)
        return
    if batch_size < RECORDS_PER_EPISODE:
        raise ValueError(
            "Shared-write evaluation batch size must hold one complete family"
        )
    families: dict[str, list[NaturalMemoryExample]] = {}
    for example in examples:
        families.setdefault(example.memory_state_id, []).append(example)
    ordered_families: list[tuple[NaturalMemoryExample, ...]] = []
    for state_id, raw_family in families.items():
        family = tuple(
            sorted(raw_family, key=lambda example: example.semantic_target_slot)
        )
        if [example.semantic_target_slot for example in family] != list(
            range(RECORDS_PER_EPISODE)
        ):
            raise ValueError(f"{condition} state family {state_id} is incomplete")
        ordered_families.append(family)
    families_per_batch = max(1, batch_size // RECORDS_PER_EPISODE)
    for family_batch in _batches(ordered_families, families_per_batch):
        yield tuple(example for family in family_batch for example in family)


def _control_state_absence_evidence(
    row_ids: Sequence[str],
    *,
    before_read_state_names: Sequence[str],
    after_read_state_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    before = list(before_read_state_names)
    after = list(after_read_state_names)
    return {
        row_id: {
            "projected_kv_state_names_before_read": before,
            "projected_kv_state_names_after_read": after,
            "projected_kv_state_absent_before_read": not before,
            "projected_kv_state_absent_after_read": not after,
        }
        for row_id in row_ids
    }


def _projected_kv_state_names(model: torch.nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    for module_name, module in iter_delta_mem_modules(model):
        for attribute in (
            "projected_kv_keys",
            "projected_kv_values",
            "projected_kv_occupied",
            "projected_kv_surprise",
        ):
            if getattr(module, attribute, None) is not None:
                names.append(f"{module_name}.{attribute}")
    return tuple(sorted(names))


def _decode_prediction(
    tokenizer: Any,
    token_ids: Sequence[int],
    expected_json: str,
) -> dict[str, Any]:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()
    parsed_json: Any = None
    parse_valid = False
    canonical = None
    try:
        parsed_json = json.loads(text)
        canonical = source.canonical_json(parsed_json)
        parse_valid = True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "text": text,
        "json_parse_valid": parse_valid,
        "canonical_json": canonical,
        "structured_json_exact": canonical == expected_json,
    }


def evaluate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[NaturalMemoryExample],
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
    if condition == "pristine_frozen_base" and module_names:
        raise RuntimeError(
            "Pristine frozen-base evaluation must use a model without Delta-Mem attached"
        )
    answer_exact: list[bool] = []
    structured_exact: list[bool] = []
    greedy_exact: list[bool] = []
    greedy_structured_exact: list[bool] = []
    token_correct = 0
    token_total = 0
    route_correct = 0
    route_total = 0
    route_by_layer = {
        name: {"correct": 0, "total": 0} for name in module_names
    }
    answer_predictions_by_row: dict[str, dict[str, Any]] = {}
    route_predictions_by_row: dict[str, dict[str, int]] = {}
    state_digest_by_row: dict[str, str] = {}
    control_state_absence_by_row: dict[str, dict[str, Any]] = {}
    occupancy_correct = 0
    occupancy_total = 0
    write_route_correct = 0
    write_route_total = 0
    absent_modules = 0
    possible_modules = 0
    task_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "teacher_forced_exact": 0,
            "teacher_forced_structured_exact": 0,
            "greedy_exact": 0,
            "greedy_structured_exact": 0,
            "route_correct": 0,
            "route_total": 0,
        }
    )

    model.eval()
    with torch.no_grad():
        for raw_batch in _evaluation_batches(
            list(examples), condition=condition, batch_size=batch_size
        ):
            batch = collate_examples(
                raw_batch,
                pad_token_id=pad_token_id,
                device=device,
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
            before_read_state_names = (
                _projected_kv_state_names(model)
                if condition in CONTROL_CONDITIONS
                else ()
            )

            logits, route_logits = _read_episode_batch(model, batch, dtype=dtype)
            exact, batch_token_correct, batch_token_total = (
                _answer_exact_predictions(logits, batch.labels)
            )
            teacher_predictions, expected_rows = _answer_prediction_token_ids(
                logits, batch.labels
            )
            generated_rows = (
                _greedy_answer_predictions(
                    model,
                    batch,
                    pad_token_id=pad_token_id,
                    dtype=dtype,
                )
                if greedy
                else [None] * len(batch.examples)
            )
            if condition in CONTROL_CONDITIONS:
                evidence = _control_state_absence_evidence(
                    [example.row_id for example in batch.examples],
                    before_read_state_names=before_read_state_names,
                    after_read_state_names=_projected_kv_state_names(model),
                )
                overlap = set(control_state_absence_by_row) & set(evidence)
                if overlap:
                    raise ValueError(
                        "Duplicate control state-absence rows: "
                        + ", ".join(sorted(overlap))
                    )
                control_state_absence_by_row.update(evidence)

            batch_row_predictions = {
                example.row_id: {} for example in batch.examples
            }
            possible_modules += len(module_names) * len(batch.examples)
            absent_modules += (len(module_names) - len(route_logits)) * len(
                batch.examples
            )
            if condition in CONTROL_CONDITIONS:
                if route_logits:
                    raise RuntimeError(
                        f"{condition} unexpectedly exposed projected-KV routes"
                    )
            else:
                if set(route_logits) != set(module_names) or not route_logits:
                    raise RuntimeError(
                        f"{condition} omitted projected-KV route evidence"
                    )
                for name, layer_logits in route_logits.items():
                    selected = selected_route_logits(layer_logits, batch.query_mask)
                    predictions = selected.argmax(dim=-1)
                    matches = predictions.eq(batch.target_slots)
                    matched = int(matches.sum().item())
                    route_correct += matched
                    route_total += len(batch.examples)
                    route_by_layer[name]["correct"] += matched
                    route_by_layer[name]["total"] += len(batch.examples)
                    for example, predicted, is_match in zip(
                        batch.examples,
                        predictions.detach().cpu().tolist(),
                        matches.detach().cpu().tolist(),
                        strict=True,
                    ):
                        batch_row_predictions[example.row_id][name] = int(predicted)
                        task_counts[example.task]["route_correct"] += int(is_match)
                        task_counts[example.task]["route_total"] += 1
                route_predictions_by_row.update(batch_row_predictions)

            for example, predicted, expected, is_exact, generated in zip(
                batch.examples,
                teacher_predictions,
                expected_rows,
                exact,
                generated_rows,
                strict=True,
            ):
                if example.row_id in answer_predictions_by_row:
                    raise ValueError(f"Duplicate evaluation row ID: {example.row_id}")
                if expected != example.expected_answer_token_ids:
                    raise RuntimeError(
                        f"Evaluation labels differ from {example.row_id} expectation"
                    )
                teacher_evidence = _decode_prediction(
                    tokenizer, predicted, example.expected_value
                )
                structured = bool(teacher_evidence["structured_json_exact"])
                structured_exact.append(structured)
                task = task_counts[example.task]
                task["rows"] += 1
                task["teacher_forced_exact"] += int(is_exact)
                task["teacher_forced_structured_exact"] += int(structured)
                greedy_evidence = None
                greedy_is_exact = None
                if generated is not None:
                    greedy_evidence = _decode_prediction(
                        tokenizer, generated, example.expected_value
                    )
                    greedy_is_exact = generated == example.expected_answer_token_ids
                    greedy_exact.append(greedy_is_exact)
                    greedy_structured = bool(
                        greedy_evidence["structured_json_exact"]
                    )
                    greedy_structured_exact.append(greedy_structured)
                    task["greedy_exact"] += int(greedy_is_exact)
                    task["greedy_structured_exact"] += int(greedy_structured)
                answer_predictions_by_row[example.row_id] = {
                    "episode_id": example.episode_id,
                    "task": example.task,
                    "expected_value_json": example.expected_value,
                    "expected_answer_token_ids": list(expected),
                    "teacher_forced_prediction_token_ids": list(predicted),
                    "teacher_forced_exact": bool(is_exact),
                    "teacher_forced_json": teacher_evidence,
                    "greedy_generated_token_ids": (
                        list(generated) if generated is not None else None
                    ),
                    "greedy_exact": greedy_is_exact,
                    "greedy_json": greedy_evidence,
                }
            answer_exact.extend(exact)
            token_correct += batch_token_correct
            token_total += batch_token_total
            reset_delta_mem_states(model)

    per_task: dict[str, dict[str, Any]] = {}
    for task, counts in sorted(task_counts.items()):
        rows = counts["rows"]
        per_task[task] = {
            **counts,
            "teacher_forced_answer_exact_accuracy": (
                counts["teacher_forced_exact"] / rows
            ),
            "teacher_forced_structured_json_exact_accuracy": (
                counts["teacher_forced_structured_exact"] / rows
            ),
            "greedy_answer_exact_accuracy": (
                counts["greedy_exact"] / rows if greedy else None
            ),
            "greedy_structured_json_exact_accuracy": (
                counts["greedy_structured_exact"] / rows if greedy else None
            ),
            "semantic_route_accuracy": (
                counts["route_correct"] / counts["route_total"]
                if counts["route_total"]
                else None
            ),
        }
    layer_metrics = {
        name: {
            **counts,
            "accuracy": counts["correct"] / counts["total"]
            if counts["total"]
            else None,
        }
        for name, counts in route_by_layer.items()
    }
    return {
        "condition": condition,
        "rows": len(examples),
        "teacher_forced_answer_exact_count": sum(answer_exact),
        "teacher_forced_answer_exact_accuracy": sum(answer_exact) / len(answer_exact),
        "teacher_forced_structured_json_exact_count": sum(structured_exact),
        "teacher_forced_structured_json_exact_accuracy": (
            sum(structured_exact) / len(structured_exact)
        ),
        "teacher_forced_answer_token_correct": token_correct,
        "teacher_forced_answer_token_total": token_total,
        "teacher_forced_answer_token_accuracy": token_correct / token_total,
        "greedy_answer_evaluated": greedy,
        "greedy_answer_exact_count": sum(greedy_exact) if greedy else None,
        "greedy_answer_exact_accuracy": (
            sum(greedy_exact) / len(greedy_exact) if greedy else None
        ),
        "greedy_structured_json_exact_count": (
            sum(greedy_structured_exact) if greedy else None
        ),
        "greedy_structured_json_exact_accuracy": (
            sum(greedy_structured_exact) / len(greedy_structured_exact)
            if greedy
            else None
        ),
        "answer_predictions_by_row": answer_predictions_by_row,
        "semantic_route_correct": route_correct,
        "semantic_route_total": route_total,
        "semantic_route_accuracy": route_correct / route_total if route_total else None,
        "route_by_layer": layer_metrics,
        "route_predictions_by_row": route_predictions_by_row,
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
        "route_absent_fraction": (
            absent_modules / possible_modules if possible_modules else 1.0
        ),
        "state_digest_by_row": state_digest_by_row,
        "runtime_state_absence_rows": len(control_state_absence_by_row),
        "runtime_state_absence_fraction": (
            sum(
                row["projected_kv_state_absent_before_read"]
                and row["projected_kv_state_absent_after_read"]
                for row in control_state_absence_by_row.values()
            )
            / len(control_state_absence_by_row)
            if control_state_absence_by_row
            else None
        ),
        "runtime_state_absence_by_row": control_state_absence_by_row,
        "delta_heads_disabled": condition == "pristine_frozen_base",
        "adapter_attached": bool(module_names),
        "attached_delta_mem_module_count": len(module_names),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "pristine_base_adapter_excluded": (
            condition == "pristine_frozen_base" and not module_names
        ),
        "per_task": per_task,
    }


def audit_correct_state_identity(
    examples: Sequence[NaturalMemoryExample],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not examples or any(example.condition != "correct_state" for example in examples):
        raise ValueError("Correct-state identity audit received other conditions")
    groups: dict[str, list[NaturalMemoryExample]] = defaultdict(list)
    for example in examples:
        groups[example.memory_state_id].append(example)
    state_digests = _require_mapping(
        evaluation.get("state_digest_by_row"), "correct state digests"
    )
    route_predictions = _require_mapping(
        evaluation.get("route_predictions_by_row"), "correct route predictions"
    )
    module_names = tuple(_require_mapping(
        evaluation.get("route_by_layer"), "correct route layers"
    ))
    identical_families = 0
    all_four_route_correct = 0
    route_family_total = 0
    families: dict[str, dict[str, Any]] = {}
    for group_id, raw_family in sorted(groups.items()):
        family = sorted(raw_family, key=lambda example: example.semantic_target_slot)
        if [example.semantic_target_slot for example in family] != list(
            range(RECORDS_PER_EPISODE)
        ):
            raise ValueError(f"Correct-state family {group_id} is incomplete")
        digests = [str(state_digests.get(example.row_id, "")) for example in family]
        if any(not digest for digest in digests):
            raise ValueError(f"Correct-state family {group_id} lacks tensor digests")
        identical = len(set(digests)) == 1
        identical_families += int(identical)
        layer_all_correct: dict[str, bool] = {}
        for module_name in module_names:
            matches = [
                int(_require_mapping(
                    route_predictions.get(example.row_id),
                    f"route row {example.row_id}",
                ).get(module_name, -1))
                == int(example.target_slot)
                for example in family
            ]
            passed = all(matches)
            layer_all_correct[module_name] = passed
            route_family_total += 1
            all_four_route_correct += int(passed)
        families[group_id] = {
            "row_ids": [example.row_id for example in family],
            "runtime_tensor_state_digests": digests,
            "runtime_byte_identical": identical,
            "layer_all_four_routes_correct": layer_all_correct,
        }
    family_count = len(families)
    return {
        "families": family_count,
        "runtime_byte_identical_families": identical_families,
        "runtime_byte_identical_state_fraction": identical_families / family_count,
        "family_layer_all_four_correct": all_four_route_correct,
        "family_layer_total": route_family_total,
        "family_layer_all_four_correct_fraction": (
            all_four_route_correct / route_family_total
        ),
        "families_by_id": families,
    }


def _validated_state_digests(
    examples: Sequence[NaturalMemoryExample],
    evaluation: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, str]:
    row_ids = [example.row_id for example in examples]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{condition} contains duplicate row IDs")
    raw_digests = _require_mapping(
        evaluation.get("state_digest_by_row"), f"{condition} state digests"
    )
    if set(raw_digests) != set(row_ids):
        raise ValueError(f"{condition} tensor-state digest rows differ")
    digests = {row_id: str(raw_digests[row_id]) for row_id in row_ids}
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests.values()
    ):
        raise ValueError(f"{condition} contains an invalid tensor-state digest")
    return digests


def audit_runtime_state_causality(
    examples_by_condition: Mapping[str, Sequence[NaturalMemoryExample]],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind shared writes and counterfactual writes to their runtime tensors."""

    if set(examples_by_condition) != set(POSITIVE_CONDITIONS):
        raise ValueError("State-causality examples do not cover all positive conditions")
    if set(evaluations) != set(POSITIVE_CONDITIONS):
        raise ValueError("State-causality evaluations do not cover all positive conditions")

    examples = {
        condition: tuple(examples_by_condition[condition])
        for condition in POSITIVE_CONDITIONS
    }
    digests = {
        condition: _validated_state_digests(
            examples[condition], evaluations[condition], condition=condition
        )
        for condition in POSITIVE_CONDITIONS
    }
    correct_by_row = {example.row_id: example for example in examples["correct_state"]}
    if not correct_by_row:
        raise ValueError("State-causality audit received no correct-state rows")
    correct_rows = set(correct_by_row)
    for condition in POSITIVE_CONDITIONS:
        if any(example.condition != condition for example in examples[condition]):
            raise ValueError(f"State-causality audit received incorrect {condition} rows")
        if {example.row_id for example in examples[condition]} != correct_rows:
            raise ValueError(f"{condition} row pairing differs from correct_state")

    shared_families: dict[str, dict[str, Any]] = {}
    shared_family_total = 0
    identical_family_total = 0
    for condition in SHARED_WRITE_CONDITIONS:
        grouped: dict[str, list[NaturalMemoryExample]] = defaultdict(list)
        for example in examples[condition]:
            grouped[example.memory_state_id].append(example)
        condition_families: dict[str, Any] = {}
        for state_id, raw_family in sorted(grouped.items()):
            family = sorted(
                raw_family, key=lambda example: example.semantic_target_slot
            )
            if [example.semantic_target_slot for example in family] != list(
                range(RECORDS_PER_EPISODE)
            ):
                raise ValueError(f"{condition} state family {state_id} is incomplete")
            payloads = {example.record_payload_sha256 for example in family}
            if len(payloads) != 1:
                raise ValueError(
                    f"{condition} state family {state_id} does not share one write payload"
                )
            family_digests = [digests[condition][example.row_id] for example in family]
            identical = len(set(family_digests)) == 1
            shared_family_total += 1
            identical_family_total += int(identical)
            condition_families[state_id] = {
                "row_ids": [example.row_id for example in family],
                "record_payload_sha256": next(iter(payloads)),
                "runtime_tensor_state_digests": family_digests,
                "runtime_byte_identical": identical,
            }
        shared_families[condition] = condition_families

    counterfactual_by_condition: dict[str, Any] = {}
    counterfactual_pair_total = 0
    pair_contract_total = 0
    payload_difference_total = 0
    state_difference_total = 0
    for condition in COUNTERFACTUAL_STATE_CONDITIONS:
        condition_by_row = {
            example.row_id: example for example in examples[condition]
        }
        pairs: dict[str, Any] = {}
        for row_id in sorted(correct_rows):
            correct = correct_by_row[row_id]
            counterfactual = condition_by_row[row_id]
            pair_contract = (
                correct.row_id == counterfactual.row_id
                and correct.episode_id == counterfactual.episode_id
                and correct.task == counterfactual.task
                and correct.source_split == counterfactual.source_split
                and correct.semantic_target_slot
                == counterfactual.semantic_target_slot
                and correct.query_prefix_length
                == counterfactual.query_prefix_length
                and correct.read_input_ids[: correct.query_prefix_length]
                == counterfactual.read_input_ids[
                    : counterfactual.query_prefix_length
                ]
                and correct.query_mask[: correct.query_prefix_length]
                == counterfactual.query_mask[
                    : counterfactual.query_prefix_length
                ]
            )
            payload_differs = (
                correct.record_payload_sha256
                != counterfactual.record_payload_sha256
            )
            state_differs = (
                digests["correct_state"][row_id] != digests[condition][row_id]
            )
            counterfactual_pair_total += 1
            pair_contract_total += int(pair_contract)
            payload_difference_total += int(payload_differs)
            state_difference_total += int(state_differs)
            pairs[row_id] = {
                "episode_id": correct.episode_id,
                "semantic_target_slot": correct.semantic_target_slot,
                "pair_contract_passed": pair_contract,
                "correct_record_payload_sha256": correct.record_payload_sha256,
                "counterfactual_record_payload_sha256": (
                    counterfactual.record_payload_sha256
                ),
                "write_payload_differs": payload_differs,
                "correct_runtime_tensor_state_digest": (
                    digests["correct_state"][row_id]
                ),
                "counterfactual_runtime_tensor_state_digest": (
                    digests[condition][row_id]
                ),
                "runtime_tensor_state_differs": state_differs,
            }
        counterfactual_by_condition[condition] = {
            "rows": len(pairs),
            "write_payload_difference_fraction": (
                sum(pair["write_payload_differs"] for pair in pairs.values())
                / len(pairs)
            ),
            "pair_contract_passed_fraction": (
                sum(pair["pair_contract_passed"] for pair in pairs.values())
                / len(pairs)
            ),
            "runtime_tensor_state_difference_fraction": (
                sum(
                    pair["runtime_tensor_state_differs"]
                    for pair in pairs.values()
                )
                / len(pairs)
            ),
            "pairs_by_row": pairs,
        }

    return {
        "shared_write_conditions": list(SHARED_WRITE_CONDITIONS),
        "shared_state_families": shared_family_total,
        "runtime_byte_identical_families": identical_family_total,
        "runtime_byte_identical_state_fraction": (
            identical_family_total / shared_family_total
        ),
        "shared_families_by_condition": shared_families,
        "counterfactual_conditions": list(COUNTERFACTUAL_STATE_CONDITIONS),
        "counterfactual_pairs": counterfactual_pair_total,
        "counterfactual_pair_contract_passed_fraction": (
            pair_contract_total / counterfactual_pair_total
        ),
        "write_payload_difference_fraction": (
            payload_difference_total / counterfactual_pair_total
        ),
        "runtime_tensor_state_difference_fraction": (
            state_difference_total / counterfactual_pair_total
        ),
        "counterfactual_by_condition": counterfactual_by_condition,
    }


def audit_rewrite_output_change(
    correct_examples: Sequence[NaturalMemoryExample],
    rewrite_examples: Sequence[NaturalMemoryExample],
    correct_evaluation: Mapping[str, Any],
    rewrite_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    correct_by_row = {example.row_id: example for example in correct_examples}
    rewrite_by_row = {example.row_id: example for example in rewrite_examples}
    if (
        len(correct_by_row) != len(correct_examples)
        or set(correct_by_row) != set(rewrite_by_row)
    ):
        raise ValueError("Correct/rewrite row pairing differs")
    correct_predictions = _require_mapping(
        correct_evaluation.get("answer_predictions_by_row"),
        "correct answer predictions",
    )
    rewrite_predictions = _require_mapping(
        rewrite_evaluation.get("answer_predictions_by_row"),
        "rewrite answer predictions",
    )
    if set(correct_predictions) != set(correct_by_row) or set(rewrite_predictions) != set(
        correct_by_row
    ):
        raise ValueError("Correct/rewrite prediction rows differ")
    greedy = bool(correct_evaluation.get("greedy_answer_evaluated"))
    if bool(rewrite_evaluation.get("greedy_answer_evaluated")) != greedy:
        raise ValueError("Correct/rewrite greedy policy differs")

    pairs: dict[str, dict[str, Any]] = {}
    for row_id in sorted(correct_by_row):
        correct = correct_by_row[row_id]
        rewrite = rewrite_by_row[row_id]
        if correct.condition != "correct_state" or rewrite.condition != "target_slot_rewrite":
            raise ValueError("Rewrite audit received incorrect conditions")
        correct_prediction = _require_mapping(
            correct_predictions[row_id], f"correct prediction {row_id}"
        )
        rewrite_prediction = _require_mapping(
            rewrite_predictions[row_id], f"rewrite prediction {row_id}"
        )
        changed_record_indices = [
            index
            for index, (left, right) in enumerate(
                zip(correct.write_value_jsons, rewrite.write_value_jsons, strict=True)
            )
            if left != right
        ]
        target_index = correct.write_semantic_slots.index(
            correct.semantic_target_slot
        )
        pair_contract = (
            correct.semantic_target_slot == rewrite.semantic_target_slot
            and correct.write_record_ids == rewrite.write_record_ids
            and correct.write_semantic_slots == rewrite.write_semantic_slots
            and correct.write_slots == rewrite.write_slots
            and correct.target_slot == rewrite.target_slot
            and changed_record_indices == [target_index]
            and correct.read_input_ids[: correct.query_prefix_length]
            == rewrite.read_input_ids[: rewrite.query_prefix_length]
            and correct.query_mask[: correct.query_prefix_length]
            == rewrite.query_mask[: rewrite.query_prefix_length]
        )
        expected_answers_differ = (
            correct.expected_value != rewrite.expected_value
            and correct.expected_answer_token_ids != rewrite.expected_answer_token_ids
        )
        correct_teacher = tuple(
            int(token)
            for token in correct_prediction["teacher_forced_prediction_token_ids"]
        )
        rewrite_teacher = tuple(
            int(token)
            for token in rewrite_prediction["teacher_forced_prediction_token_ids"]
        )
        teacher_change = correct_teacher != rewrite_teacher
        teacher_joint_exact_change = (
            correct_prediction.get("teacher_forced_exact") is True
            and rewrite_prediction.get("teacher_forced_exact") is True
            and teacher_change
        )
        greedy_change = None
        greedy_joint_exact_change = None
        if greedy:
            correct_greedy = tuple(
                int(token)
                for token in correct_prediction["greedy_generated_token_ids"]
            )
            rewrite_greedy = tuple(
                int(token)
                for token in rewrite_prediction["greedy_generated_token_ids"]
            )
            greedy_change = correct_greedy != rewrite_greedy
            greedy_joint_exact_change = (
                correct_prediction.get("greedy_exact") is True
                and rewrite_prediction.get("greedy_exact") is True
                and greedy_change
            )
        pairs[row_id] = {
            "episode_id": correct.episode_id,
            "task": correct.task,
            "semantic_target_slot": correct.semantic_target_slot,
            "changed_write_record_indices": changed_record_indices,
            "target_write_record_only_changed": changed_record_indices == [target_index],
            "pair_contract_passed": pair_contract,
            "correct_expected_value_json": correct.expected_value,
            "rewrite_expected_value_json": rewrite.expected_value,
            "expected_answers_differ": expected_answers_differ,
            "teacher_forced_output_changed": teacher_change,
            "teacher_forced_joint_exact_output_flip": teacher_joint_exact_change,
            "greedy_output_changed": greedy_change,
            "greedy_joint_exact_output_flip": greedy_joint_exact_change,
        }

    def fraction(field: str) -> float:
        return sum(bool(pair[field]) for pair in pairs.values()) / len(pairs)

    per_task: dict[str, dict[str, Any]] = {}
    for task in sorted({pair["task"] for pair in pairs.values()}):
        task_pairs = [pair for pair in pairs.values() if pair["task"] == task]
        per_task[task] = {
            "rows": len(task_pairs),
            "pair_contract_passed_fraction": sum(
                bool(pair["pair_contract_passed"]) for pair in task_pairs
            )
            / len(task_pairs),
            "expected_answers_differ_fraction": sum(
                bool(pair["expected_answers_differ"]) for pair in task_pairs
            )
            / len(task_pairs),
            "teacher_forced_output_change_fraction": sum(
                bool(pair["teacher_forced_output_changed"]) for pair in task_pairs
            )
            / len(task_pairs),
            "teacher_forced_joint_exact_output_flip_fraction": sum(
                bool(pair["teacher_forced_joint_exact_output_flip"])
                for pair in task_pairs
            )
            / len(task_pairs),
            "greedy_output_change_fraction": (
                sum(bool(pair["greedy_output_changed"]) for pair in task_pairs)
                / len(task_pairs)
                if greedy
                else None
            ),
            "greedy_joint_exact_output_flip_fraction": (
                sum(
                    bool(pair["greedy_joint_exact_output_flip"])
                    for pair in task_pairs
                )
                / len(task_pairs)
                if greedy
                else None
            ),
        }
    return {
        "rows": len(pairs),
        "pair_contract_passed_fraction": fraction("pair_contract_passed"),
        "expected_answers_differ_fraction": fraction("expected_answers_differ"),
        "teacher_forced_output_change_fraction": fraction(
            "teacher_forced_output_changed"
        ),
        "teacher_forced_joint_exact_output_flip_fraction": fraction(
            "teacher_forced_joint_exact_output_flip"
        ),
        "greedy_answer_evaluated": greedy,
        "greedy_output_change_fraction": (
            fraction("greedy_output_changed") if greedy else None
        ),
        "greedy_joint_exact_output_flip_fraction": (
            fraction("greedy_joint_exact_output_flip") if greedy else None
        ),
        "per_task": per_task,
        "pairs_by_row": pairs,
    }


def audit_control_equivalence(
    no_state_evaluation: Mapping[str, Any],
    pristine_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    no_state = _require_mapping(
        no_state_evaluation.get("answer_predictions_by_row"),
        "no-state predictions",
    )
    pristine = _require_mapping(
        pristine_evaluation.get("answer_predictions_by_row"),
        "pristine predictions",
    )
    if set(no_state) != set(pristine) or not no_state:
        raise ValueError("No-state/pristine prediction rows differ")
    greedy = bool(no_state_evaluation.get("greedy_answer_evaluated"))
    if bool(pristine_evaluation.get("greedy_answer_evaluated")) != greedy:
        raise ValueError("No-state/pristine greedy policy differs")
    rows: dict[str, dict[str, Any]] = {}
    for row_id in sorted(no_state):
        left = _require_mapping(no_state[row_id], f"no-state row {row_id}")
        right = _require_mapping(pristine[row_id], f"pristine row {row_id}")
        teacher_equal = left.get("teacher_forced_prediction_token_ids") == right.get(
            "teacher_forced_prediction_token_ids"
        )
        greedy_equal = (
            left.get("greedy_generated_token_ids")
            == right.get("greedy_generated_token_ids")
            if greedy
            else None
        )
        rows[row_id] = {
            "teacher_forced_outputs_equal": teacher_equal,
            "greedy_outputs_equal": greedy_equal,
        }
    return {
        "rows": len(rows),
        "teacher_forced_output_equivalence_fraction": sum(
            row["teacher_forced_outputs_equal"] for row in rows.values()
        )
        / len(rows),
        "greedy_answer_evaluated": greedy,
        "greedy_output_equivalence_fraction": (
            sum(bool(row["greedy_outputs_equal"]) for row in rows.values())
            / len(rows)
            if greedy
            else None
        ),
        "rows_by_id": rows,
    }


def audit_trainable_parameters(
    model: torch.nn.Module,
    *,
    expected_trainable_names: Sequence[str] | None = None,
    allow_zero: bool = False,
) -> dict[str, Any]:
    actual = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    allowed: list[str] = []
    for module_name, module in iter_delta_mem_modules(model):
        predicate = getattr(module, "is_trainable_parameter", None)
        for sub_name, parameter in module.named_parameters():
            if sub_name.startswith("base."):
                continue
            if predicate is None or predicate(sub_name):
                allowed.append(f"{module_name}.{sub_name}")
    allowed = sorted(set(allowed))
    expected = (
        sorted(str(name) for name in expected_trainable_names)
        if expected_trainable_names is not None
        else allowed
    )
    only_delta_mem = set(actual).issubset(set(allowed))
    expected_binding = actual == expected
    passed = only_delta_mem and expected_binding and (allow_zero or bool(actual))
    return {
        "actual_trainable_names": actual,
        "allowed_delta_mem_trainable_names": allowed,
        "expected_trainable_names": expected,
        "only_delta_mem_parameters_trainable": only_delta_mem,
        "trainable_name_binding_passed": expected_binding,
        "nonempty_trainable_set": bool(actual),
        "allow_zero": allow_zero,
        "passed": passed,
    }


def _minimum_task_metric(
    evaluation: Mapping[str, Any],
    field: str,
) -> float | None:
    per_task = evaluation.get("per_task")
    if isinstance(per_task, Mapping) and per_task:
        values = [
            float(metrics[field])
            for metrics in per_task.values()
            if isinstance(metrics, Mapping) and metrics.get(field) is not None
        ]
        if values:
            return min(values)
    value = evaluation.get(field)
    return None if value is None else float(value)


def build_gate(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    state_identity: Mapping[str, Any],
    state_causality: Mapping[str, Any],
    rewrite_audit: Mapping[str, Any],
    control_equivalence: Mapping[str, Any],
    profile_eligibility: Mapping[str, Any],
    trainable_audit: Mapping[str, Any],
    immutability_passed: bool,
    training: Mapping[str, Any] | None = None,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """Return a deliberately conjunctive causal acceptance gate."""

    if set(evaluations) != set(CONDITIONS):
        raise ValueError("Gate evaluations do not cover all seven conditions")
    checks: dict[str, bool] = {}
    formal_profile = profile_eligibility.get("profile") in FORMAL_PROFILES

    for condition in POSITIVE_CONDITIONS:
        evaluation = evaluations[condition]
        answer_metric = _minimum_task_metric(
            evaluation, "teacher_forced_structured_json_exact_accuracy"
        )
        route_metric = _minimum_task_metric(evaluation, "semantic_route_accuracy")
        checks[f"{condition}.structured_json_exact_min"] = (
            answer_metric is not None and answer_metric >= thresholds.answer_exact_min
        )
        checks[f"{condition}.semantic_route_min"] = (
            route_metric is not None and route_metric >= thresholds.route_accuracy_min
        )
        checks[f"{condition}.full_occupancy"] = (
            evaluation.get("full_occupancy_fraction") == 1.0
        )
        checks[f"{condition}.forced_write_route"] = (
            evaluation.get("forced_write_route_accuracy") == 1.0
        )
        if evaluation.get("greedy_answer_evaluated"):
            greedy_metric = _minimum_task_metric(
                evaluation, "greedy_structured_json_exact_accuracy"
            )
            checks[f"{condition}.greedy_structured_json_exact_min"] = (
                greedy_metric is not None
                and greedy_metric >= thresholds.answer_exact_min
            )

    for condition in CONTROL_CONDITIONS:
        evaluation = evaluations[condition]
        checks[f"{condition}.zero_writes"] = (
            evaluation.get("full_occupancy_total") == 0
            and evaluation.get("forced_write_route_total") == 0
        )
        checks[f"{condition}.routes_absent"] = (
            evaluation.get("semantic_route_total") == 0
            and evaluation.get("route_absent_fraction") == 1.0
        )
        state_absence_by_row = evaluation.get("runtime_state_absence_by_row")
        checks[f"{condition}.runtime_state_absent"] = (
            evaluation.get("runtime_state_absence_rows")
            == evaluation.get("rows")
            and evaluation.get("runtime_state_absence_fraction") == 1.0
            and isinstance(state_absence_by_row, Mapping)
            and len(state_absence_by_row) == evaluation.get("rows")
            and not bool(evaluation.get("state_digest_by_row"))
        )
    checks["formal.greedy_answer_evaluation"] = (
        not formal_profile
        or all(
            evaluation.get("greedy_answer_evaluated") is True
            for evaluation in evaluations.values()
        )
    )
    checks["pristine_frozen_base.heads_disabled"] = (
        evaluations["pristine_frozen_base"].get("delta_heads_disabled") is True
    )
    checks["pristine_frozen_base.adapter_excluded"] = (
        evaluations["pristine_frozen_base"].get(
            "pristine_base_adapter_excluded"
        )
        is True
        and evaluations["pristine_frozen_base"].get("adapter_attached") is False
        and evaluations["pristine_frozen_base"].get(
            "attached_delta_mem_module_count"
        )
        == 0
        and evaluations["pristine_frozen_base"].get("trainable_parameter_count")
        == 0
    )

    checks["correct_state.runtime_byte_identical"] = (
        state_identity.get("runtime_byte_identical_state_fraction") == 1.0
    )
    checks["correct_state.all_four_routes"] = (
        state_identity.get("family_layer_all_four_correct_fraction") == 1.0
    )
    checks["positive_states.shared_write_identity"] = (
        state_causality.get("runtime_byte_identical_state_fraction") == 1.0
    )
    checks["counterfactual_states.write_payloads_differ"] = (
        state_causality.get("write_payload_difference_fraction") == 1.0
    )
    checks["counterfactual_states.pair_contract"] = (
        state_causality.get("counterfactual_pair_contract_passed_fraction")
        == 1.0
    )
    checks["counterfactual_states.runtime_tensors_differ"] = (
        state_causality.get("runtime_tensor_state_difference_fraction") == 1.0
    )
    checks["rewrite.expected_answers_differ"] = (
        rewrite_audit.get("expected_answers_differ_fraction") == 1.0
    )
    checks["rewrite.pair_contract"] = (
        rewrite_audit.get("pair_contract_passed_fraction") == 1.0
    )
    checks["rewrite.teacher_output_change"] = (
        rewrite_audit.get("teacher_forced_output_change_fraction", 0.0)
        >= thresholds.rewrite_output_change_min
    )
    checks["rewrite.teacher_joint_exact_flip"] = (
        rewrite_audit.get("teacher_forced_joint_exact_output_flip_fraction", 0.0)
        >= thresholds.rewrite_output_change_min
    )
    if rewrite_audit.get("greedy_answer_evaluated"):
        checks["rewrite.greedy_output_change"] = (
            rewrite_audit.get("greedy_output_change_fraction", 0.0)
            >= thresholds.rewrite_output_change_min
        )
        checks["rewrite.greedy_joint_exact_flip"] = (
            rewrite_audit.get("greedy_joint_exact_output_flip_fraction", 0.0)
            >= thresholds.rewrite_output_change_min
        )
    checks["no_state.pristine_equivalence"] = (
        control_equivalence.get("teacher_forced_output_equivalence_fraction") == 1.0
        and (
            not control_equivalence.get("greedy_answer_evaluated")
            or control_equivalence.get("greedy_output_equivalence_fraction") == 1.0
        )
    )
    checks["profile.eligibility"] = profile_eligibility.get("passed") is True
    checks["model.only_delta_mem_trainable"] = (
        trainable_audit.get("only_delta_mem_parameters_trainable") is True
        and trainable_audit.get("passed") is True
    )
    checks["source_and_model.immutable"] = bool(immutability_passed)
    if training is not None and training.get("optimizer_skipped") is not True:
        checks["training.adapter_changed"] = training.get("adapter_changed") is True
        checks["training.router_gradient"] = training.get(
            "router_gradient_audit", {}
        ).get("all_modules_finite_nonzero") is True

    return {
        "schema": "rwkv_ms_natural_memory_gate_acceptance.v1",
        "thresholds": {
            "answer_exact_min": thresholds.answer_exact_min,
            "route_accuracy_min": thresholds.route_accuracy_min,
            "rewrite_output_change_min": thresholds.rewrite_output_change_min,
        },
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "passed": all(checks.values()),
    }


def _json_payload_hash(path: Path) -> str:
    return source.sha256_text(
        source.canonical_json(_read_json_file(path, f"JSON artifact {path.name}"))
    )


def snapshot_directory_files(path: Path) -> dict[str, dict[str, Any]]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ValueError(f"Artifact directory must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Artifact directory is invalid: {resolved}")
    result: dict[str, dict[str, Any]] = {}
    for artifact in sorted(resolved.iterdir(), key=lambda value: value.name):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Artifact directory contains a non-file entry: {artifact}")
        result[artifact.name] = {
            "bytes": artifact.stat().st_size,
            "sha256": source.sha256_file(artifact),
        }
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source.sha256_file(path)


def validate_sealed_lock_chain(
    sealed_manifest: Mapping[str, Any],
    development_run_dir: Path,
    *,
    adapter_path: Path,
) -> dict[str, Any]:
    """Validate the sealed lock against the immutable development receipt."""

    sealed_lock = _require_mapping(sealed_manifest.get("sealed_lock"), "sealed_lock")
    lock = _require_mapping(sealed_lock.get("receipt"), "sealed lock receipt")
    requested_run_dir = development_run_dir.expanduser()
    if requested_run_dir.is_symlink():
        raise ValueError("Development run directory must not be a symbolic link")
    run_dir = requested_run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError(f"Development run directory is invalid: {run_dir}")
    protocol_path = run_dir / "protocol.json"
    training_path = run_dir / "training_configuration.json"
    receipt_path = run_dir / "run_receipt.json"
    for path in (protocol_path, training_path, receipt_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Sealed lock chain artifact is missing: {path}")
    development_protocol = _read_json_file(protocol_path, "development protocol")
    development_training = _read_json_file(
        training_path, "development training configuration"
    )
    protocol_hash = source.sha256_text(source.canonical_json(development_protocol))
    training_hash = source.sha256_text(source.canonical_json(development_training))
    receipt = _read_json_file(receipt_path, "development run receipt")
    unsigned_receipt = dict(receipt)
    recorded_receipt_hash = unsigned_receipt.pop("run_receipt_sha256", None)
    if recorded_receipt_hash != _signed_payload(
        unsigned_receipt, "run_receipt_sha256"
    )["run_receipt_sha256"]:
        raise ValueError("Development run receipt signature is invalid")
    if receipt.get("profile") != "development":
        raise ValueError("Sealed lock chain does not point to a development run")
    manifest_payload_hash = receipt.get("source_manifest_payload_sha256")
    required = {
        "development_manifest_payload_sha256": manifest_payload_hash,
        "runner_protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_hash,
    }
    for name, value in required.items():
        if lock.get(name) != value:
            raise ValueError(f"Sealed lock chain mismatch: {name}")
    if (
        receipt.get("protocol_sha256") != protocol_hash
        or receipt.get("training_configuration_sha256") != training_hash
    ):
        raise ValueError("Development receipt does not bind its protocol files")
    if receipt.get("gate_passed") is not True:
        raise ValueError("Sealed lock chain points to a failing development run")
    if lock.get("development_gate_passed") is not True:
        raise ValueError("Sealed lock does not assert a passing development gate")
    if lock.get("development_run_receipt_sha256") != recorded_receipt_hash:
        raise ValueError("Sealed lock does not bind the signed development receipt")
    requested_adapter = adapter_path.expanduser()
    if requested_adapter.is_symlink():
        raise ValueError("Sealed adapter path must not be a symbolic link")
    adapter = requested_adapter.resolve(strict=True)
    if not adapter.is_dir():
        raise ValueError(f"Sealed adapter path is invalid: {adapter}")
    adapter_files = snapshot_directory_files(adapter)
    if not adapter_files:
        raise ValueError("Sealed adapter directory is empty")
    if receipt.get("adapter_files") != adapter_files:
        raise ValueError("Sealed adapter artifacts differ from the development receipt")
    adapter_files_sha256 = _sha256_json(adapter_files)
    if lock.get("adapter_files_sha256") != adapter_files_sha256:
        raise ValueError("Sealed lock does not bind the exact adapter artifacts")
    return {
        "development_run_dir": str(run_dir),
        "protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_hash,
        "development_manifest_payload_sha256": manifest_payload_hash,
        "adapter_path": str(adapter),
        "adapter_files": adapter_files,
        "adapter_files_sha256": adapter_files_sha256,
        "development_run_receipt_sha256": recorded_receipt_hash,
        "development_protocol": development_protocol,
        "development_training_configuration": development_training,
        "passed": True,
    }


def _parse_layers(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        layers = tuple(int(layer) for layer in value)
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("Target layers must be a nonempty unique list of nonnegative integers")
    return layers


def _parse_training_conditions(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        conditions = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        conditions = tuple(str(condition) for condition in value)
    if not conditions:
        raise ValueError("Training conditions must be a nonempty list")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Training conditions must be unique")
    invalid = sorted(set(conditions) - set(POSITIVE_CONDITIONS))
    if invalid:
        raise ValueError(
            "Training conditions must be positive memory conditions: "
            + ", ".join(invalid)
        )
    return conditions


def run_experiment(
    *,
    source_manifest: Path,
    output_dir: Path,
    profile: str = "development",
    model_path: Path | None = None,
    adapter_path: Path | None = None,
    development_run_dir: Path | None = None,
    seed: int = 42,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    epochs: int = 8,
    max_steps: int | None = 768,
    batch_size: int = 4,
    eval_batch_size: int = 8,
    learning_rate: float = 2e-4,
    answer_weight: float = 1.0,
    route_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    device_name: str = "cuda",
    dtype_name: str = "bfloat16",
    attn_implementation: str = "sdpa",
    target_layers: Sequence[int] = DEFAULT_TARGET_LAYERS,
    training_conditions: Sequence[str] = DEFAULT_TRAINING_CONDITIONS,
    rank: int = 4,
    key_dim: int = 32,
    temperature: float = 16.0,
    greedy: bool = True,
    answer_exact_min: float = 0.80,
    route_accuracy_min: float = 0.95,
    rewrite_output_change_min: float = 0.80,
) -> dict[str, Any]:
    """Run a train/development screen or a sealed, optimizer-free evaluation."""

    configure_hf_mirror()
    if profile == "sealed_validation" and adapter_path is None:
        raise ValueError("Sealed validation requires a frozen development adapter")
    if profile != "sealed_validation" and adapter_path is not None:
        raise ValueError("Training profiles cannot inject a pre-trained adapter")
    if profile == "sealed_validation" and development_run_dir is None:
        raise ValueError("Sealed validation requires the development run receipt directory")
    if profile in FORMAL_PROFILES and not greedy:
        raise ValueError(
            "Formal development and sealed-validation runs require greedy evaluation"
        )
    if profile in {"development", "sealed_validation"} and (
        train_limit is not None or eval_limit is not None
    ):
        raise ValueError(
            "Formal development and sealed-validation runs require complete splits"
        )
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied")
    if epochs <= 0 or batch_size <= 0 or eval_batch_size <= 0:
        raise ValueError("Training and evaluation sizes must be positive")
    if eval_batch_size < RECORDS_PER_EPISODE:
        raise ValueError(
            "Evaluation batch size must hold a complete four-query state family"
        )
    thresholds = GateThresholds(
        answer_exact_min=answer_exact_min,
        route_accuracy_min=route_accuracy_min,
        rewrite_output_change_min=rewrite_output_change_min,
    )
    if not all(
        0.0 <= value <= 1.0
        for value in (
            answer_exact_min,
            route_accuracy_min,
            rewrite_output_change_min,
        )
    ):
        raise ValueError("Gate thresholds must be fractions in [0, 1]")

    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError(f"Natural run output must not be a symbolic link: {requested_output}")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Natural run output must be fresh: {resolved_output}")
    bundle = load_profile_bundle(source_manifest, profile=profile)
    model_root, model_artifact_paths = resolve_model_artifacts(
        bundle.model_binding,
        model_path=model_path,
    )
    source_before = snapshot_files(bundle.source_paths)
    model_before = snapshot_files(model_artifact_paths)
    sealed_chain: Mapping[str, Any] | None = None
    if profile == "sealed_validation":
        sealed_chain = validate_sealed_lock_chain(
            bundle.sealed_manifest or bundle.development_manifest,
            development_run_dir or Path(),
            adapter_path=adapter_path or Path(),
        )

    device = torch.device(device_name)
    dtype = _dtype(dtype_name)
    layers = _parse_layers(target_layers)
    selected_training_conditions = _parse_training_conditions(training_conditions)
    delta_config = build_delta_config(
        target_layers=layers,
        rank=rank,
        key_dim=key_dim,
        temperature=temperature,
    )
    model_source = {"model": {"path": str(model_root)}}
    model, tokenizer, replaced_layers, trainable_names, checkpointed_mlps = (
        _load_model_and_tokenizer(
            model_source,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            delta_config=delta_config,
        )
    )
    if profile == "sealed_validation":
        loaded_delta_config = load_delta_mem_adapter(model, adapter_path or Path())
        if loaded_delta_config.to_dict() != delta_config.to_dict():
            raise ValueError(
                "Sealed adapter configuration differs from the frozen requested configuration"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        trainable_audit = audit_trainable_parameters(
            model,
            expected_trainable_names=(),
            allow_zero=True,
        )
    else:
        trainable_audit = audit_trainable_parameters(
            model,
            expected_trainable_names=trainable_names,
        )
    if not trainable_audit["passed"]:
        raise ValueError("Only Delta-Mem parameters may be trainable")

    pre_adapter_hash = _state_dict_sha256(snapshot_delta_mem_weights(model))
    train_episodes = select_complete_episodes(bundle.train_episodes, train_limit)
    eval_episodes = select_complete_episodes(bundle.evaluation_episodes, eval_limit)
    if profile != "sealed_validation" and not train_episodes:
        raise ValueError("Training profile selected no training episodes")
    if not eval_episodes:
        raise ValueError("Selected profile has no evaluation episodes")

    resolved_output.mkdir(parents=True, exist_ok=False)
    progress_path = resolved_output / "training_progress.jsonl"
    if profile == "sealed_validation":
        training: dict[str, Any] = {
            "optimizer_skipped": True,
            "steps": 0,
            "adapter_changed": None,
            "router_gradient_audit": {"all_modules_finite_nonzero": True},
        }
        adapter_files = dict((sealed_chain or {})["adapter_files"])
    else:
        training_examples = build_training_examples(
            train_episodes,
            tokenizer,
            selected_training_conditions,
        )
        training = dict(
            train_model(
                model,
                training_examples,
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
                training_conditions=selected_training_conditions,
            )
        )
        post_train_audit = audit_trainable_parameters(
            model,
            expected_trainable_names=trainable_names,
        )
        if not post_train_audit["passed"]:
            raise ValueError("Training changed the trainable-parameter boundary")
        post_adapter_hash = _state_dict_sha256(snapshot_delta_mem_weights(model))
        training["adapter_changed"] = post_adapter_hash != pre_adapter_hash
        training["adapter_state_sha256_before"] = pre_adapter_hash
        training["adapter_state_sha256_after"] = post_adapter_hash
        if not training["adapter_changed"]:
            raise RuntimeError("Training produced no Delta-Mem parameter update")
        save_delta_mem_adapter(model, resolved_output / "adapter", delta_config)
        adapter_files = snapshot_directory_files(resolved_output / "adapter")

    evaluation_examples: dict[str, list[NaturalMemoryExample]] = {
        condition: build_condition_examples(eval_episodes, tokenizer, condition)
        for condition in CONDITIONS
    }
    evaluations: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        if condition == "pristine_frozen_base":
            continue
        evaluations[condition] = evaluate_condition(
            model,
            tokenizer,
            evaluation_examples[condition],
            condition=condition,
            batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            dtype=dtype,
            greedy=greedy,
        )

    del model
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    pristine_model = load_pristine_base_model(
        model_root,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    try:
        evaluations["pristine_frozen_base"] = evaluate_condition(
            pristine_model,
            tokenizer,
            evaluation_examples["pristine_frozen_base"],
            condition="pristine_frozen_base",
            batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
            dtype=dtype,
            greedy=greedy,
        )
    finally:
        del pristine_model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    correct_identity = audit_correct_state_identity(
        evaluation_examples["correct_state"], evaluations["correct_state"]
    )
    state_causality = audit_runtime_state_causality(
        {
            condition: evaluation_examples[condition]
            for condition in POSITIVE_CONDITIONS
        },
        {condition: evaluations[condition] for condition in POSITIVE_CONDITIONS},
    )
    rewrite_audit = audit_rewrite_output_change(
        evaluation_examples["correct_state"],
        evaluation_examples["target_slot_rewrite"],
        evaluations["correct_state"],
        evaluations["target_slot_rewrite"],
    )
    control_equivalence = audit_control_equivalence(
        evaluations["no_state"], evaluations["pristine_frozen_base"]
    )

    source_after = assert_snapshot_unchanged(
        source_before,
        description="Natural source",
    )
    model_after = assert_snapshot_unchanged(
        model_before,
        description="Local model",
    )
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "runner_schema": RUN_SCHEMA,
        "source_schema": source.SCHEMA,
        "profile": profile,
        "conditions": list(CONDITIONS),
        "training_conditions": list(selected_training_conditions),
        "opened_splits": list(bundle.eligibility["opened_splits"]),
        "hf_endpoint": HF_MIRROR_ENDPOINT,
        "model_binding_sha256": bundle.model_binding["binding_sha256"],
        "target_layers": list(layers),
        "rank": rank,
        "key_dim": key_dim,
        "temperature": temperature,
        "eval_batch_size": eval_batch_size,
        "greedy_answer_evaluation": greedy,
        "dtype": dtype_name,
        "attn_implementation": attn_implementation,
        "write_read_cache_policy": "every model invocation passes use_cache=False; writes and reads are separate",
        "query_encoding_policy": "Gemma chat-template address-only prefix and canonical JSON label tokenized as one full string with offset-derived disjoint masks and boundary-crossing rejection",
        "answer_logit_policy": ANSWER_LOGIT_POLICY,
        "shared_state_batching_policy": SHARED_STATE_BATCHING_POLICY,
        "sealed_chain": sealed_chain,
        "thresholds": {
            "answer_exact_min": answer_exact_min,
            "route_accuracy_min": route_accuracy_min,
            "rewrite_output_change_min": rewrite_output_change_min,
        },
    }
    if sealed_chain is not None:
        locked_protocol = _require_mapping(
            sealed_chain.get("development_protocol"), "locked development protocol"
        )
        frozen_fields = (
            "conditions",
            "training_conditions",
            "hf_endpoint",
            "model_binding_sha256",
            "target_layers",
            "rank",
            "key_dim",
            "temperature",
            "eval_batch_size",
            "greedy_answer_evaluation",
            "dtype",
            "attn_implementation",
            "write_read_cache_policy",
            "query_encoding_policy",
            "answer_logit_policy",
            "shared_state_batching_policy",
            "thresholds",
        )
        mismatches = [
            field
            for field in frozen_fields
            if locked_protocol.get(field) != protocol.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Sealed protocol differs from frozen development fields: "
                + ", ".join(mismatches)
            )
    training_configuration = {
        "schema": "rwkv_ms_natural_memory_gate_training_configuration.v1",
        "profile": profile,
        "seed": seed,
        "training_conditions": list(selected_training_conditions),
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "greedy_answer_evaluation": greedy,
        "learning_rate": learning_rate,
        "answer_weight": answer_weight,
        "route_weight": route_weight,
        "max_grad_norm": max_grad_norm,
        "device": device_name,
        "dtype": dtype_name,
        "thresholds": dict(protocol["thresholds"]),
    }
    gate = build_gate(
        evaluations,
        state_identity=correct_identity,
        state_causality=state_causality,
        rewrite_audit=rewrite_audit,
        control_equivalence=control_equivalence,
        profile_eligibility=bundle.eligibility,
        trainable_audit=trainable_audit,
        immutability_passed=source_before == source_after and model_before == model_after,
        training=training,
        thresholds=thresholds,
    )
    protocol_hash = _sha256_json(protocol)
    training_configuration_hash = _sha256_json(training_configuration)
    evaluation_payload = {
        "schema": EVALUATION_SCHEMA,
        "profile": profile,
        "training": training,
        "conditions": evaluations,
        "correct_state_identity": correct_identity,
        "runtime_state_causality": state_causality,
        "rewrite_audit": rewrite_audit,
        "control_equivalence": control_equivalence,
        "trainable_audit": trainable_audit,
        "gate": gate,
    }
    _write_json(resolved_output / "protocol.json", protocol)
    _write_json(
        resolved_output / "training_configuration.json",
        training_configuration,
    )
    _write_json(resolved_output / "evaluation.json", evaluation_payload)
    adapter_files_sha256 = _sha256_json(adapter_files)
    receipt = {
        "schema": RUN_SCHEMA,
        "profile": profile,
        "gate_passed": gate["passed"],
        "source_manifest_payload_sha256": (
            (bundle.development_manifest or bundle.sealed_manifest or {})
            .get("manifest_receipt", {})
            .get("payload_sha256")
        ),
        "source_files_before": source_before,
        "source_files_after": source_after,
        "model_files_before": model_before,
        "model_files_after": model_after,
        "protocol_sha256": protocol_hash,
        "training_configuration_sha256": training_configuration_hash,
        "evaluation_sha256": _sha256_json(evaluation_payload),
        "replaced_layers": list(replaced_layers),
        "trainable_names": list(trainable_names),
        "checkpointed_frozen_mlps": list(checkpointed_mlps),
        "adapter_files": adapter_files,
        "adapter_files_sha256": adapter_files_sha256,
        "gate": gate,
    }
    receipt = _signed_payload(receipt, "run_receipt_sha256")
    _write_json(resolved_output / "run_receipt.json", receipt)
    if profile == "development" and gate["passed"]:
        sealed_lock_receipt = {
            "schema": source.SEALED_LOCK_SCHEMA,
            "configuration_frozen": True,
            "development_gate_passed": True,
            "benchmark_contract_sha256": bundle.development_manifest[
                "benchmark_contract_sha256"
            ],
            "development_manifest_payload_sha256": bundle.development_manifest[
                "manifest_receipt"
            ]["payload_sha256"],
            "runner_protocol_sha256": protocol_hash,
            "training_configuration_sha256": training_configuration_hash,
            "development_run_receipt_sha256": receipt["run_receipt_sha256"],
            "adapter_files_sha256": adapter_files_sha256,
        }
        _write_json(resolved_output / "sealed_lock_receipt.json", sealed_lock_receipt)
    if not gate["passed"]:
        raise RuntimeError(
            "Natural memory gate failed: " + ", ".join(gate["failed_checks"])
        )
    return {
        "output_dir": str(resolved_output),
        "receipt": receipt,
        "evaluation": evaluation_payload,
        "gate": gate,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="development")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--target-layers", default=",".join(map(str, DEFAULT_TARGET_LAYERS)))
    parser.add_argument(
        "--training-conditions",
        default=",".join(DEFAULT_TRAINING_CONDITIONS),
        help=(
            "Comma-separated positive memory conditions; formal default is "
            "correct_state"
        ),
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=16.0)
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    parser.add_argument("--answer-exact-min", type=float, default=0.80)
    parser.add_argument("--route-accuracy-min", type=float, default=0.95)
    parser.add_argument("--rewrite-output-change-min", type=float, default=0.80)
    parser.set_defaults(greedy=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_experiment(
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        profile=args.profile,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        development_run_dir=args.development_run_dir,
        seed=args.seed,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        answer_weight=args.answer_weight,
        route_weight=args.route_weight,
        max_grad_norm=args.max_grad_norm,
        device_name=args.device,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        target_layers=_parse_layers(args.target_layers),
        training_conditions=_parse_training_conditions(args.training_conditions),
        rank=args.rank,
        key_dim=args.key_dim,
        temperature=args.temperature,
        greedy=args.greedy,
        answer_exact_min=args.answer_exact_min,
        route_accuracy_min=args.route_accuracy_min,
        rewrite_output_change_min=args.rewrite_output_change_min,
    )
    print(json.dumps({"output_dir": result["output_dir"], "gate": result["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
