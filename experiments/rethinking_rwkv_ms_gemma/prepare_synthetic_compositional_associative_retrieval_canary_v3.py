#!/usr/bin/env python3
"""Build the held-out compositional associative-retrieval canary v3.

The read dialogue contains only the query and supervised answer.  Memory records
are encoded independently and carry exact key/value token masks, so a consumer
cannot accidentally recover the old contextual whole-mapping write path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer


SOURCE_SCHEMA = "rwkv_ms_synthetic_compositional_associative_source.v3"
ROW_SCHEMA = "rwkv_ms_synthetic_compositional_associative_row.v3"
ROW_MANIFEST_SCHEMA = "rwkv_ms_synthetic_compositional_associative_row_manifest.v3"
SPLIT_MANIFEST_SCHEMA = "rwkv_ms_synthetic_compositional_associative_split.v3"
TASK_NAME = "synthetic-compositional-associative-retrieval"
SOURCE_PURPOSE = "heldout_semantic_addressing_canary_v3"
GENERATOR_VERSION = "isolated-record-factorized-pairs-v2"

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MODEL_PATH = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "local_artifacts/synthetic_compositional_associative_canary_v3"
)
MODEL_ARTIFACT_NAMES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

GENERATION_SEED = 2026080603
NONCE_COUNT = 24
RECORDS_PER_EPISODE = 4
RWKV_MS_NUM_STATES = 4
TRAIN_OFFSETS = tuple(range(16))
HELDOUT_OFFSETS = tuple(range(16, NONCE_COUNT))
TRAIN_TEMPLATES_PER_OFFSET_PAIR = 6
HELDOUT_TEMPLATES_PER_OFFSET_PAIR = 6
PARTITION_ORDER = ("train", "heldout")

SYSTEM_PROMPT = (
    "You are a deterministic associative lookup engine. The four memory records "
    "are supplied through external memory. Resolve the query key and reply with "
    'exactly {"value":"NONCE"} and no other text.'
)
RECORD_TEMPLATE = "MEMORY KEY <{key}>. MEMORY VALUE <{value}>."
QUERY_TEMPLATE = "QUERY KEY <{key}>. Return its stored value."
RESPONSE_TEMPLATE = '{{"value":"{value}"}}'

SOURCE_CONTRACT = {
    "synthetic_data_only": True,
    "external_dataset_access": False,
    "hard32_accessed": False,
    "protected_evaluation_included": False,
    "partition_names": list(PARTITION_ORDER),
    "heldout_is_synthetic": True,
    "records_per_episode": RECORDS_PER_EPISODE,
    "slots_per_episode": RWKV_MS_NUM_STATES,
    "read_messages_contain_memory_records": False,
    "record_encoding_scope": "one_record_content_only_per_encoder_call",
    "requires_record_local_collator": True,
    "compatible_with_flat_episode_write_collator": False,
    "query_visible_during_read": True,
    "query_excluded_from_record_writes": True,
    "key_value_write_masks_explicit": True,
    "answer_target_mask_and_labels_explicit": True,
    "heldout_record_order_patterns_unseen_in_train": True,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _derived_rng(*parts: object) -> random.Random:
    payload = canonical_json_bytes([GENERATION_SEED, *parts])
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return random.Random(seed)


def _nonce_labels(prefix: str, namespace: str) -> tuple[str, ...]:
    rng = _derived_rng("nonce-labels", namespace)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    labels: list[str] = []
    while len(labels) < NONCE_COUNT:
        candidate = prefix + "".join(rng.choice(alphabet) for _ in range(8))
        if candidate not in labels:
            labels.append(candidate)
    return tuple(labels)


KEY_LABELS = _nonce_labels("K", "keys")
VALUE_LABELS = _nonce_labels("V", "values")


def _permutation(namespace: str) -> tuple[int, ...]:
    values = list(range(NONCE_COUNT))
    _derived_rng("factorization", namespace).shuffle(values)
    return tuple(values)


KEY_CYCLE = _permutation("keys")
VALUE_CYCLE = _permutation("values")
KEY_CYCLE_POSITION = {key_index: position for position, key_index in enumerate(KEY_CYCLE)}


def _record_order_pattern_pools() -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    patterns = list(itertools.permutations(range(RECORDS_PER_EPISODE)))
    _derived_rng("record-order-patterns").shuffle(patterns)
    return tuple(patterns[:18]), tuple(patterns[18:])


TRAIN_RECORD_ORDER_PATTERNS, HELDOUT_RECORD_ORDER_PATTERNS = (
    _record_order_pattern_pools()
)


def configure_hf_mirror() -> str:
    current = os.environ.get("HF_ENDPOINT")
    if current is not None and current.rstrip("/") != HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be {HF_MIRROR_ENDPOINT}, not {current!r}"
        )
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    return HF_MIRROR_ENDPOINT


def bind_model_artifacts(model_path: Path) -> dict[str, Any]:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"Model path is not a regular directory: {resolved}")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in MODEL_ARTIFACT_NAMES:
        path = resolved / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required model artifact is invalid: {path}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "path": str(resolved),
        "artifacts": artifacts,
        "identity_sha256": canonical_sha256(artifacts),
    }


def load_local_tokenizer(model_path: Path = DEFAULT_MODEL_PATH):
    configure_hf_mirror()
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"Tokenizer model path is invalid: {resolved}")
    return AutoTokenizer.from_pretrained(
        resolved,
        local_files_only=True,
        trust_remote_code=False,
    )


def _mapped_value_index(key_index: int, offset: int) -> int:
    if key_index not in KEY_CYCLE_POSITION:
        raise ValueError(f"Unknown key index: {key_index}")
    if offset < 0 or offset >= NONCE_COUNT:
        raise ValueError(f"Mapping offset is out of range: {offset}")
    position = (KEY_CYCLE_POSITION[key_index] + offset) % NONCE_COUNT
    return VALUE_CYCLE[position]


def _offset_pairs(offsets: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if len(offsets) % 2:
        raise ValueError("Donor offsets must form reciprocal pairs")
    return tuple(
        (int(offsets[index]), int(offsets[index + 1]))
        for index in range(0, len(offsets), 2)
    )


def canary_spec() -> dict[str, Any]:
    return {
        "schema": SOURCE_SCHEMA,
        "task": TASK_NAME,
        "purpose": SOURCE_PURPOSE,
        "generator_version": GENERATOR_VERSION,
        "generation_seed": GENERATION_SEED,
        "key_labels": list(KEY_LABELS),
        "value_labels": list(VALUE_LABELS),
        "key_cycle": list(KEY_CYCLE),
        "value_cycle": list(VALUE_CYCLE),
        "mapping_factorization": (
            "value_cycle[(key_cycle_position[key] + mapping_offset) % 24]"
        ),
        "partitions": {
            "train": {
                "mapping_offsets": list(TRAIN_OFFSETS),
                "record_order_patterns": [
                    list(pattern) for pattern in TRAIN_RECORD_ORDER_PATTERNS
                ],
                "offset_pairs": [list(pair) for pair in _offset_pairs(TRAIN_OFFSETS)],
                "templates_per_offset_pair": TRAIN_TEMPLATES_PER_OFFSET_PAIR,
                "mapping_states_per_template": 2,
                "query_variants_per_mapping_state": RECORDS_PER_EPISODE,
                "rows_per_template": 2 * RECORDS_PER_EPISODE,
            },
            "heldout": {
                "mapping_offsets": list(HELDOUT_OFFSETS),
                "record_order_patterns": [
                    list(pattern) for pattern in HELDOUT_RECORD_ORDER_PATTERNS
                ],
                "offset_pairs": [list(pair) for pair in _offset_pairs(HELDOUT_OFFSETS)],
                "templates_per_offset_pair": HELDOUT_TEMPLATES_PER_OFFSET_PAIR,
                "mapping_states_per_template": 2,
                "query_variants_per_mapping_state": RECORDS_PER_EPISODE,
                "rows_per_template": 2 * RECORDS_PER_EPISODE,
            },
        },
        "episode": {
            "records": RECORDS_PER_EPISODE,
            "slots": RWKV_MS_NUM_STATES,
            "record_order": "deterministically_seeded_shuffle",
            "query_target_slots": list(range(RECORDS_PER_EPISODE)),
            "query_counterfactuals_per_byte_identical_state": RECORDS_PER_EPISODE,
            "record_encoding": "independent_content_only_with_special_tokens",
            "read_context": "system_plus_query_plus_teacher_forced_answer",
        },
        "split_guarantees": {
            "keys_seen_in_both_splits": True,
            "values_seen_in_both_splits": True,
            "key_value_pair_intersection": 0,
            "mapping_intersection": 0,
            "mapping_query_intersection": 0,
            "record_order_pattern_intersection": 0,
        },
        "interventions": {
            "reciprocal_same_query_donor": True,
            "donor_differs_at_every_value": True,
            "within_row_value_slot_derangement": True,
            "no_write_metadata": True,
        },
        "acceptance_gate": {
            "training_seeds": [42, 43, 44],
            "required_seed_passes": 3,
            "heldout_answer_accuracy_min": 0.95,
            "heldout_semantic_route_accuracy_min": 0.95,
            "heldout_query_counterfactual_route_accuracy_min": 0.95,
            "heldout_donor_expected_answer_accuracy_min": 0.95,
            "heldout_value_swap_expected_answer_accuracy_min": 0.95,
            "heldout_no_write_answer_accuracy_max": 0.35,
            "heldout_no_write_route_absent_fraction_min": 1.0,
        },
    }


CANARY_SPEC_SHA256 = canonical_sha256(canary_spec())


def _templates(
    split: str,
    offset_pair_index: int,
    count: int,
) -> list[dict[str, Any]]:
    if count < NONCE_COUNT // RECORDS_PER_EPISODE:
        raise ValueError("Every offset pair must cover every nonce key")
    rng = _derived_rng("templates", split, offset_pair_index)
    coverage_order = list(range(NONCE_COUNT))
    rng.shuffle(coverage_order)
    templates: list[dict[str, Any]] = []
    used_key_sets: set[tuple[int, ...]] = set()
    coverage_templates = NONCE_COUNT // RECORDS_PER_EPISODE
    order_patterns = (
        TRAIN_RECORD_ORDER_PATTERNS
        if split == "train"
        else HELDOUT_RECORD_ORDER_PATTERNS
    )
    for local_index in range(count):
        while True:
            if local_index < coverage_templates:
                start = local_index * RECORDS_PER_EPISODE
                key_indices = coverage_order[start : start + RECORDS_PER_EPISODE]
            else:
                key_indices = rng.sample(range(NONCE_COUNT), RECORDS_PER_EPISODE)
            identity = tuple(sorted(key_indices))
            if identity not in used_key_sets:
                used_key_sets.add(identity)
                break
        sorted_keys = sorted(key_indices)
        pattern_index = (offset_pair_index * count + local_index) % len(
            order_patterns
        )
        record_order_permutation = order_patterns[pattern_index]
        key_indices = [sorted_keys[index] for index in record_order_permutation]
        templates.append(
            {
                "local_index": local_index,
                "key_indices": list(key_indices),
                "record_order_permutation": list(record_order_permutation),
            }
        )
    return templates


def _span(text: str, value: str) -> list[int]:
    start = text.find(value)
    if start < 0 or text.find(value, start + 1) >= 0:
        raise ValueError(f"Expected exactly one {value!r} span in {text!r}")
    return [start, start + len(value)]


def _encode_spans(
    tokenizer: Any,
    text: str,
    spans: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded["attention_mask"]]
    token_offsets = [
        [int(offset[0]), int(offset[1])] for offset in encoded["offset_mapping"]
    ]
    return _encode_spans_from_tokenization(
        text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_offsets=token_offsets,
        spans=spans,
    )


def _encode_spans_from_tokenization(
    text: str,
    *,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    token_offsets: Sequence[Sequence[int]],
    spans: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    input_ids = [int(value) for value in input_ids]
    attention_mask = [int(value) for value in attention_mask]
    token_offsets = [
        [int(offset[0]), int(offset[1])] for offset in token_offsets
    ]
    if not input_ids or not len(input_ids) == len(attention_mask) == len(token_offsets):
        raise ValueError("Tokenizer returned misaligned span features")
    if any(value != 1 for value in attention_mask):
        raise ValueError("Unpadded record-local tokenization must have an all-one mask")

    result: dict[str, Any] = {
        "text_sha256": sha256_bytes(text.encode("utf-8")),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_offsets": token_offsets,
        "input_ids_sha256": canonical_sha256(input_ids),
    }
    selected_by_name: dict[str, set[int]] = {}
    for name, raw_span in spans.items():
        if len(raw_span) != 2:
            raise ValueError(f"{name} character span must have two entries")
        start, end = (int(raw_span[0]), int(raw_span[1]))
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"{name} character span is invalid")
        positions = [
            index
            for index, (token_start, token_end) in enumerate(token_offsets)
            if token_end > token_start and token_start < end and token_end > start
        ]
        if not positions:
            raise ValueError(f"Tokenizer produced no tokens for {name}")
        covered = set()
        for position in positions:
            token_start, token_end = token_offsets[position]
            covered.update(range(max(start, token_start), min(end, token_end)))
        if covered != set(range(start, end)):
            raise ValueError(f"Tokenizer offsets do not fully cover {name}")
        mask = [index in positions for index in range(len(input_ids))]
        result[f"{name}_char_span"] = [start, end]
        result[f"{name}_token_positions"] = positions
        result[f"{name}_token_mask"] = mask
        result[f"{name}_token_ids"] = [input_ids[index] for index in positions]
        selected_by_name[name] = set(positions)
    for left, right in itertools.combinations(selected_by_name, 2):
        if selected_by_name[left] & selected_by_name[right]:
            raise ValueError(f"Tokenizer spans {left} and {right} overlap")
    return result


def _validate_span_encoding(
    encoding: Mapping[str, Any],
    text: str,
    span_names: Iterable[str],
) -> None:
    input_ids = encoding.get("input_ids")
    attention = encoding.get("attention_mask")
    offsets = encoding.get("token_offsets")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or not isinstance(attention, list)
        or not isinstance(offsets, list)
        or not len(input_ids) == len(attention) == len(offsets)
        or any(type(value) is not int for value in input_ids)
        or any(type(value) is not int or value != 1 for value in attention)
    ):
        raise ValueError("Record-local token arrays are invalid")
    normalized_offsets: list[tuple[int, int]] = []
    for offset in offsets:
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
        ):
            raise ValueError("Record-local token offsets are invalid")
        normalized_offsets.append((offset[0], offset[1]))
    if encoding.get("text_sha256") != sha256_bytes(text.encode("utf-8")):
        raise ValueError("Record-local text hash differs")
    if encoding.get("input_ids_sha256") != canonical_sha256(input_ids):
        raise ValueError("Record-local token hash differs")

    occupied: set[int] = set()
    for name in span_names:
        span = encoding.get(f"{name}_char_span")
        positions = encoding.get(f"{name}_token_positions")
        mask = encoding.get(f"{name}_token_mask")
        token_ids = encoding.get(f"{name}_token_ids")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not isinstance(positions, list)
            or not positions
            or any(type(value) is not int for value in positions)
            or not isinstance(mask, list)
            or len(mask) != len(input_ids)
            or any(type(value) is not bool for value in mask)
        ):
            raise ValueError(f"Record-local {name} mask metadata is invalid")
        derived_positions = [index for index, selected in enumerate(mask) if selected]
        if positions != derived_positions:
            raise ValueError(f"Record-local {name} positions and mask differ")
        if token_ids != [input_ids[index] for index in positions]:
            raise ValueError(f"Record-local {name} token IDs differ")
        start, end = span
        covered = set()
        for position in positions:
            if position < 0 or position >= len(input_ids):
                raise ValueError(f"Record-local {name} position is out of range")
            token_start, token_end = normalized_offsets[position]
            if token_start >= end or token_end <= start:
                raise ValueError(f"Record-local {name} token does not overlap its span")
            covered.update(range(max(start, token_start), min(end, token_end)))
        if covered != set(range(start, end)):
            raise ValueError(f"Record-local {name} tokens do not cover the character span")
        if occupied.intersection(positions):
            raise ValueError("Record-local semantic token masks overlap")
        occupied.update(positions)


def _mapping_signature(records: Sequence[Mapping[str, Any]]) -> str:
    pairs = sorted((str(record["key"]), str(record["value"])) for record in records)
    return canonical_sha256(pairs)


def _mapping_query_signature(
    records: Sequence[Mapping[str, Any]],
    query_key: str,
) -> str:
    return canonical_sha256(
        {
            "mapping_sha256": _mapping_signature(records),
            "query_key": query_key,
        }
    )


def _pair_signature(key: str, value: str) -> str:
    return canonical_sha256({"key": key, "value": value})


def _record_content(key: str, value: str) -> str:
    return RECORD_TEMPLATE.format(key=key, value=value)


def _query_content(key: str) -> str:
    return QUERY_TEMPLATE.format(key=key)


def _response_content(value: str) -> str:
    return RESPONSE_TEMPLATE.format(value=value)


def _encode_read_route(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    query_key: str,
    answer_text: str,
) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("Tokenizer chat template did not render text")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    direct_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(direct_ids, Mapping):
        direct_ids = direct_ids.get("input_ids")
    if not isinstance(direct_ids, Sequence):
        raise ValueError("Tokenizer chat template did not return input IDs")
    input_ids = [int(value) for value in encoded["input_ids"]]
    if input_ids != [int(value) for value in direct_ids]:
        raise ValueError("Rendered and direct chat-template token IDs differ")
    key_span = _span(rendered, query_key)
    answer_span = _span(rendered, answer_text)
    route = _encode_spans_from_tokenization(
        rendered,
        input_ids=input_ids,
        attention_mask=[int(value) for value in encoded["attention_mask"]],
        token_offsets=[
            [int(offset[0]), int(offset[1])]
            for offset in encoded["offset_mapping"]
        ],
        spans={"query_key": key_span, "answer": answer_span},
    )
    return {
        "rendered_text": rendered,
        "text_sha256": route["text_sha256"],
        "input_ids": route["input_ids"],
        "attention_mask": route["attention_mask"],
        "query_key_token_mask": route["query_key_token_mask"],
        "query_key_token_positions": route["query_key_token_positions"],
        "query_key_token_ids": route["query_key_token_ids"],
        "query_key_char_span": route["query_key_char_span"],
        "answer_token_mask": route["answer_token_mask"],
        "answer_token_positions": route["answer_token_positions"],
        "answer_token_ids": route["answer_token_ids"],
        "answer_char_span": route["answer_char_span"],
        "token_offsets": route["token_offsets"],
        "input_ids_sha256": route["input_ids_sha256"],
    }


def _partition_parameters(split: str) -> tuple[tuple[tuple[int, int], ...], int]:
    if split == "train":
        return _offset_pairs(TRAIN_OFFSETS), TRAIN_TEMPLATES_PER_OFFSET_PAIR
    if split == "heldout":
        return _offset_pairs(HELDOUT_OFFSETS), HELDOUT_TEMPLATES_PER_OFFSET_PAIR
    raise ValueError(f"Unsupported partition: {split}")


def build_partition_rows(tokenizer: Any, split: str) -> list[dict[str, Any]]:
    offset_pairs, templates_per_pair = _partition_parameters(split)
    record_encoding_cache: dict[tuple[str, str], dict[str, Any]] = {}
    query_encoding_cache: dict[str, dict[str, Any]] = {}
    response_encoding_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def record_encoding(key: str, value: str) -> dict[str, Any]:
        identity = (key, value)
        if identity not in record_encoding_cache:
            text = _record_content(key, value)
            record_encoding_cache[identity] = _encode_spans(
                tokenizer,
                text,
                {"key": _span(text, key), "value": _span(text, value)},
            )
        return copy.deepcopy(record_encoding_cache[identity])

    def query_encoding(key: str) -> dict[str, Any]:
        if key not in query_encoding_cache:
            text = _query_content(key)
            query_encoding_cache[key] = _encode_spans(
                tokenizer,
                text,
                {"key": _span(text, key)},
            )
        return copy.deepcopy(query_encoding_cache[key])

    def response_encoding(value: str) -> dict[str, Any]:
        if value not in response_encoding_cache:
            text = _response_content(value)
            response_encoding_cache[value] = _encode_spans(
                tokenizer,
                text,
                {"value": _span(text, value)},
            )
        return copy.deepcopy(response_encoding_cache[value])

    for pair_index, offset_pair in enumerate(offset_pairs):
        pair_templates = _templates(split, pair_index, templates_per_pair)
        for template in pair_templates:
            template_id = (
                f"{split}-offset-pair-{pair_index:02d}-"
                f"template-{template['local_index']:03d}"
            )
            offset_row_groups: list[list[dict[str, Any]]] = []
            for offset in offset_pair:
                records: list[dict[str, Any]] = []
                for slot, key_index in enumerate(template["key_indices"]):
                    value_index = _mapped_value_index(key_index, offset)
                    key = KEY_LABELS[key_index]
                    value = VALUE_LABELS[value_index]
                    content = _record_content(key, value)
                    encoding = record_encoding(key, value)
                    records.append(
                        {
                            "slot": slot,
                            "key_index": key_index,
                            "key": key,
                            "value_index": value_index,
                            "value": value,
                            "content": content,
                            "pair_signature_sha256": _pair_signature(key, value),
                            "tokenization": encoding,
                            "tokenization_sha256": canonical_sha256(encoding),
                        }
                    )
                memory_state_id = f"{template_id}-offset-{offset:02d}"
                memory_state_sha256 = canonical_sha256(
                    [
                        {
                            "slot": record["slot"],
                            "key": record["key"],
                            "value": record["value"],
                            "tokenization_sha256": record["tokenization_sha256"],
                        }
                        for record in records
                    ]
                )
                swap_rng = _derived_rng("value-swap", memory_state_id)
                shift = 1 + swap_rng.randrange(RECORDS_PER_EPISODE - 1)
                source_slot_by_destination_slot = [
                    (destination + shift) % RECORDS_PER_EPISODE
                    for destination in range(RECORDS_PER_EPISODE)
                ]
                state_rows: list[dict[str, Any]] = []
                for target_slot, target_record in enumerate(records):
                    query_key = str(target_record["key"])
                    target_value = str(target_record["value"])
                    query_text = _query_content(query_key)
                    answer_text = _response_content(target_value)
                    query_tokens = query_encoding(query_key)
                    answer_tokens = response_encoding(target_value)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query_text},
                        {"role": "assistant", "content": answer_text},
                    ]
                    read_route = _encode_read_route(
                        tokenizer,
                        messages,
                        query_key,
                        answer_text,
                    )
                    read_answer_labels = [
                        token_id if selected else -100
                        for token_id, selected in zip(
                            read_route["input_ids"],
                            read_route["answer_token_mask"],
                            strict=True,
                        )
                    ]
                    row_id = f"{memory_state_id}-query-slot-{target_slot}"
                    swapped_source_slot = source_slot_by_destination_slot[target_slot]
                    swapped_target_value = str(records[swapped_source_slot]["value"])
                    row_records = copy.deepcopy(records)
                    row = {
                        "schema": ROW_SCHEMA,
                        "source_split": split,
                        "row_id": row_id,
                        "template_id": template_id,
                        "memory_state_id": memory_state_id,
                        "memory_state_sha256": memory_state_sha256,
                        "mapping_offset": offset,
                        "mapping_signature_sha256": _mapping_signature(records),
                        "mapping_query_signature_sha256": _mapping_query_signature(
                            records, query_key
                        ),
                        "record_order_permutation": list(
                            template["record_order_permutation"]
                        ),
                        "messages": messages,
                        "record_local_writes": row_records,
                        "write_record_input_ids": [
                            record["tokenization"]["input_ids"]
                            for record in row_records
                        ],
                        "write_record_attention_mask": [
                            record["tokenization"]["attention_mask"]
                            for record in row_records
                        ],
                        "write_record_key_mask": [
                            record["tokenization"]["key_token_mask"]
                            for record in row_records
                        ],
                        "write_record_value_mask": [
                            record["tokenization"]["value_token_mask"]
                            for record in row_records
                        ],
                        "write_record_key_positions": [
                            record["tokenization"]["key_token_positions"]
                            for record in row_records
                        ],
                        "write_record_value_positions": [
                            record["tokenization"]["value_token_positions"]
                            for record in row_records
                        ],
                        "write_record_slot_indices": list(
                            range(RECORDS_PER_EPISODE)
                        ),
                        "read_route_input_ids": read_route["input_ids"],
                        "read_route_attention_mask": read_route["attention_mask"],
                        "read_route_target_mask": read_route[
                            "query_key_token_mask"
                        ],
                        "read_answer_target_mask": read_route[
                            "answer_token_mask"
                        ],
                        "read_answer_labels": read_answer_labels,
                        "query_route_target_slot": target_slot,
                        "read_route": read_route,
                        "read_route_sha256": canonical_sha256(read_route),
                        "query": {
                            "key_index": int(target_record["key_index"]),
                            "key": query_key,
                            "content": query_text,
                            "tokenization": query_tokens,
                            "tokenization_sha256": canonical_sha256(query_tokens),
                            "target_slot": target_slot,
                            "target_value_index": int(target_record["value_index"]),
                            "target_value": target_value,
                            "answer": answer_text,
                            "answer_tokenization": answer_tokens,
                            "answer_tokenization_sha256": canonical_sha256(
                                answer_tokens
                            ),
                        },
                        "query_counterfactuals": None,
                        "donor": None,
                        "value_swap": {
                            "semantics": (
                                "destination_slot_receives_value_from_source_slot"
                            ),
                            "source_slot_by_destination_slot": (
                                source_slot_by_destination_slot
                            ),
                            "query_target_slot": target_slot,
                            "target_value_source_slot": swapped_source_slot,
                            "expected_target_value_index": int(
                                records[swapped_source_slot]["value_index"]
                            ),
                            "expected_target_value": swapped_target_value,
                            "expected_answer": _response_content(
                                swapped_target_value
                            ),
                        },
                        "no_write": {
                            "record_count": 0,
                            "expected_memory_state": "absent",
                            "expected_route": "absent",
                        },
                    }
                    state_rows.append(row)
                offset_row_groups.append(state_rows)

            if (
                len(offset_row_groups) != 2
                or any(len(group) != RECORDS_PER_EPISODE for group in offset_row_groups)
            ):
                raise AssertionError(
                    "Every template must materialize two four-query mapping states"
                )
            first_ordinal = len(rows)
            for offset_index, state_rows in enumerate(offset_row_groups):
                sibling_ids = [row["row_id"] for row in state_rows]
                donor_rows = offset_row_groups[1 - offset_index]
                for target_slot, row in enumerate(state_rows):
                    donor = donor_rows[target_slot]
                    row["source_row_ordinal"] = (
                        first_ordinal
                        + offset_index * RECORDS_PER_EPISODE
                        + target_slot
                    )
                    row["query_counterfactuals"] = {
                        "memory_state_id": row["memory_state_id"],
                        "byte_identical_record_writes_required": True,
                        "row_id_by_target_slot": sibling_ids,
                    }
                    row["donor"] = {
                        "row_id": donor["row_id"],
                        "row_ordinal": (
                            first_ordinal
                            + (1 - offset_index) * RECORDS_PER_EPISODE
                            + target_slot
                        ),
                        "mapping_offset": donor["mapping_offset"],
                        "mapping_signature_sha256": donor[
                            "mapping_signature_sha256"
                        ],
                        "query_key": donor["query"]["key"],
                        "query_target_slot": donor["query"]["target_slot"],
                        "expected_target_value_index": donor["query"][
                            "target_value_index"
                        ],
                        "expected_target_value": donor["query"]["target_value"],
                        "expected_answer": donor["query"]["answer"],
                    }
            rows.extend(row for group in offset_row_groups for row in group)
    _validate_partition_rows(split, rows)
    return rows


def build_partitions(tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    partitions = {
        split: build_partition_rows(tokenizer, split) for split in PARTITION_ORDER
    }
    audit = audit_split_leakage(partitions)
    if not audit["passed"]:
        raise ValueError("Generated train/heldout partitions violate the no-leakage contract")
    return partitions


def _expected_partition_rows(split: str) -> int:
    pairs, templates_per_pair = _partition_parameters(split)
    return len(pairs) * templates_per_pair * 2 * RECORDS_PER_EPISODE


def _validate_partition_rows(split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != _expected_partition_rows(split):
        raise ValueError(f"{split} row count differs from the locked generation contract")
    allowed_offsets = set(TRAIN_OFFSETS if split == "train" else HELDOUT_OFFSETS)
    row_by_id: dict[str, Mapping[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("source_split") != split
            or row.get("source_row_ordinal") != ordinal
            or not isinstance(row.get("row_id"), str)
            or row["row_id"] in row_by_id
            or row.get("mapping_offset") not in allowed_offsets
        ):
            raise ValueError(f"{split} row identity differs at ordinal {ordinal}")
        row_by_id[str(row["row_id"])] = row
        records = row.get("record_local_writes")
        if not isinstance(records, list) or len(records) != RECORDS_PER_EPISODE:
            raise ValueError(f"{split} row {ordinal} does not have four record writes")
        keys: list[str] = []
        values: list[str] = []
        for slot, record in enumerate(records):
            if not isinstance(record, dict) or record.get("slot") != slot:
                raise ValueError(f"{split} row {ordinal} record slots are not canonical")
            key_index = record.get("key_index")
            value_index = record.get("value_index")
            if (
                type(key_index) is not int
                or key_index < 0
                or key_index >= NONCE_COUNT
                or record.get("key") != KEY_LABELS[key_index]
                or type(value_index) is not int
                or value_index != _mapped_value_index(key_index, int(row["mapping_offset"]))
                or record.get("value") != VALUE_LABELS[value_index]
            ):
                raise ValueError(f"{split} row {ordinal} record mapping differs")
            key = str(record["key"])
            value = str(record["value"])
            content = _record_content(key, value)
            encoding = record.get("tokenization")
            if (
                record.get("content") != content
                or not isinstance(encoding, dict)
                or record.get("pair_signature_sha256") != _pair_signature(key, value)
                or record.get("tokenization_sha256") != canonical_sha256(encoding)
            ):
                raise ValueError(f"{split} row {ordinal} record binding differs")
            _validate_span_encoding(encoding, content, ("key", "value"))
            keys.append(key)
            values.append(value)
        if len(set(keys)) != RECORDS_PER_EPISODE or len(set(values)) != RECORDS_PER_EPISODE:
            raise ValueError(f"{split} row {ordinal} keys and values must be unique")
        expected_record_major = {
            "write_record_input_ids": [
                record["tokenization"]["input_ids"] for record in records
            ],
            "write_record_attention_mask": [
                record["tokenization"]["attention_mask"] for record in records
            ],
            "write_record_key_mask": [
                record["tokenization"]["key_token_mask"] for record in records
            ],
            "write_record_value_mask": [
                record["tokenization"]["value_token_mask"] for record in records
            ],
            "write_record_key_positions": [
                record["tokenization"]["key_token_positions"] for record in records
            ],
            "write_record_value_positions": [
                record["tokenization"]["value_token_positions"]
                for record in records
            ],
            "write_record_slot_indices": list(range(RECORDS_PER_EPISODE)),
        }
        if any(row.get(field) != value for field, value in expected_record_major.items()):
            raise ValueError(f"{split} row {ordinal} record-major tensor projection differs")
        expected_memory_state_sha256 = canonical_sha256(
            [
                {
                    "slot": record["slot"],
                    "key": record["key"],
                    "value": record["value"],
                    "tokenization_sha256": record["tokenization_sha256"],
                }
                for record in records
            ]
        )
        if (
            not isinstance(row.get("memory_state_id"), str)
            or row.get("memory_state_sha256") != expected_memory_state_sha256
        ):
            raise ValueError(f"{split} row {ordinal} memory-state binding differs")
        if row.get("mapping_signature_sha256") != _mapping_signature(records):
            raise ValueError(f"{split} row {ordinal} mapping hash differs")

        query = row.get("query")
        if not isinstance(query, dict):
            raise ValueError(f"{split} row {ordinal} query metadata is absent")
        target_slot = query.get("target_slot")
        if type(target_slot) is not int or target_slot not in range(RECORDS_PER_EPISODE):
            raise ValueError(f"{split} row {ordinal} target slot is invalid")
        target = records[target_slot]
        query_text = _query_content(str(target["key"]))
        answer_text = _response_content(str(target["value"]))
        query_encoding = query.get("tokenization")
        answer_encoding = query.get("answer_tokenization")
        if (
            query.get("key_index") != target["key_index"]
            or query.get("key") != target["key"]
            or query.get("target_value_index") != target["value_index"]
            or query.get("target_value") != target["value"]
            or query.get("content") != query_text
            or query.get("answer") != answer_text
            or not isinstance(query_encoding, dict)
            or not isinstance(answer_encoding, dict)
            or query.get("tokenization_sha256") != canonical_sha256(query_encoding)
            or query.get("answer_tokenization_sha256")
            != canonical_sha256(answer_encoding)
        ):
            raise ValueError(f"{split} row {ordinal} query target differs")
        _validate_span_encoding(query_encoding, query_text, ("key",))
        _validate_span_encoding(answer_encoding, answer_text, ("value",))
        expected_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": answer_text},
        ]
        if row.get("messages") != expected_messages:
            raise ValueError(f"{split} row {ordinal} read dialogue contains write context")
        read_route = row.get("read_route")
        if not isinstance(read_route, dict) or not isinstance(
            read_route.get("rendered_text"), str
        ):
            raise ValueError(f"{split} row {ordinal} read-route metadata is absent")
        _validate_span_encoding(
            read_route,
            str(read_route["rendered_text"]),
            ("query_key", "answer"),
        )
        expected_answer_labels = [
            token_id if selected else -100
            for token_id, selected in zip(
                read_route["input_ids"],
                read_route["answer_token_mask"],
                strict=True,
            )
        ]
        if (
            str(read_route["rendered_text"]).count(str(target["key"])) != 1
            or str(read_route["rendered_text"]).count(answer_text) != 1
            or row.get("read_route_sha256") != canonical_sha256(read_route)
            or row.get("read_route_input_ids") != read_route["input_ids"]
            or row.get("read_route_attention_mask") != read_route["attention_mask"]
            or row.get("read_route_target_mask")
            != read_route["query_key_token_mask"]
            or row.get("read_answer_target_mask")
            != read_route["answer_token_mask"]
            or row.get("read_answer_labels") != expected_answer_labels
            or row.get("query_route_target_slot") != target_slot
        ):
            raise ValueError(f"{split} row {ordinal} route-supervision projection differs")
        if row.get("mapping_query_signature_sha256") != _mapping_query_signature(
            records, str(target["key"])
        ):
            raise ValueError(f"{split} row {ordinal} mapping-query hash differs")
        permutation = row.get("record_order_permutation")
        if (
            not isinstance(permutation, list)
            or sorted(permutation) != list(range(RECORDS_PER_EPISODE))
        ):
            raise ValueError(f"{split} row {ordinal} record order is not a permutation")

        value_swap = row.get("value_swap")
        if not isinstance(value_swap, dict):
            raise ValueError(f"{split} row {ordinal} value-swap metadata is absent")
        source_slots = value_swap.get("source_slot_by_destination_slot")
        if (
            not isinstance(source_slots, list)
            or sorted(source_slots) != list(range(RECORDS_PER_EPISODE))
            or any(source == destination for destination, source in enumerate(source_slots))
            or value_swap.get("query_target_slot") != target_slot
        ):
            raise ValueError(f"{split} row {ordinal} value swap is not a derangement")
        source_slot = source_slots[target_slot]
        swapped_record = records[source_slot]
        if (
            value_swap.get("target_value_source_slot") != source_slot
            or value_swap.get("expected_target_value_index")
            != swapped_record["value_index"]
            or value_swap.get("expected_target_value") != swapped_record["value"]
            or value_swap.get("expected_answer")
            != _response_content(str(swapped_record["value"]))
            or swapped_record["value"] == target["value"]
        ):
            raise ValueError(f"{split} row {ordinal} value-swap target differs")
        if row.get("no_write") != {
            "record_count": 0,
            "expected_memory_state": "absent",
            "expected_route": "absent",
        }:
            raise ValueError(f"{split} row {ordinal} no-write metadata differs")

    for ordinal, row in enumerate(rows):
        donor_metadata = row.get("donor")
        if not isinstance(donor_metadata, dict):
            raise ValueError(f"{split} row {ordinal} donor metadata is absent")
        donor_ordinal = donor_metadata.get("row_ordinal")
        if type(donor_ordinal) is not int or donor_ordinal not in range(len(rows)):
            raise ValueError(f"{split} row {ordinal} donor ordinal is invalid")
        donor = rows[donor_ordinal]
        reciprocal = donor.get("donor")
        source_records = row["record_local_writes"]
        donor_records = donor["record_local_writes"]
        if (
            donor_metadata.get("row_id") != donor["row_id"]
            or donor_metadata.get("mapping_offset") != donor["mapping_offset"]
            or donor_metadata.get("mapping_signature_sha256")
            != donor["mapping_signature_sha256"]
            or donor_metadata.get("query_key") != donor["query"]["key"]
            or donor_metadata.get("query_target_slot")
            != donor["query"]["target_slot"]
            or donor_metadata.get("expected_target_value_index")
            != donor["query"]["target_value_index"]
            or donor_metadata.get("expected_target_value")
            != donor["query"]["target_value"]
            or donor_metadata.get("expected_answer") != donor["query"]["answer"]
            or not isinstance(reciprocal, dict)
            or reciprocal.get("row_ordinal") != ordinal
            or row["template_id"] != donor["template_id"]
            or row["query"]["key"] != donor["query"]["key"]
            or row["query"]["target_slot"] != donor["query"]["target_slot"]
            or [record["key"] for record in source_records]
            != [record["key"] for record in donor_records]
            or any(
                source["value"] == donor_record["value"]
                for source, donor_record in zip(
                    source_records, donor_records, strict=True
                )
            )
        ):
            raise ValueError(f"{split} row {ordinal} donor contract differs")

    state_families: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        state_families.setdefault(str(row["memory_state_id"]), []).append(row)
    expected_state_count = len(rows) // RECORDS_PER_EPISODE
    if len(state_families) != expected_state_count:
        raise ValueError(f"{split} memory-state family count differs")
    for state_id, family in state_families.items():
        family = sorted(family, key=lambda item: int(item["query"]["target_slot"]))
        if (
            len(family) != RECORDS_PER_EPISODE
            or [row["query"]["target_slot"] for row in family]
            != list(range(RECORDS_PER_EPISODE))
        ):
            raise ValueError(f"{split} state {state_id} lacks all four query variants")
        row_ids = [str(row["row_id"]) for row in family]
        reference_write_hash = canonical_sha256(
            {
                field: family[0][field]
                for field in (
                    "write_record_input_ids",
                    "write_record_attention_mask",
                    "write_record_key_mask",
                    "write_record_value_mask",
                    "write_record_slot_indices",
                )
            }
        )
        for row in family:
            counterfactuals = row.get("query_counterfactuals")
            current_write_hash = canonical_sha256(
                {
                    field: row[field]
                    for field in (
                        "write_record_input_ids",
                        "write_record_attention_mask",
                        "write_record_key_mask",
                        "write_record_value_mask",
                        "write_record_slot_indices",
                    )
                }
            )
            if (
                not isinstance(counterfactuals, dict)
                or counterfactuals.get("memory_state_id") != state_id
                or counterfactuals.get("byte_identical_record_writes_required")
                is not True
                or counterfactuals.get("row_id_by_target_slot") != row_ids
                or current_write_hash != reference_write_hash
                or row["memory_state_sha256"] != family[0]["memory_state_sha256"]
                or row["mapping_signature_sha256"]
                != family[0]["mapping_signature_sha256"]
            ):
                raise ValueError(
                    f"{split} state {state_id} query-counterfactual contract differs"
                )


def audit_split_leakage(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(partitions) != set(PARTITION_ORDER):
        raise ValueError("Canary requires exact train and heldout partitions")

    inventory: dict[str, dict[str, Any]] = {}
    for split in PARTITION_ORDER:
        rows = partitions[split]
        _validate_partition_rows(split, rows)
        pairs = sorted(
            {
                (str(record["key"]), str(record["value"]))
                for row in rows
                for record in row["record_local_writes"]
            }
        )
        mappings = sorted({str(row["mapping_signature_sha256"]) for row in rows})
        mapping_queries = sorted(
            {str(row["mapping_query_signature_sha256"]) for row in rows}
        )
        keys = sorted({key for key, _ in pairs})
        values = sorted({value for _, value in pairs})
        permutations = sorted(
            {tuple(int(value) for value in row["record_order_permutation"]) for row in rows}
        )
        target_slot_counts = {
            str(slot): sum(row["query"]["target_slot"] == slot for row in rows)
            for slot in range(RECORDS_PER_EPISODE)
        }
        inventory[split] = {
            "row_count": len(rows),
            "row_ids": [str(row["row_id"]) for row in rows],
            "key_value_pairs": [[key, value] for key, value in pairs],
            "mapping_signatures_sha256": mappings,
            "mapping_query_signatures_sha256": mapping_queries,
            "keys": keys,
            "values": values,
            "record_order_permutations": [list(value) for value in permutations],
            "query_target_slot_counts": target_slot_counts,
        }

    train = inventory["train"]
    heldout = inventory["heldout"]
    pair_intersection = sorted(
        set(map(tuple, train["key_value_pairs"]))
        & set(map(tuple, heldout["key_value_pairs"]))
    )
    mapping_intersection = sorted(
        set(train["mapping_signatures_sha256"])
        & set(heldout["mapping_signatures_sha256"])
    )
    mapping_query_intersection = sorted(
        set(train["mapping_query_signatures_sha256"])
        & set(heldout["mapping_query_signatures_sha256"])
    )
    row_id_intersection = sorted(set(train["row_ids"]) & set(heldout["row_ids"]))
    order_pattern_intersection = sorted(
        set(map(tuple, train["record_order_permutations"]))
        & set(map(tuple, heldout["record_order_permutations"]))
    )
    expected_keys = sorted(KEY_LABELS)
    expected_values = sorted(VALUE_LABELS)
    target_slots_balanced = all(
        len(set(partition["query_target_slot_counts"].values())) == 1
        for partition in inventory.values()
    )
    randomized_orders = (
        train["record_order_permutations"]
        == sorted([list(pattern) for pattern in TRAIN_RECORD_ORDER_PATTERNS])
        and heldout["record_order_permutations"]
        == sorted([list(pattern) for pattern in HELDOUT_RECORD_ORDER_PATTERNS])
    )
    passed = (
        not pair_intersection
        and not mapping_intersection
        and not mapping_query_intersection
        and not row_id_intersection
        and not order_pattern_intersection
        and train["keys"] == heldout["keys"] == expected_keys
        and train["values"] == heldout["values"] == expected_values
        and target_slots_balanced
        and randomized_orders
    )
    return {
        "passed": passed,
        "train_key_coverage": len(train["keys"]),
        "heldout_key_coverage": len(heldout["keys"]),
        "train_value_coverage": len(train["values"]),
        "heldout_value_coverage": len(heldout["values"]),
        "shared_key_count": len(set(train["keys"]) & set(heldout["keys"])),
        "shared_value_count": len(
            set(train["values"]) & set(heldout["values"])
        ),
        "key_value_pair_intersection": [list(value) for value in pair_intersection],
        "key_value_pair_intersection_count": len(pair_intersection),
        "mapping_intersection_sha256": mapping_intersection,
        "mapping_intersection_count": len(mapping_intersection),
        "mapping_query_intersection_sha256": mapping_query_intersection,
        "mapping_query_intersection_count": len(mapping_query_intersection),
        "row_id_intersection": row_id_intersection,
        "row_id_intersection_count": len(row_id_intersection),
        "record_order_pattern_intersection": [
            list(value) for value in order_pattern_intersection
        ],
        "record_order_pattern_intersection_count": len(
            order_pattern_intersection
        ),
        "train_record_order_pattern_count": len(
            train["record_order_permutations"]
        ),
        "heldout_record_order_pattern_count": len(
            heldout["record_order_permutations"]
        ),
        "query_target_slots_balanced": target_slots_balanced,
        "record_orders_randomized": randomized_orders,
        "partition_inventory_sha256": canonical_sha256(inventory),
        "partitions": inventory,
    }


def build_split_manifest(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    audit = audit_split_leakage(partitions)
    if not audit["passed"]:
        raise ValueError("Split leakage audit failed")
    manifest: dict[str, Any] = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "generation_seed": GENERATION_SEED,
        "spec_sha256": CANARY_SPEC_SHA256,
        "policy": {
            "train_mapping_offsets": list(TRAIN_OFFSETS),
            "heldout_mapping_offsets": list(HELDOUT_OFFSETS),
            "train_record_order_patterns": [
                list(pattern) for pattern in TRAIN_RECORD_ORDER_PATTERNS
            ],
            "heldout_record_order_patterns": [
                list(pattern) for pattern in HELDOUT_RECORD_ORDER_PATTERNS
            ],
            "unseen_key_value_pairings_required": True,
            "unseen_mappings_required": True,
            "unseen_mapping_query_combinations_required": True,
            "unseen_record_order_patterns_required": True,
            "same_nonce_keys_seen_in_both_splits": True,
            "same_nonce_values_seen_in_both_splits": True,
        },
        "audit": audit,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _row_manifest_record(
    row: Mapping[str, Any],
    raw_line: str,
) -> dict[str, Any]:
    return {
        "schema": ROW_MANIFEST_SCHEMA,
        "source_split": row["source_split"],
        "source_row_ordinal": row["source_row_ordinal"],
        "row_id": row["row_id"],
        "template_id": row["template_id"],
        "memory_state_id": row["memory_state_id"],
        "memory_state_sha256": row["memory_state_sha256"],
        "mapping_offset": row["mapping_offset"],
        "mapping_signature_sha256": row["mapping_signature_sha256"],
        "mapping_query_signature_sha256": row["mapping_query_signature_sha256"],
        "query_key": row["query"]["key"],
        "query_target_slot": row["query"]["target_slot"],
        "query_target_value": row["query"]["target_value"],
        "donor_row_id": row["donor"]["row_id"],
        "donor_row_ordinal": row["donor"]["row_ordinal"],
        "query_counterfactual_row_ids": row["query_counterfactuals"][
            "row_id_by_target_slot"
        ],
        "value_swap_expected_target_value": row["value_swap"][
            "expected_target_value"
        ],
        "record_tokenization_sha256": [
            record["tokenization_sha256"] for record in row["record_local_writes"]
        ],
        "query_tokenization_sha256": row["query"]["tokenization_sha256"],
        "answer_tokenization_sha256": row["query"][
            "answer_tokenization_sha256"
        ],
        "row_sha256": sha256_bytes(raw_line.encode("utf-8")),
    }


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, list[str]]:
    lines = [canonical_json_bytes(row).decode("utf-8") for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8"), lines


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
    }


def write_bundle(
    output_dir: Path,
    *,
    model: Mapping[str, Any],
    tokenizer: Any,
    partitions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise ValueError(f"V3 output directory must be fresh or empty: {resolved_output}")
    resolved_output.mkdir(parents=True, exist_ok=True)
    materialized = dict(partitions) if partitions is not None else build_partitions(tokenizer)
    split_manifest = build_split_manifest(materialized)

    partition_records: dict[str, Any] = {}
    output_files: dict[str, str] = {}
    for split in PARTITION_ORDER:
        rows = materialized[split]
        data_payload, data_lines = _jsonl_payload(rows)
        row_records = [
            _row_manifest_record(row, line)
            for row, line in zip(rows, data_lines, strict=True)
        ]
        row_payload, _ = _jsonl_payload(row_records)
        data_name = f"{split}.jsonl"
        rows_name = f"source_rows_{split}.jsonl"
        atomic_write(resolved_output / data_name, data_payload)
        atomic_write(resolved_output / rows_name, row_payload)
        partition_records[split] = {
            "rows": len(rows),
            "source_split": split,
            "data": {
                "path": data_name,
                "bytes": len(data_payload),
                "sha256": sha256_bytes(data_payload),
            },
            "row_manifest": {
                "path": rows_name,
                "bytes": len(row_payload),
                "sha256": sha256_bytes(row_payload),
            },
            "ordered_row_ids_sha256": canonical_sha256(
                [row["row_id"] for row in rows]
            ),
        }
        output_files[f"{split}_file"] = str(resolved_output / data_name)
        output_files[f"{split}_rows_file"] = str(resolved_output / rows_name)

    split_payload = (
        json.dumps(split_manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    split_name = "split_manifest.json"
    atomic_write(resolved_output / split_name, split_payload)
    manifest: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "task": TASK_NAME,
        "purpose": SOURCE_PURPOSE,
        "spec_sha256": CANARY_SPEC_SHA256,
        "spec": canary_spec(),
        "contract": dict(SOURCE_CONTRACT),
        "hf": {
            "endpoint": HF_MIRROR_ENDPOINT,
            "local_files_only": True,
            "tokenizer": _tokenizer_identity(tokenizer),
        },
        "model": dict(model),
        "partitions": partition_records,
        "split_manifest": {
            "path": split_name,
            "bytes": len(split_payload),
            "sha256": sha256_bytes(split_payload),
            "manifest_sha256": split_manifest["manifest_sha256"],
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path = resolved_output / "source_manifest.json"
    atomic_write(manifest_path, manifest_payload)
    return {
        "output_dir": str(resolved_output),
        **output_files,
        "split_manifest": str(resolved_output / split_name),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_file_sha256": sha256_bytes(manifest_payload),
        "model_identity_sha256": model["identity_sha256"],
        "train_rows": len(materialized["train"]),
        "heldout_rows": len(materialized["heldout"]),
    }


def build_bundle(model_path: Path, output_dir: Path) -> dict[str, Any]:
    configure_hf_mirror()
    model = bind_model_artifacts(model_path)
    tokenizer = load_local_tokenizer(model_path)
    return write_bundle(output_dir, model=model, tokenizer=tokenizer)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _verify_canonical_manifest(
    manifest: Mapping[str, Any],
    *,
    hash_field: str = "manifest_sha256",
    description: str,
) -> str:
    unsigned = dict(manifest)
    declared = unsigned.pop(hash_field, None)
    actual = canonical_sha256(unsigned)
    if declared != actual:
        raise ValueError(f"{description} canonical SHA-256 differs")
    return actual


def _bound_artifact(
    manifest_dir: Path,
    record: Any,
    description: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Source manifest omits {description}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise ValueError(f"Source manifest {description} path must be relative")
    path = (manifest_dir / raw_path).resolve()
    if path.parent != manifest_dir or not path.is_file() or path.is_symlink():
        raise ValueError(f"Source manifest {description} path is invalid: {path}")
    if (
        record.get("bytes") != path.stat().st_size
        or record.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"Source manifest {description} binding differs")
    return path


def _read_jsonl(path: Path, description: str) -> tuple[list[dict[str, Any]], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{description} row {ordinal} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{description} row {ordinal} must be a JSON object")
        rows.append(row)
    return rows, lines


def load_source_bundle(
    source_manifest: Path,
    *,
    model_path: Path | None = None,
    verify_model_hashes: bool = False,
) -> dict[str, Any]:
    manifest_path = source_manifest.expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"V3 source manifest is invalid: {manifest_path}")
    manifest = _load_json_object(manifest_path, "V3 source manifest")
    manifest_sha256 = _verify_canonical_manifest(
        manifest, description="V3 source manifest"
    )
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("task") != TASK_NAME
        or manifest.get("purpose") != SOURCE_PURPOSE
        or manifest.get("spec_sha256") != CANARY_SPEC_SHA256
        or manifest.get("spec") != canary_spec()
        or manifest.get("contract") != SOURCE_CONTRACT
        or manifest.get("hf", {}).get("endpoint") != HF_MIRROR_ENDPOINT
        or manifest.get("hf", {}).get("local_files_only") is not True
    ):
        raise ValueError("V3 source-manifest identity differs")

    manifest_dir = manifest_path.parent
    declared_partitions = manifest.get("partitions")
    if not isinstance(declared_partitions, dict) or set(declared_partitions) != set(
        PARTITION_ORDER
    ):
        raise ValueError("V3 source manifest requires exact train and heldout partitions")
    partitions: dict[str, list[dict[str, Any]]] = {}
    row_manifests: dict[str, list[dict[str, Any]]] = {}
    for split in PARTITION_ORDER:
        record = declared_partitions[split]
        if (
            not isinstance(record, dict)
            or record.get("rows") != _expected_partition_rows(split)
            or record.get("source_split") != split
        ):
            raise ValueError(f"V3 {split} partition identity differs")
        data_path = _bound_artifact(manifest_dir, record.get("data"), f"{split} data")
        rows_path = _bound_artifact(
            manifest_dir, record.get("row_manifest"), f"{split} row manifest"
        )
        rows, raw_lines = _read_jsonl(data_path, f"V3 {split} data")
        row_records, _ = _read_jsonl(rows_path, f"V3 {split} row manifest")
        _validate_partition_rows(split, rows)
        expected_records = [
            _row_manifest_record(row, line)
            for row, line in zip(rows, raw_lines, strict=True)
        ]
        if row_records != expected_records:
            raise ValueError(f"V3 {split} row manifest differs from bound data")
        if record.get("ordered_row_ids_sha256") != canonical_sha256(
            [row["row_id"] for row in rows]
        ):
            raise ValueError(f"V3 {split} ordered row IDs differ")
        partitions[split] = rows
        row_manifests[split] = row_records

    split_path = _bound_artifact(
        manifest_dir, manifest.get("split_manifest"), "split manifest"
    )
    split_manifest = _load_json_object(split_path, "V3 split manifest")
    split_sha256 = _verify_canonical_manifest(
        split_manifest, description="V3 split manifest"
    )
    if (
        split_manifest != build_split_manifest(partitions)
        or manifest["split_manifest"].get("manifest_sha256") != split_sha256
    ):
        raise ValueError("V3 split manifest differs from recomputed leakage audit")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("V3 source manifest omits model identity")
    resolved_model_path = Path(str(model.get("path", ""))).expanduser().resolve()
    artifacts = model.get("artifacts")
    if (
        not resolved_model_path.is_dir()
        or resolved_model_path.is_symlink()
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(MODEL_ARTIFACT_NAMES)
        or model.get("identity_sha256") != canonical_sha256(artifacts)
    ):
        raise ValueError("V3 model identity differs")
    if model_path is not None and resolved_model_path != model_path.expanduser().resolve():
        raise ValueError("V3 model path differs from source provenance")
    if verify_model_hashes and bind_model_artifacts(resolved_model_path) != model:
        raise ValueError("Current model artifacts differ from V3 provenance")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": sha256_file(manifest_path),
        "split_manifest": split_manifest,
        "split_manifest_path": split_path,
        "partitions": partitions,
        "row_manifests": row_manifests,
        "model": model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bundle(args.model_path, args.output_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
