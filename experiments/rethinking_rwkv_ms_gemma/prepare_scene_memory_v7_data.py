#!/usr/bin/env python3
"""Build frozen-Hard32-aligned V7 scene-memory training artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import networkx as nx

from deltamem.chat_templates import apply_chat_template
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_failure_pairs import (
    BaseRecord,
    SourceRow,
    join_train_records,
    load_base_records,
    load_source_split,
    sha256_file,
    sha256_text,
    validate_base_train_producer_bundle,
)
from experiments.rethinking_rwkv_ms_gemma.run_novel_agent_eval import score_prediction
from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (
    first_pair_distinguishing_scene_target,
    rendered_scene_decision_features,
)


SCHEMA = "rwkv_ms_scene_memory_v7_bundle.v1"
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v7_source.v1"
PAIRING_SCHEMA = "rwkv_ms_scene_memory_v7_pairing.v1"
PAIRING_BINDING_SCHEMA = "rwkv_ms_scene_memory_v7_pairing_binding.v1"
ROW_SCHEMA = "rwkv_ms_scene_memory_v7_row.v1"
TASK = "scene-v4-current"
FAILURE_STRATA = (
    "invalid_schema",
    "false_positive_only",
    "false_negative_only",
    "mixed",
)
TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
GOLD_CARDINALITY_QUOTAS = {0: 9, 1: 16, 2: 5, 3: 2}
TRAIN_DIRECTED_PAIR_QUOTAS = {
    "presence": 18,
    "same_cardinality_value": 10,
    "cross_cardinality_value": 4,
}
TINY_DIRECTED_PAIR_QUOTAS = {
    "presence": 0,
    "same_cardinality_value": 2,
    "cross_cardinality_value": 0,
}
SELECTION_NAMESPACE = "rwkv_ms_scene_memory_v7_train32_selection.v1"
PARAGRAPH_ANCHOR = __import__("re").compile(r"^\[P\d+\]\s*", __import__("re").MULTILINE)

DATASET_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/datasets/novel-agent-sft-dataset"
)
TASK_DIR = DATASET_ROOT / "training/v4-scene-boundary-detection"
TRAIN_FILE = TASK_DIR / "train.jsonl"
VAL_FILE = TASK_DIR / "val.jsonl"
TEST_FILE = TASK_DIR / "test.jsonl"
TRAIN_FILE_SHA256 = "785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22"
VAL_FILE_SHA256 = "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513"
TEST_FILE_SHA256 = "d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d"

BASE_TRAIN_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "base_train_seed3407_n64_v1"
)
BASE_TRAIN_RECORDS = BASE_TRAIN_DIR / "base.jsonl"
BASE_TRAIN_RECORDS_SHA256 = "853d52aca502e431479a07ac62b5973354720a9c623cd4109ec2332059d52b18"
BASE_TRAIN_MANIFEST_SHA256 = "44c9836cd8433cf352ff606b3fa36e9fc7c7453ed51d3b8b3bc7727770bb9eeb"
BASE_TRAIN_SELECTION_SHA256 = "0fb9595ee8587508390157189dc49184ff024cbc37590a4bb0ffbf3be85717b5"
BASE_TRAIN_SUMMARY_SHA256 = "51cf7d902a20f92dbea33b456b88a6a49a53ed9ce474dda3196a04797cde3c87"

HARD32_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "pairs_candidate64_failure32_holdout32_v1"
)
HARD32_FILE = HARD32_DIR / "holdout.jsonl"
HARD32_FILE_SHA256 = "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
HARD32_SELECTION = HARD32_DIR / "holdout_source_indices.json"
HARD32_SELECTION_SHA256 = "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
HARD32_BASE_RECORDS = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_outputs/novel_rwkv_ms_memory/"
    "eval_scene_v6_identityproof_run2_step32_hard32_v2/base_full.jsonl"
)
HARD32_BASE_RECORDS_SHA256 = "4740695691bad3ba6c808cc29d734022bf25b0a58ce301a02d551905ec27b4a1"

TOKENIZER_PATH = Path(
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
)
DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_memory_v7_fixed_hard32_aligned_train32_v1"
)


class ContractError(ValueError):
    pass


@dataclass
class Candidate:
    source: SourceRow
    base_record: BaseRecord
    base_payload: dict[str, Any]
    strict_score: dict[str, Any]
    failure_stratum: str
    boundary_count: int
    label_sha256: str
    paragraph_hashes: tuple[str, ...]
    selection_sha256: str
    token_metadata: dict[str, Any] | None = None
    decision_features: dict[str, Any] | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_self_hash(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> dict[str, Any]:
    result = dict(payload)
    require(field not in result, f"self-hash field already exists: {field}")
    result[field] = canonical_sha256(result)
    return result


def validate_self_hash(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> None:
    unsigned = dict(payload)
    recorded = unsigned.pop(field, None)
    require(recorded == canonical_sha256(unsigned), f"{field} differs")


def normalize_overlap_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(character for character in normalized if not character.isspace())


def normalized_paragraph_hashes(prompt: str) -> tuple[str, ...]:
    anchors = list(PARAGRAPH_ANCHOR.finditer(prompt))
    paragraphs = []
    for index, anchor in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(prompt)
        normalized = normalize_overlap_text(prompt[anchor.end() : end])
        if normalized:
            paragraphs.append(sha256_text(normalized))
    return tuple(dict.fromkeys(paragraphs))


def load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {description}: {path}") from exc
    require(isinstance(payload, dict), f"{description} must be an object")
    return payload


def read_jsonl(path: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    require(path.is_file() and not path.is_symlink(), f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\r\n")
            require(bool(raw_line.strip()), f"blank JSONL row at {path}:{line_number}")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at {path}:{line_number}") from exc
            require(isinstance(payload, dict), f"JSONL row must be an object at {path}:{line_number}")
            records.append((raw_line, payload))
    return records


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, rendered)


def strict_failure_stratum(score: Mapping[str, Any]) -> str | None:
    if not bool(score.get("schema_valid")):
        return "invalid_schema"
    fp = int(score.get("fp", 0))
    fn = int(score.get("fn", 0))
    if fp and fn:
        return "mixed"
    if fp:
        return "false_positive_only"
    if fn:
        return "false_negative_only"
    return None


def target_stratum(left: Candidate, right: Candidate) -> str:
    if left.boundary_count == 0 or right.boundary_count == 0:
        require(
            left.boundary_count != right.boundary_count,
            "empty labels cannot be paired with identical empty labels",
        )
        return "presence"
    if left.boundary_count == right.boundary_count:
        return "same_cardinality_value"
    return "cross_cardinality_value"


def _hard32_contract(
    hard32_file: Path,
    hard32_selection: Path,
    hard32_base_records: Path,
) -> tuple[dict[int, dict[str, int]], dict[str, int], set[str]]:
    require(sha256_file(hard32_file) == HARD32_FILE_SHA256, "fixed Hard32 data SHA-256 differs")
    require(
        sha256_file(hard32_selection) == HARD32_SELECTION_SHA256,
        "fixed Hard32 selection SHA-256 differs",
    )
    require(
        sha256_file(hard32_base_records) == HARD32_BASE_RECORDS_SHA256,
        "fixed Hard32 base records SHA-256 differs",
    )
    hard_rows = read_jsonl(hard32_file)
    selection = load_json_object(hard32_selection, description="fixed Hard32 selection")
    selection_rows = selection.get("rows")
    require(isinstance(selection_rows, list) and len(selection_rows) == 32, "Hard32 selection must contain 32 rows")
    expected_by_source = {
        int(item["source_index"]): str(item["row_sha256"])
        for item in selection_rows
        if isinstance(item, dict)
    }
    require(len(expected_by_source) == 32, "Hard32 selection source indices are invalid")
    hard_by_hash: dict[str, dict[str, Any]] = {}
    hard_paragraphs: set[str] = set()
    hard_cardinality = Counter()
    for raw_line, row in hard_rows:
        messages = row.get("messages")
        require(isinstance(messages, list) and len(messages) == 3, "Hard32 row messages differ")
        row_hash = sha256_text(raw_line)
        hard_by_hash[row_hash] = row
        gold = json.loads(messages[2]["content"])
        score = score_prediction("scene", gold, gold)
        require(bool(score["schema_valid"]) and not score["fp"] and not score["fn"], "Hard32 gold is invalid")
        hard_cardinality[len(score["gold_boundaries"])] += 1
        hard_paragraphs.update(normalized_paragraph_hashes(messages[1]["content"]))
    require(dict(sorted(hard_cardinality.items())) == GOLD_CARDINALITY_QUOTAS, "fixed Hard32 cardinality quotas differ")

    target_matrix: dict[int, Counter[str]] = defaultdict(Counter)
    hard_failure_counts = Counter()
    base_rows = read_jsonl(hard32_base_records)
    require(len(base_rows) == 32, "Hard32 base records must contain 32 rows")
    for _, record in base_rows:
        source_index = record.get("source_index")
        require(isinstance(source_index, int) and not isinstance(source_index, bool), "Hard32 base source index is invalid")
        row_hash = record.get("row_sha256")
        require(expected_by_source.get(source_index) == row_hash, "Hard32 base row binding differs")
        hard_row = hard_by_hash.get(str(row_hash))
        require(hard_row is not None, "Hard32 base record does not bind fixed data")
        gold = json.loads(hard_row["messages"][2]["content"])
        require(record.get("gold") == gold, "Hard32 base record gold differs")
        strict = score_prediction("scene", record.get("parsed_json"), gold)
        require(record.get("score_strict") == strict, "Hard32 stored strict score differs")
        failure = strict_failure_stratum(strict)
        require(failure is not None, "fixed Hard32 contains a strict base success")
        cardinality = len(strict["gold_boundaries"])
        target_matrix[cardinality][failure] += 1
        hard_failure_counts[failure] += 1
    return (
        {
            cardinality: {stratum: target_matrix[cardinality][stratum] for stratum in FAILURE_STRATA}
            for cardinality in GOLD_CARDINALITY_QUOTAS
        },
        {stratum: hard_failure_counts[stratum] for stratum in FAILURE_STRATA},
        hard_paragraphs,
    )


def _base_payloads_by_row(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_line, payload in read_jsonl(path):
        row_hash = payload.get("row_sha256")
        require(isinstance(row_hash, str) and row_hash not in result, "base record row hash is invalid or duplicate")
        payload = dict(payload)
        payload["_raw_record_sha256"] = sha256_text(raw_line)
        result[row_hash] = payload
    return result


def load_candidate_pool(
    *,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    base_records_path: Path,
    hard_paragraph_hashes: set[str],
) -> tuple[list[Candidate], dict[str, Any]]:
    require(sha256_file(train_file) == TRAIN_FILE_SHA256, "official train SHA-256 differs")
    require(sha256_file(val_file) == VAL_FILE_SHA256, "official val SHA-256 differs")
    require(sha256_file(test_file) == TEST_FILE_SHA256, "official test SHA-256 differs")
    require(sha256_file(base_records_path) == BASE_TRAIN_RECORDS_SHA256, "base train records SHA-256 differs")
    require(
        sha256_file(base_records_path.parent / "manifest.json") == BASE_TRAIN_MANIFEST_SHA256,
        "base train producer manifest SHA-256 differs",
    )
    require(
        sha256_file(base_records_path.parent / "candidate_selection.json") == BASE_TRAIN_SELECTION_SHA256,
        "base train producer selection SHA-256 differs",
    )
    require(
        sha256_file(base_records_path.parent / "summary.json") == BASE_TRAIN_SUMMARY_SHA256,
        "base train producer summary SHA-256 differs",
    )
    train_rows = load_source_split(train_file, split="train")
    val_rows = load_source_split(val_file, split="val")
    test_rows = load_source_split(test_file, split="test")
    base_records, total_records = load_base_records(base_records_path, task_name=TASK)
    require(total_records == len(base_records) == 64, "only the complete predeclared candidate64 producer is supported")
    joined = join_train_records(train_rows, base_records)
    producer_bundle = validate_base_train_producer_bundle(
        base_train_eval_path=base_records_path,
        train_source_path=train_file,
        joined_records=joined,
        total_eval_records=total_records,
        task_name=TASK,
    )
    base_payloads = _base_payloads_by_row(base_records_path)
    val_hashes = {row.row_sha256 for row in val_rows}
    test_hashes = {row.row_sha256 for row in test_rows}
    candidates: list[Candidate] = []
    excluded_overlap = 0
    strict_successes = 0
    producer_score_drift_rows = 0
    for source, base_record in joined:
        require(source.row_sha256 not in val_hashes and source.row_sha256 not in test_hashes, "candidate row entered val or test")
        payload = base_payloads[source.row_sha256]
        strict = score_prediction("scene", base_record.parsed_json, source.gold)
        if payload.get("score") != strict:
            producer_score_drift_rows += 1
        failure = strict_failure_stratum(strict)
        if failure is None:
            strict_successes += 1
            continue
        paragraph_hashes = normalized_paragraph_hashes(source.messages[1]["content"])
        if set(paragraph_hashes) & hard_paragraph_hashes:
            excluded_overlap += 1
            continue
        boundaries = list(strict["gold_boundaries"])
        boundary_count = len(boundaries)
        if boundary_count not in GOLD_CARDINALITY_QUOTAS:
            continue
        label_sha256 = canonical_sha256(boundaries)
        candidates.append(
            Candidate(
                source=source,
                base_record=base_record,
                base_payload=payload,
                strict_score=strict,
                failure_stratum=failure,
                boundary_count=boundary_count,
                label_sha256=label_sha256,
                paragraph_hashes=paragraph_hashes,
                selection_sha256=sha256_text(
                    f"{SELECTION_NAMESPACE}\0{source.prompt_sha256}"
                ),
            )
        )
    return candidates, {
        "candidate_pool_rows": len(joined),
        "eligible_strict_failures": len(candidates),
        "strict_successes_excluded": strict_successes,
        "producer_score_drift_rows": producer_score_drift_rows,
        "eligibility_score_source": "current_exact_run_novel_agent_eval.score_prediction_scene",
        "hard32_paragraph_overlap_rows_excluded": excluded_overlap,
        "producer_bundle": producer_bundle,
    }


def _allocation_options(
    *,
    quota: int,
    available: Mapping[str, int],
    target: Mapping[str, int],
) -> list[tuple[tuple[int, int, tuple[int, ...]], dict[str, int]]]:
    options = []
    ranges = [range(min(quota, int(available.get(stratum, 0))) + 1) for stratum in FAILURE_STRATA]
    for values in itertools.product(*ranges):
        if sum(values) != quota:
            continue
        counts = dict(zip(FAILURE_STRATA, values))
        cell_l1 = sum(abs(counts[stratum] - int(target.get(stratum, 0))) for stratum in FAILURE_STRATA)
        invalid_excess = max(0, counts["invalid_schema"] - int(target.get("invalid_schema", 0)))
        options.append(((cell_l1, invalid_excess, values), counts))
    return sorted(options, key=lambda item: item[0])


def select_train32(
    candidates: Sequence[Candidate],
    target_matrix: Mapping[int, Mapping[str, int]],
) -> tuple[list[Candidate], dict[str, Any]]:
    grouped: dict[int, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        grouped[candidate.boundary_count][candidate.failure_stratum].append(candidate)
    selected: list[Candidate] = []
    actual_matrix: dict[int, dict[str, int]] = {}
    cell_l1_total = 0
    for cardinality, quota in GOLD_CARDINALITY_QUOTAS.items():
        for rows in grouped[cardinality].values():
            rows.sort(
                key=lambda row: (
                    row.selection_sha256,
                    row.source.prompt_sha256,
                    row.source.line_index,
                )
            )
        available = {
            stratum: len(grouped[cardinality][stratum]) for stratum in FAILURE_STRATA
        }
        options = _allocation_options(
            quota=quota,
            available=available,
            target=target_matrix[cardinality],
        )
        require(options, f"candidate pool cannot satisfy cardinality quota {cardinality}:{quota}")
        objective, allocation = options[0]
        cell_l1_total += objective[0]
        actual_matrix[cardinality] = allocation
        for stratum in FAILURE_STRATA:
            selected.extend(grouped[cardinality][stratum][: allocation[stratum]])
    selected.sort(key=lambda row: row.source.line_index)
    require(len(selected) == 32, "V7 Train32 selection did not produce 32 rows")
    actual_cardinality = Counter(row.boundary_count for row in selected)
    require(dict(sorted(actual_cardinality.items())) == GOLD_CARDINALITY_QUOTAS, "V7 Train32 cardinality quotas differ")
    target_failure = Counter()
    actual_failure = Counter(row.failure_stratum for row in selected)
    for target in target_matrix.values():
        target_failure.update(target)
    overall_l1 = sum(
        abs(actual_failure[stratum] - target_failure[stratum])
        for stratum in FAILURE_STRATA
    )
    return selected, {
        "gold_cardinality_quotas": dict(GOLD_CARDINALITY_QUOTAS),
        "hard32_failure_matrix": {
            str(cardinality): dict(target_matrix[cardinality])
            for cardinality in GOLD_CARDINALITY_QUOTAS
        },
        "selected_failure_matrix": {
            str(cardinality): actual_matrix[cardinality]
            for cardinality in GOLD_CARDINALITY_QUOTAS
        },
        "hard32_failure_counts": {
            stratum: target_failure[stratum] for stratum in FAILURE_STRATA
        },
        "selected_failure_counts": {
            stratum: actual_failure[stratum] for stratum in FAILURE_STRATA
        },
        "minimum_cellwise_l1": cell_l1_total,
        "overall_failure_count_l1": overall_l1,
        "matching_mode": "globally_exact_per_cardinality_exhaustive_count_allocation_v1",
        "relaxed_silently": False,
    }


def _token_ids(tokenizer, rendered: str) -> list[int]:
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded.input_ids
    if ids and isinstance(ids[0], list):
        require(len(ids) == 1, "tokenizer returned multiple rows")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def materialize_token_metadata(candidate: Candidate, tokenizer) -> None:
    messages = candidate.source.messages
    write_messages = messages[:2]
    write_rendered = apply_chat_template(
        tokenizer,
        write_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    generation_prefix_rendered = apply_chat_template(
        tokenizer,
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True,
    )
    require(isinstance(write_rendered, str), "write chat template must render text")
    require(isinstance(generation_prefix_rendered, str), "generation prefix must render text")
    write_ids = _token_ids(tokenizer, write_rendered)
    generation_prefix_ids = _token_ids(tokenizer, generation_prefix_rendered)
    sample = {
        "source_index": candidate.source.line_index,
        "row_sha256": candidate.source.row_sha256,
        "messages": messages[:2],
        "gold_content": messages[2]["content"],
    }
    decision_features = rendered_scene_decision_features(tokenizer, sample, "cpu")
    canonical_read_ids = [int(value) for value in decision_features["input_ids"][0].tolist()]
    semantic_positions = [int(value) for value in decision_features["selected_positions"]]
    semantic_token_ids = [canonical_read_ids[position] for position in semantic_positions]
    require(
        canonical_read_ids[: len(generation_prefix_ids)] == generation_prefix_ids,
        "canonical read prefix differs from state-only generation prefix",
    )
    candidate.token_metadata = {
        "write_rendered_sha256": sha256_text(write_rendered),
        "write_input_ids_sha256": canonical_sha256(write_ids),
        "write_token_count": len(write_ids),
        "generation_prefix_rendered_sha256": sha256_text(generation_prefix_rendered),
        "generation_prefix_input_ids_sha256": canonical_sha256(generation_prefix_ids),
        "generation_prefix_token_count": len(generation_prefix_ids),
        "gold_content_sha256": sha256_text(messages[2]["content"]),
        "canonical_read_rendered_sha256": sha256_text(decision_features["rendered"]),
        "canonical_read_input_ids_sha256": canonical_sha256(canonical_read_ids),
        "canonical_read_token_count": len(canonical_read_ids),
        "semantic_target_positions": semantic_positions,
        "semantic_target_token_ids": semantic_token_ids,
        "semantic_target_mask_sha256": canonical_sha256(
            [index in set(semantic_positions) for index in range(len(canonical_read_ids))]
        ),
    }
    candidate.decision_features = decision_features


def _edge_key(left: Candidate, right: Candidate) -> tuple[str, str]:
    return tuple(sorted((left.source.row_sha256, right.source.row_sha256)))


def optimize_pairing(
    rows: Sequence[Candidate],
    *,
    directed_quotas: Mapping[str, int],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    require(len(rows) % 2 == 0 and rows, "pairing requires a positive even row count")
    require(all(value >= 0 and value % 2 == 0 for value in directed_quotas.values()), "directed pairing quotas must be non-negative even counts")
    require(sum(directed_quotas.values()) == len(rows), "directed pairing quotas must cover every row")
    require(all(row.token_metadata is not None for row in rows), "pairing requires token metadata")
    indices = range(len(rows))
    all_edges = [
        (left, right)
        for left in indices
        for right in range(left + 1, len(rows))
        if rows[left].label_sha256 != rows[right].label_sha256
    ]
    edge_ranks = {
        edge: rank
        for rank, edge in enumerate(
            sorted(all_edges, key=lambda edge: (_edge_key(rows[edge[0]], rows[edge[1]]), edge))
        )
    }
    tie_scale = (len(rows) // 2) * (len(all_edges) + 1) + 1

    def delta(edge: tuple[int, int]) -> int:
        left_count = int(rows[edge[0]].token_metadata["write_token_count"])
        right_count = int(rows[edge[1]].token_metadata["write_token_count"])
        return abs(left_count - right_count)

    cross_edges = [
        edge
        for edge in all_edges
        if target_stratum(rows[edge[0]], rows[edge[1]]) == "cross_cardinality_value"
    ]
    cross_pair_count = directed_quotas["cross_cardinality_value"] // 2
    cross_choices: Iterable[tuple[tuple[int, int], ...]]
    if cross_pair_count == 0:
        cross_choices = [tuple()]
    else:
        cross_choices = itertools.combinations(cross_edges, cross_pair_count)
    best_pairs: list[tuple[int, int]] | None = None
    best_objective: tuple[int, tuple[tuple[str, str], ...]] | None = None
    feasible_cross_choices = 0
    evaluated_cross_choices = 0
    for cross_choice in cross_choices:
        evaluated_cross_choices += 1
        cross_nodes = [node for edge in cross_choice for node in edge]
        if len(set(cross_nodes)) != len(cross_nodes):
            continue
        remaining = [index for index in indices if index not in set(cross_nodes)]
        graph = nx.Graph()
        graph.add_nodes_from(remaining)
        for left_position, left in enumerate(remaining):
            for right in remaining[left_position + 1 :]:
                if rows[left].label_sha256 == rows[right].label_sha256:
                    continue
                stratum = target_stratum(rows[left], rows[right])
                if stratum not in {"presence", "same_cardinality_value"}:
                    continue
                edge = (min(left, right), max(left, right))
                graph.add_edge(
                    left,
                    right,
                    weight=delta(edge) * tie_scale + edge_ranks[edge],
                )
        matching = nx.min_weight_matching(graph, weight="weight")
        if len(matching) * 2 != len(remaining):
            continue
        remainder_pairs = [tuple(sorted((int(left), int(right)))) for left, right in matching]
        pairs = sorted([*cross_choice, *remainder_pairs])
        strata = Counter(target_stratum(rows[left], rows[right]) for left, right in pairs)
        directed_counts = {stratum: strata[stratum] * 2 for stratum in TARGET_STRATA}
        if directed_counts != dict(directed_quotas):
            continue
        feasible_cross_choices += 1
        pair_keys = tuple(sorted(_edge_key(rows[left], rows[right]) for left, right in pairs))
        objective = (sum(delta(edge) for edge in pairs), pair_keys)
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_pairs = pairs
    require(best_pairs is not None and best_objective is not None, "no exact-quota label-distinct pairing exists")
    return best_pairs, {
        "directed_quotas": dict(directed_quotas),
        "total_write_token_delta": best_objective[0],
        "max_write_token_delta": max(delta(edge) for edge in best_pairs),
        "optimization": "enumerate_cross_edges_then_exact_min_weight_perfect_matching_v1",
        "evaluated_cross_choices": evaluated_cross_choices,
        "feasible_cross_choices": feasible_cross_choices,
        "global_minimum_after_exact_quotas": True,
    }


def _entry_for_direction(
    *,
    rows: Sequence[Candidate],
    source_ordinal: int,
    donor_ordinal: int,
) -> dict[str, Any]:
    source = rows[source_ordinal]
    donor = rows[donor_ordinal]
    require(source.token_metadata is not None and donor.token_metadata is not None, "pair metadata requires tokens")
    require(source.decision_features is not None and donor.decision_features is not None, "pair metadata requires decision features")
    pair_target = first_pair_distinguishing_scene_target(
        source_features=source.decision_features,
        donor_features=donor.decision_features,
        donor_sample={
            "source_index": donor.source.line_index,
            "row_sha256": donor.source.row_sha256,
        },
    )
    selected_positions = [int(value) for value in pair_target["selected_target_positions"]]
    entry = {
        "train_row_ordinal": source_ordinal,
        "donor_train_row_ordinal": donor_ordinal,
        "official_source_index": source.source.line_index,
        "donor_official_source_index": donor.source.line_index,
        "source_row_sha256": source.source.row_sha256,
        "donor_row_sha256": donor.source.row_sha256,
        "source_label_sha256": source.label_sha256,
        "donor_label_sha256": donor.label_sha256,
        "source_base_record_sha256": source.base_record.raw_record_sha256,
        "donor_base_record_sha256": donor.base_record.raw_record_sha256,
        "source_strict_failure_stratum": source.failure_stratum,
        "donor_strict_failure_stratum": donor.failure_stratum,
        "source_strict_score_sha256": canonical_sha256(source.strict_score),
        "donor_strict_score_sha256": canonical_sha256(donor.strict_score),
        "source_boundary_count": source.boundary_count,
        "donor_boundary_count": donor.boundary_count,
        "target_stratum": target_stratum(source, donor),
        "source_generation_prefix_sha256": source.token_metadata[
            "generation_prefix_input_ids_sha256"
        ],
        "donor_generation_prefix_sha256": donor.token_metadata[
            "generation_prefix_input_ids_sha256"
        ],
        "source_write_sha256": source.token_metadata["write_input_ids_sha256"],
        "donor_write_sha256": donor.token_metadata["write_input_ids_sha256"],
        "source_write_token_count": source.token_metadata["write_token_count"],
        "donor_write_token_count": donor.token_metadata["write_token_count"],
        "write_token_count_delta": abs(
            int(source.token_metadata["write_token_count"])
            - int(donor.token_metadata["write_token_count"])
        ),
        "first_differing_semantic_ordinal": pair_target[
            "first_differing_semantic_ordinal"
        ],
        "selected_target_positions": selected_positions,
        "selected_target_predictor_positions": [position - 1 for position in selected_positions],
        "selected_target_token_ids": pair_target["selected_target_token_ids"],
        "donor_target_token_ids": pair_target["donor_target_token_ids"],
        "causal_prefix_sha256": pair_target["causal_prefix_sha256"],
    }
    return with_self_hash(entry, field="entry_sha256")


def build_pair_manifest(
    *,
    rows: Sequence[Candidate],
    pairs: Sequence[tuple[int, int]],
    dataset_path: Path,
    directed_quotas: Mapping[str, int],
    optimization: Mapping[str, Any],
) -> dict[str, Any]:
    directed = []
    for left, right in pairs:
        directed.append(_entry_for_direction(rows=rows, source_ordinal=left, donor_ordinal=right))
        directed.append(_entry_for_direction(rows=rows, source_ordinal=right, donor_ordinal=left))
    directed.sort(key=lambda entry: int(entry["train_row_ordinal"]))
    require(
        [int(entry["train_row_ordinal"]) for entry in directed] == list(range(len(rows))),
        "directed pairing does not cover emitted row order",
    )
    payload = {
        "schema": PAIRING_SCHEMA,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "rows": len(rows),
            "ordered_row_sha256": canonical_sha256(
                [row.source.row_sha256 for row in rows]
            ),
        },
        "quotas": dict(directed_quotas),
        "optimization": dict(optimization),
        "directed_pairs": directed,
        "entries_sha256": canonical_sha256(directed),
    }
    return with_self_hash(payload)


def row_manifest(candidate: Candidate, *, train_row_ordinal: int) -> dict[str, Any]:
    require(candidate.token_metadata is not None, "row manifest requires token metadata")
    score = dict(candidate.strict_score)
    record = {
        "schema": ROW_SCHEMA,
        "train_row_ordinal": train_row_ordinal,
        "official_source_index": candidate.source.line_index,
        "source_split": "train",
        "row_sha256": candidate.source.row_sha256,
        "prompt_sha256": candidate.source.prompt_sha256,
        "label_sha256": candidate.label_sha256,
        "gold_boundaries": score["gold_boundaries"],
        "gold_boundary_count": candidate.boundary_count,
        "paragraph_hashes": list(candidate.paragraph_hashes),
        "paragraph_hashes_sha256": canonical_sha256(list(candidate.paragraph_hashes)),
        "selection_sha256": candidate.selection_sha256,
        "base_record_sha256": candidate.base_record.raw_record_sha256,
        "base_record_key": candidate.base_record.key,
        "base_producer_fingerprint": candidate.base_record.producer_fingerprint,
        "strict_score": score,
        "producer_stored_score": candidate.base_payload.get("score"),
        "producer_score_matches_current_strict": (
            candidate.base_payload.get("score") == score
        ),
        "strict_failure_stratum": candidate.failure_stratum,
        "token_metadata": candidate.token_metadata,
    }
    return with_self_hash(record, field="record_sha256")


def pairing_binding(pair_path: Path, pair_manifest: Mapping[str, Any]) -> dict[str, Any]:
    dataset = pair_manifest["dataset"]
    return {
        "schema": PAIRING_BINDING_SCHEMA,
        "pair_manifest": {
            "path": str(pair_path.resolve()),
            "sha256": sha256_file(pair_path),
            "manifest_sha256": pair_manifest["manifest_sha256"],
        },
        "dataset_sha256": dataset["sha256"],
        "directed_entry_count": len(pair_manifest["directed_pairs"]),
        "quotas": dict(pair_manifest["quotas"]),
        "entries_sha256": pair_manifest["entries_sha256"],
    }


def build_source_manifest(
    *,
    dataset_path: Path,
    row_manifest_path: Path,
    pair_path: Path,
    pair_manifest: Mapping[str, Any],
    rows: int,
    purpose: str,
    parent_train32_sha256: str | None,
) -> dict[str, Any]:
    payload = {
        "schema": SOURCE_SCHEMA,
        "task": TASK,
        "purpose": purpose,
        "contract": {
            "source_split": "train",
            "val_rows": 0,
            "test_rows": 0,
            "episode_contract": {
                "episode_recent_messages": 0,
                "write_phase": "system + user",
                "read_supervision": "system + assistant",
            },
        },
        "partitions": {
            "train": {
                "source_split": "train",
                "rows": rows,
                "data": {
                    "path": str(dataset_path.resolve()),
                    "sha256": sha256_file(dataset_path),
                },
                "row_manifest": {
                    "path": str(row_manifest_path.resolve()),
                    "sha256": sha256_file(row_manifest_path),
                },
            }
        },
        "v7_pairing": pairing_binding(pair_path, pair_manifest),
        "parent_train32_sha256": parent_train32_sha256,
    }
    return with_self_hash(payload)


def _tiny_pair(
    rows: Sequence[Candidate], pairs: Sequence[tuple[int, int]]
) -> tuple[Candidate, Candidate]:
    choices = []
    for left, right in pairs:
        left_row = rows[left]
        right_row = rows[right]
        if (
            left_row.boundary_count <= 0
            or left_row.boundary_count != right_row.boundary_count
            or left_row.label_sha256 == right_row.label_sha256
        ):
            continue
        severity = sum(
            int(row.strict_score["fp"])
            + int(row.strict_score["fn"])
            + int(not row.strict_score["schema_valid"])
            for row in (left_row, right_row)
        )
        choices.append(
            (
                -severity,
                _edge_key(left_row, right_row),
                left_row,
                right_row,
            )
        )
    require(bool(choices), "Train32 pairing contains no positive same-cardinality pair")
    _, _, left, right = min(choices, key=lambda item: (item[0], item[1]))
    return left, right


def _ensure_fresh_output(output_dir: Path, *, overwrite: bool) -> None:
    filenames = (
        "train32.jsonl",
        "train32_rows.jsonl",
        "train32_pair_manifest.json",
        "train32_source_manifest.json",
        "tiny2.jsonl",
        "tiny2_rows.jsonl",
        "tiny2_pair_manifest.json",
        "tiny2_source_manifest.json",
        "manifest.json",
    )
    existing = [output_dir / filename for filename in filenames if (output_dir / filename).exists()]
    require(overwrite or not existing, "V7 output already exists: " + ", ".join(map(str, existing)))
    if overwrite:
        for path in existing:
            require(path.is_file() and not path.is_symlink(), f"refusing to replace non-regular output: {path}")
            path.unlink()


def prepare_v7_data(
    *,
    output_dir: Path,
    tokenizer_path: Path,
    train_file: Path = TRAIN_FILE,
    val_file: Path = VAL_FILE,
    test_file: Path = TEST_FILE,
    base_train_records: Path = BASE_TRAIN_RECORDS,
    hard32_file: Path = HARD32_FILE,
    hard32_selection: Path = HARD32_SELECTION,
    hard32_base_records: Path = HARD32_BASE_RECORDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    _ensure_fresh_output(output_dir, overwrite=overwrite)
    target_matrix, hard_failure_counts, hard_paragraphs = _hard32_contract(
        hard32_file,
        hard32_selection,
        hard32_base_records,
    )
    candidates, pool_audit = load_candidate_pool(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        base_records_path=base_train_records,
        hard_paragraph_hashes=hard_paragraphs,
    )
    selected, selection_audit = select_train32(candidates, target_matrix)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path.expanduser().resolve(),
        local_files_only=True,
    )
    for candidate in selected:
        materialize_token_metadata(candidate, tokenizer)

    train_path = output_dir / "train32.jsonl"
    train_rows_path = output_dir / "train32_rows.jsonl"
    train_pair_path = output_dir / "train32_pair_manifest.json"
    train_source_path = output_dir / "train32_source_manifest.json"
    atomic_write_text(
        train_path,
        "".join(candidate.source.raw_line + "\n" for candidate in selected),
    )
    write_jsonl(
        train_rows_path,
        (row_manifest(candidate, train_row_ordinal=index) for index, candidate in enumerate(selected)),
    )
    pairs, pairing_audit = optimize_pairing(
        selected,
        directed_quotas=TRAIN_DIRECTED_PAIR_QUOTAS,
    )
    train_pair_manifest = build_pair_manifest(
        rows=selected,
        pairs=pairs,
        dataset_path=train_path,
        directed_quotas=TRAIN_DIRECTED_PAIR_QUOTAS,
        optimization=pairing_audit,
    )
    write_json(train_pair_path, train_pair_manifest)
    train_source_manifest = build_source_manifest(
        dataset_path=train_path,
        row_manifest_path=train_rows_path,
        pair_path=train_pair_path,
        pair_manifest=train_pair_manifest,
        rows=32,
        purpose="fixed_hard32_aligned_real_training",
        parent_train32_sha256=None,
    )
    write_json(train_source_path, train_source_manifest)

    tiny_left, tiny_right = _tiny_pair(selected, pairs)
    tiny_rows = [tiny_left, tiny_right]
    tiny_path = output_dir / "tiny2.jsonl"
    tiny_rows_path = output_dir / "tiny2_rows.jsonl"
    tiny_pair_path = output_dir / "tiny2_pair_manifest.json"
    tiny_source_path = output_dir / "tiny2_source_manifest.json"
    atomic_write_text(
        tiny_path,
        "".join(candidate.source.raw_line + "\n" for candidate in tiny_rows),
    )
    write_jsonl(
        tiny_rows_path,
        (row_manifest(candidate, train_row_ordinal=index) for index, candidate in enumerate(tiny_rows)),
    )
    tiny_pairs, tiny_pairing_audit = optimize_pairing(
        tiny_rows,
        directed_quotas=TINY_DIRECTED_PAIR_QUOTAS,
    )
    tiny_pair_manifest = build_pair_manifest(
        rows=tiny_rows,
        pairs=tiny_pairs,
        dataset_path=tiny_path,
        directed_quotas=TINY_DIRECTED_PAIR_QUOTAS,
        optimization={
            **tiny_pairing_audit,
            "selection": "maximum_combined_strict_error_positive_same_cardinality_pair_v1",
        },
    )
    write_json(tiny_pair_path, tiny_pair_manifest)
    tiny_source_manifest = build_source_manifest(
        dataset_path=tiny_path,
        row_manifest_path=tiny_rows_path,
        pair_path=tiny_pair_path,
        pair_manifest=tiny_pair_manifest,
        rows=2,
        purpose="hardest_positive_same_cardinality_tiny_overfit_preflight",
        parent_train32_sha256=sha256_file(train_path),
    )
    write_json(tiny_source_path, tiny_source_manifest)

    artifacts = {}
    for name, path in (
        ("train32", train_path),
        ("train32_rows", train_rows_path),
        ("train32_pair_manifest", train_pair_path),
        ("train32_source_manifest", train_source_path),
        ("tiny2", tiny_path),
        ("tiny2_rows", tiny_rows_path),
        ("tiny2_pair_manifest", tiny_pair_path),
        ("tiny2_source_manifest", tiny_source_path),
    ):
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = with_self_hash(
        {
            "schema": SCHEMA,
            "task": TASK,
            "fixed_hard32": {
                "data": {"path": str(hard32_file.resolve()), "sha256": HARD32_FILE_SHA256},
                "selection": {
                    "path": str(hard32_selection.resolve()),
                    "sha256": HARD32_SELECTION_SHA256,
                },
                "base_records": {
                    "path": str(hard32_base_records.resolve()),
                    "sha256": HARD32_BASE_RECORDS_SHA256,
                },
                "gold_cardinality_quotas": dict(GOLD_CARDINALITY_QUOTAS),
                "strict_failure_counts": hard_failure_counts,
            },
            "base_train_pool": {
                "records": {
                    "path": str(base_train_records.resolve()),
                    "sha256": BASE_TRAIN_RECORDS_SHA256,
                },
                "scope": "complete_predeclared_candidate64",
                "full_train_base_producer_available": False,
                **pool_audit,
            },
            "selection": selection_audit,
            "pairing": pairing_audit,
            "leakage": {
                "normalization": "anchored paragraphs; Unicode NFKC; remove Unicode whitespace",
                "train32_hard32_shared_normalized_paragraphs": 0,
                "val_rows_in_training": 0,
                "test_rows_in_training": 0,
            },
            "tokenizer": {
                "path": str(tokenizer_path.expanduser().resolve()),
                "tokenizer_json_sha256": sha256_file(
                    tokenizer_path.expanduser().resolve() / "tokenizer.json"
                ),
                "chat_template_sha256": sha256_file(
                    tokenizer_path.expanduser().resolve() / "chat_template.jinja"
                ),
            },
            "artifacts": artifacts,
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--base-train-records", type=Path, default=BASE_TRAIN_RECORDS)
    parser.add_argument("--hard32-base-records", type=Path, default=HARD32_BASE_RECORDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_v7_data(
            output_dir=args.output_dir,
            tokenizer_path=args.tokenizer_path,
            base_train_records=args.base_train_records.expanduser().resolve(),
            hard32_base_records=args.hard32_base_records.expanduser().resolve(),
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "train32_sha256": manifest["artifacts"]["train32"]["sha256"],
                "tiny2_sha256": manifest["artifacts"]["tiny2"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
