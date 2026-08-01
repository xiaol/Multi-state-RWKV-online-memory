#!/usr/bin/env python3
"""Build a train-only, benchmark-shaped scene failure curriculum.

The curriculum is mined exclusively from official ``scene-v4-current`` TRAIN
rows that a pinned frozen-base evaluation fails under the benchmark's strict
scorer.  Protected validation/Hard32 and test identities are recorded as
opaque SHA-256 exclusions; this builder accepts no path for, and never opens,
those artifacts.

Each emitted row keeps the source JSONL bytes and exact three-message
``[system, user, assistant]`` schema.  Rows are paired reciprocally in two
equally represented strata:

* ``presence``: one empty and one non-empty gold boundary list; and
* ``same_cardinality_value``: two non-empty, same-cardinality, distinct labels.

Pair selection minimizes write-token length differences, then prefers larger
strict base errors, with a hash-only tie-break.  The output pair/source schemas
remain V7-compatible so the existing scene-state trainer can consume the new
data without translating its identity metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma.prepare_scene_failure_pairs import (
    BaseRecord,
    SourceRow,
    join_train_records,
    load_base_records,
    load_source_split,
    sha256_file,
    sha256_text,
)
from experiments.rethinking_rwkv_ms_gemma.prepare_scene_memory_v7_data import (
    Candidate,
    ContractError,
    atomic_write_text,
    build_pair_manifest,
    canonical_sha256,
    load_json_object,
    materialize_token_metadata,
    normalized_paragraph_hashes,
    pairing_binding,
    read_jsonl,
    require,
    row_manifest,
    strict_failure_stratum,
    with_self_hash,
    write_json,
    write_jsonl,
)
from experiments.rethinking_rwkv_ms_gemma.run_novel_agent_eval import (
    score_prediction,
)


SCHEMA = "rwkv_ms_scene_hard_failure_curriculum.v1"
CURRICULUM_SCHEMA = "rwkv_ms_scene_hard_failure_pairing.v1"
PAIR_SCHEDULE_SCHEMA = "rwkv_ms_scene_hard_failure_pair_schedule.v1"
PAIR_SCHEDULE_ENTRY_SCHEMA = "rwkv_ms_scene_hard_failure_pair_schedule_entry.v1"
PAIR_SCHEDULE_MANIFEST_SCHEMA = (
    "rwkv_ms_scene_hard_failure_pair_schedule_manifest.v1"
)
PAIR_CURRICULUM_BINDING_SCHEMA = (
    "rwkv_ms_scene_hard_failure_pair_curriculum_binding.v1"
)
TASK = "scene-v4-current"
SELECTION_NAMESPACE = "rwkv_ms_scene_hard_failure_curriculum_selection.v1"
PAIR_RANK_NAMESPACE = "rwkv_ms_scene_hard_failure_curriculum_pair.v1"
PAIR_SHUFFLE_NAMESPACE = "rwkv_ms_scene_hard_failure_four_cycle_shuffle.v1"
SOURCE_SCHEMA = "rwkv_ms_scene_memory_v7_source.v1"
PAIRING_SCHEMA = "rwkv_ms_scene_memory_v7_pairing.v1"

DEFAULT_MAX_PAIRS = 16
PAIR_CYCLES = 4
GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_DATASET_ROOT = Path(
    "/run/media/xiaol/B214449214445C0B/datasets/novel-agent-sft-dataset"
)
DEFAULT_TRAIN_FILE = (
    DEFAULT_DATASET_ROOT / "training/v4-scene-boundary-detection/train.jsonl"
)
DEFAULT_BASE_EVAL = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "base_train_seed3407_n64_v1/base.jsonl"
)
DEFAULT_TOKENIZER_PATH = Path(
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"
)
DEFAULT_OUTPUT_DIR = Path(
    "/run/media/xiaol/B214449214445C0B/delta_mem_data/scene_failure_state/"
    "scene_hard_failure_curriculum_base64_pairs16_v1"
)

TRAIN_FILE_SHA256 = "785fe54c0a4e5c64e33f64f9bc88d64719576407c21eb0d520f9dec5a59b8e22"
VAL_FILE_SHA256 = "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513"
TEST_FILE_SHA256 = "d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d"
HARD32_FILE_SHA256 = "b5b1137de89f82eee4b3ae3e3c7b5305240699ec7b65e84b61cb415a7a000d4a"
HARD32_SELECTION_SHA256 = (
    "76d510e5f02c30f2cf3a0262cc4c97d69ef8f52861e9ced855d530b233d916db"
)

BASE_EVAL_SHA256 = "853d52aca502e431479a07ac62b5973354720a9c623cd4109ec2332059d52b18"
BASE_MANIFEST_SHA256 = "44c9836cd8433cf352ff606b3fa36e9fc7c7453ed51d3b8b3bc7727770bb9eeb"
BASE_SELECTION_SHA256 = "0fb9595ee8587508390157189dc49184ff024cbc37590a4bb0ffbf3be85717b5"
BASE_SUMMARY_SHA256 = "51cf7d902a20f92dbea33b456b88a6a49a53ed9ce474dda3196a04797cde3c87"
TOKENIZER_JSON_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
CHAT_TEMPLATE_SHA256 = "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082"

PRODUCER_SCHEMA = "rwkv_ms_scene_train_base_eval.v1"
PRODUCER_SELECTION_SCHEMA = "rwkv_ms_scene_train_base_selection.v1"
ARTIFACT_FILENAMES = {
    "train": "train.jsonl",
    "rows": "train_rows.jsonl",
    "pair_manifest": "pair_manifest.json",
    "pair_schedule": "pair_schedule.jsonl",
    "pair_schedule_manifest": "pair_schedule_manifest.json",
    "source_manifest": "source_manifest.json",
    "bundle_manifest": "manifest.json",
}


def _verify_sha256(value: str, *, description: str) -> str:
    normalized = str(value).lower()
    require(
        len(normalized) == 64
        and all(character in "0123456789abcdef" for character in normalized),
        f"{description} must be a lowercase SHA-256",
    )
    return normalized


def _verify_regular_file(
    path: Path,
    expected_sha256: str,
    *,
    description: str,
) -> Path:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"missing or non-regular {description}: {resolved}",
    )
    expected = _verify_sha256(expected_sha256, description=f"{description} hash")
    actual = sha256_file(resolved)
    require(
        actual == expected,
        f"{description} SHA-256 differs: expected={expected} actual={actual}",
    )
    return resolved


def _declared_path(
    value: Any,
    *,
    relative_to: Path,
    description: str,
) -> Path:
    require(isinstance(value, str) and bool(value), f"{description} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to.parent / path
    return path.resolve()


def protected_evaluation_bindings() -> dict[str, Any]:
    """Return hash-only exclusions; no protected locator enters the builder."""

    return {
        "policy": "hash_bound_exclusion_only; never_resolved_opened_or_parsed",
        "official_validation": {
            "included": False,
            "path": None,
            "rows": 170,
            "sha256": VAL_FILE_SHA256,
        },
        "hard32": {
            "included": False,
            "path": None,
            "rows": 32,
            "data_sha256": HARD32_FILE_SHA256,
            "selection_sha256": HARD32_SELECTION_SHA256,
        },
        "official_test": {
            "included": False,
            "path": None,
            "rows": 149,
            "sha256": TEST_FILE_SHA256,
        },
    }


def _base_payloads_by_hash(path: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for raw_line, payload in read_jsonl(path):
        row_hash = payload.get("row_sha256")
        require(
            isinstance(row_hash, str) and row_hash not in payloads,
            "base-eval row_sha256 is invalid or duplicated",
        )
        copied = dict(payload)
        copied["_raw_record_sha256"] = sha256_text(raw_line)
        payloads[row_hash] = copied
    return payloads


def validate_pinned_base_bundle(
    *,
    train_file: Path,
    train_rows: Sequence[SourceRow],
    base_eval: Path,
    expected_base_eval_sha256: str,
    expected_base_manifest_sha256: str,
    expected_base_selection_sha256: str,
    expected_base_summary_sha256: str,
) -> tuple[list[tuple[SourceRow, BaseRecord]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate a producer-managed base run without fixing its candidate count."""

    base_eval = _verify_regular_file(
        base_eval,
        expected_base_eval_sha256,
        description="frozen-base evaluation",
    )
    require(
        base_eval.name == "base.jsonl",
        "frozen-base evaluation must be the producer-managed base.jsonl",
    )
    manifest_path = _verify_regular_file(
        base_eval.parent / "manifest.json",
        expected_base_manifest_sha256,
        description="base producer manifest",
    )
    selection_path = _verify_regular_file(
        base_eval.parent / "candidate_selection.json",
        expected_base_selection_sha256,
        description="base producer selection",
    )
    summary_path = _verify_regular_file(
        base_eval.parent / "summary.json",
        expected_base_summary_sha256,
        description="base producer summary",
    )
    manifest = load_json_object(manifest_path, description="base producer manifest")
    selection = load_json_object(selection_path, description="base producer selection")
    summary = load_json_object(summary_path, description="base producer summary")

    require(manifest.get("schema") == PRODUCER_SCHEMA, "base producer schema differs")
    require(
        selection.get("schema") == PRODUCER_SELECTION_SCHEMA,
        "base producer selection schema differs",
    )
    selection_rows = selection.get("rows")
    candidate_count = selection.get("candidate_count")
    require(
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count > 0,
        "base producer candidate_count must be positive",
    )
    require(
        isinstance(selection_rows, list) and len(selection_rows) == candidate_count,
        "base producer selection row count differs",
    )
    expected_selection_fields = {
        "task": TASK,
        "split": "train",
        "selection_uses_gold_labels": False,
        "selection_uses_model_output": False,
        "selection_basis": "sha256(selection_seed + NUL + user_prompt_sha256)",
    }
    require(
        all(selection.get(key) == value for key, value in expected_selection_fields.items()),
        "base producer selection protocol differs",
    )
    require(
        selection.get("dataset_sha256") == sha256_file(train_file),
        "base producer selection binds a different train dataset hash",
    )
    require(
        _declared_path(
            selection.get("dataset_file"),
            relative_to=selection_path,
            description="base producer selection dataset",
        )
        == train_file,
        "base producer selection binds a different train dataset path",
    )

    train_by_index = {row.line_index: row for row in train_rows}
    expected_selected: dict[int, tuple[str, str]] = {}
    for ordinal, item in enumerate(selection_rows):
        require(isinstance(item, dict), f"selection row {ordinal} must be an object")
        source_index = item.get("source_index")
        require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and source_index not in expected_selected,
            f"selection row {ordinal} source_index is invalid or duplicated",
        )
        source = train_by_index.get(source_index)
        require(source is not None, f"selection row {ordinal} is outside official TRAIN")
        expected = (source.row_sha256, source.prompt_sha256)
        actual = (item.get("row_sha256"), item.get("user_prompt_sha256"))
        require(actual == expected, f"selection row {ordinal} source binding differs")
        expected_selected[source_index] = expected

    records, total_records = load_base_records(base_eval, task_name=TASK)
    require(
        total_records == len(records) == candidate_count,
        "base.jsonl must contain exactly the dynamically declared candidate set",
    )
    joined = join_train_records(list(train_rows), records)
    actual_selected = {
        source.line_index: (source.row_sha256, source.prompt_sha256)
        for source, _ in joined
    }
    require(
        actual_selected == expected_selected,
        "base.jsonl rows differ from the producer selection",
    )

    fingerprint_payload = manifest.get("fingerprint_payload")
    fingerprint = manifest.get("fingerprint")
    require(isinstance(fingerprint_payload, dict), "base producer fingerprint payload is missing")
    require(
        fingerprint == canonical_sha256(fingerprint_payload),
        "base producer fingerprint differs from its canonical payload",
    )
    expected_fingerprint_fields = {
        "schema": PRODUCER_SCHEMA,
        "task": TASK,
        "task_kind": "scene",
        "condition": "base",
        "split": "train",
        "candidate_count": candidate_count,
        "dataset_sha256": sha256_file(train_file),
        "selection_sha256": canonical_sha256(selection),
        "selected_rows": selection_rows,
        "selection_seed": selection.get("selection_seed"),
    }
    require(
        all(fingerprint_payload.get(key) == value for key, value in expected_fingerprint_fields.items()),
        "base producer fingerprint protocol differs",
    )
    require(
        _declared_path(
            fingerprint_payload.get("dataset_file"),
            relative_to=manifest_path,
            description="base producer fingerprint dataset",
        )
        == train_file,
        "base producer fingerprint binds a different TRAIN path",
    )
    record_fingerprints = {record.producer_fingerprint for record in records}
    require(
        record_fingerprints == {fingerprint},
        "base records do not share the producer fingerprint",
    )

    manifest_selection = manifest.get("selection")
    manifest_output = manifest.get("output")
    require(isinstance(manifest_selection, dict), "base producer manifest selection is missing")
    require(isinstance(manifest_output, dict), "base producer manifest output is missing")
    require(
        _declared_path(
            manifest_selection.get("path"),
            relative_to=manifest_path,
            description="base producer manifest selection",
        )
        == selection_path
        and manifest_selection.get("sha256") == sha256_file(selection_path)
        and manifest_selection.get("rows") == candidate_count
        and manifest_selection.get("uses_gold_labels") is False
        and manifest_selection.get("uses_model_output") is False,
        "base producer manifest selection binding differs",
    )
    require(
        _declared_path(
            manifest_output.get("base_records"),
            relative_to=manifest_path,
            description="base producer manifest records",
        )
        == base_eval,
        "base producer manifest points to a different base.jsonl",
    )
    expected_summary = {
        "schema": PRODUCER_SCHEMA,
        "fingerprint": fingerprint,
        "complete": True,
        "completed": candidate_count,
        "expected": candidate_count,
        "condition": "base",
        "task": TASK,
        "split": "train",
    }
    require(
        all(summary.get(key) == value for key, value in expected_summary.items()),
        "base producer summary is incomplete or differs",
    )
    payloads = _base_payloads_by_hash(base_eval)
    require(set(payloads) == {record.row_sha256 for record in records}, "base payload binding differs")
    return joined, payloads, {
        "schema": PRODUCER_SCHEMA,
        "candidate_count": candidate_count,
        "fingerprint": fingerprint,
        "selection_seed": selection.get("selection_seed"),
        "selection_uses_gold_labels": False,
        "selection_uses_model_output": False,
        "artifacts": {
            "base_eval": {"path": str(base_eval), "sha256": sha256_file(base_eval)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "selection": {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }


def build_failure_candidates(
    joined: Sequence[tuple[SourceRow, BaseRecord]],
    base_payloads: Mapping[str, dict[str, Any]],
) -> tuple[list[Candidate], dict[str, Any]]:
    failures: list[Candidate] = []
    strict_successes = 0
    for source, record in joined:
        strict_score = score_prediction("scene", record.parsed_json, source.gold)
        failure_stratum = strict_failure_stratum(strict_score)
        if failure_stratum is None:
            strict_successes += 1
            continue
        boundaries = list(strict_score["gold_boundaries"])
        failures.append(
            Candidate(
                source=source,
                base_record=record,
                base_payload=base_payloads[source.row_sha256],
                strict_score=strict_score,
                failure_stratum=failure_stratum,
                boundary_count=len(boundaries),
                label_sha256=canonical_sha256(boundaries),
                paragraph_hashes=normalized_paragraph_hashes(
                    source.messages[1]["content"]
                ),
                selection_sha256=sha256_text(
                    f"{SELECTION_NAMESPACE}\0{source.prompt_sha256}"
                ),
            )
        )
    failures.sort(key=lambda candidate: candidate.source.line_index)
    require(failures, "pinned frozen-base evaluation contains no strict TRAIN failures")
    return failures, {
        "evaluated_candidates": len(joined),
        "eligible_strict_failures": len(failures),
        "strict_successes_excluded": strict_successes,
        "failure_strata": dict(sorted(Counter(row.failure_stratum for row in failures).items())),
        "gold_cardinalities": {
            str(key): value
            for key, value in sorted(Counter(row.boundary_count for row in failures).items())
        },
        "criterion": "exact run_novel_agent_eval.score_prediction(scene) strict failure",
    }


def _write_token_count(candidate: Candidate) -> int:
    require(candidate.token_metadata is not None, "pairing requires token metadata")
    value = candidate.token_metadata.get("write_token_count")
    require(isinstance(value, int) and value > 0, "write_token_count is invalid")
    return value


def _strict_error_severity(candidate: Candidate) -> int:
    score = candidate.strict_score
    return int(score.get("fp", 0)) + int(score.get("fn", 0)) + int(
        not bool(score.get("schema_valid"))
    )


def _edge_rank(
    candidates: Sequence[Candidate],
    left: int,
    right: int,
) -> tuple[int, int, str, int, int]:
    left_hash, right_hash = sorted(
        (
            candidates[left].source.row_sha256,
            candidates[right].source.row_sha256,
        )
    )
    return (
        abs(_write_token_count(candidates[left]) - _write_token_count(candidates[right])),
        -(
            _strict_error_severity(candidates[left])
            + _strict_error_severity(candidates[right])
        ),
        sha256_text(f"{PAIR_RANK_NAMESPACE}\0{left_hash}\0{right_hash}"),
        min(candidates[left].source.line_index, candidates[right].source.line_index),
        max(candidates[left].source.line_index, candidates[right].source.line_index),
    )


def _same_cardinality_capacity(
    candidates: Sequence[Candidate],
    indices: Iterable[int],
) -> int:
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for index in indices:
        candidate = candidates[index]
        if candidate.boundary_count > 0:
            grouped[candidate.boundary_count][candidate.label_sha256] += 1
    capacity = 0
    for labels in grouped.values():
        total = sum(labels.values())
        capacity += min(total // 2, total - max(labels.values()))
    return capacity


def _feasible_same_edges(
    candidates: Sequence[Candidate],
    remaining: set[int],
    *,
    cardinality: int | None = None,
) -> list[tuple[tuple[int, int, str, int, int], int, int]]:
    ordered = sorted(remaining)
    edges = []
    for position, left in enumerate(ordered):
        left_row = candidates[left]
        if left_row.boundary_count <= 0 or (
            cardinality is not None and left_row.boundary_count != cardinality
        ):
            continue
        for right in ordered[position + 1 :]:
            right_row = candidates[right]
            if (
                right_row.boundary_count != left_row.boundary_count
                or right_row.label_sha256 == left_row.label_sha256
            ):
                continue
            edges.append((_edge_rank(candidates, left, right), left, right))
    return sorted(edges)


def _select_same_cardinality_pairs(
    candidates: Sequence[Candidate],
    nonempty: set[int],
    *,
    quota: int,
) -> tuple[list[tuple[int, int]], set[int], dict[str, Any]]:
    remaining = set(nonempty)
    pairs: list[tuple[int, int]] = []
    capacities = {
        cardinality: _same_cardinality_capacity(
            candidates,
            [index for index in remaining if candidates[index].boundary_count == cardinality],
        )
        for cardinality in sorted(
            {candidates[index].boundary_count for index in remaining}
        )
    }
    eligible_cards = [cardinality for cardinality, value in capacities.items() if value > 0]
    require(eligible_cards, "no label-distinct non-empty same-cardinality pair exists")

    # Cover every feasible non-empty cardinality when the quota allows it.  If
    # the cap is smaller, choose the cardinalities with the closest first edge.
    if quota >= len(eligible_cards):
        seeded_cards = eligible_cards
    else:
        first_edges = []
        for cardinality in eligible_cards:
            edge = _feasible_same_edges(
                candidates,
                remaining,
                cardinality=cardinality,
            )[0]
            first_edges.append((edge[0], cardinality))
        seeded_cards = [cardinality for _, cardinality in sorted(first_edges)[:quota]]

    for cardinality in seeded_cards:
        required_after = quota - len(pairs) - 1
        selected: tuple[int, int] | None = None
        for _, left, right in _feasible_same_edges(
            candidates,
            remaining,
            cardinality=cardinality,
        ):
            proposed = remaining - {left, right}
            if _same_cardinality_capacity(candidates, proposed) >= required_after:
                selected = (left, right)
                break
        require(selected is not None, f"cannot seed same-cardinality stratum {cardinality}")
        pairs.append(selected)
        remaining.difference_update(selected)

    while len(pairs) < quota:
        required_after = quota - len(pairs) - 1
        selected = None
        for _, left, right in _feasible_same_edges(candidates, remaining):
            proposed = remaining - {left, right}
            if _same_cardinality_capacity(candidates, proposed) >= required_after:
                selected = (left, right)
                break
        require(selected is not None, "cannot complete same-cardinality pair quota")
        pairs.append(selected)
        remaining.difference_update(selected)

    return pairs, remaining, {
        "available_capacity_by_cardinality": {
            str(key): value for key, value in capacities.items()
        },
        "represented_cardinalities": sorted(
            {candidates[left].boundary_count for left, _ in pairs}
        ),
        "selected_pairs_by_cardinality": {
            str(key): value
            for key, value in sorted(
                Counter(candidates[left].boundary_count for left, _ in pairs).items()
            )
        },
    }


def select_balanced_pairs(
    candidates: Sequence[Candidate],
    *,
    max_pairs: int,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Select equal presence and same-cardinality pair counts under a cap."""

    require(
        isinstance(max_pairs, int) and not isinstance(max_pairs, bool) and max_pairs >= 2,
        "max_pairs must be an integer of at least two",
    )
    require(
        all(candidate.token_metadata is not None for candidate in candidates),
        "balanced pairing requires token metadata for every failure candidate",
    )
    empty = {
        index for index, candidate in enumerate(candidates) if candidate.boundary_count == 0
    }
    nonempty = set(range(len(candidates))) - empty
    same_capacity = _same_cardinality_capacity(candidates, nonempty)
    quota = min(
        max_pairs // 2,
        len(empty),
        len(nonempty) // 3,
        same_capacity,
    )
    require(
        quota > 0,
        "candidate pool cannot form one balanced presence/same-cardinality pair block",
    )
    same_pairs, remaining_nonempty, same_audit = _select_same_cardinality_pairs(
        candidates,
        nonempty,
        quota=quota,
    )

    presence_edges = sorted(
        (
            _edge_rank(candidates, left, right),
            left,
            right,
        )
        for left in empty
        for right in remaining_nonempty
    )
    used_empty: set[int] = set()
    used_nonempty: set[int] = set()
    presence_pairs: list[tuple[int, int]] = []
    for _, left, right in presence_edges:
        if left in used_empty or right in used_nonempty:
            continue
        presence_pairs.append((left, right))
        used_empty.add(left)
        used_nonempty.add(right)
        if len(presence_pairs) == quota:
            break
    require(len(presence_pairs) == quota, "cannot complete presence pair quota")
    pairs = [*same_pairs, *presence_pairs]
    used = [index for pair in pairs for index in pair]
    require(len(used) == len(set(used)) == len(pairs) * 2, "pair selection reused a row")

    deltas = [
        abs(_write_token_count(candidates[left]) - _write_token_count(candidates[right]))
        for left, right in pairs
    ]
    return pairs, {
        "max_pairs_cap": max_pairs,
        "selected_pairs": len(pairs),
        "selected_rows": len(used),
        "pairs_per_balanced_stratum": quota,
        "pair_strata": {
            "presence": len(presence_pairs),
            "same_cardinality_value": len(same_pairs),
            "cross_cardinality_value": 0,
        },
        "directed_strata": {
            "presence": len(presence_pairs) * 2,
            "same_cardinality_value": len(same_pairs) * 2,
            "cross_cardinality_value": 0,
        },
        "available_empty_failures": len(empty),
        "available_nonempty_failures": len(nonempty),
        "available_same_cardinality_pair_capacity": same_capacity,
        "same_cardinality": same_audit,
        "write_token_delta": {
            "total": sum(deltas),
            "maximum": max(deltas),
            "ordered": deltas,
        },
        "objective": (
            "exactly balance presence and same-cardinality pair counts; cover feasible "
            "non-empty cardinalities; minimize write-token delta; prefer larger strict "
            "base error; hash-only tie-break"
        ),
    }


def _validate_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    description: str,
) -> str:
    expected = payload.get(field)
    require(
        isinstance(expected, str) and len(expected) == 64,
        f"{description} self-hash is missing",
    )
    unsigned = dict(payload)
    unsigned.pop(field, None)
    require(
        canonical_sha256(unsigned) == expected,
        f"{description} self-hash differs",
    )
    return expected


def _directed_entries_by_ordinal(
    pair_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    require(pair_manifest.get("schema") == PAIRING_SCHEMA, "pair schema differs")
    entries = pair_manifest.get("directed_pairs")
    require(isinstance(entries, list) and entries, "directed pairs are missing")
    by_ordinal: dict[int, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "directed pair entry must be an object")
        _validate_self_hash(
            entry,
            field="entry_sha256",
            description="directed pair entry",
        )
        ordinal = entry.get("train_row_ordinal")
        require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and ordinal not in by_ordinal,
            "directed pair ordinal is invalid or duplicated",
        )
        by_ordinal[ordinal] = dict(entry)
    require(
        sorted(by_ordinal) == list(range(len(by_ordinal))),
        "directed pair ordinals are not dense",
    )
    for ordinal, entry in by_ordinal.items():
        donor = entry.get("donor_train_row_ordinal")
        require(
            isinstance(donor, int)
            and donor in by_ordinal
            and donor != ordinal
            and by_ordinal[donor].get("donor_train_row_ordinal") == ordinal,
            "directed pair reciprocity differs",
        )
    return by_ordinal


def _canonical_pairs(
    directed: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[int, int], ...]:
    pairs = {
        tuple(sorted((ordinal, int(entry["donor_train_row_ordinal"]))))
        for ordinal, entry in directed.items()
    }
    require(
        len(pairs) * 2 == len(directed),
        "canonical pair count differs from directed rows",
    )
    return tuple(sorted(pairs))


def _pair_identity_sha256(
    pair: tuple[int, int],
    directed: Mapping[int, Mapping[str, Any]],
) -> str:
    low, high = pair
    return canonical_sha256(
        {
            "canonical_pair_ordinals": [low, high],
            "source_row_sha256": [
                directed[low]["source_row_sha256"],
                directed[high]["source_row_sha256"],
            ],
            "source_label_sha256": [
                directed[low]["source_label_sha256"],
                directed[high]["source_label_sha256"],
            ],
            "directed_pair_entry_sha256": [
                directed[low]["entry_sha256"],
                directed[high]["entry_sha256"],
            ],
        }
    )


def ordered_pairs_for_cycle(
    canonical_pairs: Sequence[tuple[int, int]],
    *,
    cycle_index: int,
    directed: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[int, int], ...]:
    require(1 <= cycle_index <= PAIR_CYCLES, "cycle_index is outside the curriculum")
    return tuple(
        sorted(
            canonical_pairs,
            key=lambda pair: sha256_text(
                f"{PAIR_SHUFFLE_NAMESPACE}\0{cycle_index}\0"
                f"{_pair_identity_sha256(pair, directed)}"
            ),
        )
    )


def _schedule_member_binding(
    *,
    ordinal: int,
    donor_ordinal: int,
    role: str,
    directed: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    entry = directed[ordinal]
    return {
        "member_role": role,
        "train_row_ordinal": ordinal,
        "official_source_index": entry["official_source_index"],
        "source_row_sha256": entry["source_row_sha256"],
        "source_label_sha256": entry["source_label_sha256"],
        "source_write_sha256": entry["source_write_sha256"],
        "source_base_record_sha256": entry["source_base_record_sha256"],
        "directed_pair_entry_sha256": entry["entry_sha256"],
        "donor_train_row_ordinal": donor_ordinal,
        "donor_row_sha256": entry["donor_row_sha256"],
        "donor_label_sha256": entry["donor_label_sha256"],
        "donor_write_sha256": entry["donor_write_sha256"],
        "donor_base_record_sha256": entry["donor_base_record_sha256"],
    }


def build_pair_schedule(
    pair_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    directed = _directed_entries_by_ordinal(pair_manifest)
    canonical_pairs = _canonical_pairs(directed)
    entries: list[dict[str, Any]] = []
    for cycle_index in range(1, PAIR_CYCLES + 1):
        ordered_pairs = ordered_pairs_for_cycle(
            canonical_pairs,
            cycle_index=cycle_index,
            directed=directed,
        )
        for cycle_position, pair in enumerate(ordered_pairs):
            low, high = pair
            entry = {
                "schema": PAIR_SCHEDULE_ENTRY_SCHEMA,
                "schedule_index": len(entries),
                "presentation": len(entries) + 1,
                "optimizer_step": len(entries) + 1,
                "cycle_index": cycle_index,
                "cycle_position": cycle_position,
                "pair_batch_size": 2,
                "canonical_pair_ordinals": [low, high],
                "canonical_pair_sha256": _pair_identity_sha256(pair, directed),
                "target_stratum": directed[low]["target_stratum"],
                "members": [
                    _schedule_member_binding(
                        ordinal=low,
                        donor_ordinal=high,
                        role="canonical_low",
                        directed=directed,
                    ),
                    _schedule_member_binding(
                        ordinal=high,
                        donor_ordinal=low,
                        role="canonical_high",
                        directed=directed,
                    ),
                ],
                "sampling": {
                    "mode": "deterministic_hash_sorted_complete_pair_cycle",
                    "namespace": PAIR_SHUFFLE_NAMESPACE,
                },
            }
            entries.append(with_self_hash(entry, field="entry_sha256"))
    require(
        len(entries) == len(canonical_pairs) * PAIR_CYCLES,
        "pair schedule length differs",
    )
    return entries


def schedule_audit(
    entries: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    directed = _directed_entries_by_ordinal(pair_manifest)
    canonical_pairs = _canonical_pairs(directed)
    pairs_per_cycle = len(canonical_pairs)
    require(
        len(entries) == pairs_per_cycle * PAIR_CYCLES,
        "schedule audit length differs",
    )
    cycle_audit: list[dict[str, Any]] = []
    flattened: list[list[int]] = []
    expected_pair_set = set(canonical_pairs)
    expected_ordinals = list(range(len(directed)))
    for cycle_index in range(1, PAIR_CYCLES + 1):
        start = (cycle_index - 1) * pairs_per_cycle
        cycle_entries = entries[start : start + pairs_per_cycle]
        expected_pairs = ordered_pairs_for_cycle(
            canonical_pairs,
            cycle_index=cycle_index,
            directed=directed,
        )
        observed_pairs = tuple(
            tuple(int(value) for value in entry["canonical_pair_ordinals"])
            for entry in cycle_entries
        )
        require(
            observed_pairs == expected_pairs
            and set(observed_pairs) == expected_pair_set
            and len(set(observed_pairs)) == pairs_per_cycle,
            f"cycle {cycle_index} pair coverage differs",
        )
        require(
            sorted(ordinal for pair in observed_pairs for ordinal in pair)
            == expected_ordinals,
            f"cycle {cycle_index} row coverage differs",
        )
        for position, entry in enumerate(cycle_entries):
            _validate_self_hash(
                entry,
                field="entry_sha256",
                description="pair schedule entry",
            )
            require(
                entry.get("schema") == PAIR_SCHEDULE_ENTRY_SCHEMA
                and entry.get("schedule_index") == start + position
                and entry.get("presentation") == start + position + 1
                and entry.get("optimizer_step") == start + position + 1
                and entry.get("cycle_index") == cycle_index
                and entry.get("cycle_position") == position,
                f"cycle {cycle_index} schedule indexing differs",
            )
        flattened.extend([list(pair) for pair in observed_pairs])
        cycle_audit.append(
            {
                "cycle_index": cycle_index,
                "optimizer_step_start": start + 1,
                "optimizer_step_end": start + pairs_per_cycle,
                "ordered_pairs": [list(pair) for pair in observed_pairs],
                "cycle_pairs_sha256": canonical_sha256(
                    [list(pair) for pair in observed_pairs]
                ),
                "checkpoint_prefix_sha256": canonical_sha256(flattened),
            }
        )
    endpoint_steps = tuple(
        pairs_per_cycle * cycle_index
        for cycle_index in range(1, PAIR_CYCLES + 1)
    )
    return {
        "canonical_pairs": [list(pair) for pair in canonical_pairs],
        "canonical_pairs_sha256": canonical_sha256(
            [list(pair) for pair in canonical_pairs]
        ),
        "scheduled_ordinals": expected_ordinals,
        "pair_cycles": PAIR_CYCLES,
        "pairs_per_cycle": pairs_per_cycle,
        "pair_presentations": len(entries),
        "optimizer_steps": len(entries),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "generation_endpoint_steps": list(endpoint_steps),
        "ordered_pairs_sha256": canonical_sha256(flattened),
        "entries_sha256": canonical_sha256(list(entries)),
        "cycle_audit": cycle_audit,
    }


def build_pair_schedule_manifest(
    *,
    schedule_path: Path,
    entries: Sequence[Mapping[str, Any]],
    pair_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit = schedule_audit(entries, pair_manifest)
    return with_self_hash(
        {
            "schema": PAIR_SCHEDULE_MANIFEST_SCHEMA,
            "task": TASK,
            "schedule_schema": PAIR_SCHEDULE_SCHEMA,
            "step_unit": "one_symmetric_reciprocal_pair_per_optimizer_update",
            "sampling_contract": {
                "sampler": "explicit_ordered_scene_hard_failure_pair_cycle_v1",
                "replacement": "four_complete_deterministic_cycles",
                "random_number_generator": "none_sha256_sort_keys_only",
                "namespace": PAIR_SHUFFLE_NAMESPACE,
            },
            "schedule": {
                "path": str(schedule_path.resolve()),
                "sha256": sha256_file(schedule_path),
                "rows": len(entries),
                "entries_sha256": audit["entries_sha256"],
                "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
            },
            "curriculum": audit,
            "protected_evaluation": protected_evaluation_bindings(),
        }
    )


def pair_curriculum_binding(
    *,
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit = schedule_manifest["curriculum"]
    return {
        "schema": PAIR_CURRICULUM_BINDING_SCHEMA,
        "pair_schedule": {
            "path": str(schedule_path.resolve()),
            "sha256": sha256_file(schedule_path),
            "rows": audit["pair_presentations"],
            "entries_sha256": audit["entries_sha256"],
            "ordered_pairs_sha256": audit["ordered_pairs_sha256"],
        },
        "pair_schedule_manifest": {
            "path": str(schedule_manifest_path.resolve()),
            "sha256": sha256_file(schedule_manifest_path),
            "manifest_sha256": schedule_manifest["manifest_sha256"],
        },
        "canonical_pairs": audit["canonical_pairs"],
        "canonical_pairs_sha256": audit["canonical_pairs_sha256"],
        "scheduled_ordinals": audit["scheduled_ordinals"],
        "pair_cycles": audit["pair_cycles"],
        "pairs_per_cycle": audit["pairs_per_cycle"],
        "pair_presentations": audit["pair_presentations"],
        "optimizer_steps": audit["optimizer_steps"],
        "gradient_accumulation_steps": audit["gradient_accumulation_steps"],
        "generation_endpoint_steps": audit["generation_endpoint_steps"],
    }


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _prepare_output(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / filename for name, filename in ARTIFACT_FILENAMES.items()
    }
    existing = [path for path in paths.values() if path.exists()]
    require(overwrite or not existing, "output artifacts already exist: " + ", ".join(map(str, existing)))
    if overwrite:
        for path in existing:
            require(path.is_file() and not path.is_symlink(), f"refusing to replace {path}")
            path.unlink()
    return paths


def _trainer_source_manifest(
    *,
    train_path: Path,
    rows_path: Path,
    pair_path: Path,
    pair_manifest: Mapping[str, Any],
    schedule_path: Path,
    schedule_manifest_path: Path,
    schedule_manifest: Mapping[str, Any],
    rows: int,
    pair_audit: Mapping[str, Any],
    producer_audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SOURCE_SCHEMA,
        "task": TASK,
        "purpose": "train_only_frozen_base_failure_benchmark_boost_v1",
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
                "data": {"path": str(train_path), "sha256": sha256_file(train_path)},
                "row_manifest": {
                    "path": str(rows_path),
                    "sha256": sha256_file(rows_path),
                },
            }
        },
        "v7_pairing": pairing_binding(pair_path, pair_manifest),
        "hard_failure_curriculum": {
            "schema": CURRICULUM_SCHEMA,
            "pairing": dict(pair_audit),
            "train_schedule": pair_curriculum_binding(
                schedule_path=schedule_path,
                schedule_manifest_path=schedule_manifest_path,
                schedule_manifest=schedule_manifest,
            ),
            "base_producer_fingerprint": producer_audit["fingerprint"],
            "protected_evaluation": protected_evaluation_bindings(),
        },
    }
    return with_self_hash(payload)


def prepare_scene_hard_failure_curriculum(
    *,
    train_file: Path = DEFAULT_TRAIN_FILE,
    expected_train_sha256: str = TRAIN_FILE_SHA256,
    base_eval: Path = DEFAULT_BASE_EVAL,
    expected_base_eval_sha256: str = BASE_EVAL_SHA256,
    expected_base_manifest_sha256: str = BASE_MANIFEST_SHA256,
    expected_base_selection_sha256: str = BASE_SELECTION_SHA256,
    expected_base_summary_sha256: str = BASE_SUMMARY_SHA256,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    expected_tokenizer_json_sha256: str = TOKENIZER_JSON_SHA256,
    expected_chat_template_sha256: str = CHAT_TEMPLATE_SHA256,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    overwrite: bool = False,
) -> dict[str, Any]:
    train_file = _verify_regular_file(
        train_file,
        expected_train_sha256,
        description="official scene-v4 TRAIN",
    )
    tokenizer_path = tokenizer_path.expanduser().resolve()
    tokenizer_json = _verify_regular_file(
        tokenizer_path / "tokenizer.json",
        expected_tokenizer_json_sha256,
        description="tokenizer.json",
    )
    chat_template = _verify_regular_file(
        tokenizer_path / "chat_template.jinja",
        expected_chat_template_sha256,
        description="chat_template.jinja",
    )
    paths = _prepare_output(output_dir.expanduser().resolve(), overwrite=overwrite)

    train_rows = load_source_split(train_file, split="train")
    joined, base_payloads, producer_audit = validate_pinned_base_bundle(
        train_file=train_file,
        train_rows=train_rows,
        base_eval=base_eval,
        expected_base_eval_sha256=expected_base_eval_sha256,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
        expected_base_selection_sha256=expected_base_selection_sha256,
        expected_base_summary_sha256=expected_base_summary_sha256,
    )
    candidates, candidate_audit = build_failure_candidates(joined, base_payloads)
    tokenizer = load_tokenizer(tokenizer_path)
    for candidate in candidates:
        materialize_token_metadata(candidate, tokenizer)
    selected_pairs, pair_audit = select_balanced_pairs(candidates, max_pairs=max_pairs)

    selected_indices = sorted(
        {index for pair in selected_pairs for index in pair},
        key=lambda index: candidates[index].source.line_index,
    )
    selected = [candidates[index] for index in selected_indices]
    new_ordinal = {old: ordinal for ordinal, old in enumerate(selected_indices)}
    remapped_pairs = sorted(
        (
            min(new_ordinal[left], new_ordinal[right]),
            max(new_ordinal[left], new_ordinal[right]),
        )
        for left, right in selected_pairs
    )

    atomic_write_text(
        paths["train"],
        "".join(candidate.source.raw_line + "\n" for candidate in selected),
    )
    write_jsonl(
        paths["rows"],
        (
            row_manifest(candidate, train_row_ordinal=ordinal)
            for ordinal, candidate in enumerate(selected)
        ),
    )
    pair_manifest = build_pair_manifest(
        rows=selected,
        pairs=remapped_pairs,
        dataset_path=paths["train"],
        directed_quotas=pair_audit["directed_strata"],
        optimization={
            "algorithm": "balanced_strata_feasibility_preserving_nearest_write_v1",
            **pair_audit,
        },
    )
    require(pair_manifest["schema"] == PAIRING_SCHEMA, "trainer pair schema drifted")
    write_json(paths["pair_manifest"], pair_manifest)
    schedule_entries = build_pair_schedule(pair_manifest)
    write_jsonl(paths["pair_schedule"], schedule_entries)
    schedule_manifest = build_pair_schedule_manifest(
        schedule_path=paths["pair_schedule"],
        entries=schedule_entries,
        pair_manifest=pair_manifest,
    )
    write_json(paths["pair_schedule_manifest"], schedule_manifest)
    source_manifest = _trainer_source_manifest(
        train_path=paths["train"],
        rows_path=paths["rows"],
        pair_path=paths["pair_manifest"],
        pair_manifest=pair_manifest,
        schedule_path=paths["pair_schedule"],
        schedule_manifest_path=paths["pair_schedule_manifest"],
        schedule_manifest=schedule_manifest,
        rows=len(selected),
        pair_audit=pair_audit,
        producer_audit=producer_audit,
    )
    write_json(paths["source_manifest"], source_manifest)

    artifacts = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
        if name != "bundle_manifest"
    }
    selected_failure_counts = Counter(candidate.failure_stratum for candidate in selected)
    selected_cardinalities = Counter(candidate.boundary_count for candidate in selected)
    bundle = with_self_hash(
        {
            "schema": SCHEMA,
            "task": TASK,
            "contract": {
                "source_split": "train",
                "selection": "strict frozen-base benchmark failures only",
                "serialization": "exact source JSONL bytes",
                "messages": ["system", "user", "assistant"],
                "pairing": "reciprocal, row-disjoint, label-distinct",
                "pair_strata_balance": "presence == same_cardinality_value",
                "max_pairs_cap": max_pairs,
                "pair_cycles": PAIR_CYCLES,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            },
            "official_train": {
                "path": str(train_file),
                "sha256": sha256_file(train_file),
                "rows": len(train_rows),
            },
            "base_evaluation": producer_audit,
            "tokenizer": {
                "path": str(tokenizer_path),
                "tokenizer_json_sha256": sha256_file(tokenizer_json),
                "chat_template_sha256": sha256_file(chat_template),
            },
            "protected_evaluation": protected_evaluation_bindings(),
            "candidate_pool": candidate_audit,
            "selection": {
                **pair_audit,
                "selected_failure_strata": dict(sorted(selected_failure_counts.items())),
                "selected_gold_cardinalities": {
                    str(key): value for key, value in sorted(selected_cardinalities.items())
                },
                "ordered_official_source_indices": [
                    candidate.source.line_index for candidate in selected
                ],
                "ordered_row_sha256": canonical_sha256(
                    [candidate.source.row_sha256 for candidate in selected]
                ),
            },
            "train_schedule": dict(schedule_manifest["curriculum"]),
            "validation": {
                "all_emitted_rows_are_official_train": True,
                "all_emitted_rows_are_strict_frozen_base_failures": True,
                "source_serialization_preserved_exactly": True,
                "one_reciprocal_donor_per_row": True,
                "pair_members_have_distinct_gold_labels": True,
                "same_cardinality_pairs_are_nonempty": True,
                "presence_pairs_have_exactly_one_empty_member": True,
                "pair_strata_are_exactly_balanced": True,
                "validation_rows_emitted": 0,
                "hard32_rows_emitted": 0,
                "test_rows_emitted": 0,
                "protected_artifacts_resolved_or_opened": 0,
            },
            "artifacts": artifacts,
        }
    )
    write_json(paths["bundle_manifest"], bundle)
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--expected-train-sha256", default=TRAIN_FILE_SHA256)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--expected-base-eval-sha256", default=BASE_EVAL_SHA256)
    parser.add_argument("--expected-base-manifest-sha256", default=BASE_MANIFEST_SHA256)
    parser.add_argument("--expected-base-selection-sha256", default=BASE_SELECTION_SHA256)
    parser.add_argument("--expected-base-summary-sha256", default=BASE_SUMMARY_SHA256)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--expected-tokenizer-json-sha256",
        default=TOKENIZER_JSON_SHA256,
    )
    parser.add_argument(
        "--expected-chat-template-sha256",
        default=CHAT_TEMPLATE_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_scene_hard_failure_curriculum(
            train_file=args.train_file,
            expected_train_sha256=args.expected_train_sha256,
            base_eval=args.base_eval,
            expected_base_eval_sha256=args.expected_base_eval_sha256,
            expected_base_manifest_sha256=args.expected_base_manifest_sha256,
            expected_base_selection_sha256=args.expected_base_selection_sha256,
            expected_base_summary_sha256=args.expected_base_summary_sha256,
            tokenizer_path=args.tokenizer_path,
            expected_tokenizer_json_sha256=args.expected_tokenizer_json_sha256,
            expected_chat_template_sha256=args.expected_chat_template_sha256,
            output_dir=args.output_dir,
            max_pairs=args.max_pairs,
            overwrite=args.overwrite,
        )
    except (ContractError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"SCENE_HARD_FAILURE_CURRICULUM_ERROR={error}") from error
    print(
        "SCENE_HARD_FAILURE_CURRICULUM="
        + json.dumps(
            {
                "manifest": str(args.output_dir.expanduser().resolve() / "manifest.json"),
                "pairs": manifest["selection"]["selected_pairs"],
                "rows": manifest["selection"]["selected_rows"],
                "schema": manifest["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
