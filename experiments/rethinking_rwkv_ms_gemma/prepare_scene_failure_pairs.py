#!/usr/bin/env python3
"""Build train-failure and validation-holdout scene-boundary episodes.

The builder intentionally has a narrow contract:

* failure mining is restricted to the official scene-v4 train split;
* evaluation records are joined to source rows by the evaluator row SHA-256;
* scene predictions use the existing conservative format recovery;
* the holdout is selected from validation without consulting model outputs; and
* the test split is read only for provenance and overlap validation.

Output train and holdout JSONL rows retain the source three-message structure so
the episode trainer can use ``--episode-recent-messages 0``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence

try:
    from . import analyze_novel_agent_eval as scene_analysis
except ImportError:  # Direct script execution.
    import analyze_novel_agent_eval as scene_analysis


SCHEMA = "rwkv_ms_scene_failure_pairs.v1"
DEFAULT_TASK_NAME = "scene-v4-current"
DEFAULT_CANDIDATE_COUNT = 64
DEFAULT_TRAIN_FAILURE_COUNT = 32
PRODUCER_SCHEMA = "rwkv_ms_scene_train_base_eval.v1"
PRODUCER_SELECTION_SCHEMA = "rwkv_ms_scene_train_base_selection.v1"
PRODUCER_MANIFEST_FILENAME = "manifest.json"
PRODUCER_SELECTION_FILENAME = "candidate_selection.json"
PRODUCER_SUMMARY_FILENAME = "summary.json"
TRAIN_FAILURE_RANK_NAMESPACE = "rwkv_ms_scene_train_failure_selection.v1"
OUTPUT_FILENAMES = (
    "train.jsonl",
    "holdout.jsonl",
    "holdout_source_indices.json",
    "train_manifest.jsonl",
    "holdout_manifest.jsonl",
    "manifest.json",
)
EXPECTED_ROLES = ("system", "user", "assistant")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceRow:
    split: str
    line_index: int
    raw_line: str
    row_sha256: str
    prompt_sha256: str
    messages: list[dict[str, str]]
    gold: dict[str, Any]
    paragraph_count: int


@dataclass(frozen=True)
class BaseRecord:
    eval_line_index: int
    raw_record_sha256: str
    row_sha256: str
    key: str
    source_line_index: int
    producer_fingerprint: str
    parsed_json: Any
    gold: dict[str, Any]


@dataclass(frozen=True)
class FailureRow:
    source: SourceRow
    record: BaseRecord
    recovered_prediction: set[int] | None
    failure_kind: str
    tp: int
    fp: int
    fn: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(serialized)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _declared_artifact_path(
    raw_path: Any,
    *,
    manifest_path: Path,
    description: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{description} path is missing from {manifest_path}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _validate_messages(
    row: Any,
    *,
    path: Path,
    line_number: int,
) -> tuple[list[dict[str, str]], dict[str, Any], int]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict) or set(row) != {"messages"}:
        raise ValueError(f"Expected a messages-only JSON object at {location}")
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 3:
        raise ValueError(f"Expected exactly three messages at {location}")

    messages: list[dict[str, str]] = []
    for message_index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ValueError(f"Message {message_index} is not an object at {location}")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"Message {message_index} has invalid role/content at {location}")
        messages.append({"role": role, "content": content})
    roles = tuple(message["role"] for message in messages)
    if roles != EXPECTED_ROLES:
        raise ValueError(
            f"Expected roles {EXPECTED_ROLES}, found {roles} at {location}"
        )

    gold = scene_analysis.extract_json(messages[-1]["content"])
    if not isinstance(gold, dict):
        raise ValueError(f"Assistant gold is not a JSON object at {location}")
    gold_boundaries = scene_analysis.strict_gold_boundaries(gold)
    paragraph_count = scene_analysis.parse_paragraph_count(messages[1]["content"])
    invalid = sorted(
        boundary
        for boundary in gold_boundaries
        if boundary < 1 or boundary >= paragraph_count
    )
    if invalid:
        raise ValueError(f"Gold contains out-of-range boundaries {invalid} at {location}")
    return messages, gold, paragraph_count


def load_source_split(path: Path, *, split: str) -> list[SourceRow]:
    rows: list[SourceRow] = []
    row_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for physical_line, raw_line_with_newline in enumerate(handle, start=1):
            if not raw_line_with_newline.strip():
                continue
            raw_line = raw_line_with_newline.rstrip("\n")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{physical_line}") from exc
            messages, gold, paragraph_count = _validate_messages(
                payload,
                path=path,
                line_number=physical_line,
            )
            row_sha256 = sha256_text(raw_line)
            prompt_sha256 = sha256_text(messages[1]["content"])
            if row_sha256 in row_hashes:
                raise ValueError(f"Duplicate source row hash in {path}: {row_sha256}")
            if prompt_sha256 in prompt_hashes:
                raise ValueError(f"Duplicate exact user prompt in {path}: {prompt_sha256}")
            row_hashes.add(row_sha256)
            prompt_hashes.add(prompt_sha256)
            rows.append(
                SourceRow(
                    split=split,
                    line_index=len(rows),
                    raw_line=raw_line,
                    row_sha256=row_sha256,
                    prompt_sha256=prompt_sha256,
                    messages=messages,
                    gold=gold,
                    paragraph_count=paragraph_count,
                )
            )
    if not rows:
        raise ValueError(f"Source split is empty: {path}")
    return rows


def _validate_disjoint_splits(splits: dict[str, list[SourceRow]]) -> dict[str, Any]:
    row_overlaps: dict[str, list[str]] = {}
    prompt_overlaps: dict[str, list[str]] = {}
    split_names = tuple(splits)
    for left_index, left_name in enumerate(split_names):
        left_rows = splits[left_name]
        left_row_hashes = {row.row_sha256 for row in left_rows}
        left_prompt_hashes = {row.prompt_sha256 for row in left_rows}
        for right_name in split_names[left_index + 1 :]:
            right_rows = splits[right_name]
            pair_name = f"{left_name}:{right_name}"
            row_overlap = sorted(
                left_row_hashes & {row.row_sha256 for row in right_rows}
            )
            prompt_overlap = sorted(
                left_prompt_hashes & {row.prompt_sha256 for row in right_rows}
            )
            if row_overlap:
                row_overlaps[pair_name] = row_overlap
            if prompt_overlap:
                prompt_overlaps[pair_name] = prompt_overlap
    if row_overlaps or prompt_overlaps:
        raise ValueError(
            "Official scene split overlap detected: "
            f"row_overlaps={row_overlaps} prompt_overlaps={prompt_overlaps}"
        )
    return {
        "row_sha256_pairwise_disjoint": True,
        "exact_user_prompt_sha256_pairwise_disjoint": True,
        "row_overlap_counts": {},
        "exact_user_prompt_overlap_counts": {},
    }


def load_base_records(
    path: Path,
    *,
    task_name: str,
) -> tuple[list[BaseRecord], int]:
    selected: list[BaseRecord] = []
    seen_hashes: set[str] = set()
    total_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for physical_line, raw_line_with_newline in enumerate(handle, start=1):
            if not raw_line_with_newline.strip():
                continue
            raw_line = raw_line_with_newline.rstrip("\n")
            total_records += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{physical_line}") from exc
            if record.get("task") != task_name:
                continue
            expected_fields = {
                "condition": "base",
                "task_kind": "scene",
                "split": "train",
                "status": "ok",
            }
            mismatches = {
                name: {"expected": expected, "actual": record.get(name)}
                for name, expected in expected_fields.items()
                if record.get(name) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Invalid base train record at {path}:{physical_line}: {mismatches}"
                )
            row_sha256 = record.get("row_sha256")
            if not isinstance(row_sha256, str) or SHA256_RE.fullmatch(row_sha256) is None:
                raise ValueError(f"Invalid row_sha256 at {path}:{physical_line}")
            if row_sha256 in seen_hashes:
                raise ValueError(f"Duplicate base-eval row hash: {row_sha256}")
            source_line_index = record.get("line_index")
            if isinstance(source_line_index, bool) or not isinstance(source_line_index, int):
                raise ValueError(f"Invalid line_index at {path}:{physical_line}")
            producer_fingerprint = record.get("fingerprint")
            if (
                not isinstance(producer_fingerprint, str)
                or SHA256_RE.fullmatch(producer_fingerprint) is None
            ):
                raise ValueError(
                    f"Invalid producer fingerprint at {path}:{physical_line}"
                )
            gold = record.get("gold")
            if not isinstance(gold, dict):
                raise ValueError(f"Invalid gold object at {path}:{physical_line}")
            expected_key = f"{task_name}:{source_line_index}"
            if record.get("key") != expected_key:
                raise ValueError(
                    f"Invalid record key at {path}:{physical_line}: "
                    f"expected={expected_key!r} actual={record.get('key')!r}"
                )
            raw_generation = record.get("raw_generation")
            if not isinstance(raw_generation, str):
                raise ValueError(f"Missing raw_generation at {path}:{physical_line}")
            parsed_json = record.get("parsed_json")
            if scene_analysis.extract_json(raw_generation) != parsed_json:
                raise ValueError(
                    f"parsed_json does not match raw_generation at {path}:{physical_line}"
                )
            seen_hashes.add(row_sha256)
            selected.append(
                BaseRecord(
                    eval_line_index=len(selected),
                    raw_record_sha256=sha256_text(raw_line),
                    row_sha256=row_sha256,
                    key=str(record.get("key", "")),
                    source_line_index=source_line_index,
                    producer_fingerprint=producer_fingerprint,
                    parsed_json=parsed_json,
                    gold=gold,
                )
            )
    if not selected:
        raise ValueError(f"No {task_name!r} base train records found in {path}")
    return selected, total_records


def join_train_records(
    train_rows: list[SourceRow],
    base_records: list[BaseRecord],
) -> list[tuple[SourceRow, BaseRecord]]:
    train_by_hash = {row.row_sha256: row for row in train_rows}
    joined: list[tuple[SourceRow, BaseRecord]] = []
    for record in base_records:
        source = train_by_hash.get(record.row_sha256)
        if source is None:
            raise ValueError(
                "Base-eval row does not belong to the official train split: "
                f"{record.row_sha256}"
            )
        if record.source_line_index != source.line_index:
            raise ValueError(
                "Base-eval line_index disagrees with its row_sha256 join: "
                f"hash={record.row_sha256} eval={record.source_line_index} "
                f"train={source.line_index}"
            )
        if record.gold != source.gold:
            raise ValueError(
                f"Base-eval gold disagrees with train source row {record.row_sha256}"
            )
        joined.append((source, record))
    return joined


def validate_base_train_producer_bundle(
    *,
    base_train_eval_path: Path,
    train_source_path: Path,
    joined_records: list[tuple[SourceRow, BaseRecord]],
    total_eval_records: int,
    task_name: str,
) -> dict[str, Any]:
    """Bind base failure records to the producer's fixed 64-row selection."""

    if base_train_eval_path.name != "base.jsonl":
        raise ValueError(
            "Base train evaluation must be the producer-managed base.jsonl, not an "
            f"arbitrary JSONL path: {base_train_eval_path}"
        )
    if total_eval_records != DEFAULT_CANDIDATE_COUNT:
        raise ValueError(
            "Producer base.jsonl must contain exactly 64 records and no unrelated rows: "
            f"records={total_eval_records}"
        )
    if len(joined_records) != DEFAULT_CANDIDATE_COUNT:
        raise ValueError(
            "Producer base.jsonl must contain exactly the 64 selected task records: "
            f"records={len(joined_records)}"
        )

    producer_dir = base_train_eval_path.parent
    manifest_path = producer_dir / PRODUCER_MANIFEST_FILENAME
    selection_path = producer_dir / PRODUCER_SELECTION_FILENAME
    summary_path = producer_dir / PRODUCER_SUMMARY_FILENAME
    manifest = _load_json_object(
        manifest_path,
        description="scene train base producer manifest",
    )
    selection = _load_json_object(
        selection_path,
        description="scene train base candidate selection",
    )
    summary = _load_json_object(
        summary_path,
        description="scene train base completed summary",
    )

    if manifest.get("schema") != PRODUCER_SCHEMA:
        raise ValueError("Unexpected scene train base producer manifest schema")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        description="producer fingerprint",
    )
    fingerprint_payload = manifest.get("fingerprint_payload")
    if not isinstance(fingerprint_payload, dict):
        raise ValueError("Producer manifest fingerprint_payload must be an object")
    if canonical_json_sha256(fingerprint_payload) != fingerprint:
        raise ValueError("Producer manifest fingerprint_payload does not match fingerprint")

    expected_payload_fields = {
        "schema": PRODUCER_SCHEMA,
        "task": task_name,
        "task_kind": "scene",
        "condition": "base",
        "split": "train",
        "candidate_count": DEFAULT_CANDIDATE_COUNT,
    }
    payload_mismatches = {
        field: {"expected": expected, "actual": fingerprint_payload.get(field)}
        for field, expected in expected_payload_fields.items()
        if fingerprint_payload.get(field) != expected
    }
    if payload_mismatches:
        raise ValueError(
            f"Producer fingerprint payload violates the fixed protocol: {payload_mismatches}"
        )

    resolved_train_source = train_source_path.expanduser().resolve()
    payload_dataset_path = _declared_artifact_path(
        fingerprint_payload.get("dataset_file"),
        manifest_path=manifest_path,
        description="producer fingerprint dataset_file",
    )
    if payload_dataset_path != resolved_train_source:
        raise ValueError("Producer fingerprint dataset_file differs from official train.jsonl")
    train_source_sha256 = sha256_file(resolved_train_source)
    if fingerprint_payload.get("dataset_sha256") != train_source_sha256:
        raise ValueError("Producer fingerprint dataset SHA-256 differs from official train.jsonl")

    if selection.get("schema") != PRODUCER_SELECTION_SCHEMA:
        raise ValueError("Unexpected scene train base candidate selection schema")
    expected_selection_fields = {
        "task": task_name,
        "split": "train",
        "candidate_count": DEFAULT_CANDIDATE_COUNT,
        "selection_uses_gold_labels": False,
        "selection_uses_model_output": False,
        "selection_basis": "sha256(selection_seed + NUL + user_prompt_sha256)",
    }
    selection_mismatches = {
        field: {"expected": expected, "actual": selection.get(field)}
        for field, expected in expected_selection_fields.items()
        if selection.get(field) != expected
    }
    if selection_mismatches:
        raise ValueError(
            f"Producer candidate selection violates the fixed protocol: {selection_mismatches}"
        )
    selection_seed = selection.get("selection_seed")
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("Producer candidate selection_seed must be an integer")
    selection_dataset_path = _declared_artifact_path(
        selection.get("dataset_file"),
        manifest_path=selection_path,
        description="producer selection dataset_file",
    )
    if selection_dataset_path != resolved_train_source:
        raise ValueError("Producer selection dataset_file differs from official train.jsonl")
    if selection.get("dataset_sha256") != train_source_sha256:
        raise ValueError("Producer selection dataset SHA-256 differs from official train.jsonl")

    selection_rows = selection.get("rows")
    if not isinstance(selection_rows, list) or len(selection_rows) != DEFAULT_CANDIDATE_COUNT:
        raise ValueError("Producer candidate selection must contain exactly 64 rows")
    expected_candidates: dict[int, tuple[str, str]] = {}
    for ordinal, row in enumerate(selection_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Producer selection row {ordinal} must be an object")
        source_index = row.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"Producer selection row {ordinal} has invalid source_index")
        if source_index in expected_candidates:
            raise ValueError(f"Producer selection repeats source_index {source_index}")
        row_sha256 = _require_sha256(
            row.get("row_sha256"),
            description=f"producer selection row {ordinal} row_sha256",
        )
        prompt_sha256 = _require_sha256(
            row.get("user_prompt_sha256"),
            description=f"producer selection row {ordinal} user_prompt_sha256",
        )
        expected_candidates[source_index] = (row_sha256, prompt_sha256)

    selection_sha256 = sha256_file(selection_path)
    manifest_selection = manifest.get("selection")
    if not isinstance(manifest_selection, dict):
        raise ValueError("Producer manifest selection record is missing")
    declared_selection_path = _declared_artifact_path(
        manifest_selection.get("path"),
        manifest_path=manifest_path,
        description="producer manifest selection",
    )
    if declared_selection_path != selection_path.resolve():
        raise ValueError("Producer manifest points to a different candidate selection")
    expected_manifest_selection = {
        "sha256": selection_sha256,
        "rows": DEFAULT_CANDIDATE_COUNT,
        "uses_gold_labels": False,
        "uses_model_output": False,
    }
    for field, expected in expected_manifest_selection.items():
        if manifest_selection.get(field) != expected:
            raise ValueError(f"Producer manifest selection.{field} differs")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("Producer manifest output record is missing")
    declared_base_records = _declared_artifact_path(
        output.get("base_records"),
        manifest_path=manifest_path,
        description="producer manifest base_records",
    )
    if declared_base_records != base_train_eval_path.resolve():
        raise ValueError("Producer manifest points to a different base.jsonl")

    canonical_selection_sha256 = canonical_json_sha256(selection)
    if fingerprint_payload.get("selection_sha256") != canonical_selection_sha256:
        raise ValueError("Producer fingerprint selection SHA-256 differs")
    if fingerprint_payload.get("selected_rows") != selection_rows:
        raise ValueError("Producer fingerprint selected_rows differ from selection artifact")
    if fingerprint_payload.get("selection_seed") != selection_seed:
        raise ValueError("Producer fingerprint selection_seed differs from selection artifact")

    base_model_artifacts = fingerprint_payload.get("base_model_artifacts")
    if not isinstance(base_model_artifacts, dict):
        raise ValueError("Producer fingerprint base_model_artifacts are missing")
    weights = base_model_artifacts.get("weights")
    runtime_artifacts = base_model_artifacts.get("runtime_artifacts")
    if not isinstance(weights, list) or not weights:
        raise ValueError("Producer fingerprint must bind at least one base-model weight")
    if not isinstance(runtime_artifacts, list) or not runtime_artifacts:
        raise ValueError("Producer fingerprint must bind base-model runtime artifacts")
    aggregate_sha256 = _require_sha256(
        base_model_artifacts.get("aggregate_sha256"),
        description="base-model artifact aggregate_sha256",
    )
    if canonical_json_sha256(
        {"weights": weights, "runtime_artifacts": runtime_artifacts}
    ) != aggregate_sha256:
        raise ValueError("Producer base-model artifact aggregate SHA-256 differs")
    base_model_path = fingerprint_payload.get("base_model")
    if not isinstance(base_model_path, str) or not base_model_path:
        raise ValueError("Producer fingerprint base_model path is missing")
    if base_model_artifacts.get("root") != base_model_path:
        raise ValueError("Producer base-model artifact root differs from base_model path")

    actual_candidates: dict[int, tuple[str, str]] = {}
    record_fingerprints: set[str] = set()
    for source, record in joined_records:
        if source.line_index in actual_candidates:
            raise ValueError(f"Producer base records repeat source index {source.line_index}")
        actual_candidates[source.line_index] = (
            source.row_sha256,
            source.prompt_sha256,
        )
        record_fingerprints.add(record.producer_fingerprint)
    if actual_candidates != expected_candidates:
        raise ValueError(
            "Producer base.jsonl rows differ from the deterministic candidate selection"
        )
    if record_fingerprints != {fingerprint}:
        raise ValueError(
            "Every producer base record must carry the common producer fingerprint"
        )

    expected_summary_fields = {
        "schema": PRODUCER_SCHEMA,
        "fingerprint": fingerprint,
        "complete": True,
        "completed": DEFAULT_CANDIDATE_COUNT,
        "expected": DEFAULT_CANDIDATE_COUNT,
        "condition": "base",
        "task": task_name,
        "split": "train",
    }
    summary_mismatches = {
        field: {"expected": expected, "actual": summary.get(field)}
        for field, expected in expected_summary_fields.items()
        if summary.get(field) != expected
    }
    if summary_mismatches:
        raise ValueError(
            f"Producer summary is not a complete 64-row run: {summary_mismatches}"
        )

    return {
        "schema": PRODUCER_SCHEMA,
        "fingerprint": fingerprint,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": selection_sha256,
            "canonical_sha256": canonical_selection_sha256,
            "candidate_count": DEFAULT_CANDIDATE_COUNT,
            "selection_seed": selection_seed,
            "selection_basis": selection["selection_basis"],
            "uses_gold_labels": False,
            "uses_model_output": False,
        },
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
            "complete": True,
            "completed": DEFAULT_CANDIDATE_COUNT,
        },
        "base_model": {
            "path": base_model_path,
            "artifact_aggregate_sha256": aggregate_sha256,
            "weight_files": len(weights),
            "runtime_artifact_files": len(runtime_artifacts),
        },
    }


def classify_failure(source: SourceRow, record: BaseRecord) -> FailureRow | None:
    prediction = scene_analysis.recover_scene(record.parsed_json)
    gold = scene_analysis.strict_gold_boundaries(source.gold)
    if prediction is None:
        return FailureRow(
            source=source,
            record=record,
            recovered_prediction=None,
            failure_kind="unrecoverable_format",
            tp=0,
            fp=0,
            fn=len(gold),
        )

    tp = len(prediction & gold)
    fp = len(prediction - gold)
    fn = len(gold - prediction)
    if fp == 0 and fn == 0:
        return None
    if fp and fn:
        failure_kind = "mixed_false_positive_false_negative"
    elif fp:
        failure_kind = "false_positive_only"
    else:
        failure_kind = "false_negative_only"
    return FailureRow(
        source=source,
        record=record,
        recovered_prediction=prediction,
        failure_kind=failure_kind,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def select_validation_holdout(
    val_rows: list[SourceRow],
    *,
    count: int,
    seed: int,
) -> list[SourceRow]:
    if count <= 0:
        raise ValueError("holdout_count must be positive")
    if count > len(val_rows):
        raise ValueError(
            f"holdout_count exceeds validation rows: count={count} rows={len(val_rows)}"
        )
    ranked = sorted(
        val_rows,
        key=lambda row: (
            sha256_text(f"{seed}\0{row.prompt_sha256}"),
            row.prompt_sha256,
            row.line_index,
        ),
    )
    selected_hashes = {row.row_sha256 for row in ranked[:count]}
    return [row for row in val_rows if row.row_sha256 in selected_hashes]


def train_failure_selection_sha256(prompt_sha256: str) -> str:
    return sha256_text(f"{TRAIN_FAILURE_RANK_NAMESPACE}\0{prompt_sha256}")


def select_train_failures(
    failures: list[FailureRow],
    *,
    count: int,
) -> list[FailureRow]:
    """Choose a fixed number by a record-order-independent prompt-hash rank."""

    if count <= 0:
        raise ValueError("train_failure_count must be positive")
    if len(failures) < count:
        raise ValueError(
            "Base train evaluation produced fewer eligible failures than required: "
            f"eligible={len(failures)} required={count}"
        )
    ranked = sorted(
        failures,
        key=lambda failure: (
            train_failure_selection_sha256(failure.source.prompt_sha256),
            failure.source.prompt_sha256,
            failure.source.line_index,
        ),
    )
    selected_indices = {
        failure.source.line_index for failure in ranked[:count]
    }
    return sorted(
        (
            failure
            for failure in failures
            if failure.source.line_index in selected_indices
        ),
        key=lambda failure: failure.source.line_index,
    )


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(handle.name)
            for line in lines:
                handle.write(line)
                handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write_lines(path, [serialized])


def _manifest_line(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _failure_manifest_record(failure: FailureRow) -> dict[str, Any]:
    source = failure.source
    prediction = failure.recovered_prediction
    return {
        "partition": "train",
        "source_split": "train",
        "source_line_index": source.line_index,
        "row_sha256": source.row_sha256,
        "prompt_sha256": source.prompt_sha256,
        "failure_selection_sha256": train_failure_selection_sha256(
            source.prompt_sha256
        ),
        "failure_selection_namespace": TRAIN_FAILURE_RANK_NAMESPACE,
        "paragraph_count": source.paragraph_count,
        "gold_boundaries": sorted(scene_analysis.strict_gold_boundaries(source.gold)),
        "base_record_key": failure.record.key,
        "base_record_sha256": failure.record.raw_record_sha256,
        "base_prediction_recovered": prediction is not None,
        "base_recovered_boundaries": None if prediction is None else sorted(prediction),
        "failure_kind": failure.failure_kind,
        "tp": failure.tp,
        "fp": failure.fp,
        "fn": failure.fn,
    }


def _holdout_manifest_record(row: SourceRow, *, seed: int) -> dict[str, Any]:
    return {
        "partition": "holdout",
        "source_split": "val",
        "source_line_index": row.line_index,
        "row_sha256": row.row_sha256,
        "prompt_sha256": row.prompt_sha256,
        "paragraph_count": row.paragraph_count,
        "gold_boundaries": sorted(scene_analysis.strict_gold_boundaries(row.gold)),
        "selection_sha256": sha256_text(f"{seed}\0{row.prompt_sha256}"),
        "selection_key": "seed + exact user prompt SHA-256",
        "selection_uses_model_output": False,
    }


def _partition_summary(
    *,
    data_path: Path,
    manifest_path: Path,
    row_hashes: list[str],
    prompt_hashes: list[str],
    source_split: str,
) -> dict[str, Any]:
    return {
        "source_split": source_split,
        "rows": len(row_hashes),
        "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "row_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "row_hashes_sha256": canonical_json_sha256(row_hashes),
        "prompt_hashes_sha256": canonical_json_sha256(prompt_hashes),
    }


def prepare_scene_failure_pairs(
    *,
    dataset_dir: str | Path,
    base_train_eval_jsonl: str | Path,
    output_dir: str | Path,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    train_failure_count: int = DEFAULT_TRAIN_FAILURE_COUNT,
    holdout_count: int = 32,
    selection_seed: int = 20260728,
    task_name: str = DEFAULT_TASK_NAME,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    base_train_eval_path = Path(base_train_eval_jsonl).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if candidate_count != DEFAULT_CANDIDATE_COUNT:
        raise ValueError(
            "Scene failure mining requires exactly 64 predeclared base candidates: "
            f"candidate_count={candidate_count}"
        )
    split_paths = {
        split: dataset_dir / f"{split}.jsonl" for split in ("train", "val", "test")
    }
    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing official {split} split: {path}")
    if not base_train_eval_path.is_file():
        raise FileNotFoundError(f"Missing base train evaluation: {base_train_eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    source_paths = {*split_paths.values(), base_train_eval_path}
    collisions = sorted(str(path) for path in source_paths & set(output_paths.values()))
    if collisions:
        raise ValueError(f"Output paths collide with protected inputs: {collisions}")
    existing = sorted(str(path) for path in output_paths.values() if path.exists())
    if existing and not overwrite:
        raise FileExistsError(
            "Output artifacts already exist; pass overwrite=True/--overwrite: "
            + ", ".join(existing)
        )

    splits = {
        split: load_source_split(path, split=split)
        for split, path in split_paths.items()
    }
    overlap_validation = _validate_disjoint_splits(splits)
    base_records, total_eval_records = load_base_records(
        base_train_eval_path,
        task_name=task_name,
    )
    if len(base_records) != candidate_count:
        raise ValueError(
            "Base train evaluation candidate count differs from the declared protocol: "
            f"records={len(base_records)} candidate_count={candidate_count}"
        )
    joined = join_train_records(splits["train"], base_records)
    producer_bundle = validate_base_train_producer_bundle(
        base_train_eval_path=base_train_eval_path,
        train_source_path=split_paths["train"],
        joined_records=joined,
        total_eval_records=total_eval_records,
        task_name=task_name,
    )
    eligible_failures = [
        failure
        for source, record in joined
        if (failure := classify_failure(source, record)) is not None
    ]
    failures = select_train_failures(
        eligible_failures,
        count=train_failure_count,
    )
    holdout_rows = select_validation_holdout(
        splits["val"],
        count=holdout_count,
        seed=selection_seed,
    )

    train_row_hashes = [failure.source.row_sha256 for failure in failures]
    holdout_row_hashes = [row.row_sha256 for row in holdout_rows]
    train_prompt_hashes = [failure.source.prompt_sha256 for failure in failures]
    holdout_prompt_hashes = [row.prompt_sha256 for row in holdout_rows]
    if set(train_row_hashes) & set(holdout_row_hashes):
        raise AssertionError("Train and validation holdout row hashes overlap")
    if set(train_prompt_hashes) & set(holdout_prompt_hashes):
        raise AssertionError("Train and validation holdout prompts overlap")

    train_data_path = output_paths["train.jsonl"]
    holdout_data_path = output_paths["holdout.jsonl"]
    holdout_indices_path = output_paths["holdout_source_indices.json"]
    train_manifest_path = output_paths["train_manifest.jsonl"]
    holdout_manifest_path = output_paths["holdout_manifest.jsonl"]
    _atomic_write_lines(train_data_path, (failure.source.raw_line for failure in failures))
    _atomic_write_lines(holdout_data_path, (row.raw_line for row in holdout_rows))
    _atomic_write_json(
        holdout_indices_path,
        {
            "schema": "rwkv_ms_scene_eval_selection.v1",
            "dataset": {
                "split": "val",
                "path": str(split_paths["val"]),
                "sha256": sha256_file(split_paths["val"]),
            },
            "rows": [
                {
                    "source_index": row.line_index,
                    "row_sha256": row.row_sha256,
                }
                for row in holdout_rows
            ],
        },
    )
    _atomic_write_lines(
        train_manifest_path,
        (_manifest_line(_failure_manifest_record(failure)) for failure in failures),
    )
    _atomic_write_lines(
        holdout_manifest_path,
        (
            _manifest_line(_holdout_manifest_record(row, seed=selection_seed))
            for row in holdout_rows
        ),
    )

    builder_path = Path(__file__).resolve()
    analyzer_path = Path(scene_analysis.__file__).resolve()
    eligible_failure_counts = Counter(
        failure.failure_kind for failure in eligible_failures
    )
    selected_failure_counts = Counter(failure.failure_kind for failure in failures)
    manifest = {
        "schema": SCHEMA,
        "task": task_name,
        "contract": {
            "failure_mining_split": "train",
            "holdout_source_split": "val",
            "test_policy": "provenance_and_overlap_audit_only; never emitted",
            "join_key": "sha256(source JSONL line without trailing newline)",
            "failure_rule": (
                "Identify base rows whose conservatively recovered boundary set differs from gold, "
                "plus rows whose base output is not conservatively recoverable; select exactly "
                "train_failure_count rows by the lowest predeclared prompt-hash ranks."
            ),
            "candidate_count": candidate_count,
            "train_failure_count": train_failure_count,
            "failure_selection_rank": (
                "sha256(fixed namespace + NUL + exact user prompt SHA-256)"
            ),
            "failure_selection_namespace": TRAIN_FAILURE_RANK_NAMESPACE,
            "failure_selection_uses_eval_record_order": False,
            "holdout_rule": (
                "Select the lowest deterministic SHA-256 ranks of official-validation user "
                "prompts; selection does not inspect labels, base outputs, or adapter outputs."
            ),
            "episode_contract": {
                "messages": ["system", "user", "assistant"],
                "episode_recent_messages": 0,
                "write_phase": "system + user",
                "read_supervision": "system + assistant",
            },
        },
        "config": {
            "candidate_count": candidate_count,
            "train_failure_count": train_failure_count,
            "holdout_count": holdout_count,
            "selection_seed": selection_seed,
        },
        "sources": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(splits[split]),
                "emitted_for_training": split == "train",
                "emitted_for_holdout": split == "val",
            }
            for split, path in split_paths.items()
        },
        "base_train_evaluation": {
            "path": str(base_train_eval_path),
            "sha256": sha256_file(base_train_eval_path),
            "producer_bundle": producer_bundle,
            "all_jsonl_records": total_eval_records,
            "selected_task_records": len(base_records),
            "joined_train_records": len(joined),
            "eligible_failures": len(eligible_failures),
            "selected_failures": len(failures),
            "eligible_failure_kinds": dict(sorted(eligible_failure_counts.items())),
            "selected_failure_kinds": dict(sorted(selected_failure_counts.items())),
            "failure_kinds": dict(sorted(selected_failure_counts.items())),
        },
        "partitions": {
            "train": _partition_summary(
                data_path=train_data_path,
                manifest_path=train_manifest_path,
                row_hashes=train_row_hashes,
                prompt_hashes=train_prompt_hashes,
                source_split="train",
            ),
            "holdout": _partition_summary(
                data_path=holdout_data_path,
                manifest_path=holdout_manifest_path,
                row_hashes=holdout_row_hashes,
                prompt_hashes=holdout_prompt_hashes,
                source_split="val",
            ),
        },
        "validation": {
            **overlap_validation,
            "all_base_records_joined_to_train_by_row_sha256": True,
            "base_gold_matches_train_source": True,
            "train_holdout_row_sha256_disjoint": True,
            "train_holdout_exact_user_prompt_sha256_disjoint": True,
            "output_rows_preserve_source_serialization": True,
            "output_rows_have_exactly_three_messages": True,
            "candidate_count_matches_protocol": len(base_records) == candidate_count,
            "train_failure_count_matches_protocol": len(failures)
            == train_failure_count,
            "failure_selection_uses_eval_record_order": False,
            "base_records_match_producer_selection": True,
            "base_records_share_producer_fingerprint": True,
            "producer_summary_complete": True,
            "holdout_selection_uses_model_output": False,
            "test_rows_emitted": 0,
        },
        "code": {
            "builder": {"path": str(builder_path), "sha256": sha256_file(builder_path)},
            "scene_recovery": {
                "path": str(analyzer_path),
                "sha256": sha256_file(analyzer_path),
                "function": "recover_scene",
                "rules": scene_analysis.recovery_rules()["scene"],
            },
        },
    }
    manifest["partitions"]["holdout"]["official_source_indices"] = {
        "path": str(holdout_indices_path),
        "sha256": sha256_file(holdout_indices_path),
        "indices": [row.line_index for row in holdout_rows],
        "consumer": "run_scene_state_eval.py --row-indices-file",
    }
    _atomic_write_json(output_paths["manifest.json"], manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Official v4-scene-boundary-detection directory containing train/val/test.jsonl.",
    )
    parser.add_argument("--base-train-eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
    )
    parser.add_argument(
        "--train-failure-count",
        type=int,
        default=DEFAULT_TRAIN_FAILURE_COUNT,
    )
    parser.add_argument("--holdout-count", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=20260728)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = prepare_scene_failure_pairs(
        dataset_dir=args.dataset_dir,
        base_train_eval_jsonl=args.base_train_eval_jsonl,
        output_dir=args.output_dir,
        candidate_count=args.candidate_count,
        train_failure_count=args.train_failure_count,
        holdout_count=args.holdout_count,
        selection_seed=args.selection_seed,
        task_name=args.task_name,
        overwrite=args.overwrite,
    )
    print(
        "SCENE_FAILURE_PAIRS="
        + json.dumps(
            {
                "schema": manifest["schema"],
                "train_rows": manifest["partitions"]["train"]["rows"],
                "holdout_rows": manifest["partitions"]["holdout"]["rows"],
                "manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
