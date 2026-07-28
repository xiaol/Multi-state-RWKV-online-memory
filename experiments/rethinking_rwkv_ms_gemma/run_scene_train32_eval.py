#!/usr/bin/env python3
"""Evaluate the protected V7 Train32 or Tiny2 scene-memory overfit contract.

This evaluator deliberately accepts only training-derived V7 artifacts.  It
reuses the state-isolating runtime from :mod:`run_scene_state_eval`, but owns a
separate split-aware contract so validation data cannot enter an overfit run.
The pass/fail decision uses canonical greedy generations and strict scene
scores only; format recovery is reported as a diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_LOCK = SCRIPT_DIR / "scene_memory_v7_source_lock.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rethinking_rwkv_ms_gemma.run_scene_state_eval import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    TASK_NAME,
    base_model_prompt_identity,
    base_model_weight_identity,
    build_comparisons,
    clear_model_memory,
    evaluate_condition,
    extract_json,
    fingerprint_payload_sha256,
    generate_messages,
    is_canonical_scene_prediction,
    load_adapter_model,
    load_selected_rows,
    memory_architecture_contract,
    memory_condition,
    prime_online_state,
    recovered_scene_score,
    render_and_tokenize,
    reset_delta_state,
    resolved_memory_layer_count,
    runtime_package_versions,
    score_prediction,
    sha256_file,
    sha256_text,
    strict_gold_boundaries,
    summarize_records,
    utc_now,
)


SOURCE_SCHEMA = "rwkv_ms_scene_memory_v7_source.v1"
PAIR_SCHEMA = "rwkv_ms_scene_memory_v7_pairing.v1"
PAIR_BINDING_SCHEMA = "rwkv_ms_scene_memory_v7_pairing_binding.v1"
ROW_SCHEMA = "rwkv_ms_scene_memory_v7_row.v1"
RECEIPT_SCHEMA = "rwkv_ms_scene_v7_overfit_receipt.v1"
RECORD_SCHEMA = "rwkv_ms_scene_v7_overfit_record.v1"
CONDITIONS = ("state_only", "state_only_donor", "state_only_no_write")
CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    "scene_v7_train32_overfit": {
        "variant": "train32",
        "rows": 32,
        "quotas": {
            "presence": 18,
            "same_cardinality_value": 10,
            "cross_cardinality_value": 4,
        },
        "max_label_multiplicity": 9,
        "empty_rows": 9,
    },
    "scene_v7_tiny_overfit": {
        "variant": "tiny2",
        "rows": 2,
        "quotas": {
            "presence": 0,
            "same_cardinality_value": 2,
            "cross_cardinality_value": 0,
        },
        "max_label_multiplicity": 1,
        "empty_rows": 0,
    },
}
TARGET_STRATA = (
    "presence",
    "same_cardinality_value",
    "cross_cardinality_value",
)
TRAIN32_GATE_REQUIREMENTS = {
    "canonical_outputs": 31,
    "strict_exact_rows": 24,
    "strict_true_positives": 24,
    "empty_exact_rows": 6,
    "same_cardinality_exact_rows": 8,
    "donor_identity_exact_rows": 24,
    "same_cardinality_donor_identity_exact_rows": 8,
    "correct_beats_donor_rows": 20,
    "same_cardinality_correct_beats_donor_rows": 8,
    "max_zero_exact_rows": 9,
    "correct_minus_zero_exact_rows": 16,
    "correct_minus_zero_micro_f1": 0.25,
    "max_predicted_to_gold_boundary_ratio": 2.0,
    "correct_minus_donor_micro_f1": 0.05,
    "correct_minus_zero_causal_micro_f1": 0.05,
}
EPISODE_CONTRACT = {
    "episode_recent_messages": 0,
    "write_phase": "system + user",
    "read_supervision": "system + assistant",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class V7EvaluationContractError(ValueError):
    """Raised when a protected evaluator binding differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V7EvaluationContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def self_hash_payload(
    payload: Mapping[str, Any], *, hash_field: str = "receipt_sha256"
) -> str:
    """Hash canonical JSON after excluding one self-hash field."""

    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    return canonical_sha256(unsigned)


def _validate_self_hash(payload: Mapping[str, Any], *, field: str) -> None:
    recorded = payload.get(field)
    _require(
        isinstance(recorded, str) and SHA256_RE.fullmatch(recorded) is not None,
        f"{field} is missing or invalid",
    )
    _require(recorded == self_hash_payload(payload, hash_field=field), f"{field} differs")


def atomic_write_text(path: Path, text: str) -> None:
    """Durably replace ``path`` using a temporary file in the same directory."""

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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


def _regular_file(path: Path, *, description: str) -> Path:
    expanded = path.expanduser()
    _require(not expanded.is_symlink(), f"symlink is forbidden for {description}: {expanded}")
    resolved = expanded.resolve()
    _require(resolved.is_file(), f"missing {description}: {resolved}")
    return resolved


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = _regular_file(path, description=description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V7EvaluationContractError(f"invalid JSON in {description}: {path}") from exc
    _require(isinstance(payload, dict), f"{description} must be a JSON object")
    return payload


def _read_jsonl(path: Path, *, description: str) -> list[tuple[str, dict[str, Any]]]:
    path = _regular_file(path, description=description)
    records: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\r\n")
            _require(bool(raw.strip()), f"blank JSONL row in {description}:{line_number}")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise V7EvaluationContractError(
                    f"invalid JSONL row in {description}:{line_number}"
                ) from exc
            _require(
                isinstance(payload, dict),
                f"JSONL row must be an object in {description}:{line_number}",
            )
            records.append((raw, payload))
    return records


def _validate_sha(value: Any, *, description: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"invalid SHA-256 for {description}",
    )
    return value


def _artifact_binding(
    path: Path, *, description: str, expected_sha256: str | None
) -> dict[str, Any]:
    path = _regular_file(path, description=description)
    actual = sha256_file(path)
    if expected_sha256 is not None:
        _validate_sha(expected_sha256, description=f"expected {description}")
        _require(actual == expected_sha256, f"{description} SHA-256 differs")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "actual_sha256": actual,
        "expected_sha256": expected_sha256,
    }


def _contract_spec(contract: str) -> dict[str, Any]:
    spec = CONTRACT_SPECS.get(contract)
    _require(spec is not None, f"unsupported V7 evaluation contract: {contract}")
    return dict(spec)


def validate_v7_source_lock(
    source_lock_file: Path | str,
    *,
    contract: str,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind protected evaluator inputs to the repository V7 source lock."""

    spec = _contract_spec(contract)
    source_lock_path = _regular_file(Path(source_lock_file), description="V7 source lock")
    lock = _load_json(source_lock_path, description="V7 source lock")
    _require(
        lock.get("schema") == "rwkv_ms_scene_memory_v7_source_lock.v1",
        "V7 source-lock schema differs",
    )
    _validate_self_hash(lock, field="lock_sha256")
    artifacts = lock.get("artifacts")
    _require(isinstance(artifacts, dict), "V7 source-lock artifacts are missing")
    variant = str(spec["variant"])
    lock_names = {
        "dataset": variant,
        "row_manifest": f"{variant}_rows",
        "pair_manifest": f"{variant}_pair_manifest",
        "source_manifest": f"{variant}_source_manifest",
    }
    for evaluator_name, lock_name in lock_names.items():
        locked = artifacts.get(lock_name)
        current = artifact_bindings.get(evaluator_name)
        _require(
            isinstance(locked, dict) and isinstance(current, Mapping),
            f"V7 source-lock binding is missing: {lock_name}",
        )
        _require(
            Path(str(locked.get("path"))).expanduser().resolve()
            == Path(str(current.get("path"))).expanduser().resolve(),
            f"V7 source-lock path differs: {lock_name}",
        )
        _require(
            locked.get("sha256") == current.get("actual_sha256"),
            f"V7 source-lock SHA-256 differs: {lock_name}",
        )
    return {
        "path": str(source_lock_path),
        "file_sha256": sha256_file(source_lock_path),
        "lock_sha256": lock["lock_sha256"],
        "artifact_names": lock_names,
    }


def _reject_validation_or_test_path(path: Path, *, description: str) -> None:
    name = path.name.casefold()
    _require(
        name not in {"val.jsonl", "validation.jsonl", "test.jsonl"},
        f"{description} is validation/test-derived: {path}",
    )


def load_v7_rows(
    dataset_file: Path | str,
    row_manifest_file: Path | str,
    *,
    contract: str | None = None,
    expected_rows: int | None = None,
    expected_dataset_sha256: str | None = None,
    expected_row_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Load and bind every V7 data row to its ordinal row-manifest record."""

    if contract is not None:
        contract_rows = int(_contract_spec(contract)["rows"])
        if expected_rows is not None:
            _require(expected_rows == contract_rows, "expected row count differs from contract")
        expected_rows = contract_rows
    _require(expected_rows in {2, 32}, "V7 row count must be exactly 2 or 32")
    dataset_path = _regular_file(Path(dataset_file), description="V7 dataset")
    rows_path = _regular_file(Path(row_manifest_file), description="V7 row manifest")
    _reject_validation_or_test_path(dataset_path, description="V7 dataset")
    _artifact_binding(
        dataset_path,
        description="V7 dataset",
        expected_sha256=expected_dataset_sha256,
    )
    _artifact_binding(
        rows_path,
        description="V7 row manifest",
        expected_sha256=expected_row_manifest_sha256,
    )
    data_records = _read_jsonl(dataset_path, description="V7 dataset")
    manifest_records = _read_jsonl(rows_path, description="V7 row manifest")
    _require(
        len(data_records) == len(manifest_records) == expected_rows,
        f"V7 contract requires exactly {expected_rows} rows",
    )

    rows: list[dict[str, Any]] = []
    source_indices: set[int] = set()
    row_hashes: set[str] = set()
    for ordinal, ((raw_line, data), (_, row_manifest)) in enumerate(
        zip(data_records, manifest_records, strict=True)
    ):
        _require(row_manifest.get("schema") == ROW_SCHEMA, "V7 row schema differs")
        _validate_self_hash(row_manifest, field="record_sha256")
        _require(row_manifest.get("train_row_ordinal") == ordinal, "V7 row ordinal differs")
        _require(row_manifest.get("source_split") == "train", "V7 row is not train-derived")
        row_sha = sha256_text(raw_line)
        _require(row_manifest.get("row_sha256") == row_sha, "V7 row SHA-256 differs")
        messages = data.get("messages")
        _require(
            isinstance(messages, list) and len(messages) == 3,
            "V7 scene row must contain exactly three messages",
        )
        _require(
            [message.get("role") for message in messages]
            == ["system", "user", "assistant"],
            "V7 scene message roles differ",
        )
        content = messages[2].get("content")
        _require(isinstance(content, str), "V7 assistant gold must be text")
        try:
            gold = json.loads(content)
        except json.JSONDecodeError as exc:
            raise V7EvaluationContractError("V7 assistant gold is not exact JSON") from exc
        _require(is_canonical_scene_prediction(gold), "V7 assistant gold is not canonical")
        strict = score_prediction("scene", gold, gold)
        _require(
            bool(strict["schema_valid"]) and not strict["fp"] and not strict["fn"],
            "V7 assistant gold is not strictly self-consistent",
        )
        _require(
            row_manifest.get("gold_boundaries") == strict["gold_boundaries"],
            "V7 row gold binding differs",
        )
        _require(
            row_manifest.get("gold_boundary_count") == len(strict["gold_boundaries"]),
            "V7 row gold cardinality differs",
        )
        baseline_score = row_manifest.get("strict_score")
        _require(isinstance(baseline_score, dict), "V7 audited base strict score is missing")
        _require(
            not (
                bool(baseline_score.get("schema_valid"))
                and int(baseline_score.get("fp", 0)) == 0
                and int(baseline_score.get("fn", 0)) == 0
            ),
            "V7 row is not an audited strict base failure",
        )
        _require(
            row_manifest.get("strict_failure_stratum")
            in {"invalid_schema", "false_positive_only", "false_negative_only", "mixed"},
            "V7 base failure stratum differs",
        )
        base_record_sha256 = _validate_sha(
            row_manifest.get("base_record_sha256"),
            description="V7 base record",
        )
        source_index = row_manifest.get("official_source_index")
        _require(
            isinstance(source_index, int)
            and not isinstance(source_index, bool)
            and source_index >= 0,
            "V7 official source index is invalid",
        )
        _require(source_index not in source_indices, "V7 official source index is duplicated")
        _require(row_sha not in row_hashes, "V7 row hash is duplicated")
        source_indices.add(source_index)
        row_hashes.add(row_sha)
        label_sha = _validate_sha(row_manifest.get("label_sha256"), description="V7 label")
        _require(label_sha == canonical_sha256(strict["gold_boundaries"]), "V7 label SHA-256 differs")
        token_metadata = row_manifest.get("token_metadata")
        _require(isinstance(token_metadata, dict), "V7 token metadata is missing")
        _require(
            {
                "write_input_ids_sha256",
                "write_rendered_sha256",
                "write_token_count",
                "generation_prefix_input_ids_sha256",
                "generation_prefix_rendered_sha256",
                "generation_prefix_token_count",
                "semantic_target_positions",
                "semantic_target_token_ids",
            }
            <= set(token_metadata),
            "V7 token metadata is incomplete",
        )
        _validate_sha(
            token_metadata.get("generation_prefix_input_ids_sha256"),
            description="V7 generation prefix input IDs",
        )
        write_count = token_metadata.get("write_token_count")
        _require(
            isinstance(write_count, int) and not isinstance(write_count, bool) and write_count > 0,
            "V7 write token count is invalid",
        )
        write_sha = _validate_sha(
            token_metadata.get("write_input_ids_sha256"), description="V7 write input IDs"
        )
        positions = token_metadata.get("semantic_target_positions")
        token_ids = token_metadata.get("semantic_target_token_ids")
        _require(
            isinstance(positions, list)
            and isinstance(token_ids, list)
            and len(positions) == len(token_ids) > 0,
            "V7 semantic target metadata differs",
        )
        prime_messages = messages[:2]
        rows.append(
            {
                "train_row_ordinal": ordinal,
                "source_index": source_index,
                "official_source_index": source_index,
                "messages": prime_messages,
                "gold": gold,
                "gold_content": content,
                "row_sha256": row_sha,
                "prime_messages_sha256": fingerprint_payload_sha256(
                    {"messages": prime_messages}
                ),
                "write_token_count": write_count,
                "write_input_ids_sha256": write_sha,
                "label_sha256": label_sha,
                "gold_boundary_count": len(strict["gold_boundaries"]),
                "base_record_sha256": base_record_sha256,
                "strict_failure_stratum": row_manifest["strict_failure_stratum"],
                "strict_score_sha256": canonical_sha256(baseline_score),
                "token_metadata": token_metadata,
                "row_manifest": row_manifest,
                "data": data,
            }
        )
    return rows


def validate_runtime_prefixes(
    tokenizer,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-tokenize every write/read prefix and bind it to frozen row metadata."""

    generation_hashes: set[str] = set()
    generation_render_hashes: set[str] = set()
    for row in rows:
        metadata = row["token_metadata"]
        write_rendered, write_ids, _ = render_and_tokenize(
            tokenizer,
            list(row["messages"]),
            add_generation_prompt=False,
            device="cpu",
        )
        generation_rendered, generation_ids, _ = render_and_tokenize(
            tokenizer,
            [row["messages"][0]],
            add_generation_prompt=True,
            device="cpu",
        )
        write_token_ids = [int(value) for value in write_ids[0].tolist()]
        generation_token_ids = [int(value) for value in generation_ids[0].tolist()]
        _require(
            canonical_sha256(write_token_ids) == metadata["write_input_ids_sha256"]
            and len(write_token_ids) == metadata["write_token_count"]
            and sha256_text(write_rendered) == metadata["write_rendered_sha256"],
            "V7 runtime write prefix differs from frozen token metadata",
        )
        _require(
            canonical_sha256(generation_token_ids)
            == metadata["generation_prefix_input_ids_sha256"]
            and len(generation_token_ids) == metadata["generation_prefix_token_count"]
            and sha256_text(generation_rendered)
            == metadata["generation_prefix_rendered_sha256"],
            "V7 runtime generation prefix differs from frozen token metadata",
        )
        generation_hashes.add(metadata["generation_prefix_input_ids_sha256"])
        generation_render_hashes.add(metadata["generation_prefix_rendered_sha256"])
    _require(len(generation_hashes) == 1, "V7 rows do not share one exact generation token prefix")
    _require(len(generation_render_hashes) == 1, "V7 rows do not share one exact generation rendered prefix")
    return {
        "rows": len(rows),
        "generation_prefix_input_ids_sha256": next(iter(generation_hashes)),
        "generation_prefix_rendered_sha256": next(iter(generation_render_hashes)),
    }


def load_v7_pairing(
    pair_manifest_file: Path | str,
    *,
    dataset_file: Path | str,
    rows: Sequence[Mapping[str, Any]],
    contract: str,
    expected_pair_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the complete symmetric V7 directed donor mapping."""

    spec = _contract_spec(contract)
    expected_rows = int(spec["rows"])
    _require(len(rows) == expected_rows, "V7 pairing row count differs from contract")
    dataset_path = _regular_file(Path(dataset_file), description="V7 dataset")
    pair_path = _regular_file(Path(pair_manifest_file), description="V7 pair manifest")
    _artifact_binding(
        pair_path,
        description="V7 pair manifest",
        expected_sha256=expected_pair_manifest_sha256,
    )
    payload = _load_json(pair_path, description="V7 pair manifest")
    _require(payload.get("schema") == PAIR_SCHEMA, "V7 pair manifest schema differs")
    _validate_self_hash(payload, field="manifest_sha256")
    dataset = payload.get("dataset")
    _require(isinstance(dataset, dict), "V7 pair dataset binding is missing")
    _require(
        Path(str(dataset.get("path"))).expanduser().resolve() == dataset_path,
        "V7 pair dataset path differs",
    )
    dataset_sha = sha256_file(dataset_path)
    _require(dataset.get("sha256") == dataset_sha, "V7 pair dataset SHA-256 differs")
    _require(dataset.get("rows") == expected_rows, "V7 pair dataset row count differs")
    ordered_hashes = [str(row["row_sha256"]) for row in rows]
    _require(
        dataset.get("ordered_row_sha256") == canonical_sha256(ordered_hashes),
        "V7 pair ordered row digest differs",
    )
    expected_quotas = dict(spec["quotas"])
    _require(payload.get("quotas") == expected_quotas, "V7 pair quotas differ")
    directed = payload.get("directed_pairs")
    _require(
        isinstance(directed, list) and len(directed) == expected_rows,
        "V7 directed pairs do not cover every row",
    )
    _require(payload.get("entries_sha256") == canonical_sha256(directed), "V7 pair list digest differs")

    by_ordinal: dict[int, dict[str, Any]] = {}
    observed = Counter()
    for entry in directed:
        _require(isinstance(entry, dict), "V7 directed pair entry must be an object")
        _validate_self_hash(entry, field="entry_sha256")
        ordinal = entry.get("train_row_ordinal")
        donor_ordinal = entry.get("donor_train_row_ordinal")
        _require(
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and isinstance(donor_ordinal, int)
            and not isinstance(donor_ordinal, bool)
            and 0 <= ordinal < expected_rows
            and 0 <= donor_ordinal < expected_rows
            and ordinal != donor_ordinal,
            "V7 directed pair ordinals are invalid",
        )
        _require(ordinal not in by_ordinal, "V7 directed pair ordinal is duplicated")
        source = rows[ordinal]
        donor = rows[donor_ordinal]
        bindings = {
            "official_source_index": source["source_index"],
            "donor_official_source_index": donor["source_index"],
            "source_row_sha256": source["row_sha256"],
            "donor_row_sha256": donor["row_sha256"],
            "source_label_sha256": source["label_sha256"],
            "donor_label_sha256": donor["label_sha256"],
            "source_boundary_count": source["gold_boundary_count"],
            "donor_boundary_count": donor["gold_boundary_count"],
            "source_write_sha256": source["write_input_ids_sha256"],
            "donor_write_sha256": donor["write_input_ids_sha256"],
            "source_write_token_count": source["write_token_count"],
            "donor_write_token_count": donor["write_token_count"],
            "source_generation_prefix_sha256": source["token_metadata"][
                "generation_prefix_input_ids_sha256"
            ],
            "donor_generation_prefix_sha256": donor["token_metadata"][
                "generation_prefix_input_ids_sha256"
            ],
            "source_base_record_sha256": source["base_record_sha256"],
            "donor_base_record_sha256": donor["base_record_sha256"],
            "source_strict_failure_stratum": source["strict_failure_stratum"],
            "donor_strict_failure_stratum": donor["strict_failure_stratum"],
            "source_strict_score_sha256": source["strict_score_sha256"],
            "donor_strict_score_sha256": donor["strict_score_sha256"],
        }
        for field, expected in bindings.items():
            _require(entry.get(field) == expected, f"V7 pair {field} differs")
        _require(
            entry["source_label_sha256"] != entry["donor_label_sha256"],
            "V7 donor label is not distinct",
        )
        _require(
            entry["source_generation_prefix_sha256"]
            == entry["donor_generation_prefix_sha256"],
            "V7 source and donor generation prefixes differ",
        )
        expected_delta = abs(int(source["write_token_count"]) - int(donor["write_token_count"]))
        _require(entry.get("write_token_count_delta") == expected_delta, "V7 pair write-token delta differs")
        positions = entry.get("selected_target_positions")
        predictors = entry.get("selected_target_predictor_positions")
        token_ids = entry.get("selected_target_token_ids")
        donor_ids = entry.get("donor_target_token_ids")
        semantic_ordinal = entry.get("first_differing_semantic_ordinal")
        _require(
            isinstance(positions, list)
            and isinstance(predictors, list)
            and isinstance(token_ids, list)
            and isinstance(donor_ids, list)
            and len(positions) == len(predictors) == len(token_ids) == len(donor_ids) == 1
            and predictors == [positions[0] - 1]
            and token_ids != donor_ids,
            "V7 selected distinguishing target differs",
        )
        _require(
            isinstance(semantic_ordinal, int)
            and not isinstance(semantic_ordinal, bool)
            and semantic_ordinal >= 0,
            "V7 semantic target ordinal is invalid",
        )
        source_meta = source["token_metadata"]
        donor_meta = donor["token_metadata"]
        _require(
            semantic_ordinal < len(source_meta["semantic_target_token_ids"])
            and semantic_ordinal < len(donor_meta["semantic_target_token_ids"])
            and source_meta["semantic_target_positions"][semantic_ordinal] == positions[0]
            and source_meta["semantic_target_token_ids"][semantic_ordinal] == token_ids[0]
            and donor_meta["semantic_target_token_ids"][semantic_ordinal] == donor_ids[0],
            "V7 selected target does not bind to row token metadata",
        )
        _validate_sha(entry.get("causal_prefix_sha256"), description="V7 causal prefix")
        stratum = entry.get("target_stratum")
        _require(stratum in TARGET_STRATA, "V7 target stratum differs")
        observed[stratum] += 1
        by_ordinal[ordinal] = entry

    _require(set(by_ordinal) == set(range(expected_rows)), "V7 pair coverage is incomplete")
    _require(
        {name: observed[name] for name in TARGET_STRATA} == expected_quotas,
        "V7 observed pair strata differ",
    )
    for ordinal, entry in by_ordinal.items():
        donor_ordinal = int(entry["donor_train_row_ordinal"])
        reverse = by_ordinal[donor_ordinal]
        _require(reverse.get("donor_train_row_ordinal") == ordinal, "V7 donor map is not symmetric")
        _require(reverse.get("source_row_sha256") == entry.get("donor_row_sha256"), "V7 reverse source row differs")
        _require(reverse.get("donor_row_sha256") == entry.get("source_row_sha256"), "V7 reverse donor row differs")
        _require(
            reverse.get("causal_prefix_sha256") == entry.get("causal_prefix_sha256"),
            "V7 reverse causal prefix differs",
        )

    if contract == "scene_v7_tiny_overfit":
        _require(
            {int(row["source_index"]) for row in rows} == {383, 619},
            "Tiny2 official source indices differ",
        )
        _require(
            {int(row["gold_boundary_count"]) for row in rows} == {2},
            "Tiny2 must be a positive same-cardinality pair",
        )
        _require(
            all(entry["target_stratum"] == "same_cardinality_value" for entry in directed)
            and all(entry["write_token_count_delta"] == 21 for entry in directed),
            "Tiny2 frozen same-cardinality pairing differs",
        )
    return {
        "path": str(pair_path),
        "file_sha256": sha256_file(pair_path),
        "manifest_sha256": payload["manifest_sha256"],
        "entries_sha256": payload["entries_sha256"],
        "quotas": expected_quotas,
        "directed_pairs": directed,
        "by_ordinal": by_ordinal,
        "donor_by_ordinal": {
            ordinal: int(entry["donor_train_row_ordinal"])
            for ordinal, entry in by_ordinal.items()
        },
        "payload": payload,
    }


def validate_v7_contract(
    *,
    contract: str,
    dataset_file: Path | str,
    row_manifest_file: Path | str,
    pair_manifest_file: Path | str,
    source_manifest_file: Path | str,
    expected_dataset_sha256: str | None = None,
    expected_row_manifest_sha256: str | None = None,
    expected_pair_manifest_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    source_lock_file: Path | str | None = None,
) -> dict[str, Any]:
    """Validate all V7 inputs and return normalized rows and donor bindings."""

    spec = _contract_spec(contract)
    dataset_path = _regular_file(Path(dataset_file), description="V7 dataset")
    rows_path = _regular_file(Path(row_manifest_file), description="V7 row manifest")
    pair_path = _regular_file(Path(pair_manifest_file), description="V7 pair manifest")
    source_path = _regular_file(Path(source_manifest_file), description="V7 source manifest")
    for path, description in (
        (dataset_path, "V7 dataset"),
        (rows_path, "V7 row manifest"),
        (pair_path, "V7 pair manifest"),
        (source_path, "V7 source manifest"),
    ):
        _reject_validation_or_test_path(path, description=description)

    artifact_bindings = {
        "dataset": _artifact_binding(
            dataset_path,
            description="V7 dataset",
            expected_sha256=expected_dataset_sha256,
        ),
        "row_manifest": _artifact_binding(
            rows_path,
            description="V7 row manifest",
            expected_sha256=expected_row_manifest_sha256,
        ),
        "pair_manifest": _artifact_binding(
            pair_path,
            description="V7 pair manifest",
            expected_sha256=expected_pair_manifest_sha256,
        ),
        "source_manifest": _artifact_binding(
            source_path,
            description="V7 source manifest",
            expected_sha256=expected_source_manifest_sha256,
        ),
    }
    source_lock = (
        None
        if source_lock_file is None
        else validate_v7_source_lock(
            source_lock_file,
            contract=contract,
            artifact_bindings=artifact_bindings,
        )
    )
    rows = load_v7_rows(
        dataset_path,
        rows_path,
        contract=contract,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_row_manifest_sha256=expected_row_manifest_sha256,
    )
    pairing = load_v7_pairing(
        pair_path,
        dataset_file=dataset_path,
        rows=rows,
        contract=contract,
        expected_pair_manifest_sha256=expected_pair_manifest_sha256,
    )
    source = _load_json(source_path, description="V7 source manifest")
    _require(source.get("schema") == SOURCE_SCHEMA, "V7 source schema differs")
    _require(source.get("task") == TASK_NAME, "V7 source task differs")
    _validate_self_hash(source, field="manifest_sha256")
    source_contract = source.get("contract")
    _require(isinstance(source_contract, dict), "V7 source contract is missing")
    _require(source_contract.get("source_split") == "train", "V7 source split is not train")
    _require(source_contract.get("val_rows") == 0, "V7 source contains validation rows")
    _require(source_contract.get("test_rows") == 0, "V7 source contains test rows")
    _require(source_contract.get("episode_contract") == EPISODE_CONTRACT, "V7 episode contract differs")
    partitions = source.get("partitions")
    _require(isinstance(partitions, dict) and set(partitions) == {"train"}, "V7 source partitions must contain only train")
    train = partitions["train"]
    _require(isinstance(train, dict), "V7 train partition is missing")
    _require(train.get("source_split") == "train", "V7 partition source split differs")
    _require(train.get("rows") == spec["rows"], "V7 source row count differs")
    data_binding = train.get("data")
    row_binding = train.get("row_manifest")
    _require(isinstance(data_binding, dict) and isinstance(row_binding, dict), "V7 source artifact binding is missing")
    _require(
        Path(str(data_binding.get("path"))).expanduser().resolve() == dataset_path
        and data_binding.get("sha256") == sha256_file(dataset_path),
        "V7 source dataset binding differs",
    )
    _require(
        Path(str(row_binding.get("path"))).expanduser().resolve() == rows_path
        and row_binding.get("sha256") == sha256_file(rows_path),
        "V7 source row-manifest binding differs",
    )
    for declared in (Path(str(data_binding["path"])), Path(str(row_binding["path"]))):
        _reject_validation_or_test_path(declared, description="V7 declared source path")
    pair_binding = source.get("v7_pairing")
    _require(isinstance(pair_binding, dict), "V7 source pair binding is missing")
    _require(pair_binding.get("schema") == PAIR_BINDING_SCHEMA, "V7 source pair-binding schema differs")
    _require(pair_binding.get("dataset_sha256") == sha256_file(dataset_path), "V7 source pair dataset digest differs")
    _require(pair_binding.get("directed_entry_count") == spec["rows"], "V7 source pair count differs")
    _require(pair_binding.get("quotas") == spec["quotas"], "V7 source pair quotas differ")
    _require(pair_binding.get("entries_sha256") == pairing["entries_sha256"], "V7 source pair-list digest differs")
    pair_file_binding = pair_binding.get("pair_manifest")
    _require(isinstance(pair_file_binding, dict), "V7 source pair-manifest binding is missing")
    _require(
        Path(str(pair_file_binding.get("path"))).expanduser().resolve() == pair_path
        and pair_file_binding.get("sha256") == sha256_file(pair_path)
        and pair_file_binding.get("manifest_sha256") == pairing["manifest_sha256"],
        "V7 source pair-manifest binding differs",
    )
    if contract == "scene_v7_tiny_overfit":
        parent_sha = source.get("parent_train32_sha256")
        _validate_sha(parent_sha, description="Tiny2 parent Train32")
    else:
        _require(source.get("parent_train32_sha256") is None, "Train32 parent binding must be null")
    return {
        "contract": contract,
        "variant": spec["variant"],
        "expected_rows": spec["rows"],
        "expected_quotas": spec["quotas"],
        "dataset_file": str(dataset_path),
        "row_manifest_file": str(rows_path),
        "pair_manifest_file": str(pair_path),
        "source_manifest_file": str(source_path),
        "artifact_bindings": artifact_bindings,
        "source_lock": source_lock,
        "source_manifest_sha256": source["manifest_sha256"],
        "pair_manifest_sha256": pairing["manifest_sha256"],
        "rows": rows,
        "pairing": pairing,
        "source_manifest": source,
    }


def _strict_score(record: Mapping[str, Any]) -> dict[str, Any]:
    prediction = record.get("parsed_json")
    gold = record.get("gold")
    expected = score_prediction("scene", prediction, gold)
    recorded = record.get("score_strict")
    if recorded is not None:
        _require(recorded == expected, "record strict score differs from generation")
    return expected


def _strict_exact(score: Mapping[str, Any]) -> bool:
    return bool(score.get("schema_valid")) and int(score.get("fp", 0)) == 0 and int(score.get("fn", 0)) == 0


def _strictly_better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_score = _strict_score(left)
    right_score = _strict_score(right)
    return (
        int(_strict_exact(left_score)),
        float(left_score["sample_f1"]),
        -int(left_score["fp"]),
        -int(left_score["fn"]),
    ) > (
        int(_strict_exact(right_score)),
        float(right_score["sample_f1"]),
        -int(right_score["fp"]),
        -int(right_score["fn"]),
    )


def build_gate(
    *,
    contract: str,
    records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    pairing: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical strict-generation overfit gate."""

    spec = _contract_spec(contract)
    expected_rows = int(spec["rows"])
    _require(set(records_by_condition) == set(CONDITIONS), "V7 gate conditions differ")
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    ordered_ordinals: list[int] | None = None
    for condition in CONDITIONS:
        records = list(records_by_condition[condition])
        _require(len(records) == expected_rows, f"V7 gate requires {expected_rows} {condition} rows")
        condition_rows: dict[int, Mapping[str, Any]] = {}
        ordinals: list[int] = []
        for record in records:
            _require(record.get("condition") == condition, "V7 gate record condition differs")
            ordinal = record.get("train_row_ordinal", record.get("selection_ordinal"))
            _require(
                isinstance(ordinal, int) and not isinstance(ordinal, bool) and 0 <= ordinal < expected_rows,
                "V7 gate record ordinal is invalid",
            )
            _require(ordinal not in condition_rows, "V7 gate record ordinal is duplicated")
            condition_rows[ordinal] = record
            ordinals.append(ordinal)
            _strict_score(record)
        _require(ordinals == list(range(expected_rows)), "V7 gate record order differs")
        if ordered_ordinals is None:
            ordered_ordinals = ordinals
        _require(ordinals == ordered_ordinals, "V7 gate condition row order differs")
        indexed[condition] = condition_rows

    correct = indexed["state_only"]
    donor = indexed["state_only_donor"]
    zero = indexed["state_only_no_write"]
    canonical_rows = sum(is_canonical_scene_prediction(row.get("parsed_json")) for row in correct.values())
    exact_rows = sum(_strict_exact(_strict_score(row)) for row in correct.values())
    strict_tp = sum(int(_strict_score(row)["tp"]) for row in correct.values())
    strict_fp = sum(int(_strict_score(row)["fp"]) for row in correct.values())
    strict_fn = sum(int(_strict_score(row)["fn"]) for row in correct.values())
    denominator = 2 * strict_tp + strict_fp + strict_fn
    strict_micro_f1 = 0.0 if denominator == 0 else 2 * strict_tp / denominator
    gold_positive_boundaries = strict_tp + strict_fn
    predicted_boundaries = strict_tp + strict_fp
    predicted_to_gold_ratio = (
        predicted_boundaries / gold_positive_boundaries
        if gold_positive_boundaries
        else 0.0
    )
    donor_raw_differences = sum(
        correct[index].get("raw_generation") != donor[index].get("raw_generation")
        for index in range(expected_rows)
    )
    zero_raw_differences = sum(
        correct[index].get("raw_generation") != zero[index].get("raw_generation")
        for index in range(expected_rows)
    )
    donor_wins = sum(_strictly_better(correct[index], donor[index]) for index in range(expected_rows))
    zero_wins = sum(_strictly_better(correct[index], zero[index]) for index in range(expected_rows))

    donor_current_scores = {
        index: _strict_score(donor[index]) for index in range(expected_rows)
    }
    zero_current_scores = {
        index: _strict_score(zero[index]) for index in range(expected_rows)
    }

    def micro_f1(scores: Iterable[Mapping[str, Any]]) -> float:
        score_rows = list(scores)
        true_positive = sum(int(score["tp"]) for score in score_rows)
        false_positive = sum(int(score["fp"]) for score in score_rows)
        false_negative = sum(int(score["fn"]) for score in score_rows)
        score_denominator = 2 * true_positive + false_positive + false_negative
        return (
            0.0
            if score_denominator == 0
            else 2 * true_positive / score_denominator
        )

    donor_current_f1 = micro_f1(donor_current_scores.values())
    zero_current_f1 = micro_f1(zero_current_scores.values())
    zero_exact_rows = sum(_strict_exact(score) for score in zero_current_scores.values())
    zero_raw_outputs = [str(zero[index].get("raw_generation")) for index in range(expected_rows)]
    zero_unique_raw_outputs = len(set(zero_raw_outputs))
    label_multiplicity = Counter(
        canonical_sha256(correct[index]["gold"]) for index in range(expected_rows)
    )
    max_label_multiplicity = max(label_multiplicity.values())
    _require(
        max_label_multiplicity == int(spec["max_label_multiplicity"]),
        "V7 frozen label multiplicity differs",
    )

    donor_by_ordinal = {
        int(entry["train_row_ordinal"]): int(entry["donor_train_row_ordinal"])
        for entry in pairing["directed_pairs"]
    }
    donor_identity_scores = {
        index: score_prediction(
            "scene",
            donor[index].get("parsed_json"),
            correct[donor_by_ordinal[index]].get("gold"),
        )
        for index in range(expected_rows)
    }
    donor_identity_exact_rows = sum(
        _strict_exact(score) for score in donor_identity_scores.values()
    )
    donor_identity_tp = sum(int(score["tp"]) for score in donor_identity_scores.values())
    donor_identity_fp = sum(int(score["fp"]) for score in donor_identity_scores.values())
    donor_identity_fn = sum(int(score["fn"]) for score in donor_identity_scores.values())
    donor_identity_denominator = (
        2 * donor_identity_tp + donor_identity_fp + donor_identity_fn
    )
    donor_identity_f1 = (
        0.0
        if donor_identity_denominator == 0
        else 2 * donor_identity_tp / donor_identity_denominator
    )

    pair_entries = pairing.get("directed_pairs")
    _require(isinstance(pair_entries, list) and len(pair_entries) == expected_rows, "V7 gate pairing is incomplete")
    same_cardinality_ordinals = sorted(
        int(entry["train_row_ordinal"])
        for entry in pair_entries
        if entry.get("target_stratum") == "same_cardinality_value"
    )
    expected_same = int(spec["quotas"]["same_cardinality_value"])
    _require(len(same_cardinality_ordinals) == expected_same, "V7 same-cardinality stratum differs")
    same_exact = sum(_strict_exact(_strict_score(correct[index])) for index in same_cardinality_ordinals)
    same_donor_wins = sum(_strictly_better(correct[index], donor[index]) for index in same_cardinality_ordinals)
    same_zero_wins = sum(_strictly_better(correct[index], zero[index]) for index in same_cardinality_ordinals)
    same_donor_identity_exact = sum(
        _strict_exact(donor_identity_scores[index])
        for index in same_cardinality_ordinals
    )
    empty_ordinals = [
        index
        for index in range(expected_rows)
        if not strict_gold_boundaries(correct[index]["gold"])
    ]
    _require(len(empty_ordinals) == int(spec["empty_rows"]), "V7 frozen empty-label stratum differs")
    empty_exact_rows = sum(
        _strict_exact(_strict_score(correct[index])) for index in empty_ordinals
    )

    recovered_outputs = sum(
        bool(row.get("score_recovered", {}).get("schema_recovered"))
        for row in correct.values()
    )
    metrics = {
        "rows": expected_rows,
        "canonical_correct_state_outputs": canonical_rows,
        "strict_exact_correct_state_rows": exact_rows,
        "strict_micro_f1": strict_micro_f1,
        "positive_gold_true_positives": strict_tp,
        "positive_gold_boundaries": gold_positive_boundaries,
        "predicted_to_gold_boundary_ratio": predicted_to_gold_ratio,
        "empty_gold": {
            "rows": len(empty_ordinals),
            "strict_exact_correct_state_rows": empty_exact_rows,
        },
        "correct_vs_donor": {
            "raw_generation_differences": donor_raw_differences,
            "strict_row_wins": donor_wins,
        },
        "correct_vs_zero": {
            "raw_generation_differences": zero_raw_differences,
            "strict_row_wins": zero_wins,
            "zero_unique_raw_outputs": zero_unique_raw_outputs,
            "zero_strict_exact_rows": zero_exact_rows,
            "correct_minus_zero_strict_exact_rows": exact_rows - zero_exact_rows,
            "correct_minus_zero_strict_micro_f1": strict_micro_f1 - zero_current_f1,
        },
        "same_cardinality": {
            "rows": expected_same,
            "strict_exact_correct_state_rows": same_exact,
            "correct_vs_donor_strict_row_wins": same_donor_wins,
            "correct_vs_zero_strict_row_wins": same_zero_wins,
            "donor_identity_strict_exact_rows": same_donor_identity_exact,
        },
        "donor_identity_recovery": {
            "scored_against": "predeclared_donor_gold",
            "strict_exact_rows": donor_identity_exact_rows,
            "strict_micro_f1": donor_identity_f1,
            "tp": donor_identity_tp,
            "fp": donor_identity_fp,
            "fn": donor_identity_fn,
            "authorization_gate": contract == "scene_v7_tiny_overfit",
        },
    }
    if contract == "scene_v7_tiny_overfit":
        gates = {
            "all_correct_state_outputs_canonical": canonical_rows == 2,
            "all_correct_state_rows_strict_exact": exact_rows == 2,
            "strict_micro_f1_is_one": strict_micro_f1 == 1.0,
            "all_positive_gold_boundaries_recovered_strictly": strict_tp == gold_positive_boundaries,
            "correct_raw_differs_from_donor_per_row": donor_raw_differences == 2,
            "correct_strictly_beats_donor_current_per_row": donor_wins == 2,
            "donor_identity_is_canonical_per_row": all(
                is_canonical_scene_prediction(donor[index].get("parsed_json"))
                for index in range(2)
            ),
            "donor_identity_is_strict_exact_per_row": donor_identity_exact_rows == 2,
            "zero_reset_outputs_are_identical": zero_unique_raw_outputs == 1,
            "zero_exact_does_not_exceed_label_multiplicity": zero_exact_rows <= 1,
            "correct_minus_zero_exact_gap_is_positive": exact_rows - zero_exact_rows >= 1,
            "correct_raw_differs_from_zero_on_distinct_label": zero_raw_differences >= 1,
            "same_cardinality_rows_all_strict_exact": same_exact == 2,
            "same_cardinality_rows_all_beat_donor": same_donor_wins == 2,
        }
    else:
        requirements = TRAIN32_GATE_REQUIREMENTS
        gates = {
            "canonical_correct_state_outputs": canonical_rows >= requirements["canonical_outputs"],
            "strict_exact_correct_state_rows": exact_rows >= requirements["strict_exact_rows"],
            "strict_positive_gold_true_positives": strict_tp >= requirements["strict_true_positives"],
            "empty_gold_strict_exact_rows": empty_exact_rows >= requirements["empty_exact_rows"],
            "same_cardinality_strict_exact_rows": same_exact >= requirements["same_cardinality_exact_rows"],
            "donor_identity_strict_exact_rows": donor_identity_exact_rows >= requirements["donor_identity_exact_rows"],
            "same_cardinality_donor_identity_strict_exact_rows": same_donor_identity_exact >= requirements["same_cardinality_donor_identity_exact_rows"],
            "correct_strictly_beats_donor_current_rows": donor_wins >= requirements["correct_beats_donor_rows"],
            "same_cardinality_correct_beats_donor_rows": same_donor_wins >= requirements["same_cardinality_correct_beats_donor_rows"],
            "zero_reset_outputs_are_identical": zero_unique_raw_outputs == 1,
            "zero_exact_does_not_exceed_label_multiplicity": zero_exact_rows <= requirements["max_zero_exact_rows"],
            "correct_minus_zero_strict_exact_rows": exact_rows - zero_exact_rows >= requirements["correct_minus_zero_exact_rows"],
            "correct_minus_zero_strict_micro_f1": strict_micro_f1 - zero_current_f1 >= requirements["correct_minus_zero_micro_f1"],
            "predicted_boundary_density": predicted_to_gold_ratio <= requirements["max_predicted_to_gold_boundary_ratio"],
            "state_minus_donor_strict_micro_f1": strict_micro_f1 - donor_current_f1 >= requirements["correct_minus_donor_micro_f1"],
            "state_minus_zero_causal_strict_micro_f1": strict_micro_f1 - zero_current_f1 >= requirements["correct_minus_zero_causal_micro_f1"],
        }
    passed = all(gates.values())
    return {
        "status": "pass" if passed else "fail",
        "all_gates_passed": passed,
        "contract": contract,
        "variant": spec["variant"],
        "criterion": "canonical_greedy_strict_generation_v1",
        "train32_gate_requirements": (
            dict(TRAIN32_GATE_REQUIREMENTS)
            if contract == "scene_v7_train32_overfit"
            else None
        ),
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "metrics": metrics,
        "gates": gates,
        "format_recovery_diagnostic_only": {
            "recovered_correct_state_outputs": recovered_outputs,
            "can_satisfy_gate": False,
        },
    }


def _record_with_self_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("record_sha256", None)
    result["record_sha256"] = canonical_sha256(result)
    return result


def validate_v7_resume_records(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
    donor_by_ordinal: Mapping[int, int],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[int, dict[str, Any]]:
    """Validate a safely resumable prefix of one condition output."""

    _require(condition in CONDITIONS, "unknown V7 resume condition")
    validated: dict[int, dict[str, Any]] = {}
    for position, raw_record in enumerate(records):
        _require(isinstance(raw_record, Mapping), "V7 resume record must be an object")
        record = dict(raw_record)
        _require(record.get("schema") == RECORD_SCHEMA, "V7 resume record schema differs")
        _validate_self_hash(record, field="record_sha256")
        _require(record.get("fingerprint") == fingerprint, "V7 resume record fingerprint differs")
        _require(record.get("condition") == condition, "V7 resume record condition differs")
        _require(record.get("split") == "train", "V7 resume record split differs")
        ordinal = record.get("train_row_ordinal")
        _require(ordinal == position, "V7 resume records must form an ordered prefix")
        _require(ordinal not in validated and ordinal < len(rows), "V7 resume record ordinal differs")
        sample = rows[ordinal]
        expected_donor = donor_by_ordinal[ordinal] if condition == "state_only_donor" else None
        literals = {
            "status": "ok",
            "task": TASK_NAME,
            "task_kind": "scene",
            "source_index": sample["source_index"],
            "row_sha256": sample["row_sha256"],
            "gold": sample["gold"],
            "donor_train_row_ordinal": expected_donor,
            "donor_source_index": None if expected_donor is None else rows[expected_donor]["source_index"],
            "donor_row_sha256": None if expected_donor is None else rows[expected_donor]["row_sha256"],
        }
        for field, expected in literals.items():
            _require(record.get(field) == expected, f"V7 resume record {field} differs")
        raw_generation = record.get("raw_generation")
        _require(isinstance(raw_generation, str), "V7 resume raw generation is invalid")
        _require(extract_json(raw_generation) == record.get("parsed_json"), "V7 resume parsed JSON differs")
        _require(record.get("score_strict") == score_prediction("scene", record.get("parsed_json"), sample["gold"]), "V7 resume strict score differs")
        _require(record.get("score_recovered") == recovered_scene_score(record.get("parsed_json"), sample["gold"]), "V7 resume recovery score differs")
        _require(
            record.get("input_rendered_sha256")
            == sample["token_metadata"]["generation_prefix_rendered_sha256"],
            "V7 resume generation prefix differs",
        )
        prime = record.get("prime")
        _require(isinstance(prime, dict), "V7 resume prime evidence is missing")
        prime_row = rows[expected_donor] if expected_donor is not None else sample
        _require(
            prime.get("rendered_sha256")
            == prime_row["token_metadata"]["write_rendered_sha256"],
            "V7 resume write prefix differs",
        )
        output_tokens = record.get("output_tokens")
        _require(
            isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and 0 <= output_tokens <= max_new_tokens,
            "V7 resume output-token count differs",
        )
        _require(record.get("hit_max_new_tokens") == (output_tokens >= max_new_tokens), "V7 resume max-token flag differs")
        validated[ordinal] = record
    return validated


def validate_existing_manifest(
    manifest: Mapping[str, Any], *, expected_fingerprint: str
) -> dict[str, Any]:
    _require(isinstance(manifest, Mapping), "existing V7 manifest must be an object")
    payload = manifest.get("fingerprint_payload")
    _require(isinstance(payload, dict), "existing V7 fingerprint payload is missing")
    _require(fingerprint_payload_sha256(payload) == manifest.get("fingerprint"), "existing V7 manifest self-fingerprint differs")
    _require(manifest.get("fingerprint") == expected_fingerprint, "existing V7 manifest fingerprint differs")
    return dict(manifest)


def build_receipt(
    *,
    contract: str,
    fingerprint: str,
    output_dir: Path,
    input_contract: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    outputs = {
        name: _artifact_binding(
            output_dir / filename,
            description=f"V7 {name}",
            expected_sha256=None,
        )
        for name, filename in {
            "manifest": "manifest.json",
            "state_only": "state_only.jsonl",
            "state_only_donor": "state_only_donor.jsonl",
            "state_only_no_write": "state_only_no_write.jsonl",
            "summary": "summary.json",
        }.items()
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "contract": contract,
        "variant": _contract_spec(contract)["variant"],
        "evaluation_fingerprint": fingerprint,
        "input_artifacts": input_contract["artifact_bindings"],
        "source_lock": input_contract["source_lock"],
        "source_manifest_sha256": input_contract["source_manifest_sha256"],
        "pair_manifest_sha256": input_contract["pair_manifest_sha256"],
        "checkpoint": dict(checkpoint),
        "outputs": outputs,
        "gate": dict(gate),
        "hard32_authorization": {
            "authorized": (
                contract == "scene_v7_train32_overfit"
                and gate.get("status") == "pass"
            ),
            "checkpoint_binding": dict(checkpoint),
            "scope": "fixed_hard32_only_no_full170",
        },
    }
    receipt["receipt_sha256"] = self_hash_payload(receipt)
    return receipt


def validate_receipt(
    receipt: Path | str | Mapping[str, Any],
    *,
    expected_contract: str | None = None,
    expected_fingerprint: str | None = None,
    expected_artifact_sha256: Mapping[str, str] | None = None,
    expected_checkpoint: Mapping[str, Any] | None = None,
    require_pass: bool = False,
) -> dict[str, Any]:
    """Validate receipt self-hash, output files, input locks, and checkpoint."""

    if isinstance(receipt, Mapping):
        payload = dict(receipt)
        receipt_path = None
    else:
        receipt_path = _regular_file(Path(receipt), description="V7 evaluation receipt")
        payload = _load_json(receipt_path, description="V7 evaluation receipt")
    _require(payload.get("schema") == RECEIPT_SCHEMA, "V7 receipt schema differs")
    _validate_self_hash(payload, field="receipt_sha256")
    if expected_contract is not None:
        _require(payload.get("contract") == expected_contract, "V7 receipt contract differs")
    if expected_fingerprint is not None:
        _require(payload.get("evaluation_fingerprint") == expected_fingerprint, "V7 receipt fingerprint differs")
    expected_artifact_sha256 = expected_artifact_sha256 or {}
    input_artifacts = payload.get("input_artifacts")
    _require(isinstance(input_artifacts, dict), "V7 receipt input artifacts are missing")
    for name, binding in input_artifacts.items():
        _require(isinstance(binding, dict), f"V7 receipt input binding is invalid: {name}")
        path = _regular_file(Path(str(binding.get("path"))), description=f"V7 input {name}")
        actual = sha256_file(path)
        _require(actual == binding.get("actual_sha256"), f"V7 receipt input SHA differs: {name}")
        if binding.get("expected_sha256") is not None:
            _require(actual == binding.get("expected_sha256"), f"V7 receipt expected SHA differs: {name}")
        if name in expected_artifact_sha256:
            _require(actual == expected_artifact_sha256[name], f"V7 caller expected SHA differs: {name}")
    source_lock = payload.get("source_lock")
    _require(isinstance(source_lock, dict), "V7 receipt source-lock binding is missing")
    source_lock_path = _regular_file(
        Path(str(source_lock.get("path"))), description="V7 receipt source lock"
    )
    _require(
        sha256_file(source_lock_path) == source_lock.get("file_sha256"),
        "V7 receipt source-lock file SHA differs",
    )
    source_lock_payload = _load_json(source_lock_path, description="V7 receipt source lock")
    _validate_self_hash(source_lock_payload, field="lock_sha256")
    _require(
        source_lock_payload["lock_sha256"] == source_lock.get("lock_sha256"),
        "V7 receipt source-lock canonical digest differs",
    )
    outputs = payload.get("outputs")
    _require(isinstance(outputs, dict), "V7 receipt outputs are missing")
    for name, binding in outputs.items():
        _require(isinstance(binding, dict), f"V7 receipt output binding is invalid: {name}")
        path = _regular_file(Path(str(binding.get("path"))), description=f"V7 output {name}")
        _require(sha256_file(path) == binding.get("actual_sha256"), f"V7 receipt output SHA differs: {name}")
    if expected_checkpoint is not None:
        _require(payload.get("checkpoint") == dict(expected_checkpoint), "V7 receipt checkpoint differs")
    if require_pass:
        _require(payload.get("gate", {}).get("status") == "pass", "V7 receipt gate did not pass")
    result = dict(payload)
    if receipt_path is not None:
        result["receipt_path"] = str(receipt_path)
        result["receipt_file_sha256"] = sha256_file(receipt_path)
    return result


def validate_fixed_hard32_authorization(
    receipt: Path | str | Mapping[str, Any],
    *,
    expected_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only fixed Hard32 for the exact Train32-passing checkpoint."""

    payload = validate_receipt(
        receipt,
        expected_contract="scene_v7_train32_overfit",
        expected_checkpoint=expected_checkpoint,
        require_pass=True,
    )
    authorization = payload.get("hard32_authorization")
    _require(isinstance(authorization, dict), "V7 Hard32 authorization is missing")
    _require(authorization.get("authorized") is True, "V7 receipt does not authorize fixed Hard32")
    _require(
        authorization.get("scope") == "fixed_hard32_only_no_full170",
        "V7 receipt Hard32 scope differs",
    )
    _require(
        authorization.get("checkpoint_binding") == dict(expected_checkpoint),
        "V7 Hard32 checkpoint binding differs",
    )
    source_lock = payload.get("source_lock")
    _require(isinstance(source_lock, dict), "V7 Hard32 source-lock binding is missing")
    _require(
        Path(str(source_lock.get("path"))).resolve() == DEFAULT_SOURCE_LOCK.resolve(),
        "V7 Hard32 authorization requires the repository source lock",
    )
    validate_v7_source_lock(
        DEFAULT_SOURCE_LOCK,
        contract="scene_v7_train32_overfit",
        artifact_bindings=payload["input_artifacts"],
    )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=tuple(CONTRACT_SPECS), required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--row-manifest-file", type=Path, required=True)
    parser.add_argument("--pair-manifest-file", type=Path, required=True)
    parser.add_argument("--source-manifest-file", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-row-manifest-sha256", required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--source-lock-file", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delta-mem-root", default=str(PROJECT_ROOT))
    parser.add_argument("--expected-memory-layer-count", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--normal-fusion-profile", default="native", choices=("native",))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_v7_checkpoint(
    memory_dir: Path | str,
    *,
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completed V7 generation-objective checkpoint."""

    memory_dir = Path(memory_dir).expanduser().resolve()
    config = _regular_file(memory_dir / "delta_mem_config.json", description="memory config")
    adapter = _regular_file(memory_dir / "delta_mem_adapter.pt", description="memory adapter")
    protocol_path = _regular_file(
        memory_dir / "training_protocol.json", description="V7 training protocol"
    )
    trainer_state_path = _regular_file(
        memory_dir / "trainer_state.json", description="V7 trainer state"
    )
    pairing_path = _regular_file(
        memory_dir / "scene_state_identity_pairing_manifest.json",
        description="V7 materialized pairing manifest",
    )
    for filename in ("optimizer.pt", "scheduler.pt"):
        _regular_file(memory_dir / filename, description=f"V7 checkpoint {filename}")
    _require(
        any(path.is_file() and not path.is_symlink() for path in memory_dir.glob("rng_state*.pth")),
        "V7 checkpoint RNG state is missing",
    )

    architecture = memory_architecture_contract(memory_dir)
    _require(architecture["target_layers"] == list(range(42)), "V7 checkpoint target layers differ")
    _require(architecture["delta_heads"] == ["q", "o"], "V7 checkpoint delta heads differ")
    _require(architecture["rank"] == 4, "V7 checkpoint rank differs")
    _require(architecture["rwkv_ms_semantics_version"] == 2, "V7 checkpoint semantics version differs")
    _require(architecture["memory_backend"] == "rwkv_ms", "V7 checkpoint memory backend differs")

    protocol = _load_json(protocol_path, description="V7 training protocol")
    expected_literals = {
        "schema_version": 10,
        "memory_objective_version": "scene_state_generation_ce_v1",
        "memory_loss_mode": "scene_state_generation_ce",
        "training_mode": "episode",
        "assistant_loss_mode": "final_assistant_only",
        "episode_recent_messages": 0,
        "episode_read_write_enabled": False,
        "max_write_length": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "frozen_mlp_activation_checkpointing": True,
        "validation_split_ratio": 0.0,
        "eval_samples": 0,
        "scene_boundary_payload_ce_weight": 0.0,
        "memory_dropout_no_memory_prob": 0.0,
        "memory_dropout_state_only_prob": 0.0,
        "memory_base_kl_weight": 0.0,
        "memory_kl_weight": 0.0,
        "memory_representation_weight": 0.0,
        "write_sparsity_weight": 0.0,
        "scene_generation_read_protocol": (
            "exact_system_only_generation_prefix_same_read_correct_donor_zero_v1"
        ),
        "scene_generation_zero_protocol": (
            "adapter_active_reset_state_writes_disabled_detached_reference_v1"
        ),
    }
    for field, expected in expected_literals.items():
        _require(protocol.get(field) == expected, f"V7 training protocol {field} differs")
    expected_formula = (
        "weighted_generation_ce(schema=2,decision=4,termination=1) + "
        "first_wrong_gold_prefix_top1_hinge(0.2) + "
        "correct_source_vs_donor_two_token_ce + "
        "donor_donor_vs_source_two_token_ce + "
        "correct_vs_detached_zero_decision_margin_hinge(0.2)"
    )
    _require(
        protocol.get("scene_generation_objective_formula") == expected_formula,
        "V7 training objective formula differs",
    )
    dataset_path = Path(str(input_contract["dataset_file"])).resolve()
    source_path = Path(str(input_contract["source_manifest_file"])).resolve()
    pair_path = Path(str(input_contract["pair_manifest_file"])).resolve()
    _require(Path(str(protocol.get("train_file"))).resolve() == dataset_path, "V7 checkpoint train file differs")
    _require(protocol.get("train_samples") == input_contract["expected_rows"], "V7 checkpoint train sample count differs")
    source_identity = protocol.get("scene_state_source_manifest")
    _require(isinstance(source_identity, dict), "V7 checkpoint source identity is missing")
    source_expected = {
        "path": str(source_path),
        "file_sha256": input_contract["artifact_bindings"]["source_manifest"]["actual_sha256"],
        "schema": SOURCE_SCHEMA,
        "train_file": str(dataset_path),
        "train_file_sha256": input_contract["artifact_bindings"]["dataset"]["actual_sha256"],
        "train_rows": input_contract["expected_rows"],
        "train_source_split": "train",
        "episode_contract": EPISODE_CONTRACT,
    }
    _require(source_identity == source_expected, "V7 checkpoint source-manifest identity differs")

    materialized_pairing = _load_json(pairing_path, description="V7 materialized pairing manifest")
    _validate_self_hash(materialized_pairing, field="manifest_sha256")
    _require(materialized_pairing.get("schema_version") == 2, "V7 materialized pairing schema differs")
    _require(
        materialized_pairing.get("objective_version") == "scene_state_generation_ce_v1",
        "V7 materialized pairing objective differs",
    )
    _require(set(materialized_pairing.get("splits", {})) == {"train"}, "V7 materialized pairing splits differ")
    train_pairing = materialized_pairing["splits"]["train"]
    _require(isinstance(train_pairing, dict), "V7 materialized train pairing is missing")
    _validate_self_hash(train_pairing, field="manifest_sha256")
    pairing_expected = {
        "sample_count": input_contract["expected_rows"],
        "source_pair_manifest_path": str(pair_path),
        "source_pair_manifest_file_sha256": input_contract["artifact_bindings"]["pair_manifest"]["actual_sha256"],
        "source_pair_manifest_sha256": input_contract["pair_manifest_sha256"],
        "source_entries_sha256": input_contract["pairing"]["entries_sha256"],
        "target_stratum_row_counts": input_contract["expected_quotas"],
    }
    for field, expected in pairing_expected.items():
        _require(train_pairing.get(field) == expected, f"V7 materialized pairing {field} differs")
    protocol_pairing = protocol.get("scene_state_identity_pairing")
    _require(isinstance(protocol_pairing, dict), "V7 protocol pairing summary is missing")
    _require(
        protocol_pairing.get("manifest_sha256") == materialized_pairing["manifest_sha256"],
        "V7 protocol pairing manifest digest differs",
    )
    _require(
        protocol_pairing.get("target_stratum_row_counts") == input_contract["expected_quotas"],
        "V7 protocol pairing quotas differ",
    )

    trainer_state = _load_json(trainer_state_path, description="V7 trainer state")
    global_step = trainer_state.get("global_step")
    max_steps = trainer_state.get("max_steps")
    _require(
        isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and global_step > 0
        and global_step == max_steps
        and global_step == protocol.get("max_steps"),
        "V7 checkpoint is not a completed training horizon",
    )
    if memory_dir.name.startswith("checkpoint-"):
        suffix = memory_dir.name.removeprefix("checkpoint-")
        _require(suffix.isdigit() and int(suffix) == global_step, "V7 checkpoint directory step differs")
    return {
        "memory_dir": str(memory_dir),
        "config_sha256": sha256_file(config),
        "adapter_sha256": sha256_file(adapter),
        "training_protocol_sha256": sha256_file(protocol_path),
        "trainer_state_sha256": sha256_file(trainer_state_path),
        "materialized_pairing_file_sha256": sha256_file(pairing_path),
        "materialized_pairing_sha256": materialized_pairing["manifest_sha256"],
        "global_step": global_step,
        "max_steps": max_steps,
        "objective_version": protocol["memory_objective_version"],
        "architecture": architecture,
        "source_manifest_sha256": source_identity["file_sha256"],
        "train_file_sha256": source_identity["train_file_sha256"],
        "source_pair_manifest_sha256": train_pairing["source_pair_manifest_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _require(args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS, "V7 protected evaluation requires max_new_tokens=128")
    input_contract = validate_v7_contract(
        contract=args.contract,
        dataset_file=args.dataset_file,
        row_manifest_file=args.row_manifest_file,
        pair_manifest_file=args.pair_manifest_file,
        source_manifest_file=args.source_manifest_file,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_row_manifest_sha256=args.expected_row_manifest_sha256,
        expected_pair_manifest_sha256=args.expected_pair_manifest_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        source_lock_file=args.source_lock_file,
    )
    rows = input_contract["rows"]
    donor_by_ordinal = input_contract["pairing"]["donor_by_ordinal"]
    memory_dir = args.memory_dir.expanduser().resolve()
    args.memory_dir = memory_dir
    output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = output_dir
    args.delta_mem_root = str(Path(args.delta_mem_root).expanduser().resolve())
    _require(Path(args.delta_mem_root) == PROJECT_ROOT, "delta-mem root must be this evaluator checkout")
    checkpoint = validate_v7_checkpoint(
        memory_dir,
        input_contract=input_contract,
    )
    expected_layers = resolved_memory_layer_count(memory_dir, args.expected_memory_layer_count)
    architecture = memory_architecture_contract(memory_dir)
    base_model = Path(args.base_model).expanduser().resolve()
    args.base_model = str(base_model)
    fingerprint_payload = {
        "schema_version": 1,
        "contract": args.contract,
        "variant": input_contract["variant"],
        "split": "train",
        "input_artifacts": input_contract["artifact_bindings"],
        "source_lock": input_contract["source_lock"],
        "source_manifest_sha256": input_contract["source_manifest_sha256"],
        "pair_manifest_sha256": input_contract["pair_manifest_sha256"],
        "rows": [
            {
                "train_row_ordinal": row["train_row_ordinal"],
                "source_index": row["source_index"],
                "row_sha256": row["row_sha256"],
                "donor_train_row_ordinal": donor_by_ordinal[row["train_row_ordinal"]],
            }
            for row in rows
        ],
        "generation_prefix_input_ids_sha256": rows[0]["token_metadata"][
            "generation_prefix_input_ids_sha256"
        ],
        "generation_prefix_rendered_sha256": rows[0]["token_metadata"][
            "generation_prefix_rendered_sha256"
        ],
        "checkpoint": checkpoint,
        "base_model": str(base_model),
        "base_model_weights": base_model_weight_identity(base_model),
        "base_model_prompt_artifacts": base_model_prompt_identity(base_model),
        "memory_architecture": architecture,
        "expected_memory_layer_count": expected_layers,
        "runtime": {
            "conditions": list(CONDITIONS),
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "do_sample": False,
            "use_cache_generation": True,
            "prime_use_cache": False,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "normal_fusion_profile": "native",
            "packages": runtime_package_versions(),
        },
        "code": {
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "state_evaluator_sha256": sha256_file(SCRIPT_DIR / "run_scene_state_eval.py"),
        },
    }
    fingerprint = fingerprint_payload_sha256(fingerprint_payload)
    manifest = {
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "evaluation_kind": "protected V7 train-split strict generation overfit gate",
    }
    output_paths = {
        condition: output_dir / f"{condition}.jsonl" for condition in CONDITIONS
    }
    output_paths.update(
        {
            "manifest": output_dir / "manifest.json",
            "summary": output_dir / "summary.json",
            "receipt": output_dir / "gate_receipt.json",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
    manifest_path = output_paths["manifest"]
    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path, description="existing V7 manifest")
        manifest = validate_existing_manifest(existing_manifest, expected_fingerprint=fingerprint)
    else:
        atomic_write_json(manifest_path, manifest)

    completed: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        path = output_paths[condition]
        raw_records = [record for _, record in _read_jsonl(path, description=f"V7 {condition} output")] if path.exists() else []
        completed[condition] = validate_v7_resume_records(
            raw_records,
            condition=condition,
            fingerprint=fingerprint,
            rows=rows,
            donor_by_ordinal=donor_by_ordinal,
        )

    if any(len(completed[condition]) < len(rows) for condition in CONDITIONS):
        model, tokenizer, runtime_profile = load_adapter_model(args, expected_layers)
        runtime_prefixes = validate_runtime_prefixes(tokenizer, rows=rows)
        recorded_prefixes = manifest.get("runtime_prefixes")
        if recorded_prefixes is not None:
            _require(recorded_prefixes == runtime_prefixes, "V7 resume runtime prefixes differ")
        else:
            manifest["runtime_prefixes"] = runtime_prefixes
        manifest_runtime = manifest.get("runtime_fusion_profile")
        if manifest_runtime is not None:
            _require(manifest_runtime == runtime_profile, "V7 resume runtime profile differs")
        else:
            manifest["runtime_fusion_profile"] = runtime_profile
            atomic_write_json(manifest_path, manifest)
        try:
            for condition in CONDITIONS:
                for ordinal, sample in enumerate(rows):
                    if ordinal in completed[condition]:
                        continue
                    donor_ordinal = donor_by_ordinal[ordinal]
                    donor_sample = rows[donor_ordinal]
                    result = evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        donor_sample=donor_sample,
                        condition=condition,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        device=args.device,
                        collect_semantic_nll=False,
                    )
                    record = _record_with_self_hash(
                        {
                            "schema": RECORD_SCHEMA,
                            "status": "ok",
                            "completed_at": utc_now(),
                            "fingerprint": fingerprint,
                            "condition": condition,
                            "task": TASK_NAME,
                            "task_kind": "scene",
                            "split": "train",
                            "train_row_ordinal": ordinal,
                            "source_index": sample["source_index"],
                            "row_sha256": sample["row_sha256"],
                            "gold": sample["gold"],
                            "donor_train_row_ordinal": donor_ordinal if condition == "state_only_donor" else None,
                            **result,
                        }
                    )
                    completed[condition][ordinal] = record
                    atomic_write_jsonl(
                        output_paths[condition],
                        [completed[condition][index] for index in sorted(completed[condition])],
                    )
        finally:
            del model
            del tokenizer
            clear_model_memory()

    ordered_records = {
        condition: [completed[condition][index] for index in range(len(rows))]
        for condition in CONDITIONS
    }
    summaries = {
        condition: summarize_records(records)
        for condition, records in ordered_records.items()
    }
    gate = build_gate(
        contract=args.contract,
        records_by_condition=ordered_records,
        pairing=input_contract["pairing"],
    )
    summary = {
        "schema": "rwkv_ms_scene_v7_overfit_summary.v1",
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "complete": True,
        "contract": args.contract,
        "variant": input_contract["variant"],
        "split": "train",
        "conditions": summaries,
        "comparisons": build_comparisons(summaries),
        "gate": gate,
        "format_recovery_is_diagnostic_only": True,
    }
    summary["summary_sha256"] = self_hash_payload(summary, hash_field="summary_sha256")
    atomic_write_json(output_paths["summary"], summary)
    receipt = build_receipt(
        contract=args.contract,
        fingerprint=fingerprint,
        output_dir=output_dir,
        input_contract=input_contract,
        checkpoint=checkpoint,
        gate=gate,
    )
    atomic_write_json(output_paths["receipt"], receipt)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
