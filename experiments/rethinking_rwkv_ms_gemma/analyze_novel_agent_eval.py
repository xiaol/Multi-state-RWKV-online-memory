#!/usr/bin/env python3
"""Build a conservative format-recovery diagnostic for a frozen evaluation.

The analyzer never regenerates model output. It validates the completed strict
evaluation and dataset rows, recovers only explicitly defined schema variants,
and writes a deterministic JSON summary with paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable

try:
    from .run_novel_agent_eval import score_prediction
except ImportError:
    from run_novel_agent_eval import score_prediction


SUPPORTED_CONDITIONS = ("base", "normal", "no_write")
CONDITIONS = ("base", "normal")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_724
SCENE_V6_CONTRACT_ROWS = {
    "scene_v6_validation": ("val", 170),
    "scene_v6_final_test": ("test", 149),
}
SCENE_V6_RECOVERED_MICRO_F1_FLOOR = 0.37
SCENE_V6_NORMAL_MINUS_NO_WRITE_FLOOR = 0.05
SCENE_V6_COVERAGE_FLOOR = 0.95
SCENE_V6_MAX_TOKEN_HIT_RATE_DELTA_CEILING = 0.05
OFFICIAL_SCENE_V4_DATASET_REVISION = "5d3040d21f51b3ce90b9396b058e552c47f43cd5"
OFFICIAL_SCENE_V4_SHA256 = {
    "val": "61e94bcc536a124b07aef2c38ba285d7073d94a223866b58ddc7e5e1f509d513",
    "test": "d8b50ca3862bd40f023155bd14aa7b25d9d5dd3db4ea1c4d5a7e6f4f79cdfd6d",
}
ATTRIBUTION_ALIASES = frozenset(
    {
        "best_candidate",
        "selected_character",
        "chosen_character",
        "target_character",
        "character",
        "role",
        "speaker",
        "角色",
    }
)
NARRATIVE_TYPES = frozenset(
    {"dialogue", "narration", "thought", "action", "scene_description"}
)
CORE_SELECTION_TASKS = (
    "attribution-v3.2",
    "narrative-v3.2",
    "scene-v4-current",
)
SELECTION_METRIC_FLOORS = {
    "attribution-v3.2": -(1 / 30),
    "narrative-v3.2": -0.02,
    "scene-v4-current": -0.03,
}
SELECTION_CI_UPPER_FLOOR = 0.0
SELECTION_COVERAGE_DELTA_FLOOR = -0.05
SELECTION_MAX_TOKEN_HIT_RATE_DELTA_CEILING = 0.05
CLEAN_SELECTION_EXCLUSIONS = {
    "val": {"narrative-v3.2": (25,)},
}
CLEAN_SELECTION_EXCLUSION_NOTES = {
    ("val", "narrative-v3.2"): (
        "Zero-based validation row 25 duplicates training content with conflicting labels; "
        "it remains in all-row metrics and is excluded only from checkpoint selection."
    ),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    relative_path: str
    kind: str
    expected_rows: int


@dataclass(frozen=True)
class DatasetSample:
    line_index: int
    row_sha256: str
    gold: Any
    candidates: tuple[str, ...]
    paragraph_count: int | None


TASK_LAYOUT = (
    ("attribution-v3.2", "v3.2-attribution-best-candidate", "attribution"),
    ("narrative-v3.2", "v3.2-narrative-type-classification", "narrative"),
    ("scene-v3.2", "v3.2-scene-boundary-detection", "scene"),
    ("scene-v4-current", "v4-scene-boundary-detection", "scene"),
)
EXPECTED_ROWS_BY_SPLIT = {
    "val": {
        "attribution-v3.2": 30,
        "narrative-v3.2": 39,
        "scene-v3.2": 35,
        "scene-v4-current": 170,
    },
    "test": {
        "attribution-v3.2": 30,
        "narrative-v3.2": 39,
        "scene-v3.2": 35,
        "scene-v4-current": 149,
    },
}


def task_specs(split: str) -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(
            name=name,
            relative_path=f"{relative_dir}/{split}.jsonl",
            kind=kind,
            expected_rows=EXPECTED_ROWS_BY_SPLIT[split][name],
        )
        for name, relative_dir, kind in TASK_LAYOUT
    )


def evaluation_task_specs(eval_dir: Path, split: str) -> tuple[TaskSpec, ...]:
    manifest = read_json(eval_dir / "manifest.json")
    datasets = manifest.get("fingerprint_payload", {}).get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        return task_specs(split)
    layout_by_name = {
        name: (relative_dir, kind) for name, relative_dir, kind in TASK_LAYOUT
    }
    specs: list[TaskSpec] = []
    for name, _, _ in TASK_LAYOUT:
        dataset = datasets.get(name)
        if not isinstance(dataset, dict):
            continue
        relative_dir, kind = layout_by_name[name]
        selected_rows = dataset.get("selected_rows")
        if not isinstance(selected_rows, int) or selected_rows <= 0:
            raise ValueError(f"Invalid selected row count for {name}: {selected_rows}")
        specs.append(
            TaskSpec(
                name=name,
                relative_path=f"{relative_dir}/{split}.jsonl",
                kind=kind,
                expected_rows=selected_rows,
            )
        )
    if not specs:
        raise ValueError("Evaluation manifest selects no recognized tasks")
    return tuple(specs)


def evaluation_conditions(eval_dir: Path) -> tuple[str, ...]:
    manifest = read_json(eval_dir / "manifest.json")
    raw_conditions = manifest.get("fingerprint_payload", {}).get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        return ("base", "normal")
    if any(
        not isinstance(condition, str) or condition not in SUPPORTED_CONDITIONS
        for condition in raw_conditions
    ):
        raise ValueError(f"Evaluation manifest has invalid conditions: {raw_conditions}")
    if len(set(raw_conditions)) != len(raw_conditions):
        raise ValueError("Evaluation manifest conditions contain duplicates")
    return tuple(raw_conditions)


def evaluation_contract(eval_dir: Path) -> dict[str, Any]:
    manifest = read_json(eval_dir / "manifest.json")
    contract = manifest.get("evaluation_contract")
    fingerprint_contract = manifest.get("fingerprint_payload", {}).get(
        "evaluation_contract"
    )
    if contract is None and fingerprint_contract is None:
        return {"name": "generic", "phase": "generic"}
    if not isinstance(contract, dict) or contract != fingerprint_contract:
        raise ValueError(
            "Evaluation contract is missing or differs between manifest and fingerprint"
        )
    return dict(contract)


TASK_SPECS = task_specs("test")
EXPECTED_ROWS_PER_CONDITION = sum(spec.expected_rows for spec in TASK_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("val", "test"))
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_payload_sha256(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Manifest fingerprint_payload must be an object")
    return sha256_text(json.dumps(payload, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def infer_evaluation_split(eval_dir: Path) -> str:
    manifest = read_json(eval_dir / "manifest.json")
    fingerprint_payload = manifest.get("fingerprint_payload", {})
    split = manifest.get("split") or fingerprint_payload.get("split")
    if split in EXPECTED_ROWS_BY_SPLIT:
        return str(split)
    dataset_names = {
        Path(str(dataset["path"])).name
        for dataset in fingerprint_payload.get("datasets", {}).values()
        if isinstance(dataset, dict) and dataset.get("path")
    }
    inferred = {name.removesuffix(".jsonl") for name in dataset_names}
    if len(inferred) == 1 and next(iter(inferred)) in EXPECTED_ROWS_BY_SPLIT:
        return next(iter(inferred))
    raise ValueError("Could not infer evaluation split from manifest; pass --split")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def extract_json(text: str) -> Any | None:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def resolve_training_root(dataset_root: Path) -> Path:
    candidates = (dataset_root, dataset_root / "training")
    for candidate in candidates:
        if all((candidate / spec.relative_path).is_file() for spec in TASK_SPECS):
            return candidate.resolve()
    expected = ", ".join(spec.relative_path for spec in TASK_SPECS)
    raise FileNotFoundError(
        f"Could not find all task files below {dataset_root}; expected {expected}"
    )


def parse_candidates(user_content: str) -> tuple[str, ...]:
    candidate_block = user_content.split("上下文:", maxsplit=1)[0]
    candidates = tuple(
        match.group(1).strip()
        for match in re.finditer(r"(?m)^-\s*(.+?)\s*$", candidate_block)
    )
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Attribution candidates are empty or duplicated")
    return candidates


def parse_paragraph_count(user_content: str) -> int:
    paragraph_ids = [
        int(match.group(1))
        for match in re.finditer(r"\[P(\d+)\]", user_content)
    ]
    if not paragraph_ids:
        raise ValueError("Scene prompt contains no paragraph identifiers")
    paragraph_count = max(paragraph_ids)
    if set(paragraph_ids) != set(range(1, paragraph_count + 1)):
        raise ValueError("Scene prompt paragraph identifiers are not contiguous")
    return paragraph_count


def load_dataset_samples(
    training_root: Path,
) -> tuple[dict[str, list[DatasetSample]], dict[str, dict[str, Any]]]:
    samples_by_task: dict[str, list[DatasetSample]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for spec in TASK_SPECS:
        path = training_root / spec.relative_path
        samples: list[DatasetSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 3:
                    raise ValueError(f"Invalid messages in {path}")
                user_content = str(messages[-2].get("content", ""))
                gold = extract_json(str(messages[-1].get("content", "")))
                if gold is None:
                    raise ValueError(f"Invalid gold JSON in {path}")
                candidates: tuple[str, ...] = ()
                paragraph_count: int | None = None
                if spec.kind == "attribution":
                    candidates = parse_candidates(user_content)
                    if not isinstance(gold, dict) or gold.get("best_candidate") not in candidates:
                        raise ValueError(f"Attribution gold is outside candidates in {path}")
                elif spec.kind == "narrative":
                    gold_label_map(gold)
                else:
                    paragraph_count = parse_paragraph_count(user_content)
                    gold_boundaries = strict_gold_boundaries(gold)
                    if any(
                        boundary < 1 or boundary >= paragraph_count
                        for boundary in gold_boundaries
                    ):
                        raise ValueError(f"Scene gold has an invalid boundary in {path}")
                samples.append(
                    DatasetSample(
                        line_index=len(samples),
                        row_sha256=sha256_text(raw_line.rstrip("\n")),
                        gold=gold,
                        candidates=candidates,
                        paragraph_count=paragraph_count,
                    )
                )
                if len(samples) >= spec.expected_rows:
                    break
        if len(samples) != spec.expected_rows:
            raise ValueError(
                f"Expected {spec.expected_rows} rows for {spec.name}, found {len(samples)}"
            )
        samples_by_task[spec.name] = samples
        provenance[spec.name] = {
            "path": str(path.resolve()),
            "rows": len(samples),
            "sha256": sha256_file(path),
        }
    return samples_by_task, provenance


def gold_label_map(gold: Any) -> dict[str, str]:
    if not isinstance(gold, dict) or not isinstance(gold.get("labels"), list):
        raise ValueError("Narrative gold does not contain a labels list")
    labels: dict[str, str] = {}
    for item in gold["labels"]:
        if not isinstance(item, dict):
            raise ValueError("Narrative gold contains a non-object label")
        unit_id = normalize_unit_id(item.get("unit_id"))
        label_type = item.get("type")
        if unit_id is None or label_type not in NARRATIVE_TYPES:
            raise ValueError("Narrative gold contains an invalid label")
        if unit_id in labels:
            raise ValueError("Narrative gold contains a duplicate unit_id")
        labels[unit_id] = label_type
    return labels


def strict_gold_boundaries(gold: Any) -> set[int]:
    if not isinstance(gold, dict) or not isinstance(gold.get("boundaries"), list):
        raise ValueError("Scene gold does not contain a boundaries list")
    boundaries: set[int] = set()
    for item in gold["boundaries"]:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("Scene gold contains a non-integer boundary")
        boundaries.add(item)
    return boundaries


def aligned_qwen_scene_reference(
    strict_summary: dict[str, Any],
    samples: list[DatasetSample],
    *,
    split: str,
) -> dict[str, Any]:
    if split != "test":
        return {
            "status": "not_applicable",
            "reason": "The aligned Qwen artifact covers only the untouched test split.",
        }
    reference_root = strict_summary.get("references")
    if not isinstance(reference_root, dict):
        raise ValueError("Strict summary is missing reference metadata")
    reference = reference_root.get("scene-v4-current")
    source_hashes = reference_root.get("source_hashes")
    if not isinstance(reference, dict) or not isinstance(source_hashes, dict):
        raise ValueError("Strict summary is missing the aligned Qwen scene reference")
    source = reference.get("artifact_source")
    expected_hash = source_hashes.get("scene_boundary_final.json")
    if not isinstance(source, str) or not isinstance(expected_hash, str):
        raise ValueError("Aligned Qwen scene reference lacks source provenance")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or sha256_file(source_path) != expected_hash:
        raise ValueError("Aligned Qwen scene reference source hash differs")
    payload = read_json(source_path)
    rows = payload.get("v4-590", {}).get("per_sample")
    if not isinstance(rows, list) or len(rows) != len(samples):
        raise ValueError(
            "Aligned Qwen scene reference row count differs from the official test rows"
        )
    contributions: list[tuple[int, int, int]] = []
    predictions: list[set[int]] = []
    for index, (row, sample) in enumerate(zip(rows, samples, strict=True)):
        if not isinstance(row, dict) or row.get("id") != index:
            raise ValueError(f"Aligned Qwen row identity differs at test index {index}")
        gold = strict_gold_boundaries(sample.gold)
        if row.get("gold") != sorted(gold):
            raise ValueError(f"Aligned Qwen gold differs at test index {index}")
        if row.get("paras") != sample.paragraph_count:
            raise ValueError(
                f"Aligned Qwen paragraph count differs at test index {index}"
            )
        raw_prediction = row.get("pred")
        if not isinstance(raw_prediction, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_prediction
        ):
            raise ValueError(f"Aligned Qwen prediction is invalid at test index {index}")
        prediction = set(raw_prediction)
        tp = len(prediction & gold)
        fp = len(prediction - gold)
        fn = len(gold - prediction)
        if (row.get("tp"), row.get("fp"), row.get("fn")) != (tp, fp, fn):
            raise ValueError(f"Aligned Qwen counts differ at test index {index}")
        contributions.append((tp, fp, fn))
        predictions.append(prediction)
    alignment_source = reference.get("alignment_manifest_source")
    alignment_hash = reference.get("alignment_manifest_sha256")
    if not isinstance(alignment_source, str) or not isinstance(alignment_hash, str):
        return {
            "status": "unverified_for_paired_ci",
            "alignment_rule": (
                "positionally_reconstructed_from_committed_generator_protocol"
            ),
            "reason": (
                "The historical Qwen artifact has positional ids, gold boundaries, "
                "and paragraph counts but no prompt or source-row hashes. An external "
                "row-hash alignment manifest is required for a paired CI."
            ),
            "rows": len(rows),
            "source": str(source_path),
            "sha256": expected_hash,
            "positionally_reconstructed_micro_f1": metric_from_contributions(
                "scene", contributions
            ),
        }
    alignment_path = Path(alignment_source).expanduser().resolve()
    if not alignment_path.is_file() or sha256_file(alignment_path) != alignment_hash:
        raise ValueError("Qwen row-hash alignment manifest source hash differs")
    alignment = read_json(alignment_path)
    if (
        not isinstance(alignment, dict)
        or alignment.get("schema") != "scene_qwen_row_alignment.v1"
        or alignment.get("qwen_artifact_sha256") != expected_hash
    ):
        raise ValueError("Qwen row-hash alignment manifest metadata differs")
    alignment_rows = alignment.get("rows")
    if not isinstance(alignment_rows, list) or len(alignment_rows) != len(samples):
        raise ValueError("Qwen row-hash alignment manifest row count differs")
    for index, (alignment_row, sample) in enumerate(
        zip(alignment_rows, samples, strict=True)
    ):
        if not isinstance(alignment_row, dict) or alignment_row != {
            "id": index,
            "row_sha256": sample.row_sha256,
        }:
            raise ValueError(
                f"Qwen row-hash alignment differs at test index {index}"
            )
    return {
        "status": "aligned",
        "alignment_rule": "external_source_row_sha256_manifest_v1",
        "rows": len(rows),
        "source": str(source_path),
        "sha256": expected_hash,
        "alignment_manifest_source": str(alignment_path),
        "alignment_manifest_sha256": alignment_hash,
        "contributions": contributions,
        "predictions": predictions,
        "micro_f1": metric_from_contributions("scene", contributions),
    }


def expected_keys() -> set[str]:
    return {
        f"{spec.name}:{line_index}"
        for spec in TASK_SPECS
        for line_index in range(spec.expected_rows)
    }


def validate_records(
    eval_dir: Path,
    samples_by_task: dict[str, list[DatasetSample]],
    *,
    expected_fingerprint: str | None = None,
    split: str | None = None,
    normal_fusion_profile: str | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    records_by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    raw_provenance: dict[str, dict[str, Any]] = {}
    required_keys = expected_keys()
    kind_by_task = {spec.name: spec.kind for spec in TASK_SPECS}
    for condition in CONDITIONS:
        path = eval_dir / f"{condition}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_ROWS_PER_CONDITION:
            raise ValueError(
                f"Expected {EXPECTED_ROWS_PER_CONDITION} {condition} rows, found {len(rows)}"
            )
        records: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("key"))
            if key in records:
                raise ValueError(f"Duplicate key in {path}: {key}")
            records[key] = row
        if set(records) != required_keys:
            missing = sorted(required_keys - set(records))
            extra = sorted(set(records) - required_keys)
            raise ValueError(f"Key mismatch in {path}; missing={missing}, extra={extra}")
        for spec in TASK_SPECS:
            for sample in samples_by_task[spec.name]:
                key = f"{spec.name}:{sample.line_index}"
                row = records[key]
                expected_fields = {
                    "key": key,
                    "condition": condition,
                    "task": spec.name,
                    "task_kind": kind_by_task[spec.name],
                    "line_index": sample.line_index,
                    "row_sha256": sample.row_sha256,
                    "gold": sample.gold,
                    "status": "ok",
                }
                if expected_fingerprint is not None:
                    expected_fields.update(
                        {
                            "fingerprint": expected_fingerprint,
                            "split": split,
                            "max_new_tokens": (
                                128 if spec.name == "scene-v4-current" else 1024
                            ),
                            "normal_fusion_profile": (
                                None
                                if condition == "base"
                                else normal_fusion_profile
                            ),
                        }
                    )
                for field_name, expected_value in expected_fields.items():
                    if row.get(field_name) != expected_value:
                        raise ValueError(
                            f"Record mismatch for {condition}:{key}:{field_name}"
                        )
                if expected_fingerprint is not None:
                    raw_generation = row.get("raw_generation")
                    if not isinstance(raw_generation, str):
                        raise ValueError(
                            f"Record mismatch for {condition}:{key}:raw_generation"
                        )
                    if extract_json(raw_generation) != row.get("parsed_json"):
                        raise ValueError(
                            f"Record raw_generation does not reproduce parsed_json for "
                            f"{condition}:{key}"
                        )
                    expected_score = score_prediction(
                        spec.kind,
                        row.get("parsed_json"),
                        sample.gold,
                    )
                    if row.get("score") != expected_score:
                        raise ValueError(
                            f"Record mismatch for {condition}:{key}:score"
                        )
        records_by_condition[condition] = records
        raw_provenance[condition] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
    for key in required_keys:
        base_row = records_by_condition["base"][key]
        for condition in CONDITIONS:
            comparison_row = records_by_condition[condition][key]
            for field_name in (
                "task",
                "task_kind",
                "line_index",
                "row_sha256",
                "gold",
            ):
                if base_row.get(field_name) != comparison_row.get(field_name):
                    raise ValueError(
                        f"Condition pairing mismatch for {condition}:{key}:{field_name}"
                    )
    return records_by_condition, raw_provenance


def validate_strict_artifacts(eval_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = eval_dir / "summary.json"
    manifest_path = eval_dir / "manifest.json"
    strict_summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    recorded_fingerprint = manifest.get("fingerprint")
    if (
        not isinstance(recorded_fingerprint, str)
        or fingerprint_payload_sha256(manifest.get("fingerprint_payload"))
        != recorded_fingerprint
    ):
        raise ValueError("Manifest fingerprint_payload does not hash to fingerprint")
    fingerprint_contract = manifest["fingerprint_payload"].get(
        "evaluation_contract", {"name": "generic"}
    )
    if (
        fingerprint_contract.get("name") != "generic"
        and manifest.get("references")
        != manifest["fingerprint_payload"].get("references")
    ):
        raise ValueError("Manifest references differ from fingerprint_payload")
    if strict_summary.get("complete") is not True:
        raise ValueError("Strict summary is not complete")
    if strict_summary.get("fingerprint") != recorded_fingerprint:
        raise ValueError("Strict summary and manifest fingerprints differ")
    if fingerprint_contract.get("name") != "generic":
        for field in (
            "references",
            "evaluation_contract",
            "split",
            "normal_fusion_profile",
        ):
            if strict_summary.get(field) != manifest["fingerprint_payload"].get(field):
                raise ValueError(
                    f"Strict summary {field} differs from fingerprint payload"
                )
    for condition in CONDITIONS:
        condition_summary = strict_summary.get("conditions", {}).get(condition)
        if not isinstance(condition_summary, dict):
            raise ValueError(f"Strict summary is missing condition {condition}")
        for spec in TASK_SPECS:
            task_summary = condition_summary.get(spec.name)
            if not isinstance(task_summary, dict):
                raise ValueError(f"Strict summary is missing {condition}:{spec.name}")
            if task_summary.get("samples") != spec.expected_rows:
                raise ValueError(f"Strict summary row count differs for {condition}:{spec.name}")
    provenance = {
        "strict_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
    }
    return strict_summary, provenance


def recover_attribution(parsed_json: Any, candidates: tuple[str, ...]) -> str | None:
    if not isinstance(parsed_json, dict):
        return None
    values = [
        parsed_json[key]
        for key in ATTRIBUTION_ALIASES
        if key in parsed_json
    ]
    if not values or not all(isinstance(value, str) for value in values):
        return None
    normalized_values = {value.strip() for value in values}
    if len(normalized_values) != 1:
        return None
    candidate = next(iter(normalized_values))
    return candidate if candidate in set(candidates) else None


def normalize_unit_id(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:\[(\d+)\]|(\d+))", value.strip())
    if match is None:
        return None
    return match.group(1) or match.group(2)


def recover_narrative(parsed_json: Any) -> dict[str, str] | None:
    items: list[Any]
    if (
        isinstance(parsed_json, dict)
        and set(parsed_json) == {"labels"}
        and isinstance(parsed_json["labels"], list)
    ):
        items = parsed_json["labels"]
    elif isinstance(parsed_json, list):
        items = parsed_json
    elif isinstance(parsed_json, dict) and parsed_json:
        items = [
            {"unit_id": unit_id, "type": label_type}
            for unit_id, label_type in parsed_json.items()
        ]
    else:
        return None
    labels: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or "unit_id" not in item or "type" not in item:
            return None
        unit_id = normalize_unit_id(item["unit_id"])
        label_type = item["type"]
        if unit_id is None or label_type not in NARRATIVE_TYPES:
            return None
        if unit_id in labels and labels[unit_id] != label_type:
            return None
        labels[unit_id] = label_type
    return labels


def normalize_boundary_scalar(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:P(\d+)|(\d+))", value.strip())
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def recover_boundary_object(value: Any) -> list[int] | None:
    if not isinstance(value, dict):
        scalar = normalize_boundary_scalar(value)
        return None if scalar is None else [scalar]
    explicit_keys = [
        key
        for key in ("boundaries", "boundary", "after_paragraph")
        if key in value
    ]
    if len(explicit_keys) != 1:
        return None
    explicit_key = explicit_keys[0]
    if explicit_key == "after_paragraph":
        scalar = normalize_boundary_scalar(value[explicit_key])
        return None if scalar is None else [scalar]
    return recover_boundary_payload(value[explicit_key])


def recover_boundary_payload(value: Any) -> list[int] | None:
    scalar = normalize_boundary_scalar(value)
    if scalar is not None:
        return [scalar]
    if isinstance(value, dict):
        return recover_boundary_object(value)
    if not isinstance(value, list):
        return None
    boundaries: list[int] = []
    for item in value:
        if isinstance(item, list):
            return None
        recovered = recover_boundary_object(item)
        if recovered is None:
            return None
        boundaries.extend(recovered)
    return boundaries


def recover_scene(parsed_json: Any) -> set[int] | None:
    if isinstance(parsed_json, dict):
        recovered = recover_boundary_object(parsed_json)
    elif isinstance(parsed_json, list):
        recovered = recover_boundary_payload(parsed_json)
    else:
        return None
    return None if recovered is None else set(recovered)


def ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return ratio(2 * tp, 2 * tp + fp + fn)


def task_rows(
    records: dict[str, dict[str, Any]], spec: TaskSpec
) -> list[dict[str, Any]]:
    return [records[f"{spec.name}:{line_index}"] for line_index in range(spec.expected_rows)]


def analyze_task(
    spec: TaskSpec,
    rows: list[dict[str, Any]],
    samples: list[DatasetSample],
) -> tuple[dict[str, Any], list[Any], list[tuple[int, ...]]]:
    if spec.kind == "attribution":
        predictions = [
            recover_attribution(row.get("parsed_json"), sample.candidates)
            for row, sample in zip(rows, samples, strict=True)
        ]
        correct = [
            int(prediction is not None and prediction == row["gold"]["best_candidate"])
            for prediction, row in zip(predictions, rows, strict=True)
        ]
        covered = sum(prediction is not None for prediction in predictions)
        contributions = [(value, 1) for value in correct]
        summary = {
            "samples": len(rows),
            "recovered_rows": covered,
            "coverage": ratio(covered, len(rows)),
            "unrecovered_rows": len(rows) - covered,
            "correct": sum(correct),
            "accuracy": ratio(sum(correct), len(rows)),
            "covered_only_accuracy": ratio(sum(correct), covered),
            "primary_metric": ratio(sum(correct), len(rows)),
            "primary_metric_name": "format_recovered_accuracy",
        }
        return summary, predictions, contributions

    if spec.kind == "narrative":
        predictions = [recover_narrative(row.get("parsed_json")) for row in rows]
        covered = sum(prediction is not None for prediction in predictions)
        contributions: list[tuple[int, ...]] = []
        extra_ids = 0
        missing_ids = 0
        sample_accuracies: list[float] = []
        for prediction, row in zip(predictions, rows, strict=True):
            predicted = prediction or {}
            gold = gold_label_map(row["gold"])
            correct_units = sum(
                predicted.get(unit_id) == label_type
                for unit_id, label_type in gold.items()
            )
            contributions.append((correct_units, len(gold)))
            extra_ids += len(set(predicted) - set(gold))
            missing_ids += len(set(gold) - set(predicted))
            sample_accuracies.append(ratio(correct_units, len(gold)))
        correct_units = sum(item[0] for item in contributions)
        gold_units = sum(item[1] for item in contributions)
        summary = {
            "samples": len(rows),
            "recovered_rows": covered,
            "coverage": ratio(covered, len(rows)),
            "unrecovered_rows": len(rows) - covered,
            "correct_units": correct_units,
            "gold_units": gold_units,
            "accuracy": ratio(correct_units, gold_units),
            "mean_sample_accuracy": statistics.fmean(sample_accuracies),
            "extra_unit_ids": extra_ids,
            "missing_gold_unit_ids": missing_ids,
            "primary_metric": ratio(correct_units, gold_units),
            "primary_metric_name": "format_recovered_unit_accuracy",
        }
        return summary, predictions, contributions

    predictions = [recover_scene(row.get("parsed_json")) for row in rows]
    covered = sum(prediction is not None for prediction in predictions)
    contributions = []
    sample_f1_values: list[float] = []
    invalid_indices = 0
    predicted_indices = 0
    for prediction, row, sample in zip(predictions, rows, samples, strict=True):
        predicted = prediction or set()
        gold = strict_gold_boundaries(row["gold"])
        if sample.paragraph_count is None:
            raise AssertionError("Scene sample is missing paragraph_count")
        invalid_indices += sum(
            boundary < 1 or boundary >= sample.paragraph_count
            for boundary in predicted
        )
        predicted_indices += len(predicted)
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        contributions.append((tp, fp, fn))
        sample_f1_values.append(f1_from_counts(tp, fp, fn))
    tp = sum(item[0] for item in contributions)
    fp = sum(item[1] for item in contributions)
    fn = sum(item[2] for item in contributions)
    summary = {
        "samples": len(rows),
        "recovered_rows": covered,
        "coverage": ratio(covered, len(rows)),
        "unrecovered_rows": len(rows) - covered,
        "predicted_unique_indices": predicted_indices,
        "invalid_indices_counted_as_fp": invalid_indices,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "micro_f1": f1_from_counts(tp, fp, fn),
        "macro_sample_f1": statistics.fmean(sample_f1_values),
        "primary_metric": f1_from_counts(tp, fp, fn),
        "primary_metric_name": "format_recovered_micro_f1",
    }
    return summary, predictions, contributions


def metric_from_contributions(kind: str, contributions: Iterable[tuple[int, ...]]) -> float:
    materialized = list(contributions)
    if kind in {"attribution", "narrative"}:
        numerator = sum(item[0] for item in materialized)
        denominator = sum(item[1] for item in materialized)
        return ratio(numerator, denominator)
    tp = sum(item[0] for item in materialized)
    fp = sum(item[1] for item in materialized)
    fn = sum(item[2] for item in materialized)
    return f1_from_counts(tp, fp, fn)


def strict_contributions(
    spec: TaskSpec,
    rows: Iterable[dict[str, Any]],
) -> list[tuple[int, ...]]:
    contributions: list[tuple[int, ...]] = []
    for row in rows:
        score = row["score"]
        if spec.kind == "attribution":
            contributions.append((int(bool(score["correct"])), 1))
        elif spec.kind == "narrative":
            contributions.append(
                (int(score["correct_units"]), int(score["gold_units"]))
            )
        else:
            contributions.append(
                (int(score["tp"]), int(score["fp"]), int(score["fn"]))
            )
    return contributions


def normalized_difference(normal: float, base: float) -> float:
    difference = normal - base
    return 0.0 if abs(difference) < 1e-15 else difference


def metric_pair(
    *,
    kind: str,
    metric_name: str,
    base_contributions: list[tuple[int, ...]],
    normal_contributions: list[tuple[int, ...]],
) -> dict[str, Any]:
    base_metric = metric_from_contributions(kind, base_contributions)
    normal_metric = metric_from_contributions(kind, normal_contributions)
    return {
        "metric_name": metric_name,
        "base": base_metric,
        "normal": normal_metric,
        "normal_minus_base": normalized_difference(normal_metric, base_metric),
    }


def coverage_pair(
    base_predictions: list[Any],
    normal_predictions: list[Any],
) -> dict[str, float]:
    base_coverage = ratio(
        sum(prediction is not None for prediction in base_predictions),
        len(base_predictions),
    )
    normal_coverage = ratio(
        sum(prediction is not None for prediction in normal_predictions),
        len(normal_predictions),
    )
    return {
        "base": base_coverage,
        "normal": normal_coverage,
        "normal_minus_base": normalized_difference(normal_coverage, base_coverage),
    }


def max_token_hit_rate_pair(
    base_rows: list[dict[str, Any]],
    normal_rows: list[dict[str, Any]],
) -> dict[str, float]:
    base_rate = ratio(
        sum(bool(row.get("hit_max_new_tokens")) for row in base_rows),
        len(base_rows),
    )
    normal_rate = ratio(
        sum(bool(row.get("hit_max_new_tokens")) for row in normal_rows),
        len(normal_rows),
    )
    return {
        "base": base_rate,
        "normal": normal_rate,
        "normal_minus_base": normalized_difference(normal_rate, base_rate),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def paired_bootstrap(
    kind: str,
    base_contributions: list[tuple[int, ...]],
    normal_contributions: list[tuple[int, ...]],
) -> dict[str, Any]:
    if len(base_contributions) != len(normal_contributions):
        raise ValueError("Bootstrap contribution lengths differ")
    sample_count = len(base_contributions)
    generator = random.Random(BOOTSTRAP_SEED)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [generator.randrange(sample_count) for _ in range(sample_count)]
        base_metric = metric_from_contributions(
            kind, (base_contributions[index] for index in indices)
        )
        normal_metric = metric_from_contributions(
            kind, (normal_contributions[index] for index in indices)
        )
        difference = normal_metric - base_metric
        differences.append(0.0 if abs(difference) < 1e-15 else difference)
    point_estimate = metric_from_contributions(kind, normal_contributions) - metric_from_contributions(
        kind, base_contributions
    )
    if abs(point_estimate) < 1e-15:
        point_estimate = 0.0
    return {
        "normal_minus_base": point_estimate,
        "ci_95_percentile": [
            percentile(differences, 0.025),
            percentile(differences, 0.975),
        ],
        "bootstrap_mean": statistics.fmean(differences),
        "bootstrap_standard_error": statistics.pstdev(differences),
        "probability_positive": ratio(
            sum(difference > 0 for difference in differences), len(differences)
        ),
        "probability_negative": ratio(
            sum(difference < 0 for difference in differences), len(differences)
        ),
        "probability_zero": ratio(
            sum(difference == 0 for difference in differences), len(differences)
        ),
    }


def paired_bootstrap_comparison(
    *,
    kind: str,
    candidate_name: str,
    comparator_name: str,
    candidate_contributions: list[tuple[int, ...]],
    comparator_contributions: list[tuple[int, ...]],
) -> dict[str, Any]:
    bootstrap = paired_bootstrap(
        kind,
        comparator_contributions,
        candidate_contributions,
    )
    point_estimate = bootstrap.pop("normal_minus_base")
    return {
        "candidate": candidate_name,
        "comparator": comparator_name,
        "difference_name": f"{candidate_name}_minus_{comparator_name}",
        "point_estimate": point_estimate,
        **bootstrap,
    }


def condition_task_diagnostics(
    rows: list[dict[str, Any]],
    predictions: list[Any],
) -> dict[str, Any]:
    recovered_rows = sum(prediction is not None for prediction in predictions)
    max_token_hits = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
    return {
        "rows": len(rows),
        "recovered_rows": recovered_rows,
        "coverage": ratio(recovered_rows, len(rows)),
        "max_token_hits": max_token_hits,
        "max_token_hit_rate": ratio(max_token_hits, len(rows)),
    }


def minimum_gate(value: float, threshold: float) -> dict[str, Any]:
    return {
        "operator": ">=",
        "threshold": threshold,
        "value": value,
        "passed": value >= threshold,
    }


def maximum_gate(value: float, threshold: float) -> dict[str, Any]:
    return {
        "operator": "<=",
        "threshold": threshold,
        "value": value,
        "passed": value <= threshold,
    }


def strict_positive_gate(value: float) -> dict[str, Any]:
    return {
        "operator": ">",
        "threshold": 0.0,
        "value": value,
        "passed": value > 0.0,
    }


def build_scene_v6_gate_analysis(
    *,
    contract: dict[str, Any],
    split: str,
    specs: tuple[TaskSpec, ...],
    strict_summary: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, list[Any]]],
    contributions: dict[str, dict[str, list[tuple[int, ...]]]],
    records_by_condition: dict[str, dict[str, dict[str, Any]]],
    samples_by_task: dict[str, list[DatasetSample]],
) -> dict[str, Any]:
    contract_name = contract.get("name")
    if contract_name == "generic":
        return {"status": "not_requested", "contract": contract}
    if contract_name not in SCENE_V6_CONTRACT_ROWS:
        raise ValueError(f"Unsupported scene V6 evaluation contract: {contract_name}")
    expected_split, expected_rows = SCENE_V6_CONTRACT_ROWS[contract_name]
    expected_phase = (
        "validation_selection" if expected_split == "val" else "final_test"
    )
    expected_contract = {
        "name": contract_name,
        "phase": expected_phase,
        "split": expected_split,
        "task": "scene-v4-current",
        "rows": expected_rows,
        "conditions": ["base", "normal", "no_write"],
        "normal_fusion_profile": "native",
        "expected_memory_layer_count": 42,
        "memory_target_layers": list(range(42)),
        "memory_delta_heads": ["q", "o"],
        "memory_rank": 4,
        "rwkv_ms_semantics_version": 2,
        "memory_backend": "rwkv_ms",
        "official_dataset_revision": OFFICIAL_SCENE_V4_DATASET_REVISION,
        "official_dataset_sha256": OFFICIAL_SCENE_V4_SHA256[expected_split],
        "overwrite_allowed": contract_name != "scene_v6_final_test",
        "generation_policy": (
            "Append-only resumable records; completed keys are never regenerated."
        ),
    }
    if contract_name == "scene_v6_final_test":
        expected_contract.update(
            {
                "checkpoint_selection_forbidden": True,
                "test_once_enforcement_scope": (
                    "per_output_directory_and_fingerprint"
                ),
                "test_once_enforcement_caveat": (
                    "A new output directory can rerun inference; global single-use "
                    "enforcement is not provided. Checkpoint selection on test remains "
                    "forbidden."
                ),
            }
        )
    if contract != expected_contract:
        raise ValueError("Scene V6 evaluation contract metadata differs from the lock")
    if split != expected_split:
        raise ValueError(f"{contract_name} analyzer split differs from the lock")
    if tuple(spec.name for spec in specs) != ("scene-v4-current",):
        raise ValueError(f"{contract_name} requires exactly scene-v4-current")
    if specs[0].expected_rows != expected_rows:
        raise ValueError(f"{contract_name} row count differs from the lock")
    if CONDITIONS != ("base", "normal", "no_write"):
        raise ValueError(f"{contract_name} condition set differs from the lock")

    task_name = "scene-v4-current"
    condition_diagnostics = {
        condition: condition_task_diagnostics(
            task_rows(records_by_condition[condition], specs[0]),
            predictions[condition][task_name],
        )
        for condition in CONDITIONS
    }
    comparisons = {
        comparator: paired_bootstrap_comparison(
            kind="scene",
            candidate_name="normal",
            comparator_name=comparator,
            candidate_contributions=contributions["normal"][task_name],
            comparator_contributions=contributions[comparator][task_name],
        )
        for comparator in ("base", "no_write")
    }
    aligned_qwen = aligned_qwen_scene_reference(
        strict_summary,
        samples_by_task[task_name],
        split=split,
    )
    qwen_aligned = split == "test" and aligned_qwen.get("status") == "aligned"
    if qwen_aligned:
        comparisons["aligned_qwen"] = paired_bootstrap_comparison(
            kind="scene",
            candidate_name="normal",
            comparator_name="aligned_qwen",
            candidate_contributions=contributions["normal"][task_name],
            comparator_contributions=aligned_qwen["contributions"],
        )

    ci_comparators = ("base", "no_write") + (
        ("aligned_qwen",) if qwen_aligned else ()
    )
    ci_gates = {
        comparator: strict_positive_gate(
            float(comparisons[comparator]["ci_95_percentile"][0])
        )
        for comparator in ci_comparators
    }
    if split == "test" and not qwen_aligned:
        ci_gates["aligned_qwen"] = {
            "operator": ">",
            "threshold": 0.0,
            "value": None,
            "passed": False,
            "status": "unavailable_without_row_hash_alignment_manifest",
        }
    normal_metric = float(metrics["normal"][task_name]["primary_metric"])
    normal_coverage = float(condition_diagnostics["normal"]["coverage"])
    no_write_delta = float(comparisons["no_write"]["point_estimate"])
    max_token_delta_gates = {}
    for comparator in ("base", "no_write"):
        delta = normalized_difference(
            float(condition_diagnostics["normal"]["max_token_hit_rate"]),
            float(condition_diagnostics[comparator]["max_token_hit_rate"]),
        )
        max_token_delta_gates[comparator] = maximum_gate(
            delta,
            SCENE_V6_MAX_TOKEN_HIT_RATE_DELTA_CEILING,
        )
    gates = {
        "normal_recovered_micro_f1": minimum_gate(
            normal_metric,
            SCENE_V6_RECOVERED_MICRO_F1_FLOOR,
        ),
        "normal_minus_no_write": minimum_gate(
            no_write_delta,
            SCENE_V6_NORMAL_MINUS_NO_WRITE_FLOOR,
        ),
        "normal_coverage": minimum_gate(
            normal_coverage,
            SCENE_V6_COVERAGE_FLOOR,
        ),
        "paired_ci_95_lower_strictly_positive": ci_gates,
        "max_token_hit_rate_delta": max_token_delta_gates,
    }
    flattened_gate_results = [
        gates["normal_recovered_micro_f1"]["passed"],
        gates["normal_minus_no_write"]["passed"],
        gates["normal_coverage"]["passed"],
        *(gate["passed"] for gate in ci_gates.values()),
        *(gate["passed"] for gate in max_token_delta_gates.values()),
    ]
    all_gates_passed = all(flattened_gate_results)
    return {
        "status": "pass" if all_gates_passed else "fail",
        "contract": contract,
        "selection_authorized": split == "val",
        "checkpoint_selection_forbidden": split == "test",
        "final_claim_authorized": split == "test" and all_gates_passed,
        "all_official_rows_verified": True,
        "condition_diagnostics": condition_diagnostics,
        "comparisons": comparisons,
        "aligned_qwen": {
            key: value
            for key, value in aligned_qwen.items()
            if key not in {"contributions", "predictions"}
        },
        "gates": gates,
        "all_gates_passed": all_gates_passed,
    }


def unavailable_selection_task(
    *,
    task_name: str,
    split: str,
) -> dict[str, Any]:
    metric_floor = SELECTION_METRIC_FLOORS[task_name]
    gates = {
        "recovered_metric_delta_floor": {
            "operator": ">=",
            "threshold": metric_floor,
            "value": None,
            "passed": False,
            "status": "not_evaluated",
        },
        "recovered_ci_upper_nonnegative": {
            "operator": ">=",
            "threshold": SELECTION_CI_UPPER_FLOOR,
            "value": None,
            "passed": False,
            "status": "not_evaluated",
        },
        "coverage_delta_floor": {
            "operator": ">=",
            "threshold": SELECTION_COVERAGE_DELTA_FLOOR,
            "value": None,
            "passed": False,
            "status": "not_evaluated",
        },
        "max_token_hit_rate_delta_ceiling": {
            "operator": "<=",
            "threshold": SELECTION_MAX_TOKEN_HIT_RATE_DELTA_CEILING,
            "value": None,
            "passed": False,
            "status": "not_evaluated",
        },
    }
    return {
        "status": "not_evaluated",
        "split": split,
        "official_expected_rows": EXPECTED_ROWS_BY_SPLIT[split][task_name],
        "evaluated_rows": 0,
        "selection_rows": 0,
        "configured_excluded_zero_based_indices": list(
            CLEAN_SELECTION_EXCLUSIONS.get(split, {}).get(task_name, ())
        ),
        "applied_excluded_zero_based_indices": [],
        "clean_selection_exclusion_note": CLEAN_SELECTION_EXCLUSION_NOTES.get(
            (split, task_name)
        ),
        "all_rows": None,
        "clean_selection": None,
        "gates": gates,
        "gates_passed": False,
        "criterion_eligible": False,
        "criterion_passed": False,
    }


def build_selection_criterion(
    *,
    split: str,
    specs: tuple[TaskSpec, ...],
    strict_summary: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, list[Any]]],
    contributions: dict[str, dict[str, list[tuple[int, ...]]]],
    records_by_condition: dict[str, dict[str, dict[str, Any]]],
    all_row_bootstraps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    specs_by_name = {spec.name: spec for spec in specs}
    task_outputs: dict[str, dict[str, Any]] = {}
    for task_name in CORE_SELECTION_TASKS:
        spec = specs_by_name.get(task_name)
        if spec is None:
            task_outputs[task_name] = unavailable_selection_task(
                task_name=task_name,
                split=split,
            )
            continue

        base_rows = task_rows(records_by_condition["base"], spec)
        normal_rows = task_rows(records_by_condition["normal"], spec)
        configured_exclusions = list(
            CLEAN_SELECTION_EXCLUSIONS.get(split, {}).get(task_name, ())
        )
        applied_exclusions = sorted(
            index for index in configured_exclusions if index < spec.expected_rows
        )
        excluded = set(applied_exclusions)
        selection_indices = [
            index for index in range(spec.expected_rows) if index not in excluded
        ]
        if not selection_indices:
            raise ValueError(f"Clean selection subset is empty for {task_name}")

        base_strict_summary = strict_summary["conditions"]["base"][task_name]
        normal_strict_summary = strict_summary["conditions"]["normal"][task_name]
        base_strict_metric = float(base_strict_summary["primary_metric"])
        normal_strict_metric = float(normal_strict_summary["primary_metric"])
        all_rows_strict = {
            "metric_name": str(base_strict_summary["primary_metric_name"]),
            "base": base_strict_metric,
            "normal": normal_strict_metric,
            "normal_minus_base": normalized_difference(
                normal_strict_metric,
                base_strict_metric,
            ),
        }
        all_rows_recovered = {
            "metric_name": metrics["base"][task_name]["primary_metric_name"],
            "base": float(metrics["base"][task_name]["primary_metric"]),
            "normal": float(metrics["normal"][task_name]["primary_metric"]),
            **all_row_bootstraps[task_name],
        }
        all_rows_coverage = coverage_pair(
            predictions["base"][task_name],
            predictions["normal"][task_name],
        )
        all_rows_max_token_hits = max_token_hit_rate_pair(base_rows, normal_rows)

        selection_base_rows = [base_rows[index] for index in selection_indices]
        selection_normal_rows = [normal_rows[index] for index in selection_indices]
        selection_base_predictions = [
            predictions["base"][task_name][index] for index in selection_indices
        ]
        selection_normal_predictions = [
            predictions["normal"][task_name][index] for index in selection_indices
        ]
        selection_base_contributions = [
            contributions["base"][task_name][index] for index in selection_indices
        ]
        selection_normal_contributions = [
            contributions["normal"][task_name][index] for index in selection_indices
        ]
        clean_strict = metric_pair(
            kind=spec.kind,
            metric_name=str(base_strict_summary["primary_metric_name"]),
            base_contributions=strict_contributions(spec, selection_base_rows),
            normal_contributions=strict_contributions(spec, selection_normal_rows),
        )
        if applied_exclusions:
            clean_bootstrap = paired_bootstrap(
                spec.kind,
                selection_base_contributions,
                selection_normal_contributions,
            )
        else:
            clean_bootstrap = dict(all_row_bootstraps[task_name])
        clean_recovered = {
            **metric_pair(
                kind=spec.kind,
                metric_name=metrics["base"][task_name]["primary_metric_name"],
                base_contributions=selection_base_contributions,
                normal_contributions=selection_normal_contributions,
            ),
            **clean_bootstrap,
        }
        clean_coverage = coverage_pair(
            selection_base_predictions,
            selection_normal_predictions,
        )
        clean_max_token_hits = max_token_hit_rate_pair(
            selection_base_rows,
            selection_normal_rows,
        )
        ci_upper = float(clean_recovered["ci_95_percentile"][1])
        gates = {
            "recovered_metric_delta_floor": minimum_gate(
                float(clean_recovered["normal_minus_base"]),
                SELECTION_METRIC_FLOORS[task_name],
            ),
            "recovered_ci_upper_nonnegative": minimum_gate(
                ci_upper,
                SELECTION_CI_UPPER_FLOOR,
            ),
            "coverage_delta_floor": minimum_gate(
                float(clean_coverage["normal_minus_base"]),
                SELECTION_COVERAGE_DELTA_FLOOR,
            ),
            "max_token_hit_rate_delta_ceiling": maximum_gate(
                float(clean_max_token_hits["normal_minus_base"]),
                SELECTION_MAX_TOKEN_HIT_RATE_DELTA_CEILING,
            ),
        }
        gates_passed = all(bool(gate["passed"]) for gate in gates.values())
        official_expected_rows = EXPECTED_ROWS_BY_SPLIT[split][task_name]
        if spec.expected_rows == official_expected_rows:
            availability = "complete"
        elif spec.expected_rows < official_expected_rows:
            availability = "limited"
        else:
            availability = "nonstandard"
        criterion_eligible = split == "val" and availability == "complete"
        if split != "val":
            status = "ineligible_split"
        elif availability != "complete":
            status = "provisional_pass" if gates_passed else "provisional_fail"
        else:
            status = "pass" if gates_passed else "fail"
        task_outputs[task_name] = {
            "status": status,
            "split": split,
            "availability": availability,
            "official_expected_rows": official_expected_rows,
            "evaluated_rows": spec.expected_rows,
            "selection_rows": len(selection_indices),
            "configured_excluded_zero_based_indices": configured_exclusions,
            "applied_excluded_zero_based_indices": applied_exclusions,
            "clean_selection_exclusion_note": CLEAN_SELECTION_EXCLUSION_NOTES.get(
                (split, task_name)
            ),
            "all_rows": {
                "strict": all_rows_strict,
                "recovered": all_rows_recovered,
                "coverage": all_rows_coverage,
                "max_token_hit_rate": all_rows_max_token_hits,
            },
            "clean_selection": {
                "strict": clean_strict,
                "recovered": clean_recovered,
                "coverage": clean_coverage,
                "max_token_hit_rate": clean_max_token_hits,
            },
            "gates": gates,
            "gates_passed": gates_passed,
            "criterion_eligible": criterion_eligible,
            "criterion_passed": criterion_eligible and gates_passed,
        }

    complete = split == "val" and all(
        task_outputs[task_name].get("availability") == "complete"
        for task_name in CORE_SELECTION_TASKS
    )
    all_gates_passed = all(
        bool(task_outputs[task_name]["gates_passed"])
        for task_name in CORE_SELECTION_TASKS
    )
    overall_passed = complete and all_gates_passed
    if split != "val":
        status = "ineligible_split"
    elif not complete:
        status = "incomplete"
    else:
        status = "pass" if overall_passed else "fail"
    return {
        "criterion_version": "novel_agent_validation_selection_v1",
        "status": status,
        "split": split,
        "required_split": "val",
        "required_tasks": list(CORE_SELECTION_TASKS),
        "thresholds": {
            "recovered_metric_normal_minus_base_floor_by_task": dict(
                SELECTION_METRIC_FLOORS
            ),
            "recovered_ci_95_upper_floor": SELECTION_CI_UPPER_FLOOR,
            "coverage_normal_minus_base_floor": SELECTION_COVERAGE_DELTA_FLOOR,
            "max_token_hit_rate_normal_minus_base_ceiling": (
                SELECTION_MAX_TOKEN_HIT_RATE_DELTA_CEILING
            ),
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "complete": complete,
        "all_gates_passed": all_gates_passed,
        "overall_passed": overall_passed,
        "tasks": task_outputs,
    }


def paired_changes(
    spec: TaskSpec,
    rows: list[dict[str, Any]],
    base_predictions: list[Any],
    normal_predictions: list[Any],
) -> dict[str, Any]:
    if spec.kind == "attribution":
        both_correct = 0
        both_wrong = 0
        normal_gain = 0
        normal_regression = 0
        prediction_disagreements = 0
        for row, base_prediction, normal_prediction in zip(
            rows, base_predictions, normal_predictions, strict=True
        ):
            gold = row["gold"]["best_candidate"]
            base_correct = base_prediction is not None and base_prediction == gold
            normal_correct = normal_prediction is not None and normal_prediction == gold
            prediction_disagreements += base_prediction != normal_prediction
            if base_correct and normal_correct:
                both_correct += 1
            elif base_correct:
                normal_regression += 1
            elif normal_correct:
                normal_gain += 1
            else:
                both_wrong += 1
        return {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "normal_gain": normal_gain,
            "normal_regression": normal_regression,
            "prediction_disagreements": prediction_disagreements,
        }

    if spec.kind == "narrative":
        counts = {
            "both_correct": 0,
            "both_wrong": 0,
            "normal_gain": 0,
            "normal_regression": 0,
            "prediction_disagreements": 0,
        }
        sample_changes = {"normal_better": 0, "normal_worse": 0, "tie": 0}
        for row, base_prediction, normal_prediction in zip(
            rows, base_predictions, normal_predictions, strict=True
        ):
            gold = gold_label_map(row["gold"])
            base_labels = base_prediction or {}
            normal_labels = normal_prediction or {}
            base_correct_count = 0
            normal_correct_count = 0
            for unit_id, label_type in gold.items():
                base_correct = base_labels.get(unit_id) == label_type
                normal_correct = normal_labels.get(unit_id) == label_type
                base_correct_count += base_correct
                normal_correct_count += normal_correct
                counts["prediction_disagreements"] += (
                    base_labels.get(unit_id) != normal_labels.get(unit_id)
                )
                if base_correct and normal_correct:
                    counts["both_correct"] += 1
                elif base_correct:
                    counts["normal_regression"] += 1
                elif normal_correct:
                    counts["normal_gain"] += 1
                else:
                    counts["both_wrong"] += 1
            if normal_correct_count > base_correct_count:
                sample_changes["normal_better"] += 1
            elif normal_correct_count < base_correct_count:
                sample_changes["normal_worse"] += 1
            else:
                sample_changes["tie"] += 1
        return {**counts, "sample_accuracy_changes": sample_changes}

    counts = {
        "gold_both_hit": 0,
        "gold_both_missed": 0,
        "gold_gained_by_normal": 0,
        "gold_lost_by_normal": 0,
        "shared_fp": 0,
        "fp_added_by_normal": 0,
        "fp_removed_by_normal": 0,
    }
    sample_changes = {"normal_better": 0, "normal_worse": 0, "tie": 0}
    for row, base_prediction, normal_prediction in zip(
        rows, base_predictions, normal_predictions, strict=True
    ):
        gold = strict_gold_boundaries(row["gold"])
        base_boundaries = base_prediction or set()
        normal_boundaries = normal_prediction or set()
        for boundary in gold:
            base_hit = boundary in base_boundaries
            normal_hit = boundary in normal_boundaries
            if base_hit and normal_hit:
                counts["gold_both_hit"] += 1
            elif base_hit:
                counts["gold_lost_by_normal"] += 1
            elif normal_hit:
                counts["gold_gained_by_normal"] += 1
            else:
                counts["gold_both_missed"] += 1
        for boundary in (base_boundaries | normal_boundaries) - gold:
            base_fp = boundary in base_boundaries
            normal_fp = boundary in normal_boundaries
            if base_fp and normal_fp:
                counts["shared_fp"] += 1
            elif base_fp:
                counts["fp_removed_by_normal"] += 1
            else:
                counts["fp_added_by_normal"] += 1
        base_f1 = f1_from_counts(
            len(base_boundaries & gold),
            len(base_boundaries - gold),
            len(gold - base_boundaries),
        )
        normal_f1 = f1_from_counts(
            len(normal_boundaries & gold),
            len(normal_boundaries - gold),
            len(gold - normal_boundaries),
        )
        if normal_f1 > base_f1:
            sample_changes["normal_better"] += 1
        elif normal_f1 < base_f1:
            sample_changes["normal_worse"] += 1
        else:
            sample_changes["tie"] += 1
    return {**counts, "sample_f1_changes": sample_changes}


def resource_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    peak_memory = [
        int(row["peak_cuda_memory_bytes"])
        for row in rows
        if row.get("peak_cuda_memory_bytes") is not None
    ]
    return {
        "samples": len(rows),
        "elapsed_seconds": sum(elapsed),
        "mean_elapsed_seconds": statistics.fmean(elapsed),
        "median_elapsed_seconds": statistics.median(elapsed),
        "p95_elapsed_seconds": percentile(elapsed, 0.95),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
        "hit_max_new_tokens": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
        "max_peak_cuda_memory_bytes": max(peak_memory) if peak_memory else None,
    }


def trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_trace = 0
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        trace = row.get("memory_trace")
        if not isinstance(trace, list) or not trace:
            continue
        records_with_trace += 1
        for point in trace:
            if not isinstance(point, dict) or not isinstance(point.get("layer_idx"), int):
                continue
            by_layer.setdefault(int(point["layer_idx"]), []).append(point)
    field_names = ("read_entropy", "read_max", "state_norm", "delta_o_ratio")
    layer_summaries: dict[str, dict[str, Any]] = {}
    for layer_index, points in sorted(by_layer.items()):
        layer_summary: dict[str, Any] = {"points": len(points)}
        for field_name in field_names:
            values = [
                float(point[field_name])
                for point in points
                if isinstance(point.get(field_name), (int, float))
                and not isinstance(point.get(field_name), bool)
            ]
            if values:
                layer_summary[f"mean_{field_name}"] = statistics.fmean(values)
                layer_summary[f"max_{field_name}"] = max(values)
        layer_summaries[str(layer_index)] = layer_summary
    return {
        "records_with_trace": records_with_trace,
        "trace_points": sum(len(points) for points in by_layer.values()),
        "by_layer": layer_summaries,
    }


def condition_resources(
    records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    all_rows = [
        records[f"{spec.name}:{line_index}"]
        for spec in TASK_SPECS
        for line_index in range(spec.expected_rows)
    ]
    return {
        "overall": resource_totals(all_rows),
        "by_task": {
            spec.name: resource_totals(task_rows(records, spec))
            for spec in TASK_SPECS
        },
        "memory_trace": trace_summary(all_rows),
    }


def recovery_rules() -> dict[str, Any]:
    return {
        "version": "conservative-explicit-schema-v1",
        "attribution": {
            "accepted_aliases": sorted(ATTRIBUTION_ALIASES),
            "rule": (
                "All present accepted aliases must be strings that agree after trimming; "
                "the value must exactly match a candidate parsed from the prompt."
            ),
            "not_recovered": "No fuzzy, substring, transliteration, or gold-name matching.",
        },
        "narrative": {
            "accepted_structures": [
                "an object whose only key is labels and whose value is a label list",
                "a top-level label list",
                "a non-empty direct unit-id to type object",
            ],
            "unit_id_normalization": "Only decimal N and exact bracketed [N] forms.",
            "allowed_types": sorted(NARRATIVE_TYPES),
            "duplicate_rule": "Identical duplicates collapse; conflicting duplicates reject the row.",
            "scoring_rule": (
                "Author unit accuracy scores gold unit IDs; extra predicted IDs are audited "
                "but do not enter the denominator."
            ),
        },
        "scene": {
            "accepted_explicit_keys": ["after_paragraph", "boundaries", "boundary"],
            "accepted_structures": (
                "Top-level dictionaries or lists recursively composed of explicit boundary "
                "keys and scalar boundary values; nested raw lists are rejected."
            ),
            "integer_normalization": "Only integer N, decimal string N, and exact P<N> string forms.",
            "range_rule": (
                "No shifting, clamping, dropping, or Boolean interpretation; literal zero and "
                "out-of-range indices remain predictions and therefore false positives."
            ),
            "not_recovered": (
                "No inference from start/end spans, segment text, prose, adjacency, type fields, "
                "or implicit binary 0/1 decisions."
            ),
        },
        "unrecoverable_row_scoring": (
            "An unrecoverable attribution row is incorrect, an unrecoverable narrative row has "
            "no predicted labels, and an unrecoverable scene row predicts an empty boundary set."
        ),
    }


def build_output(
    eval_dir: Path,
    training_root: Path,
    split: str,
    strict_summary: dict[str, Any],
    strict_provenance: dict[str, Any],
    raw_provenance: dict[str, Any],
    dataset_provenance: dict[str, Any],
    records_by_condition: dict[str, dict[str, dict[str, Any]]],
    samples_by_task: dict[str, list[DatasetSample]],
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {condition: {} for condition in CONDITIONS}
    predictions: dict[str, dict[str, list[Any]]] = {
        condition: {} for condition in CONDITIONS
    }
    contributions: dict[str, dict[str, list[tuple[int, ...]]]] = {
        condition: {} for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for spec in TASK_SPECS:
            rows = task_rows(records_by_condition[condition], spec)
            task_metrics, task_predictions, task_contributions = analyze_task(
                spec, rows, samples_by_task[spec.name]
            )
            metrics[condition][spec.name] = task_metrics
            predictions[condition][spec.name] = task_predictions
            contributions[condition][spec.name] = task_contributions

    normal_vs_base: dict[str, dict[str, Any]] = {}
    normal_vs_no_write: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    strict_comparison: dict[str, dict[str, Any]] = {
        condition: {} for condition in CONDITIONS
    }
    reference_comparison: dict[str, dict[str, Any]] = {
        condition: {} for condition in CONDITIONS
    }
    for spec in TASK_SPECS:
        base_metric = float(metrics["base"][spec.name]["primary_metric"])
        normal_metric = float(metrics["normal"][spec.name]["primary_metric"])
        bootstrap = paired_bootstrap(
            spec.kind,
            contributions["base"][spec.name],
            contributions["normal"][spec.name],
        )
        normal_vs_base[spec.name] = {
            "metric_name": metrics["base"][spec.name]["primary_metric_name"],
            "base": base_metric,
            "normal": normal_metric,
            **bootstrap,
        }
        if "no_write" in CONDITIONS:
            normal_vs_no_write[spec.name] = {
                "metric_name": metrics["normal"][spec.name]["primary_metric_name"],
                **paired_bootstrap_comparison(
                    kind=spec.kind,
                    candidate_name="normal",
                    comparator_name="no_write",
                    candidate_contributions=contributions["normal"][spec.name],
                    comparator_contributions=contributions["no_write"][spec.name],
                ),
            }
        paired[spec.name] = paired_changes(
            spec,
            task_rows(records_by_condition["base"], spec),
            predictions["base"][spec.name],
            predictions["normal"][spec.name],
        )
        reference = strict_summary.get("references", {}).get(spec.name, {})
        reference_metric = reference.get("artifact_metric")
        for condition in CONDITIONS:
            recovered_metric = float(metrics[condition][spec.name]["primary_metric"])
            strict_metric = float(
                strict_summary["conditions"][condition][spec.name]["primary_metric"]
            )
            strict_comparison[condition][spec.name] = {
                "strict_metric_name": strict_summary["conditions"][condition][spec.name][
                    "primary_metric_name"
                ],
                "strict_metric": strict_metric,
                "format_recovered_metric": recovered_metric,
                "format_recovered_minus_strict": recovered_metric - strict_metric,
            }
            if isinstance(reference_metric, (int, float)):
                reference_comparison[condition][spec.name] = {
                    "artifact_metric": float(reference_metric),
                    "artifact_metric_name": reference.get("metric_name"),
                    "artifact_source": reference.get("artifact_source"),
                    "reference_model": reference.get("reference_model"),
                    "comparison_caveat": reference.get("comparison_caveat"),
                    "format_recovered_metric": recovered_metric,
                    "format_recovered_minus_artifact": recovered_metric
                    - float(reference_metric),
                }

    legacy_selection_criterion = build_selection_criterion(
        split=split,
        specs=TASK_SPECS,
        strict_summary=strict_summary,
        metrics=metrics,
        predictions=predictions,
        contributions=contributions,
        records_by_condition=records_by_condition,
        all_row_bootstraps=normal_vs_base,
    )
    condition_diagnostics = {
        spec.name: {
            condition: condition_task_diagnostics(
                task_rows(records_by_condition[condition], spec),
                predictions[condition][spec.name],
            )
            for condition in CONDITIONS
        }
        for spec in TASK_SPECS
    }
    contract = evaluation_contract(eval_dir)
    scene_v6_gate_analysis = build_scene_v6_gate_analysis(
        contract=contract,
        split=split,
        specs=TASK_SPECS,
        strict_summary=strict_summary,
        metrics=metrics,
        predictions=predictions,
        contributions=contributions,
        records_by_condition=records_by_condition,
        samples_by_task=samples_by_task,
    )
    if contract.get("name") == "scene_v6_validation":
        selection_criterion = {
            "criterion_version": "scene_v6_validation_selection_v1",
            "status": scene_v6_gate_analysis["status"],
            "complete": True,
            "overall_passed": scene_v6_gate_analysis["all_gates_passed"],
            "selection_rows": contract["rows"],
            "source": "scene_v6_gate_analysis",
        }
    elif contract.get("name") == "scene_v6_final_test":
        selection_criterion = {
            "criterion_version": "scene_v6_no_test_selection_v1",
            "status": "not_applicable",
            "complete": True,
            "overall_passed": False,
            "selection_rows": 0,
            "reason": "The untouched test split is forbidden for checkpoint selection.",
        }
    else:
        selection_criterion = legacy_selection_criterion
    analyzer_path = Path(__file__).resolve()
    return {
        "non_official": True,
        "split": split,
        "diagnostic_kind": "conservative format-recovered structured-task transfer evaluation",
        "warning": (
            "These metrics are diagnostic only. They recover explicit formatting variants from "
            "frozen raw generations and do not replace the strict author-compatible evaluation."
        ),
        "transfer_evaluation_caveat": strict_summary.get(
            "training_scope",
            "The adapter was not trained on these structured task prompts or labels.",
        ),
        "recovery_rules": recovery_rules(),
        "validation": {
            "expected_rows_per_condition": EXPECTED_ROWS_PER_CONDITION,
            "matched_rows_per_condition": {
                condition: len(records_by_condition[condition])
                for condition in CONDITIONS
            },
            "task_rows": {
                spec.name: spec.expected_rows for spec in TASK_SPECS
            },
            "condition_keys_identical": True,
            "row_hashes_match_dataset": True,
            "gold_matches_dataset": True,
            "strict_summary_complete": True,
            "strict_fingerprint": strict_summary.get("fingerprint"),
        },
        "provenance": {
            "analyzer": {
                "path": str(analyzer_path),
                "sha256": sha256_file(analyzer_path),
            },
            "eval_dir": str(eval_dir.resolve()),
            "dataset_training_root": str(training_root),
            "strict_artifacts": strict_provenance,
            "raw_generations": raw_provenance,
            "datasets": dataset_provenance,
        },
        "bootstrap_protocol": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "interval": "paired percentile interval with deterministic type-7 interpolation",
            "resampling_unit": (
                "Paired evaluation rows. Narrative units and scene TP/FP/FN remain clustered "
                "within their source row."
            ),
        },
        "conditions": {
            condition: {
                "metrics": metrics[condition],
                "resources": condition_resources(records_by_condition[condition]),
            }
            for condition in CONDITIONS
        },
        "normal_vs_base": normal_vs_base,
        "normal_vs_no_write": normal_vs_no_write,
        "paired_bootstrap_comparisons": {
            "normal_minus_base": normal_vs_base,
            "normal_minus_no_write": normal_vs_no_write,
        },
        "condition_diagnostics": condition_diagnostics,
        "paired_changes": paired,
        "selection_criterion": selection_criterion,
        "legacy_multitask_selection_criterion": legacy_selection_criterion,
        "scene_v6_gate_analysis": scene_v6_gate_analysis,
        "strict_comparison": strict_comparison,
        "reference_comparison": reference_comparison,
    }


def main() -> None:
    global CONDITIONS, TASK_SPECS, EXPECTED_ROWS_PER_CONDITION
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    split = args.split or infer_evaluation_split(eval_dir)
    CONDITIONS = evaluation_conditions(eval_dir)
    TASK_SPECS = evaluation_task_specs(eval_dir, split)
    EXPECTED_ROWS_PER_CONDITION = sum(spec.expected_rows for spec in TASK_SPECS)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else eval_dir / "format_recovered_summary.json"
    )
    protected_paths = {
        *(eval_dir / f"{condition}.jsonl" for condition in CONDITIONS),
        eval_dir / "manifest.json",
        eval_dir / "summary.json",
        eval_dir / "progress.json",
    }
    if output_path in protected_paths:
        raise ValueError(f"Refusing to overwrite protected evaluation input: {output_path}")
    training_root = resolve_training_root(args.dataset_root.resolve())
    strict_summary, strict_provenance = validate_strict_artifacts(eval_dir)
    samples_by_task, dataset_provenance = load_dataset_samples(training_root)
    contract = evaluation_contract(eval_dir)
    if contract.get("name") in SCENE_V6_CONTRACT_ROWS:
        scene_provenance = dataset_provenance.get("scene-v4-current", {})
        expected_dataset_hash = OFFICIAL_SCENE_V4_SHA256[split]
        if scene_provenance.get("sha256") != expected_dataset_hash:
            raise ValueError(
                "Scene V6 analyzer requires the official scene-v4 dataset at "
                f"revision {OFFICIAL_SCENE_V4_DATASET_REVISION}"
            )
        if contract.get("official_dataset_sha256") != expected_dataset_hash:
            raise ValueError("Scene V6 contract official dataset hash differs")
    protected_contract = contract.get("name") != "generic"
    records_by_condition, raw_provenance = validate_records(
        eval_dir,
        samples_by_task,
        expected_fingerprint=(
            str(strict_summary["fingerprint"]) if protected_contract else None
        ),
        split=split,
        normal_fusion_profile=(
            str(contract.get("normal_fusion_profile"))
            if protected_contract
            else None
        ),
    )
    output = build_output(
        eval_dir=eval_dir,
        training_root=training_root,
        split=split,
        strict_summary=strict_summary,
        strict_provenance=strict_provenance,
        raw_provenance=raw_provenance,
        dataset_provenance=dataset_provenance,
        records_by_condition=records_by_condition,
        samples_by_task=samples_by_task,
    )
    write_json_atomic(output_path, output)
    print(
        "FORMAT_RECOVERY_COMPLETE "
        f"non_official=true split={split} rows_per_condition={EXPECTED_ROWS_PER_CONDITION} "
        f"bootstrap_replicates={BOOTSTRAP_REPLICATES} output={output_path}"
    )


if __name__ == "__main__":
    main()
