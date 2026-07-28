#!/usr/bin/env python3
"""Run a bounded, provenance-locked base evaluation on scene-v4 train rows.

This producer exists solely to mine base-model failures for paired-data training.
It is deliberately separate from ``run_scene_state_eval.py``, whose validation
rows and ``base_full`` condition are diagnostic and must never become training
data.

Candidate rows are selected deterministically from hashes of user prompts. The
selection therefore does not inspect gold labels, model outputs, or adapter
outputs. ``--prepare-only`` writes and validates all provenance artifacts
without loading a model or starting inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from .common import load_model_and_tokenizer  # type: ignore[import-not-found]
    from .prepare_scene_failure_pairs import (
        SourceRow,
        load_source_split,
        sha256_file,
        sha256_text,
    )
    from .run_novel_agent_eval import (
        append_record,
        clear_model_memory,
        extract_json,
        generate_one,
        git_revision,
        read_records,
        score_prediction,
        utc_now,
        write_json_atomic,
    )
except ImportError:  # Direct script execution.
    from common import load_model_and_tokenizer
    from prepare_scene_failure_pairs import (
        SourceRow,
        load_source_split,
        sha256_file,
        sha256_text,
    )
    from run_novel_agent_eval import (
        append_record,
        clear_model_memory,
        extract_json,
        generate_one,
        git_revision,
        read_records,
        score_prediction,
        utc_now,
        write_json_atomic,
    )


SCHEMA = "rwkv_ms_scene_train_base_eval.v1"
SELECTION_SCHEMA = "rwkv_ms_scene_train_base_selection.v1"
TASK_NAME = "scene-v4-current"
TASK_KIND = "scene"
SPLIT = "train"
CONDITION = "base"
DEFAULT_CANDIDATE_COUNT = 64
MAX_CANDIDATE_ROWS = 64
DEFAULT_SELECTION_SEED = 3407
DEFAULT_MAX_NEW_TOKENS = 128
SHA256_RE = re.compile(r"[0-9a-f]{64}")
OUTPUT_FILENAMES = (
    "candidate_selection.json",
    "base.jsonl",
    "manifest.json",
    "progress.json",
    "summary.json",
)
BASE_RUNTIME_ARTIFACT_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
GENERATION_PACKAGE_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "safetensors",
    "tokenizers",
    "huggingface-hub",
    "accelerate",
    "numpy",
    "sentencepiece",
    "protobuf",
)


@dataclass(frozen=True)
class PreparedRun:
    dataset_file: Path
    selected_rows: list[SourceRow]
    fingerprint: str
    manifest: dict[str, Any]

    @property
    def selected_by_index(self) -> dict[int, SourceRow]:
        return {row.line_index: row for row in self.selected_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset-sha256",
        required=True,
        help="Required SHA-256 lock for the official train.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--prepare-only",
        "--dry-run",
        dest="prepare_only",
        action="store_true",
        help="Write selection/provenance artifacts without loading the model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _artifact_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def local_base_model_artifacts(base_model: Path) -> dict[str, Any]:
    """Hash all local weights and generation-relevant model assets."""

    weight_paths = sorted(
        (path for path in base_model.rglob("*.safetensors") if path.is_file()),
        key=lambda path: path.relative_to(base_model).as_posix(),
    )
    if not weight_paths:
        raise FileNotFoundError(
            f"Local base model contains no *.safetensors weights: {base_model}"
        )
    runtime_paths = sorted(
        (
            path
            for path in base_model.rglob("*")
            if path.is_file()
            and (
                path.name in BASE_RUNTIME_ARTIFACT_NAMES
                or path.name.startswith("tokenizer.")
                or path.name.startswith("chat_template.")
            )
            and path.suffix != ".safetensors"
        ),
        key=lambda path: path.relative_to(base_model).as_posix(),
    )
    runtime_names = {path.name for path in runtime_paths}
    if "config.json" not in runtime_names:
        raise FileNotFoundError(f"Missing base-model config: {base_model / 'config.json'}")
    if not ({"tokenizer.json", "tokenizer.model", "spiece.model"} & runtime_names):
        raise FileNotFoundError(
            "Local base model is missing a tokenizer payload "
            f"(tokenizer.json/tokenizer.model/spiece.model): {base_model}"
        )

    weights = [_artifact_record(path, root=base_model) for path in weight_paths]
    runtime = [_artifact_record(path, root=base_model) for path in runtime_paths]
    aggregate_payload = {"weights": weights, "runtime_artifacts": runtime}
    return {
        "root": str(base_model),
        **aggregate_payload,
        "aggregate_sha256": canonical_json_sha256(aggregate_payload),
    }


def generation_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in GENERATION_PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def select_candidate_rows(
    train_rows: list[SourceRow],
    *,
    count: int,
    seed: int,
) -> list[SourceRow]:
    """Select rows using user-prompt hashes only, then restore source order."""

    if count <= 0:
        raise ValueError("candidate_count must be positive")
    if count > MAX_CANDIDATE_ROWS:
        raise ValueError(
            f"candidate_count is capped at {MAX_CANDIDATE_ROWS}; received {count}"
        )
    if count > len(train_rows):
        raise ValueError(
            f"candidate_count exceeds train rows: count={count} rows={len(train_rows)}"
        )
    ranked = sorted(
        train_rows,
        key=lambda row: (
            sha256_text(f"{seed}\0{row.prompt_sha256}"),
            row.prompt_sha256,
            row.line_index,
        ),
    )
    selected_indices = {row.line_index for row in ranked[:count]}
    return [row for row in train_rows if row.line_index in selected_indices]


def selection_payload(
    *,
    dataset_file: Path,
    dataset_sha256: str,
    selected_rows: list[SourceRow],
    candidate_count: int,
    selection_seed: int,
) -> dict[str, Any]:
    return {
        "schema": SELECTION_SCHEMA,
        "task": TASK_NAME,
        "split": SPLIT,
        "dataset_file": str(dataset_file),
        "dataset_sha256": dataset_sha256,
        "candidate_count": candidate_count,
        "selection_seed": selection_seed,
        "selection_basis": "sha256(selection_seed + NUL + user_prompt_sha256)",
        "selection_uses_gold_labels": False,
        "selection_uses_model_output": False,
        "rows": [
            {
                "source_index": row.line_index,
                "row_sha256": row.row_sha256,
                "user_prompt_sha256": row.prompt_sha256,
            }
            for row in selected_rows
        ],
    }


def code_fingerprints() -> dict[str, str]:
    return {
        "producer_sha256": sha256_file(Path(__file__)),
        "source_loader_sha256": sha256_file(
            SCRIPT_DIR / "prepare_scene_failure_pairs.py"
        ),
        "generation_runner_sha256": sha256_file(
            SCRIPT_DIR / "run_novel_agent_eval.py"
        ),
        "common_sha256": sha256_file(SCRIPT_DIR / "common.py"),
        "scene_recovery_sha256": sha256_file(
            SCRIPT_DIR / "analyze_novel_agent_eval.py"
        ),
        "chat_templates_sha256": sha256_file(
            PROJECT_ROOT / "deltamem" / "chat_templates.py"
        ),
    }


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _reject_unmanaged_jsonl(output_dir: Path) -> None:
    managed_jsonl = output_dir / "base.jsonl"
    unmanaged = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*.jsonl")
        if path != managed_jsonl
    )
    if unmanaged:
        raise ValueError(
            "Output directory contains unrelated stale JSONL state; use a dedicated "
            f"output directory or remove it explicitly: {unmanaged}"
        )


def _remove_outputs(output_dir: Path) -> None:
    _reject_unmanaged_jsonl(output_dir)
    for filename in OUTPUT_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)


def prepare_run(args: argparse.Namespace) -> PreparedRun:
    dataset_file = args.dataset_file.expanduser().resolve()
    base_model = args.base_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if dataset_file.name != "train.jsonl":
        raise ValueError(
            "Scene failure mining requires the official train.jsonl; "
            "validation and test rows are forbidden"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Missing train dataset: {dataset_file}")
    base_config_path = base_model / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Missing base-model config: {base_config_path}")

    expected_dataset_sha256 = _validate_sha256(
        args.expected_dataset_sha256,
        name="expected_dataset_sha256",
    )
    dataset_sha256 = sha256_file(dataset_file)
    if dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "Dataset SHA-256 differs from --expected-dataset-sha256: "
            f"expected={expected_dataset_sha256} actual={dataset_sha256}"
        )

    train_rows = load_source_split(dataset_file, split=SPLIT)
    selected_rows = select_candidate_rows(
        train_rows,
        count=args.candidate_count,
        seed=args.selection_seed,
    )
    selection = selection_payload(
        dataset_file=dataset_file,
        dataset_sha256=dataset_sha256,
        selected_rows=selected_rows,
        candidate_count=args.candidate_count,
        selection_seed=args.selection_seed,
    )
    code = code_fingerprints()
    base_artifacts = local_base_model_artifacts(base_model)
    runtime_versions = generation_runtime_versions()
    fingerprint_payload = {
        "schema": SCHEMA,
        "task": TASK_NAME,
        "task_kind": TASK_KIND,
        "condition": CONDITION,
        "split": SPLIT,
        "code": code,
        "base_model": str(base_model),
        "base_config_sha256": sha256_file(base_config_path),
        "base_model_artifacts": base_artifacts,
        "generation_runtime_versions": runtime_versions,
        "dataset_file": str(dataset_file),
        "dataset_sha256": dataset_sha256,
        "selection_sha256": canonical_json_sha256(selection),
        "selected_rows": selection["rows"],
        "candidate_count": args.candidate_count,
        "selection_seed": args.selection_seed,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    fingerprint = canonical_json_sha256(fingerprint_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_unmanaged_jsonl(output_dir)
    if args.overwrite:
        _remove_outputs(output_dir)

    selection_path = output_dir / "candidate_selection.json"
    if selection_path.is_file():
        existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing_selection != selection:
            raise ValueError(
                f"Candidate selection differs at {selection_path}; use --overwrite"
            )
    else:
        write_json_atomic(selection_path, selection)

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "evaluation_kind": "bounded official-train base failure mining",
        "warning": (
            "These selected train rows may be used only for base-failure mining. "
            "Never substitute validation or test records."
        ),
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "rows": len(selected_rows),
            "uses_gold_labels": False,
            "uses_model_output": False,
        },
        "output": {
            "base_records": str(output_dir / "base.jsonl"),
            "builder_argument": "--base-train-eval-jsonl",
        },
        "code": {
            "rwkv_repo": git_revision(PROJECT_ROOT),
            **code,
        },
    }
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_payload = existing_manifest.get("fingerprint_payload")
        recorded_fingerprint = existing_manifest.get("fingerprint")
        if (
            recorded_payload != fingerprint_payload
            or recorded_fingerprint != fingerprint
            or canonical_json_sha256(recorded_payload) != recorded_fingerprint
        ):
            raise ValueError(
                f"Output manifest fingerprint differs at {manifest_path}; "
                "use --overwrite or a new output directory"
            )
        manifest = existing_manifest
    else:
        write_json_atomic(manifest_path, manifest)

    return PreparedRun(
        dataset_file=dataset_file,
        selected_rows=selected_rows,
        fingerprint=fingerprint,
        manifest=manifest,
    )


def build_base_record(
    source: SourceRow,
    generation: dict[str, Any],
    *,
    fingerprint: str,
    max_new_tokens: int,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build the exact JSONL record contract consumed by the pair builder."""

    if generation.get("status") != "ok":
        raise ValueError("Generation status must be 'ok'")
    raw_generation = generation.get("raw_generation")
    if not isinstance(raw_generation, str):
        raise ValueError("Generation is missing raw_generation")
    if extract_json(raw_generation) != generation.get("parsed_json"):
        raise ValueError("Generation parsed_json does not match raw_generation")
    protected = {
        "key",
        "condition",
        "task",
        "task_kind",
        "split",
        "line_index",
        "row_sha256",
        "gold",
        "fingerprint",
        "max_new_tokens",
        "completed_at",
        "score",
    }
    conflicts = protected & set(generation)
    if conflicts:
        raise ValueError(f"Generation contains protected record fields: {sorted(conflicts)}")
    runtime_fields = dict(generation)
    runtime_fields.pop("status")
    parsed_json = runtime_fields["parsed_json"]
    return {
        "key": f"{TASK_NAME}:{source.line_index}",
        "condition": CONDITION,
        "task": TASK_NAME,
        "task_kind": TASK_KIND,
        "split": SPLIT,
        "line_index": source.line_index,
        "row_sha256": source.row_sha256,
        "gold": source.gold,
        "status": "ok",
        "fingerprint": fingerprint,
        "max_new_tokens": max_new_tokens,
        "score": score_prediction(TASK_KIND, parsed_json, source.gold),
        "completed_at": completed_at or utc_now(),
        **runtime_fields,
    }


def validate_resume_records(
    records: Iterable[dict[str, Any]],
    *,
    selected_by_index: dict[int, SourceRow],
    fingerprint: str,
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for record in records:
        line_index = record.get("line_index")
        if isinstance(line_index, bool) or not isinstance(line_index, int):
            raise ValueError("Resume record has an invalid line_index")
        source = selected_by_index.get(line_index)
        if source is None:
            raise ValueError(f"Resume record row is not in the candidate selection: {line_index}")
        expected_fields = {
            "key": f"{TASK_NAME}:{line_index}",
            "condition": CONDITION,
            "task": TASK_NAME,
            "task_kind": TASK_KIND,
            "split": SPLIT,
            "status": "ok",
            "fingerprint": fingerprint,
            "row_sha256": source.row_sha256,
            "gold": source.gold,
        }
        mismatches = {
            field: {"expected": expected, "actual": record.get(field)}
            for field, expected in expected_fields.items()
            if record.get(field) != expected
        }
        if mismatches:
            raise ValueError(
                f"Resume record contract differs for source index {line_index}: {mismatches}"
            )
        raw_generation = record.get("raw_generation")
        if not isinstance(raw_generation, str):
            raise ValueError(f"Resume record is missing raw_generation: {line_index}")
        if extract_json(raw_generation) != record.get("parsed_json"):
            raise ValueError(
                f"Resume record parsed_json differs from raw_generation: {line_index}"
            )
        if line_index in completed:
            raise ValueError(f"Duplicate resume record for source index {line_index}")
        completed[line_index] = record
    return completed


def _summary(
    prepared: PreparedRun,
    completed: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    ordered = [
        completed[row.line_index]
        for row in prepared.selected_rows
        if row.line_index in completed
    ]
    tp = sum(int(record["score"]["tp"]) for record in ordered)
    fp = sum(int(record["score"]["fp"]) for record in ordered)
    fn = sum(int(record["score"]["fn"]) for record in ordered)
    denominator = 2 * tp + fp + fn
    return {
        "schema": SCHEMA,
        "updated_at": utc_now(),
        "fingerprint": prepared.fingerprint,
        "complete": len(ordered) == len(prepared.selected_rows),
        "completed": len(ordered),
        "expected": len(prepared.selected_rows),
        "condition": CONDITION,
        "task": TASK_NAME,
        "split": SPLIT,
        "strict_micro_f1": 0.0 if denominator == 0 else (2 * tp) / denominator,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_inference(args: argparse.Namespace, prepared: PreparedRun) -> dict[str, Any]:
    records_path = args.output_dir.expanduser().resolve() / "base.jsonl"
    completed = validate_resume_records(
        read_records(records_path),
        selected_by_index=prepared.selected_by_index,
        fingerprint=prepared.fingerprint,
    )
    pending = [
        row for row in prepared.selected_rows if row.line_index not in completed
    ]
    if pending:
        model, tokenizer = load_model_and_tokenizer(
            base_model=str(args.base_model.expanduser().resolve()),
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        try:
            for source in pending:
                generation = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    messages=source.messages[:-1],
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                )
                record = build_base_record(
                    source,
                    generation,
                    fingerprint=prepared.fingerprint,
                    max_new_tokens=args.max_new_tokens,
                )
                append_record(records_path, record)
                completed[source.line_index] = record
                write_json_atomic(
                    args.output_dir / "progress.json",
                    {
                        "schema": SCHEMA,
                        "updated_at": utc_now(),
                        "fingerprint": prepared.fingerprint,
                        "completed": len(completed),
                        "expected": len(prepared.selected_rows),
                        "last_key": record["key"],
                    },
                )
                print(
                    "SCENE_TRAIN_BASE_EVAL "
                    f"source_index={source.line_index} "
                    f"completed={len(completed)}/{len(prepared.selected_rows)}",
                    flush=True,
                )
        finally:
            del model
            del tokenizer
            clear_model_memory()

    summary = _summary(prepared, completed)
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_json_atomic(
        args.output_dir / "progress.json",
        {
            "schema": SCHEMA,
            "updated_at": utc_now(),
            "fingerprint": prepared.fingerprint,
            "completed": len(completed),
            "expected": len(prepared.selected_rows),
            "complete": len(completed) == len(prepared.selected_rows),
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepared = prepare_run(args)
    records_path = args.output_dir / "base.jsonl"
    completed = validate_resume_records(
        read_records(records_path),
        selected_by_index=prepared.selected_by_index,
        fingerprint=prepared.fingerprint,
    )
    if args.prepare_only:
        progress = {
            "schema": SCHEMA,
            "updated_at": utc_now(),
            "fingerprint": prepared.fingerprint,
            "mode": "prepare_only",
            "model_loaded": False,
            "completed": len(completed),
            "expected": len(prepared.selected_rows),
        }
        write_json_atomic(args.output_dir / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
        return
    summary = run_inference(args, prepared)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
