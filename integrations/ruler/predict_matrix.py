#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import sys
import threading
from typing import TextIO

from task_config import (
    EVAL_TASKS,
    GEMMA4_CHAT_TEMPLATE_NAME,
    OFFICIAL_EVAL_SEED,
    TASK_TOKEN_BUDGETS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICT_SCRIPT = Path(__file__).with_name("predict.py")
DEFAULT_SCORE_SCRIPT = Path(__file__).with_name("score.py")
TOP_LEVEL_MANIFEST = "prediction_matrix.manifest.json"
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
MODEL_METADATA_SUFFIXES = (".json", ".jinja", ".model", ".txt")


@dataclass(frozen=True)
class MatrixJob:
    context_length: int
    task: str
    input_file: Path
    input_sha256: str
    rows: int


@dataclass(frozen=True)
class InputMatrix:
    manifest_file: Path
    manifest_sha256: str
    subset: str
    template_name: str
    seed: int
    lengths: tuple[int, ...]
    tasks: tuple[str, ...]
    rows_per_job: int
    jobs: tuple[MatrixJob, ...]


@dataclass(frozen=True)
class WorkUnit:
    job: MatrixJob
    chunk_index: int
    chunk_count: int
    output_file: Path
    log_file: Path

    @property
    def selected_ordinals(self) -> tuple[int, ...]:
        return tuple(range(self.chunk_index, self.job.rows, self.chunk_count))


@dataclass(frozen=True)
class RunConfig:
    variant: str
    model_path: str
    model_identity: dict[str, object]
    adapter_dir: Path | None
    adapter_sha256: str | None
    adapter_config_sha256: str | None
    runtime_root: Path
    runtime_git_identity: dict[str, object]
    dtype: str
    attn_implementation: str
    max_new_tokens: int
    python_bin: str
    predict_script: Path
    predict_script_sha256: str
    score_script: Path
    score_script_sha256: str
    overwrite_output: bool
    prefill_chunk_tokens: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one verified RULER generation matrix through predict.py across exclusive GPUs."
        )
    )
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "hybrid"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument(
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated physical GPU IDs or UUIDs (default: CUDA_VISIBLE_DEVICES or 0)",
    )
    parser.add_argument("--chunks-per-input", type=int, default=1)
    parser.add_argument(
        "--expected-rows-per-job",
        type=int,
        default=0,
        help="Optional exact sample-count assertion (default: trust the verified manifest)",
    )
    parser.add_argument(
        "--expected-seed",
        type=int,
        default=OFFICIAL_EVAL_SEED,
        help=(
            "Required seed for every input job; defaults to the official evaluation seed. "
            "Use an explicit held-out seed only for checkpoint selection."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=0,
        help="Prefill cache chunk size in tokens; 0 keeps monolithic prefill",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--predict-script", type=Path, default=DEFAULT_PREDICT_SCRIPT)
    parser.add_argument("--score-script", type=Path, default=DEFAULT_SCORE_SCRIPT)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_identity(path: Path) -> dict[str, object]:
    revision = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        raise ValueError(f"Runtime root is not a Git checkout: {path}")
    status = subprocess.run(
        ("git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    listed_files = subprocess.run(
        (
            "git",
            "-C",
            str(path),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ),
        check=True,
        capture_output=True,
    ).stdout
    tree_records = []
    for raw_relative_path in listed_files.split(b"\0"):
        if not raw_relative_path:
            continue
        relative_path = raw_relative_path.decode("utf-8", errors="surrogateescape")
        candidate = path / relative_path
        tree_records.append(
            {
                "path": relative_path,
                "sha256": sha256_file(candidate) if candidate.is_file() else None,
            }
        )
    tree_canonical = json.dumps(
        tree_records,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "root": str(path),
        "revision": revision.stdout.strip(),
        "dirty": bool(status),
        "status_sha256": sha256_text(status),
        "tree_sha256": sha256_text(tree_canonical),
        "tree_file_count": len(tree_records),
    }


def identity_file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def model_identity(model_path: str) -> dict[str, object]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "Matrix prediction requires a local model directory for weight provenance: "
            f"{root}"
        )
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found: {root / 'config.json'}")
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    weight_paths = tuple(path for path in files if path.suffix in MODEL_WEIGHT_SUFFIXES)
    if not weight_paths:
        raise FileNotFoundError(f"No local model weight files found under {root}")
    metadata_paths = tuple(
        path
        for path in files
        if path.suffix in MODEL_METADATA_SUFFIXES and path not in weight_paths
    )
    weight_files = [identity_file_record(path, root) for path in weight_paths]
    metadata_files = [identity_file_record(path, root) for path in metadata_paths]
    canonical = json.dumps(
        {"metadata_files": metadata_files, "weight_files": weight_files},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "path": str(root),
        "sha256": sha256_text(canonical),
        "metadata_files": metadata_files,
        "weight_files": weight_files,
        "weight_bytes": sum(int(record["size_bytes"]) for record in weight_files),
    }


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def inspect_input_jsonl(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if not isinstance(row.get("input"), str) or not row["input"]:
                raise ValueError(f"Missing RULER input at {path}:{line_number}")
            if not isinstance(row.get("outputs"), list):
                raise ValueError(f"Missing RULER outputs at {path}:{line_number}")
            rows += 1
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def resolve_record_path(raw_path: object, manifest_file: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Invalid input path in {manifest_file}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_file.parent / path
    return path.resolve()


def load_input_matrix(
    manifest_file: Path,
    *,
    expected_seed: int = OFFICIAL_EVAL_SEED,
) -> InputMatrix:
    manifest_file = manifest_file.expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Generation manifest not found: {manifest_file}")
    payload = load_json(manifest_file)
    if payload.get("mode") != "eval":
        raise ValueError("Prediction matrices must come from prepare_data.py --mode eval")
    subset = str(payload.get("subset", ""))
    if subset != "validation":
        raise ValueError(f"RULER evaluation subset must be validation, found {subset!r}")
    template_name = str(payload.get("template_name", ""))
    if template_name != GEMMA4_CHAT_TEMPLATE_NAME:
        raise ValueError(
            f"RULER evaluation template must be {GEMMA4_CHAT_TEMPLATE_NAME!r}, "
            f"found {template_name!r}"
        )

    lengths = tuple(int(value) for value in payload.get("lengths", ()))
    tasks = tuple(str(value) for value in payload.get("tasks", ()))
    if not lengths or len(set(lengths)) != len(lengths) or any(value <= 0 for value in lengths):
        raise ValueError("Generation manifest has invalid or duplicate context lengths")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Generation manifest has invalid or duplicate tasks")
    unknown_tasks = sorted(set(tasks) - set(EVAL_TASKS))
    if unknown_tasks:
        raise ValueError(f"Generation manifest has unknown RULER tasks: {unknown_tasks}")
    rows_per_job = int(payload.get("num_samples_per_task_length", 0))
    if rows_per_job <= 0:
        raise ValueError("Generation manifest has invalid num_samples_per_task_length")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Generation manifest has no jobs")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(raw_jobs):
        raise ValueError("Generation manifest has invalid per-job seeds")
    if {int(seed) for seed in seeds} != {expected_seed}:
        raise ValueError(f"RULER evaluation matrix must use expected seed {expected_seed}")
    jobs = []
    seen_keys: set[tuple[int, str]] = set()
    seen_paths: set[Path] = set()
    for record in raw_jobs:
        if not isinstance(record, dict):
            raise ValueError("Generation manifest jobs must be objects")
        context_length = int(record.get("context_length", 0))
        task = str(record.get("task", ""))
        if int(record.get("seed", -1)) != expected_seed:
            raise ValueError(
                f"RULER evaluation job {context_length}/{task} does not use "
                f"expected seed {expected_seed}"
            )
        if record.get("subset") != subset or record.get("template_name") != template_name:
            raise ValueError(f"RULER job metadata mismatch for {context_length}/{task}")
        key = (context_length, task)
        if key in seen_keys:
            raise ValueError(f"Duplicate matrix job for length={context_length}, task={task}")
        seen_keys.add(key)
        input_file = resolve_record_path(record.get("path"), manifest_file)
        if input_file in seen_paths:
            raise ValueError(f"Matrix jobs reuse an input file: {input_file}")
        seen_paths.add(input_file)
        if not input_file.is_file():
            raise FileNotFoundError(f"RULER input not found: {input_file}")
        expected_digest = record.get("sha256")
        actual_digest = sha256_file(input_file)
        if expected_digest != actual_digest:
            raise ValueError(f"Input hash mismatch for {input_file}")
        actual_rows = inspect_input_jsonl(input_file)
        if int(record.get("rows", 0)) != actual_rows:
            raise ValueError(f"Input row-count mismatch for {input_file}")
        if actual_rows != rows_per_job:
            raise ValueError(f"Non-uniform row count for {input_file}")
        jobs.append(
            MatrixJob(
                context_length=context_length,
                task=task,
                input_file=input_file,
                input_sha256=actual_digest,
                rows=actual_rows,
            )
        )

    expected_keys = {(length, task) for length in lengths for task in tasks}
    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        unexpected = sorted(seen_keys - expected_keys)
        raise ValueError(
            f"Generation manifest is not a complete matrix; missing={missing}, "
            f"extra={unexpected}"
        )
    expected_total = sum(job.rows for job in jobs)
    if int(payload.get("total_rows", -1)) != expected_total:
        raise ValueError("Generation manifest total_rows does not match its verified inputs")

    return InputMatrix(
        manifest_file=manifest_file,
        manifest_sha256=sha256_file(manifest_file),
        subset=subset,
        template_name=template_name,
        seed=expected_seed,
        lengths=lengths,
        tasks=tasks,
        rows_per_job=rows_per_job,
        jobs=tuple(jobs),
    )


def parse_devices(raw_devices: str) -> tuple[str, ...]:
    devices = tuple(value.strip() for value in raw_devices.split(",") if value.strip())
    if not devices:
        raise ValueError("At least one GPU device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU devices must be unique")
    return devices


def build_work_units(
    matrix: InputMatrix,
    output_root: Path,
    chunks_per_input: int,
) -> tuple[WorkUnit, ...]:
    if chunks_per_input < 1:
        raise ValueError("chunks-per-input must be >= 1")
    output_root = output_root.expanduser().resolve()
    units = []
    for job in matrix.jobs:
        if chunks_per_input > job.rows:
            raise ValueError(
                f"chunks-per-input={chunks_per_input} exceeds rows={job.rows} for {job.input_file}"
            )
        for chunk_index in range(chunks_per_input):
            if chunks_per_input == 1:
                stem = job.task
            else:
                stem = f"{job.task}-chunk-{chunk_index:03d}-of-{chunks_per_input:03d}"
            units.append(
                WorkUnit(
                    job=job,
                    chunk_index=chunk_index,
                    chunk_count=chunks_per_input,
                    output_file=output_root / str(job.context_length) / f"{stem}.jsonl",
                    log_file=output_root / "logs" / str(job.context_length) / f"{stem}.log",
                )
            )
    output_files = [unit.output_file for unit in units]
    if len(set(output_files)) != len(output_files):
        raise ValueError("Work-unit output paths are not unique")
    return tuple(units)


def output_manifest_path(output_file: Path) -> Path:
    return output_file.with_suffix(output_file.suffix + ".manifest.json")


def effective_max_new_tokens(unit: WorkUnit, config: RunConfig) -> int:
    return config.max_new_tokens or TASK_TOKEN_BUDGETS[unit.job.task]


def expected_output_identity(unit: WorkUnit, config: RunConfig) -> dict[str, object]:
    return {
        "variant": config.variant,
        "task": unit.job.task,
        "input_file": str(unit.job.input_file),
        "input_sha256": unit.job.input_sha256,
        "output_file": str(unit.output_file),
        "model_path": str(Path(config.model_path).expanduser().resolve()),
        "adapter_dir": None if config.adapter_dir is None else str(config.adapter_dir),
        "adapter_sha256": config.adapter_sha256,
        "adapter_config_sha256": config.adapter_config_sha256,
        "runtime_root": str(config.runtime_root),
        "device": "cuda:0",
        "dtype": config.dtype,
        "attn_implementation": config.attn_implementation,
        "max_new_tokens": effective_max_new_tokens(unit, config),
        "prefill_chunk_tokens": config.prefill_chunk_tokens,
        "chunk_index": unit.chunk_index,
        "chunk_count": unit.chunk_count,
        "selected_rows": len(unit.selected_ordinals),
    }


def inspect_output_rows(unit: WorkUnit, config: RunConfig) -> set[int]:
    source_rows = []
    with unit.job.input_file.open("r", encoding="utf-8") as source_handle:
        for raw_line in source_handle:
            if raw_line.strip():
                source_rows.append(json.loads(raw_line))
    expected_state_modules = 0
    if config.variant == "hybrid":
        child_manifest = load_json(output_manifest_path(unit.output_file))
        replaced_modules = child_manifest.get("replaced_modules")
        if not isinstance(replaced_modules, list) or not replaced_modules:
            raise ValueError(f"Hybrid output has no replaced modules: {unit.output_file}")
        if len(set(replaced_modules)) != len(replaced_modules):
            raise ValueError(f"Hybrid output has duplicate replaced modules: {unit.output_file}")
        expected_state_modules = len(replaced_modules)
    ordinals: set[int] = set()
    with unit.output_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {unit.output_file}:{line_number}")
            if row.get("variant") != config.variant or row.get("task") != unit.job.task:
                raise ValueError(f"Cross-run prediction row at {unit.output_file}:{line_number}")
            ordinal = int(row.get("sample_ordinal", -1))
            if ordinal in ordinals:
                raise ValueError(f"Duplicate sample_ordinal={ordinal} in {unit.output_file}")
            if ordinal < 0 or ordinal >= len(source_rows):
                raise ValueError(f"Out-of-range sample_ordinal={ordinal} in {unit.output_file}")
            source_row = source_rows[ordinal]
            for key in ("input", "outputs"):
                if row.get(key) != source_row.get(key):
                    raise ValueError(
                        f"Prediction row does not match source {key} at ordinal={ordinal}: "
                        f"{unit.output_file}"
                    )
            state_stats = row.get("state_stats")
            if config.variant == "hybrid":
                if not isinstance(state_stats, dict):
                    raise ValueError(f"Hybrid row has no state stats: {unit.output_file}:{line_number}")
                if state_stats.get("num_modules") != expected_state_modules:
                    raise ValueError(
                        f"Hybrid state-module count mismatch: {unit.output_file}:{line_number}"
                    )
                if state_stats.get("nonzero_modules") != expected_state_modules:
                    raise ValueError(
                        f"Hybrid state is not nonzero in every module: "
                        f"{unit.output_file}:{line_number}"
                    )
            elif state_stats is not None:
                raise ValueError(f"Base prediction row has hybrid state stats: {unit.output_file}")
            prompt_tokens = row.get("prompt_tokens")
            if type(prompt_tokens) is not int or prompt_tokens < 1:
                raise ValueError(
                    f"Prediction row has invalid prompt token count: "
                    f"{unit.output_file}:{line_number}"
                )
            expected_prefill_chunks = (
                1
                if config.prefill_chunk_tokens <= 0
                else (prompt_tokens + config.prefill_chunk_tokens - 1)
                // config.prefill_chunk_tokens
            )
            if row.get("prefill_chunk_count") != expected_prefill_chunks:
                raise ValueError(
                    f"Prediction row has invalid prefill chunk count: "
                    f"{unit.output_file}:{line_number}"
                )
            peak_allocated = row.get("cuda_peak_allocated_bytes")
            peak_reserved = row.get("cuda_peak_reserved_bytes")
            if (
                type(peak_allocated) is not int
                or peak_allocated < 0
                or type(peak_reserved) is not int
                or peak_reserved < peak_allocated
            ):
                raise ValueError(
                    f"Prediction row has invalid CUDA peak memory stats: "
                    f"{unit.output_file}:{line_number}"
                )
            ordinals.add(ordinal)
    unexpected = ordinals - set(unit.selected_ordinals)
    if unexpected:
        raise ValueError(f"Unexpected sample ordinals in {unit.output_file}: {sorted(unexpected)}")
    return ordinals


def output_status(unit: WorkUnit, config: RunConfig) -> str:
    manifest_file = output_manifest_path(unit.output_file)
    output_exists = unit.output_file.is_file()
    manifest_exists = manifest_file.is_file()
    if not output_exists and not manifest_exists:
        return "pending"
    if output_exists and not manifest_exists:
        raise ValueError(f"Prediction output has no identity manifest: {unit.output_file}")
    manifest = load_json(manifest_file)
    for key, expected in expected_output_identity(unit, config).items():
        if manifest.get(key) != expected:
            raise ValueError(f"Resume metadata mismatch for {unit.output_file}: {key}")
    status = manifest.get("status")
    if status not in ("running", "complete"):
        raise ValueError(f"Invalid prediction status for {unit.output_file}: {status!r}")
    if not output_exists:
        if status == "complete":
            raise FileNotFoundError(f"Completed prediction output is missing: {unit.output_file}")
        return "resume"

    ordinals = inspect_output_rows(unit, config)
    if status == "running":
        return "resume"
    expected_ordinals = set(unit.selected_ordinals)
    if ordinals != expected_ordinals:
        raise ValueError(
            f"Completed prediction output has {len(ordinals)} of {len(expected_ordinals)} rows: "
            f"{unit.output_file}"
        )
    if manifest.get("completed_rows") != len(expected_ordinals):
        raise ValueError(f"Completed-row metadata mismatch for {unit.output_file}")
    if manifest.get("output_sha256") != sha256_file(unit.output_file):
        raise ValueError(f"Completed prediction hash mismatch for {unit.output_file}")
    return "complete"


def verify_matrix_coverage(
    matrix: InputMatrix,
    units: tuple[WorkUnit, ...],
    config: RunConfig,
) -> list[dict[str, object]]:
    coverage = []
    for job in matrix.jobs:
        observed: set[int] = set()
        output_files = []
        for unit in (candidate for candidate in units if candidate.job == job):
            if output_status(unit, config) != "complete":
                raise ValueError(f"Incomplete prediction shard: {unit.output_file}")
            ordinals = inspect_output_rows(unit, config)
            overlap = observed & ordinals
            if overlap:
                raise ValueError(
                    f"Overlapping prediction shards for length={job.context_length}, "
                    f"task={job.task}: {sorted(overlap)}"
                )
            observed.update(ordinals)
            output_files.append(
                {
                    "path": str(unit.output_file),
                    "sha256": sha256_file(unit.output_file),
                    "rows": len(ordinals),
                }
            )
        expected = set(range(job.rows))
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"Prediction coverage mismatch for length={job.context_length}, "
                f"task={job.task}; missing={missing}, extra={unexpected}"
            )
        coverage.append(
            {
                "context_length": job.context_length,
                "task": job.task,
                "generation_manifest_sha256": matrix.manifest_sha256,
                "input_file": str(job.input_file),
                "input_sha256": job.input_sha256,
                "expected_rows": job.rows,
                "observed_rows": len(observed),
                "ordinal_min": min(observed),
                "ordinal_max": max(observed),
                "complete": True,
                "output_files": output_files,
            }
        )
    return coverage


def build_predict_command(unit: WorkUnit, config: RunConfig) -> list[str]:
    command = [
        config.python_bin,
        str(config.predict_script),
        "--input-file",
        str(unit.job.input_file),
        "--output-file",
        str(unit.output_file),
        "--task",
        unit.job.task,
        "--variant",
        config.variant,
        "--model-path",
        config.model_path,
        "--runtime-root",
        str(config.runtime_root),
        "--device",
        "cuda:0",
        "--dtype",
        config.dtype,
        "--attn-implementation",
        config.attn_implementation,
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--prefill-chunk-tokens",
        str(config.prefill_chunk_tokens),
        "--chunk-index",
        str(unit.chunk_index),
        "--chunk-count",
        str(unit.chunk_count),
    ]
    if config.adapter_dir is not None:
        command.extend(("--adapter-dir", str(config.adapter_dir)))
    if config.overwrite_output:
        command.append("--overwrite-output")
    return command


def run_work_unit(
    unit: WorkUnit,
    device: str,
    config: RunConfig,
    log_handle: TextIO | None = None,
) -> None:
    unit.output_file.parent.mkdir(parents=True, exist_ok=True)
    unit.log_file.parent.mkdir(parents=True, exist_ok=True)
    command = build_predict_command(unit, config)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment["PYTHONUNBUFFERED"] = "1"
    owns_log = log_handle is None
    if log_handle is None:
        log_handle = unit.log_file.open("a", encoding="utf-8", buffering=1)
    try:
        log_handle.write("$ " + shlex.join(command) + "\n")
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        if owns_log:
            log_handle.close()


def run_work_queue(
    units: tuple[WorkUnit, ...],
    devices: tuple[str, ...],
    config: RunConfig,
) -> None:
    queue: Queue[WorkUnit] = Queue()
    for unit in units:
        queue.put(unit)
    stop = threading.Event()
    failures: list[tuple[WorkUnit, BaseException]] = []
    failure_lock = threading.Lock()

    def worker(device: str) -> None:
        while not stop.is_set():
            try:
                unit = queue.get_nowait()
            except Empty:
                return
            try:
                assert_runtime_unchanged(config)
                print(
                    f"RULER_START device={device} length={unit.job.context_length} "
                    f"task={unit.job.task} chunk={unit.chunk_index}/{unit.chunk_count}",
                    flush=True,
                )
                run_work_unit(unit, device, config)
                if output_status(unit, config) != "complete":
                    raise RuntimeError(f"Predictor did not complete {unit.output_file}")
                print(
                    f"RULER_DONE device={device} length={unit.job.context_length} "
                    f"task={unit.job.task} chunk={unit.chunk_index}/{unit.chunk_count}",
                    flush=True,
                )
            except BaseException as error:
                with failure_lock:
                    failures.append((unit, error))
                stop.set()
            finally:
                queue.task_done()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures:
            future.result()
    if failures:
        unit, error = failures[0]
        raise RuntimeError(
            f"Prediction failed for {unit.output_file}; see {unit.log_file}"
        ) from error


def assert_runtime_unchanged(config: RunConfig) -> None:
    current = git_identity(config.runtime_root)
    if current != config.runtime_git_identity:
        raise ValueError("Runtime Git worktree changed during matrix prediction")


def score_paths(output_root: Path, context_length: int) -> tuple[Path, Path, Path]:
    score_root = output_root / "scores"
    return (
        score_root / f"{context_length}.json",
        score_root / f"{context_length}.csv",
        score_root / f"{context_length}.log",
    )


def verify_score_output(
    matrix: InputMatrix,
    units: tuple[WorkUnit, ...],
    config: RunConfig,
    context_length: int,
    output_json: Path,
    output_csv: Path,
) -> dict[str, object]:
    if not output_json.is_file() or not output_csv.is_file():
        raise FileNotFoundError(f"Scorer did not create both outputs for {context_length}")
    payload = load_json(output_json)
    expected_tasks = {
        job.task: job for job in matrix.jobs if job.context_length == context_length
    }
    if payload.get("variant") != config.variant:
        raise ValueError(f"Score variant mismatch for {context_length}")
    if payload.get("context_length") != context_length:
        raise ValueError(f"Score context-length mismatch for {context_length}")
    missing_tasks = [task for task in EVAL_TASKS if task not in matrix.tasks]
    complete_task_matrix = not missing_tasks
    if payload.get("complete_task_matrix") is not complete_task_matrix:
        raise ValueError(f"Score completeness mismatch for {context_length}")
    if payload.get("missing_tasks") != missing_tasks:
        raise ValueError(f"Score missing-task mismatch for {context_length}")
    task_results = payload.get("tasks")
    if not isinstance(task_results, dict) or set(task_results) != set(expected_tasks):
        raise ValueError(f"Score task set mismatch for {context_length}")
    if payload.get("task_count") != len(expected_tasks):
        raise ValueError(f"Score task-count mismatch for {context_length}")
    for task, job in expected_tasks.items():
        result = task_results[task]
        if not isinstance(result, dict) or result.get("rows") != job.rows:
            raise ValueError(f"Score row-count mismatch for {context_length}/{task}")
        expected_units = [
            unit
            for unit in units
            if unit.job.context_length == context_length and unit.job.task == task
        ]
        expected_files = {
            str(unit.output_file): sha256_file(unit.output_file) for unit in expected_units
        }
        score_files = result.get("files")
        if not isinstance(score_files, list):
            raise ValueError(f"Missing score file records for {context_length}/{task}")
        actual_files = {
            str(Path(record["path"]).resolve()): record.get("sha256")
            for record in score_files
            if isinstance(record, dict) and "path" in record
        }
        if actual_files != expected_files:
            raise ValueError(f"Score input-file mismatch for {context_length}/{task}")
    macro_score = payload.get("macro_score")
    if not isinstance(macro_score, (int, float)):
        raise ValueError(f"Missing macro score for {context_length}")
    return {
        "context_length": context_length,
        "macro_score": macro_score,
        "task_count": payload.get("task_count"),
        "rows": sum(job.rows for job in expected_tasks.values()),
        "json": str(output_json),
        "json_sha256": sha256_file(output_json),
        "csv": str(output_csv),
        "csv_sha256": sha256_file(output_csv),
    }


def score_matrix(
    matrix: InputMatrix,
    units: tuple[WorkUnit, ...],
    output_root: Path,
    config: RunConfig,
) -> list[dict[str, object]]:
    records = []
    for context_length in matrix.lengths:
        assert_runtime_unchanged(config)
        output_json, output_csv, log_file = score_paths(output_root, context_length)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            config.python_bin,
            str(config.score_script),
            "--prediction-dir",
            str(output_root / str(context_length)),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--variant",
            config.variant,
            "--context-length",
            str(context_length),
        ]
        if set(matrix.tasks) != set(EVAL_TASKS):
            command.append("--allow-partial")
        mode = "w" if config.overwrite_output else "a"
        with log_file.open(mode, encoding="utf-8", buffering=1) as log_handle:
            log_handle.write("$ " + shlex.join(command) + "\n")
            subprocess.run(
                command,
                check=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        records.append(
            verify_score_output(
                matrix,
                units,
                config,
                context_length,
                output_json,
                output_csv,
            )
        )
    assert_runtime_unchanged(config)
    return records


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_specification(
    matrix: InputMatrix,
    units: tuple[WorkUnit, ...],
    config: RunConfig,
) -> dict[str, object]:
    return {
        "generation_manifest": str(matrix.manifest_file),
        "generation_manifest_sha256": matrix.manifest_sha256,
        "rows_per_job": matrix.rows_per_job,
        "variant": config.variant,
        "model_path": str(Path(config.model_path).expanduser().resolve()),
        "model_identity": config.model_identity,
        "adapter_dir": None if config.adapter_dir is None else str(config.adapter_dir),
        "adapter_sha256": config.adapter_sha256,
        "adapter_config_sha256": config.adapter_config_sha256,
        "runtime_root": str(config.runtime_root),
        "runtime_git_identity": config.runtime_git_identity,
        "dtype": config.dtype,
        "attn_implementation": config.attn_implementation,
        "max_new_tokens": config.max_new_tokens,
        "prefill_chunk_tokens": config.prefill_chunk_tokens,
        "chunks_per_input": units[0].chunk_count,
        "python_bin": config.python_bin,
        "predict_script": str(config.predict_script),
        "predict_script_sha256": config.predict_script_sha256,
        "score_script": str(config.score_script),
        "score_script_sha256": config.score_script_sha256,
    }


def unit_record(unit: WorkUnit, config: RunConfig) -> dict[str, object]:
    return {
        "context_length": unit.job.context_length,
        "task": unit.job.task,
        "input_file": str(unit.job.input_file),
        "input_sha256": unit.job.input_sha256,
        "input_rows": unit.job.rows,
        "chunk_index": unit.chunk_index,
        "chunk_count": unit.chunk_count,
        "selected_rows": len(unit.selected_ordinals),
        "output_file": str(unit.output_file),
        "log_file": str(unit.log_file),
        "status": output_status(unit, config),
    }


def failure_unit_record(unit: WorkUnit, config: RunConfig) -> dict[str, object]:
    try:
        return unit_record(unit, config)
    except Exception as error:
        return {
            "context_length": unit.job.context_length,
            "task": unit.job.task,
            "input_file": str(unit.job.input_file),
            "input_sha256": unit.job.input_sha256,
            "input_rows": unit.job.rows,
            "chunk_index": unit.chunk_index,
            "chunk_count": unit.chunk_count,
            "selected_rows": len(unit.selected_ordinals),
            "output_file": str(unit.output_file),
            "log_file": str(unit.log_file),
            "status": "invalid",
            "status_error": f"{type(error).__name__}: {error}",
        }


def reject_unplanned_outputs(units: tuple[WorkUnit, ...], output_root: Path) -> None:
    planned = {unit.output_file for unit in units}
    lengths = {unit.job.context_length for unit in units}
    unexpected = []
    for context_length in lengths:
        directory = output_root / str(context_length)
        if directory.is_dir():
            unexpected.extend(path for path in directory.glob("*.jsonl") if path not in planned)
    if unexpected:
        raise ValueError(
            "Output root contains prediction files outside this exact matrix: "
            + ", ".join(str(path) for path in sorted(unexpected))
        )


def prepare_run(
    matrix: InputMatrix,
    units: tuple[WorkUnit, ...],
    output_root: Path,
    devices: tuple[str, ...],
    config: RunConfig,
) -> tuple[Path, dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    reject_unplanned_outputs(units, output_root)
    top_manifest = output_root / TOP_LEVEL_MANIFEST
    specification = run_specification(matrix, units, config)
    if config.overwrite_output:
        for unit in units:
            unit.output_file.unlink(missing_ok=True)
            output_manifest_path(unit.output_file).unlink(missing_ok=True)
            unit.log_file.unlink(missing_ok=True)
        for context_length in matrix.lengths:
            for score_path in score_paths(output_root, context_length):
                score_path.unlink(missing_ok=True)
        top_manifest.unlink(missing_ok=True)
    elif top_manifest.is_file():
        existing = load_json(top_manifest)
        if existing.get("run_specification") != specification:
            raise ValueError(f"Output root belongs to another prediction matrix: {output_root}")
    else:
        orphaned = [
            path
            for unit in units
            for path in (unit.output_file, output_manifest_path(unit.output_file))
            if path.exists()
        ]
        if orphaned:
            raise ValueError(
                "Refusing outputs without a matching top-level run manifest: "
                + ", ".join(str(path) for path in orphaned)
            )

    payload = {
        "status": "running",
        "run_specification": specification,
        "devices": list(devices),
        "subset": matrix.subset,
        "template_name": matrix.template_name,
        "seed": matrix.seed,
        "lengths": list(matrix.lengths),
        "tasks": list(matrix.tasks),
        "rows_per_job": matrix.rows_per_job,
        "input_rows": sum(job.rows for job in matrix.jobs),
        "work_units": [unit_record(unit, config) for unit in units],
    }
    atomic_write_json(top_manifest, payload)
    return top_manifest, payload


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    if args.chunks_per_input < 1:
        raise ValueError("chunks-per-input must be >= 1")
    if args.max_new_tokens < 0:
        raise ValueError("max-new-tokens must be >= 0")
    if args.prefill_chunk_tokens < 0:
        raise ValueError("prefill-chunk-tokens must be >= 0")
    predict_script = args.predict_script.expanduser().resolve()
    if not predict_script.is_file():
        raise FileNotFoundError(f"Prediction script not found: {predict_script}")
    score_script = args.score_script.expanduser().resolve()
    if not score_script.is_file():
        raise FileNotFoundError(f"Score script not found: {score_script}")
    runtime_root = args.runtime_root.expanduser().resolve()
    adapter_dir = None
    adapter_sha256 = None
    adapter_config_sha256 = None
    if args.variant == "hybrid":
        if args.adapter_dir is None:
            raise ValueError("Hybrid matrix prediction requires --adapter-dir")
        adapter_dir = args.adapter_dir.expanduser().resolve()
        adapter_file = adapter_dir / "delta_mem_adapter.pt"
        adapter_config_file = adapter_dir / "delta_mem_config.json"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"Missing adapter weights: {adapter_file}")
        if not adapter_config_file.is_file():
            raise FileNotFoundError(f"Missing adapter config: {adapter_config_file}")
        if not (runtime_root / "deltamem").is_dir():
            raise FileNotFoundError(f"Bundled deltamem package not found under {runtime_root}")
        adapter_sha256 = sha256_file(adapter_file)
        adapter_config_sha256 = sha256_file(adapter_config_file)
    elif args.adapter_dir is not None:
        raise ValueError("Base matrix prediction does not accept --adapter-dir")
    return RunConfig(
        variant=args.variant,
        model_path=args.model_path,
        model_identity=model_identity(args.model_path),
        adapter_dir=adapter_dir,
        adapter_sha256=adapter_sha256,
        adapter_config_sha256=adapter_config_sha256,
        runtime_root=runtime_root,
        runtime_git_identity=git_identity(runtime_root),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        python_bin=args.python_bin,
        predict_script=predict_script,
        predict_script_sha256=sha256_file(predict_script),
        score_script=score_script,
        score_script_sha256=sha256_file(score_script),
        overwrite_output=args.overwrite_output,
    )


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    matrix = load_input_matrix(
        args.generation_manifest,
        expected_seed=args.expected_seed,
    )
    if args.expected_rows_per_job < 0:
        raise ValueError("expected-rows-per-job must be >= 0")
    if args.expected_rows_per_job and matrix.rows_per_job != args.expected_rows_per_job:
        raise ValueError(
            f"Expected {args.expected_rows_per_job} rows per job, found {matrix.rows_per_job}"
        )
    output_root = args.output_root.expanduser().resolve()
    units = build_work_units(matrix, output_root, args.chunks_per_input)
    config = resolve_run_config(args)
    top_manifest, payload = prepare_run(matrix, units, output_root, devices, config)
    pending = tuple(unit for unit in units if output_status(unit, config) != "complete")
    try:
        if pending:
            run_work_queue(pending, devices, config)
        payload["work_units"] = [unit_record(unit, config) for unit in units]
        payload["coverage"] = verify_matrix_coverage(matrix, units, config)
        payload["output_rows"] = sum(record["observed_rows"] for record in payload["coverage"])
        payload["scores"] = score_matrix(matrix, units, output_root, config)
        payload["status"] = "complete"
        atomic_write_json(top_manifest, payload)
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["work_units"] = [failure_unit_record(unit, config) for unit in units]
        atomic_write_json(top_manifest, payload)
        raise
    print("RULER_MATRIX=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
