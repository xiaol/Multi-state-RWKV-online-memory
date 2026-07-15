#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from task_config import (
    DEFAULT_EVAL_LENGTHS,
    DEFAULT_TRAIN_LENGTHS,
    DEFAULT_TRAIN_SEEDS,
    EVAL_TASKS,
    GEMMA4_CHAT_TEMPLATE,
    GEMMA4_CHAT_TEMPLATE_NAME,
    OFFICIAL_EVAL_SEED,
    TRAIN_TASKS,
    parse_csv_ints,
    parse_csv_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic NVIDIA RULER data with the local Gemma4 tokenizer."
    )
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("eval", "train"), required=True)
    parser.add_argument("--lengths", help="Comma-separated target context lengths")
    parser.add_argument("--tasks", help="Comma-separated task names")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seeds", help="Comma-separated seeds aligned with --lengths")
    parser.add_argument("--subset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_templates(path: Path) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("ruler_data_template", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RULER templates from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.Templates)


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            count += 1
    return count


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("transformers", "nltk", "wonderwords", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def tokenizer_file_records(path: Path) -> list[dict[str, str]]:
    records = []
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        candidate = path / name
        if candidate.is_file():
            records.append({"path": str(candidate.resolve()), "sha256": sha256_file(candidate)})
    if not records:
        raise FileNotFoundError(f"No tokenizer metadata files found under {path}")
    return records


def resolve_jobs(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...], int]:
    if args.mode == "train":
        lengths = parse_csv_ints(args.lengths) if args.lengths else DEFAULT_TRAIN_LENGTHS
        tasks = parse_csv_strings(args.tasks) if args.tasks else TRAIN_TASKS
        unknown = sorted(set(tasks) - set(TRAIN_TASKS))
        if unknown:
            raise ValueError(f"Training is restricted to the 11 non-QA tasks; invalid tasks: {unknown}")
        if args.seeds:
            seeds = parse_csv_ints(args.seeds)
        elif lengths == DEFAULT_TRAIN_LENGTHS:
            seeds = DEFAULT_TRAIN_SEEDS
        else:
            seeds = tuple(100000 + 100 * index for index in range(len(lengths)))
        num_samples = args.num_samples or 128
        if len(seeds) != len(lengths):
            raise ValueError("Training seeds must have one entry per context length")
        if len(set(seeds)) != len(seeds) or OFFICIAL_EVAL_SEED in seeds:
            raise ValueError("Training seeds must be unique and disjoint from official eval seed 42")
    else:
        lengths = parse_csv_ints(args.lengths) if args.lengths else DEFAULT_EVAL_LENGTHS
        tasks = parse_csv_strings(args.tasks) if args.tasks else EVAL_TASKS
        unknown = sorted(set(tasks) - set(EVAL_TASKS))
        if unknown:
            raise ValueError(f"Unknown RULER tasks: {unknown}")
        seeds = parse_csv_ints(args.seeds) if args.seeds else (OFFICIAL_EVAL_SEED,) * len(lengths)
        num_samples = args.num_samples or 500
        if len(seeds) != len(lengths):
            raise ValueError("Evaluation seeds must have one entry per context length")
    if num_samples <= 0:
        raise ValueError("num-samples must be positive")
    return lengths, tasks, seeds, num_samples


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    ruler_root = args.ruler_root.resolve()
    scripts_root = ruler_root / "scripts"
    prepare_script = scripts_root / "data" / "prepare.py"
    template_file = scripts_root / "data" / "template.py"
    if not prepare_script.is_file() or not template_file.is_file():
        raise FileNotFoundError(f"Not a classic NVIDIA RULER checkout: {ruler_root}")
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {args.tokenizer_path}")

    templates = load_templates(template_file)
    template_name = "base" if args.mode == "train" else GEMMA4_CHAT_TEMPLATE_NAME
    expected_template = "{task_template}" if args.mode == "train" else GEMMA4_CHAT_TEMPLATE
    if templates.get(template_name) != expected_template:
        raise ValueError(
            f"RULER template.py must define {template_name!r} as {expected_template!r}"
        )
    lengths, tasks, seeds, num_samples = resolve_jobs(args)
    subset = args.subset or ("train" if args.mode == "train" else "validation")
    args.output_root.mkdir(parents=True, exist_ok=True)

    corpus_file = ruler_root / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    if args.mode == "train" and any(task.startswith("niah_") for task in tasks):
        if not corpus_file.is_file():
            raise FileNotFoundError(f"Essay-backed RULER tasks require {corpus_file}")
        corpus_record = {"path": str(corpus_file.resolve()), "sha256": sha256_file(corpus_file)}
    else:
        corpus_record = None
    qa_records = []
    qa_files = {
        "qa_1": ruler_root / "scripts" / "data" / "synthetic" / "json" / "squad.json",
        "qa_2": ruler_root / "scripts" / "data" / "synthetic" / "json" / "hotpotqa.json",
    }
    for task, path in qa_files.items():
        if task not in tasks:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"RULER task {task} requires {path}")
        qa_records.append({"task": task, "path": str(path.resolve()), "sha256": sha256_file(path)})

    environment = os.environ.copy()
    python_bin = str(Path(sys.executable).parent)
    environment["PATH"] = python_bin + os.pathsep + environment.get("PATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    jobs = []
    for length_index, (length, seed_base) in enumerate(zip(lengths, seeds)):
        save_dir = args.output_root / str(length)
        for task_index, task in enumerate(tasks):
            seed = seed_base + task_index if args.mode == "train" else seed_base
            output_file = save_dir / task / f"{subset}.jsonl"
            sidecar_file = output_file.with_suffix(output_file.suffix + ".generation.json")
            expected_job = {
                "task": task,
                "context_length": length,
                "length_index": length_index,
                "task_index": task_index,
                "seed": seed,
                "rows": num_samples,
                "template_name": template_name,
                "subset": subset,
            }
            if args.overwrite:
                output_file.unlink(missing_ok=True)
                sidecar_file.unlink(missing_ok=True)
            elif output_file.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Refusing to reuse existing data without --resume or --overwrite: {output_file}"
                    )
                if not sidecar_file.is_file():
                    raise FileNotFoundError(f"Missing resume sidecar for {output_file}")
                sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
                for key, value in expected_job.items():
                    if sidecar.get(key) != value:
                        raise ValueError(f"Resume metadata mismatch for {output_file}: {key}")
                rows = count_jsonl(output_file)
                digest = sha256_file(output_file)
                if rows != num_samples or sidecar.get("sha256") != digest:
                    raise ValueError(f"Resume hash or row-count mismatch for {output_file}")
                jobs.append(
                    {
                        **expected_job,
                        "path": str(output_file.resolve()),
                        "sha256": digest,
                    }
                )
                continue
            command = (
                sys.executable,
                str(prepare_script),
                "--save_dir",
                str(save_dir),
                "--benchmark",
                "synthetic",
                "--task",
                task,
                "--subset",
                subset,
                "--tokenizer_path",
                str(args.tokenizer_path.resolve()),
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(length),
                "--model_template_type",
                template_name,
                "--num_samples",
                str(num_samples),
                "--random_seed",
                str(seed),
            )
            subprocess.run(command, cwd=scripts_root, env=environment, check=True)
            if not output_file.is_file():
                raise FileNotFoundError(f"RULER did not create {output_file}")
            rows = count_jsonl(output_file)
            if rows != num_samples:
                raise ValueError(f"Expected {num_samples} rows in {output_file}, found {rows}")
            job = {
                **expected_job,
                "path": str(output_file.resolve()),
                "sha256": sha256_file(output_file),
            }
            atomic_write(sidecar_file, json.dumps(job, indent=2) + "\n")
            jobs.append(job)

    manifest = {
        "mode": args.mode,
        "ruler_root": str(ruler_root),
        "ruler_revision": git_revision(ruler_root),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "tokenizer_files": tokenizer_file_records(args.tokenizer_path),
        "dependency_versions": dependency_versions(),
        "paul_graham_corpus": corpus_record,
        "qa_corpora": qa_records,
        "template_name": template_name,
        "template": expected_template,
        "subset": subset,
        "lengths": list(lengths),
        "tasks": list(tasks),
        "seed_bases": list(seeds),
        "seeds": [job["seed"] for job in jobs],
        "num_samples_per_task_length": num_samples,
        "total_rows": len(jobs) * num_samples,
        "jobs": jobs,
    }
    manifest_path = args.output_root / "generation_manifest.json"
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print("RULER_GENERATION=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
