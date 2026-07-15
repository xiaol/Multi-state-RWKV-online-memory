#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


TASK_TYPES = {
    "niah_single_1": "all",
    "niah_single_2": "all",
    "niah_single_3": "all",
    "niah_multikey_1": "all",
    "niah_multikey_2": "all",
    "niah_multikey_3": "all",
    "niah_multivalue": "all",
    "niah_multiquery": "all",
    "vt": "all",
    "cwe": "all",
    "fwe": "all",
    "qa_1": "part",
    "qa_2": "part",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score RULER prediction JSONL files with the official substring metrics."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_prediction(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"[\x00-\x1f]", "\n", text).strip()


def sample_score(prediction: str, references: list[str], mode: str) -> float:
    matches = [1.0 if str(reference).lower() in prediction.lower() else 0.0 for reference in references]
    if not matches:
        return 0.0
    return max(matches) if mode == "part" else sum(matches) / len(matches)


def main() -> None:
    args = parse_args()
    grouped_files: dict[str, list[Path]] = {task: [] for task in TASK_TYPES}
    for path in sorted(args.prediction_dir.glob("*.jsonl")):
        name = path.stem
        for task in TASK_TYPES:
            if name == task or name.startswith(task + "-"):
                grouped_files[task].append(path)
                break

    missing = [task for task, paths in grouped_files.items() if not paths]
    if missing and not args.allow_partial:
        raise FileNotFoundError(f"Missing prediction files for: {missing}")
    if not any(grouped_files.values()):
        raise FileNotFoundError(f"No prediction JSONL files found in {args.prediction_dir}")

    task_results = {}
    for task, paths in grouped_files.items():
        if not paths:
            continue
        rows = []
        file_records = []
        for path in paths:
            file_rows = load_jsonl(path)
            rows.extend(file_rows)
            file_records.append(
                {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(file_rows)}
            )
        if not rows:
            raise ValueError(f"No prediction rows found for task {task}")
        seen_ordinals = set()
        scores = []
        null_count = 0
        elapsed_seconds = 0.0
        prompt_tokens = 0
        for row in rows:
            ordinal = row.get("sample_ordinal")
            if ordinal is not None:
                ordinal = int(ordinal)
                if ordinal in seen_ordinals:
                    raise ValueError(f"Duplicate sample_ordinal={ordinal} for task {task}")
                seen_ordinals.add(ordinal)
            prediction = clean_prediction(row.get("pred"))
            references = [str(value) for value in row.get("outputs", [])]
            if not prediction:
                null_count += 1
            scores.append(sample_score(prediction, references, TASK_TYPES[task]))
            elapsed_seconds += float(row.get("elapsed_seconds") or 0.0)
            prompt_tokens += int(row.get("prompt_tokens") or 0)
        official_score = round(sum(scores) / len(scores) * 100.0, 2)
        task_results[task] = {
            "score": official_score,
            "rows": len(rows),
            "null_predictions": null_count,
            "mean_prompt_tokens": prompt_tokens / len(rows),
            "total_elapsed_seconds": elapsed_seconds,
            "files": file_records,
        }

    macro_score = round(
        sum(result["score"] for result in task_results.values()) / len(task_results),
        2,
    )
    payload = {
        "variant": args.variant,
        "context_length": args.context_length,
        "prediction_dir": str(args.prediction_dir.resolve()),
        "macro_score": macro_score,
        "task_count": len(task_results),
        "complete_task_matrix": not missing,
        "missing_tasks": missing,
        "tasks": task_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task", "score", "rows", "null_predictions"))
        for task, result in task_results.items():
            writer.writerow((task, result["score"], result["rows"], result["null_predictions"]))
        writer.writerow(("macro", macro_score, sum(item["rows"] for item in task_results.values()), ""))
    print("RULER_SCORE=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
