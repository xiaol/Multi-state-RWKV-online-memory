#!/usr/bin/env python3
"""Score corrected write-then-read native evaluations with format recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import analyze_novel_agent_eval as recovery
from experiments.rethinking_rwkv_ms_gemma import run_novel_agent_eval as evaluator


SCHEMA = "rwkv_ms_natural_memory_native_corrected_analysis.v1"
TASK_KINDS = {
    "attribution-v3.2": "attribution",
    "narrative-v3.2": "narrative",
    "scene-v4-current": "scene",
}


def canonical_sha256(value: Any) -> str:
    return evaluator.sha256_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected an object at {path}")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    rows = evaluator.read_records(path)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Evaluation records must be objects: {path}")
    return rows


def _dataset_samples(
    path: Path,
    *,
    task_kind: str,
    expected_rows: int,
) -> list[recovery.DatasetSample]:
    samples: list[recovery.DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid publisher messages in {path}")
            user_content = str(messages[-2].get("content", ""))
            gold = recovery.extract_json(str(messages[-1].get("content", "")))
            if gold is None:
                raise ValueError(f"Invalid publisher gold JSON in {path}")
            candidates: tuple[str, ...] = ()
            paragraph_count: int | None = None
            if task_kind == "attribution":
                candidates = recovery.parse_candidates(user_content)
            elif task_kind == "narrative":
                recovery.gold_label_map(gold)
            else:
                paragraph_count = recovery.parse_paragraph_count(user_content)
                recovery.strict_gold_boundaries(gold)
            samples.append(
                recovery.DatasetSample(
                    line_index=len(samples),
                    row_sha256=evaluator.sha256_text(raw_line.rstrip("\n")),
                    gold=gold,
                    candidates=candidates,
                    paragraph_count=paragraph_count,
                )
            )
            if len(samples) == expected_rows:
                break
    if len(samples) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in {path}, found {len(samples)}"
        )
    return samples


def analyze_evaluation_root(eval_root: Path) -> Mapping[str, Any]:
    resolved_root = eval_root.expanduser().resolve(strict=True)
    tasks: dict[str, Any] = {}
    total_raw_disagreements = 0
    total_rows = 0
    for task_name, task_kind in TASK_KINDS.items():
        task_dir = resolved_root / task_name
        consolidated = not task_dir.is_dir()
        if consolidated:
            task_dir = resolved_root
        manifest_path = task_dir / "manifest.json"
        summary_path = task_dir / "summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        strict_summary = _read_json(summary_path)
        payload = manifest.get("fingerprint_payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Manifest fingerprint payload is missing: {task_dir}")
        if payload.get("online_memory_protocol") != "write_then_read":
            raise ValueError(f"Task did not use write_then_read: {task_name}")
        datasets = payload.get("datasets")
        if not isinstance(datasets, Mapping) or (
            task_name not in datasets
            or (not consolidated and set(datasets) != {task_name})
        ):
            raise ValueError(f"Task manifest dataset selection differs: {task_name}")
        dataset = datasets[task_name]
        if not isinstance(dataset, Mapping):
            raise ValueError(f"Task dataset binding is invalid: {task_name}")
        row_count = int(dataset["selected_rows"])
        samples = _dataset_samples(
            Path(str(dataset["path"])),
            task_kind=task_kind,
            expected_rows=row_count,
        )
        spec = recovery.TaskSpec(
            name=task_name,
            relative_path=str(dataset["path"]),
            kind=task_kind,
            expected_rows=row_count,
        )
        condition_rows: dict[str, list[dict[str, Any]]] = {}
        condition_summaries: dict[str, Any] = {}
        recovered_predictions: dict[str, list[Any]] = {}
        for condition in ("base", "normal"):
            rows = _read_records(task_dir / f"{condition}.jsonl")
            if consolidated:
                rows = [row for row in rows if row.get("task") == task_name]
            rows_by_index = {int(row["line_index"]): row for row in rows}
            if set(rows_by_index) != set(range(row_count)):
                raise ValueError(f"Incomplete {condition} rows for {task_name}")
            ordered = [rows_by_index[index] for index in range(row_count)]
            if any(
                row.get("fingerprint") != manifest.get("fingerprint")
                or row.get("condition") != condition
                for row in ordered
            ):
                raise ValueError(f"{condition} record binding differs for {task_name}")
            summary, predictions, _ = recovery.analyze_task(spec, ordered, samples)
            condition_rows[condition] = ordered
            condition_summaries[condition] = summary
            recovered_predictions[condition] = predictions
        raw_disagreements = sum(
            base.get("raw_generation") != normal.get("raw_generation")
            for base, normal in zip(
                condition_rows["base"],
                condition_rows["normal"],
                strict=True,
            )
        )
        recovered_disagreements = sum(
            base != normal
            for base, normal in zip(
                recovered_predictions["base"],
                recovered_predictions["normal"],
                strict=True,
            )
        )
        base_metric = float(condition_summaries["base"]["primary_metric"])
        normal_metric = float(condition_summaries["normal"]["primary_metric"])
        strict_conditions = strict_summary.get("conditions", {})
        base_elapsed = float(strict_conditions["base"][task_name]["elapsed_seconds"])
        normal_elapsed = float(strict_conditions["normal"][task_name]["elapsed_seconds"])
        tasks[task_name] = {
            "kind": task_kind,
            "rows": row_count,
            "base": condition_summaries["base"],
            "normal": condition_summaries["normal"],
            "normal_minus_base": normal_metric - base_metric,
            "raw_generation_disagreements": raw_disagreements,
            "recovered_prediction_disagreements": recovered_disagreements,
            "normal_to_base_elapsed_ratio": normal_elapsed / base_elapsed,
            "artifacts": {
                "manifest_sha256": evaluator.sha256_file(manifest_path),
                "strict_summary_sha256": evaluator.sha256_file(summary_path),
                "base_records_sha256": evaluator.sha256_file(task_dir / "base.jsonl"),
                "normal_records_sha256": evaluator.sha256_file(task_dir / "normal.jsonl"),
            },
        }
        total_raw_disagreements += raw_disagreements
        total_rows += row_count
    if set(tasks) != set(TASK_KINDS):
        raise ValueError("Corrected analysis requires all three native tasks")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "evaluation_root": str(resolved_root),
        "online_memory_protocol": "write_then_read",
        "tasks": tasks,
        "totals": {
            "rows": total_rows,
            "raw_generation_disagreements": total_raw_disagreements,
            "raw_generation_disagreement_fraction": (
                total_raw_disagreements / total_rows
            ),
        },
    }
    result["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_analysis_without_receipt",
        "payload_sha256": canonical_sha256(result),
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze_evaluation_root(args.eval_root)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Corrected analysis output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
