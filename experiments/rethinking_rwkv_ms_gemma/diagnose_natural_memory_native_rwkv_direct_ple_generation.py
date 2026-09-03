#!/usr/bin/env python3
"""Diagnose native direct-PLE generation sensitivity on fixed development rows."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_rwkv_direct_ple_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_recurrent_routed_final as final,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_generation_diagnostic.v1"
DEFAULT_TRAINING_RESULT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_train_v10/result.json"
)
DEFAULT_CANDIDATE_ADAPTER = DEFAULT_TRAINING_RESULT.parent / "adapter"
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_rwkv_direct_ple_generation_diagnostic_v1"
)


def fixed_rows() -> Mapping[str, Mapping[str, Any]]:
    rows_by_task = evaluator.development.read_v2_rows()
    return {
        task: min(
            (row for row in rows_by_task[task] if int(row["prompt_variant"]) == 0),
            key=lambda row: int(row["source_ordinal"]),
        )
        for task in evaluator.TASKS
    }


def donor_for(
    task: str,
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = evaluator.development.read_v2_rows()[task]
    sources = tuple(
        final.as_source_row(task, value)
        for value in rows
        if int(value["prompt_variant"]) == 0
    )
    target = next(
        value for value in sources if value.source_ordinal == int(row["source_ordinal"])
    )
    donor_ordinal = final.choose_control_donor(target, sources).source_ordinal
    return next(
        value
        for value in rows
        if int(value["source_ordinal"]) == donor_ordinal
        and int(value["prompt_variant"]) == int(row["prompt_variant"])
    )


def pairwise_sensitivity(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values: dict[str, Any] = {}
    for condition in evaluator.CONDITIONS[1:]:
        raw_changed = 0
        parsed_changed = 0
        score_changed = 0
        tasks: dict[str, Any] = {}
        for task in evaluator.TASKS:
            correct = next(
                row
                for row in records
                if row["task"] == task
                and row["condition"] == "correct_recurrent_state"
            )
            control = next(
                row
                for row in records
                if row["task"] == task and row["condition"] == condition
            )
            task_raw_changed = correct["raw_generation"] != control["raw_generation"]
            task_parsed_changed = correct["prediction"] != control["prediction"]
            task_score_changed = correct["score"] != control["score"]
            raw_changed += int(task_raw_changed)
            parsed_changed += int(task_parsed_changed)
            score_changed += int(task_score_changed)
            tasks[task] = {
                "raw_generation_changed": task_raw_changed,
                "prediction_changed": task_parsed_changed,
                "score_changed": task_score_changed,
            }
        values[condition] = {
            "raw_generation_changed_rows": raw_changed,
            "prediction_changed_rows": parsed_changed,
            "score_changed_rows": score_changed,
            "tasks": tasks,
        }
    return values


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("HF_ENDPOINT") != evaluator.HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be exactly {evaluator.HF_MIRROR_ENDPOINT}"
        )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Diagnostic output must be fresh: {output}")
    output.mkdir(parents=True)
    training_result = evaluator.validate_training_result(
        args.training_result,
        args.candidate_adapter,
    )
    baseline_commit, baseline_records = evaluator.validate_baseline_commit(
        args.baseline_commit,
        base_model=args.base_model,
        v9_adapter=args.v9_adapter,
    )
    model, tokenizer, precision = evaluator.load_system(
        "direct_ple_candidate",
        base_model=args.base_model.expanduser().resolve(strict=True),
        v9_adapter=args.v9_adapter.expanduser().resolve(strict=True),
        candidate_adapter=args.candidate_adapter.expanduser().resolve(strict=True),
        candidate_result=training_result,
        device=args.device,
    )
    model.eval()
    selected = fixed_rows()
    module_names = tuple(name for name, _ in common.ordered_modules(model))
    records: list[Mapping[str, Any]] = []
    try:
        for task in evaluator.TASKS:
            row = selected[task]
            records.extend(
                evaluator.candidate_row_conditions(
                    model,
                    tokenizer,
                    row,
                    donor_for(task, row),
                    module_names=module_names,
                    device=args.device,
                )
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    raw_path = output / "records.jsonl"
    evaluator.write_jsonl(raw_path, records)
    selected_keys = {
        (task, int(row["line_index"])) for task, row in selected.items()
    }
    system_records = {
        system: [
            row
            for row in baseline_records[system]
            if (str(row["task"]), int(row["line_index"])) in selected_keys
        ]
        for system in evaluator.SYSTEMS[:2]
    }
    system_records["direct_ple_candidate"] = [
        row for row in records if row["condition"] == "correct_recurrent_state"
    ]
    condition_summaries = {
        condition: evaluator.summarize_records(
            [row for row in records if row["condition"] == condition]
        )
        for condition in evaluator.CONDITIONS
    }
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "diagnostic_complete_not_final_evaluation",
        "input_binding": {
            "training_result": str(args.training_result.expanduser().resolve(strict=True)),
            "training_result_sha256": evaluator.sha256_file(
                args.training_result.expanduser().resolve(strict=True)
            ),
            "training_result_receipt": training_result["receipt"]["payload_sha256"],
            "candidate_adapter": str(
                args.candidate_adapter.expanduser().resolve(strict=True)
            ),
            "baseline_commit": evaluator.baseline_commit_binding(
                args.baseline_commit,
                baseline_commit,
            ),
            "rows": {
                task: {
                    "line_index": int(row["line_index"]),
                    "source_ordinal": int(row["source_ordinal"]),
                    "prompt_variant": int(row["prompt_variant"]),
                    "row_sha256": row["row_sha256"],
                }
                for task, row in selected.items()
            },
            "conditions": list(evaluator.CONDITIONS),
            "candidate_precision": precision,
            "task_router": False,
            "template_matcher": False,
            "dual_pass_selector": False,
            "benchmark_specific_decoder": False,
            "final_rows_opened": False,
            "publisher_validation_opened": False,
            "publisher_test_opened": False,
            "runner_sha256": evaluator.sha256_file(Path(__file__)),
            "evaluator_sha256": evaluator.sha256_file(Path(evaluator.__file__)),
        },
        "summary": {
            "systems": {
                system: evaluator.summarize_records(system_rows)
                for system, system_rows in system_records.items()
            },
            "conditions": condition_summaries,
            "sensitivity": pairwise_sensitivity(records),
        },
        "raw_records": {
            "path": str(raw_path),
            "rows": len(records),
            "sha256": evaluator.sha256_file(raw_path),
        },
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": evaluator.canonical_sha256(value),
    }
    evaluator.write_json(output / "result.json", value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, default=DEFAULT_TRAINING_RESULT)
    parser.add_argument("--candidate-adapter", type=Path, default=DEFAULT_CANDIDATE_ADAPTER)
    parser.add_argument("--baseline-commit", type=Path, default=evaluator.DEFAULT_BASELINE_COMMIT)
    parser.add_argument("--base-model", type=Path, default=evaluator.DEFAULT_BASE_MODEL)
    parser.add_argument("--v9-adapter", type=Path, default=evaluator.DEFAULT_V9_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    value = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": value["status"],
                "receipt": value["receipt"]["payload_sha256"],
                "sensitivity": value["summary"]["sensitivity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
