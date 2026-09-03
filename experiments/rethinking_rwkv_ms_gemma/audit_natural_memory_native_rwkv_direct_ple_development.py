#!/usr/bin/env python3
"""Apply stricter paraphrase-level checks to a signed direct-PLE development run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_rwkv_direct_ple_development as evaluator,
)


SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development_audit.v1"
EXPECTED_RESULT_SCHEMA = "rwkv_ms_natural_memory_native_rwkv_direct_ple_development.v1"
SYSTEMS = evaluator.SYSTEMS
CONDITIONS = evaluator.CONDITIONS
TASKS = evaluator.TASKS
PROMPT_VARIANTS = tuple(range(4))


def read_signed_result(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    unsigned = dict(value)
    unsigned.pop("receipt", None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != evaluator.canonical_sha256(unsigned)
        or value.get("schema") != EXPECTED_RESULT_SCHEMA
    ):
        raise ValueError("Direct-PLE development result signature or schema differs")
    return value


def read_bound_rows(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(metadata["path"])).expanduser().resolve(strict=True)
    rows = evaluator.read_jsonl(path)
    if (
        len(rows) != int(metadata["rows"])
        or evaluator.sha256_file(path) != metadata["sha256"]
    ):
        raise ValueError(f"Bound prediction file differs: {path}")
    return rows


def variant_metric(summary: Mapping[str, Any], task: str, variant: int) -> float:
    return float(
        summary["by_task_prompt_variant"][f"{task}:{variant}"][
            evaluator.TASK_METRICS[task]
        ]
    )


def schema_rate(summary: Mapping[str, Any], task: str, variant: int) -> float:
    metrics = summary["by_task_prompt_variant"][f"{task}:{variant}"]
    return float(metrics["strict_schema_valid"]) / float(metrics["rows"])


def verify_system_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    rows_per_task: Mapping[str, int],
) -> bool:
    expected_rows = sum(rows_per_task.values())
    expected_groups = {
        (task, variant): rows_per_task[task] // len(PROMPT_VARIANTS)
        for task in TASKS
        for variant in PROMPT_VARIANTS
    }
    keys = {
        (str(row["task"]), int(row["line_index"]), int(row["prompt_variant"]))
        for row in records
    }
    groups = Counter(
        (str(row["task"]), int(row["prompt_variant"])) for row in records
    )
    return (
        all(rows_per_task[task] % len(PROMPT_VARIANTS) == 0 for task in TASKS)
        and len(records) == expected_rows
        and len(keys) == expected_rows
        and groups == expected_groups
    )


def verify_control_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    rows_per_task: Mapping[str, int],
) -> bool:
    expected_rows = sum(rows_per_task.values())
    expected_groups = {
        (task, variant, condition): rows_per_task[task] // len(PROMPT_VARIANTS)
        for task in TASKS
        for variant in PROMPT_VARIANTS
        for condition in CONDITIONS
    }
    keys = {
        (
            str(row["task"]),
            int(row["line_index"]),
            int(row["prompt_variant"]),
            str(row["condition"]),
        )
        for row in records
    }
    groups = Counter(
        (
            str(row["task"]),
            int(row["prompt_variant"]),
            str(row["condition"]),
        )
        for row in records
    )
    return (
        all(rows_per_task[task] % len(PROMPT_VARIANTS) == 0 for task in TASKS)
        and len(records) == expected_rows * len(CONDITIONS)
        and len(keys) == len(records)
        and groups == expected_groups
        and all(row.get("projected_carrier_fixed") is True for row in records)
        and all(row.get("projected_carrier_byte_identical") is True for row in records)
        and all(
            isinstance(row.get("zero_vs_projected_output_identical"), bool)
            for row in records
        )
        and all(row.get("system") == "direct_ple_candidate" for row in records)
        and all(
            int(row["donor_source_ordinal"]) != int(row["source_ordinal"])
            and row["donor_row_sha256"] != row["row_sha256"]
            for row in records
        )
        and all(row.get("benchmark_time_task_router") is False for row in records)
        and all(row.get("benchmark_time_template_matcher") is False for row in records)
        and all(row.get("benchmark_time_dual_pass_selector") is False for row in records)
        and all(row.get("benchmark_specific_decoder") is False for row in records)
    )


def candidate_matches_correct_controls(
    candidate: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> bool:
    def key(row: Mapping[str, Any]) -> tuple[str, int, int]:
        return (
            str(row["task"]),
            int(row["line_index"]),
            int(row["prompt_variant"]),
        )

    candidate_rows = {
        key(row): evaluator.canonical_sha256(row)
        for row in candidate
    }
    correct_rows = {
        key(row): evaluator.canonical_sha256(row)
        for row in controls
        if row["condition"] == "correct_recurrent_state"
    }
    return len(candidate_rows) == len(candidate) and candidate_rows == correct_rows


def expanded_checks(
    system_summaries: Mapping[str, Mapping[str, Any]],
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    candidate = system_summaries["direct_ple_candidate"]
    base = system_summaries["frozen_gemma_base"]
    projected = system_summaries["v9_projected_slot_baseline"]
    candidate_variant_nonregression = {
        f"{task}:{variant}": variant_metric(candidate, task, variant)
        >= max(
            variant_metric(base, task, variant),
            variant_metric(projected, task, variant),
        )
        for task in TASKS
        for variant in PROMPT_VARIANTS
    }
    schema_variant_nonregression = {
        f"{task}:{variant}": schema_rate(candidate, task, variant)
        >= max(
            schema_rate(base, task, variant),
            schema_rate(projected, task, variant),
        )
        for task in TASKS
        for variant in PROMPT_VARIANTS
    }
    correct = condition_summaries["correct_recurrent_state"]
    causal_variant_nonregression = {
        condition: {
            f"{task}:{variant}": variant_metric(correct, task, variant)
            >= variant_metric(condition_summaries[condition], task, variant)
            for task in TASKS
            for variant in PROMPT_VARIANTS
        }
        for condition in CONDITIONS[1:]
    }
    causal_prompt_variant_strict = {
        condition: {
            str(variant): sum(
                variant_metric(correct, task, variant) for task in TASKS
            )
            / len(TASKS)
            > sum(
                variant_metric(condition_summaries[condition], task, variant)
                for task in TASKS
            )
            / len(TASKS)
            for variant in PROMPT_VARIANTS
        }
        for condition in CONDITIONS[1:]
    }
    strict_variant_improvement = any(
        variant_metric(candidate, task, variant) > variant_metric(base, task, variant)
        and variant_metric(candidate, task, variant)
        > variant_metric(projected, task, variant)
        for task in TASKS
        for variant in PROMPT_VARIANTS
    )
    passed = bool(
        all(candidate_variant_nonregression.values())
        and all(schema_variant_nonregression.values())
        and strict_variant_improvement
        and all(
            passed
            for condition in causal_variant_nonregression.values()
            for passed in condition.values()
        )
        and all(
            passed
            for condition in causal_prompt_variant_strict.values()
            for passed in condition.values()
        )
    )
    return {
        "candidate_metric_nonregression_every_task_prompt_variant": (
            candidate_variant_nonregression
        ),
        "candidate_schema_nonregression_every_task_prompt_variant": (
            schema_variant_nonregression
        ),
        "candidate_strictly_better_than_both_on_one_task_prompt_variant": (
            strict_variant_improvement
        ),
        "correct_state_nonworse_every_control_task_prompt_variant": (
            causal_variant_nonregression
        ),
        "correct_state_strict_every_control_prompt_variant_aggregate": (
            causal_prompt_variant_strict
        ),
        "passed": passed,
    }


def audit(result_path: Path) -> Mapping[str, Any]:
    resolved_result = result_path.expanduser().resolve(strict=True)
    source = read_signed_result(resolved_result)
    input_binding = source.get("input_binding", {})
    rows_per_task = {
        str(task): int(count)
        for task, count in input_binding.get("rows_per_task", {}).items()
    }
    expected_rows = sum(rows_per_task.values())
    if (
        rows_per_task.keys() != set(TASKS)
        or expected_rows <= 0
        or input_binding.get("task_router") is not False
        or input_binding.get("template_matcher") is not False
        or input_binding.get("dual_pass_selector") is not False
        or input_binding.get("benchmark_specific_decoder") is not False
        or input_binding.get("projected_carrier_fixed") is not True
        or input_binding.get("final_rows_opened") is not False
        or source.get("final_rows_opened") is not False
        or source.get("publisher_validation_opened") is not False
        or source.get("publisher_test_opened") is not False
    ):
        raise ValueError("Direct-PLE development input contract differs")
    runner_hash_matches = input_binding.get("runner_sha256") == evaluator.sha256_file(
        Path(evaluator.__file__).resolve(strict=True)
    )
    baseline_binding = input_binding.get("baseline_commit", {})
    baseline_commit, committed_baselines = evaluator.validate_baseline_commit(
        Path(str(baseline_binding.get("path", ""))),
        base_model=Path(str(input_binding.get("base_model", ""))),
        v9_adapter=Path(str(input_binding.get("v9_adapter", ""))),
    )
    baseline_binding_matches = evaluator.canonical_sha256(
        baseline_binding
    ) == evaluator.canonical_sha256(
        evaluator.baseline_commit_binding(
            Path(str(baseline_binding["path"])),
            baseline_commit,
        )
    )
    system_records = {
        system: read_bound_rows(source["raw_prediction_files"][system])
        for system in SYSTEMS
    }
    control_records = read_bound_rows(source["control_file"])
    system_rows_valid = {
        system: verify_system_rows(records, rows_per_task=rows_per_task)
        for system, records in system_records.items()
    }
    control_rows_valid = verify_control_rows(
        control_records,
        rows_per_task=rows_per_task,
    )
    candidate_control_match = candidate_matches_correct_controls(
        system_records["direct_ple_candidate"],
        control_records,
    )
    baseline_rows_match = {
        system: evaluator.canonical_sha256(system_records[system])
        == evaluator.canonical_sha256(committed_baselines[system])
        for system in SYSTEMS[:2]
    }
    system_summaries = {
        system: evaluator.summarize_records(records)
        for system, records in system_records.items()
    }
    condition_summaries = {
        condition: evaluator.summarize_records(
            [row for row in control_records if row["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    base_promotion = evaluator.promotion(system_summaries, condition_summaries)
    embedded_summary_matches = evaluator.canonical_sha256(source["summary"]) == evaluator.canonical_sha256(
        {
            "systems": system_summaries,
            "conditions": condition_summaries,
            "promotion": base_promotion,
        }
    )
    expanded = expanded_checks(system_summaries, condition_summaries)
    passed = bool(
        source.get("passed") is True
        and base_promotion["passed"] is True
        and expanded["passed"] is True
        and runner_hash_matches
        and baseline_binding_matches
        and all(baseline_rows_match.values())
        and embedded_summary_matches
        and all(system_rows_valid.values())
        and control_rows_valid
        and candidate_control_match
    )
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "expanded_development_passed_final_preregistration_authorized"
            if passed
            else "expanded_development_failed_final_blocked"
        ),
        "passed": passed,
        "source": {
            "path": str(resolved_result),
            "sha256": evaluator.sha256_file(resolved_result),
            "receipt": source["receipt"]["payload_sha256"],
            "runner_hash_matches": runner_hash_matches,
            "baseline_commit_binding_matches": baseline_binding_matches,
            "baseline_rows_match": baseline_rows_match,
            "embedded_summary_matches": embedded_summary_matches,
        },
        "row_integrity": {
            "expected_rows_per_system": expected_rows,
            "systems": system_rows_valid,
            "controls": control_rows_valid,
            "candidate_matches_correct_controls": candidate_control_match,
        },
        "base_promotion": base_promotion,
        "expanded_paraphrase_checks": expanded,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "final_evaluation_authorized": passed,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
        "runner_sha256": evaluator.sha256_file(Path(__file__).resolve(strict=True)),
    }
    value["receipt"] = {
        "algorithm": "sha256",
        "payload_scope": "canonical_result_without_receipt",
        "payload_sha256": evaluator.canonical_sha256(value),
    }
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Audit output must be fresh: {output}")
    value = audit(args.development_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "passed": value["passed"], "receipt": value["receipt"]["payload_sha256"]}, sort_keys=True))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
