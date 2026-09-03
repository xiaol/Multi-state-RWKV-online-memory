#!/usr/bin/env python3
"""Evaluate a stage-9 candidate on the independent development-v2 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from torch.distributed.elastic.multiprocessing.errors import record

from experiments.rethinking_rwkv_ms_gemma import evaluate_natural_memory_native_recurrent_routed_posttrain_development as evaluator
from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common


V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
HYBRID_MODE = "recurrent_routed_query_value"
DEV_COUNTS = {"attribution": 8, "narrative": 32, "scene": 32}
SLOT_SHUFFLE_EXPECTATION = "negative_control"
SLOT_SHUFFLE_INVARIANCE_ATOL = 1e-8
ORIGINAL_VALIDATE_SPLIT = common.validate_split_artifacts
ORIGINAL_LOAD_ROWS = common.load_open_rows


def v2_manifest() -> Mapping[str, Any]:
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if not isinstance(receipt, Mapping) or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest receipt differs")
    if common.canonical_sha256(value) != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest payload differs")
    value["receipt"] = receipt
    return value


def load_v2_rows(split: str, *, manifest: Mapping[str, Any]) -> dict[str, tuple[common.SourceRow, ...]]:
    if split != "development":
        return ORIGINAL_LOAD_ROWS(split, manifest=manifest)
    source_rows = ORIGINAL_LOAD_ROWS("train", manifest=manifest)
    rows_by_task = {}
    selected = v2_manifest()["development_source_ordinals"]
    for task in common.TASKS:
        lookup = {row.source_ordinal: row for row in source_rows[task]}
        rows_by_task[task] = tuple(lookup[ordinal] for ordinal in selected[task])
    return rows_by_task


def choose_hard_donor(target: common.SourceRow, rows: Sequence[common.SourceRow]) -> common.SourceRow:
    target_user = str(json.loads(target.raw_line)["messages"][1]["content"])
    candidates = [
        row
        for row in rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"Development-v2 row has no different-answer donor: {target}")
    return max(
        candidates,
        key=lambda row: (
            __import__("difflib").SequenceMatcher(
                None,
                target_user,
                str(json.loads(row.raw_line)["messages"][1]["content"]),
            ).ratio(),
            -abs(row.user_characters - target.user_characters),
            row.row_sha256,
        ),
    )


def build_v2_schedule(rows_by_task: Mapping[str, Sequence[common.SourceRow]]):
    schedule = []
    payload = []
    for task in common.TASKS:
        rows = sorted(rows_by_task[task], key=lambda row: row.source_ordinal)
        if len(rows) != DEV_COUNTS[task]:
            raise ValueError(f"Development-v2 row count differs for {task}")
        for target in rows:
            donor = choose_hard_donor(target, rows)
            for variant in range(4):
                schedule.append((target, donor, variant))
                payload.append({
                    "task": task,
                    "source_ordinal": target.source_ordinal,
                    "source_row_sha256": target.row_sha256,
                    "donor_source_ordinal": donor.source_ordinal,
                    "donor_row_sha256": donor.row_sha256,
                    "prompt_variant": variant,
                })
    expected = sum(DEV_COUNTS.values()) * 4
    if len(schedule) != expected:
        raise RuntimeError("Development-v2 schedule size differs")
    return tuple(schedule), payload


def summarize_v2(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ordered = sorted(rows, key=lambda row: (common.TASKS.index(str(row["task"])), int(row["source_ordinal"]), int(row["prompt_variant"])))
    overall = evaluator.mean_metrics(ordered)
    by_task = {task: evaluator.mean_metrics([row for row in ordered if row["task"] == task]) for task in common.TASKS}
    by_variant = {
        f"{task}:{variant}": evaluator.mean_metrics([
            row for row in ordered if row["task"] == task and row["prompt_variant"] == variant
        ])
        for task in common.TASKS for variant in range(4)
    }
    controls = common.CONDITIONS[1:]
    causal_controls = tuple(
        condition
        for condition in controls
        if condition != "slot_shuffled_recurrent_state"
        or SLOT_SHUFFLE_EXPECTATION == "negative_control"
    )
    overall_pass = all(overall["mean_control_minus_correct_ce"][condition] > 0.0 for condition in causal_controls)
    per_task_pass = all(by_task[task]["mean_control_minus_correct_ce"][condition] > 0.0 for task in common.TASKS for condition in causal_controls)
    variant_donor_pass = all(metrics["mean_control_minus_correct_ce"]["matched_donor_recurrent_state"] > 0.0 for metrics in by_variant.values())
    slot_shuffle_invariance_pass = bool(
        SLOT_SHUFFLE_EXPECTATION != "invariance"
        or (
            abs(overall["mean_control_minus_correct_ce"]["slot_shuffled_recurrent_state"])
            <= SLOT_SHUFFLE_INVARIANCE_ATOL
            and all(
                abs(by_task[task]["mean_control_minus_correct_ce"]["slot_shuffled_recurrent_state"])
                <= SLOT_SHUFFLE_INVARIANCE_ATOL
                for task in common.TASKS
            )
            and all(
                abs(metrics["mean_control_minus_correct_ce"]["slot_shuffled_recurrent_state"])
                <= SLOT_SHUFFLE_INVARIANCE_ATOL
                for metrics in by_variant.values()
            )
        )
    )
    carrier_pass = all(row["projected_carrier_fixed"] is True for row in ordered)
    row_count_pass = len(ordered) == sum(DEV_COUNTS.values()) * 4 and all(by_task[task]["rows"] == DEV_COUNTS[task] * 4 for task in common.TASKS) and all(metrics["rows"] == DEV_COUNTS[task] for task in common.TASKS for metrics in [by_variant[f"{task}:{variant}"] for variant in range(4)])
    passed = bool(overall_pass and per_task_pass and variant_donor_pass and slot_shuffle_invariance_pass and carrier_pass and row_count_pass)
    return {
        "rows": len(ordered),
        "overall": overall,
        "by_task": by_task,
        "by_task_prompt_variant": by_variant,
        "causal_criteria": {
            "overall_correct_over_all_controls": overall_pass,
            "per_task_correct_over_all_controls": per_task_pass,
            "all_task_prompt_variants_correct_over_donor": variant_donor_pass,
            "slot_shuffle_expectation": SLOT_SHUFFLE_EXPECTATION,
            "slot_shuffle_invariance": slot_shuffle_invariance_pass,
            "projected_carriers_fixed": carrier_pass,
            "row_counts_exact": row_count_pass,
        },
        "passed": passed,
    }


def fake_split_artifacts():
    manifest, original_receipt = ORIGINAL_VALIDATE_SPLIT()
    value = v2_manifest()
    receipt = {
        "schema": value["schema"],
        "manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "materialized_splits": ["development_v2"],
        "final_files_written": [],
        "files": value["development_files"],
    }
    return manifest, receipt


def configure() -> None:
    common.HYBRID_MODE = HYBRID_MODE
    common.validate_split_artifacts = fake_split_artifacts
    common.load_open_rows = load_v2_rows
    evaluator.build_schedule = build_v2_schedule
    evaluator.summarize = summarize_v2
    evaluator.EXPECTED_ROWS = sum(DEV_COUNTS.values()) * 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-result-receipt", required=True)
    parser.add_argument(
        "--training-status",
        default="stage9_training_complete_development_v2_evaluation_authorized",
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage9_protocol_v1.json",
    )
    parser.add_argument(
        "--protocol-receipt",
        default="f651c0450abf8c09123fb8ab745dbe0796cd290d5ea80006f8d497bbb5116391",
    )
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--hybrid-gain", type=float, default=0.125)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common.HYBRID_GAIN = args.hybrid_gain
    configure()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Development-v2 evaluation requires four ranks")
    try:
        result = evaluator.run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=args.training_result_receipt,
            training_status=args.training_status,
            protocol_file=args.protocol_file.expanduser().resolve(strict=True),
            protocol_receipt=args.protocol_receipt,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps({"rank": context.process_rank, "status": result["status"], "passed": result["passed"], "result_receipt": result.get("receipt", {}).get("payload_sha256") if context.is_primary else None}, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
