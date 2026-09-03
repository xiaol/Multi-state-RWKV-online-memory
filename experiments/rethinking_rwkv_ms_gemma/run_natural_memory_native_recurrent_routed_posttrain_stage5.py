#!/usr/bin/env python3
"""Run the low-rate broad-coverage recurrent-routing continuation stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain_stage3 as stage3,
)


runner = stage3.runner
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage5_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "66a7b0190dd6d2245612162dd2ca85a60e70e139d583dee1948724382e1caa36"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = "7af3769fa34631329a54fb8caf44797a3a5598344e104680b6aa2cb108339248"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage4_development_v1"
DEVELOPMENT_RESULT_RECEIPT = "85d1807bd747c45de9abf4a99f4bddc7119ec5a8372a8a5bdee9ffe5f359b1e9"
STAGE2_DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_development_v1"
STAGE2_DEVELOPMENT_RECEIPT = "88129262892d28795b23752d44289133c2f5416245847d944af17c9b4853a47a"
STAGE3_DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage3_development_v1"
STAGE3_DEVELOPMENT_RECEIPT = "8abc7c1954f85352b1bdd763d6c794938cef423768a37df785ed661c917a62c4"
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 0.1
MARGIN = 0.02
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 0.75,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}
TASK_COUNTS_PER_STEP = {"attribution": 1, "narrative": 5, "scene": 2}


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(
        PREDECESSOR_ROOT / "result.json",
        PREDECESSOR_RESULT_RECEIPT,
    )
    stage2_development = common.validate_signed_json(
        STAGE2_DEVELOPMENT_ROOT / "result.json",
        STAGE2_DEVELOPMENT_RECEIPT,
    )
    stage3_development = common.validate_signed_json(
        STAGE3_DEVELOPMENT_ROOT / "result.json",
        STAGE3_DEVELOPMENT_RECEIPT,
    )
    stage4_development = common.validate_signed_json(
        DEVELOPMENT_ROOT / "result.json",
        DEVELOPMENT_RESULT_RECEIPT,
    )
    if (
        predecessor.get("status")
        != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or any(
            result.get("status") != "development_failed_final_evaluation_blocked"
            or result.get("passed") is not False
            or result.get("final_rows_opened") is not False
            for result in (
                stage2_development,
                stage3_development,
                stage4_development,
            )
        )
        or stage2_development.get("summary", {}).get("gates", {}).get(
            "overall_correct_over_all_controls"
        )
        is not True
        or stage2_development.get("summary", {}).get("gates", {}).get(
            "per_task_correct_over_all_controls"
        )
        is not True
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt")
        != PREDECESSOR_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json")
        != PREDECESSOR_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Stage-5 recurrent-routing lineage differs")
    return protocol, predecessor


def stage5_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    prior_schedule, prior_payload = common.build_training_schedule(
        rows_by_task,
        updates=96,
    )
    used_ordinals = {
        task: {
            row.target.source_ordinal
            for row in prior_schedule
            if row.target.task == task
        }
        for task in common.TASKS
    }
    remaining = {
        task: sorted(
            [
                row
                for row in rows_by_task[task]
                if row.source_ordinal not in used_ordinals[task]
            ],
            key=lambda row: (
                hashlib.sha256(
                    (
                        "rwkv-ms-recurrent-routed-train-v1:"
                        + task
                        + ":"
                        + row.row_sha256
                    ).encode("utf-8")
                ).hexdigest(),
                row.source_ordinal,
            ),
        )
        for task in common.TASKS
    }
    cursors = {task: 0 for task in common.TASKS}
    variant_cursors = {task: 0 for task in common.TASKS}
    schedule = []
    payload = []
    for step in range(1, TRAIN_UPDATES + 1):
        step_rows = []
        for task, count in TASK_COUNTS_PER_STEP.items():
            start = cursors[task]
            selected = remaining[task][start : start + count]
            if len(selected) != count:
                raise RuntimeError(f"Stage-5 has insufficient untouched {task} rows")
            step_rows.extend(selected)
            cursors[task] += count
        step_rows.sort(
            key=lambda row: hashlib.sha256(
                f"rwkv-ms-recurrent-routed-stage5:{step}:{row.row_sha256}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        step_payload = []
        for position, target in enumerate(step_rows):
            donor = stage3.choose_donor(target, rows_by_task[target.task])
            variant = variant_cursors[target.task] % 4
            variant_cursors[target.task] += 1
            schedule.append(
                common.ScheduledRow(
                    step=step,
                    position=position,
                    target=target,
                    donor=donor,
                    prompt_variant=variant,
                )
            )
            step_payload.append(
                {
                    "position": position,
                    "task": target.task,
                    "source_ordinal": target.source_ordinal,
                    "source_row_sha256": target.row_sha256,
                    "donor_source_ordinal": donor.source_ordinal,
                    "donor_row_sha256": donor.row_sha256,
                    "prompt_variant": variant,
                }
            )
        payload.append(
            {
                "step": step,
                "source_step": 96 + step,
                "rows": step_payload,
                "payload_sha256": common.canonical_sha256(step_payload),
            }
        )
    selected_ordinals = {
        task: {
            row.target.source_ordinal
            for row in schedule
            if row.target.task == task
        }
        for task in common.TASKS
    }
    expected_task_rows = {"attribution": 20, "narrative": 100, "scene": 40}
    expected_task_variant_rows = {
        "attribution": 5,
        "narrative": 25,
        "scene": 10,
    }
    if (
        len(schedule) != TRAIN_UPDATES * 8
        or any(used_ordinals[task] & selected_ordinals[task] for task in common.TASKS)
        or cursors != expected_task_rows
        or any(
            sum(
                row.target.task == task and row.prompt_variant == variant
                for row in schedule
            )
            != expected_task_variant_rows[task]
            for task in common.TASKS
            for variant in range(4)
        )
    ):
        raise RuntimeError("Stage-5 untouched task-balanced schedule differs")
    lineage_hash = common.canonical_sha256(
        {"prior_96_steps": prior_payload, "stage5_steps": payload}
    )
    return tuple(schedule), payload, lineage_hash


def configure_runner() -> None:
    stage3.configure_runner()
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage5.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage5_input.v1"
    runner.PREFLIGHT_STATUS = "stage5_preflight_passed"
    runner.TRAINING_STATUS = "stage5_training_complete_development_evaluation_authorized"
    runner.FAILURE_STATUS = "stage5_training_failed_development_evaluation_blocked"
    runner.RUNNER_FILE = Path(__file__)
    runner.PROTOCOL = PROTOCOL
    runner.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    runner.STAGE1_ROOT = PREDECESSOR_ROOT
    runner.STAGE1_RESULT_RECEIPT = PREDECESSOR_RESULT_RECEIPT
    runner.STAGE1_ADAPTER_WEIGHTS_SHA256 = PREDECESSOR_ADAPTER_WEIGHTS_SHA256
    runner.STAGE1_ADAPTER_CONFIG_SHA256 = PREDECESSOR_ADAPTER_CONFIG_SHA256
    runner.DEVELOPMENT_ROOT = DEVELOPMENT_ROOT
    runner.DEVELOPMENT_RESULT_RECEIPT = DEVELOPMENT_RESULT_RECEIPT
    runner.SOURCE_START_STEP = 97
    runner.SOURCE_END_STEP = 116
    runner.TRAIN_UPDATES = TRAIN_UPDATES
    runner.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    runner.LEARNING_RATE = LEARNING_RATE
    runner.MAX_GRAD_NORM = MAX_GRAD_NORM
    runner.MARGIN = MARGIN
    runner.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    runner.validate_lineage = validate_lineage
    runner.stage2_schedule = stage5_schedule


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, 20))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runner()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Stage-5 post-training requires four-rank torchrun")
    try:
        result = runner.run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": (
                    result.get("receipt", {}).get("payload_sha256")
                    if context.is_primary
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
