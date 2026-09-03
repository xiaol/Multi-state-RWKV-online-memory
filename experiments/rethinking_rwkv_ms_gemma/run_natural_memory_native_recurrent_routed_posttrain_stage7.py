#!/usr/bin/env python3
"""Run same-example paraphrase-invariant recurrent-routing post-training."""

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
    run_natural_memory_native_recurrent_routed_posttrain_stage5 as stage5,
)


runner = stage5.runner
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage7_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "f7b4833c3b5442fb31568f255e88a6cdda80b319fb11db66255017d3231bbdec"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = "7af3769fa34631329a54fb8caf44797a3a5598344e104680b6aa2cb108339248"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage6_development_v1"
DEVELOPMENT_RESULT_RECEIPT = "07f17fe62b69312783d997102b7edeb146bac7eddddb36fccba023e88c88080f"
STAGE2_DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_development_v1"
STAGE2_DEVELOPMENT_RECEIPT = "88129262892d28795b23752d44289133c2f5416245847d944af17c9b4853a47a"
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 5e-6
MAX_GRAD_NORM = 0.05
MARGIN = 0.02
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}


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
    latest_development = common.validate_signed_json(
        DEVELOPMENT_ROOT / "result.json",
        DEVELOPMENT_RESULT_RECEIPT,
    )
    if (
        predecessor.get("status")
        != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or stage2_development.get("passed") is not False
        or latest_development.get("passed") is not False
        or stage2_development.get("final_rows_opened") is not False
        or latest_development.get("final_rows_opened") is not False
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
        raise ValueError("Stage-7 recurrent-routing lineage differs")
    return protocol, predecessor


def stage7_schedule(
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
                        "rwkv-ms-recurrent-routed-stage7-target:"
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
    required = {"attribution": 10, "narrative": 20, "scene": 10}
    selected = {
        task: remaining[task][:count]
        for task, count in required.items()
    }
    if any(len(selected[task]) != required[task] for task in common.TASKS):
        raise RuntimeError("Stage-7 has insufficient stage-2-untouched targets")
    task_cursors = {task: 0 for task in common.TASKS}
    schedule = []
    payload = []
    for step in range(1, TRAIN_UPDATES + 1):
        paired_task = "attribution" if step % 2 else "scene"
        step_targets = []
        for task in ("narrative", paired_task):
            target = selected[task][task_cursors[task]]
            task_cursors[task] += 1
            step_targets.append(target)
        step_rows = []
        for target in step_targets:
            donor = stage5.stage3.choose_donor(target, rows_by_task[target.task])
            for variant in range(4):
                step_rows.append((target, donor, variant))
        step_rows.sort(
            key=lambda item: hashlib.sha256(
                (
                    f"rwkv-ms-recurrent-routed-stage7:{step}:"
                    f"{item[0].row_sha256}:{item[2]}"
                ).encode("utf-8")
            ).hexdigest()
        )
        step_payload = []
        for position, (target, donor, variant) in enumerate(step_rows):
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
    unique_targets = {
        (row.target.task, row.target.source_ordinal)
        for row in schedule
    }
    if (
        len(schedule) != TRAIN_UPDATES * 8
        or len(unique_targets) != 40
        or task_cursors != required
        or any(
            {
                row.prompt_variant
                for row in schedule
                if row.target.task == task
                and row.target.source_ordinal == ordinal
            }
            != {0, 1, 2, 3}
            for task, ordinal in unique_targets
        )
    ):
        raise RuntimeError("Stage-7 same-example paraphrase schedule differs")
    lineage_hash = common.canonical_sha256(
        {"prior_96_steps": prior_payload, "stage7_steps": payload}
    )
    return tuple(schedule), payload, lineage_hash


def configure_runner() -> None:
    stage5.configure_runner()
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage7.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage7_input.v1"
    runner.PREFLIGHT_STATUS = "stage7_preflight_passed"
    runner.TRAINING_STATUS = "stage7_training_complete_development_evaluation_authorized"
    runner.FAILURE_STATUS = "stage7_training_failed_development_evaluation_blocked"
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
    runner.stage2_schedule = stage7_schedule


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
        raise ValueError("Stage-7 post-training requires four-rank torchrun")
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
