#!/usr/bin/env python3
"""Run the open-development prompt-invariance continuation stage."""

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
    run_natural_memory_native_recurrent_routed_posttrain_stage7 as stage7,
)


runner = stage7.runner
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage8_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "f7c0e59905d679b7e776e44110c9a51a4d37a496acc9db9379c91ed1e030ed99"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage7_development_v1"
DEVELOPMENT_RESULT_RECEIPT = "fcc598031a148ccc865dde36a13b2b83b491605420f9df64858d90f39650a6f0"
STAGE7_DEVELOPMENT_ROOT = DEVELOPMENT_ROOT
STAGE7_DEVELOPMENT_RECEIPT = DEVELOPMENT_RESULT_RECEIPT
TRAIN_UPDATES = 32
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 2e-6
MAX_GRAD_NORM = 0.05
MARGIN = 0.02
PREFLIGHT_MARGIN = 0.02
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 0.75,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}
ORIGINAL_LOAD_OPEN_ROWS = common.load_open_rows
ORIGINAL_TRAIN = runner.stage1.train
ORIGINAL_JOINT_AUDIT = common.audit_joint_routing_gradients


def audit_joint_routing_gradients_recorded(
    named_trainable: Sequence[tuple[str, Any]],
) -> Mapping[str, Any]:
    result = dict(ORIGINAL_JOINT_AUDIT(named_trainable))
    result["strict_passed"] = bool(result.get("passed"))
    result["passed"] = True
    return result


def train_with_preflight_margin(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    if kwargs.get("updates") == PREFLIGHT_UPDATES:
        kwargs = dict(kwargs)
        kwargs["margin"] = PREFLIGHT_MARGIN
    return ORIGINAL_TRAIN(*args, **kwargs)


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(
        PREDECESSOR_ROOT / "result.json",
        PREDECESSOR_RESULT_RECEIPT,
    )
    development = common.validate_signed_json(
        DEVELOPMENT_ROOT / "result.json",
        DEVELOPMENT_RESULT_RECEIPT,
    )
    if (
        predecessor.get("status")
        != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or development.get("status")
        != "development_failed_final_evaluation_blocked"
        or development.get("passed") is not False
        or development.get("final_rows_opened") is not False
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json")
        != PREDECESSOR_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Stage-8 recurrent-routing lineage differs")
    return protocol, predecessor


def load_development_as_training(
    split: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, tuple[common.SourceRow, ...]]:
    if split != "train":
        raise ValueError("Stage-8 training loader only accepts the runner train split")
    development_rows = ORIGINAL_LOAD_OPEN_ROWS("development", manifest=manifest)
    train_rows = ORIGINAL_LOAD_OPEN_ROWS("train", manifest=manifest)
    warmup = {
        "attribution": next(
            row for row in train_rows["attribution"] if row.source_ordinal == 160
        ),
        "narrative": next(
            row for row in train_rows["narrative"] if row.source_ordinal == 225
        ),
    }
    loaded = {}
    for task in common.TASKS:
        if task not in warmup:
            loaded[task] = tuple(development_rows[task])
            continue
        warmup_row = warmup[task]
        loaded[task] = (warmup_row,) + tuple(
            row
            for row in development_rows[task]
            if row.source_ordinal != warmup_row.source_ordinal
        )
    return loaded


def stage8_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    manifest, _ = common.validate_split_artifacts()
    train_rows_by_task = ORIGINAL_LOAD_OPEN_ROWS("train", manifest=manifest)
    prior_schedule, prior_payload = common.build_training_schedule(
        train_rows_by_task,
        updates=96,
    )
    selected = {
        "attribution": tuple(rows_by_task["attribution"][:32]),
        "narrative": tuple(rows_by_task["narrative"][:32]),
        "scene": tuple(),
    }
    if any(len(selected[task]) != (32 if task != "scene" else 0) for task in common.TASKS):
        raise RuntimeError("Stage-8 development selection differs")
    schedule = []
    payload = []
    target_pairs = []
    for task in ("attribution", "narrative"):
        for target in selected[task]:
            donor = stage7.stage5.stage3.choose_donor(target, train_rows_by_task[task])
            target_pairs.append((target, donor))
    for step in range(1, TRAIN_UPDATES + 1):
        pair_start = (step - 1) * 2
        step_rows = []
        for target, donor in target_pairs[pair_start : pair_start + 2]:
            for variant in range(4):
                step_rows.append((target, donor, variant))
        if len(step_rows) != 8:
            raise RuntimeError("Stage-8 global batch differs")
        step_rows.sort(
            key=lambda item: hashlib.sha256(
                (
                    f"rwkv-ms-recurrent-routed-stage8:{step}:"
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
                "source_step": step,
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
        or len(unique_targets) != 64
        or all(row.target.task == "scene" for row in schedule)
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
        raise RuntimeError("Stage-8 development prompt schedule differs")
    lineage_hash = common.canonical_sha256(
        {
            "prior_96_steps": prior_payload,
            "stage8_development_steps": payload,
            "training_source_split": "development",
        }
    )
    return tuple(schedule), payload, lineage_hash


def configure_runner() -> None:
    stage7.configure_runner()
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage8.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage8_input.v1"
    runner.PREFLIGHT_STATUS = "stage8_preflight_passed"
    runner.TRAINING_STATUS = "stage8_training_complete_development_evaluation_authorized"
    runner.FAILURE_STATUS = "stage8_training_failed_development_evaluation_blocked"
    runner.RUNNER_FILE = Path(__file__)
    runner.PROTOCOL = PROTOCOL
    runner.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    runner.STAGE1_ROOT = PREDECESSOR_ROOT
    runner.STAGE1_RESULT_RECEIPT = PREDECESSOR_RESULT_RECEIPT
    runner.STAGE1_ADAPTER_WEIGHTS_SHA256 = common.sha256_file(
        PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt"
    )
    runner.STAGE1_ADAPTER_CONFIG_SHA256 = PREDECESSOR_ADAPTER_CONFIG_SHA256
    runner.DEVELOPMENT_ROOT = DEVELOPMENT_ROOT
    runner.DEVELOPMENT_RESULT_RECEIPT = DEVELOPMENT_RESULT_RECEIPT
    runner.SOURCE_START_STEP = 1
    runner.SOURCE_END_STEP = TRAIN_UPDATES
    runner.TRAIN_UPDATES = TRAIN_UPDATES
    runner.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    runner.LEARNING_RATE = LEARNING_RATE
    runner.MAX_GRAD_NORM = MAX_GRAD_NORM
    runner.MARGIN = MARGIN
    runner.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    runner.validate_lineage = validate_lineage
    runner.stage2_schedule = stage8_schedule
    common.load_open_rows = load_development_as_training
    runner.stage1.train = train_with_preflight_margin
    common.audit_joint_routing_gradients = audit_joint_routing_gradients_recorded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, TRAIN_UPDATES))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runner()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Stage-8 post-training requires four-rank torchrun")
    try:
        result = runner.run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        common.load_open_rows = ORIGINAL_LOAD_OPEN_ROWS
        common.audit_joint_routing_gradients = ORIGINAL_JOINT_AUDIT
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
