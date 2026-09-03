#!/usr/bin/env python3
"""Train a donor-contrast recurrent-routing continuation on disjoint rows."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
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

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain_stage2 as stage2


runner = stage2
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage9_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "f651c0450abf8c09123fb8ab745dbe0796cd290d5ea80006f8d497bbb5116391"
V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 0.05
MARGIN = 0.02
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.5,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}
ALWAYS_ACTIVE_CONTROLS = ("matched_donor_recurrent_state",)
DEV_COUNTS = {"attribution": 8, "narrative": 32, "scene": 32}
TRAIN_COUNTS = {"attribution": 12, "narrative": 14, "scene": 14}
ORIGINAL_TRAIN = stage2.stage1.train


def load_v2_manifest() -> Mapping[str, Any]:
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if not isinstance(receipt, Mapping) or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest receipt differs")
    if common.canonical_sha256(value) != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest payload differs")
    value["receipt"] = receipt
    return value


def row_user_content(row: common.SourceRow) -> str:
    return str(json.loads(row.raw_line)["messages"][1]["content"])


def choose_hard_donor(target: common.SourceRow, rows: Sequence[common.SourceRow]) -> common.SourceRow:
    target_user = row_user_content(target)
    candidates = [
        row
        for row in rows
        if row.source_ordinal != target.source_ordinal
        and row.assistant_identity != target.assistant_identity
    ]
    if not candidates:
        raise ValueError(f"Stage-9 target has no different-answer donor: {target}")
    return max(
        candidates,
        key=lambda row: (
            SequenceMatcher(None, target_user, row_user_content(row)).ratio(),
            -abs(row.user_characters - target.user_characters),
            row.row_sha256,
        ),
    )


def stage9_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    manifest = load_v2_manifest()
    base_schedule, base_payload = common.build_training_schedule(rows_by_task, updates=96)
    used = {
        task: {row.target.source_ordinal for row in base_schedule if row.target.task == task}
        for task in common.TASKS
    }
    lookup = {
        task: {row.source_ordinal: row for row in rows_by_task[task]}
        for task in common.TASKS
    }
    selected: dict[str, tuple[common.SourceRow, ...]] = {}
    for task in common.TASKS:
        ordinals = tuple(manifest["candidate_training_source_ordinals"][task])
        if len(ordinals) != TRAIN_COUNTS[task] or any(ordinal in used[task] for ordinal in ordinals):
            raise RuntimeError(f"Stage-9 candidate rows overlap prior schedule for {task}")
        selected[task] = tuple(lookup[task][ordinal] for ordinal in ordinals)
    dev_ordinals = manifest["development_source_ordinals"]
    if any(set(dev_ordinals[task]) & {row.source_ordinal for row in selected[task]} for task in common.TASKS):
        raise RuntimeError("Stage-9 candidate rows overlap development-v2 rows")
    targets: list[common.SourceRow] = []
    remaining = {task: list(selected[task]) for task in common.TASKS}
    desired = {task: len(remaining[task]) for task in common.TASKS}
    for _ in range(sum(desired.values())):
        task = max(
            common.TASKS,
            key=lambda value: (
                len(remaining[value]) / desired[value] if desired[value] else -1.0,
                hashlib.sha256(f"stage9-task:{value}:{len(targets)}".encode()).hexdigest(),
            ),
        )
        if not remaining[task]:
            task = max(common.TASKS, key=lambda value: len(remaining[value]))
        targets.append(remaining[task].pop(0))
    if len(targets) != TRAIN_UPDATES * 2:
        raise RuntimeError("Stage-9 target count differs")
    schedule: list[common.ScheduledRow] = []
    payload: list[dict[str, Any]] = []
    for step in range(1, TRAIN_UPDATES + 1):
        step_targets = targets[(step - 1) * 2 : step * 2]
        step_rows = []
        for target in step_targets:
            donor = choose_hard_donor(target, selected[target.task])
            for variant in range(4):
                step_rows.append((target, donor, variant))
        step_rows.sort(
            key=lambda item: hashlib.sha256(
                f"rwkv-ms-recurrent-routed-stage9:{step}:{item[0].row_sha256}:{item[2]}".encode()
            ).hexdigest()
        )
        step_payload = []
        for position, (target, donor, variant) in enumerate(step_rows):
            schedule.append(common.ScheduledRow(step, position, target, donor, variant))
            step_payload.append({
                "position": position,
                "task": target.task,
                "source_ordinal": target.source_ordinal,
                "source_row_sha256": target.row_sha256,
                "donor_source_ordinal": donor.source_ordinal,
                "donor_row_sha256": donor.row_sha256,
                "prompt_variant": variant,
            })
        payload.append({
            "step": step,
            "source_step": 96 + step,
            "rows": step_payload,
            "payload_sha256": common.canonical_sha256(step_payload),
        })
    if len(schedule) != TRAIN_UPDATES * stage2.stage1.GLOBAL_BATCH_SIZE:
        raise RuntimeError("Stage-9 schedule size differs")
    if any(sum(row.target.task == task for row in schedule) != TRAIN_COUNTS[task] * 4 for task in common.TASKS):
        raise RuntimeError("Stage-9 task balance differs")
    lineage = common.canonical_sha256({"prior_96_steps": base_payload, "stage9_steps": payload, "objective": "always_on_donor_state_contrast_v1"})
    return tuple(schedule), payload, lineage


def train_with_always_on_donor(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    kwargs = dict(kwargs)
    kwargs["always_active_controls"] = ALWAYS_ACTIVE_CONTROLS
    return ORIGINAL_TRAIN(*args, **kwargs)


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(PREDECESSOR_ROOT / "result.json", PREDECESSOR_RESULT_RECEIPT)
    manifest = load_v2_manifest()
    if (
        predecessor.get("status") != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or predecessor.get("final_rows_opened") is not False
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json") != PREDECESSOR_ADAPTER_CONFIG_SHA256
        or manifest.get("final_rows_opened") is not False
        or protocol.get("development_v2_manifest_receipt") != V2_MANIFEST_RECEIPT
    ):
        raise ValueError("Stage-9 recurrent-routing lineage differs")
    return protocol, predecessor


def configure_runner() -> None:
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage9.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage9_input.v1"
    runner.PREFLIGHT_STATUS = "stage9_preflight_passed"
    runner.TRAINING_STATUS = "stage9_training_complete_development_v2_evaluation_authorized"
    runner.FAILURE_STATUS = "stage9_training_failed_development_v2_evaluation_blocked"
    runner.RUNNER_FILE = Path(__file__)
    runner.PROTOCOL = PROTOCOL
    runner.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    runner.STAGE1_ROOT = PREDECESSOR_ROOT
    runner.STAGE1_RESULT_RECEIPT = PREDECESSOR_RESULT_RECEIPT
    runner.STAGE1_ADAPTER_WEIGHTS_SHA256 = common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt")
    runner.STAGE1_ADAPTER_CONFIG_SHA256 = PREDECESSOR_ADAPTER_CONFIG_SHA256
    runner.DEVELOPMENT_ROOT = V2_ROOT
    runner.DEVELOPMENT_RESULT_RECEIPT = V2_MANIFEST_RECEIPT
    runner.SOURCE_START_STEP = 97
    runner.SOURCE_END_STEP = 96 + TRAIN_UPDATES
    runner.TRAIN_UPDATES = TRAIN_UPDATES
    runner.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    runner.LEARNING_RATE = LEARNING_RATE
    runner.MAX_GRAD_NORM = MAX_GRAD_NORM
    runner.MARGIN = MARGIN
    runner.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    runner.validate_lineage = validate_lineage
    runner.stage2_schedule = stage9_schedule
    runner.stage1.train = train_with_always_on_donor


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
        raise ValueError("Stage-9 post-training requires four-rank torchrun")
    try:
        result = runner.run(context=context, output_dir=args.output_dir, updates=args.updates, base_model=args.base_model)
    finally:
        runner.stage1.train = ORIGINAL_TRAIN
        distributed.destroy_distributed_training(context)
    print(json.dumps({"rank": context.process_rank, "status": result["status"], "passed": result["passed"], "result_receipt": result.get("receipt", {}).get("payload_sha256") if context.is_primary else None}, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
