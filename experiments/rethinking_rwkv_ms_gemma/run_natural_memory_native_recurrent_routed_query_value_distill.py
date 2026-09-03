#!/usr/bin/env python3
"""Continue recurrent routing with projected-slot semantic distillation."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_query_value_gain_half as gain_half
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_query_value_gain_one as predecessor_runner
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_query_value_narrative_repair as narrative_repair
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain_stage2 as stage2
from deltamem.core.delta_impl import load_delta_mem_state_dict


MODE = "recurrent_routed_query_value"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_query_value_distill_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "345fbea5325d1eef8cd4c071a64500972a31e2c0bd9346f248a178a0fcba8d75"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_gain_one_train20_v1"
PREDECESSOR_RESULT_RECEIPT = "74939af17b1296e04cd37a67f17d73b8865de69e6e29cded5d3a77a05259e858"
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = "a593d778d0587319464bc9a6520d9c0b08c52be98cdab7fda39f48946825dd04"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "4484ad76ff6523626a9ee11bb04d40723e35d90443d0b58c41c4f8a4e652c84b"
V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
TEACHER_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"
DISTILL_CACHE_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_query_value_distill_teacher_cache_v1"
SOURCE_START_STEP = 137
SOURCE_END_STEP = 156
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 0.1
MARGIN = 0.05
HYBRID_GAIN = 1.0
DISTILL_WEIGHT = 0.5
DISTILL_TEMPERATURE = 1.0
DISTILL_TOP_K = 64
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.125,
}
ALWAYS_ACTIVE_CONTROLS = ("matched_donor_recurrent_state",)
ORIGINAL_TRAIN = predecessor_runner.ORIGINAL_TRAIN


def load_v2_manifest() -> Mapping[str, Any]:
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if not isinstance(receipt, Mapping) or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest receipt differs")
    if common.canonical_sha256(value) != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest payload differs")
    value["receipt"] = receipt
    return value


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(PREDECESSOR_ROOT / "result.json", PREDECESSOR_RESULT_RECEIPT)
    manifest = load_v2_manifest()
    if (
        predecessor.get("status") != "query_value_gain_one_training_complete_development_v2_evaluation_authorized"
        or predecessor.get("passed") is not True
        or predecessor.get("final_rows_opened") is not False
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt") != PREDECESSOR_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json") != PREDECESSOR_ADAPTER_CONFIG_SHA256
        or manifest.get("final_rows_opened") is not False
    ):
        raise ValueError("Distillation predecessor lineage differs")
    return protocol, predecessor


def row_user_content(row: common.SourceRow) -> str:
    return str(json.loads(row.raw_line)["messages"][1]["content"])


def stage14_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    initial_schedule, initial_payload = common.build_training_schedule(rows_by_task, updates=96)
    prior_targets = {row.target.row_sha256 for row in initial_schedule}
    stage12, stage12_payload, _ = gain_half.stage12_schedule(rows_by_task)
    stage13, stage13_payload, _ = narrative_repair.stage13_schedule(rows_by_task)
    prior_targets.update(row.target.row_sha256 for row in stage12)
    prior_targets.update(row.target.row_sha256 for row in stage13)
    development = {task: set(load_v2_manifest()["development_source_ordinals"][task]) for task in common.TASKS}
    available: dict[str, list[common.SourceRow]] = {}
    for task in common.TASKS:
        candidates = [
            row for row in rows_by_task[task]
            if row.row_sha256 not in prior_targets and row.source_ordinal not in development[task]
        ]
        candidates.sort(key=lambda row: hashlib.sha256(f"query-value-distill-stage14:{task}:{row.row_sha256}".encode()).hexdigest())
        available[task] = candidates
    target_counts = {"attribution": 4, "narrative": 8, "scene": 28}
    selected_targets = {task: tuple(available[task][: target_counts[task]]) for task in common.TASKS}
    if any(len(selected_targets[task]) != target_counts[task] for task in common.TASKS):
        raise RuntimeError("Distillation untouched schedule differs")
    donors: dict[str, common.SourceRow] = {}
    for task in common.TASKS:
        for target in selected_targets[task]:
            candidates = [
                row for row in available[task]
                if row.source_ordinal != target.source_ordinal and row.assistant_identity != target.assistant_identity
            ]
            donors[target.row_sha256] = max(
                candidates,
                key=lambda row: (
                    SequenceMatcher(None, row_user_content(target), row_user_content(row)).ratio(),
                    -abs(row.user_characters - target.user_characters),
                    row.row_sha256,
                ),
            )
    remaining = {task: list(selected_targets[task]) for task in common.TASKS}
    desired = {task: len(remaining[task]) for task in common.TASKS}
    targets: list[common.SourceRow] = []
    while any(remaining.values()):
        task = max(
            common.TASKS,
            key=lambda value: (
                len(remaining[value]) / desired[value] if desired[value] else -1.0,
                hashlib.sha256(f"query-value-distill-stage14-task:{value}:{len(targets)}".encode()).hexdigest(),
            ),
        )
        targets.append(remaining[task].pop(0))
    selected: list[common.ScheduledRow] = []
    selected_payload: list[dict[str, Any]] = []
    for step in range(1, TRAIN_UPDATES + 1):
        step_rows = []
        for target in targets[(step - 1) * 2 : step * 2]:
            for variant in range(4):
                step_rows.append((target, donors[target.row_sha256], variant))
        step_rows.sort(key=lambda item: hashlib.sha256(f"query-value-distill-stage14-step:{step}:{item[0].row_sha256}:{item[2]}".encode()).hexdigest())
        payload_rows = []
        for position, (target, donor, variant) in enumerate(step_rows):
            selected.append(common.ScheduledRow(step, position, target, donor, variant))
            payload_rows.append({
                "position": position,
                "task": target.task,
                "source_ordinal": target.source_ordinal,
                "source_row_sha256": target.row_sha256,
                "donor_source_ordinal": donor.source_ordinal,
                "donor_row_sha256": donor.row_sha256,
                "prompt_variant": variant,
            })
        selected_payload.append({
            "step": step,
            "source_step": SOURCE_START_STEP - 1 + step,
            "rows": payload_rows,
            "payload_sha256": common.canonical_sha256(payload_rows),
        })
    if len(selected) != TRAIN_UPDATES * 8:
        raise RuntimeError("Distillation schedule row count differs")
    return tuple(selected), selected_payload, common.canonical_sha256({
        "initial_schedule": initial_payload,
        "stage12_schedule": stage12_payload,
        "stage13_schedule": stage13_payload,
        "stage14_schedule": selected_payload,
    })


def load_predecessor_adapter(model: Any, input_dir: Path) -> Any:
    state = __import__("torch").load(input_dir / "delta_mem_adapter.pt", map_location="cpu", weights_only=True)
    load_delta_mem_state_dict(model, state)
    return common.build_config()


def train_with_distillation(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    kwargs = dict(kwargs)
    kwargs["always_active_controls"] = ALWAYS_ACTIVE_CONTROLS
    return ORIGINAL_TRAIN(*args, **kwargs)


def configure() -> None:
    predecessor_runner.configure()
    common.HYBRID_MODE = MODE
    common.HYBRID_GAIN = HYBRID_GAIN
    common.PROTOCOL = PROTOCOL
    common.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    stage2.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_query_value_distill.v1"
    stage2.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_query_value_distill_input.v1"
    stage2.PREFLIGHT_STATUS = "query_value_distill_preflight_passed"
    stage2.TRAINING_STATUS = "query_value_distill_training_complete_development_v2_evaluation_authorized"
    stage2.FAILURE_STATUS = "query_value_distill_training_failed_development_v2_evaluation_blocked"
    stage2.RUNNER_FILE = Path(__file__)
    stage2.PROTOCOL = PROTOCOL
    stage2.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    stage2.STAGE1_ROOT = PREDECESSOR_ROOT
    stage2.STAGE1_RESULT_RECEIPT = PREDECESSOR_RESULT_RECEIPT
    stage2.STAGE1_ADAPTER_WEIGHTS_SHA256 = PREDECESSOR_ADAPTER_WEIGHTS_SHA256
    stage2.STAGE1_ADAPTER_CONFIG_SHA256 = PREDECESSOR_ADAPTER_CONFIG_SHA256
    stage2.DEVELOPMENT_ROOT = V2_ROOT
    stage2.DEVELOPMENT_RESULT_RECEIPT = V2_MANIFEST_RECEIPT
    stage2.SOURCE_START_STEP = SOURCE_START_STEP
    stage2.SOURCE_END_STEP = SOURCE_END_STEP
    stage2.TRAIN_UPDATES = TRAIN_UPDATES
    stage2.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    stage2.LEARNING_RATE = LEARNING_RATE
    stage2.MAX_GRAD_NORM = MAX_GRAD_NORM
    stage2.MARGIN = MARGIN
    stage2.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    stage2.validate_lineage = validate_lineage
    stage2.stage2_schedule = stage14_schedule
    stage2.load_delta_mem_adapter = load_predecessor_adapter
    stage2.stage1.train = train_with_distillation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(PREFLIGHT_UPDATES, TRAIN_UPDATES))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if PROTOCOL_PAYLOAD_SHA256 == "PLACEHOLDER":
        raise ValueError("Distillation protocol receipt is not installed")
    configure()
    stage2.stage1.DISTILL_TEACHER_ADAPTER = TEACHER_ADAPTER
    stage2.stage1.DISTILL_BASE_MODEL = args.base_model
    stage2.stage1.DISTILL_CACHE_ROOT = DISTILL_CACHE_ROOT
    stage2.stage1.DISTILL_WEIGHT = DISTILL_WEIGHT
    stage2.stage1.DISTILL_TEMPERATURE = DISTILL_TEMPERATURE
    stage2.stage1.DISTILL_TOP_K = DISTILL_TOP_K
    context = distributed.initialize_distributed_training(args.device, timeout_seconds=7200)
    if context is None:
        raise ValueError("Distillation continuation requires four-rank torchrun")
    try:
        result = stage2.run(context=context, output_dir=args.output_dir, updates=args.updates, base_model=args.base_model)
    finally:
        stage2.stage1.train = ORIGINAL_TRAIN
        stage2.stage1.DISTILL_TEACHER_ADAPTER = None
        stage2.stage1.DISTILL_BASE_MODEL = None
        stage2.stage1.DISTILL_CACHE_ROOT = None
        stage2.stage1.DISTILL_WEIGHT = 0.0
        distributed.destroy_distributed_training(context)
    print(json.dumps({"rank": context.process_rank, "status": result["status"], "passed": result["passed"], "result_receipt": result.get("receipt", {}).get("payload_sha256") if context.is_primary else None}, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
