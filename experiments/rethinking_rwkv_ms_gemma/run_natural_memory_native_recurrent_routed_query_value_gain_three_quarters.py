#!/usr/bin/env python3
"""Continue recurrent routing with a learned query-conditioned value term."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import os
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta_impl import load_delta_mem_state_dict  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain_stage2 as stage2  # noqa: E402


MODE = "recurrent_routed_query_value"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_query_value_gain_three_quarters_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "2809d6f40cf68ef6b75bcf2d58fffc336c1350be87ceb748153f154da45b23aa"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = "7af3769fa34631329a54fb8caf44797a3a5598344e104680b6aa2cb108339248"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
SOURCE_START_STEP = 97
SOURCE_END_STEP = 116
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 5e-6
MAX_GRAD_NORM = 0.1
MARGIN = 0.05
HYBRID_GAIN = 0.75
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.125,
}
ALWAYS_ACTIVE_CONTROLS = ("matched_donor_recurrent_state",)
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


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(
        PREDECESSOR_ROOT / "result.json",
        PREDECESSOR_RESULT_RECEIPT,
    )
    manifest = load_v2_manifest()
    if (
        predecessor.get("status") != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or predecessor.get("final_rows_opened") is not False
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt")
        != PREDECESSOR_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json")
        != PREDECESSOR_ADAPTER_CONFIG_SHA256
        or manifest.get("final_rows_opened") is not False
    ):
        raise ValueError("Query-value predecessor lineage differs")
    return protocol, predecessor


def row_user_content(row: common.SourceRow) -> str:
    value = json.loads(row.raw_line)
    return str(value["messages"][1]["content"])


def stage12_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    full_schedule, full_payload = common.build_training_schedule(
        rows_by_task,
        updates=96,
    )
    prior_targets = {
        row.target.row_sha256 for row in full_schedule if row.step < SOURCE_START_STEP
    }
    development_ordinals = load_v2_manifest()["development_source_ordinals"]
    development = {
        task: set(ordinals) for task, ordinals in development_ordinals.items()
    }
    available = {
        task: [
            row
            for row in rows_by_task[task]
            if row.row_sha256 not in prior_targets
            and row.source_ordinal not in development[task]
        ]
        for task in common.TASKS
    }
    for task in common.TASKS:
        available[task].sort(
            key=lambda row: hashlib.sha256(
                f"query-value-stage12:{task}:{row.row_sha256}".encode("utf-8")
            ).hexdigest()
        )
    target_counts = {"attribution": 4, "narrative": 32, "scene": 4}
    selected_targets = {
        task: tuple(available[task][: target_counts[task]]) for task in common.TASKS
    }
    if any(len(selected_targets[task]) != target_counts[task] for task in common.TASKS):
        raise RuntimeError("Query-value untouched schedule differs")
    donors = {}
    for task in common.TASKS:
        for target_index, target in enumerate(selected_targets[task]):
            candidates = [
                row
                for row in available[task]
                if row.source_ordinal != target.source_ordinal
                and row.assistant_identity != target.assistant_identity
            ]
            if not candidates:
                raise RuntimeError(f"Query-value target has no donor: {target}")
            target_user = row_user_content(target)
            donor = max(
                candidates,
                key=lambda row: (
                    SequenceMatcher(None, target_user, row_user_content(row)).ratio(),
                    -abs(row.user_characters - target.user_characters),
                    row.row_sha256,
                ),
            )
            donors[target.row_sha256] = donor
    remaining = {task: list(selected_targets[task]) for task in common.TASKS}
    targets = []
    desired = {task: len(remaining[task]) for task in common.TASKS}
    while any(remaining.values()):
        task = max(
            common.TASKS,
            key=lambda value: (
                len(remaining[value]) / desired[value] if desired[value] else -1.0,
                hashlib.sha256(
                    f"query-value-stage12-task:{value}:{len(targets)}".encode()
                ).hexdigest(),
            ),
        )
        targets.append(remaining[task].pop(0))
    if len(targets) != TRAIN_UPDATES * 2:
        raise RuntimeError("Query-value target count differs")
    selected = []
    selected_payload = []
    for step in range(1, TRAIN_UPDATES + 1):
        step_targets = targets[(step - 1) * 2 : step * 2]
        step_rows = []
        for target in step_targets:
            donor = donors[target.row_sha256]
            for variant in range(4):
                step_rows.append((target, donor, variant))
        step_rows.sort(
            key=lambda item: hashlib.sha256(
                f"query-value-stage12-step:{step}:{item[0].row_sha256}:{item[2]}".encode()
            ).hexdigest()
        )
        step_payload = []
        for position, (target, donor, variant) in enumerate(step_rows):
            selected.append(common.ScheduledRow(step, position, target, donor, variant))
            step_payload.append({
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
            "rows": step_payload,
            "payload_sha256": common.canonical_sha256(step_payload),
        })
    return tuple(selected), selected_payload, common.canonical_sha256(
        {"prior_96_steps": full_payload, "stage12_steps": selected_payload}
    )


def load_predecessor_adapter(
    model: torch.nn.Module,
    input_dir: Path,
) -> Any:
    state = torch.load(
        input_dir / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_delta_mem_state_dict(
        model,
        state,
        initialize_missing_rwkv_pair_value=True,
    )
    return common.build_config()


def train_with_always_on_donor(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    kwargs = dict(kwargs)
    kwargs["always_active_controls"] = ALWAYS_ACTIVE_CONTROLS
    return ORIGINAL_TRAIN(*args, **kwargs)


def configure() -> None:
    common.HYBRID_GAIN = HYBRID_GAIN
    common.HYBRID_MODE = MODE
    common.PROTOCOL = PROTOCOL
    common.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    stage2.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_query_value.v1"
    stage2.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_query_value_input.v1"
    stage2.PREFLIGHT_STATUS = "query_value_preflight_passed"
    stage2.TRAINING_STATUS = "query_value_training_complete_development_v2_evaluation_authorized"
    stage2.FAILURE_STATUS = "query_value_training_failed_development_v2_evaluation_blocked"
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
    stage2.stage2_schedule = stage12_schedule
    stage2.load_delta_mem_adapter = load_predecessor_adapter
    stage2.stage1.train = train_with_always_on_donor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, TRAIN_UPDATES))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    global PROTOCOL_PAYLOAD_SHA256
    if PROTOCOL_PAYLOAD_SHA256 == "PLACEHOLDER":
        raise ValueError("Query-value protocol receipt is not installed")
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Query-value post-training requires four-rank torchrun")
    try:
        result = stage2.run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        stage2.stage1.train = ORIGINAL_TRAIN
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "result_receipt": result.get("receipt", {}).get("payload_sha256")
                if context.is_primary
                else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
