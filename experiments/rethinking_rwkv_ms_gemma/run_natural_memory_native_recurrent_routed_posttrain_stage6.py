#!/usr/bin/env python3
"""Run donor-discriminative recurrent-routing post-training."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.distributed as torch_distributed
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
stage1 = runner.stage1
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage6_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "15f5bbbf59ef0a52f669615f613e25bdf6226c2891b3dae1768f2ff4a81a3c4f"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = "7af3769fa34631329a54fb8caf44797a3a5598344e104680b6aa2cb108339248"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
STAGE5_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage5_train20_v1"
STAGE5_RESULT_RECEIPT = "7f780f6731136dacc2bcf0c26da5813d7a284b0cb70afa70d758a9366001888e"
DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage5_development_v1"
DEVELOPMENT_RESULT_RECEIPT = "a4117c333c64ccbc0b91c19d8c2f7ecf58f6dfb0b0b28a8cc573dac45fe00508"
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 0.05
MARGIN = 0.1
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}
ORIGINAL_TRAIN = stage1.train
ORIGINAL_COLLATE = stage1.evolution.collate_native_examples


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(
        PREDECESSOR_ROOT / "result.json",
        PREDECESSOR_RESULT_RECEIPT,
    )
    stage5_result = common.validate_signed_json(
        STAGE5_ROOT / "result.json",
        STAGE5_RESULT_RECEIPT,
    )
    development = common.validate_signed_json(
        DEVELOPMENT_ROOT / "result.json",
        DEVELOPMENT_RESULT_RECEIPT,
    )
    if (
        predecessor.get("status")
        != "stage2_training_complete_development_evaluation_authorized"
        or predecessor.get("passed") is not True
        or stage5_result.get("status")
        != "stage5_training_complete_development_evaluation_authorized"
        or stage5_result.get("passed") is not True
        or development.get("status")
        != "development_failed_final_evaluation_blocked"
        or development.get("passed") is not False
        or development.get("summary", {}).get("gates", {}).get(
            "overall_correct_over_all_controls"
        )
        is not True
        or development.get("summary", {}).get("gates", {}).get(
            "projected_carriers_fixed"
        )
        is not True
        or development.get("final_rows_opened") is not False
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt")
        != PREDECESSOR_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json")
        != PREDECESSOR_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Stage-6 recurrent-routing lineage differs")
    return protocol, predecessor


def donor_discriminative_positions(
    target_token_ids: Sequence[int],
    donor_token_ids: Sequence[int],
) -> tuple[int, ...]:
    matcher = SequenceMatcher(
        None,
        list(target_token_ids),
        list(donor_token_ids),
        autojunk=True,
    )
    equal_positions = set()
    for block in matcher.get_matching_blocks():
        equal_positions.update(range(block.a, block.a + block.size))
    selected = tuple(
        index for index in range(len(target_token_ids)) if index not in equal_positions
    )
    if not selected:
        raise ValueError("Different donor answer selected no discriminative target tokens")
    return selected


def mask_target_labels(
    target: Any,
    donor: Any,
) -> tuple[int, int]:
    if target.labels.ndim != 2 or target.labels.shape[0] != 1:
        raise ValueError("Stage-6 masking requires one target row")
    if donor.labels.ndim != 2 or donor.labels.shape[0] != 1:
        raise ValueError("Stage-6 masking requires one donor row")
    target_positions = torch.nonzero(
        target.labels[0].ne(-100),
        as_tuple=False,
    ).flatten()
    donor_positions = torch.nonzero(
        donor.labels[0].ne(-100),
        as_tuple=False,
    ).flatten()
    target_ids = target.labels[0, target_positions].detach().cpu().tolist()
    donor_ids = donor.labels[0, donor_positions].detach().cpu().tolist()
    selected_offsets = donor_discriminative_positions(target_ids, donor_ids)
    selected_positions = target_positions[
        torch.tensor(selected_offsets, device=target_positions.device)
    ]
    masked = torch.full_like(target.labels, -100)
    masked[0, selected_positions] = target.labels[0, selected_positions]
    target.labels.copy_(masked)
    return len(selected_offsets), len(target_ids)


def train_discriminative(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    pending_target = None
    local_rows = 0
    local_selected_tokens = 0
    local_answer_tokens = 0

    def collate_with_discriminative_labels(*collate_args: Any, **collate_kwargs: Any) -> Any:
        nonlocal pending_target
        nonlocal local_rows
        nonlocal local_selected_tokens
        nonlocal local_answer_tokens
        batch = ORIGINAL_COLLATE(*collate_args, **collate_kwargs)
        if pending_target is None:
            pending_target = batch
        else:
            selected, total = mask_target_labels(pending_target, batch)
            local_rows += 1
            local_selected_tokens += selected
            local_answer_tokens += total
            pending_target = None
        return batch

    if stage1.evolution.collate_native_examples is not ORIGINAL_COLLATE:
        raise RuntimeError("Stage-6 collate hook already installed")
    stage1.evolution.collate_native_examples = collate_with_discriminative_labels
    try:
        trained = ORIGINAL_TRAIN(*args, **kwargs)
    finally:
        stage1.evolution.collate_native_examples = ORIGINAL_COLLATE
    if pending_target is not None:
        raise RuntimeError("Stage-6 collate hook ended with an unmatched target")
    context = kwargs["context"]
    audit_values = torch.tensor(
        [local_rows, local_selected_tokens, local_answer_tokens],
        dtype=torch.long,
        device=context.device,
    )
    torch_distributed.all_reduce(audit_values, op=torch_distributed.ReduceOp.SUM)
    global_rows, selected_tokens, answer_tokens = (
        int(value) for value in audit_values.cpu().tolist()
    )
    expected_rows = int(kwargs["updates"]) * stage1.GLOBAL_BATCH_SIZE
    if (
        global_rows != expected_rows
        or selected_tokens < global_rows
        or answer_tokens < selected_tokens
    ):
        raise RuntimeError("Stage-6 discriminative label audit differs")
    return {
        **trained,
        "discriminative_label_audit": {
            "rows": global_rows,
            "selected_target_tokens": selected_tokens,
            "total_target_answer_tokens": answer_tokens,
            "selected_fraction": selected_tokens / answer_tokens,
            "task_specific_parsing": False,
            "task_specific_token_rules": False,
            "passed": True,
        },
    }


def stage6_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    schedule, payload, _ = stage5.stage5_schedule(rows_by_task)
    prior_schedule, prior_payload = common.build_training_schedule(
        rows_by_task,
        updates=96,
    )
    if len(prior_schedule) != 96 * stage1.GLOBAL_BATCH_SIZE:
        raise RuntimeError("Stage-6 prior schedule differs")
    lineage_hash = common.canonical_sha256(
        {
            "prior_96_steps": prior_payload,
            "stage6_steps": payload,
            "label_objective": "donor_discriminative_sequence_alignment_v1",
        }
    )
    return schedule, payload, lineage_hash


def configure_runner() -> None:
    stage5.configure_runner()
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage6.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage6_input.v1"
    runner.PREFLIGHT_STATUS = "stage6_preflight_passed"
    runner.TRAINING_STATUS = "stage6_training_complete_development_evaluation_authorized"
    runner.FAILURE_STATUS = "stage6_training_failed_development_evaluation_blocked"
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
    runner.stage2_schedule = stage6_schedule
    runner.stage1.train = train_discriminative


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
        raise ValueError("Stage-6 post-training requires four-rank torchrun")
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
