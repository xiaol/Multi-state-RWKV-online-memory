#!/usr/bin/env python3
"""Continue recurrent-routed post-training on untouched locked train rows."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    load_delta_mem_adapter,
    save_delta_mem_adapter,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain as stage1,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage2.v1"
INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage2_input.v1"
PREFLIGHT_STATUS = "stage2_preflight_passed"
TRAINING_STATUS = "stage2_training_complete_development_evaluation_authorized"
FAILURE_STATUS = "stage2_training_failed_development_evaluation_blocked"
RUNNER_FILE = Path(__file__)
PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage2_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = (
    "1b7a7f2c5d6fe7641131f175dfa16d42dcc9237dacec708f3c89854fe540137e"
)
STAGE1_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_train32_v4"
STAGE1_RESULT_RECEIPT = (
    "2c83c06a323b08d150e8b91263fffe3c9fb9a69f28bbf398085a352c353bf85f"
)
STAGE1_ADAPTER_WEIGHTS_SHA256 = (
    "6a424d3c8dde13ca68f269d6b2fd09e27b22a177c4d71aa9177ce31044fc3449"
)
STAGE1_ADAPTER_CONFIG_SHA256 = (
    "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
)
DEVELOPMENT_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v1"
DEVELOPMENT_RESULT_RECEIPT = (
    "76617f9bc77261d7fbfecacc1dd8685f5a07f74b6d20f81dd9d384f528caf10a"
)
WORLD_SIZE = 4
SEED = 20260828
SOURCE_START_STEP = 33
SOURCE_END_STEP = 96
TRAIN_UPDATES = 64
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.2
MARGIN = 0.1
LEARNING_RATE_MULTIPLIERS: Mapping[str, float] = {}
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 0.25,
    "slot_shuffled_recurrent_state": 0.25,
    "layer_permuted_recurrent_state": 0.5,
}


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    stage1_result = common.validate_signed_json(
        STAGE1_ROOT / "result.json",
        STAGE1_RESULT_RECEIPT,
    )
    development_result = common.validate_signed_json(
        DEVELOPMENT_ROOT / "result.json",
        DEVELOPMENT_RESULT_RECEIPT,
    )
    if (
        stage1_result.get("status")
        != "training_complete_development_evaluation_authorized"
        or stage1_result.get("passed") is not True
        or development_result.get("status")
        != "development_failed_final_evaluation_blocked"
        or development_result.get("passed") is not False
        or development_result.get("final_rows_opened") is not False
        or common.sha256_file(STAGE1_ROOT / "adapter/delta_mem_adapter.pt")
        != STAGE1_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(STAGE1_ROOT / "adapter/delta_mem_config.json")
        != STAGE1_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Stage-2 recurrent-routed lineage differs")
    return protocol, stage1_result


def stage2_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]], str]:
    full_schedule, full_payload = common.build_training_schedule(
        rows_by_task,
        updates=SOURCE_END_STEP,
    )
    selected = tuple(
        replace(row, step=row.step - SOURCE_START_STEP + 1)
        for row in full_schedule
        if SOURCE_START_STEP <= row.step <= SOURCE_END_STEP
    )
    selected_payload = []
    for source in full_payload[SOURCE_START_STEP - 1 : SOURCE_END_STEP]:
        selected_payload.append(
            {
                **source,
                "source_step": source["step"],
                "step": source["step"] - SOURCE_START_STEP + 1,
            }
        )
    earlier_rows = {
        row.target.row_sha256
        for row in full_schedule
        if row.step < SOURCE_START_STEP
    }
    selected_rows = {row.target.row_sha256 for row in selected}
    if (
        len(selected) != TRAIN_UPDATES * stage1.GLOBAL_BATCH_SIZE
        or len(selected_payload) != TRAIN_UPDATES
        or earlier_rows & selected_rows
    ):
        raise RuntimeError("Stage-2 untouched schedule differs")
    return selected, selected_payload, common.canonical_sha256(full_payload)


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Stage-2 post-training requires exactly four ranks")
    if updates not in {PREFLIGHT_UPDATES, TRAIN_UPDATES}:
        raise ValueError("Stage-2 updates must be 1 or 64")
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    protocol, stage1_result = validate_lineage()
    manifest, open_receipt = common.validate_split_artifacts()

    resolved_output = output_dir.expanduser().resolve()
    creation_error = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-stage2-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    model, tokenizer, delta_config, model_audit = common.load_model(
        base_model,
        device=context.device,
        trainable=True,
    )
    named_trainable = model_audit.pop("named_trainable")
    loaded_config = load_delta_mem_adapter(model, STAGE1_ROOT / "adapter")
    if loaded_config.to_dict() != delta_config.to_dict():
        raise ValueError("Stage-2 initial adapter configuration differs")
    rows_by_task = common.load_open_rows("train", manifest=manifest)
    schedule, schedule_payload, full_schedule_sha256 = stage2_schedule(rows_by_task)

    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "seed": SEED,
        "updates": updates,
        "source_steps": [SOURCE_START_STEP, SOURCE_END_STEP],
        "schedule_prefix_sha256": common.canonical_sha256(
            schedule_payload[:updates]
        ),
        "full_96_step_schedule_sha256": full_schedule_sha256,
        "training_rows_reused_from_stage1": 0,
        "learning_rate": LEARNING_RATE,
        "learning_rate_multipliers": dict(LEARNING_RATE_MULTIPLIERS),
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_margin": MARGIN,
        "control_weights": CONTROL_WEIGHTS,
        "global_batch_size": stage1.GLOBAL_BATCH_SIZE,
        "local_rows": stage1.LOCAL_ROWS,
        "base_model": str(base_model.expanduser().resolve()),
        "base_model_revision": common.BASE_MODEL_REVISION,
        "stage1_result_receipt": STAGE1_RESULT_RECEIPT,
        "stage1_adapter_files_sha256": stage1_result["adapter_files_sha256"],
        "development_failure_receipt": DEVELOPMENT_RESULT_RECEIPT,
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "open_split_files": open_receipt["files"],
        "rank_devices": list(context.rank_devices),
        "model_audit": model_audit,
        "runner_sha256": common.sha256_file(RUNNER_FILE),
        "stage1_runner_sha256": common.sha256_file(Path(stage1.__file__)),
        "common_helper_sha256": common.sha256_file(Path(common.__file__)),
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    if stage1.DISTILL_TEACHER_ADAPTER is not None:
        if stage1.DISTILL_CACHE_ROOT is None:
            raise ValueError("Distillation cache is required")
        input_binding["distillation"] = {
            "teacher_adapter": str(
                stage1.DISTILL_TEACHER_ADAPTER.expanduser().resolve(strict=True)
            ),
            "teacher_adapter_files_sha256": contrast.gate._sha256_json(
                contrast.gate.snapshot_directory_files(stage1.DISTILL_TEACHER_ADAPTER)
            ),
            "cache_root": str(stage1.DISTILL_CACHE_ROOT.expanduser().resolve(strict=True)),
            "cache_files_sha256": contrast.gate._sha256_json(
                contrast.gate.snapshot_directory_files(stage1.DISTILL_CACHE_ROOT)
            ),
            "weight": float(stage1.DISTILL_WEIGHT),
            "temperature": float(stage1.DISTILL_TEMPERATURE),
            "top_k": int(stage1.DISTILL_TOP_K),
            "teacher_mode": "projected_kv_slots",
        }
    distributed.require_consensus(
        context,
        common.canonical_sha256(input_binding),
        description="recurrent-routed stage2 input binding",
    )
    if context.is_primary:
        stage1.write_fresh_json(resolved_output / "input_binding.json", input_binding)

    trained = stage1.train(
        model,
        tokenizer,
        schedule,
        schedule_payload,
        updates=updates,
        context=context,
        output_dir=resolved_output,
        named_trainable=named_trainable,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        margin=MARGIN,
        control_weights=CONTROL_WEIGHTS,
        learning_rate_multipliers=LEARNING_RATE_MULTIPLIERS,
        protocol_payload_sha256=PROTOCOL_PAYLOAD_SHA256,
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": trained["peak_cuda_memory_bytes"],
        },
    )
    passed = (
        stage1.four_distinct_a100s(context.rank_devices)
        and trained["route_subset_changed"] is True
        and trained["trainable_subset_changed"] is True
        and trained["maximum_global_inactive_parameter_tensors"] == 0
        and trained["projected_carrier_fixed_every_row"] is True
        and trained["first_update_joint_routing_gradient_audit"]["passed"] is True
    )
    result = {}
    save_error = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, delta_config)
            adapter_files = contrast.gate.snapshot_directory_files(adapter_dir)
            result = {
                "schema": SCHEMA,
                "status": (
                    PREFLIGHT_STATUS
                    if updates == PREFLIGHT_UPDATES and passed
                    else TRAINING_STATUS
                    if passed
                    else FAILURE_STATUS
                ),
                "passed": passed,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "input_binding": input_binding,
                "training": trained,
                "adapter_files": adapter_files,
                "adapter_files_sha256": contrast.gate._sha256_json(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_development_evaluation_authorized": (
                    passed and updates == TRAIN_UPDATES
                ),
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": common.canonical_sha256(result),
            }
            stage1.write_fresh_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="recurrent-routed-stage2-result-save",
        error=save_error,
    )
    del model, tokenizer, rows_by_task
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "passed": passed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, 64))
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Stage-2 post-training requires four-rank torchrun")
    try:
        result = run(
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
