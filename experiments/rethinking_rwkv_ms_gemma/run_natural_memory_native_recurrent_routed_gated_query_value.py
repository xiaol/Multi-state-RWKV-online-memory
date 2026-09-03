#!/usr/bin/env python3
"""Post-train recurrent query-value routing with a learned abstention gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    run_natural_memory_native_recurrent_routed_posttrain_stage2 as stage2,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_query_value_distill as distill,
)


MODE = "recurrent_routed_gated_query_value"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_recurrent_routed_gated_query_value_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "1cffaf898a3c4c77ea87bc3ec92b8d1095d7d397a354ed9633f5b4f6d041fb7b"
)
PREDECESSOR_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_query_value_distill_blend50_v2"
)
PREDECESSOR_RESULT_RECEIPT = (
    "75fc307f2c5db320c0cbca4f7bc92c0829db5a842cf0f052a6b536d3b0016206"
)
PREDECESSOR_ADAPTER_WEIGHTS_SHA256 = (
    "cffdb73e4ac0f86265b82cceecae211b5f62bd52bad875209794b93357c3dfbd"
)
PREDECESSOR_ADAPTER_CONFIG_SHA256 = (
    "4484ad76ff6523626a9ee11bb04d40723e35d90443d0b58c41c4f8a4e652c84b"
)
PREDECESSOR_DEVELOPMENT_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_query_value_distill_blend50_development_v2"
)
PREDECESSOR_DEVELOPMENT_RECEIPT = (
    "224e7e86ecdf9d3673542e182e6420600a7cbec3daed1c4308fc6dcca33af015"
)
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 0.1
MARGIN = 0.05
HYBRID_GAIN = 1.0
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.125,
}
ALWAYS_ACTIVE_CONTROLS = tuple(CONTROL_WEIGHTS)
LEARNING_RATE_MULTIPLIERS = {
    ".rwkv_pair_gate_weight": 10000.0,
    ".rwkv_pair_gate_bias": 10000.0,
}
ORIGINAL_TRAIN = stage1.train


def validate_lineage() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(
        PREDECESSOR_ROOT / "result.json",
        PREDECESSOR_RESULT_RECEIPT,
    )
    development = common.validate_signed_json(
        PREDECESSOR_DEVELOPMENT_ROOT / "result.json",
        PREDECESSOR_DEVELOPMENT_RECEIPT,
    )
    if (
        predecessor.get("status")
        != "adapter_blend_complete_development_v2_evaluation_authorized"
        or predecessor.get("passed") is not True
        or predecessor.get("final_rows_opened") is not False
        or development.get("status")
        != "development_passed_final_evaluation_authorized"
        or development.get("passed") is not True
        or development.get("publisher_test_opened") is not False
        or common.sha256_file(
            PREDECESSOR_ROOT / "adapter/delta_mem_adapter.pt"
        )
        != PREDECESSOR_ADAPTER_WEIGHTS_SHA256
        or common.sha256_file(
            PREDECESSOR_ROOT / "adapter/delta_mem_config.json"
        )
        != PREDECESSOR_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Gated query-value predecessor lineage differs")
    return protocol, predecessor


def load_predecessor_adapter(model: Any, input_dir: Path) -> Any:
    state = torch.load(
        input_dir / "delta_mem_adapter.pt",
        map_location="cpu",
        weights_only=True,
    )
    load_delta_mem_state_dict(
        model,
        state,
        initialize_missing_rwkv_pair_gate=True,
    )
    return common.build_config()


def train_with_all_controls(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    kwargs = dict(kwargs)
    kwargs["always_active_controls"] = ALWAYS_ACTIVE_CONTROLS
    return ORIGINAL_TRAIN(*args, **kwargs)


def configure() -> None:
    distill.configure()
    common.HYBRID_MODE = MODE
    common.HYBRID_GAIN = HYBRID_GAIN
    common.PROTOCOL = PROTOCOL
    common.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    stage2.SCHEMA = (
        "rwkv_ms_natural_memory_native_recurrent_routed_gated_query_value.v1"
    )
    stage2.INPUT_SCHEMA = (
        "rwkv_ms_natural_memory_native_recurrent_routed_gated_query_value_input.v1"
    )
    stage2.PREFLIGHT_STATUS = "gated_query_value_preflight_passed"
    stage2.TRAINING_STATUS = (
        "gated_query_value_training_complete_development_evaluation_authorized"
    )
    stage2.FAILURE_STATUS = (
        "gated_query_value_training_failed_development_evaluation_blocked"
    )
    stage2.RUNNER_FILE = Path(__file__)
    stage2.PROTOCOL = PROTOCOL
    stage2.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    stage2.STAGE1_ROOT = PREDECESSOR_ROOT
    stage2.STAGE1_RESULT_RECEIPT = PREDECESSOR_RESULT_RECEIPT
    stage2.STAGE1_ADAPTER_WEIGHTS_SHA256 = PREDECESSOR_ADAPTER_WEIGHTS_SHA256
    stage2.STAGE1_ADAPTER_CONFIG_SHA256 = PREDECESSOR_ADAPTER_CONFIG_SHA256
    stage2.DEVELOPMENT_ROOT = PREDECESSOR_DEVELOPMENT_ROOT
    stage2.DEVELOPMENT_RESULT_RECEIPT = PREDECESSOR_DEVELOPMENT_RECEIPT
    stage2.SOURCE_START_STEP = distill.SOURCE_START_STEP
    stage2.SOURCE_END_STEP = distill.SOURCE_END_STEP
    stage2.TRAIN_UPDATES = TRAIN_UPDATES
    stage2.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    stage2.LEARNING_RATE = LEARNING_RATE
    stage2.LEARNING_RATE_MULTIPLIERS = LEARNING_RATE_MULTIPLIERS
    stage2.MAX_GRAD_NORM = MAX_GRAD_NORM
    stage2.MARGIN = MARGIN
    stage2.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    stage2.validate_lineage = validate_lineage
    stage2.stage2_schedule = distill.stage14_schedule
    stage2.load_delta_mem_adapter = load_predecessor_adapter
    stage2.stage1.train = train_with_all_controls


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--updates",
        type=int,
        required=True,
        choices=(PREFLIGHT_UPDATES, TRAIN_UPDATES),
    )
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if PROTOCOL_PAYLOAD_SHA256 == "PLACEHOLDER":
        raise ValueError("Gated query-value protocol receipt is not installed")
    configure()
    stage1.DISTILL_TEACHER_ADAPTER = distill.TEACHER_ADAPTER
    stage1.DISTILL_BASE_MODEL = args.base_model
    stage1.DISTILL_CACHE_ROOT = distill.DISTILL_CACHE_ROOT
    stage1.DISTILL_WEIGHT = distill.DISTILL_WEIGHT
    stage1.DISTILL_TEMPERATURE = distill.DISTILL_TEMPERATURE
    stage1.DISTILL_TOP_K = distill.DISTILL_TOP_K
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None:
        raise ValueError("Gated query-value training requires four-rank torchrun")
    try:
        result = stage2.run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        stage2.stage1.train = ORIGINAL_TRAIN
        stage1.DISTILL_TEACHER_ADAPTER = None
        stage1.DISTILL_BASE_MODEL = None
        stage1.DISTILL_CACHE_ROOT = None
        stage1.DISTILL_WEIGHT = 0.0
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
