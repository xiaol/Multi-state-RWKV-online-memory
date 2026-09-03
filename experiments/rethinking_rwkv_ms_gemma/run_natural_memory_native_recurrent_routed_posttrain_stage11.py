#!/usr/bin/env python3
"""Run the higher-gain recurrent-routing continuation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import recurrent_routed_posttrain_common as common
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain_stage10 as stage10
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_posttrain_stage2 as runner


PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_posttrain_stage11_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "037ddeff4ce64cca9279ff292b3862af7ad452fe47b117fca732fa9c29263548"
PREDECESSOR_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_stage2_train64_v1"
PREDECESSOR_RESULT_RECEIPT = "01ada5458eca9c1f53987862585bba71c3fc7c2832dd737bf77cb095f479e712"
PREDECESSOR_ADAPTER_CONFIG_SHA256 = "dc20cbe794479c9f802b45bedef83ff65543445dd85bfdb36419396cadd95d37"
V2_ROOT = SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
V2_MANIFEST_RECEIPT = "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
TRAIN_UPDATES = 20
PREFLIGHT_UPDATES = 1
HYBRID_GAIN = 0.25
ORIGINAL_TRAIN = runner.stage1.train
ORIGINAL_LOAD_ADAPTER = runner.load_delta_mem_adapter


def load_v2_manifest():
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if not isinstance(receipt, dict) or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT or common.canonical_sha256(value) != V2_MANIFEST_RECEIPT:
        raise ValueError("Development-v2 manifest differs")
    value["receipt"] = receipt
    return value


def validate_lineage():
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    predecessor = common.validate_signed_json(PREDECESSOR_ROOT / "result.json", PREDECESSOR_RESULT_RECEIPT)
    if predecessor.get("status") != "stage2_training_complete_development_evaluation_authorized" or predecessor.get("passed") is not True or predecessor.get("final_rows_opened") is not False or common.sha256_file(PREDECESSOR_ROOT / "adapter/delta_mem_config.json") != PREDECESSOR_ADAPTER_CONFIG_SHA256 or load_v2_manifest().get("final_rows_opened") is not False:
        raise ValueError("Stage-11 lineage differs")
    return protocol, predecessor


def load_predecessor_adapter(model, path):
    modules = common.ordered_modules(model)
    for _, module in modules:
        module.delta_config = replace(module.delta_config, rwkv_ms_hybrid_gain=0.125)
        module.rwkv_ms_hybrid_gain = 0.125
    ORIGINAL_LOAD_ADAPTER(model, path)
    for _, module in modules:
        module.delta_config = replace(module.delta_config, rwkv_ms_hybrid_gain=HYBRID_GAIN)
        module.rwkv_ms_hybrid_gain = HYBRID_GAIN
    return common.build_config()


def configure_runner() -> None:
    common.HYBRID_GAIN = HYBRID_GAIN
    runner.SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage11.v1"
    runner.INPUT_SCHEMA = "rwkv_ms_natural_memory_native_recurrent_routed_posttrain_stage11_input.v1"
    runner.PREFLIGHT_STATUS = "stage11_preflight_passed"
    runner.TRAINING_STATUS = "stage11_training_complete_development_v2_evaluation_authorized"
    runner.FAILURE_STATUS = "stage11_training_failed_development_v2_evaluation_blocked"
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
    runner.SOURCE_END_STEP = 116
    runner.TRAIN_UPDATES = TRAIN_UPDATES
    runner.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    runner.LEARNING_RATE = stage10.LEARNING_RATE
    runner.MAX_GRAD_NORM = stage10.MAX_GRAD_NORM
    runner.MARGIN = stage10.MARGIN
    runner.CONTROL_WEIGHTS = stage10.CONTROL_WEIGHTS
    runner.validate_lineage = validate_lineage
    runner.stage2_schedule = stage10.stage10_schedule
    runner.stage1.train = stage10.train_with_always_on_donor
    runner.load_delta_mem_adapter = load_predecessor_adapter


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
        raise ValueError("Stage-11 post-training requires four-rank torchrun")
    try:
        result = runner.run(context=context, output_dir=args.output_dir, updates=args.updates, base_model=args.base_model)
    finally:
        runner.stage1.train = ORIGINAL_TRAIN
        runner.load_delta_mem_adapter = ORIGINAL_LOAD_ADAPTER
        distributed.destroy_distributed_training(context)
    print(json.dumps({"rank": context.process_rank, "status": result["status"], "passed": result["passed"], "result_receipt": result.get("receipt", {}).get("payload_sha256") if context.is_primary else None}, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
