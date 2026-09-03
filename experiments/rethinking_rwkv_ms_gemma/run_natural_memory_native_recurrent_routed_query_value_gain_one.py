#!/usr/bin/env python3
"""Continue query-value routing with a bounded unit recurrent gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from torch.distributed.elastic.multiprocessing.errors import record

from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_native_recurrent_routed_query_value_narrative_repair as base


PROTOCOL = SCRIPT_DIR / "natural_memory_native_recurrent_routed_query_value_gain_one_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "c4a7653657ff6f26918d147575b1c221f2be4f5a0ea7c1a4a023a75ed4053861"
HYBRID_GAIN = 1.0
ORIGINAL_TRAIN = base.ORIGINAL_TRAIN


def configure() -> None:
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    base.HYBRID_GAIN = HYBRID_GAIN
    base.configure()
    base.stage2.RUNNER_FILE = Path(__file__)
    base.stage2.PREFLIGHT_STATUS = "query_value_gain_one_preflight_passed"
    base.stage2.TRAINING_STATUS = "query_value_gain_one_training_complete_development_v2_evaluation_authorized"
    base.stage2.FAILURE_STATUS = "query_value_gain_one_training_failed_development_v2_evaluation_blocked"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True, choices=(1, base.TRAIN_UPDATES))
    parser.add_argument("--base-model", type=Path, default=base.common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    global PROTOCOL_PAYLOAD_SHA256
    if PROTOCOL_PAYLOAD_SHA256 == "PLACEHOLDER":
        raise ValueError("Gain-one protocol receipt is not installed")
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Gain-one post-training requires four-rank torchrun")
    try:
        result = base.stage2.run(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
        )
    finally:
        base.stage2.stage1.train = ORIGINAL_TRAIN
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
