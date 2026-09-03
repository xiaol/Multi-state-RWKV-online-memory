#!/usr/bin/env python3
"""Evaluate the signed residual-v3 checkpoint on causal development controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_recurrent_routed_posttrain_development as evaluator,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    evaluate_natural_memory_native_recurrent_routed_posttrain_development_v2 as development_v2,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_residual_query_value as residual,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_residual_query_value_v3 as residual_v3,
)


TRAINING_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_residual_query_value_v3_train32_v1"
)
TRAINING_RESULT_RECEIPT = (
    "b48ea22238a2c60d35dfd54dc3aa26a6ac3bf37d7c0608a5cd009bdf29811609"
)
TRAINING_STATUS = (
    "residual_query_value_v3_training_complete_development_evaluation_authorized"
)


def configure() -> None:
    residual_v3.configure()
    development_v2.HYBRID_MODE = residual.MODE
    development_v2.SLOT_SHUFFLE_EXPECTATION = "invariance"
    development_v2.configure()
    evaluator.TRAINED_SUFFIXES = residual_v3.TRAINABLE_SUFFIXES
    evaluator.SCHEMA = (
        "rwkv_ms_recurrent_routed_residual_query_value_development.v3"
    )
    evaluator.INPUT_SCHEMA = (
        "rwkv_ms_recurrent_routed_residual_query_value_development_input.v3"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None or context.world_size != 4:
        raise ValueError("Residual-v3 development requires exactly four ranks")
    try:
        result = evaluator.run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=TRAINING_RESULT_RECEIPT,
            training_status=TRAINING_STATUS,
            protocol_file=residual_v3.PROTOCOL,
            protocol_receipt=residual_v3.PROTOCOL_PAYLOAD_SHA256,
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
