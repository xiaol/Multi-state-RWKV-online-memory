#!/usr/bin/env python3
"""Evaluate broad query-value causal controls on development-v2."""

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
    evaluate_natural_memory_native_recurrent_routed_posttrain_development_v2 as v2,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_query_value_broad as broad,
)


TRAINING_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_query_value_broad_capped_train32_v2"
)
TRAINING_RESULT_RECEIPT = (
    "220456a389675ed42d63cd825490f99459cdc8be2af0771e979f40554ff63f08"
)
TRAINING_STATUS = (
    "broad_training_complete_development_evaluation_authorized"
)


def configure() -> None:
    common.HYBRID_MODE = broad.MODE
    evaluator.TRAINED_SUFFIXES = broad.TRAINABLE_SUFFIXES
    evaluator.SCHEMA = "rwkv_ms_recurrent_routed_query_value_broad_development.v1"
    evaluator.INPUT_SCHEMA = (
        "rwkv_ms_recurrent_routed_query_value_broad_development_input.v1"
    )
    v2.configure()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument(
        "--training-result-receipt",
        default=TRAINING_RESULT_RECEIPT,
    )
    parser.add_argument("--base-model", type=Path, default=common.BASE_MODEL)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Broad development evaluation requires four-rank torchrun")
    try:
        result = evaluator.run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=args.training_result_receipt,
            training_status=TRAINING_STATUS,
            protocol_file=broad.PROTOCOL,
            protocol_receipt=broad.PROTOCOL_PAYLOAD_SHA256,
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
