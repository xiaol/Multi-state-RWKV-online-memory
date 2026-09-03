#!/usr/bin/env python3
"""Run a non-promoting causal diagnostic on the rejected residual checkpoint."""

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
    run_natural_memory_native_recurrent_routed_residual_query_value as residual,
)


TRAINING_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_residual_query_value_train32_v2"
)
TRAINING_RESULT_RECEIPT = (
    "916cbad3fe7509b91132cf87c8dad77bfeb95726541e88bf9aaedaaa03cff7c0"
)
TRAINING_STATUS = (
    "residual_query_value_training_failed_development_evaluation_blocked"
)


def configure() -> None:
    v2.configure()
    common.HYBRID_MODE = residual.MODE
    common.HYBRID_GAIN = residual.HYBRID_GAIN
    evaluator.TRAINED_SUFFIXES = residual.TRAINABLE_SUFFIXES
    evaluator.SCHEMA = (
        "rwkv_ms_recurrent_routed_residual_query_value_development_diagnostic.v1"
    )
    evaluator.INPUT_SCHEMA = (
        "rwkv_ms_recurrent_routed_residual_query_value_development_diagnostic_input.v1"
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
        raise ValueError("Residual diagnostic requires exactly four ranks")
    try:
        result = evaluator.run(
            context=context,
            output_dir=args.output_dir,
            training_root=args.training_root.expanduser().resolve(strict=True),
            base_model=args.base_model,
            training_result_receipt=TRAINING_RESULT_RECEIPT,
            training_status=TRAINING_STATUS,
            protocol_file=residual.PROTOCOL,
            protocol_receipt=residual.PROTOCOL_PAYLOAD_SHA256,
            diagnostic_only=True,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(
        json.dumps(
            {
                "rank": context.process_rank,
                "status": result["status"],
                "passed": result["passed"],
                "diagnostic_causal_criteria_passed": result.get(
                    "diagnostic_causal_criteria_passed"
                ),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
