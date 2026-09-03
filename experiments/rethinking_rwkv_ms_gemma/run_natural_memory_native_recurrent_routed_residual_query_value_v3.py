#!/usr/bin/env python3
"""Post-train the balanced ungated residual query-value model."""

from __future__ import annotations

import argparse
import json
import os
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
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_query_value_broad as broad,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_residual_query_value as base,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_residual_query_value_v2 as v2,
)


SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain.v3"
INPUT_SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain_input.v3"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_recurrent_routed_residual_query_value_protocol_v3.json"
)
PROTOCOL_PAYLOAD_SHA256 = "eb7b5dd9e8c1f31a4ed0cafddd67f52a09bcc4f19f6b61deac34094745426ed3"
SEED = 20260902
PREFLIGHT_UPDATES = 3
TRAIN_UPDATES = 32
PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = True
TARGET_COUNTS = {"attribution": 16, "narrative": 32, "scene": 16}
CONTROL_WEIGHTS = dict(v2.CONTROL_WEIGHTS)
BASELINE_ANCHOR_WEIGHT = v2.BASELINE_ANCHOR_WEIGHT
MAX_SOURCE_USER_CHARACTERS = v2.MAX_SOURCE_USER_CHARACTERS
FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES = (
    v2.FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
)
TRAINABLE_SUFFIXES = v2.TRAINABLE_SUFFIXES
FIRST_STEP_ZERO_ALLOWED = v2.FIRST_STEP_ZERO_ALLOWED
LEARNING_RATE_MULTIPLIERS = dict(v2.LEARNING_RATE_MULTIPLIERS)
ORIGINAL_PRIOR_BUILD_SCHEDULE = v2.ORIGINAL_PRIOR_BUILD_SCHEDULE
PRIOR_TARGET_COUNTS = dict(v2.PRIOR_TARGET_COUNTS)
PRIOR_TRAIN_UPDATES = v2.PRIOR_TRAIN_UPDATES
PRIOR_MAX_SOURCE_USER_CHARACTERS = v2.PRIOR_MAX_SOURCE_USER_CHARACTERS


def build_locked_prior_schedule(rows_by_task):
    current = (
        broad.TARGET_COUNTS,
        broad.TRAIN_UPDATES,
        broad.MAX_SOURCE_USER_CHARACTERS,
    )
    broad.TARGET_COUNTS = PRIOR_TARGET_COUNTS
    broad.TRAIN_UPDATES = PRIOR_TRAIN_UPDATES
    broad.MAX_SOURCE_USER_CHARACTERS = PRIOR_MAX_SOURCE_USER_CHARACTERS
    try:
        return ORIGINAL_PRIOR_BUILD_SCHEDULE(rows_by_task)
    finally:
        (
            broad.TARGET_COUNTS,
            broad.TRAIN_UPDATES,
            broad.MAX_SOURCE_USER_CHARACTERS,
        ) = current


def configure() -> None:
    base.SCHEMA = SCHEMA
    base.INPUT_SCHEMA = INPUT_SCHEMA
    base.PROTOCOL = PROTOCOL
    base.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    base.SEED = SEED
    base.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    base.TRAIN_UPDATES = TRAIN_UPDATES
    base.PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = (
        PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES
    )
    base.TARGET_COUNTS = TARGET_COUNTS
    base.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    base.BASELINE_ANCHOR_WEIGHT = BASELINE_ANCHOR_WEIGHT
    base.MAX_SOURCE_USER_CHARACTERS = MAX_SOURCE_USER_CHARACTERS
    base.FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES = (
        FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
    )
    base.TRAINABLE_SUFFIXES = TRAINABLE_SUFFIXES
    base.FIRST_STEP_ZERO_ALLOWED = FIRST_STEP_ZERO_ALLOWED
    base.LEARNING_RATE_MULTIPLIERS = LEARNING_RATE_MULTIPLIERS
    base.ORIGINAL_BROAD_BUILD_SCHEDULE = build_locked_prior_schedule
    base.configure()
    broad.RUNNER_FILE = Path(__file__)
    broad.PREFLIGHT_STATUS = "residual_query_value_v3_preflight_passed"
    broad.TRAINING_STATUS = (
        "residual_query_value_v3_training_complete_development_evaluation_authorized"
    )
    broad.FAILURE_STATUS = (
        "residual_query_value_v3_training_failed_development_evaluation_blocked"
    )


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
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}")
    args = parse_args(argv)
    configure()
    context = distributed.initialize_distributed_training(
        args.device,
        timeout_seconds=7200,
    )
    if context is None or context.world_size != 4:
        raise ValueError("Residual v3 post-training requires exactly four ranks")
    try:
        result = broad.run(
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
