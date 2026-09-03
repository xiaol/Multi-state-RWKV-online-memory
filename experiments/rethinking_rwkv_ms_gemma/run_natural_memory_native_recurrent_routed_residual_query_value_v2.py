#!/usr/bin/env python3
"""Post-train the narrative-focused ungated residual query-value model."""

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


SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain.v2"
INPUT_SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain_input.v2"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_recurrent_routed_residual_query_value_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "5688755fccd84b2d777a1d9774a8da7b564c574c5be0ed5dd685ac5e7811e3e2"
)
SEED = 20260901
PREFLIGHT_UPDATES = 3
TRAIN_UPDATES = 32
PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = True
TARGET_COUNTS = {"attribution": 8, "narrative": 48, "scene": 8}
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 1.0,
    "slot_shuffled_recurrent_state": 0.25,
    "layer_permuted_recurrent_state": 0.5,
}
BASELINE_ANCHOR_WEIGHT = 2.0
MAX_SOURCE_USER_CHARACTERS = 1451
FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES = frozenset(
    {
        ".hrm_rwkv7_core.x_w",
        ".hrm_rwkv7_core.x_a",
        ".hrm_rwkv7_core.x_g",
    }
)
TRAINABLE_SUFFIXES = tuple(
    suffix
    for suffix in base.TRAINABLE_SUFFIXES
    if suffix not in FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
)
FIRST_STEP_ZERO_ALLOWED = frozenset(
    suffix
    for suffix in base.FIRST_STEP_ZERO_ALLOWED
    if suffix not in FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
)
LEARNING_RATE_MULTIPLIERS = {
    **{
        suffix: 0.01
        for suffix in base.RWKV_CORE_SUFFIXES
        if suffix not in FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES
    },
    ".rwkv_route_query_proj": 0.1,
    ".rwkv_route_state_proj": 0.1,
    ".rwkv_recurrent_value_proj": 1.0,
    ".rwkv_pair_value_proj": 1.0,
}
ORIGINAL_PRIOR_BUILD_SCHEDULE = base.ORIGINAL_BROAD_BUILD_SCHEDULE
PRIOR_TARGET_COUNTS = {"attribution": 16, "narrative": 24, "scene": 24}
PRIOR_TRAIN_UPDATES = 32
PRIOR_MAX_SOURCE_USER_CHARACTERS = 1400


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
    broad.PREFLIGHT_STATUS = "residual_query_value_v2_preflight_passed"
    broad.TRAINING_STATUS = (
        "residual_query_value_v2_training_complete_development_evaluation_authorized"
    )
    broad.FAILURE_STATUS = (
        "residual_query_value_v2_training_failed_development_evaluation_blocked"
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
        raise ValueError("Residual v2 post-training requires exactly four ranks")
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
