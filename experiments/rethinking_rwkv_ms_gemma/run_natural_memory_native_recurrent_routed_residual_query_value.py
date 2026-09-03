#!/usr/bin/env python3
"""Post-train a zero-initialized additive recurrent residual over fixed V9."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.distributed.elastic.multiprocessing.errors import record

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import freeze_non_delta_mem_params  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    recurrent_routed_posttrain_common as common,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_posttrain as training,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_recurrent_routed_query_value_broad as broad,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain.v1"
INPUT_SCHEMA = "rwkv_ms_recurrent_routed_residual_query_value_posttrain_input.v1"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_recurrent_routed_residual_query_value_protocol_v1.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "d4c8ee2baff9468bfa655009f47a73f456706bb992bda4389e7c55b9fb03e915"
)
MODE = "recurrent_routed_residual_query_value"
HYBRID_GAIN = 0.125
SEED = 20260831
PREFLIGHT_UPDATES = 2
TRAIN_UPDATES = 32
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
MARGIN = 0.02
MAX_SOURCE_USER_CHARACTERS = 1450
BASELINE_ANCHOR_WEIGHT = 4.0
BASELINE_ANCHOR_TEMPERATURE = 1.0
BASELINE_ANCHOR_TOP_K = 64
PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = False
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.0625,
    "matched_donor_recurrent_state": 0.5,
    "slot_shuffled_recurrent_state": 0.25,
    "layer_permuted_recurrent_state": 0.25,
}
TARGET_COUNTS = {
    "attribution": 16,
    "narrative": 32,
    "scene": 16,
}
V9_ROOT = broad.V9_ROOT
V9_RESULT_RECEIPT = broad.V9_RESULT_RECEIPT
V2_ROOT = broad.V2_ROOT
V2_MANIFEST_RECEIPT = broad.V2_MANIFEST_RECEIPT
BROAD_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/recurrent_routed_query_value_broad_capped_train32_v2"
)
BROAD_RESULT_RECEIPT = (
    "220456a389675ed42d63cd825490f99459cdc8be2af0771e979f40554ff63f08"
)
ORIGINAL_BROAD_BUILD_SCHEDULE = broad.build_schedule

RWKV_CORE_SUFFIXES = broad.RWKV_CORE_SUFFIXES
RESIDUAL_SUFFIXES = (
    ".rwkv_route_query_proj",
    ".rwkv_route_state_proj",
    ".rwkv_recurrent_value_proj",
    ".rwkv_pair_value_proj",
)
TRAINABLE_SUFFIXES = RWKV_CORE_SUFFIXES + RESIDUAL_SUFFIXES
FIRST_STEP_ZERO_ALLOWED = frozenset(
    set(RWKV_CORE_SUFFIXES)
    | {".rwkv_route_query_proj", ".rwkv_route_state_proj"}
)
LEARNING_RATE_MULTIPLIERS = {
    **{suffix: 0.01 for suffix in RWKV_CORE_SUFFIXES},
    ".rwkv_route_query_proj": 0.1,
    ".rwkv_route_state_proj": 0.1,
    ".rwkv_recurrent_value_proj": 1.0,
    ".rwkv_pair_value_proj": 1.0,
}


def load_v2_manifest() -> Mapping[str, Any]:
    return broad.load_v2_manifest()


def validate_protocol(
    updates: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    v9_result = common.validate_signed_json(
        V9_ROOT / "result.json",
        V9_RESULT_RECEIPT,
    )
    broad_result = common.validate_signed_json(
        BROAD_ROOT / "result.json",
        BROAD_RESULT_RECEIPT,
    )
    frozen = protocol.get("frozen_inputs", {})
    expected_frozen = {
        "base_model_revision": common.BASE_MODEL_REVISION,
        "base_model_weights_sha256": common.BASE_MODEL_WEIGHTS_SHA256,
        "base_config_sha256": common.BASE_CONFIG_SHA256,
        "tokenizer_sha256": common.TOKENIZER_SHA256,
        "v9_result_receipt": V9_RESULT_RECEIPT,
        "v9_adapter_weights_sha256": common.WARMSTART_WEIGHTS_SHA256,
        "v9_adapter_config_sha256": common.WARMSTART_CONFIG_SHA256,
        "excluded_broad_training_receipt": BROAD_RESULT_RECEIPT,
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "final_commitment_receipt": common.FINAL_COMMITMENT_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "development_v2_manifest_receipt": V2_MANIFEST_RECEIPT,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    expected_architecture = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "rwkv_ms_hybrid_mode": MODE,
        "rwkv_ms_hybrid_gain": HYBRID_GAIN,
        "projected_carrier": "fixed V9 write, route, value, and fusion path",
        "recurrent_residual": "zero-initialized additive query-state value",
        "dynamic_pair_gate": False,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "baseline_fallback": False,
        "benchmark_time_parameter_override": False,
    }
    expected_training = {
        "hardware": "exactly four distinct NVIDIA A100 GPUs",
        "hf_endpoint": common.HF_MIRROR_ENDPOINT,
        "seed": SEED,
        "optimizer": "fused AdamW with fresh moments",
        "learning_rate": LEARNING_RATE,
        "learning_rate_multipliers": LEARNING_RATE_MULTIPLIERS,
        "weight_decay": 0.0,
        "max_gradient_norm": MAX_GRAD_NORM,
        "preflight_optimizer_updates": PREFLIGHT_UPDATES,
        "optimizer_updates": TRAIN_UPDATES,
        "global_batch_rows": training.GLOBAL_BATCH_SIZE,
        "local_rows_per_rank": training.LOCAL_ROWS,
        "target_source_rows_by_task": TARGET_COUNTS,
        "prompt_variants_per_target": 4,
        "max_source_user_characters": MAX_SOURCE_USER_CHARACTERS,
        "preflight_schedule": "maximum estimated row characters first",
        "cuda_allocator": "expandable_segments:True",
        "contrast_margin": MARGIN,
        "control_weights": CONTROL_WEIGHTS,
        "baseline_anchor": {
            "weight": BASELINE_ANCHOR_WEIGHT,
            "temperature": BASELINE_ANCHOR_TEMPERATURE,
            "top_k": BASELINE_ANCHOR_TOP_K,
            "teacher_condition": "zero_recurrent_state",
        },
        "final_rows_opened_during_training": False,
    }
    if PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES:
        expected_training[
            "preflight_requires_all_trainable_families_changed"
        ] = True
    frozen_negligible = globals().get(
        "FROZEN_NEGLIGIBLE_GRADIENT_SUFFIXES",
        frozenset(),
    )
    if frozen_negligible:
        expected_training["frozen_negligible_gradient_suffixes"] = sorted(
            frozen_negligible
        )
    if (
        any(frozen.get(key) != value for key, value in expected_frozen.items())
        or any(
            protocol.get("architecture", {}).get(key) != value
            for key, value in expected_architecture.items()
        )
        or any(
            protocol.get("training", {}).get(key) != value
            for key, value in expected_training.items()
        )
        or protocol.get("trainable_parameter_suffixes")
        != list(TRAINABLE_SUFFIXES)
        or updates not in {PREFLIGHT_UPDATES, TRAIN_UPDATES}
        or v9_result.get("status") != "training_complete_evaluation_pending"
        or broad_result.get("passed") is not True
        or broad_result.get("final_rows_opened") is not False
    ):
        raise ValueError("Residual query-value protocol or lineage differs")
    load_v2_manifest()
    return protocol, v9_result


def _parameter_family(name: str) -> str | None:
    matching = [suffix for suffix in TRAINABLE_SUFFIXES if name.endswith(suffix)]
    if len(matching) > 1:
        raise RuntimeError(f"Residual parameter matches multiple families: {name}")
    return matching[0] if matching else None


def configure_trainable_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], Mapping[str, Any]]:
    freeze_non_delta_mem_params(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_parameter_family(name) is not None)
    runtime._promote_trainable_parameters_to_fp32(model)
    selected = distributed.stable_named_parameters(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    )
    names = [name for name, _ in selected]
    family_counts = Counter(_parameter_family(name) for name in names)
    expected_tensors = common.EXPECTED_LAYERS * len(TRAINABLE_SUFFIXES)
    modules = [module for _, module in common.iter_delta_mem_modules(model)]
    residuals_zero = all(
        torch.equal(
            parameter.detach(),
            torch.zeros_like(parameter.detach()),
        )
        for module in modules
        for parameter in (
            module.rwkv_recurrent_value_proj,
            module.rwkv_pair_value_proj,
        )
    )
    routes_identity = all(
        torch.equal(
            module.rwkv_route_query_proj.detach().cpu(),
            torch.eye(module.rank),
        )
        and torch.equal(
            module.rwkv_route_state_proj.detach().cpu(),
            torch.eye(module.rank),
        )
        for module in modules
    )
    projected_carrier_trainables = [
        name
        for name in names
        if name.endswith(
            (
                ".projected_kv_key_proj",
                ".memory_v_proj",
                ".delta_q_proj",
                ".delta_o_proj",
                ".memory_fusion_hidden_weight",
                ".memory_fusion_read_weight",
                ".memory_fusion_bias",
                ".delta_scale_raw",
            )
        )
    ]
    passed = bool(
        len(selected) == expected_tensors
        and len(modules) == common.EXPECTED_LAYERS
        and set(family_counts) == set(TRAINABLE_SUFFIXES)
        and all(
            family_counts[suffix] == common.EXPECTED_LAYERS
            for suffix in TRAINABLE_SUFFIXES
        )
        and all(parameter.dtype == torch.float32 for _, parameter in selected)
        and residuals_zero
        and routes_identity
        and not projected_carrier_trainables
    )
    audit = {
        "parameter_tensors": len(selected),
        "expected_parameter_tensors": expected_tensors,
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": common.canonical_sha256(names),
        "family_tensor_counts": dict(sorted(family_counts.items())),
        "trainable_parameter_suffixes": list(TRAINABLE_SUFFIXES),
        "all_trainable_fp32": all(
            parameter.dtype == torch.float32 for _, parameter in selected
        ),
        "zero_initialized_residual_tensors": common.EXPECTED_LAYERS * 2,
        "residuals_exactly_zero": residuals_zero,
        "routes_exactly_identity": routes_identity,
        "projected_carrier_trainable_tensors": projected_carrier_trainables,
        "initial_checkpoint_is_v9_plus_exact_zero_residual": bool(
            residuals_zero and routes_identity and not projected_carrier_trainables
        ),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Residual trainable isolation failed: {audit!r}")
    return selected, audit


def audit_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    families = {
        suffix: audit_gradient_family(named_trainable, suffix)
        for suffix in TRAINABLE_SUFFIXES
    }
    return {
        "families": families,
        "audited_parameter_families": len(families),
        "first_step_zero_allowed_for_exact_baseline_initialization": sorted(
            FIRST_STEP_ZERO_ALLOWED
        ),
        "passed": all(family["passed"] for family in families.values()),
    }


def audit_gradient_family(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    suffix: str,
) -> Mapping[str, Any]:
    rows = []
    for name, parameter in named_trainable:
        if not name.endswith(suffix):
            continue
        gradient = parameter.grad
        finite = gradient is not None and bool(
            torch.isfinite(gradient).all().item()
        )
        norm = (
            0.0
            if gradient is None or not finite
            else float(gradient.detach().float().norm().item())
        )
        rows.append(
            {
                "name": name,
                "gradient_present": gradient is not None,
                "gradient_finite": finite,
                "gradient_l2_norm": norm,
                "gradient_nonzero": norm > 0.0,
            }
        )
    require_nonzero = suffix not in FIRST_STEP_ZERO_ALLOWED
    passed = bool(
        len(rows) == common.EXPECTED_LAYERS
        and all(row["gradient_present"] for row in rows)
        and all(row["gradient_finite"] for row in rows)
        and (
            not require_nonzero
            or all(row["gradient_nonzero"] for row in rows)
        )
    )
    return {
        "suffix": suffix,
        "parameter_tensors": len(rows),
        "gradient_present_tensors": sum(
            bool(row["gradient_present"]) for row in rows
        ),
        "gradient_nonzero_tensors": sum(
            bool(row["gradient_nonzero"]) for row in rows
        ),
        "requires_nonzero_on_first_update": require_nonzero,
        "minimum_l2_norm": min(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "maximum_l2_norm": max(
            (float(row["gradient_l2_norm"]) for row in rows),
            default=0.0,
        ),
        "parameter_names_sha256": common.canonical_sha256(
            [row["name"] for row in rows]
        ),
        "passed": passed,
    }


def row_user_content(row: common.SourceRow) -> str:
    return str(json.loads(row.raw_line)["messages"][1]["content"])


def build_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]]]:
    prior_schedule, _ = ORIGINAL_BROAD_BUILD_SCHEDULE(rows_by_task)
    prior_targets = {row.target.row_sha256 for row in prior_schedule}
    development = load_v2_manifest()["development_source_ordinals"]
    available = {}
    for task in common.TASKS:
        unique_rows: dict[str, common.SourceRow] = {}
        for row in rows_by_task[task]:
            if (
                row.row_sha256 in prior_targets
                or row.source_ordinal in set(development[task])
                or row.user_characters > MAX_SOURCE_USER_CHARACTERS
            ):
                continue
            current = unique_rows.get(row.row_sha256)
            if current is None or row.source_ordinal < current.source_ordinal:
                unique_rows[row.row_sha256] = row
        available[task] = list(unique_rows.values())
    selected_targets: dict[str, tuple[common.SourceRow, ...]] = {}
    for task in common.TASKS:
        ordered = sorted(
            available[task],
            key=lambda row: hashlib.sha256(
                f"residual-query-value-v1:{task}:{row.row_sha256}".encode()
            ).hexdigest(),
        )
        selected_targets[task] = tuple(ordered[: TARGET_COUNTS[task]])
        if len(selected_targets[task]) != TARGET_COUNTS[task]:
            raise RuntimeError(f"Residual schedule lacks target rows for {task}")

    donors: dict[str, common.SourceRow] = {}
    for task in common.TASKS:
        for target in selected_targets[task]:
            candidates = [
                row
                for row in available[task]
                if row.source_ordinal != target.source_ordinal
                and row.assistant_identity != target.assistant_identity
            ]
            donors[target.row_sha256] = max(
                candidates,
                key=lambda row: (
                    SequenceMatcher(
                        None,
                        row_user_content(target),
                        row_user_content(row),
                    ).ratio(),
                    -abs(row.user_characters - target.user_characters),
                    row.row_sha256,
                ),
            )

    remaining = {task: list(selected_targets[task]) for task in common.TASKS}
    targets: list[common.SourceRow] = []
    while any(remaining.values()):
        task = max(
            (value for value in common.TASKS if remaining[value]),
            key=lambda value: (
                len(remaining[value]) / TARGET_COUNTS[value],
                hashlib.sha256(
                    f"residual-task-order:{len(targets)}:{value}".encode()
                ).hexdigest(),
            ),
        )
        targets.append(remaining[task].pop(0))
    if len(targets) != TRAIN_UPDATES * 2:
        raise RuntimeError("Residual schedule target count differs")

    target_pairs = [targets[index : index + 2] for index in range(0, len(targets), 2)]
    target_pairs.sort(
        key=lambda pair: (
            max(
                target.user_characters + donors[target.row_sha256].user_characters
                for target in pair
            ),
            sum(
                target.user_characters + donors[target.row_sha256].user_characters
                for target in pair
            ),
            common.canonical_sha256([target.row_sha256 for target in pair]),
        ),
        reverse=True,
    )

    schedule: list[common.ScheduledRow] = []
    payload: list[dict[str, Any]] = []
    for step, pair in enumerate(target_pairs, start=1):
        expanded = [
            (target, donors[target.row_sha256], variant)
            for target in pair
            for variant in range(4)
        ]
        expanded.sort(
            key=lambda item: hashlib.sha256(
                f"residual-step:{step}:{item[0].row_sha256}:{item[2]}".encode()
            ).hexdigest()
        )
        step_rows = []
        for position, (target, donor, variant) in enumerate(expanded):
            schedule.append(
                common.ScheduledRow(
                    step=step,
                    position=position,
                    target=target,
                    donor=donor,
                    prompt_variant=variant,
                )
            )
            step_rows.append(
                {
                    "position": position,
                    "task": target.task,
                    "source_ordinal": target.source_ordinal,
                    "source_row_sha256": target.row_sha256,
                    "donor_source_ordinal": donor.source_ordinal,
                    "donor_row_sha256": donor.row_sha256,
                    "prompt_variant": variant,
                }
            )
        payload.append(
            {
                "step": step,
                "rows": step_rows,
                "payload_sha256": common.canonical_sha256(step_rows),
            }
        )
    if len(schedule) != TRAIN_UPDATES * training.GLOBAL_BATCH_SIZE:
        raise RuntimeError("Residual schedule global row count differs")
    return tuple(schedule), payload


def configure() -> None:
    common.HYBRID_MODE = MODE
    common.HYBRID_GAIN = HYBRID_GAIN
    broad.SCHEMA = SCHEMA
    broad.INPUT_SCHEMA = INPUT_SCHEMA
    broad.RUNNER_FILE = Path(__file__)
    broad.PREFLIGHT_STATUS = "residual_query_value_preflight_passed"
    broad.TRAINING_STATUS = (
        "residual_query_value_training_complete_development_evaluation_authorized"
    )
    broad.FAILURE_STATUS = (
        "residual_query_value_training_failed_development_evaluation_blocked"
    )
    broad.PROTOCOL = PROTOCOL
    broad.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    broad.MODE = MODE
    broad.SEED = SEED
    broad.PREFLIGHT_UPDATES = PREFLIGHT_UPDATES
    broad.TRAIN_UPDATES = TRAIN_UPDATES
    broad.LEARNING_RATE = LEARNING_RATE
    broad.MAX_GRAD_NORM = MAX_GRAD_NORM
    broad.MARGIN = MARGIN
    broad.CONTROL_WEIGHTS = CONTROL_WEIGHTS
    broad.TARGET_COUNTS = TARGET_COUNTS
    broad.MAX_SOURCE_USER_CHARACTERS = MAX_SOURCE_USER_CHARACTERS
    broad.BASELINE_ANCHOR_WEIGHT = BASELINE_ANCHOR_WEIGHT
    broad.BASELINE_ANCHOR_TEMPERATURE = BASELINE_ANCHOR_TEMPERATURE
    broad.BASELINE_ANCHOR_TOP_K = BASELINE_ANCHOR_TOP_K
    broad.PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = (
        PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES
    )
    broad.TRAINABLE_SUFFIXES = TRAINABLE_SUFFIXES
    broad.FIRST_STEP_ZERO_ALLOWED = FIRST_STEP_ZERO_ALLOWED
    broad.LEARNING_RATE_MULTIPLIERS = LEARNING_RATE_MULTIPLIERS
    broad.validate_protocol = validate_protocol
    broad.configure_broad_trainable_parameters = configure_trainable_parameters
    broad.audit_broad_gradients = audit_gradients
    broad.build_schedule = build_schedule


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
    if context is None:
        raise ValueError("Residual post-training requires four-rank torchrun")
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
