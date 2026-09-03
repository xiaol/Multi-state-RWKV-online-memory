#!/usr/bin/env python3
"""Broadly post-train one ungated recurrent query-value checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    freeze_non_delta_mem_params,
    save_delta_mem_adapter,
)
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
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_recurrent_routed_query_value_broad_posttrain.v2"
INPUT_SCHEMA = "rwkv_ms_recurrent_routed_query_value_broad_posttrain_input.v2"
RUNNER_FILE = Path(__file__)
PREFLIGHT_STATUS = "broad_preflight_passed"
TRAINING_STATUS = "broad_training_complete_development_evaluation_authorized"
FAILURE_STATUS = "broad_training_failed_development_evaluation_blocked"
PROTOCOL = (
    SCRIPT_DIR
    / "natural_memory_native_recurrent_routed_query_value_broad_protocol_v2.json"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "4a5d47643ba5958dbfc16cfcd21e7d5ff4cf0e84eaf0dc85bb02f2bcb61c6d76"
)
MODE = "recurrent_routed_query_value"
WORLD_SIZE = 4
SEED = 20260831
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 32
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
MARGIN = 0.05
CONTROL_WEIGHTS = {
    "zero_recurrent_state": 0.125,
    "matched_donor_recurrent_state": 0.25,
    "slot_shuffled_recurrent_state": 0.125,
    "layer_permuted_recurrent_state": 0.25,
}
TARGET_COUNTS = {
    "attribution": 16,
    "narrative": 24,
    "scene": 24,
}
MAX_SOURCE_USER_CHARACTERS = 1400
BASELINE_ANCHOR_WEIGHT = 0.0
BASELINE_ANCHOR_TEMPERATURE = 1.0
BASELINE_ANCHOR_TOP_K = 64
PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES = False
V9_ROOT = (
    SCRIPT_DIR
    / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9"
)
V9_RESULT_RECEIPT = (
    "6b53092a7020a10d995495b4496a88509d015c9ea01321f6216b28bdc90b4e31"
)
V2_ROOT = (
    SCRIPT_DIR / "local_artifacts/recurrent_routed_posttrain_development_v2"
)
V2_MANIFEST_RECEIPT = (
    "2236d1e3e980ce92787e34500a40a38634ea7017835e629759d9564ba99036d6"
)

RWKV_CORE_SUFFIXES = (
    ".hrm_rwkv7_core.x_r",
    ".hrm_rwkv7_core.x_w",
    ".hrm_rwkv7_core.x_k",
    ".hrm_rwkv7_core.x_v",
    ".hrm_rwkv7_core.x_a",
    ".hrm_rwkv7_core.x_g",
    ".hrm_rwkv7_core.w1",
    ".hrm_rwkv7_core.w2",
    ".hrm_rwkv7_core.w0",
    ".hrm_rwkv7_core.a1",
    ".hrm_rwkv7_core.a2",
    ".hrm_rwkv7_core.a0",
    ".hrm_rwkv7_core.g1",
    ".hrm_rwkv7_core.g2",
    ".hrm_rwkv7_core.k_k",
    ".hrm_rwkv7_core.k_a",
    ".hrm_rwkv7_core.receptance.weight",
    ".hrm_rwkv7_core.key.weight",
    ".hrm_rwkv7_core.value.weight",
    ".hrm_rwkv7_core.output.weight",
    ".hrm_rwkv7_core.ln_x.weight",
)
ACTIVE_PATH_SUFFIXES = (
    ".memory_v_proj",
    ".beta_proj",
    ".beta_bias",
    ".rwkv_route_query_proj",
    ".rwkv_route_state_proj",
    ".rwkv_pair_value_proj",
    ".delta_q_proj",
    ".delta_o_proj",
    ".memory_fusion_hidden_weight",
    ".memory_fusion_read_weight",
    ".memory_fusion_bias",
    ".delta_scale_raw",
)
TRAINABLE_SUFFIXES = RWKV_CORE_SUFFIXES + ACTIVE_PATH_SUFFIXES
FIRST_STEP_ZERO_ALLOWED = frozenset(
    {
        ".hrm_rwkv7_core.x_w",
        ".hrm_rwkv7_core.w2",
        ".hrm_rwkv7_core.x_a",
        ".hrm_rwkv7_core.a2",
        ".hrm_rwkv7_core.x_g",
        ".hrm_rwkv7_core.g2",
    }
)
LEARNING_RATE_MULTIPLIERS = {
    **{
        suffix: 0.05
        for suffix in RWKV_CORE_SUFFIXES
        if suffix != ".hrm_rwkv7_core.output.weight"
    },
    ".hrm_rwkv7_core.output.weight": 0.25,
    ".memory_v_proj": 0.05,
    ".beta_proj": 0.05,
    ".beta_bias": 0.05,
    ".rwkv_route_query_proj": 1.0,
    ".rwkv_route_state_proj": 1.0,
    ".rwkv_pair_value_proj": 1.0,
    ".delta_q_proj": 0.025,
    ".delta_o_proj": 0.025,
    ".memory_fusion_hidden_weight": 0.025,
    ".memory_fusion_read_weight": 0.025,
    ".memory_fusion_bias": 0.025,
    ".delta_scale_raw": 0.025,
}


def load_v2_manifest() -> Mapping[str, Any]:
    value = json.loads((V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    receipt = value.pop("receipt", None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("payload_sha256") != V2_MANIFEST_RECEIPT
        or common.canonical_sha256(value) != V2_MANIFEST_RECEIPT
        or value.get("final_rows_opened") is not False
    ):
        raise ValueError("Development-v2 manifest differs")
    value["receipt"] = receipt
    return value


def validate_protocol(updates: int) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = common.validate_signed_json(PROTOCOL, PROTOCOL_PAYLOAD_SHA256)
    v9_result = common.validate_signed_json(
        V9_ROOT / "result.json",
        V9_RESULT_RECEIPT,
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
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "final_commitment_receipt": common.FINAL_COMMITMENT_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "development_v2_manifest_receipt": V2_MANIFEST_RECEIPT,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
    }
    architecture = protocol.get("architecture", {})
    expected_architecture = {
        "memory_backend": "rwkv_ms",
        "memory_readout_mode": "projected_kv_rwkv_hybrid",
        "rwkv_ms_hybrid_mode": MODE,
        "rwkv_pair_gate": False,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "baseline_fallback": False,
        "benchmark_time_parameter_override": False,
    }
    train_contract = protocol.get("training", {})
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
        "final_rows_opened_during_training": False,
    }
    if BASELINE_ANCHOR_WEIGHT > 0.0:
        expected_training["baseline_anchor"] = {
            "weight": BASELINE_ANCHOR_WEIGHT,
            "temperature": BASELINE_ANCHOR_TEMPERATURE,
            "top_k": BASELINE_ANCHOR_TOP_K,
        }
    if (
        any(frozen.get(key) != value for key, value in expected_frozen.items())
        or any(
            architecture.get(key) != value
            for key, value in expected_architecture.items()
        )
        or any(
            train_contract.get(key) != value
            for key, value in expected_training.items()
        )
        or protocol.get("trainable_parameter_suffixes")
        != list(TRAINABLE_SUFFIXES)
        or updates not in {PREFLIGHT_UPDATES, TRAIN_UPDATES}
        or v9_result.get("status") != "training_complete_evaluation_pending"
        or common.sha256_file(V9_ROOT / "adapter/delta_mem_adapter.pt")
        != common.WARMSTART_WEIGHTS_SHA256
        or common.sha256_file(V9_ROOT / "adapter/delta_mem_config.json")
        != common.WARMSTART_CONFIG_SHA256
    ):
        raise ValueError("Broad query-value protocol or V9 lineage differs")
    load_v2_manifest()
    return protocol, v9_result


def _parameter_family(name: str) -> str | None:
    matching = [suffix for suffix in TRAINABLE_SUFFIXES if name.endswith(suffix)]
    if len(matching) > 1:
        raise RuntimeError(f"Trainable parameter matches multiple families: {name}")
    return matching[0] if matching else None


def configure_broad_trainable_parameters(
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
    dead_families = (
        ".projected_kv_key_proj",
        ".memory_q_proj",
        ".memory_k_proj",
        ".delta_k_proj",
        ".delta_v_proj",
        ".rwkv_pair_gate_weight",
        ".rwkv_pair_gate_bias",
        ".hrm_rwkv7_core.ln_x.bias",
    )
    dead_selected = [
        name for name in names if name.endswith(dead_families)
    ]
    passed = bool(
        len(selected) == expected_tensors
        and set(family_counts) == set(TRAINABLE_SUFFIXES)
        and all(
            family_counts[suffix] == common.EXPECTED_LAYERS
            for suffix in TRAINABLE_SUFFIXES
        )
        and all(parameter.dtype == torch.float32 for _, parameter in selected)
        and not dead_selected
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
        "dead_or_pair_gate_tensors": dead_selected,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Broad trainable isolation failed: {audit!r}")
    return selected, audit


def _audit_gradient_family(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    suffix: str,
) -> Mapping[str, Any]:
    rows = []
    for name, parameter in named_trainable:
        if not name.endswith(suffix):
            continue
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
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


def audit_broad_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    families = {
        suffix: _audit_gradient_family(named_trainable, suffix)
        for suffix in TRAINABLE_SUFFIXES
    }
    return {
        "families": families,
        "audited_parameter_families": len(families),
        "zero_allowed_only_for_zero_initialized_lora_staging": sorted(
            FIRST_STEP_ZERO_ALLOWED
        ),
        "passed": all(family["passed"] for family in families.values()),
    }


def family_parameter_hashes(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, str]:
    hashes = {}
    for suffix in TRAINABLE_SUFFIXES:
        state = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in named_trainable
            if name.endswith(suffix)
        }
        if len(state) != common.EXPECTED_LAYERS:
            raise RuntimeError(f"Broad family topology differs: {suffix}")
        hashes[suffix] = runtime._state_dict_sha256(state)
    return hashes


def build_schedule(
    rows_by_task: Mapping[str, Sequence[common.SourceRow]],
) -> tuple[tuple[common.ScheduledRow, ...], list[dict[str, Any]]]:
    development_ordinals = load_v2_manifest()["development_source_ordinals"]
    available = {
        task: [
            row
            for row in rows_by_task[task]
            if row.source_ordinal not in set(development_ordinals[task])
            and row.user_characters <= MAX_SOURCE_USER_CHARACTERS
        ]
        for task in common.TASKS
    }
    selected_targets = {}
    for task in common.TASKS:
        ordered = sorted(
            available[task],
            key=lambda row: hashlib.sha256(
                f"broad-query-value-v1:{task}:{row.row_sha256}".encode("utf-8")
            ).hexdigest(),
        )
        selected_targets[task] = tuple(ordered[: TARGET_COUNTS[task]])
        if len(selected_targets[task]) != TARGET_COUNTS[task]:
            raise RuntimeError(f"Broad schedule lacks target rows for {task}")

    donors: dict[str, common.SourceRow] = {}
    for task in common.TASKS:
        for target in selected_targets[task]:
            candidates = [
                row
                for row in available[task]
                if row.source_ordinal != target.source_ordinal
                and row.assistant_identity != target.assistant_identity
            ]
            if not candidates:
                raise RuntimeError(f"Broad target has no donor: {target}")
            donors[target.row_sha256] = min(
                candidates,
                key=lambda row: (
                    abs(row.user_characters - target.user_characters),
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
                    f"broad-task-order:{len(targets)}:{value}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        targets.append(remaining[task].pop(0))
    if len(targets) != TRAIN_UPDATES * 2:
        raise RuntimeError("Broad schedule target count differs")

    target_pairs = [
        targets[index : index + 2]
        for index in range(0, len(targets), 2)
    ]
    target_pairs.sort(
        key=lambda pair: (
            max(
                target.user_characters
                + donors[target.row_sha256].user_characters
                for target in pair
            ),
            sum(
                target.user_characters
                + donors[target.row_sha256].user_characters
                for target in pair
            ),
            common.canonical_sha256(
                [target.row_sha256 for target in pair]
            ),
        ),
        reverse=True,
    )

    schedule: list[common.ScheduledRow] = []
    payload: list[dict[str, Any]] = []
    for step in range(1, TRAIN_UPDATES + 1):
        expanded = [
            (target, donors[target.row_sha256], prompt_variant)
            for target in target_pairs[step - 1]
            for prompt_variant in range(4)
        ]
        expanded.sort(
            key=lambda item: hashlib.sha256(
                (
                    f"broad-step:{step}:{item[0].row_sha256}:"
                    f"{item[2]}"
                ).encode("utf-8")
            ).hexdigest()
        )
        step_rows = []
        for position, (target, donor, prompt_variant) in enumerate(expanded):
            schedule.append(
                common.ScheduledRow(
                    step=step,
                    position=position,
                    target=target,
                    donor=donor,
                    prompt_variant=prompt_variant,
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
                    "prompt_variant": prompt_variant,
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
        raise RuntimeError("Broad schedule global row count differs")
    return tuple(schedule), payload


def snapshot_directory(path: Path) -> Mapping[str, Mapping[str, Any]]:
    return {
        str(file.relative_to(path)): {
            "bytes": file.stat().st_size,
            "sha256": common.sha256_file(file),
        }
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Broad post-training requires exactly four ranks")
    if os.environ.get("HF_ENDPOINT") != common.HF_MIRROR_ENDPOINT:
        raise ValueError(
            f"HF_ENDPOINT must be exactly {common.HF_MIRROR_ENDPOINT}"
        )
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError(
            "PYTORCH_CUDA_ALLOC_CONF must be exactly expandable_segments:True"
        )
    protocol, v9_result = validate_protocol(updates)
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
        phase="broad-query-value-output-creation",
        error=creation_error,
    )

    set_seed(SEED)
    common.HYBRID_MODE = MODE
    model, tokenizer, delta_config, model_audit = common.load_model(
        base_model,
        device=context.device,
        trainable=True,
        configure_trainables=configure_broad_trainable_parameters,
    )
    named_trainable = model_audit.pop("named_trainable")
    rows_by_task = common.load_open_rows("train", manifest=manifest)
    schedule, schedule_payload = build_schedule(rows_by_task)
    initial_family_hashes = family_parameter_hashes(named_trainable)
    input_binding = {
        "schema": INPUT_SCHEMA,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_objective": protocol["objective"],
        "seed": SEED,
        "updates": updates,
        "world_size": context.world_size,
        "rank_devices": list(context.rank_devices),
        "global_batch_size": training.GLOBAL_BATCH_SIZE,
        "local_rows": training.LOCAL_ROWS,
        "schedule_prefix_sha256": common.canonical_sha256(
            schedule_payload[:updates]
        ),
        "full_schedule_sha256": common.canonical_sha256(schedule_payload),
        "target_source_rows_by_task": TARGET_COUNTS,
        "max_source_user_characters": MAX_SOURCE_USER_CHARACTERS,
        "preflight_schedule": "maximum estimated row characters first",
        "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "learning_rate": LEARNING_RATE,
        "learning_rate_multipliers": LEARNING_RATE_MULTIPLIERS,
        "max_gradient_norm": MAX_GRAD_NORM,
        "contrast_margin": MARGIN,
        "control_weights": CONTROL_WEIGHTS,
        "baseline_anchor": {
            "weight": BASELINE_ANCHOR_WEIGHT,
            "temperature": BASELINE_ANCHOR_TEMPERATURE,
            "top_k": BASELINE_ANCHOR_TOP_K,
            "teacher_condition": "zero_recurrent_state",
        },
        "base_model": str(base_model.expanduser().resolve()),
        "base_model_revision": common.BASE_MODEL_REVISION,
        "v9_result_receipt": V9_RESULT_RECEIPT,
        "v9_adapter_files": v9_result["adapter_files"],
        "split_manifest_receipt": common.SPLIT_MANIFEST_RECEIPT,
        "open_split_receipt": common.OPEN_SPLIT_RECEIPT,
        "open_split_files": open_receipt["files"],
        "development_v2_manifest_receipt": V2_MANIFEST_RECEIPT,
        "model_audit": model_audit,
        "task_router": False,
        "template_matcher": False,
        "dual_pass_selector": False,
        "benchmark_specific_decoder": False,
        "baseline_fallback": False,
        "benchmark_time_parameter_override": False,
        "final_rows_opened": False,
        "publisher_validation_opened": False,
        "publisher_test_opened": False,
        "runner_sha256": common.sha256_file(RUNNER_FILE),
        "common_helper_sha256": common.sha256_file(Path(common.__file__)),
    }
    distributed.require_consensus(
        context,
        common.canonical_sha256(input_binding),
        description="broad query-value input binding",
    )
    if context.is_primary:
        training.write_fresh_json(
            resolved_output / "input_binding.json",
            input_binding,
        )

    trained = training.train(
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
        gradient_audit_fn=audit_broad_gradients,
        baseline_anchor_weight=BASELINE_ANCHOR_WEIGHT,
        baseline_anchor_temperature=BASELINE_ANCHOR_TEMPERATURE,
        baseline_anchor_top_k=BASELINE_ANCHOR_TOP_K,
    )
    final_family_hashes = family_parameter_hashes(named_trainable)
    family_changed = {
        suffix: initial_family_hashes[suffix] != final_family_hashes[suffix]
        for suffix in TRAINABLE_SUFFIXES
    }
    all_families_changed = all(family_changed.values())
    family_change_requirement_passed = bool(
        all_families_changed
        or (
            updates != TRAIN_UPDATES
            and not PREFLIGHT_REQUIRE_ALL_FAMILY_CHANGES
        )
    )
    rank_runtime = distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "peak_cuda_memory_bytes": trained["peak_cuda_memory_bytes"],
        },
    )
    passed = bool(
        training.four_distinct_a100s(context.rank_devices)
        and trained["route_subset_changed"] is True
        and trained["trainable_subset_changed"] is True
        and trained["maximum_global_inactive_parameter_tensors"] == 0
        and trained["projected_carrier_fixed_every_row"] is True
        and trained["first_update_joint_routing_gradient_audit"]["passed"]
        is True
        and family_change_requirement_passed
    )
    result: dict[str, Any] = {}
    save_error = None
    if context.is_primary:
        try:
            adapter_dir = resolved_output / "adapter"
            save_delta_mem_adapter(model, adapter_dir, delta_config)
            adapter_files = snapshot_directory(adapter_dir)
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
                "trainable_family_changes": {
                    "initial_hashes": initial_family_hashes,
                    "final_hashes": final_family_hashes,
                    "changed": family_changed,
                    "all_required_families_changed": all_families_changed,
                    "requirement_passed": family_change_requirement_passed,
                },
                "adapter_files": adapter_files,
                "adapter_files_sha256": common.canonical_sha256(adapter_files),
                "rank_runtime": list(rank_runtime),
                "open_development_evaluation_authorized": (
                    passed and updates == TRAIN_UPDATES
                ),
                "benchmark_evaluation_uses_saved_checkpoint_only": True,
                "final_rows_opened": False,
                "publisher_validation_opened": False,
                "publisher_test_opened": False,
                "code_bindings": {
                    "runner_sha256": common.sha256_file(RUNNER_FILE),
                    "common_helper_sha256": common.sha256_file(
                        Path(common.__file__)
                    ),
                    "training_helper_sha256": common.sha256_file(
                        Path(training.__file__)
                    ),
                    "protocol_file_sha256": common.sha256_file(PROTOCOL),
                    "delta_impl_sha256": common.sha256_file(
                        PROJECT_ROOT / "deltamem/core/delta_impl.py"
                    ),
                    "rwkv_core_sha256": common.sha256_file(
                        PROJECT_ROOT / "deltamem/core/hrm_rwkv7.py"
                    ),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": common.canonical_sha256(result),
            }
            training.write_fresh_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="broad-query-value-result-save",
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
    if PROTOCOL_PAYLOAD_SHA256 == "PLACEHOLDER":
        raise ValueError("Broad query-value protocol receipt is not installed")
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Broad post-training requires four-rank torchrun")
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
