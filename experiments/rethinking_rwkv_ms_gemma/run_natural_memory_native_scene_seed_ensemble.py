#!/usr/bin/env python3
"""Train one locked robust native-scene gate seed on four GPU ranks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    reset_delta_mem_states,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_gate as gate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_training_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_training_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_seed_training_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_seed_ensemble_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "3294adb35ec7438e9f8e9f69e3fd825701e7480a9fb7722975a9ef0d8029ebd9"
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 16
LOCAL_ROWS = 4
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 16
SEEDS = (17, 29, 43)
EXPECTED_ELIGIBLE_ROWS = 1435
MAX_DONOR_TOKEN_DELTA = 16
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
LEARNING_RATE = 5e-5
MAX_GRAD_NORM = 0.1
POST_STEP_DELTA_RETENTION = 0.995
TRAIN_SALT_PREFIX = "rwkv-ms-native-scene-robust-seed-v1:"
DROP_SALT_PREFIX = "rwkv-ms-native-scene-robust-drop-v1:"
SEED_BINDINGS = {
    17: {
        "selected_rows_payload_sha256": "b366ea915f0b6a6649ee0d5c35026b882f5d36b30a8a08030127c24432914a86",
        "schedule_payload_sha256": "a05ed538aa1f220f7ea6ae398738ce32b0197ed5d7e4435b5db3711ec87454ae",
    },
    29: {
        "selected_rows_payload_sha256": "9c0e546a191c436183093731df47ee1bc95540b75db49a80b463f8d43414f2c1",
        "schedule_payload_sha256": "d17a8dff6cda43c766913ee6f9cb7d58b35c487861806048cfbe021ceea1ca17",
    },
    43: {
        "selected_rows_payload_sha256": "1fce8276918871453ee07550caaf348eab7eafac8ce88bc079f455aea17ad9a5",
        "schedule_payload_sha256": "f5713a96c6b5a5dc816c1ce2271c0b8f04231c06c675087dc66d3fafaadca13f",
    },
}


@dataclass(frozen=True)
class RobustScheduleStep:
    step: int
    source_ordinals: tuple[int, ...]
    donor_ordinals: tuple[int, ...]
    no_state_ordinals: frozenset[int]
    payload_sha256: str


def canonical_sha256(value: Any) -> str:
    return contrast.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return contrast.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Seed-ensemble protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Seed-ensemble protocol hash differs")
    return value


def build_schedule(
    rows: Sequence[contrast.SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
    *,
    seed: int,
) -> tuple[tuple[RobustScheduleStep, ...], list[dict[str, Any]]]:
    if seed not in SEED_BINDINGS:
        raise ValueError(f"Unsupported robust seed: {seed}")
    eligible = [
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= MAX_DONOR_TOKEN_DELTA
    ]
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("Seed-ensemble eligible row count differs")
    train_salt = f"{TRAIN_SALT_PREFIX}{seed}:"
    selected = sorted(
        eligible,
        key=lambda source_ordinal: (
            hashlib.sha256(
                (train_salt + rows[source_ordinal].example.row_sha256).encode("utf-8")
            ).hexdigest(),
            source_ordinal,
        ),
    )[: TRAIN_UPDATES * GLOBAL_BATCH_SIZE]
    selected_hashes = [rows[index].example.row_sha256 for index in selected]
    if canonical_sha256(selected_hashes) != SEED_BINDINGS[seed][
        "selected_rows_payload_sha256"
    ]:
        raise ValueError("Seed-ensemble selected row hash differs")
    schedule: list[RobustScheduleStep] = []
    payload: list[dict[str, Any]] = []
    for offset in range(0, len(selected), GLOBAL_BATCH_SIZE):
        step = offset // GLOBAL_BATCH_SIZE + 1
        group = selected[offset : offset + GLOBAL_BATCH_SIZE]
        no_state = frozenset(
            sorted(
                group,
                key=lambda source_ordinal: (
                    hashlib.sha256(
                        (
                            f"{DROP_SALT_PREFIX}{seed}:{step}:"
                            + rows[source_ordinal].example.row_sha256
                        ).encode("utf-8")
                    ).hexdigest(),
                    source_ordinal,
                ),
            )[:4]
        )
        row_payload = [
            {
                "source_ordinal": source_ordinal,
                "source_row_sha256": rows[source_ordinal].example.row_sha256,
                "donor_ordinal": mapping[source_ordinal],
                "positive_condition": (
                    "no_state" if source_ordinal in no_state else "correct_state"
                ),
            }
            for source_ordinal in group
        ]
        step_payload = {"step": step, "rows": row_payload}
        payload.append(step_payload)
        schedule.append(
            RobustScheduleStep(
                step=step,
                source_ordinals=tuple(group),
                donor_ordinals=tuple(mapping[index] for index in group),
                no_state_ordinals=no_state,
                payload_sha256=canonical_sha256(step_payload),
            )
        )
    if canonical_sha256(payload) != SEED_BINDINGS[seed]["schedule_payload_sha256"]:
        raise ValueError("Seed-ensemble schedule hash differs")
    return tuple(schedule), payload


def backward_condition(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    no_state: bool,
    coefficient: float,
    dtype: torch.dtype,
) -> tuple[float, int, int]:
    if no_state:
        logits = contrast.checkpointed_read_only(model, batch, dtype=dtype)
        occupancy = 0
    else:
        write_audit, logits = evolution.checkpointed_native_write_read(
            model,
            batch,
            dtype=dtype,
        )
        occupancy = int(write_audit["occupied_rows"])
    loss_sum, tokens, chunks = evolution.checkpointed_native_answer_loss_sum_and_count(
        logits,
        batch.labels,
        chunk_tokens=contrast.CE_CHUNK_TOKENS,
    )
    mean_ce = loss_sum / tokens
    scaled = mean_ce * (coefficient / GLOBAL_BATCH_SIZE)
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Seed-ensemble signed loss is non-finite")
    scaled.backward()
    value = float(mean_ce.detach().float().item())
    reset_delta_mem_states(model)
    del logits, loss_sum, mean_ce, scaled
    return value, chunks, occupancy


def apply_proximal_shrinkage(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
    anchors: Mapping[str, torch.Tensor],
) -> Mapping[str, float]:
    before_sq = 0.0
    after_sq = 0.0
    with torch.no_grad():
        for name, parameter in named_trainable:
            anchor = anchors[name]
            before = parameter.detach().float() - anchor.float()
            before_sq += float(before.square().sum().item())
            updated = anchor.float() + POST_STEP_DELTA_RETENTION * before
            parameter.copy_(updated.to(dtype=parameter.dtype))
            after = parameter.detach().float() - anchor.float()
            after_sq += float(after.square().sum().item())
    before_l2 = math.sqrt(before_sq)
    after_l2 = math.sqrt(after_sq)
    if not math.isfinite(after_l2) or after_l2 <= 0.0 or after_l2 > before_l2 * 1.001:
        raise RuntimeError("Seed-ensemble proximal shrinkage audit failed")
    return {
        "delta_l2_before": before_l2,
        "delta_l2_after": after_l2,
        "observed_l2_retention": after_l2 / before_l2,
    }


def save_gate_patch(
    model: torch.nn.Module,
    *,
    output_dir: Path,
    seed: int,
    step: int,
    context: distributed.DistributedTrainingContext,
) -> Mapping[str, Any] | None:
    gate_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(name.endswith(f".{family}") for family in contrast.GATE_FAMILIES)
    }
    state_sha256 = runtime._state_dict_sha256(gate_state)
    distributed.require_consensus(
        context,
        state_sha256,
        description=f"seed-ensemble seed {seed} checkpoint {step} gate state",
    )
    value: dict[str, Any] | None = None
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            patch_path = checkpoint_dir / "gate_patch.pt"
            torch.save(
                {
                    "schema": PATCH_SCHEMA,
                    "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                    "source_adapter_files_sha256": contrast.V9_ADAPTER_FILES_SHA256,
                    "seed": seed,
                    "step": step,
                    "gate_state_sha256": state_sha256,
                    "state_dict": gate_state,
                },
                patch_path,
            )
            value = {
                "schema": PATCH_SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "source_adapter_files_sha256": contrast.V9_ADAPTER_FILES_SHA256,
                "seed": seed,
                "step": step,
                "gate_state_sha256": state_sha256,
                "parameter_tensors": len(gate_state),
                "parameter_elements": sum(tensor.numel() for tensor in gate_state.values()),
                "parameter_names_sha256": canonical_sha256(sorted(gate_state)),
                "patch_file": {
                    "path": str(patch_path),
                    "bytes": patch_path.stat().st_size,
                    "sha256": sha256_file(patch_path),
                },
            }
            value["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_manifest_without_receipt",
                "payload_sha256": canonical_sha256(value),
            }
            contrast._write_json(checkpoint_dir / "manifest.json", value)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase=f"seed-ensemble-seed-{seed}-checkpoint-{step}",
        error=save_error,
    )
    return value


def train(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[RobustScheduleStep],
    *,
    seed: int,
    updates: int,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    dtype: torch.dtype,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    initial_state = snapshot_delta_mem_weights(model)
    initial_gate_sha256 = contrast._state_subset_sha256(initial_state, gate_only=True)
    initial_non_gate_sha256 = contrast._state_subset_sha256(initial_state, gate_only=False)
    anchors = {
        name: parameter.detach().clone() for name, parameter in named_trainable
    }
    total_active = 0.0
    total_positive_ce = 0.0
    total_donor_ce = 0.0
    total_margin = 0.0
    minimum_gate_gradient_norm = math.inf
    shrinkage_audits: list[Mapping[str, float]] = []
    started = time.time()
    progress_path = output_dir / "training_progress.jsonl"
    final_manifest: Mapping[str, Any] | None = None
    for schedule_step in schedule[:updates]:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        if len(local_sources) != LOCAL_ROWS:
            raise RuntimeError("Seed-ensemble local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        local_metrics = [0.0] * 10
        for source_ordinal in local_sources:
            position = schedule_step.source_ordinals.index(source_ordinal)
            target = rows[source_ordinal].example
            donor = rows[schedule_step.donor_ordinals[position]].example
            no_state = source_ordinal in schedule_step.no_state_ordinals
            target_batch = evolution.collate_native_examples(
                [target],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            donor_batch = contrast.build_donor_batch(
                target_batch,
                donor,
                device=context.device,
            )
            evolution.release_native_row_allocator_cache(context.device)
            positive_probe_ce, positive_tokens = contrast.evaluate_condition_ce(
                model,
                target_batch,
                no_state=no_state,
                dtype=dtype,
            )
            evolution.release_native_row_allocator_cache(context.device)
            donor_probe_ce, donor_tokens = contrast.evaluate_condition_ce(
                model,
                donor_batch,
                no_state=False,
                dtype=dtype,
            )
            margin_value = donor_probe_ce - positive_probe_ce
            active = margin_value < MARGIN
            positive_coefficient = 1.0 + (CONTRAST_WEIGHT if active else 0.0)
            evolution.release_native_row_allocator_cache(context.device)
            _, chunks, occupancy = backward_condition(
                model,
                target_batch,
                no_state=no_state,
                coefficient=positive_coefficient,
                dtype=dtype,
            )
            local_metrics[8] += chunks
            local_metrics[9] += occupancy
            if active:
                evolution.release_native_row_allocator_cache(context.device)
                donor_train_ce, chunks, occupancy = backward_condition(
                    model,
                    donor_batch,
                    no_state=False,
                    coefficient=-CONTRAST_WEIGHT,
                    dtype=dtype,
                )
                if not math.isfinite(donor_train_ce):
                    raise RuntimeError("Seed-ensemble donor train CE is non-finite")
                local_metrics[8] += chunks
                local_metrics[9] += occupancy
            local_metrics[0] += positive_probe_ce
            local_metrics[1] += donor_probe_ce
            local_metrics[2] += margin_value
            local_metrics[3] += float(active)
            local_metrics[4] += float(no_state)
            local_metrics[5] += float(not no_state)
            local_metrics[6] += positive_tokens
            local_metrics[7] += donor_tokens
            del target_batch, donor_batch
            evolution.release_native_row_allocator_cache(context.device)
        scalar_tensor = gate._prepare_distributed_scalar_sums(context, local_metrics)
        metrics = gate._distributed_scalar_sums(context, scalar_tensor)
        if metrics[4] != 4 or metrics[5] != 12:
            raise RuntimeError("Seed-ensemble state-dropout balance differs")
        if updates == PREFLIGHT_UPDATES and metrics[3] < 1:
            raise RuntimeError("Seed-ensemble preflight found no active hinge")
        gradient_validation = distributed.validate_local_gradients(named_trainable)
        if gradient_validation["passed"] is not True:
            raise RuntimeError("Seed-ensemble produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        gate_gradient_audit = evolution.audit_content_gate_gradients(named_trainable)
        if gate_gradient_audit["passed"] is not True:
            raise RuntimeError("Seed-ensemble produced invalid gate gradients")
        minimum_gate_gradient_norm = min(
            minimum_gate_gradient_norm,
            float(gate_gradient_audit["minimum_family_l2_norm"]),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError("Seed-ensemble gradient norm is non-finite")
        optimizer.step()
        shrinkage = apply_proximal_shrinkage(named_trainable, anchors)
        shrinkage_audits.append(shrinkage)
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "seed": seed,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "world_size": context.world_size,
            "correct_state_rows": int(metrics[5]),
            "no_state_rows": int(metrics[4]),
            "active_hinge_rows": int(metrics[3]),
            "mean_positive_probe_ce": metrics[0] / GLOBAL_BATCH_SIZE,
            "mean_donor_probe_ce": metrics[1] / GLOBAL_BATCH_SIZE,
            "mean_donor_minus_positive_ce": metrics[2] / GLOBAL_BATCH_SIZE,
            "positive_target_tokens": int(metrics[6]),
            "donor_target_tokens": int(metrics[7]),
            "checkpointed_ce_chunks": int(metrics[8]),
            "written_condition_occupancy_rows": int(metrics[9]),
            "gradient_norm_before_clip": float(grad_norm.detach().float().item()),
            "gradient_collective_sha256": canonical_sha256(collective),
            "gate_gradient_audit": gate_gradient_audit,
            "proximal_shrinkage": shrinkage,
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
            "no_state_ordinals": sorted(schedule_step.no_state_ordinals),
        }
        if context.is_primary:
            contrast._append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "step": schedule_step.step,
                        "positive_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "donor_ce": round(metrics[1] / GLOBAL_BATCH_SIZE, 6),
                        "active": int(metrics[3]),
                        "delta_l2": round(shrinkage["delta_l2_after"], 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_positive_ce += metrics[0]
        total_donor_ce += metrics[1]
        total_margin += metrics[2]
        total_active += metrics[3]
        if schedule_step.step == updates:
            final_manifest = save_gate_patch(
                model,
                output_dir=output_dir,
                seed=seed,
                step=schedule_step.step,
                context=context,
            )
    final_state = snapshot_delta_mem_weights(model)
    final_gate_sha256 = contrast._state_subset_sha256(final_state, gate_only=True)
    final_non_gate_sha256 = contrast._state_subset_sha256(final_state, gate_only=False)
    distributed.require_consensus(
        context,
        final_gate_sha256,
        description=f"seed-ensemble seed {seed} final gate state",
    )
    distributed.require_consensus(
        context,
        final_non_gate_sha256,
        description=f"seed-ensemble seed {seed} final non-gate state",
    )
    if final_gate_sha256 == initial_gate_sha256:
        raise RuntimeError("Seed-ensemble training did not change the gate state")
    if final_non_gate_sha256 != initial_non_gate_sha256:
        raise RuntimeError("Seed-ensemble training changed frozen non-gate state")
    return {
        "seed": seed,
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": updates * GLOBAL_BATCH_SIZE,
        "mean_positive_probe_ce": total_positive_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_probe_ce": total_donor_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_minus_positive_ce": total_margin / (updates * GLOBAL_BATCH_SIZE),
        "active_hinge_fraction": total_active / (updates * GLOBAL_BATCH_SIZE),
        "minimum_gate_gradient_norm": minimum_gate_gradient_norm,
        "post_step_delta_retention": POST_STEP_DELTA_RETENTION,
        "minimum_observed_l2_retention": min(
            audit["observed_l2_retention"] for audit in shrinkage_audits
        ),
        "maximum_observed_l2_retention": max(
            audit["observed_l2_retention"] for audit in shrinkage_audits
        ),
        "initial_gate_state_sha256": initial_gate_sha256,
        "final_gate_state_sha256": final_gate_sha256,
        "initial_non_gate_state_sha256": initial_non_gate_sha256,
        "final_non_gate_state_sha256": final_non_gate_sha256,
        "non_gate_unchanged": True,
        "progress_sha256": sha256_file(progress_path) if context.is_primary else None,
        "checkpoint": final_manifest,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    seed: int,
    updates: int,
    base_model: Path = contrast.BASE_MODEL,
    adapter_path: Path = contrast.V9_ADAPTER,
    dataset_root: Path = contrast.NATIVE_DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Seed-ensemble training requires exactly four ranks")
    if seed not in SEEDS:
        raise ValueError("Seed-ensemble training seed differs")
    gate.configure_hf_mirror()
    validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError("Seed-ensemble output may not be a symbolic link")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Seed-ensemble output must be fresh: {resolved_output}")
    if sha256_file(base_model / "config.json") != contrast.BASE_CONFIG_SHA256:
        raise ValueError("Seed-ensemble base configuration hash differs")
    adapter_files = gate.snapshot_directory_files(adapter_path)
    if gate._sha256_json(adapter_files) != contrast.V9_ADAPTER_FILES_SHA256:
        raise ValueError("Seed-ensemble V9 adapter files hash differs")
    if (
        sha256_file(adapter_path / "delta_mem_adapter.pt")
        != contrast.V9_ADAPTER_WEIGHTS_SHA256
        or sha256_file(adapter_path / "delta_mem_config.json")
        != contrast.V9_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Seed-ensemble V9 adapter component hash differs")
    runtime.set_seed(seed)
    delta_config = evolution.build_evolution_delta_config(
        "shared_qo_content_gated_attention_output"
    )
    model, tokenizer, _, _, _ = gate._load_model_and_tokenizer(
        {"model": {"path": str(base_model)}},
        device=context.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        delta_config=delta_config,
    )
    loaded_config = load_delta_mem_adapter(model, adapter_path)
    if loaded_config.to_dict() != delta_config.to_dict():
        raise ValueError("Seed-ensemble V9 adapter configuration differs")
    if len(list(iter_delta_mem_modules(model))) != 42:
        raise ValueError("Seed-ensemble requires 42 wrapped layers")
    named_trainable, trainable_audit = contrast.configure_gate_only_training(model)
    rows = contrast.load_scene_rows(tokenizer, dataset_root)
    mapping, donor_deltas, donor_payload = contrast.build_donor_mapping(rows)
    if canonical_sha256(donor_payload) != contrast.DONOR_MAPPING_SHA256:
        raise ValueError("Seed-ensemble donor mapping hash differs")
    schedule, schedule_payload = build_schedule(
        rows,
        mapping,
        donor_deltas,
        seed=seed,
    )
    input_binding = {
        "schema": SCHEMA,
        "phase": "preflight" if updates == PREFLIGHT_UPDATES else "training",
        "seed": seed,
        "updates": updates,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "base_model": str(base_model),
        "base_config_sha256": contrast.BASE_CONFIG_SHA256,
        "adapter_path": str(adapter_path),
        "adapter_files": adapter_files,
        "adapter_files_sha256": contrast.V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "scene_file_sha256": contrast.SCENE_FILE_SHA256,
        "scene_rows": len(rows),
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "eligible_rows": sum(
            delta <= MAX_DONOR_TOKEN_DELTA for delta in donor_deltas.values()
        ),
        "full_schedule_payload_sha256": canonical_sha256(schedule_payload),
        "executed_schedule_payload_sha256": canonical_sha256(schedule_payload[:updates]),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "world_size": context.world_size,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "post_step_delta_retention": POST_STEP_DELTA_RETENTION,
        "trainable_audit": trainable_audit,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description=f"seed-ensemble seed {seed} input binding",
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
            contrast._write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase=f"seed-ensemble-seed-{seed}-output-creation",
        error=creation_error,
    )
    training = train(
        model,
        rows,
        schedule,
        seed=seed,
        updates=updates,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        dtype=torch.bfloat16,
        output_dir=resolved_output,
        named_trainable=named_trainable,
    )
    result: dict[str, Any] = {}
    save_error: BaseException | None = None
    if context.is_primary:
        try:
            result = {
                "schema": SCHEMA,
                "status": (
                    "preflight_passed"
                    if updates == PREFLIGHT_UPDATES
                    else "training_complete_evaluation_pending"
                ),
                "input_binding": input_binding,
                "training": dict(training),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
                    "contrast_helper_sha256": sha256_file(Path(contrast.__file__)),
                    "evolution_sha256": sha256_file(Path(evolution.__file__)),
                    "natural_gate_sha256": sha256_file(Path(gate.__file__)),
                    "distributed_sha256": sha256_file(Path(distributed.__file__)),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": canonical_sha256(result),
            }
            contrast._write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase=f"seed-ensemble-seed-{seed}-result-save",
        error=save_error,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "status": "worker_complete",
        "rank": context.process_rank,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--updates",
        type=int,
        choices=(PREFLIGHT_UPDATES, TRAIN_UPDATES),
        required=True,
    )
    parser.add_argument("--base-model", type=Path, default=contrast.BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=contrast.V9_ADAPTER)
    parser.add_argument("--dataset-root", type=Path, default=contrast.NATIVE_DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Seed-ensemble training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            seed=args.seed,
            updates=args.updates,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            dataset_root=args.dataset_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
