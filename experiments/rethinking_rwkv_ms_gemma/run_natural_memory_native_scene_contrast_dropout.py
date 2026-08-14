#!/usr/bin/env python3
"""Train V9 shared Q/O gates with native scene contrast and state dropout."""

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
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.checkpoint import checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_projected_kv_read_query_mask,
    set_delta_mem_projected_kv_write_spans,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    prepare_natural_memory_gate as source,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_gate as gate,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_dropout_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_dropout_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_contrast_dropout_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_contrast_dropout_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "040d31c266282554e82027b00aeac5299251cad322e11f651936fc2ced185347"
BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
BASE_CONFIG_SHA256 = "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4"
V9_ADAPTER = SCRIPT_DIR / "local_artifacts/natural_memory_native_shared_qo_gate_stage1_v9/adapter"
V9_ADAPTER_FILES_SHA256 = "1ceda7e288e832df14ace3fb9b4c5db0edc4395945ecbd34c76363f0d0f9e6fb"
V9_ADAPTER_WEIGHTS_SHA256 = "b063940a9be0712f830a992e9114055e4488297b0842245e6e26563b303545a9"
V9_ADAPTER_CONFIG_SHA256 = "94b4649a2b14f178dfd2b2de18bcc77894a5606f8b67426bf753f033690273f4"
NATIVE_DATASET_ROOT = SCRIPT_DIR / "local_artifacts/natural_memory_native_development_v1"
SCENE_RELATIVE_PATH = Path("v4-scene-boundary-detection/train_derived_fit.jsonl")
SCENE_FILE_SHA256 = "8b0552cf1ddd39230896ce1ed6a3842aef94212e70bbc9e76ee8f13c546e6e57"
DONOR_MAPPING_SHA256 = "55247488b4e8297ca9bbe66b9cb4e5e8959093a86972b947530a792472581e6d"
SELECTED_ROWS_SHA256 = "069bb99296b34980d357df0032383a654a13257f75cdf29c011d1b739ad1dcf2"
FULL_SCHEDULE_SHA256 = "12141aecb4f4952e2d76d5d6926d798cc1200ac5753c6407f131a79ad1799aa2"
GATE_FAMILIES = evolution.CONTENT_GATE_PARAMETER_FAMILIES
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 8
LOCAL_ROWS = 2
PREFLIGHT_UPDATES = 1
TRAIN_UPDATES = 32
CHECKPOINT_UPDATES = (8, 16, 32)
MAX_DONOR_TOKEN_DELTA = 16
EXPECTED_ELIGIBLE_ROWS = 1435
CONTRAST_WEIGHT = 0.25
MARGIN = 0.05
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 0.1
CE_CHUNK_TOKENS = 64
TRAIN_SALT = "rwkv-ms-native-scene-contrast-train-v1:"
DROP_SALT = "rwkv-ms-native-scene-state-drop-v1:"


@dataclass(frozen=True)
class SceneContrastRow:
    example: evolution.NativeFullRowExample
    assistant_identity: str


@dataclass(frozen=True)
class ContrastScheduleStep:
    step: int
    source_ordinals: tuple[int, ...]
    donor_ordinals: tuple[int, ...]
    no_state_ordinals: frozenset[int]
    payload_sha256: str


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return source.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Scene contrast-dropout protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = source.sha256_text(source.canonical_json(unsigned))
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Scene contrast-dropout protocol hash differs")
    return value


def load_scene_rows(
    tokenizer: Any,
    dataset_root: Path,
) -> list[SceneContrastRow]:
    path = dataset_root / SCENE_RELATIVE_PATH
    if sha256_file(path) != SCENE_FILE_SHA256:
        raise ValueError("Scene contrast-dropout training file hash differs")
    rows: list[SceneContrastRow] = []
    for source_ordinal, raw_line in enumerate(
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ):
        value = json.loads(raw_line)
        messages = value.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError("Scene contrast-dropout row messages differ")
        assistant_identity = str(messages[-1].get("content"))
        rows.append(
            SceneContrastRow(
                example=evolution.encode_native_full_row(
                    tokenizer,
                    task="scene",
                    source_ordinal=source_ordinal,
                    raw_line=raw_line,
                ),
                assistant_identity=assistant_identity,
            )
        )
    if len(rows) != 1443:
        raise ValueError(f"Expected 1443 scene fit rows, found {len(rows)}")
    return rows


def build_donor_mapping(
    rows: Sequence[SceneContrastRow],
) -> tuple[dict[int, int], dict[int, int], list[dict[str, Any]]]:
    mapping: dict[int, int] = {}
    deltas: dict[int, int] = {}
    payload: list[dict[str, Any]] = []
    for source_ordinal, row in enumerate(rows):
        candidates = [
            donor_ordinal
            for donor_ordinal, donor in enumerate(rows)
            if donor_ordinal != source_ordinal
            and donor.assistant_identity != row.assistant_identity
        ]
        if not candidates:
            raise ValueError(f"Scene contrast row has no donor: {source_ordinal}")
        donor_ordinal = min(
            candidates,
            key=lambda candidate: (
                abs(
                    len(row.example.write_input_ids)
                    - len(rows[candidate].example.write_input_ids)
                ),
                rows[candidate].example.row_sha256,
                candidate,
            ),
        )
        donor = rows[donor_ordinal]
        delta = abs(
            len(row.example.write_input_ids)
            - len(donor.example.write_input_ids)
        )
        mapping[source_ordinal] = donor_ordinal
        deltas[source_ordinal] = delta
        payload.append(
            {
                "source_ordinal": source_ordinal,
                "source_row_sha256": row.example.row_sha256,
                "source_write_tokens": len(row.example.write_input_ids),
                "donor_ordinal": donor_ordinal,
                "donor_row_sha256": donor.example.row_sha256,
                "donor_write_tokens": len(donor.example.write_input_ids),
                "absolute_write_token_delta": delta,
            }
        )
    return mapping, deltas, payload


def build_schedule(
    rows: Sequence[SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
) -> tuple[tuple[ContrastScheduleStep, ...], list[dict[str, Any]]]:
    eligible = [
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= MAX_DONOR_TOKEN_DELTA
    ]
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError("Scene contrast eligible row count differs")
    selected = sorted(
        eligible,
        key=lambda source_ordinal: (
            hashlib.sha256(
                (TRAIN_SALT + rows[source_ordinal].example.row_sha256).encode("utf-8")
            ).hexdigest(),
            source_ordinal,
        ),
    )[: TRAIN_UPDATES * GLOBAL_BATCH_SIZE]
    selected_hashes = [rows[index].example.row_sha256 for index in selected]
    if canonical_sha256(selected_hashes) != SELECTED_ROWS_SHA256:
        raise ValueError("Scene contrast selected row hash differs")
    schedule: list[ContrastScheduleStep] = []
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
                            f"{DROP_SALT}{step}:"
                            + rows[source_ordinal].example.row_sha256
                        ).encode("utf-8")
                    ).hexdigest(),
                    source_ordinal,
                ),
            )[:2]
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
            ContrastScheduleStep(
                step=step,
                source_ordinals=tuple(group),
                donor_ordinals=tuple(mapping[index] for index in group),
                no_state_ordinals=no_state,
                payload_sha256=canonical_sha256(step_payload),
            )
        )
    if canonical_sha256(payload) != FULL_SCHEDULE_SHA256:
        raise ValueError("Scene contrast schedule hash differs")
    return tuple(schedule), payload


def configure_gate_only_training(
    model: torch.nn.Module,
) -> tuple[list[tuple[str, torch.nn.Parameter]], Mapping[str, Any]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    frozen_trainable = []
    for name, parameter in model.named_parameters():
        gate_parameter = any(name.endswith(f".{family}") for family in GATE_FAMILIES)
        parameter.requires_grad_(gate_parameter)
        if gate_parameter:
            selected.append((name, parameter))
        elif parameter.requires_grad:
            frozen_trainable.append(name)
    family_counts = {
        family: sum(name.endswith(f".{family}") for name, _ in selected)
        for family in GATE_FAMILIES
    }
    audit = {
        "parameter_tensors": len(selected),
        "parameter_elements": sum(parameter.numel() for _, parameter in selected),
        "parameter_names_sha256": canonical_sha256([name for name, _ in selected]),
        "family_counts": family_counts,
        "unexpected_trainable": frozen_trainable,
        "passed": (
            not frozen_trainable
            and len(selected) == 126
            and all(count == 42 for count in family_counts.values())
        ),
    }
    if audit["passed"] is not True:
        raise ValueError(f"Scene contrast gate-only trainables differ: {audit!r}")
    return selected, audit


def _state_subset_sha256(
    state: Mapping[str, torch.Tensor],
    *,
    gate_only: bool,
) -> str:
    selected = {
        name: tensor
        for name, tensor in state.items()
        if any(name.endswith(f".{family}") for family in GATE_FAMILIES)
        == gate_only
    }
    if not selected:
        raise ValueError("Scene contrast adapter state subset is empty")
    return runtime._state_dict_sha256(selected)


def build_donor_batch(
    target_batch: evolution.NativeFullRowBatch,
    donor: evolution.NativeFullRowExample,
    *,
    device: torch.device,
) -> evolution.NativeFullRowBatch:
    return evolution.NativeFullRowBatch(
        examples=target_batch.examples,
        write_input_ids=runtime._pad_1d(
            [donor.write_input_ids],
            padding_value=0,
            dtype=torch.long,
            device=device,
        ),
        write_attention_mask=runtime._pad_1d(
            [donor.write_attention_mask],
            padding_value=0,
            dtype=torch.long,
            device=device,
        ),
        read_input_ids=target_batch.read_input_ids,
        read_attention_mask=target_batch.read_attention_mask,
        labels=target_batch.labels,
    )


def checkpointed_read_only(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    def read_only(
        read_input_ids: torch.Tensor,
        read_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        reset_delta_mem_states(model)
        set_delta_mem_projected_kv_read_query_mask(model, None)
        set_delta_mem_projected_kv_write_spans(model, None, None, None)
        set_delta_mem_write_enabled(model, False)
        recompute = evolution.NativeFullRowBatch(
            examples=batch.examples,
            write_input_ids=batch.write_input_ids,
            write_attention_mask=batch.write_attention_mask,
            read_input_ids=read_input_ids,
            read_attention_mask=read_attention_mask,
            labels=batch.labels,
        )
        return evolution._native_read(model, recompute, dtype=dtype)

    return checkpoint(
        read_only,
        batch.read_input_ids,
        batch.read_attention_mask,
        use_reentrant=False,
    )


def _selected_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = labels[:, 1:].ne(-100)
    if not bool(supervised.any().item()):
        raise ValueError("Scene contrast labels contain no targets")
    predictor_indices = supervised.any(dim=0).nonzero(as_tuple=False).flatten()
    if logits.size(1) == labels.size(1):
        selected_logits = logits.index_select(1, predictor_indices)
    elif logits.size(1) == predictor_indices.numel():
        selected_logits = logits
    else:
        raise ValueError("Scene contrast logits do not cover target predictors")
    return selected_logits, labels.index_select(1, predictor_indices + 1)


def detached_answer_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int]:
    selected_logits, selected_labels = _selected_logits_and_labels(logits, labels)
    count = int(selected_labels.ne(-100).sum().item())
    total = 0.0
    for start in range(0, selected_logits.size(1), CE_CHUNK_TOKENS):
        total += float(
            F.cross_entropy(
                selected_logits[:, start : start + CE_CHUNK_TOKENS]
                .contiguous()
                .float()
                .view(-1, selected_logits.size(-1)),
                selected_labels[:, start : start + CE_CHUNK_TOKENS]
                .contiguous()
                .view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
        )
    return total / count, count


def evaluate_condition_ce(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    no_state: bool,
    dtype: torch.dtype,
) -> tuple[float, int]:
    with torch.no_grad():
        if no_state:
            reset_delta_mem_states(model)
            set_delta_mem_write_enabled(model, False)
            logits = evolution._native_read(model, batch, dtype=dtype)
        else:
            evolution._native_write(model, batch, dtype=dtype)
            logits = evolution._native_read(model, batch, dtype=dtype)
        mean_ce, tokens = detached_answer_ce(logits, batch.labels)
    reset_delta_mem_states(model)
    del logits
    return mean_ce, tokens


def backward_condition(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    no_state: bool,
    coefficient: float,
    dtype: torch.dtype,
) -> tuple[float, int, int]:
    if no_state:
        logits = checkpointed_read_only(model, batch, dtype=dtype)
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
        chunk_tokens=CE_CHUNK_TOKENS,
    )
    mean_ce = loss_sum / tokens
    scaled = mean_ce * (coefficient / GLOBAL_BATCH_SIZE)
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Scene contrast signed loss is non-finite")
    scaled.backward()
    value = float(mean_ce.detach().float().item())
    reset_delta_mem_states(model)
    del logits, loss_sum, mean_ce, scaled
    return value, chunks, occupancy


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"Scene contrast output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def save_gate_patch(
    model: torch.nn.Module,
    *,
    output_dir: Path,
    step: int,
    context: distributed.DistributedTrainingContext,
) -> Mapping[str, Any] | None:
    gate_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(name.endswith(f".{family}") for family in GATE_FAMILIES)
    }
    state_sha256 = runtime._state_dict_sha256(gate_state)
    distributed.require_consensus(
        context,
        state_sha256,
        description=f"scene contrast checkpoint {step} gate state",
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
                    "source_adapter_files_sha256": V9_ADAPTER_FILES_SHA256,
                    "step": step,
                    "gate_state_sha256": state_sha256,
                    "state_dict": gate_state,
                },
                patch_path,
            )
            value = {
                "schema": PATCH_SCHEMA,
                "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
                "source_adapter_files_sha256": V9_ADAPTER_FILES_SHA256,
                "step": step,
                "gate_state_sha256": state_sha256,
                "parameter_tensors": len(gate_state),
                "parameter_names_sha256": canonical_sha256(sorted(gate_state)),
                "patch_file": {
                    "path": str(patch_path),
                    "bytes": patch_path.stat().st_size,
                    "sha256": sha256_file(patch_path),
                },
            }
            value["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_patch_manifest_without_receipt",
                "payload_sha256": canonical_sha256(value),
            }
            _write_json(checkpoint_dir / "manifest.json", value)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase=f"scene-contrast-checkpoint-{step}",
        error=save_error,
    )
    return value


def train(
    model: torch.nn.Module,
    rows: Sequence[SceneContrastRow],
    schedule: Sequence[ContrastScheduleStep],
    *,
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
    initial_gate_sha256 = _state_subset_sha256(initial_state, gate_only=True)
    initial_non_gate_sha256 = _state_subset_sha256(initial_state, gate_only=False)
    checkpoints: list[Mapping[str, Any]] = []
    total_active = 0.0
    total_positive_ce = 0.0
    total_donor_ce = 0.0
    total_margin = 0.0
    minimum_gate_gradient_norm = math.inf
    started = time.time()
    progress_path = output_dir / "training_progress.jsonl"
    for schedule_step in schedule[:updates]:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        if len(local_sources) != LOCAL_ROWS:
            raise RuntimeError("Scene contrast local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        local_positive_ce = 0.0
        local_donor_ce = 0.0
        local_margin = 0.0
        local_active = 0.0
        local_dropped = 0.0
        local_correct = 0.0
        local_positive_tokens = 0.0
        local_donor_tokens = 0.0
        local_ce_chunks = 0.0
        local_occupied = 0.0
        for source_ordinal in local_sources:
            target = rows[source_ordinal].example
            donor_ordinal = schedule_step.donor_ordinals[
                schedule_step.source_ordinals.index(source_ordinal)
            ]
            donor = rows[donor_ordinal].example
            no_state = source_ordinal in schedule_step.no_state_ordinals
            target_batch = evolution.collate_native_examples(
                [target],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            donor_batch = build_donor_batch(
                target_batch,
                donor,
                device=context.device,
            )
            evolution.release_native_row_allocator_cache(context.device)
            positive_probe_ce, positive_tokens = evaluate_condition_ce(
                model,
                target_batch,
                no_state=no_state,
                dtype=dtype,
            )
            evolution.release_native_row_allocator_cache(context.device)
            donor_probe_ce, donor_tokens = evaluate_condition_ce(
                model,
                donor_batch,
                no_state=False,
                dtype=dtype,
            )
            margin_value = donor_probe_ce - positive_probe_ce
            active = margin_value < MARGIN
            positive_coefficient = 1.0 + (CONTRAST_WEIGHT if active else 0.0)
            evolution.release_native_row_allocator_cache(context.device)
            positive_train_ce, chunks, occupancy = backward_condition(
                model,
                target_batch,
                no_state=no_state,
                coefficient=positive_coefficient,
                dtype=dtype,
            )
            local_ce_chunks += chunks
            local_occupied += occupancy
            if active:
                evolution.release_native_row_allocator_cache(context.device)
                donor_train_ce, chunks, occupancy = backward_condition(
                    model,
                    donor_batch,
                    no_state=False,
                    coefficient=-CONTRAST_WEIGHT,
                    dtype=dtype,
                )
                local_ce_chunks += chunks
                local_occupied += occupancy
                if not math.isfinite(donor_train_ce):
                    raise RuntimeError("Scene contrast donor train CE is non-finite")
            local_positive_ce += positive_probe_ce
            local_donor_ce += donor_probe_ce
            local_margin += margin_value
            local_active += float(active)
            local_dropped += float(no_state)
            local_correct += float(not no_state)
            local_positive_tokens += positive_tokens
            local_donor_tokens += donor_tokens
            del target_batch, donor_batch
            evolution.release_native_row_allocator_cache(context.device)
        scalar_tensor = gate._prepare_distributed_scalar_sums(
            context,
            (
                local_positive_ce,
                local_donor_ce,
                local_margin,
                local_active,
                local_dropped,
                local_correct,
                local_positive_tokens,
                local_donor_tokens,
                local_ce_chunks,
                local_occupied,
            ),
        )
        metrics = gate._distributed_scalar_sums(context, scalar_tensor)
        if metrics[4] != 2 or metrics[5] != 6:
            raise RuntimeError("Scene contrast state-dropout balance differs")
        if updates == PREFLIGHT_UPDATES and metrics[3] < 1:
            raise RuntimeError("Scene contrast preflight found no active hinge")
        gradient_validation = distributed.validate_local_gradients(named_trainable)
        if gradient_validation["passed"] is not True:
            raise RuntimeError("Scene contrast produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        gate_gradient_audit = evolution.audit_content_gate_gradients(named_trainable)
        if gate_gradient_audit["passed"] is not True:
            raise RuntimeError("Scene contrast produced invalid gate gradients")
        minimum_gate_gradient_norm = min(
            minimum_gate_gradient_norm,
            float(gate_gradient_audit["minimum_family_l2_norm"]),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError("Scene contrast gradient norm is non-finite")
        optimizer.step()
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
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
            "source_ordinals": list(schedule_step.source_ordinals),
            "donor_ordinals": list(schedule_step.donor_ordinals),
            "no_state_ordinals": sorted(schedule_step.no_state_ordinals),
        }
        if context.is_primary:
            _append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "positive_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "donor_ce": round(metrics[1] / GLOBAL_BATCH_SIZE, 6),
                        "margin": round(metrics[2] / GLOBAL_BATCH_SIZE, 6),
                        "active": int(metrics[3]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_positive_ce += metrics[0]
        total_donor_ce += metrics[1]
        total_margin += metrics[2]
        total_active += metrics[3]
        if schedule_step.step in CHECKPOINT_UPDATES or schedule_step.step == updates:
            manifest = save_gate_patch(
                model,
                output_dir=output_dir,
                step=schedule_step.step,
                context=context,
            )
            if manifest is not None:
                checkpoints.append(manifest)
    final_state = snapshot_delta_mem_weights(model)
    final_gate_sha256 = _state_subset_sha256(final_state, gate_only=True)
    final_non_gate_sha256 = _state_subset_sha256(final_state, gate_only=False)
    distributed.require_consensus(
        context,
        final_gate_sha256,
        description="scene contrast final gate state",
    )
    distributed.require_consensus(
        context,
        final_non_gate_sha256,
        description="scene contrast final non-gate state",
    )
    if final_gate_sha256 == initial_gate_sha256:
        raise RuntimeError("Scene contrast training did not change the gate state")
    if final_non_gate_sha256 != initial_non_gate_sha256:
        raise RuntimeError("Scene contrast training changed frozen non-gate state")
    return {
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": updates * GLOBAL_BATCH_SIZE,
        "mean_positive_probe_ce": total_positive_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_probe_ce": total_donor_ce / (updates * GLOBAL_BATCH_SIZE),
        "mean_donor_minus_positive_ce": total_margin / (updates * GLOBAL_BATCH_SIZE),
        "active_hinge_fraction": total_active / (updates * GLOBAL_BATCH_SIZE),
        "minimum_gate_gradient_norm": minimum_gate_gradient_norm,
        "initial_gate_state_sha256": initial_gate_sha256,
        "final_gate_state_sha256": final_gate_sha256,
        "initial_non_gate_state_sha256": initial_non_gate_sha256,
        "final_non_gate_state_sha256": final_non_gate_sha256,
        "non_gate_unchanged": True,
        "progress_sha256": sha256_file(progress_path) if context.is_primary else None,
        "checkpoints": checkpoints,
    }


def run(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path = BASE_MODEL,
    adapter_path: Path = V9_ADAPTER,
    dataset_root: Path = NATIVE_DATASET_ROOT,
) -> Mapping[str, Any]:
    if context.world_size != WORLD_SIZE:
        raise ValueError("Scene contrast training requires exactly four ranks")
    gate.configure_hf_mirror()
    validate_protocol()
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError("Scene contrast output may not be a symbolic link")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Scene contrast output must be fresh: {resolved_output}")
    if sha256_file(base_model / "config.json") != BASE_CONFIG_SHA256:
        raise ValueError("Scene contrast base configuration hash differs")
    adapter_files = gate.snapshot_directory_files(adapter_path)
    if gate._sha256_json(adapter_files) != V9_ADAPTER_FILES_SHA256:
        raise ValueError("Scene contrast V9 adapter files hash differs")
    if (
        sha256_file(adapter_path / "delta_mem_adapter.pt")
        != V9_ADAPTER_WEIGHTS_SHA256
        or sha256_file(adapter_path / "delta_mem_config.json")
        != V9_ADAPTER_CONFIG_SHA256
    ):
        raise ValueError("Scene contrast V9 adapter component hash differs")
    runtime.set_seed(evolution.SEED)
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
        raise ValueError("Scene contrast V9 adapter configuration differs")
    if len(list(iter_delta_mem_modules(model))) != 42:
        raise ValueError("Scene contrast requires 42 wrapped layers")
    named_trainable, trainable_audit = configure_gate_only_training(model)
    rows = load_scene_rows(tokenizer, dataset_root)
    mapping, donor_deltas, donor_payload = build_donor_mapping(rows)
    if canonical_sha256(donor_payload) != DONOR_MAPPING_SHA256:
        raise ValueError("Scene contrast donor mapping hash differs")
    schedule, schedule_payload = build_schedule(rows, mapping, donor_deltas)
    input_binding = {
        "schema": SCHEMA,
        "phase": "preflight" if updates == PREFLIGHT_UPDATES else "training",
        "updates": updates,
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "base_model": str(base_model),
        "base_config_sha256": BASE_CONFIG_SHA256,
        "adapter_path": str(adapter_path),
        "adapter_files": adapter_files,
        "adapter_files_sha256": V9_ADAPTER_FILES_SHA256,
        "dataset_root": str(dataset_root),
        "scene_file_sha256": SCENE_FILE_SHA256,
        "scene_rows": len(rows),
        "donor_mapping_payload_sha256": canonical_sha256(donor_payload),
        "donor_mapping_rows": len(donor_payload),
        "eligible_rows": sum(delta <= MAX_DONOR_TOKEN_DELTA for delta in donor_deltas.values()),
        "full_schedule_payload_sha256": canonical_sha256(schedule_payload),
        "executed_schedule_payload_sha256": canonical_sha256(schedule_payload[:updates]),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "local_rows": LOCAL_ROWS,
        "world_size": context.world_size,
        "contrast_weight": CONTRAST_WEIGHT,
        "margin": MARGIN,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "trainable_audit": trainable_audit,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="scene contrast input binding",
    )
    creation_error: BaseException | None = None
    if context.is_primary:
        try:
            resolved_output.mkdir(parents=True, exist_ok=False)
            _write_json(resolved_output / "input_binding.json", input_binding)
        except BaseException as error:
            creation_error = error
    distributed.phase_consensus(
        context,
        phase="scene-contrast-output-creation",
        error=creation_error,
    )
    training = train(
        model,
        rows,
        schedule,
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
                "status": "preflight_passed" if updates == PREFLIGHT_UPDATES else "training_complete_evaluation_pending",
                "input_binding": input_binding,
                "training": dict(training),
                "protected_splits_opened": [],
                "code_bindings": {
                    "runner_sha256": sha256_file(Path(__file__)),
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
            _write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="scene-contrast-result-save",
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
    parser.add_argument(
        "--updates",
        type=int,
        choices=(PREFLIGHT_UPDATES, TRAIN_UPDATES),
        required=True,
    )
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=V9_ADAPTER)
    parser.add_argument("--dataset-root", type=Path, default=NATIVE_DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Scene contrast training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
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
