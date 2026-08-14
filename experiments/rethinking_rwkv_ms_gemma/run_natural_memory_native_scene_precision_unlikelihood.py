#!/usr/bin/env python3
"""Train the locked checkpoint-16 precision-unlikelihood candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import (  # noqa: E402
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
    run_natural_memory_native_scene_c16_residual as c16,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble as robust,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_training_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_training_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_precision_unlikelihood_training_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_precision_unlikelihood_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "ad397d198a841b9734745ff1ff8faa6f559428a41e216fbcf540a36cf98da51e"
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 16
LOCAL_ROWS = 4
TRAIN_UPDATES = 16
SEED = 131
SEEDS = (SEED,)
LEARNING_RATE = 1.5e-5
UNLIKELIHOOD_WEIGHT = 0.5
MAX_GRAD_NORM = 0.1
POST_STEP_DELTA_RETENTION = 0.995
TRAIN_SALT = "rwkv-ms-native-scene-c16-precision-unlikelihood-v1:131:"
NEGATIVE_SALT = "rwkv-ms-native-scene-c16-precision-negative-v1:131:"
EXPECTED_ELIGIBLE_ROWS = 1435
EXPECTED_EXCLUDED_ROWS = 1076
EXPECTED_AVAILABLE_ROWS = 359
SELECTED_ROWS_PAYLOAD_SHA256 = "71adccf460b96458e8449ef273e98bec0a00d05633dd88d9a79a667984cfef5d"
NEGATIVE_PAYLOAD_SHA256 = "9698c14ce5061b0ce39a90a55eb32941a9cb83bca06dbf97aed6b2d67c377e03"
SCHEDULE_PAYLOAD_SHA256 = "2a15bb468eeae808b7ca6d83806ad97b6e108e4398b3c1bd183c8df52e2cffc6"
STARTING_STEP = c16.STARTING_STEP
STARTING_GATE_STATE_SHA256 = c16.STARTING_GATE_STATE_SHA256
STARTING_PATCH_SHA256 = c16.STARTING_PATCH_SHA256


@dataclass(frozen=True)
class PrecisionNegative:
    example: evolution.NativeFullRowExample
    gold_boundaries: tuple[int, ...]
    wrong_boundary: int
    wrong_target_positions: tuple[int, ...]
    wrong_token_ids: tuple[int, ...]
    negative_content: str
    payload_sha256: str


@dataclass(frozen=True)
class PrecisionScheduleStep:
    step: int
    source_ordinals: tuple[int, ...]
    donor_ordinals: tuple[int, ...]
    no_state_ordinals: frozenset[int]
    payload_sha256: str


_ACTIVE_NEGATIVES: dict[int, PrecisionNegative] = {}
_ORIGINAL_LOAD_SCENE_ROWS = contrast.load_scene_rows


def canonical_sha256(value: Any) -> str:
    return robust.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return robust.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Precision-unlikelihood protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Precision-unlikelihood protocol hash differs")
    return value


def paragraph_count(messages: Sequence[Mapping[str, Any]]) -> int:
    if len(messages) != 3:
        raise ValueError("Precision-unlikelihood messages differ")
    user_content = str(messages[1].get("content", ""))
    indices = [int(value) for value in re.findall(r"\[P(\d+)\]", user_content)]
    if not indices or sorted(set(indices)) != list(range(1, max(indices) + 1)):
        raise ValueError("Precision-unlikelihood paragraph numbering differs")
    return max(indices)


def choose_wrong_boundary(
    *,
    gold_boundaries: Sequence[int],
    maximum_boundary: int,
    row_sha256: str,
) -> int:
    gold = frozenset(int(value) for value in gold_boundaries)
    available = [value for value in range(1, maximum_boundary + 1) if value not in gold]
    if not available:
        raise ValueError("Precision-unlikelihood row has no absent valid boundary")
    if 1 in available:
        return 1
    return min(
        available,
        key=lambda value: (
            hashlib.sha256(
                f"{NEGATIVE_SALT}{row_sha256}:{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )


def _find_subsequence(values: Sequence[int], target: Sequence[int]) -> int:
    starts = [
        start
        for start in range(len(values) - len(target) + 1)
        if tuple(values[start : start + len(target)]) == tuple(target)
    ]
    if len(starts) != 1:
        raise ValueError("Precision-unlikelihood content token span is not unique")
    return starts[0]


def build_negative(
    tokenizer: Any,
    *,
    raw_line: str,
    row: contrast.SceneContrastRow,
) -> PrecisionNegative:
    source = json.loads(raw_line)
    messages = source.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Precision-unlikelihood row messages are missing")
    try:
        target = json.loads(str(messages[-1]["content"]))
        gold_boundaries = tuple(int(value) for value in target["boundaries"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Precision-unlikelihood gold target differs") from error
    maximum_boundary = paragraph_count(messages) - 1
    if (
        tuple(sorted(set(gold_boundaries))) != gold_boundaries
        or any(value < 1 or value > maximum_boundary for value in gold_boundaries)
    ):
        raise ValueError("Precision-unlikelihood gold boundaries differ")
    wrong_boundary = choose_wrong_boundary(
        gold_boundaries=gold_boundaries,
        maximum_boundary=maximum_boundary,
        row_sha256=row.example.row_sha256,
    )
    negative_boundaries = tuple(sorted((*gold_boundaries, wrong_boundary)))
    negative_content = json.dumps(
        {"boundaries": list(negative_boundaries)},
        ensure_ascii=True,
        separators=(",", ": "),
    )
    messages[-1]["content"] = negative_content
    negative_raw = json.dumps(source, ensure_ascii=False, separators=(",", ":"))
    encoded = evolution.encode_native_full_row(
        tokenizer,
        task="scene",
        source_ordinal=row.example.source_ordinal,
        raw_line=negative_raw,
    )
    negative = replace(
        encoded,
        row_id=row.example.row_id,
        row_sha256=row.example.row_sha256,
    )
    gold_start = next(
        index for index, token_id in enumerate(row.example.labels) if token_id != -100
    )
    negative_start = next(
        index for index, token_id in enumerate(negative.labels) if token_id != -100
    )
    if (
        row.example.write_input_ids != negative.write_input_ids
        or row.example.write_attention_mask != negative.write_attention_mask
        or gold_start != negative_start
        or row.example.read_input_ids[:gold_start]
        != negative.read_input_ids[:negative_start]
    ):
        raise ValueError("Precision-unlikelihood negative changed the write or read prefix")
    target_ids = [token_id for token_id in negative.labels if token_id != -100]
    tokenized = tokenizer(
        negative_content,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    content_ids = [int(value) for value in tokenized["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]
    content_start = _find_subsequence(target_ids, content_ids)
    number_start = negative_content.find(str(wrong_boundary), negative_content.index("["))
    number_end = number_start + len(str(wrong_boundary))
    content_positions = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start < number_end and end > number_start
    )
    if not content_positions:
        raise ValueError("Precision-unlikelihood wrong boundary emitted no tokens")
    wrong_positions = tuple(content_start + index for index in content_positions)
    wrong_token_ids = tuple(target_ids[index] for index in wrong_positions)
    if tokenizer.decode(list(wrong_token_ids)).strip() != str(wrong_boundary):
        raise ValueError("Precision-unlikelihood wrong-token decode differs")
    payload = {
        "source_ordinal": row.example.source_ordinal,
        "source_row_sha256": row.example.row_sha256,
        "gold_boundaries": list(gold_boundaries),
        "wrong_boundary": wrong_boundary,
        "negative_content": negative_content,
        "wrong_target_positions": list(wrong_positions),
        "wrong_token_ids": list(wrong_token_ids),
    }
    return PrecisionNegative(
        example=negative,
        gold_boundaries=gold_boundaries,
        wrong_boundary=wrong_boundary,
        wrong_target_positions=wrong_positions,
        wrong_token_ids=wrong_token_ids,
        negative_content=negative_content,
        payload_sha256=canonical_sha256(payload),
    )


def load_scene_rows(
    tokenizer: Any,
    dataset_root: Path,
) -> list[contrast.SceneContrastRow]:
    global _ACTIVE_NEGATIVES
    rows = _ORIGINAL_LOAD_SCENE_ROWS(tokenizer, dataset_root)
    path = dataset_root / contrast.SCENE_RELATIVE_PATH
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) != len(rows):
        raise ValueError("Precision-unlikelihood raw row count differs")
    _ACTIVE_NEGATIVES = {
        index: build_negative(tokenizer, raw_line=raw_line, row=rows[index])
        for index, raw_line in enumerate(raw_lines)
    }
    return rows


def prior_excluded_rows(
    rows: Sequence[contrast.SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
) -> set[int]:
    original, _ = contrast.build_schedule(rows, mapping, deltas)
    excluded = {
        source_ordinal
        for step in original[:STARTING_STEP]
        for source_ordinal in step.source_ordinals
    }
    for prior_seed in robust.SEEDS:
        schedule, _ = robust.build_schedule(rows, mapping, deltas, seed=prior_seed)
        excluded.update(
            source_ordinal for step in schedule for source_ordinal in step.source_ordinals
        )
    available = {
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= contrast.MAX_DONOR_TOKEN_DELTA
    } - excluded
    for seed in (71, 89, 107):
        salt = f"rwkv-ms-native-scene-c16-residual-v1:{seed}:"
        selected = sorted(
            available,
            key=lambda source_ordinal: (
                hashlib.sha256(
                    (salt + rows[source_ordinal].example.row_sha256).encode("utf-8")
                ).hexdigest(),
                source_ordinal,
            ),
        )[: 8 * GLOBAL_BATCH_SIZE]
        excluded.update(selected)
        available.difference_update(selected)
    return excluded


def build_schedules(
    rows: Sequence[contrast.SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
    *,
    enforce_bindings: bool = True,
) -> tuple[
    dict[int, tuple[PrecisionScheduleStep, ...]],
    dict[int, list[dict[str, Any]]],
]:
    if len(_ACTIVE_NEGATIVES) != len(rows):
        raise ValueError("Precision-unlikelihood negative rows are not loaded")
    eligible = {
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= contrast.MAX_DONOR_TOKEN_DELTA
    }
    excluded = prior_excluded_rows(rows, mapping, deltas)
    available = eligible - excluded
    if (
        len(eligible) != EXPECTED_ELIGIBLE_ROWS
        or len(excluded) != EXPECTED_EXCLUDED_ROWS
        or len(available) != EXPECTED_AVAILABLE_ROWS
    ):
        raise ValueError("Precision-unlikelihood row partition differs")
    selected = sorted(
        available,
        key=lambda source_ordinal: (
            hashlib.sha256(
                (TRAIN_SALT + rows[source_ordinal].example.row_sha256).encode("utf-8")
            ).hexdigest(),
            source_ordinal,
        ),
    )[: TRAIN_UPDATES * GLOBAL_BATCH_SIZE]
    selected_rows_hash = canonical_sha256(
        [rows[index].example.row_sha256 for index in selected]
    )
    negative_hash = canonical_sha256(
        [
            {
                "source_ordinal": index,
                "payload_sha256": _ACTIVE_NEGATIVES[index].payload_sha256,
            }
            for index in selected
        ]
    )
    schedule: list[PrecisionScheduleStep] = []
    payload: list[dict[str, Any]] = []
    for offset in range(0, len(selected), GLOBAL_BATCH_SIZE):
        step = offset // GLOBAL_BATCH_SIZE + 1
        group = tuple(selected[offset : offset + GLOBAL_BATCH_SIZE])
        row_payload = [
            {
                "source_ordinal": source_ordinal,
                "source_row_sha256": rows[source_ordinal].example.row_sha256,
                "condition": "correct_state",
                "wrong_boundary": _ACTIVE_NEGATIVES[source_ordinal].wrong_boundary,
                "negative_payload_sha256": _ACTIVE_NEGATIVES[
                    source_ordinal
                ].payload_sha256,
            }
            for source_ordinal in group
        ]
        step_payload = {"step": step, "rows": row_payload}
        payload.append(step_payload)
        schedule.append(
            PrecisionScheduleStep(
                step=step,
                source_ordinals=group,
                donor_ordinals=tuple(mapping[index] for index in group),
                no_state_ordinals=frozenset(),
                payload_sha256=canonical_sha256(step_payload),
            )
        )
    schedule_hash = canonical_sha256(payload)
    if enforce_bindings and (
        selected_rows_hash != SELECTED_ROWS_PAYLOAD_SHA256
        or negative_hash != NEGATIVE_PAYLOAD_SHA256
        or schedule_hash != SCHEDULE_PAYLOAD_SHA256
    ):
        raise ValueError("Precision-unlikelihood schedule binding differs")
    return {SEED: tuple(schedule)}, {SEED: payload}


def unlikelihood_from_logits(
    selected_logits: torch.Tensor,
    wrong_token_ids: torch.Tensor,
) -> torch.Tensor:
    if (
        selected_logits.ndim != 2
        or selected_logits.size(0) == 0
        or wrong_token_ids.ndim != 1
        or wrong_token_ids.numel() != selected_logits.size(0)
    ):
        raise ValueError("Precision-unlikelihood logits and token IDs differ")
    logits = selected_logits.float()
    wrong_ids = wrong_token_ids.to(device=logits.device, dtype=torch.long)
    wrong_logits = logits.gather(1, wrong_ids.unsqueeze(1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, wrong_ids.unsqueeze(1), -torch.inf)
    return F.softplus(wrong_logits - torch.logsumexp(other_logits, dim=1)).mean()


def backward_gold(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> tuple[float, int, int, int]:
    audit, logits = evolution.checkpointed_native_write_read(model, batch, dtype=dtype)
    loss_sum, tokens, chunks = evolution.checkpointed_native_answer_loss_sum_and_count(
        logits,
        batch.labels,
        chunk_tokens=contrast.CE_CHUNK_TOKENS,
    )
    mean_ce = loss_sum / tokens
    scaled = mean_ce / GLOBAL_BATCH_SIZE
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Precision-unlikelihood gold loss is non-finite")
    scaled.backward()
    value = float(mean_ce.detach().float().item())
    reset_delta_mem_states(model)
    del logits, loss_sum, mean_ce, scaled
    return value, tokens, chunks, int(audit["occupied_rows"])


def backward_negative(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    negative: PrecisionNegative,
    *,
    dtype: torch.dtype,
) -> tuple[float, int, int]:
    audit, logits = evolution.checkpointed_native_write_read(model, batch, dtype=dtype)
    selected_logits, selected_labels = contrast._selected_logits_and_labels(
        logits,
        batch.labels,
    )
    positions = torch.tensor(
        negative.wrong_target_positions,
        device=selected_logits.device,
        dtype=torch.long,
    )
    chosen_logits = selected_logits[0].index_select(0, positions)
    wrong_ids = torch.tensor(
        negative.wrong_token_ids,
        device=selected_logits.device,
        dtype=torch.long,
    )
    observed = selected_labels[0].index_select(0, positions)
    if not torch.equal(observed, wrong_ids):
        raise ValueError("Precision-unlikelihood selected wrong tokens differ")
    loss = unlikelihood_from_logits(chosen_logits, wrong_ids)
    scaled = loss * (UNLIKELIHOOD_WEIGHT / GLOBAL_BATCH_SIZE)
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Precision-unlikelihood negative loss is non-finite")
    scaled.backward()
    value = float(loss.detach().float().item())
    reset_delta_mem_states(model)
    del logits, selected_logits, selected_labels, chosen_logits, loss, scaled
    return value, len(negative.wrong_token_ids), int(audit["occupied_rows"])


def train(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[PrecisionScheduleStep],
    *,
    seed: int,
    updates: int,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    dtype: torch.dtype,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    if seed != SEED or updates != TRAIN_UPDATES:
        raise ValueError("Precision-unlikelihood training controls differ")
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
    anchors = {name: parameter.detach().clone() for name, parameter in named_trainable}
    progress_path = output_dir / "training_progress.jsonl"
    total_gold_ce = 0.0
    total_unlikelihood = 0.0
    total_gold_tokens = 0.0
    total_wrong_tokens = 0.0
    minimum_gate_gradient_norm = math.inf
    shrinkage_audits: list[Mapping[str, float]] = []
    final_manifest: Mapping[str, Any] | None = None
    started = time.time()
    for schedule_step in schedule:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        if len(local_sources) != LOCAL_ROWS:
            raise RuntimeError("Precision-unlikelihood local schedule size differs")
        optimizer.zero_grad(set_to_none=True)
        local_metrics = [0.0] * 6
        for source_ordinal in local_sources:
            negative = _ACTIVE_NEGATIVES[source_ordinal]
            gold_batch = evolution.collate_native_examples(
                [rows[source_ordinal].example],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            evolution.release_native_row_allocator_cache(context.device)
            gold_ce, gold_tokens, chunks, occupancy = backward_gold(
                model,
                gold_batch,
                dtype=dtype,
            )
            negative_batch = evolution.collate_native_examples(
                [negative.example],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            evolution.release_native_row_allocator_cache(context.device)
            unlikelihood, wrong_tokens, negative_occupancy = backward_negative(
                model,
                negative_batch,
                negative,
                dtype=dtype,
            )
            local_metrics[0] += gold_ce
            local_metrics[1] += unlikelihood
            local_metrics[2] += gold_tokens
            local_metrics[3] += wrong_tokens
            local_metrics[4] += occupancy + negative_occupancy
            local_metrics[5] += chunks
            del gold_batch, negative_batch
            evolution.release_native_row_allocator_cache(context.device)
        scalar_tensor = gate._prepare_distributed_scalar_sums(context, local_metrics)
        metrics = gate._distributed_scalar_sums(context, scalar_tensor)
        gradient_validation = distributed.validate_local_gradients(named_trainable)
        if gradient_validation["passed"] is not True:
            raise RuntimeError("Precision-unlikelihood produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        gate_gradient_audit = evolution.audit_content_gate_gradients(named_trainable)
        if gate_gradient_audit["passed"] is not True:
            raise RuntimeError("Precision-unlikelihood produced invalid gate gradients")
        minimum_gate_gradient_norm = min(
            minimum_gate_gradient_norm,
            float(gate_gradient_audit["minimum_family_l2_norm"]),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError("Precision-unlikelihood gradient norm is non-finite")
        optimizer.step()
        shrinkage = robust.apply_proximal_shrinkage(named_trainable, anchors)
        shrinkage_audits.append(shrinkage)
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "seed": seed,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "world_size": context.world_size,
            "correct_state_rows": GLOBAL_BATCH_SIZE,
            "mean_gold_ce": metrics[0] / GLOBAL_BATCH_SIZE,
            "mean_unlikelihood": metrics[1] / GLOBAL_BATCH_SIZE,
            "gold_target_tokens": int(metrics[2]),
            "wrong_boundary_tokens": int(metrics[3]),
            "written_condition_occupancy_rows": int(metrics[4]),
            "checkpointed_ce_chunks": int(metrics[5]),
            "gradient_norm_before_clip": float(grad_norm.detach().float().item()),
            "gradient_collective_sha256": canonical_sha256(collective),
            "gate_gradient_audit": gate_gradient_audit,
            "proximal_shrinkage": shrinkage,
            "source_ordinals": list(schedule_step.source_ordinals),
            "wrong_boundaries": [
                _ACTIVE_NEGATIVES[index].wrong_boundary
                for index in schedule_step.source_ordinals
            ],
        }
        if context.is_primary:
            contrast._append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "step": schedule_step.step,
                        "gold_ce": round(metrics[0] / GLOBAL_BATCH_SIZE, 6),
                        "unlikelihood": round(metrics[1] / GLOBAL_BATCH_SIZE, 6),
                        "delta_l2": round(shrinkage["delta_l2_after"], 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_gold_ce += metrics[0]
        total_unlikelihood += metrics[1]
        total_gold_tokens += metrics[2]
        total_wrong_tokens += metrics[3]
        if schedule_step.step == updates:
            final_manifest = robust.save_gate_patch(
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
        description="precision-unlikelihood final gate state",
    )
    distributed.require_consensus(
        context,
        final_non_gate_sha256,
        description="precision-unlikelihood final non-gate state",
    )
    if final_gate_sha256 == initial_gate_sha256:
        raise RuntimeError("Precision-unlikelihood did not change the gate state")
    if final_non_gate_sha256 != initial_non_gate_sha256:
        raise RuntimeError("Precision-unlikelihood changed frozen non-gate state")
    denominator = updates * GLOBAL_BATCH_SIZE
    return {
        "seed": seed,
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": denominator,
        "mean_gold_ce": total_gold_ce / denominator,
        "mean_unlikelihood": total_unlikelihood / denominator,
        "gold_target_tokens": int(total_gold_tokens),
        "wrong_boundary_tokens": int(total_wrong_tokens),
        "unlikelihood_weight": UNLIKELIHOOD_WEIGHT,
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


def configure_engine() -> None:
    c16.SCHEMA = SCHEMA
    c16.STEP_SCHEMA = STEP_SCHEMA
    c16.PATCH_SCHEMA = PATCH_SCHEMA
    c16.PROTOCOL = PROTOCOL
    c16.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    c16.WORLD_SIZE = WORLD_SIZE
    c16.GLOBAL_BATCH_SIZE = GLOBAL_BATCH_SIZE
    c16.LOCAL_ROWS = LOCAL_ROWS
    c16.TRAIN_UPDATES = TRAIN_UPDATES
    c16.SEEDS = SEEDS
    c16.LEARNING_RATE = LEARNING_RATE
    c16.POST_STEP_DELTA_RETENTION = POST_STEP_DELTA_RETENTION
    c16.validate_protocol = validate_protocol
    c16.build_schedules = build_schedules
    c16.configure_training_engine = configure_training_engine
    contrast.load_scene_rows = load_scene_rows


def configure_training_engine() -> None:
    robust.PROTOCOL_PAYLOAD_SHA256 = PROTOCOL_PAYLOAD_SHA256
    robust.PATCH_SCHEMA = PATCH_SCHEMA
    robust.STEP_SCHEMA = STEP_SCHEMA
    robust.LEARNING_RATE = LEARNING_RATE
    robust.MAX_GRAD_NORM = MAX_GRAD_NORM
    robust.POST_STEP_DELTA_RETENTION = POST_STEP_DELTA_RETENTION
    robust.train = train


def run(**kwargs: Any) -> Mapping[str, Any]:
    configure_engine()
    return c16.run(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEED)
    parser.add_argument("--base-model", type=Path, default=contrast.BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=contrast.V9_ADAPTER)
    parser.add_argument("--dataset-root", type=Path, default=contrast.NATIVE_DATASET_ROOT)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Precision-unlikelihood training requires four-rank torchrun")
    try:
        result = run(
            context=context,
            output_dir=args.output_dir,
            seed=args.seed,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            dataset_root=args.dataset_root,
            training_root=args.training_root,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
