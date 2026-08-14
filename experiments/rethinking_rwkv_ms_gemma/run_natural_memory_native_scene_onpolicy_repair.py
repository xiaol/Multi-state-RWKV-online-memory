#!/usr/bin/env python3
"""Train the locked checkpoint-16 on-policy first-divergence repair."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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

from deltamem.chat_templates import apply_chat_template  # noqa: E402
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
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_precision_unlikelihood as precision,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_seed_ensemble as robust,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_onpolicy_repair_training_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_onpolicy_repair_training_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_onpolicy_repair_training_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_onpolicy_repair_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "a01497344e733e53caf3f49f4db12e2190076bda34807a4a137c5dc3b001f4d6"
WORLD_SIZE = 4
GLOBAL_BATCH_SIZE = 16
LOCAL_ROWS = 4
TRAIN_UPDATES = 6
SEED = 149
SEEDS = (SEED,)
LEARNING_RATE = 5e-6
PAIRWISE_MARGIN = 0.5
MAX_GRAD_NORM = 0.05
POST_STEP_DELTA_RETENTION = 0.99
MAX_NEW_TOKENS = 64
TRAIN_SALT = "rwkv-ms-native-scene-c16-onpolicy-repair-v1:149:"
EXPECTED_ELIGIBLE_ROWS = 1435
EXPECTED_PRIOR_EXCLUDED_ROWS = 1076
EXPECTED_PRECISION_ROWS = 256
EXPECTED_AVAILABLE_ROWS = 103
SELECTED_ROWS_PAYLOAD_SHA256 = (
    "4bdb444aace7e1684b00f2a10b1b17812d01aa3b7ebeaab7eb7d330f0fdfc332"
)
SCHEDULE_PAYLOAD_SHA256 = (
    "3b81436961d24252d3222ff63dffd185e7e8a8d71d3822d2f94e6cd9fe649eaa"
)
STARTING_STEP = c16.STARTING_STEP
STARTING_GATE_STATE_SHA256 = c16.STARTING_GATE_STATE_SHA256
STARTING_PATCH_SHA256 = c16.STARTING_PATCH_SHA256
_ORIGINAL_LOAD_SCENE_ROWS = precision._ORIGINAL_LOAD_SCENE_ROWS


@dataclass(frozen=True)
class RepairSource:
    messages: tuple[Mapping[str, Any], ...]
    gold_boundaries: tuple[int, ...]


@dataclass(frozen=True)
class RepairPlan:
    status: str
    prediction: tuple[int, ...] | None
    gold_boundaries: tuple[int, ...]
    false_positive_boundaries: tuple[int, ...]
    divergence_index: int | None
    correct_token_id: int | None
    wrong_token_id: int | None


@dataclass(frozen=True)
class MinedRepair:
    example: evolution.NativeFullRowExample | None
    plan: RepairPlan
    generated_token_count: int
    generated_tokens_sha256: str
    hit_max_new_tokens: bool
    payload_sha256: str


@dataclass(frozen=True)
class RepairScheduleStep:
    step: int
    source_ordinals: tuple[int, ...]
    donor_ordinals: tuple[int, ...]
    no_state_ordinals: frozenset[int]
    payload_sha256: str


_ACTIVE_REPAIR_SOURCES: dict[int, RepairSource] = {}
_ACTIVE_TOKENIZER: Any | None = None


def canonical_sha256(value: Any) -> str:
    return robust.canonical_sha256(value)


def sha256_file(path: Path) -> str:
    return robust.sha256_file(path)


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("On-policy repair protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("On-policy repair protocol hash differs")
    return value


def _gold_boundaries(messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    try:
        value = json.loads(str(messages[-1]["content"]))
        boundaries = tuple(int(item) for item in value["boundaries"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("On-policy repair gold target differs") from error
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("On-policy repair gold boundaries differ")
    return boundaries


def load_scene_rows(
    tokenizer: Any,
    dataset_root: Path,
) -> list[contrast.SceneContrastRow]:
    global _ACTIVE_REPAIR_SOURCES, _ACTIVE_TOKENIZER
    rows = _ORIGINAL_LOAD_SCENE_ROWS(tokenizer, dataset_root)
    path = dataset_root / contrast.SCENE_RELATIVE_PATH
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(raw_lines) != len(rows):
        raise ValueError("On-policy repair raw row count differs")
    sources: dict[int, RepairSource] = {}
    for index, raw_line in enumerate(raw_lines):
        value = json.loads(raw_line)
        messages = value.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError("On-policy repair row messages differ")
        sources[index] = RepairSource(
            messages=tuple(dict(message) for message in messages),
            gold_boundaries=_gold_boundaries(messages),
        )
    _ACTIVE_REPAIR_SOURCES = sources
    _ACTIVE_TOKENIZER = tokenizer
    return rows


def _precision_selected_rows(
    rows: Sequence[contrast.SceneContrastRow],
    eligible: set[int],
    prior_excluded: set[int],
) -> set[int]:
    selected = sorted(
        eligible - prior_excluded,
        key=lambda source_ordinal: (
            hashlib.sha256(
                (
                    precision.TRAIN_SALT
                    + rows[source_ordinal].example.row_sha256
                ).encode("utf-8")
            ).hexdigest(),
            source_ordinal,
        ),
    )[:EXPECTED_PRECISION_ROWS]
    return set(selected)


def build_schedules(
    rows: Sequence[contrast.SceneContrastRow],
    mapping: Mapping[int, int],
    deltas: Mapping[int, int],
    *,
    enforce_bindings: bool = True,
) -> tuple[
    dict[int, tuple[RepairScheduleStep, ...]],
    dict[int, list[dict[str, Any]]],
]:
    if len(_ACTIVE_REPAIR_SOURCES) != len(rows):
        raise ValueError("On-policy repair sources are not loaded")
    eligible = {
        source_ordinal
        for source_ordinal in range(len(rows))
        if deltas[source_ordinal] <= contrast.MAX_DONOR_TOKEN_DELTA
    }
    prior_excluded = precision.prior_excluded_rows(rows, mapping, deltas)
    precision_rows = _precision_selected_rows(rows, eligible, prior_excluded)
    available = eligible - prior_excluded - precision_rows
    if (
        len(eligible) != EXPECTED_ELIGIBLE_ROWS
        or len(prior_excluded) != EXPECTED_PRIOR_EXCLUDED_ROWS
        or len(precision_rows) != EXPECTED_PRECISION_ROWS
        or len(available) != EXPECTED_AVAILABLE_ROWS
    ):
        raise ValueError("On-policy repair row partition differs")
    selected = sorted(
        available,
        key=lambda source_ordinal: (
            hashlib.sha256(
                (TRAIN_SALT + rows[source_ordinal].example.row_sha256).encode(
                    "utf-8"
                )
            ).hexdigest(),
            source_ordinal,
        ),
    )[: TRAIN_UPDATES * GLOBAL_BATCH_SIZE]
    selected_payload = [
        {
            "source_ordinal": index,
            "source_row_sha256": rows[index].example.row_sha256,
        }
        for index in selected
    ]
    schedule: list[RepairScheduleStep] = []
    payload: list[dict[str, Any]] = []
    for offset in range(0, len(selected), GLOBAL_BATCH_SIZE):
        step = offset // GLOBAL_BATCH_SIZE + 1
        group = tuple(selected[offset : offset + GLOBAL_BATCH_SIZE])
        row_payload = [
            {
                "source_ordinal": source_ordinal,
                "source_row_sha256": rows[source_ordinal].example.row_sha256,
                "condition": "correct_state",
                "repair": "current_greedy_first_divergence",
            }
            for source_ordinal in group
        ]
        step_payload = {"step": step, "rows": row_payload}
        payload.append(step_payload)
        schedule.append(
            RepairScheduleStep(
                step=step,
                source_ordinals=group,
                donor_ordinals=tuple(mapping[index] for index in group),
                no_state_ordinals=frozenset(),
                payload_sha256=canonical_sha256(step_payload),
            )
        )
    if enforce_bindings and (
        canonical_sha256(selected_payload) != SELECTED_ROWS_PAYLOAD_SHA256
        or canonical_sha256(payload) != SCHEDULE_PAYLOAD_SHA256
    ):
        raise ValueError("On-policy repair schedule binding differs")
    return {SEED: tuple(schedule)}, {SEED: payload}


def first_divergence(
    gold_token_ids: Sequence[int], generated_token_ids: Sequence[int]
) -> int | None:
    for index, (gold, generated) in enumerate(
        zip(gold_token_ids, generated_token_ids, strict=False)
    ):
        if int(gold) != int(generated):
            return index
    return None


def repair_plan(
    *,
    gold_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    gold_boundaries: Sequence[int],
    prediction: Sequence[int] | None,
) -> RepairPlan:
    gold = tuple(int(value) for value in gold_boundaries)
    predicted = None if prediction is None else tuple(sorted(set(int(v) for v in prediction)))
    false_positives = (
        () if predicted is None else tuple(sorted(set(predicted) - set(gold)))
    )
    if predicted is None:
        status = "invalid_prediction"
        divergence = None
    elif not false_positives:
        status = "no_false_positive"
        divergence = None
    else:
        divergence = first_divergence(gold_token_ids, generated_token_ids)
        status = "actionable" if divergence is not None else "no_actionable_divergence"
    return RepairPlan(
        status=status,
        prediction=predicted,
        gold_boundaries=gold,
        false_positive_boundaries=false_positives,
        divergence_index=divergence,
        correct_token_id=(
            int(gold_token_ids[divergence]) if divergence is not None else None
        ),
        wrong_token_id=(
            int(generated_token_ids[divergence]) if divergence is not None else None
        ),
    )


def pairwise_margin_loss(
    logits: torch.Tensor,
    *,
    correct_token_id: int,
    wrong_token_id: int,
) -> torch.Tensor:
    if logits.ndim != 1 or correct_token_id == wrong_token_id:
        raise ValueError("On-policy repair pairwise tokens differ")
    correct = logits[int(correct_token_id)].float()
    wrong = logits[int(wrong_token_id)].float()
    return F.softplus(wrong - correct + PAIRWISE_MARGIN)


def _gold_generation_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[int],
) -> tuple[int, ...]:
    rendered = apply_chat_template(
        tokenizer,
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tuple(
        int(value)
        for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
    )
    prefix = tuple(int(value) for value in prompt_ids)
    if full_ids[: len(prefix)] != prefix or len(full_ids) <= len(prefix):
        raise ValueError("On-policy repair gold generation prefix differs")
    return full_ids[len(prefix) :]


def mine_repair(
    model: torch.nn.Module,
    tokenizer: Any,
    row: contrast.SceneContrastRow,
    source: RepairSource,
    *,
    device: torch.device,
) -> MinedRepair:
    generated_ids: tuple[int, ...] = ()
    model.eval()
    try:
        causal.prime_messages(
            model,
            tokenizer,
            source.messages[:-1],
            device=str(device),
        )
        encoded = causal.encode_prompt(
            tokenizer,
            source.messages[:-1],
            generation=True,
        )
        prompt_ids = tuple(int(value) for value in encoded.input_ids[0].tolist())
        config = copy.deepcopy(causal.generation_config(model, tokenizer))
        config.max_new_tokens = MAX_NEW_TOKENS
        causal.set_delta_write_enabled(model, False)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                generation_config=config,
            )
        generated_ids = tuple(
            int(value) for value in outputs[0, len(prompt_ids) :].tolist()
        )
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = causal.recovery.extract_json(raw)
        recovered = causal.recovery.recover_scene(parsed)
        prediction = None if recovered is None else tuple(sorted(recovered))
        gold_ids = _gold_generation_ids(tokenizer, source.messages, prompt_ids)
        plan = repair_plan(
            gold_token_ids=gold_ids,
            generated_token_ids=generated_ids,
            gold_boundaries=source.gold_boundaries,
            prediction=prediction,
        )
        example: evolution.NativeFullRowExample | None = None
        if plan.status == "actionable":
            if (
                plan.divergence_index is None
                or plan.correct_token_id is None
                or plan.wrong_token_id is None
            ):
                raise RuntimeError("On-policy repair actionable plan is incomplete")
            read_ids = (
                prompt_ids
                + generated_ids[: plan.divergence_index]
                + (plan.correct_token_id,)
            )
            example = evolution.NativeFullRowExample(
                row_id=f"{row.example.row_id}:onpolicy-repair",
                task="scene",
                source_ordinal=row.example.source_ordinal,
                row_sha256=row.example.row_sha256,
                write_input_ids=row.example.write_input_ids,
                write_attention_mask=row.example.write_attention_mask,
                read_input_ids=read_ids,
                read_attention_mask=(1,) * len(read_ids),
                labels=(-100,) * (len(read_ids) - 1) + (plan.correct_token_id,),
                assistant_target_tokens=1,
            )
        payload = {
            "source_ordinal": row.example.source_ordinal,
            "source_row_sha256": row.example.row_sha256,
            "status": plan.status,
            "prediction": None if plan.prediction is None else list(plan.prediction),
            "gold_boundaries": list(plan.gold_boundaries),
            "false_positive_boundaries": list(plan.false_positive_boundaries),
            "divergence_index": plan.divergence_index,
            "correct_token_id": plan.correct_token_id,
            "wrong_token_id": plan.wrong_token_id,
            "generated_token_count": len(generated_ids),
            "generated_tokens_sha256": canonical_sha256(list(generated_ids)),
            "hit_max_new_tokens": len(generated_ids) >= MAX_NEW_TOKENS,
        }
        return MinedRepair(
            example=example,
            plan=plan,
            generated_token_count=len(generated_ids),
            generated_tokens_sha256=payload["generated_tokens_sha256"],
            hit_max_new_tokens=payload["hit_max_new_tokens"],
            payload_sha256=canonical_sha256(payload),
        )
    finally:
        reset_delta_mem_states(model)
        causal.set_delta_write_enabled(model, True)
        model.train()


def backward_repair(
    model: torch.nn.Module,
    batch: evolution.NativeFullRowBatch,
    mined: MinedRepair,
    *,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    if (
        mined.example is None
        or mined.plan.correct_token_id is None
        or mined.plan.wrong_token_id is None
    ):
        raise ValueError("On-policy repair backward plan differs")
    audit, logits = evolution.checkpointed_native_write_read(model, batch, dtype=dtype)
    selected_logits, selected_labels = contrast._selected_logits_and_labels(
        logits,
        batch.labels,
    )
    if selected_logits.shape[:2] != (1, 1):
        raise ValueError("On-policy repair must supervise exactly one token")
    observed = int(selected_labels[0, 0].item())
    if observed != mined.plan.correct_token_id:
        raise ValueError("On-policy repair correct token differs")
    token_logits = selected_logits[0, 0]
    margin_before = float(
        (
            token_logits[mined.plan.correct_token_id].float()
            - token_logits[mined.plan.wrong_token_id].float()
        )
        .detach()
        .item()
    )
    loss = pairwise_margin_loss(
        token_logits,
        correct_token_id=mined.plan.correct_token_id,
        wrong_token_id=mined.plan.wrong_token_id,
    )
    scaled = loss / GLOBAL_BATCH_SIZE
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("On-policy repair loss is non-finite")
    scaled.backward()
    value = float(loss.detach().item())
    reset_delta_mem_states(model)
    del logits, selected_logits, selected_labels, token_logits, loss, scaled
    return value, margin_before, int(audit["occupied_rows"])


def _repair_evidence(
    source_ordinal: int,
    mined: MinedRepair,
) -> Mapping[str, Any]:
    plan = mined.plan
    return {
        "source_ordinal": source_ordinal,
        "status": plan.status,
        "prediction": None if plan.prediction is None else list(plan.prediction),
        "gold_boundaries": list(plan.gold_boundaries),
        "false_positive_boundaries": list(plan.false_positive_boundaries),
        "divergence_index": plan.divergence_index,
        "correct_token_id": plan.correct_token_id,
        "wrong_token_id": plan.wrong_token_id,
        "generated_token_count": mined.generated_token_count,
        "generated_tokens_sha256": mined.generated_tokens_sha256,
        "hit_max_new_tokens": mined.hit_max_new_tokens,
        "payload_sha256": mined.payload_sha256,
    }


def train(
    model: torch.nn.Module,
    rows: Sequence[contrast.SceneContrastRow],
    schedule: Sequence[RepairScheduleStep],
    *,
    seed: int,
    updates: int,
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    dtype: torch.dtype,
    output_dir: Path,
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    if seed != SEED or updates != TRAIN_UPDATES or _ACTIVE_TOKENIZER is None:
        raise ValueError("On-policy repair training controls differ")
    tokenizer = _ACTIVE_TOKENIZER
    started = time.time()
    local_ordinals = tuple(
        source_ordinal
        for step in schedule
        for source_ordinal in step.source_ordinals[
            context.process_rank * LOCAL_ROWS : (context.process_rank + 1) * LOCAL_ROWS
        ]
    )
    if len(local_ordinals) != TRAIN_UPDATES * LOCAL_ROWS:
        raise ValueError("On-policy repair local mining schedule differs")
    model.eval()
    mined_by_ordinal = {
        source_ordinal: mine_repair(
            model,
            tokenizer,
            rows[source_ordinal],
            _ACTIVE_REPAIR_SOURCES[source_ordinal],
            device=context.device,
        )
        for source_ordinal in local_ordinals
    }
    distributed.gather_objects(
        context,
        {
            "rank": context.process_rank,
            "mined_rows": len(mined_by_ordinal),
            "payload_sha256": canonical_sha256(
                [
                    mined_by_ordinal[index].payload_sha256
                    for index in local_ordinals
                ]
            ),
        },
    )
    mining_elapsed_seconds = time.time() - started
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
    total_loss = 0.0
    total_margin = 0.0
    total_repairs = 0
    total_false_positives = 0
    total_invalid = 0
    total_no_false_positive = 0
    total_no_actionable = 0
    total_hit_max = 0
    minimum_gate_gradient_norm = math.inf
    shrinkage_audits: list[Mapping[str, float]] = []
    final_manifest: Mapping[str, Any] | None = None
    for schedule_step in schedule:
        local_start = context.process_rank * LOCAL_ROWS
        local_sources = schedule_step.source_ordinals[
            local_start : local_start + LOCAL_ROWS
        ]
        optimizer.zero_grad(set_to_none=True)
        local_metrics = [0.0] * 8
        local_evidence: list[Mapping[str, Any]] = []
        for source_ordinal in local_sources:
            mined = mined_by_ordinal[source_ordinal]
            local_evidence.append(_repair_evidence(source_ordinal, mined))
            local_metrics[3] += len(mined.plan.false_positive_boundaries)
            local_metrics[7] += int(mined.hit_max_new_tokens)
            if mined.plan.status == "invalid_prediction":
                local_metrics[4] += 1
                continue
            if mined.plan.status == "no_false_positive":
                local_metrics[5] += 1
                continue
            if mined.plan.status != "actionable" or mined.example is None:
                local_metrics[6] += 1
                continue
            batch = evolution.collate_native_examples(
                [mined.example],
                pad_token_id=pad_token_id,
                device=context.device,
            )
            evolution.release_native_row_allocator_cache(context.device)
            loss, margin_before, occupancy = backward_repair(
                model,
                batch,
                mined,
                dtype=dtype,
            )
            local_metrics[0] += loss
            local_metrics[1] += margin_before
            local_metrics[2] += 1
            local_metrics[3] += 0
            del batch
            evolution.release_native_row_allocator_cache(context.device)
            if occupancy <= 0:
                raise RuntimeError("On-policy repair produced no written state")
        gathered_evidence = distributed.gather_objects(context, local_evidence)
        scalar_tensor = gate._prepare_distributed_scalar_sums(context, local_metrics)
        metrics = gate._distributed_scalar_sums(context, scalar_tensor)
        repairs = int(metrics[2])
        if repairs <= 0:
            raise RuntimeError("On-policy repair step has no actionable rows")
        gradient_validation = distributed.validate_local_gradients(named_trainable)
        if gradient_validation["passed"] is not True:
            raise RuntimeError("On-policy repair produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        gate_gradient_audit = evolution.audit_content_gate_gradients(named_trainable)
        if gate_gradient_audit["passed"] is not True:
            raise RuntimeError("On-policy repair produced invalid gate gradients")
        minimum_gate_gradient_norm = min(
            minimum_gate_gradient_norm,
            float(gate_gradient_audit["minimum_family_l2_norm"]),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError("On-policy repair gradient norm is non-finite")
        optimizer.step()
        shrinkage = robust.apply_proximal_shrinkage(named_trainable, anchors)
        shrinkage_audits.append(shrinkage)
        flattened_evidence = [item for rank_rows in gathered_evidence for item in rank_rows]
        record_value = {
            "schema": STEP_SCHEMA,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "seed": seed,
            "step": schedule_step.step,
            "schedule_step_sha256": schedule_step.payload_sha256,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "world_size": context.world_size,
            "mined_from_frozen_checkpoint_16": True,
            "generated_rows": GLOBAL_BATCH_SIZE,
            "actionable_repairs": repairs,
            "mean_pairwise_loss": metrics[0] / repairs,
            "mean_correct_minus_wrong_margin_before": metrics[1] / repairs,
            "false_positive_boundaries": int(metrics[3]),
            "invalid_predictions": int(metrics[4]),
            "rows_without_false_positives": int(metrics[5]),
            "rows_without_actionable_divergence": int(metrics[6]),
            "rows_hitting_max_new_tokens": int(metrics[7]),
            "gradient_norm_before_clip": float(grad_norm.detach().item()),
            "gradient_collective_sha256": canonical_sha256(collective),
            "gate_gradient_audit": gate_gradient_audit,
            "proximal_shrinkage": shrinkage,
            "repair_evidence": flattened_evidence,
            "repair_evidence_sha256": canonical_sha256(flattened_evidence),
        }
        if context.is_primary:
            contrast._append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": schedule_step.step,
                        "repairs": repairs,
                        "loss": round(metrics[0] / repairs, 6),
                        "margin": round(metrics[1] / repairs, 6),
                        "delta_l2": round(shrinkage["delta_l2_after"], 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        total_loss += metrics[0]
        total_margin += metrics[1]
        total_repairs += repairs
        total_false_positives += int(metrics[3])
        total_invalid += int(metrics[4])
        total_no_false_positive += int(metrics[5])
        total_no_actionable += int(metrics[6])
        total_hit_max += int(metrics[7])
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
        description="on-policy repair final gate state",
    )
    distributed.require_consensus(
        context,
        final_non_gate_sha256,
        description="on-policy repair final non-gate state",
    )
    if final_gate_sha256 == initial_gate_sha256:
        raise RuntimeError("On-policy repair did not change the gate state")
    if final_non_gate_sha256 != initial_non_gate_sha256:
        raise RuntimeError("On-policy repair changed frozen non-gate state")
    return {
        "seed": seed,
        "updates": updates,
        "elapsed_seconds": time.time() - started,
        "rows": updates * GLOBAL_BATCH_SIZE,
        "generated_rows": updates * GLOBAL_BATCH_SIZE,
        "mining_elapsed_seconds": mining_elapsed_seconds,
        "actionable_repairs": total_repairs,
        "mean_pairwise_loss": total_loss / total_repairs,
        "mean_correct_minus_wrong_margin_before": total_margin / total_repairs,
        "false_positive_boundaries": total_false_positives,
        "invalid_predictions": total_invalid,
        "rows_without_false_positives": total_no_false_positive,
        "rows_without_actionable_divergence": total_no_actionable,
        "rows_hitting_max_new_tokens": total_hit_max,
        "pairwise_margin": PAIRWISE_MARGIN,
        "full_sequence_gold_ce_weight": 0.0,
        "synthetic_negative_unlikelihood_weight": 0.0,
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
        "intervention_runner_sha256": sha256_file(Path(__file__)),
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
        raise ValueError("On-policy repair training requires four-rank torchrun")
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
