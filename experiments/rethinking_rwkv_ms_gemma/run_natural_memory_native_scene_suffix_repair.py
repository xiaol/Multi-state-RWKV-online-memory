#!/usr/bin/env python3
"""Train local gold-suffix repair from the locked checkpoint-16 endpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.core.delta import reset_delta_mem_states  # noqa: E402
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    natural_memory_distributed as distributed,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_evolution as evolution,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_causal as causal,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_contrast_dropout as contrast,
)
from experiments.rethinking_rwkv_ms_gemma import (  # noqa: E402
    run_natural_memory_native_scene_onpolicy_repair as shared,
)


SCHEMA = "rwkv_ms_natural_memory_native_scene_suffix_repair_training_result.v1"
STEP_SCHEMA = "rwkv_ms_natural_memory_native_scene_suffix_repair_training_step.v1"
PATCH_SCHEMA = "rwkv_ms_natural_memory_native_scene_suffix_repair_training_patch.v1"
PROTOCOL = SCRIPT_DIR / "natural_memory_native_scene_suffix_repair_protocol_v1.json"
PROTOCOL_PAYLOAD_SHA256 = "2a65f6105b1c6d31acc6b683b894914b743e7c645dc4ba98d667d32b2880af0c"
SEED = shared.SEED
SEEDS = (SEED,)
WORLD_SIZE = shared.WORLD_SIZE
GLOBAL_BATCH_SIZE = shared.GLOBAL_BATCH_SIZE
LOCAL_ROWS = shared.LOCAL_ROWS
TRAIN_UPDATES = shared.TRAIN_UPDATES
LEARNING_RATE = 2.5e-6
PAIRWISE_MARGIN = shared.PAIRWISE_MARGIN
MAX_GRAD_NORM = 0.025
POST_STEP_DELTA_RETENTION = 0.995
MAX_NEW_TOKENS = shared.MAX_NEW_TOKENS
LOCAL_GOLD_SUFFIX_TOKENS = 4
LOCAL_GOLD_SUFFIX_CE_WEIGHT = 0.25

RepairSource = shared.RepairSource
RepairPlan = shared.RepairPlan
MinedRepair = shared.MinedRepair
RepairScheduleStep = shared.RepairScheduleStep
canonical_sha256 = shared.canonical_sha256
sha256_file = shared.sha256_file
build_schedules = shared.build_schedules
first_divergence = shared.first_divergence
repair_plan = shared.repair_plan
pairwise_margin_loss = shared.pairwise_margin_loss
load_scene_rows = shared.load_scene_rows

_BASE_TRAIN = shared.train


def validate_protocol() -> Mapping[str, Any]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = value.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Suffix-repair protocol receipt is missing")
    unsigned = dict(value)
    unsigned.pop("receipt")
    digest = canonical_sha256(unsigned)
    if digest != PROTOCOL_PAYLOAD_SHA256 or receipt.get("payload_sha256") != digest:
        raise ValueError("Suffix-repair protocol hash differs")
    return value


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
        gold_ids = shared._gold_generation_ids(tokenizer, source.messages, prompt_ids)
        plan = repair_plan(
            gold_token_ids=gold_ids,
            generated_token_ids=generated_ids,
            gold_boundaries=source.gold_boundaries,
            prediction=prediction,
        )
        example: evolution.NativeFullRowExample | None = None
        supervised_ids: tuple[int, ...] = ()
        if plan.status == "actionable":
            if plan.divergence_index is None or plan.correct_token_id is None:
                raise RuntimeError("Suffix-repair actionable plan is incomplete")
            supervised_ids = tuple(
                gold_ids[
                    plan.divergence_index :
                    plan.divergence_index + LOCAL_GOLD_SUFFIX_TOKENS
                ]
            )
            if not supervised_ids or supervised_ids[0] != plan.correct_token_id:
                raise RuntimeError("Suffix-repair gold suffix differs")
            read_ids = (
                prompt_ids
                + generated_ids[: plan.divergence_index]
                + supervised_ids
            )
            example = evolution.NativeFullRowExample(
                row_id=f"{row.example.row_id}:suffix-repair",
                task="scene",
                source_ordinal=row.example.source_ordinal,
                row_sha256=row.example.row_sha256,
                write_input_ids=row.example.write_input_ids,
                write_attention_mask=row.example.write_attention_mask,
                read_input_ids=read_ids,
                read_attention_mask=(1,) * len(read_ids),
                labels=(-100,) * (len(read_ids) - len(supervised_ids))
                + supervised_ids,
                assistant_target_tokens=len(supervised_ids),
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
            "supervised_suffix_tokens": len(supervised_ids),
            "supervised_suffix_sha256": canonical_sha256(list(supervised_ids)),
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
        raise ValueError("Suffix-repair backward plan differs")
    audit, logits = evolution.checkpointed_native_write_read(model, batch, dtype=dtype)
    selected_logits, selected_labels = contrast._selected_logits_and_labels(
        logits,
        batch.labels,
    )
    if selected_logits.shape[0] != 1 or not 1 <= selected_logits.shape[1] <= 4:
        raise ValueError("Suffix-repair supervision length differs")
    observed = int(selected_labels[0, 0].item())
    if observed != mined.plan.correct_token_id:
        raise ValueError("Suffix-repair correct token differs")
    token_logits = selected_logits[0, 0]
    margin_before = float(
        (
            token_logits[mined.plan.correct_token_id].float()
            - token_logits[mined.plan.wrong_token_id].float()
        )
        .detach()
        .item()
    )
    objective, pairwise, closure = suffix_repair_loss(
        selected_logits[0],
        selected_labels[0],
        wrong_token_id=mined.plan.wrong_token_id,
    )
    scaled = objective / GLOBAL_BATCH_SIZE
    if not bool(torch.isfinite(scaled).item()):
        raise RuntimeError("Suffix-repair loss is non-finite")
    scaled.backward()
    pairwise_value = float(pairwise.detach().item())
    reset_delta_mem_states(model)
    del logits, selected_logits, selected_labels, token_logits
    del pairwise, closure, objective, scaled
    return pairwise_value, margin_before, int(audit["occupied_rows"])


def suffix_repair_loss(
    selected_logits: torch.Tensor,
    selected_labels: torch.Tensor,
    *,
    wrong_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        selected_logits.ndim != 2
        or selected_labels.ndim != 1
        or selected_logits.shape[0] != selected_labels.shape[0]
        or not 1 <= selected_logits.shape[0] <= LOCAL_GOLD_SUFFIX_TOKENS
    ):
        raise ValueError("Suffix-repair selected trajectory differs")
    correct_token_id = int(selected_labels[0].item())
    pairwise = pairwise_margin_loss(
        selected_logits[0],
        correct_token_id=correct_token_id,
        wrong_token_id=wrong_token_id,
    )
    closure = torch.zeros((), device=pairwise.device, dtype=torch.float32)
    if selected_logits.shape[0] > 1:
        closure = F.cross_entropy(
            selected_logits[1:].float(),
            selected_labels[1:],
        )
    return pairwise + LOCAL_GOLD_SUFFIX_CE_WEIGHT * closure, pairwise, closure


def train(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    result = dict(_BASE_TRAIN(*args, **kwargs))
    result.update(
        {
            "intervention_runner_sha256": sha256_file(Path(__file__)),
            "local_gold_suffix_tokens": LOCAL_GOLD_SUFFIX_TOKENS,
            "local_gold_suffix_ce_weight": LOCAL_GOLD_SUFFIX_CE_WEIGHT,
            "full_sequence_gold_ce_weight": 0.0,
            "synthetic_negative_unlikelihood_weight": 0.0,
        }
    )
    return result


def configure_engine() -> None:
    replacements = {
        "SCHEMA": SCHEMA,
        "STEP_SCHEMA": STEP_SCHEMA,
        "PATCH_SCHEMA": PATCH_SCHEMA,
        "PROTOCOL": PROTOCOL,
        "PROTOCOL_PAYLOAD_SHA256": PROTOCOL_PAYLOAD_SHA256,
        "LEARNING_RATE": LEARNING_RATE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "POST_STEP_DELTA_RETENTION": POST_STEP_DELTA_RETENTION,
        "validate_protocol": validate_protocol,
        "mine_repair": mine_repair,
        "backward_repair": backward_repair,
        "train": train,
    }
    for name, value in replacements.items():
        setattr(shared, name, value)


def run(**kwargs: Any) -> Mapping[str, Any]:
    configure_engine()
    return shared.run(**kwargs)


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
        raise ValueError("Suffix-repair training requires four-rank torchrun")
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
