#!/usr/bin/env python3
"""Warm-start R12 and alternate synthetic memory CE with native full-row CE."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.checkpoint import checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deltamem.chat_templates import apply_chat_template
from deltamem.core.delta import (
    iter_delta_mem_modules,
    load_delta_mem_adapter,
    reset_delta_mem_states,
    save_delta_mem_adapter,
    set_delta_mem_projected_kv_read_query_mask,
    set_delta_mem_projected_kv_write_spans,
    set_delta_mem_write_enabled,
    snapshot_delta_mem_weights,
)
from deltamem.train.delta_sft import tokenize_messages_for_sft
from experiments.rethinking_rwkv_ms_gemma import natural_memory_distributed as distributed
from experiments.rethinking_rwkv_ms_gemma import prepare_natural_memory_gate as source
from experiments.rethinking_rwkv_ms_gemma import run_natural_memory_gate as gate
from experiments.rethinking_rwkv_ms_gemma import (
    run_synthetic_compositional_associative_retrieval_v3 as runtime,
)


SCHEMA = "rwkv_ms_natural_memory_native_evolution_run.v1"
TRAIN_STEP_SCHEMA = "rwkv_ms_natural_memory_native_evolution_step.v1"
MIXED_SCHEDULE_SCHEMA = "rwkv_ms_natural_memory_native_mixed_schedule.v1"
HF_ENDPOINT = "https://hf-mirror.com"
BASE_MODEL = Path("/root/X/.cache/hf/gemma-4-E4B-it-a4c2d58")
R12_ADAPTER = Path(
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_gate_replication_r12_development_run_split20260825_seed53/adapter"
)
R12_ADAPTER_FILES_SHA256 = (
    "cf745e0d795ff1aae28521f2207d2282fe3d0ec2c88e744e27f82463186a8a63"
)
R12_SOURCE_MANIFEST = Path(
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_gate_replication_r12_development_split20260825_seed53/manifest.json"
)
NATIVE_DATASET_ROOT = Path(
    "experiments/rethinking_rwkv_ms_gemma/local_artifacts/"
    "natural_memory_native_development_v1"
)
EVOLUTION_PROTOCOL = Path(__file__).with_name(
    "natural_memory_native_evolution_protocol_v1.json"
)
EVOLUTION_PROTOCOL_PAYLOAD_SHA256 = (
    "219ea71766d3859569f1f49cc428eab5411e4ce8fa2bcae0a333ddfe78bbc749"
)
RESIDUAL_HYBRID_PROTOCOL = Path(__file__).with_name(
    "natural_memory_native_residual_hybrid_protocol_v1.json"
)
RESIDUAL_HYBRID_PROTOCOL_PAYLOAD_SHA256 = (
    "acea98b9d7b12a21208e3333370453ba81b15e478666a1496c358e7401597050"
)
CONTENT_GATE_PROTOCOL = Path(__file__).with_name(
    "natural_memory_native_content_gate_protocol_v3.json"
)
CONTENT_GATE_PROTOCOL_PAYLOAD_SHA256 = (
    "ee9cea48667657e4b4db9808a97a7e3a7fdd3b9e44ed8b83fff34352c31ee68c"
)
SHARED_QO_GATE_PROTOCOL = Path(__file__).with_name(
    "natural_memory_native_shared_qo_gate_protocol_v4.json"
)
SHARED_QO_GATE_PROTOCOL_PAYLOAD_SHA256 = (
    "b6680ad52477058fd2c20eb6ca13e492907e9a0b19806b451bff0ffa99a258a5"
)
FUSION_TOPOLOGIES = (
    "attention_output",
    "post_attention_residual_hybrid",
    "content_gated_attention_output",
    "shared_qo_content_gated_attention_output",
)
TASK_FILES = {
    "attribution": "v3.2-attribution-best-candidate/train_derived_fit.jsonl",
    "narrative": "v3.2-narrative-type-classification/train_derived_fit.jsonl",
    "scene": "v4-scene-boundary-detection/train_derived_fit.jsonl",
}
SEED = 20260812
STAGE1_UPDATES = 192
PREFLIGHT_UPDATES = 2
GLOBAL_BATCH_SIZE = 16
LOCAL_BATCH_SIZE = 4
LOCAL_MICROBATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 1.0
MAX_SEQUENCE_LENGTH = 32768
NATIVE_EXECUTION_SUBBATCH_SIZE = 1
NATIVE_CE_CHUNK_TOKENS = 64
NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES = 20 * 1024 * 1024
CONTENT_GATE_PARAMETER_FAMILIES = (
    "memory_fusion_hidden_weight",
    "memory_fusion_read_weight",
    "memory_fusion_bias",
)


@dataclass(frozen=True)
class NativeFullRowExample:
    row_id: str
    task: str
    source_ordinal: int
    row_sha256: str
    write_input_ids: tuple[int, ...]
    write_attention_mask: tuple[int, ...]
    read_input_ids: tuple[int, ...]
    read_attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_target_tokens: int


@dataclass
class NativeFullRowBatch:
    examples: list[NativeFullRowExample]
    write_input_ids: torch.Tensor
    write_attention_mask: torch.Tensor
    read_input_ids: torch.Tensor
    read_attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class OffloadedNativeActivation:
    device: torch.device
    tensor: torch.Tensor


@dataclass
class NativeSelectiveOffloadStats:
    tensors: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class MixedTrainingStep:
    step: int
    update_kind: str
    update_kind_index: int
    task: str | None
    epoch: int
    global_indices: tuple[int, ...]
    global_row_ids: tuple[str, ...]
    step_sha256: str


def canonical_sha256(value: Any) -> str:
    return distributed.canonical_sha256(value)


def _read_json(path: Path, description: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def load_evolution_protocol(
    fusion_topology: str = "attention_output",
    path: Path | None = None,
) -> Mapping[str, Any]:
    if fusion_topology not in FUSION_TOPOLOGIES:
        raise ValueError(f"Unknown evolution fusion topology: {fusion_topology!r}")
    protocol_bindings = {
        "attention_output": (
            EVOLUTION_PROTOCOL,
            EVOLUTION_PROTOCOL_PAYLOAD_SHA256,
        ),
        "post_attention_residual_hybrid": (
            RESIDUAL_HYBRID_PROTOCOL,
            RESIDUAL_HYBRID_PROTOCOL_PAYLOAD_SHA256,
        ),
        "content_gated_attention_output": (
            CONTENT_GATE_PROTOCOL,
            CONTENT_GATE_PROTOCOL_PAYLOAD_SHA256,
        ),
        "shared_qo_content_gated_attention_output": (
            SHARED_QO_GATE_PROTOCOL,
            SHARED_QO_GATE_PROTOCOL_PAYLOAD_SHA256,
        ),
    }
    expected_path, expected_payload_sha256 = protocol_bindings[fusion_topology]
    path = expected_path if path is None else path
    protocol = _read_json(path.resolve(strict=True), "evolution protocol")
    receipt = protocol.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Evolution protocol receipt is missing")
    unsigned = dict(protocol)
    unsigned.pop("receipt")
    payload_sha256 = source.sha256_text(source.canonical_json(unsigned))
    if (
        receipt.get("payload_sha256") != payload_sha256
        or payload_sha256 != expected_payload_sha256
    ):
        raise ValueError("Evolution protocol payload hash differs")
    return protocol


def build_evolution_delta_config(fusion_topology: str) -> Any:
    if fusion_topology not in FUSION_TOPOLOGIES:
        raise ValueError(f"Unknown evolution fusion topology: {fusion_topology!r}")
    config = gate.build_delta_config(
        target_layers=gate.DEFAULT_TARGET_LAYERS,
        rank=gate.PRODUCTION_ADAPTER_RANK,
        key_dim=gate.PRODUCTION_KEY_DIM,
        temperature=gate.PRODUCTION_TEMPERATURE,
    )
    if fusion_topology == "attention_output":
        return config
    if fusion_topology in {
        "content_gated_attention_output",
        "shared_qo_content_gated_attention_output",
    }:
        return replace(
            config,
            memory_fusion_mode=(
                "content_gated_qo_add"
                if fusion_topology == "shared_qo_content_gated_attention_output"
                else "content_gated_add"
            ),
            memory_fusion_gate_init=0.1,
        )
    return replace(
        config,
        memory_fusion_placement="post_attention_residual_hybrid",
        memory_fusion_residual_scale=0.01,
        memory_fusion_residual_scale_max=0.02,
    )


def validate_native_dataset_root(root: Path) -> Mapping[str, Any]:
    resolved = root.expanduser().resolve(strict=True)
    manifest_path = resolved / "manifest.json"
    manifest = _read_json(manifest_path, "native development manifest")
    receipt = manifest.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("Native development receipt is missing")
    unsigned = dict(manifest)
    unsigned.pop("receipt")
    if receipt.get("payload_sha256") != source.sha256_text(
        source.canonical_json(unsigned)
    ):
        raise ValueError("Native development manifest receipt differs")
    leakage = manifest.get("leakage_audit")
    if (
        not isinstance(leakage, Mapping)
        or leakage.get("cross_split_normalized_32_character_shingle_overlap") != 0
        or leakage.get("component_ids_crossing_splits") != 0
        or leakage.get("protected_splits_opened") != []
    ):
        raise ValueError("Native development leakage audit failed")
    if manifest.get("split_seed") != SEED:
        raise ValueError("Native development split seed differs")
    for relative_path in TASK_FILES.values():
        path = resolved / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def encode_native_full_row(
    tokenizer: Any,
    *,
    task: str,
    source_ordinal: int,
    raw_line: str,
) -> NativeFullRowExample:
    row = json.loads(raw_line)
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Native {task} row must contain exactly three messages")
    if [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError(f"Native {task} row roles differ")
    features = tokenize_messages_for_sft(
        tokenizer,
        messages,
        MAX_SEQUENCE_LENGTH,
        assistant_loss_mode="final_assistant_only",
    )
    input_ids = tuple(int(value) for value in features["input_ids"])
    attention_mask = tuple(int(value) for value in features["attention_mask"])
    labels = tuple(int(value) for value in features["labels"])
    if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
        raise ValueError(f"Native {task} tokenization is misaligned")
    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised or supervised != list(
        range(supervised[0], supervised[-1] + 1)
    ):
        raise ValueError(f"Native {task} assistant target is not contiguous")
    write_end = supervised[0]
    write_rendered = apply_chat_template(
        tokenizer,
        messages[:-1],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded_write = tokenizer(write_rendered, add_special_tokens=False)
    write_input_ids = tuple(int(value) for value in encoded_write["input_ids"])
    if write_input_ids != input_ids[:write_end]:
        raise ValueError(
            f"Native {task} system/user prefill is not a stable full-row prefix"
        )
    rendered_full = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    untruncated = tuple(
        int(value)
        for value in tokenizer(rendered_full, add_special_tokens=False)["input_ids"]
    )
    if untruncated != input_ids:
        raise ValueError(f"Native {task} full row was truncated or rewritten")
    row_sha256 = source.sha256_text(raw_line)
    return NativeFullRowExample(
        row_id=f"native:{task}:{source_ordinal}:{row_sha256[:20]}",
        task=task,
        source_ordinal=source_ordinal,
        row_sha256=row_sha256,
        write_input_ids=write_input_ids,
        write_attention_mask=(1,) * len(write_input_ids),
        read_input_ids=input_ids,
        read_attention_mask=attention_mask,
        labels=labels,
        assistant_target_tokens=len(supervised),
    )


def load_native_examples(root: Path, tokenizer: Any) -> list[NativeFullRowExample]:
    resolved = root.expanduser().resolve(strict=True)
    examples: list[NativeFullRowExample] = []
    for task, relative_path in TASK_FILES.items():
        path = resolved / relative_path
        for source_ordinal, raw_line in enumerate(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ):
            examples.append(
                encode_native_full_row(
                    tokenizer,
                    task=task,
                    source_ordinal=source_ordinal,
                    raw_line=raw_line,
                )
            )
    row_ids = [example.row_id for example in examples]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Native full-row example IDs are not unique")
    return examples


def collate_native_examples(
    examples: Sequence[NativeFullRowExample],
    *,
    pad_token_id: int,
    device: torch.device,
) -> NativeFullRowBatch:
    if not examples:
        raise ValueError("Cannot collate an empty native batch")
    return NativeFullRowBatch(
        examples=list(examples),
        write_input_ids=runtime._pad_1d(
            [example.write_input_ids for example in examples],
            padding_value=pad_token_id,
            dtype=torch.long,
            device=device,
        ),
        write_attention_mask=runtime._pad_1d(
            [example.write_attention_mask for example in examples],
            padding_value=0,
            dtype=torch.long,
            device=device,
        ),
        read_input_ids=runtime._pad_1d(
            [example.read_input_ids for example in examples],
            padding_value=pad_token_id,
            dtype=torch.long,
            device=device,
        ),
        read_attention_mask=runtime._pad_1d(
            [example.read_attention_mask for example in examples],
            padding_value=0,
            dtype=torch.long,
            device=device,
        ),
        labels=runtime._pad_1d(
            [example.labels for example in examples],
            padding_value=-100,
            dtype=torch.long,
            device=device,
        ),
    )


def build_task_balanced_native_schedule(
    examples: Sequence[NativeFullRowExample],
    *,
    updates: int,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    seed: int = SEED,
) -> tuple[tuple[MixedTrainingStep, ...], str]:
    if updates <= 0 or global_batch_size <= 0:
        raise ValueError("Native schedule sizes must be positive")
    tasks = tuple(sorted(TASK_FILES))
    indices_by_task = {
        task: [index for index, example in enumerate(examples) if example.task == task]
        for task in tasks
    }
    if any(len(indices) < global_batch_size for indices in indices_by_task.values()):
        raise ValueError("Every native task requires one complete global batch")
    base_count, remainder = divmod(updates, len(tasks))
    task_counts = {
        task: base_count + int(index < remainder)
        for index, task in enumerate(tasks)
    }
    task_sequence = [
        task for task in tasks for _ in range(task_counts[task])
    ]
    random.Random(seed).shuffle(task_sequence)
    task_rngs = {
        task: random.Random(int(source.sha256_text(f"{seed}:{task}")[:16], 16))
        for task in tasks
    }
    task_epochs = {task: 0 for task in tasks}
    task_queues: dict[str, list[tuple[int, tuple[int, ...]]]] = {
        task: [] for task in tasks
    }

    def refill(task: str) -> None:
        shuffled = list(indices_by_task[task])
        task_rngs[task].shuffle(shuffled)
        epoch = task_epochs[task]
        task_epochs[task] += 1
        complete = len(shuffled) // global_batch_size * global_batch_size
        task_queues[task].extend(
            (epoch, tuple(shuffled[start : start + global_batch_size]))
            for start in range(0, complete, global_batch_size)
        )

    steps: list[MixedTrainingStep] = []
    for task_index, task in enumerate(task_sequence, 1):
        if not task_queues[task]:
            refill(task)
        epoch, selected = task_queues[task].pop(0)
        rows = tuple(examples[index].row_id for index in selected)
        payload = {
            "schema": MIXED_SCHEDULE_SCHEMA,
            "update_kind": "native",
            "update_kind_index": task_index,
            "task": task,
            "epoch": epoch,
            "global_indices": list(selected),
            "global_row_ids": list(rows),
        }
        steps.append(
            MixedTrainingStep(
                step=0,
                update_kind="native",
                update_kind_index=task_index,
                task=task,
                epoch=epoch,
                global_indices=selected,
                global_row_ids=rows,
                step_sha256=canonical_sha256(payload),
            )
        )
    audit = Counter(step.task for step in steps)
    if max(audit.values()) - min(audit.values()) > 1:
        raise RuntimeError("Native task schedule is not probability-balanced")
    payload = [
        {
            "update_kind_index": step.update_kind_index,
            "task": step.task,
            "epoch": step.epoch,
            "global_indices": list(step.global_indices),
            "global_row_ids": list(step.global_row_ids),
            "step_sha256": step.step_sha256,
        }
        for step in steps
    ]
    return tuple(steps), canonical_sha256(payload)


def build_mixed_schedule(
    synthetic_examples: Sequence[Any],
    native_examples: Sequence[NativeFullRowExample],
    *,
    total_updates: int,
    seed: int = SEED,
) -> tuple[tuple[MixedTrainingStep, ...], Mapping[str, Any]]:
    if total_updates not in {PREFLIGHT_UPDATES, STAGE1_UPDATES}:
        raise ValueError("Evolution updates must be the 2-step preflight or 192-step Stage 1")
    if total_updates % 2:
        raise ValueError("Mixed evolution schedule requires an even update count")
    per_kind = total_updates // 2
    row_ids = [str(example.row_id) for example in synthetic_examples]
    family_ids = [
        f"{example.condition}:{example.episode_id}" for example in synthetic_examples
    ]
    member_orders = [int(example.semantic_target_slot) for example in synthetic_examples]
    synthetic_schedule, synthetic_sha256 = (
        distributed.build_family_balanced_training_schedule(
            row_ids,
            family_ids,
            member_orders,
            seed=seed,
            epochs=1,
            max_steps=per_kind,
            world_size=distributed.REQUIRED_WORLD_SIZE,
            local_batch_size=distributed.REQUIRED_LOCAL_BATCH_SIZE,
        )
    )
    native_schedule, native_sha256 = build_task_balanced_native_schedule(
        native_examples,
        updates=per_kind,
        seed=seed,
    )
    mixed: list[MixedTrainingStep] = []
    for index in range(per_kind):
        synthetic = synthetic_schedule[index]
        synthetic_payload = {
            "schema": MIXED_SCHEDULE_SCHEMA,
            "step": 2 * index + 1,
            "update_kind": "synthetic",
            "update_kind_index": index + 1,
            "epoch": synthetic.epoch,
            "global_indices": list(synthetic.global_indices),
            "global_row_ids": list(synthetic.global_row_ids),
            "source_step_sha256": synthetic.step_sha256,
        }
        mixed.append(
            MixedTrainingStep(
                step=2 * index + 1,
                update_kind="synthetic",
                update_kind_index=index + 1,
                task=None,
                epoch=synthetic.epoch,
                global_indices=synthetic.global_indices,
                global_row_ids=synthetic.global_row_ids,
                step_sha256=canonical_sha256(synthetic_payload),
            )
        )
        native = native_schedule[index]
        native_payload = {
            "schema": MIXED_SCHEDULE_SCHEMA,
            "step": 2 * index + 2,
            "update_kind": "native",
            "update_kind_index": index + 1,
            "task": native.task,
            "epoch": native.epoch,
            "global_indices": list(native.global_indices),
            "global_row_ids": list(native.global_row_ids),
            "source_step_sha256": native.step_sha256,
        }
        mixed.append(
            MixedTrainingStep(
                step=2 * index + 2,
                update_kind="native",
                update_kind_index=index + 1,
                task=native.task,
                epoch=native.epoch,
                global_indices=native.global_indices,
                global_row_ids=native.global_row_ids,
                step_sha256=canonical_sha256(native_payload),
            )
        )
    schedule_payload = [
        {
            "step": step.step,
            "update_kind": step.update_kind,
            "update_kind_index": step.update_kind_index,
            "task": step.task,
            "epoch": step.epoch,
            "global_row_ids": list(step.global_row_ids),
            "step_sha256": step.step_sha256,
        }
        for step in mixed
    ]
    return tuple(mixed), {
        "schema": MIXED_SCHEDULE_SCHEMA,
        "total_updates": total_updates,
        "synthetic_updates": per_kind,
        "native_updates": per_kind,
        "alternation": "odd_synthetic_even_native",
        "synthetic_schedule_sha256": synthetic_sha256,
        "native_schedule_sha256": native_sha256,
        "mixed_schedule_sha256": canonical_sha256(schedule_payload),
        "native_task_updates": dict(
            sorted(Counter(step.task for step in native_schedule).items())
        ),
    }


def _native_write(
    model: torch.nn.Module,
    batch: NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> Mapping[str, Any]:
    reset_delta_mem_states(model)
    set_delta_mem_projected_kv_read_query_mask(model, None)
    set_delta_mem_projected_kv_write_spans(model, None, None, None)
    set_delta_mem_write_enabled(model, True)
    with runtime._autocast_context(batch.write_input_ids.device, dtype):
        model(
            input_ids=batch.write_input_ids,
            attention_mask=batch.write_attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    occupied_rows = 0
    occupied_total = 0
    for _, module in iter_delta_mem_modules(model):
        occupied = module.projected_kv_occupied
        if occupied is None or occupied.ndim != 2:
            raise RuntimeError("Native write did not produce projected-KV occupancy")
        occupied_rows += int(occupied.any(dim=-1).sum().item())
        occupied_total += int(occupied.size(0))
    set_delta_mem_write_enabled(model, False)
    return {
        "occupied_rows": occupied_rows,
        "occupied_total": occupied_total,
    }


def _native_read(
    model: torch.nn.Module,
    batch: NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    set_delta_mem_write_enabled(model, False)
    set_delta_mem_projected_kv_read_query_mask(model, None)
    predictor_indices = runtime._answer_predictor_indices(batch.labels)
    with runtime._autocast_context(batch.read_input_ids.device, dtype):
        outputs = model(
            input_ids=batch.read_input_ids,
            attention_mask=batch.read_attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=predictor_indices,
        )
    return outputs.logits


def release_native_row_allocator_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def should_selectively_offload_native_activation(
    tensor: torch.Tensor,
    *,
    min_bytes: int = NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES,
) -> bool:
    if min_bytes <= 0:
        raise ValueError("Native selective offload threshold must be positive")
    return bool(
        tensor.device.type == "cuda"
        and not tensor.is_leaf
        and tensor.grad_fn is not None
        and tensor.numel() * tensor.element_size() >= min_bytes
    )


def native_selective_offload_hooks(
    stats: NativeSelectiveOffloadStats,
    *,
    min_bytes: int = NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES,
) -> tuple[Any, Any]:
    def pack(tensor: torch.Tensor) -> torch.Tensor | OffloadedNativeActivation:
        if not should_selectively_offload_native_activation(
            tensor,
            min_bytes=min_bytes,
        ):
            return tensor
        stats.tensors += 1
        stats.bytes += tensor.numel() * tensor.element_size()
        return OffloadedNativeActivation(
            device=tensor.device,
            tensor=tensor.detach().to("cpu"),
        )

    def unpack(
        packed: torch.Tensor | OffloadedNativeActivation,
    ) -> torch.Tensor:
        if isinstance(packed, OffloadedNativeActivation):
            return packed.tensor.to(packed.device)
        return packed

    return pack, unpack


def _native_write_read_selectively_offloaded(
    model: torch.nn.Module,
    batch: NativeFullRowBatch,
    *,
    dtype: torch.dtype,
) -> tuple[Mapping[str, Any], torch.Tensor, NativeSelectiveOffloadStats]:
    stats = NativeSelectiveOffloadStats()
    pack, unpack = native_selective_offload_hooks(stats)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        write_audit = _native_write(model, batch, dtype=dtype)
        release_native_row_allocator_cache(batch.read_input_ids.device)
        logits = _native_read(model, batch, dtype=dtype)
    return write_audit, logits, stats


def execution_subbatch_size(update_kind: str) -> int:
    if update_kind == "synthetic":
        return LOCAL_MICROBATCH_SIZE
    if update_kind == "native":
        return NATIVE_EXECUTION_SUBBATCH_SIZE
    raise ValueError(f"Unknown mixed update kind: {update_kind!r}")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def local_objective_denominators(
    update_kind: str,
    batches: Sequence[Any],
) -> tuple[int, int]:
    if update_kind not in {"synthetic", "native"}:
        raise ValueError(f"Unknown mixed update kind: {update_kind!r}")
    local_answer_tokens = sum(
        int(batch.labels[:, 1:].ne(-100).sum().item()) for batch in batches
    )
    local_route_rows = LOCAL_BATCH_SIZE if update_kind == "synthetic" else 0
    if local_answer_tokens <= 0:
        raise ValueError("Mixed update has no supervised answer tokens")
    return local_answer_tokens, local_route_rows


def _float32_cross_entropy_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.contiguous().float().view(-1, logits.size(-1)),
        labels.contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def checkpointed_native_answer_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_tokens: int = NATIVE_CE_CHUNK_TOKENS,
) -> tuple[torch.Tensor, int, int]:
    if chunk_tokens <= 0:
        raise ValueError("Native CE chunk size must be positive")
    if logits.ndim != 3 or labels.ndim != 2 or logits.size(0) != labels.size(0):
        raise ValueError("Native answer logits and labels are misaligned")
    if labels.size(1) < 2:
        raise ValueError("Native answer labels have no causal sequence axis")
    supervised = labels[:, 1:].ne(-100)
    if not bool(supervised.any().item()):
        raise ValueError("Native answer labels contain no supervised targets")
    predictor_indices = supervised.any(dim=0).nonzero(as_tuple=False).flatten()
    if logits.size(1) == labels.size(1):
        selected_logits = logits.index_select(1, predictor_indices)
    elif logits.size(1) == predictor_indices.numel():
        selected_logits = logits
    else:
        raise ValueError("Native answer logits do not cover supervised predictors")
    selected_labels = labels.index_select(1, predictor_indices + 1)
    count = int(selected_labels.ne(-100).sum().item())
    losses = [
        checkpoint(
            _float32_cross_entropy_sum,
            selected_logits[:, start : start + chunk_tokens],
            selected_labels[:, start : start + chunk_tokens],
            use_reentrant=False,
        )
        for start in range(0, selected_logits.size(1), chunk_tokens)
    ]
    if not losses:
        raise RuntimeError("Native checkpointed CE emitted no chunks")
    return torch.stack(losses).sum(), count, len(losses)


def audit_content_gate_gradients(
    named_trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> Mapping[str, Any]:
    family_audits: dict[str, dict[str, Any]] = {}
    for family in CONTENT_GATE_PARAMETER_FAMILIES:
        selected = [
            (name, parameter)
            for name, parameter in named_trainable
            if name.endswith(f".{family}")
        ]
        active = [
            (name, parameter)
            for name, parameter in selected
            if parameter.grad is not None
        ]
        missing = [name for name, parameter in selected if parameter.grad is None]
        nonfinite = [
            name
            for name, parameter in active
            if not bool(torch.isfinite(parameter.grad).all().item())
        ]
        finite_gradients = [
            parameter.grad.detach().float()
            for name, parameter in active
            if name not in nonfinite
        ]
        squared_norm = sum(
            float(gradient.square().sum().item())
            for gradient in finite_gradients
        )
        nonzero_elements = sum(
            int(torch.count_nonzero(gradient).item())
            for gradient in finite_gradients
        )
        family_audits[family] = {
            "parameter_tensors": len(selected),
            "parameter_names_sha256": canonical_sha256(
                [name for name, _ in selected]
            ),
            "active_gradient_tensors": len(active),
            "missing_gradient_tensors": len(missing),
            "missing_preview": missing[:8],
            "nonfinite_gradient_tensors": len(nonfinite),
            "nonfinite_preview": nonfinite[:8],
            "nonzero_gradient_elements": nonzero_elements,
            "l2_norm": math.sqrt(squared_norm),
            "passed": (
                bool(selected)
                and not missing
                and not nonfinite
                and nonzero_elements > 0
            ),
        }
    return {
        "families": family_audits,
        "parameter_tensors": sum(
            audit["parameter_tensors"] for audit in family_audits.values()
        ),
        "all_families_finite_nonzero": all(
            audit["passed"] for audit in family_audits.values()
        ),
        "minimum_family_l2_norm": min(
            audit["l2_norm"] for audit in family_audits.values()
        ),
        "passed": all(audit["passed"] for audit in family_audits.values()),
    }


def train_mixed_distributed(
    model: torch.nn.Module,
    synthetic_examples: Sequence[Any],
    native_examples: Sequence[NativeFullRowExample],
    *,
    schedule: Sequence[MixedTrainingStep],
    schedule_audit: Mapping[str, Any],
    context: distributed.DistributedTrainingContext,
    pad_token_id: int,
    dtype: torch.dtype,
    progress_path: Path,
    require_content_gate_gradients: bool = False,
) -> Mapping[str, Any]:
    if (
        context.world_size != 4
        or GLOBAL_BATCH_SIZE != 16
        or LOCAL_BATCH_SIZE != 4
        or LOCAL_MICROBATCH_SIZE != 2
        or NATIVE_EXECUTION_SUBBATCH_SIZE != 1
        or GRADIENT_ACCUMULATION_STEPS != 2
    ):
        raise ValueError("Evolution training requires world4/local4/micro2/accum2/global16")
    named_trainable = gate._named_trainable_parameters(model)
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        fused=True,
    )
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    content_gate_gradient_steps = 0
    minimum_content_gate_gradient_norm = math.inf
    started = time.time()
    for mixed_step in schedule:
        examples = (
            synthetic_examples
            if mixed_step.update_kind == "synthetic"
            else native_examples
        )
        local_indices = distributed.local_step_indices(
            distributed.GlobalTrainingStep(
                step=mixed_step.step,
                epoch=mixed_step.epoch,
                global_indices=mixed_step.global_indices,
                global_row_ids=mixed_step.global_row_ids,
                step_sha256=mixed_step.step_sha256,
            ),
            process_rank=context.process_rank,
            world_size=context.world_size,
            local_batch_size=LOCAL_BATCH_SIZE,
        )
        selected = [examples[index] for index in local_indices]
        batches: list[Any] = []
        execution_size = execution_subbatch_size(mixed_step.update_kind)
        for start in range(0, LOCAL_BATCH_SIZE, execution_size):
            rows = selected[start : start + execution_size]
            if mixed_step.update_kind == "synthetic":
                batch = gate.collate_examples(
                    rows,
                    pad_token_id=pad_token_id,
                    device=context.device,
                )
            else:
                batch = collate_native_examples(
                    rows,
                    pad_token_id=pad_token_id,
                    device=context.device,
                )
            batches.append(batch)
        local_answer_tokens, local_route_rows = local_objective_denominators(
            mixed_step.update_kind,
            batches,
        )
        denominators = gate._prepare_distributed_scalar_sums(
            context,
            (local_answer_tokens, local_route_rows),
        )
        global_answer_tokens, global_route_rows = (
            int(value)
            for value in gate._distributed_scalar_sums(context, denominators)
        )
        if global_answer_tokens <= 0 or (
            mixed_step.update_kind == "synthetic"
            and global_route_rows != GLOBAL_BATCH_SIZE
        ) or (
            mixed_step.update_kind == "native" and global_route_rows != 0
        ):
            raise distributed.DistributedTrainingError(
                "Mixed objective denominators are invalid"
            )
        optimizer.zero_grad(set_to_none=True)
        local_answer_loss_sum = 0.0
        local_route_loss_sum = 0.0
        local_answer_correct = 0.0
        local_answer_total = 0.0
        local_route_correct = 0.0
        local_route_total = 0.0
        local_occupied_rows = 0.0
        local_occupied_total = 0.0
        local_native_ce_chunks = 0
        local_native_offloaded_tensors = 0
        local_native_offloaded_bytes = 0
        for batch_index, batch in enumerate(batches):
            if mixed_step.update_kind == "synthetic":
                write_audit = gate._write_episode_batch(model, batch, dtype=dtype)
                logits, route_logits = gate._read_episode_batch(
                    model,
                    batch,
                    dtype=dtype,
                )
                route_sum, route_rows, route_predictions = (
                    distributed.route_loss_sum_and_predictions(
                        route_logits,
                        batch.query_mask,
                        batch.target_slots,
                    )
                )
                local_occupied_rows += float(write_audit["full_occupancy_count"])
                local_occupied_total += float(write_audit["full_occupancy_total"])
                local_route_correct += float(
                    sum(
                        prediction.eq(batch.target_slots).sum().item()
                        for prediction in route_predictions.values()
                    )
                )
                local_route_total += float(
                    len(route_predictions) * batch.target_slots.numel()
                )
            else:
                release_native_row_allocator_cache(context.device)
                write_audit, logits, offload_stats = (
                    _native_write_read_selectively_offloaded(
                        model,
                        batch,
                        dtype=dtype,
                    )
                )
                route_sum = logits.sum() * 0.0
                route_rows = 0
                local_occupied_rows += float(write_audit["occupied_rows"])
                local_occupied_total += float(write_audit["occupied_total"])
                local_native_offloaded_tensors += offload_stats.tensors
                local_native_offloaded_bytes += offload_stats.bytes
            if mixed_step.update_kind == "native":
                answer_sum, answer_tokens, ce_chunks = (
                    checkpointed_native_answer_loss_sum_and_count(
                        logits,
                        batch.labels,
                    )
                )
                local_native_ce_chunks += ce_chunks
            else:
                answer_sum, answer_tokens = (
                    distributed.answer_loss_sum_and_count(
                        logits,
                        batch.labels,
                    )
                )
            total_loss = answer_sum / global_answer_tokens
            if global_route_rows:
                total_loss = total_loss + route_sum / global_route_rows
            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError("Mixed evolution loss is non-finite")
            total_loss.backward()
            predicted_rows, expected_rows = runtime._answer_prediction_token_ids(
                logits.detach(),
                batch.labels,
            )
            local_answer_correct += float(
                sum(
                    predicted == expected
                    for predictions, expected in zip(
                        predicted_rows,
                        expected_rows,
                        strict=True,
                    )
                    for predicted, expected in zip(
                        predictions,
                        expected,
                        strict=True,
                    )
                )
            )
            local_answer_total += float(
                sum(len(expected) for expected in expected_rows)
            )
            local_answer_loss_sum += float(answer_sum.detach().float().item())
            local_route_loss_sum += float(route_sum.detach().float().item())
            if answer_tokens <= 0 or route_rows != (
                len(batch.examples) if mixed_step.update_kind == "synthetic" else 0
            ):
                raise RuntimeError("Mixed microbatch objective counts differ")
            reset_delta_mem_states(model)
            batches[batch_index] = None
            answer_sum = None
            batch = None
            logits = None
            predicted_rows = None
            expected_rows = None
            route_logits = None
            route_predictions = None
            route_sum = None
            total_loss = None
            write_audit = None
            offload_stats = None
        gradient_validation = distributed.validate_local_gradients(named_trainable)
        if gradient_validation["passed"] is not True:
            raise RuntimeError("Mixed evolution produced invalid local gradients")
        collective = distributed.sum_gradients(context, named_trainable)
        content_gate_gradient_audit = None
        if require_content_gate_gradients:
            content_gate_gradient_audit = audit_content_gate_gradients(
                named_trainable
            )
            if content_gate_gradient_audit["passed"] is not True:
                raise RuntimeError(
                    "Content-gated evolution produced invalid gate gradients: "
                    f"{content_gate_gradient_audit!r}"
                )
            content_gate_gradient_steps += 1
            minimum_content_gate_gradient_norm = min(
                minimum_content_gate_gradient_norm,
                float(content_gate_gradient_audit["minimum_family_l2_norm"]),
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError("Mixed evolution gradient norm is non-finite")
        optimizer.step()
        metric_tensor = gate._prepare_distributed_scalar_sums(
            context,
            (
                local_answer_loss_sum,
                local_route_loss_sum,
                local_answer_correct,
                local_answer_total,
                local_route_correct,
                local_route_total,
                local_occupied_rows,
                local_occupied_total,
            ),
        )
        metrics = gate._distributed_scalar_sums(context, metric_tensor)
        answer_loss = metrics[0] / global_answer_tokens
        route_loss = metrics[1] / global_route_rows if global_route_rows else 0.0
        record_value = {
            "schema": TRAIN_STEP_SCHEMA,
            "step": mixed_step.step,
            "update_kind": mixed_step.update_kind,
            "update_kind_index": mixed_step.update_kind_index,
            "task": mixed_step.task,
            "epoch": mixed_step.epoch,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "local_microbatch_size": LOCAL_MICROBATCH_SIZE,
            "execution_subbatch_size": execution_size,
            "backward_calls_per_rank": len(batches),
            "native_checkpointed_ce_chunks_per_rank": (
                local_native_ce_chunks
                if mixed_step.update_kind == "native"
                else None
            ),
            "native_selective_offloaded_tensors_per_rank": (
                local_native_offloaded_tensors
                if mixed_step.update_kind == "native"
                else None
            ),
            "native_selective_offloaded_bytes_per_rank": (
                local_native_offloaded_bytes
                if mixed_step.update_kind == "native"
                else None
            ),
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "global_answer_tokens": global_answer_tokens,
            "global_route_rows": global_route_rows,
            "answer_loss": answer_loss,
            "route_loss": route_loss,
            "total_loss": answer_loss + route_loss,
            "teacher_forced_token_accuracy": metrics[2] / metrics[3],
            "semantic_route_accuracy": (
                metrics[4] / metrics[5] if metrics[5] else None
            ),
            "memory_occupancy_fraction": metrics[6] / metrics[7],
            "gradient_norm_before_clip": float(grad_norm.detach().float().item()),
            "gradient_reduction": "sum_before_global_clip",
            "global_row_ids": list(mixed_step.global_row_ids),
            "schedule_step_sha256": mixed_step.step_sha256,
            "gradient_collective_sha256": canonical_sha256(collective),
            "content_gate_gradient_audit": content_gate_gradient_audit,
        }
        if context.is_primary:
            _append_jsonl(progress_path, record_value)
            print(
                json.dumps(
                    {
                        "step": mixed_step.step,
                        "kind": mixed_step.update_kind,
                        "task": mixed_step.task,
                        "answer_loss": round(answer_loss, 6),
                        "route_loss": round(route_loss, 6),
                        "token_accuracy": round(record_value["teacher_forced_token_accuracy"], 4),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        totals[f"{mixed_step.update_kind}_answer_loss"] += answer_loss
        totals[f"{mixed_step.update_kind}_route_loss"] += route_loss
        totals[f"{mixed_step.update_kind}_updates"] += 1
    adapter_hash = runtime._state_dict_sha256(snapshot_delta_mem_weights(model))
    adapter_hashes = distributed.require_consensus(
        context,
        adapter_hash,
        description="final mixed evolution adapter",
    )
    return {
        "updates": len(schedule),
        "elapsed_seconds": time.time() - started,
        "schedule": dict(schedule_audit),
        "mean_losses": {
            kind: {
                "answer": totals[f"{kind}_answer_loss"]
                / totals[f"{kind}_updates"],
                "route": totals[f"{kind}_route_loss"]
                / totals[f"{kind}_updates"],
            }
            for kind in ("synthetic", "native")
        },
        "final_adapter_state_sha256": adapter_hashes[0],
        "progress_sha256": (
            source.sha256_file(progress_path) if context.is_primary else None
        ),
        "content_gate_gradient_audit": {
            "required": require_content_gate_gradients,
            "steps_audited": content_gate_gradient_steps,
            "all_steps_passed": (
                not require_content_gate_gradients
                or content_gate_gradient_steps == len(schedule)
            ),
            "minimum_family_l2_norm": (
                minimum_content_gate_gradient_norm
                if require_content_gate_gradients
                else None
            ),
        },
        "distributed": {
            "world_size": context.world_size,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "local_microbatch_size": LOCAL_MICROBATCH_SIZE,
            "native_execution_subbatch_size": NATIVE_EXECUTION_SUBBATCH_SIZE,
            "native_ce_chunk_tokens": NATIVE_CE_CHUNK_TOKENS,
            "native_selective_offload_min_bytes": (
                NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES
            ),
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "gradient_reduction": "explicit_sum",
            "rank_devices": list(context.rank_devices),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_evolution(
    *,
    context: distributed.DistributedTrainingContext,
    output_dir: Path,
    updates: int,
    base_model: Path = BASE_MODEL,
    adapter_path: Path = R12_ADAPTER,
    source_manifest: Path = R12_SOURCE_MANIFEST,
    native_dataset_root: Path = NATIVE_DATASET_ROOT,
    fusion_topology: str = "attention_output",
) -> Mapping[str, Any]:
    gate.configure_hf_mirror()
    protocol = load_evolution_protocol(fusion_topology)
    native_manifest = validate_native_dataset_root(native_dataset_root)
    stage_names = {
        "attention_output": ("preflight", "stage1"),
        "post_attention_residual_hybrid": (
            "residual_hybrid_preflight",
            "residual_hybrid_stage1",
        ),
        "content_gated_attention_output": (
            "content_gate_preflight",
            "content_gate_stage1",
        ),
        "shared_qo_content_gated_attention_output": (
            "shared_qo_gate_preflight",
            "shared_qo_gate_stage1",
        ),
    }
    if updates == STAGE1_UPDATES:
        stage = stage_names[fusion_topology][1]
    elif updates == PREFLIGHT_UPDATES:
        stage = stage_names[fusion_topology][0]
    else:
        raise ValueError("Evolution run must request 2 or 192 updates")
    requested_output = output_dir.expanduser()
    if requested_output.is_symlink():
        raise ValueError("Evolution output may not be a symbolic link")
    resolved_output = requested_output.resolve()
    if resolved_output.exists():
        raise ValueError(f"Evolution output must be fresh: {resolved_output}")
    base_model = base_model.expanduser().resolve(strict=True)
    adapter_path = adapter_path.expanduser().resolve(strict=True)
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    native_dataset_root = native_dataset_root.expanduser().resolve(strict=True)
    adapter_files = gate.snapshot_directory_files(adapter_path)
    if gate._sha256_json(adapter_files) != R12_ADAPTER_FILES_SHA256:
        raise ValueError("Warm-start R12 adapter hash differs")
    bundle = gate.load_profile_bundle(source_manifest, profile="development")
    if Path(bundle.model_binding["local_model_path"]).resolve() != base_model:
        raise ValueError("R12 source manifest base model differs")
    source_delta_config = build_evolution_delta_config("attention_output")
    delta_config = build_evolution_delta_config(fusion_topology)
    runtime.set_seed(SEED)
    model, tokenizer, _, trainable_names, _ = gate._load_model_and_tokenizer(
        {"model": {"path": str(base_model)}},
        device=context.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        delta_config=delta_config,
    )
    loaded_config = load_delta_mem_adapter(
        model,
        adapter_path,
        initialize_missing_residual_hybrid_gain=(
            fusion_topology == "post_attention_residual_hybrid"
        ),
        initialize_missing_content_gate=(
            fusion_topology in {
                "content_gated_attention_output",
                "shared_qo_content_gated_attention_output",
            }
        ),
    )
    if loaded_config.to_dict() != source_delta_config.to_dict():
        raise ValueError("Warm-start R12 adapter configuration differs")
    trainable_audit = gate.audit_trainable_parameters(
        model,
        expected_trainable_names=trainable_names,
    )
    if trainable_audit["passed"] is not True:
        raise ValueError("Only Delta-Mem parameters may train in evolution")
    initial_adapter_hash = runtime._state_dict_sha256(
        snapshot_delta_mem_weights(model)
    )
    distributed.require_consensus(
        context,
        initial_adapter_hash,
        description="warm-start R12 adapter state",
    )
    synthetic_episodes = gate.select_complete_episodes(bundle.train_episodes, None)
    synthetic_examples = gate.build_training_examples(
        synthetic_episodes,
        tokenizer,
        gate.DEFAULT_TRAINING_CONDITIONS,
    )
    native_examples = load_native_examples(native_dataset_root, tokenizer)
    schedule, schedule_audit = build_mixed_schedule(
        synthetic_examples,
        native_examples,
        total_updates=updates,
    )
    input_binding = {
        "stage": stage,
        "fusion_topology": fusion_topology,
        "updates": updates,
        "base_model": str(base_model),
        "base_model_config_sha256": source.sha256_file(base_model / "config.json"),
        "warm_start_adapter": str(adapter_path),
        "warm_start_adapter_files": adapter_files,
        "warm_start_adapter_files_sha256": R12_ADAPTER_FILES_SHA256,
        "warm_start_adapter_state_sha256": initial_adapter_hash,
        "warm_start_source_delta_config": source_delta_config.to_dict(),
        "target_delta_config": delta_config.to_dict(),
        "synthetic_source_manifest": str(source_manifest),
        "synthetic_source_manifest_sha256": source.sha256_file(source_manifest),
        "native_dataset_root": str(native_dataset_root),
        "native_dataset_manifest_sha256": source.sha256_file(
            native_dataset_root / "manifest.json"
        ),
        "native_dataset_receipt_sha256": native_manifest["receipt"][
            "payload_sha256"
        ],
        "synthetic_examples": len(synthetic_examples),
        "native_examples": len(native_examples),
        "synthetic_examples_sha256": canonical_sha256(
            [example.row_id for example in synthetic_examples]
        ),
        "native_examples_sha256": canonical_sha256(
            [example.row_id for example in native_examples]
        ),
        "native_task_rows": dict(
            sorted(Counter(example.task for example in native_examples).items())
        ),
        "native_max_write_tokens": max(
            len(example.write_input_ids) for example in native_examples
        ),
        "native_max_read_tokens": max(
            len(example.read_input_ids) for example in native_examples
        ),
        "native_max_assistant_target_tokens": max(
            example.assistant_target_tokens for example in native_examples
        ),
        "native_execution_memory_policy": {
            "logical_local_microbatch_size": LOCAL_MICROBATCH_SIZE,
            "execution_subbatch_size": NATIVE_EXECUTION_SUBBATCH_SIZE,
            "local_rows_per_update": LOCAL_BATCH_SIZE,
            "backward_calls_per_rank": (
                LOCAL_BATCH_SIZE // NATIVE_EXECUTION_SUBBATCH_SIZE
            ),
            "gradient_equivalence": (
                "sum_each_row_answer_ce_over_the_same_global_token_denominator_"
                "before_one_global_gradient_sum_and_optimizer_step"
            ),
            "saved_tensor_cpu_offload": False,
            "selective_saved_tensor_cpu_offload": {
                "enabled": True,
                "min_bytes": NATIVE_SELECTIVE_OFFLOAD_MIN_BYTES,
                "eligibility": "cuda_nonleaf_with_grad_fn_only",
                "scope": "native_write_and_read_forward_only",
                "pin_memory": False,
            },
            "checkpointed_float32_ce_chunk_tokens": NATIVE_CE_CHUNK_TOKENS,
            "content_gate_activation_checkpointing": (
                fusion_topology in {
                    "content_gated_attention_output",
                    "shared_qo_content_gated_attention_output",
                }
            ),
            "content_gate_checkpoint_implementation": (
                "torch_non_reentrant"
                if fusion_topology in {
                    "content_gated_attention_output",
                    "shared_qo_content_gated_attention_output",
                }
                else None
            ),
            "serialized_row_graph_release": True,
            "native_row_allocator_cache_release": (
                "gc_collect_and_cuda_empty_cache_before_native_write_and_read"
            ),
            "cuda_allocator_configuration": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        },
        "schedule": dict(schedule_audit),
        "protocol_payload_sha256": {
            "attention_output": EVOLUTION_PROTOCOL_PAYLOAD_SHA256,
            "post_attention_residual_hybrid": (
                RESIDUAL_HYBRID_PROTOCOL_PAYLOAD_SHA256
            ),
            "content_gated_attention_output": (
                CONTENT_GATE_PROTOCOL_PAYLOAD_SHA256
            ),
            "shared_qo_content_gated_attention_output": (
                SHARED_QO_GATE_PROTOCOL_PAYLOAD_SHA256
            ),
        }[fusion_topology],
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    distributed.require_consensus(
        context,
        canonical_sha256(input_binding),
        description="evolution input binding",
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
        phase="evolution-output-creation",
        error=creation_error,
    )
    training = train_mixed_distributed(
        model,
        synthetic_examples,
        native_examples,
        schedule=schedule,
        schedule_audit=schedule_audit,
        context=context,
        pad_token_id=int(tokenizer.pad_token_id),
        dtype=torch.bfloat16,
        progress_path=resolved_output / "training_progress.jsonl",
        require_content_gate_gradients=(
            fusion_topology in {
                "content_gated_attention_output",
                "shared_qo_content_gated_attention_output",
            }
        ),
    )
    final_adapter_hash = runtime._state_dict_sha256(
        snapshot_delta_mem_weights(model)
    )
    if final_adapter_hash == initial_adapter_hash:
        raise RuntimeError("Evolution training did not change the adapter")
    save_error: BaseException | None = None
    result: dict[str, Any] = {}
    if context.is_primary:
        try:
            save_delta_mem_adapter(model, resolved_output / "adapter", delta_config)
            result = {
                "schema": SCHEMA,
                "stage": stage,
                "status": "training_complete_evaluation_pending",
                "input_binding": input_binding,
                "training": dict(training),
                "adapter_files": gate.snapshot_directory_files(
                    resolved_output / "adapter"
                ),
                "code_bindings": {
                    "runner_sha256": source.sha256_file(Path(__file__)),
                    "natural_gate_sha256": source.sha256_file(Path(gate.__file__)),
                    "distributed_sha256": source.sha256_file(Path(distributed.__file__)),
                    "synthetic_runtime_sha256": source.sha256_file(Path(runtime.__file__)),
                },
            }
            result["receipt"] = {
                "algorithm": "sha256",
                "payload_scope": "canonical_result_without_receipt",
                "payload_sha256": source.sha256_text(source.canonical_json(result)),
            }
            _write_json(resolved_output / "result.json", result)
        except BaseException as error:
            save_error = error
    distributed.phase_consensus(
        context,
        phase="evolution-result-save",
        error=save_error,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result if context.is_primary else {
        "stage": stage,
        "status": "worker_complete",
        "rank": context.process_rank,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--updates",
        type=int,
        choices=(PREFLIGHT_UPDATES, STAGE1_UPDATES),
        default=PREFLIGHT_UPDATES,
    )
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=R12_ADAPTER)
    parser.add_argument("--source-manifest", type=Path, default=R12_SOURCE_MANIFEST)
    parser.add_argument("--native-dataset-root", type=Path, default=NATIVE_DATASET_ROOT)
    parser.add_argument(
        "--fusion-topology",
        choices=FUSION_TOPOLOGIES,
        default="attention_output",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@record
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = distributed.initialize_distributed_training(args.device)
    if context is None:
        raise ValueError("Native evolution requires four-rank torchrun")
    try:
        result = run_evolution(
            context=context,
            output_dir=args.output_dir,
            updates=args.updates,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            source_manifest=args.source_manifest,
            native_dataset_root=args.native_dataset_root,
            fusion_topology=args.fusion_topology,
        )
    finally:
        distributed.destroy_distributed_training(context)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
